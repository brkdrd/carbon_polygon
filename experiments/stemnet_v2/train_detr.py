#!/usr/bin/env python3
"""StemNet v2: train the DETR-style head (detr.py) on frozen Sonata + the 12
covariance features, against the survey bases.

One step = V2_ACCUM crops (gradient accumulation), each as v2_train draws
them: an 8 m square wholly in the training split (>= 45% of its 1 m cells
scanned), subsampled to V2_CAP points, TreeLearn augmentation; the crop's
survey bases (label_bases.csv, shifted, tier-corrected) go through the same
x / y transform. Sonata is frozen; only the head learns.
AdamW, lr V2_DETR_LR (2e-4), weight decay 1e-4, V2_DETR_WARMUP (1,000) steps
of warm-up then cosine to 0 over V2_DETR_STEPS (50,000); gradient norm clipped
at 0.1 (DETR's setting).

Every V2_EVAL_EVERY (5,000) steps and at the end: the whole val square predicted
(detr.predict_region), scored against the val survey bases the way
v2_curves / v2_compare score: ranked by probability, matched one-to-one at
0.5 m and 0.3 m -> AP@0.5, AP@0.3, maximum recall, and the best F1 on the val
curve with its probability cut. The best AP@0.5 is kept.

Non-finite guards as v2_train: a crop whose inputs or loss are not finite is
skipped, a step whose gradient is not finite is not applied, non-finite
weights are never saved, a resume whose latest state is not finite falls back
to the best one. Resumes from detr_last.pth (V2_RESUME=0 starts over).

Outputs:
  data/stemnet_v2/models/detr_best.pth, detr_last.pth
  results/stemnet_v2_train_detr.txt         progress, losses, every evaluation
  results/stemnet_v2_val_detections_detr.csv  best evaluation's detections
  results/59_stemnet_v2_val_detr.png        val square: bases found / missed
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS  # noqa: E402
import curves as C  # noqa: E402
import data as D  # noqa: E402
import detr as R  # noqa: E402
import model as M  # noqa: E402
import train as T  # noqa: E402
import vote as V  # noqa: E402

STEPS = int(float(os.environ.get("V2_DETR_STEPS", "50000")))
ACCUM = int(os.environ.get("V2_ACCUM", "4"))
LR = float(os.environ.get("V2_DETR_LR", "2e-4"))
WD = float(os.environ.get("V2_WD", "1e-4"))
WARMUP = int(float(os.environ.get("V2_DETR_WARMUP", "1000")))
CLIP = 0.1
EVAL_EVERY = T.EVAL_EVERY
CAP = T.CAP
D_MODEL = int(os.environ.get("V2_DETR_D", "256"))
QUERIES = int(os.environ.get("V2_DETR_QUERIES", "49"))
LAYERS = int(os.environ.get("V2_DETR_LAYERS", "6"))
BEST, LAST = T.MODELS / "detr_best.pth", T.MODELS / "detr_last.pth"
LOG = RESULTS / "stemnet_v2_train_detr.txt"


def log(msg):
    line = f"[v2-detr] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class DetrRunner:
    """Sonata + covariance features (as v2_train's Runner) -> voxels -> the head."""

    def __init__(self, enc, head, mu, sd, dev):
        self.feat = T.Runner(enc, None, mu, sd, dev)
        self.head, self.dev = head, dev

    def outs(self, c, centre):
        f = self.feat.inputs(c)
        if not bool(torch.isfinite(f).all()):
            return None
        vf, vp = R.voxelise(f, c["xyz"], centre)
        return self.head(vf, vp)

    @torch.no_grad()
    def predict(self, c, centre):
        self.head.eval()
        o = self.outs(c, centre)
        self.head.train()
        if o is None:
            return np.zeros((0, 2)), np.zeros(0)
        logits, ref = o[-1]
        return centre + ref.float().cpu().numpy() * R.HALF, torch.sigmoid(logits).float().cpu().numpy()


def evaluate(runner, store, gt, bbox, split, rng):
    t0 = time.time()
    det, sc = R.predict_region(runner.predict, store, bbox, CAP, rng)
    ign = C.split_at(split, det) != 1 if len(det) else None
    c5, c3 = C.curve(det, sc, gt, 0.5, ign), C.curve(det, sc, gt, 0.3, ign)
    ev = dict(ap5=C.ap(c5), ap3=C.ap(c3), det=int(len(det)), gt=int(len(gt)), seconds=time.time() - t0,
              max_rec=float(c5["tp"][-1] / max(len(gt), 1)) if len(c5["tp"]) else 0.0)
    if len(c5["tp"]):
        rank = np.arange(1, len(c5["tp"]) + 1)
        f1 = 2 * c5["tp"] / (rank + len(gt))
        k = int(np.argmax(f1))
        ev.update(best_f1=float(f1[k]), cut=float(c5["score"][k]), p_at=float(c5["tp"][k] / (k + 1)),
                  r_at=float(c5["tp"][k] / max(len(gt), 1)))
    else:
        ev.update(best_f1=0.0, cut=1.0, p_at=0.0, r_at=0.0)
    return ev, det, sc, ign


def save(path, head, opt, step, best, mu, sd, ev):
    if not T.finite(head):
        log(f"  NOT saving {path.name} at step {step}: weights are not finite")
        return
    T.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(dict(head=head.state_dict(), opt=opt.state_dict(), step=step, best=best, mu=mu, sd=sd, d_in=head.d_in,
                    task="detr", hcfg=head.cfg, eval=ev or {},
                    cfg=dict(steps=STEPS, lr=LR, accum=ACCUM, cap=CAP, voxel=R.VOXEL, encoder=T.ENCODER)), path)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = json.loads((OUT / "frame.json").read_text())
    if "labels" not in frame:
        raise SystemExit("[v2-detr] no labels — run v2_labels first")
    rng = np.random.default_rng(T.SEED)
    torch.manual_seed(T.SEED)
    dev = T.DEVICE
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[v2-detr] V2_DEVICE=cuda but no GPU is visible in the container — check the NVIDIA runtime")
    from scipy.spatial import cKDTree
    regions = np.load(OUT / "regions.npz")
    store = D.open_store(log)
    centres = D.eligible_centres(regions, R.CROP, 0.45, 0)
    rows = list(csv.DictReader(open(OUT / "label_bases.csv")))
    bxy = np.array([[float(r["x"]), float(r["y"])] for r in rows]).reshape(-1, 2)
    bkd = cKDTree(bxy)
    gt, _, bbox = T.val_truth(regions)
    log(f"{len(centres):,} training crop positions | {len(bxy):,} survey bases | val square {bbox} with {len(gt)} bases "
        f"| device {dev}")
    enc = M.load_sonata(dev) if T.ENCODER == "sonata" else M.FakeEncoder()

    ck = None
    for path in ((LAST, BEST) if T.RESUME else ()):
        if path.exists():
            c_ = torch.load(path, map_location=dev, weights_only=False)
            if all(bool(torch.isfinite(v).all()) for v in c_["head"].values()):
                ck = c_
                break
            log(f"{path.name} (step {c_['step']}) has non-finite weights — not resuming from it")
    mu, sd = (ck["mu"], ck["sd"]) if ck is not None else T.feature_stats(store, centres, rng)
    probe = store.crop(*centres[len(centres) // 2], R.CROP)
    probe = {k: v[:20000] for k, v in probe.items()}
    d_in = T.Runner(enc, None, mu, sd, dev).inputs(probe).shape[1]
    head = R.DetrHead(d_in, d=D_MODEL, n_query=QUERIES, layers=LAYERS).to(dev)
    log(f"DETR head: {d_in} values per point -> {R.VOXEL:g} m voxels -> {LAYERS} decoder layers, {QUERIES} queries, "
        f"d {D_MODEL} | {sum(p.numel() for p in head.parameters()):,} parameters")
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    start, best = 0, -1.0
    if ck is not None:
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
        start, best = int(ck["step"]), float(ck["best"])
        log(f"resuming from step {start} (best val AP@0.5 so far {best:.3f})")
    for g in opt.param_groups:
        g.setdefault("initial_lr", LR)

    def lr_at(s):
        if s < WARMUP:
            return (s + 1) / WARMUP
        return 0.5 * (1 + np.cos(np.pi * min(s - WARMUP, STEPS - WARMUP) / max(STEPS - WARMUP, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=start - 1)
    runner = DetrRunner(enc, head, mu, sd, dev)

    run, t0, last, ev = {}, time.time(), time.time(), None
    skipped_crops = skipped_steps = 0
    for step in range(start + 1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        used, stats, l_sum = 0, [], 0.0
        for _ in range(ACCUM):
            for _try in range(20):
                cx, cy = centres[rng.integers(len(centres))] + rng.uniform(-0.5, 0.5, 2)
                c = store.crop(cx, cy, R.CROP)
                if c is not None and len(c["xyz"]) >= 200:
                    break
            if len(c["xyz"]) > CAP:
                sel = rng.choice(len(c["xyz"]), CAP, replace=False)
                c = {k: v[sel] for k, v in c.items()}
            centre = np.array([cx, cy])
            b = R.crop_bases(bkd, bxy, cx, cy)
            A = D.augment(c, rng, centre)
            b = (b - centre) @ A.T + centre if len(b) else b
            o = runner.outs(c, centre)
            loss = None
            if o is not None:
                gtn = torch.as_tensor((b - centre) / R.HALF, dtype=torch.float32, device=dev).reshape(-1, 2)
                loss, st = R.set_loss(o, gtn)
            if loss is None or not bool(torch.isfinite(loss)):
                skipped_crops += 1
                if skipped_crops <= 50:
                    log(f"  skipped crop at ({cx:.1f}, {cy:.1f}), step {step}: non-finite {'inputs' if o is None else 'loss'}")
                continue
            (loss / ACCUM).backward()
            l_sum += float(loss); used += 1; stats.append(st)
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP) if used else torch.tensor(0.0)
        if used and bool(torch.isfinite(gn)):
            opt.step()
        elif used:
            skipped_steps += 1
            log(f"  step {step} not applied: gradient norm {float(gn)}")
        sched.step()
        if stats:
            errs = [s["err_m"] for s in stats if np.isfinite(s["err_m"])]
            pm = [s["p_matched"] for s in stats if np.isfinite(s["p_matched"])]
            for k, v in (("loss", l_sum / used), ("err", float(np.mean(errs)) if errs else None),
                         ("pm", float(np.mean(pm)) if pm else None),
                         ("gt", float(np.mean([s["gt"] for s in stats]))), ("conf", float(np.mean([s["confident"] for s in stats])))):
                if v is not None:
                    run[k] = v if k not in run else 0.95 * run[k] + 0.05 * v
        if time.time() - last > T.PROGRESS_S or step == start + 1:
            last = time.time()
            rate = (time.time() - t0) / max(step - start, 1)
            mem = f" | gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if dev.startswith("cuda") else ""
            log(f"step {step}/{STEPS} | loss {run.get('loss', float('nan')):.4f} | matched: position error "
                f"{run.get('err', float('nan')) * 100:.1f} cm, p {run.get('pm', float('nan')):.2f} | "
                f"{run.get('gt', 0):.1f} bases / crop, {run.get('conf', 0):.1f} queries >= 0.5 | lr {sched.get_last_lr()[0]:.2e} | "
                f"{rate:.2f} s/step, eta {rate * (STEPS - step) / 3600:.1f} h{mem}"
                + (f" | skipped {skipped_crops} crops, {skipped_steps} steps" if skipped_crops or skipped_steps else ""))
        if step % EVAL_EVERY == 0 or step == STEPS:
            ev, det, sc, ign = evaluate(runner, store, gt, bbox, regions["split"], rng)
            log(f"EVAL step {step}: AP@0.5 {ev['ap5']:.3f} | AP@0.3 {ev['ap3']:.3f} | max recall {ev['max_rec']:.3f} | "
                f"best F1 on the val curve {ev['best_f1']:.3f} (P {ev['p_at']:.3f} R {ev['r_at']:.3f} at p >= {ev['cut']:.2f}) | "
                f"{ev['det']} det (p >= {R.MIN_P:g}) / {ev['gt']} bases | {ev['seconds'] / 60:.1f} min")
            if ev["ap5"] > best:
                best = ev["ap5"]
                save(BEST, head, opt, step, best, mu, sd, ev)
                log(f"  new best -> {BEST}")
                k = sc >= ev["cut"]
                mm = V.match(det[k], sc[k], gt, 0.5, ignore=ign[k] if ign is not None else None)
                with open(RESULTS / "stemnet_v2_val_detections_detr.csv", "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["x", "y", "probability", "kept_at_best_f1_cut", "matched_base"])
                    pair_all = V.match(det, sc, gt, 0.5, ignore=ign)["pair"] if len(det) else []
                    for (x, y), s_, pr in zip(det, sc, pair_all):
                        w.writerow([f"{x:.3f}", f"{y:.3f}", f"{s_:.4f}", int(s_ >= ev["cut"]), int(pr)])
                T.figure(gt, det[k], mm["pair"], bbox, RESULTS / "59_stemnet_v2_val_detr.png",
                         title=f"StemNet v2 DETR head, val square at p >= {ev['cut']:.2f}: F1 {ev['best_f1']:.3f}, "
                               f"AP@0.5 {ev['ap5']:.3f}")
            save(LAST, head, opt, step, best, mu, sd, ev)
        elif step % 1000 == 0:
            save(LAST, head, opt, step, best, mu, sd, ev)
    log(f"done: {STEPS} steps in {(time.time() - t0) / 3600:.1f} h | best val AP@0.5 {best:.3f} -> {BEST}")


if __name__ == "__main__":
    main()
