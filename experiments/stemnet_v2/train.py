#!/usr/bin/env python3
"""StemNet v2 training: frozen Sonata + a head that learns, for every point,
"am I close to a tree base" (gate) and the 2-D offset to it.

One step = V2_ACCUM crops (gradient accumulation), each:
  1. an 8 m square, centre drawn fresh from every position that lies wholly in
     the training split with >= 45% of its 1 m cells scanned
  2. points, normals, 12 covariance features, labels from the tiles;
     random-subsampled to V2_CAP points if denser
  3. TreeLearn augmentation (data.py; x/y-only matrix jitter, grid-scaled
     point jitter)
  4. Sonata (frozen, no gradients) -> 1,232 features per point; with the 12
     standardised covariance features -> the head -> gate logit + offset
  5. loss on the INNER 6 m only — the 1 m rim is context for Sonata, its own
     predictions are not graded (its points cannot see past the crop edge):
       gate    binary cross-entropy, positives weighted by the crop's
               background/close ratio, capped at V2_POS_W_CAP (20)
       offset  mean distance between predicted and true offset, close points
               only (TreeLearn's offset loss), x V2_LAMBDA_OFF
AdamW, lr V2_LR with a cosine schedule over V2_STEPS.

Every V2_EVAL_EVERY steps (and at the end): the whole val square predicted
with overlapping crops, gated points shifted, HDBSCAN clusters -> bases, and
scored against the val bases at 0.5 m (the headline, comparable to every
earlier number) and 0.3 m (placement precision, now meaningful because the
bases sit within centimetres). The best F1 at 0.5 m is kept; the latest state
is always saved, and a restart resumes from it (V2_RESUME=0 starts over).

Outputs:
  data/stemnet_v2/models/head_best.pth, head_last.pth
  results/stemnet_v2_train.txt          progress, losses, every evaluation
  results/stemnet_v2_val_detections.csv
  results/46_stemnet_v2_val.png
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS  # noqa: E402
import data as D  # noqa: E402
import model as M  # noqa: E402
import vote as V  # noqa: E402

STEPS = int(float(os.environ.get("V2_STEPS", "150000")))
ACCUM = int(os.environ.get("V2_ACCUM", "4"))
LR = float(os.environ.get("V2_LR", "1e-3"))
WD = float(os.environ.get("V2_WD", "1e-4"))
CAP = int(float(os.environ.get("V2_CAP", "300000")))
POS_W_CAP = float(os.environ.get("V2_POS_W_CAP", "20"))
LAMBDA_OFF = float(os.environ.get("V2_LAMBDA_OFF", "1.0"))
EVAL_EVERY = int(float(os.environ.get("V2_EVAL_EVERY", "5000")))
DEVICE = os.environ.get("V2_DEVICE", "cuda")
ENCODER = os.environ.get("V2_ENCODER", "sonata")
RESUME = os.environ.get("V2_RESUME", "1") != "0"
PROGRESS_S = float(os.environ.get("V2_PROGRESS_S", "60"))
SEED = int(os.environ.get("V2_SEED", "0"))
CROP, INNER = 8.0, 6.0
MODELS = OUT / "models"
BEST, LAST = MODELS / "head_best.pth", MODELS / "head_last.pth"
LOG = RESULTS / "stemnet_v2_train.txt"


def log(msg):
    line = f"[v2-train] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def feature_stats(store, centres, rng, n=150):
    """Mean / std of the 12 covariance features over a sample of training crops."""
    acc = []
    for c in centres[rng.choice(len(centres), min(n, len(centres)), replace=False)]:
        cr = store.crop(c[0], c[1], CROP)
        if cr is not None:
            acc.append(cr["feat"][rng.choice(len(cr["feat"]), min(5000, len(cr["feat"])), replace=False)])
    f = np.concatenate(acc).astype(np.float64)
    f = f[np.isfinite(f).all(1)]
    return f.mean(0).astype(np.float32), np.maximum(f.std(0), 1e-6).astype(np.float32)


def val_truth(regions):
    split = regions["split"]
    gt, tier = [], []
    for r in csv.DictReader(open(OUT / "label_bases.csv")):
        x, y = float(r["x"]), float(r["y"])
        cy, cx = int(np.floor(y)), int(np.floor(x))
        if 0 <= cy < split.shape[0] and 0 <= cx < split.shape[1] and split[cy, cx] == 1:
            gt.append([x, y]); tier.append(r["tier"])
    jj, ii = np.nonzero(split == 1)
    bbox = (float(ii.min()), float(ii.max() + 1), float(jj.min()), float(jj.max() + 1))
    return np.array(gt), np.array(tier), bbox


class Runner:
    def __init__(self, enc, head, mu, sd, device):
        self.enc, self.head, self.mu, self.sd, self.dev = enc, head, mu, sd, device

    def inputs(self, c):
        local = torch.from_numpy(M.to_local(c["xyz"])).to(self.dev)
        normal = torch.from_numpy(np.ascontiguousarray(c["normal"], np.float32)).to(self.dev)
        f = M.dense_features(self.enc, local, normal)
        ft = torch.from_numpy(np.nan_to_num((c["feat"] - self.mu) / self.sd).astype(np.float32)).to(self.dev)
        return torch.cat([f, ft], 1)

    @torch.no_grad()
    def predict(self, c):
        self.head.eval()
        g, o = self.head(self.inputs(c))
        self.head.train()
        return torch.sigmoid(g).cpu().numpy(), o.cpu().numpy()


def crop_loss(runner, c, inner):
    x = runner.inputs(c)
    g, o = runner.head(x)
    m = torch.from_numpy(inner).to(runner.dev)
    y = torch.from_numpy(c["gate"].astype(np.float32)).to(runner.dev)[m]
    g = g[m]
    n_pos = int(y.sum())
    pw = torch.tensor(min(POS_W_CAP, max(1.0, (len(y) - n_pos) / max(n_pos, 1))), device=runner.dev)
    l_gate = F.binary_cross_entropy_with_logits(g, y, pos_weight=pw)
    if n_pos:
        t = torch.from_numpy(c["offset"]).to(runner.dev)[m][y > 0.5]
        l_off = torch.linalg.vector_norm(o[m][y > 0.5] - t, dim=1).mean()
    else:
        l_off = o.sum() * 0.0
    return l_gate, l_off, n_pos, len(y)


def evaluate(runner, store, gt, tier, bbox, rng):
    t0 = time.time()
    xy, p, off = V.predict_region(runner.predict, store, bbox, CAP, rng)
    det, score, nv = V.detect(xy, p, off)
    out = dict(points=int(len(xy)), gated=float((p >= V.P_THRESH).mean()) if len(p) else 0.0,
               detections=int(len(det)), gt=int(len(gt)), seconds=time.time() - t0)
    for r in (0.5, 0.3):
        mm = V.match(det, score, gt, r, bbox)
        found = np.zeros(len(gt), bool); found[mm["pair"][mm["pair"] >= 0]] = True
        out[f"@{r}"] = {k: v for k, v in mm.items() if k != "pair"}
        out[f"@{r}"]["recall_by_tier"] = {t: float(found[tier == t].mean()) for t in ("A", "B", "C") if (tier == t).any()}
        if r == 0.5:
            out["_pair"] = mm["pair"]
    return out, det, score, nv


def report_eval(step, ev):
    a, b = ev["@0.5"], ev["@0.3"]
    log(f"EVAL step {step}: @0.5 m F1 {a['f1']:.3f} (P {a['precision']:.3f} R {a['recall']:.3f}, "
        f"{a['tp']} TP / {a['fp']} FP / {a['fn']} FN) | @0.3 m F1 {b['f1']:.3f} "
        f"(P {b['precision']:.3f} R {b['recall']:.3f}) | {ev['detections']} det / {ev['gt']} bases | "
        f"recall@0.5 by tier " + " ".join(f"{t} {v:.2f}" for t, v in a["recall_by_tier"].items())
        + f" | {ev['gated'] * 100:.1f}% of {ev['points']:,} points gated | {ev['seconds'] / 60:.1f} min")


def save(path, head, opt, step, best, mu, sd, ev):
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(dict(head=head.state_dict(), opt=opt.state_dict(), step=step, best=best, mu=mu, sd=sd,
                    d_in=head.d_in, eval={k: v for k, v in (ev or {}).items() if not k.startswith("_")},
                    cfg=dict(crop=CROP, inner=INNER, cap=CAP, lr=LR, steps=STEPS, accum=ACCUM,
                             encoder=ENCODER, colour="zeros")), path)


def figure(gt, det, pair, bbox, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    found = np.zeros(len(gt), bool); found[pair[pair >= 0]] = True
    ax.scatter(gt[found, 0], gt[found, 1], s=40, facecolors="none", edgecolors="tab:green", label=f"base found ({found.sum()})")
    ax.scatter(gt[~found, 0], gt[~found, 1], s=40, facecolors="none", edgecolors="tab:red", label=f"base missed ({(~found).sum()})")
    tp = pair >= 0
    ax.scatter(det[tp, 0], det[tp, 1], s=12, c="tab:green", marker="x", label="detection, correct")
    ax.scatter(det[pair == -1, 0], det[pair == -1, 1], s=12, c="tab:red", marker="x", label="detection, false")
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3]); ax.set_aspect("equal")
    ax.legend(loc="upper right"); ax.set_title("StemNet v2 on the val square, match 0.5 m")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = json.loads((OUT / "frame.json").read_text())
    if "labels" not in frame:
        raise SystemExit("[v2-train] no labels — run v2_labels first")
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[v2-train] V2_DEVICE=cuda but no GPU is visible in the container — check the "
                         "NVIDIA runtime (run_experiments gpu_check); refusing to crawl on the CPU")
    dev = DEVICE
    regions = np.load(OUT / "regions.npz")
    store = D.TileStore()
    centres = D.eligible_centres(regions, CROP, 0.45, 0)
    gt, tier, bbox = val_truth(regions)
    log(f"{len(centres):,} training crop positions | val square {bbox} with {len(gt)} bases | device {dev}")

    enc = M.load_sonata(dev) if ENCODER == "sonata" else M.FakeEncoder()
    start, best, mu, sd = 0, -1.0, None, None
    ck = torch.load(LAST, map_location=dev, weights_only=False) if (RESUME and LAST.exists()) else None
    if ck is not None:
        mu, sd = ck["mu"], ck["sd"]
    else:
        mu, sd = feature_stats(store, centres, rng)
    probe = store.crop(*centres[len(centres) // 2], CROP)
    probe = {k: v[:20000] for k, v in probe.items()}
    d_in = Runner(enc, None, mu, sd, dev).inputs(probe).shape[1]   # 1,232 Sonata + 12 covariance
    log(f"head input: {d_in} values per point")
    head = M.Head(d_in).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    if ck is not None:
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
        start, best = int(ck["step"]), float(ck["best"])
        log(f"resuming from step {start} (best F1@0.5 so far {best:.3f})")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, STEPS) / STEPS)))
    for _ in range(start):
        sched.step()
    runner = Runner(enc, head, mu, sd, dev)

    run, t0, last, ev = {}, time.time(), time.time(), None
    for step in range(start + 1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        lg_s = lo_s = 0.0; npos = npts = 0
        for _ in range(ACCUM):
            for _try in range(20):
                cx, cy = centres[rng.integers(len(centres))] + rng.uniform(-0.5, 0.5, 2)
                c = store.crop(cx, cy, CROP)
                if c is not None and len(c["xyz"]) >= 200:
                    break
            if len(c["xyz"]) > CAP:
                sel = rng.choice(len(c["xyz"]), CAP, replace=False)
                c = {k: v[sel] for k, v in c.items()}
            inner = (np.abs(c["xyz"][:, 0] - cx) <= INNER / 2) & (np.abs(c["xyz"][:, 1] - cy) <= INNER / 2)
            D.augment(c, rng, np.array([cx, cy]))
            lg, lo, p_, n_ = crop_loss(runner, c, inner)
            ((lg + LAMBDA_OFF * lo) / ACCUM).backward()
            lg_s += float(lg) / ACCUM; lo_s += float(lo) / ACCUM; npos += p_; npts += n_
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        for k, v in (("gate", lg_s), ("off", lo_s), ("pos", npos / max(npts, 1)), ("pts", npts / ACCUM)):
            run[k] = v if k not in run else 0.95 * run[k] + 0.05 * v
        if time.time() - last > PROGRESS_S or step == start + 1:
            last = time.time()
            rate = (time.time() - t0) / max(step - start, 1)
            mem = (f" | gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                   if dev.startswith("cuda") else "")
            log(f"step {step}/{STEPS} | gate {run['gate']:.4f} off {run['off'] * 100:.1f} cm | close {run['pos'] * 100:.2f}% "
                f"of {run['pts']:,.0f} inner pts | lr {sched.get_last_lr()[0]:.2e} | {rate:.2f} s/step, "
                f"eta {rate * (STEPS - step) / 3600:.1f} h{mem}")
        if step % EVAL_EVERY == 0 or step == STEPS:
            ev, det, score, nv = evaluate(runner, store, gt, tier, bbox, rng)
            report_eval(step, ev)
            if ev["@0.5"]["f1"] > best:
                best = ev["@0.5"]["f1"]
                save(BEST, head, opt, step, best, mu, sd, ev)
                log(f"  new best -> {BEST}")
                with open(RESULTS / "stemnet_v2_val_detections.csv", "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["x", "y", "score", "votes", "matched_base"])
                    for (x, y), s_, n_, pr in zip(det, score, nv, ev["_pair"]):
                        w.writerow([f"{x:.3f}", f"{y:.3f}", f"{s_:.2f}", int(n_), int(pr)])
                figure(gt, det, ev["_pair"], bbox, RESULTS / "46_stemnet_v2_val.png")
            save(LAST, head, opt, step, best, mu, sd, ev)
        elif step % 1000 == 0:                       # a crash loses at most 1,000 steps
            save(LAST, head, opt, step, best, mu, sd, ev)
    log(f"done: {STEPS} steps in {(time.time() - t0) / 3600:.1f} h | best val F1@0.5 {best:.3f} -> {BEST}")


if __name__ == "__main__":
    main()
