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

Non-finite guard: a crop whose inputs or loss are not finite is skipped and
logged with its position; a step whose gradient is not finite is not applied;
weights that are not finite are never saved; and a resume whose latest state
is not finite falls back to the best one. (Run 1 lost everything after step
~18,850 to a single inf feature value.)

The progress line splits each step's time into data (tiles, crop,
augmentation), encoder (Sonata) and head (forward + backward).

V2_TASK=base+tree (the v2_train_tree target) adds TreeLearn's semantic half:
a third output, "is this point tree", trained with plain binary cross-entropy
on the tree / non-tree labels carried over from campus_pred_B24
(prep_semantic.py: target = tree, loss only where the label is known),
x V2_LAMBDA_TREE, beside the gate and offset. Evaluation adds accuracy, IoU
per class and mIoU on the val square, for all labelled points and for the
points whose label came from a matched point (i.e. without the removed
ground). Its own checkpoints and log (head_tree_*.pth,
stemnet_v2_train_tree.txt); the best F1 and the best mIoU are both kept.

Outputs (V2_TASK=base+tree: head_tree_*, *_tree.txt / .csv, 46_..._tree.png):
  data/stemnet_v2/models/head_best.pth, head_last.pth (+ head_tree_best_miou.pth)
  results/stemnet_v2_train.txt          progress, losses, every evaluation
  results/stemnet_v2_val_detections.csv
  results/46_stemnet_v2_val.png
  results/51_stemnet_v2_val_tree.png    (tree task) val square: tree right / wrong
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

STEPS = int(float(os.environ.get("V2_STEPS", "50000")))
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
TASK = os.environ.get("V2_TASK", "base")
if TASK not in ("base", "base+tree"):
    raise SystemExit(f"[v2-train] V2_TASK={TASK!r}: use base or base+tree")
TREE = TASK == "base+tree"
LAMBDA_TREE = float(os.environ.get("V2_LAMBDA_TREE", "1.0"))
CROP, INNER = 8.0, 6.0
MODELS = OUT / "models"
SUF = "_tree" if TREE else ""
BEST, LAST = MODELS / f"head{SUF}_best.pth", MODELS / f"head{SUF}_last.pth"
BEST_MIOU = MODELS / "head_tree_best_miou.pth"
LOG = RESULTS / f"stemnet_v2_train{SUF}.txt"


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
        out = self.head(self.inputs(c))
        self.head.train()
        res = (torch.sigmoid(out[0]).cpu().numpy(), out[1].cpu().numpy())
        if len(out) > 2:
            res += (torch.sigmoid(out[2]).cpu().numpy(),)
        return res


def archive_longer_run():
    """Checkpoints of a run with a LONGER schedule than V2_STEPS belong to a
    different run (run 1: 150,000 steps, killed by a NaN at ~18,850): they are
    moved aside with their log and outputs (name + _<steps>steps), never
    continued. A shorter one is resumed and extended, as before. -> note or None."""
    if not RESUME:
        return None
    for path in (LAST, BEST):
        if path.exists():
            n = int(torch.load(path, map_location="cpu", weights_only=False).get("cfg", {}).get("steps", STEPS))
            if n <= STEPS:
                return None
            moved = []
            for q in (LAST, BEST, BEST_MIOU, LOG, RESULTS / f"stemnet_v2_val_detections{SUF}.csv",
                      RESULTS / f"46_stemnet_v2_val{SUF}.png"):
                if q.exists():
                    q.rename(q.with_name(f"{q.stem}_{n}steps{q.suffix}"))
                    moved.append(q.name)
            return (f"previous run had a {n:,}-step schedule, this one {STEPS:,}: not continuing it — moved aside "
                    f"as *_{n}steps: {', '.join(moved)}; starting from step 0")
    return None


def finite(head):
    return all(bool(torch.isfinite(p).all()) for p in head.parameters())


def crop_loss(runner, c, inner, x):
    out = runner.head(x)
    g, o = out[0], out[1]
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
    tr = None
    if len(out) > 2:
        s_ = torch.from_numpy(c["sem"].astype(np.int64)).to(runner.dev)[m]
        k = s_ >= 0
        tl = out[2][m][k]
        if int(k.sum()):
            yt = (s_[k] == 1).float()
            l_tree = F.binary_cross_entropy_with_logits(tl, yt)
            acc = float(((tl > 0).float() == yt).float().mean())
        else:
            l_tree, acc = out[2].sum() * 0.0, float("nan")
        tr = dict(loss=l_tree, acc=acc, labelled=int(k.sum()), tree=int((s_ == 1).sum()))
    return l_gate, l_off, n_pos, len(y), tr


def tree_metrics(pt, sem):
    """Tree probability vs labels: accuracy, IoU per class, mIoU, P / R of tree."""
    k = sem >= 0
    pred, y = pt[k] >= 0.5, sem[k] == 1
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    iou_t = tp / max(tp + fp + fn, 1); iou_n = tn / max(tn + fp + fn, 1)
    return dict(points=int(k.sum()), acc=(tp + tn) / max(k.sum(), 1), iou_tree=iou_t, iou_non=iou_n,
                miou=(iou_t + iou_n) / 2, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                tree_share=float(y.mean()) if k.any() else 0.0)


def evaluate(runner, store, gt, tier, bbox, rng):
    t0 = time.time()
    R = V.predict_region(runner.predict, store, bbox, CAP, rng, full=True)
    xy, p, off = R["xy"], R["p"], R["off"]
    det, score, nv = V.detect(xy, p, off)
    out = dict(points=int(len(xy)), gated=float((p >= V.P_THRESH).mean()) if len(p) else 0.0,
               detections=int(len(det)), gt=int(len(gt)), seconds=time.time() - t0)
    if TREE:
        out["tree"] = tree_metrics(R["tree"], R["sem"])
        out["tree_matched"] = tree_metrics(R["tree"], np.where(R["sem"] == 2, -1, R["sem"]))
        out["_region"] = R
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
    for k, nm in (("tree", "all labelled"), ("tree_matched", "matched only (no removed ground)")):
        if k in ev:
            t = ev[k]
            log(f"  TREE {nm}: mIoU {t['miou']:.3f} (IoU tree {t['iou_tree']:.3f}, non-tree {t['iou_non']:.3f}) | "
                f"acc {t['acc']:.3f} | tree P {t['precision']:.3f} R {t['recall']:.3f} | "
                f"{t['points']:,} pts, {t['tree_share'] * 100:.1f}% tree")


def save(path, head, opt, step, best, mu, sd, ev, best_miou=-1.0):
    if not finite(head):
        log(f"  NOT saving {path.name} at step {step}: weights are not finite")
        return
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(dict(head=head.state_dict(), opt=opt.state_dict(), step=step, best=best, best_miou=best_miou,
                    mu=mu, sd=sd, d_in=head.d_in, task=TASK,
                    eval={k: v for k, v in (ev or {}).items() if not k.startswith("_")},
                    cfg=dict(crop=CROP, inner=INNER, cap=CAP, lr=LR, steps=STEPS, accum=ACCUM,
                             encoder=ENCODER, colour="zeros", task=TASK, lambda_tree=LAMBDA_TREE)), path)


def figure(gt, det, pair, bbox, path, title="StemNet v2 on the val square, match 0.5 m"):
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
    ax.legend(loc="upper right"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def tree_figure(R, bbox, step, t, path, px=0.1, name="Val square"):
    """A region from above: per pixel, the highest point's outcome."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x0, x1, y0, y1 = bbox
    nx, ny = int(np.ceil((x1 - x0) / px)), int(np.ceil((y1 - y0) / px))
    i = np.clip(((R["xy"][:, 0] - x0) / px).astype(np.int64), 0, nx - 1)
    j = np.clip(((R["xy"][:, 1] - y0) / px).astype(np.int64), 0, ny - 1)
    pred, sem = R["tree"] >= 0.5, R["sem"]
    cls = np.full(len(sem), 4)                                   # 0 TP 1 TN 2 FP 3 FN 4 unlabelled
    lab = sem >= 0; y = sem == 1
    cls[lab & pred & y] = 0; cls[lab & ~pred & ~y] = 1; cls[lab & pred & ~y] = 2; cls[lab & ~pred & y] = 3
    key = j * nx + i
    o = np.lexsort((R["z"], key))
    last = np.r_[key[o][1:] != key[o][:-1], True]
    pal = np.array([[0.13, 0.55, 0.13], [0.75, 0.75, 0.75], [0.85, 0.1, 0.1], [0.1, 0.3, 0.9], [0.95, 0.85, 0.3]])
    img = np.ones((ny * nx, 3)); img[key[o][last]] = pal[cls[o][last]]
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(img.reshape(ny, nx, 3), origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(pal, ("tree, right", "non-tree, right",
              "said tree, is non-tree", "said non-tree, is tree", "no label"))], loc="upper right")
    ax.set_title(f"{name} from above (highest point per {px * 100:g} cm pixel), step {step}: "
                 f"mIoU {t['miou']:.3f}, acc {t['acc']:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    note = archive_longer_run()
    if note:
        log(note)
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
    store = D.open_store(log)
    centres = D.eligible_centres(regions, CROP, 0.45, 0)
    gt, tier, bbox = val_truth(regions)
    log(f"{len(centres):,} training crop positions | val square {bbox} with {len(gt)} bases | device {dev}")

    if TREE:
        have = 0
        for c_ in centres[rng.choice(len(centres), min(20, len(centres)), replace=False)]:
            cr = store.crop(c_[0], c_[1], CROP)
            have += int((cr["sem"] >= 0).sum()) if cr is not None else 0
        if not have:
            raise SystemExit("[v2-train] V2_TASK=base+tree but the crops carry no tree / non-tree labels — "
                             "run v2_semantic, then v2_cache")
        log(f"task base+tree: tree loss x {LAMBDA_TREE:g} | train-split labels: "
            f"{json.dumps(frame.get('semantic', {}).get('tally', {}).get('train', {}))}")
    enc = M.load_sonata(dev) if ENCODER == "sonata" else M.FakeEncoder()
    start, best, best_miou, mu, sd = 0, -1.0, -1.0, None, None
    ck = None
    for path in ((LAST, BEST) if RESUME else ()):
        if path.exists():
            c_ = torch.load(path, map_location=dev, weights_only=False)
            if all(bool(torch.isfinite(v).all()) for v in c_["head"].values()):
                ck = c_
                break
            log(f"{path.name} (step {c_['step']}) has non-finite weights — not resuming from it")
    if ck is not None:
        mu, sd = ck["mu"], ck["sd"]
    else:
        mu, sd = feature_stats(store, centres, rng)
    probe = store.crop(*centres[len(centres) // 2], CROP)
    probe = {k: v[:20000] for k, v in probe.items()}
    d_in = Runner(enc, None, mu, sd, dev).inputs(probe).shape[1]   # 1,232 Sonata + 12 covariance
    log(f"head input: {d_in} values per point")
    head = M.Head(d_in, tree=TREE).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    if ck is not None:
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
        start, best, best_miou = int(ck["step"]), float(ck["best"]), float(ck.get("best_miou", -1.0))
        log(f"resuming from step {start} (best F1@0.5 so far {best:.3f})")
        if any(torch.is_tensor(v) and not bool(torch.isfinite(v).all())
               for st in opt.state.values() for v in st.values()):
            raise SystemExit("[v2-train] optimizer state is not finite — V2_RESUME=0 to start over")
    for g in opt.param_groups:
        g.setdefault("initial_lr", LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, STEPS) / STEPS)),
                                              last_epoch=start - 1)
    runner = Runner(enc, head, mu, sd, dev)

    def sync():
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    run, t0, last, ev = {}, time.time(), time.time(), None
    skipped_crops = skipped_steps = 0
    for step in range(start + 1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        lg_s = lo_s = lt_s = 0.0; npos = npts = 0; used = 0; t_acc = []
        t_data = t_enc = t_head = 0.0
        for _ in range(ACCUM):
            ta = time.time()
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
            tb = time.time()
            x = runner.inputs(c)
            sync(); tc = time.time()
            t_data += tb - ta; t_enc += tc - tb
            bad = "inputs" if not bool(torch.isfinite(x).all()) else None
            if bad is None:
                lg, lo, p_, n_, tr = crop_loss(runner, c, inner, x)
                lt = tr["loss"] if tr is not None else lg * 0.0
                if not (bool(torch.isfinite(lg)) and bool(torch.isfinite(lo)) and bool(torch.isfinite(lt))):
                    bad = "loss"
            if bad is not None:
                skipped_crops += 1
                if skipped_crops <= 50:
                    log(f"  skipped crop at ({cx:.1f}, {cy:.1f}), step {step}: non-finite {bad}")
                continue
            ((lg + LAMBDA_OFF * lo + LAMBDA_TREE * lt) / ACCUM).backward()
            sync(); t_head += time.time() - tc
            lg_s += float(lg); lo_s += float(lo); lt_s += float(lt); npos += p_; npts += n_; used += 1
            if tr is not None and np.isfinite(tr["acc"]):
                t_acc.append(tr["acc"])
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0) if used else torch.tensor(0.0)
        if used and bool(torch.isfinite(gn)):
            opt.step()
        elif used:
            skipped_steps += 1
            log(f"  step {step} not applied: gradient norm {float(gn)}")
        sched.step()
        tm = dict(data=t_data, enc=t_enc, head=t_head)
        for k, v in (("gate", lg_s / max(used, 1)), ("off", lo_s / max(used, 1)), ("pos", npos / max(npts, 1)),
                     ("pts", npts / max(used, 1)), ("tree", lt_s / max(used, 1)),
                     ("tacc", float(np.mean(t_acc)) if t_acc else 0.0), *tm.items()):
            run[k] = v if k not in run else 0.95 * run[k] + 0.05 * v
        if time.time() - last > PROGRESS_S or step == start + 1:
            last = time.time()
            rate = (time.time() - t0) / max(step - start, 1)
            mem = (f" | gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                   if dev.startswith("cuda") else "")
            log(f"step {step}/{STEPS} | gate {run['gate']:.4f} off {run['off'] * 100:.1f} cm"
                + (f" | tree {run['tree']:.4f} acc {run['tacc']:.3f}" if TREE else "")
                + f" | close {run['pos'] * 100:.2f}% "
                f"of {run['pts']:,.0f} inner pts | lr {sched.get_last_lr()[0]:.2e} | {rate:.2f} s/step "
                f"(data {run['data']:.2f} encoder {run['enc']:.2f} head {run['head']:.2f}), "
                f"eta {rate * (STEPS - step) / 3600:.1f} h{mem}"
                + (f" | skipped {skipped_crops} crops, {skipped_steps} steps" if skipped_crops or skipped_steps else ""))
        if step % EVAL_EVERY == 0 or step == STEPS:
            ev, det, score, nv = evaluate(runner, store, gt, tier, bbox, rng)
            report_eval(step, ev)
            if TREE and ev["tree"]["miou"] > best_miou:
                best_miou = ev["tree"]["miou"]
                save(BEST_MIOU, head, opt, step, best, mu, sd, ev, best_miou)
                log(f"  new best tree mIoU -> {BEST_MIOU}")
                tree_figure(ev["_region"], bbox, step, ev["tree"], RESULTS / "51_stemnet_v2_val_tree.png")
            ev.pop("_region", None)
            if ev["@0.5"]["f1"] > best:
                best = ev["@0.5"]["f1"]
                save(BEST, head, opt, step, best, mu, sd, ev, best_miou)
                log(f"  new best -> {BEST}")
                with open(RESULTS / f"stemnet_v2_val_detections{SUF}.csv", "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["x", "y", "score", "votes", "matched_base"])
                    for (x, y), s_, n_, pr in zip(det, score, nv, ev["_pair"]):
                        w.writerow([f"{x:.3f}", f"{y:.3f}", f"{s_:.2f}", int(n_), int(pr)])
                figure(gt, det, ev["_pair"], bbox, RESULTS / f"46_stemnet_v2_val{SUF}.png")
            save(LAST, head, opt, step, best, mu, sd, ev, best_miou)
        elif step % 1000 == 0:                       # a crash loses at most 1,000 steps
            save(LAST, head, opt, step, best, mu, sd, ev, best_miou)
    log(f"done: {STEPS} steps in {(time.time() - t0) / 3600:.1f} h | best val F1@0.5 {best:.3f} -> {BEST}"
        + (f" | best tree mIoU {best_miou:.3f} -> {BEST_MIOU}" if TREE else ""))


if __name__ == "__main__":
    main()
