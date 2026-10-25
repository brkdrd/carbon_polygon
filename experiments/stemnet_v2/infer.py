#!/usr/bin/env python3
"""StemNet v2 inference: the trained head over the whole campus (or a box).

The area is cut into V2_INFER_BLOCK (50 m) blocks. Each block is predicted with
V2_INFER_MARGIN (4 m) extra on every side (overlapping 8 m crops, inner 6 m
counted, as in training's evaluation), so a point near a block edge still gets
full crop context and a tree on the edge gets all its votes. A block keeps
only its own points and the bases detected inside it: every point is written
once, every base found once.

Detection settings: models/detect_params.json if v2_validate wrote it for this
checkpoint (tuned on val), else the defaults in vote.py — the log says which.

Checkpoint: V2_CKPT (a file name in models/), default head_tree_best.pth, else
head_best.pth.

Area: V2_INFER_BBOX "x0,x1,y0,y1" (frame metres), default every scanned 1 m
cell of the campus.

Outputs:
  data/stemnet_v2/infer/<checkpoint>/block_<x0>_<y0>.laz   (V2_INFER_LAZ=0 skips)
      every grid point, UTM x / y / z and scan RGB, plus
        tree_prob   tree probability (tree head; absent for a base-only head)
        pred_tree   1 where tree_prob >= 0.5
        gate_prob   "close to a tree base" probability
        label_tree  campus_pred_B24 label carried over (1 tree, 0 non-tree,
                    2 removed ground, -1 unknown) — to compare, not used
        split       0 train, 1 val, 2 test, 3 buffer, 255 outside
  results/stemnet_v2_infer_bases.csv    every detected base: frame and UTM x / y,
                                        score (vote mass), votes, split
  results/stemnet_v2_infer.txt          progress; tree head vs the B24 labels per
                                        split; survey bases recalled per split
  results/52_stemnet_v2_infer_map.png   from above: tree probability of the
                                        highest point per pixel, bases on top
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
import data as D  # noqa: E402
import model as M  # noqa: E402
import train as T  # noqa: E402
import vote as V  # noqa: E402

BLOCK = float(os.environ.get("V2_INFER_BLOCK", "50"))
MARGIN = float(os.environ.get("V2_INFER_MARGIN", "4"))
BBOX = os.environ.get("V2_INFER_BBOX", "")
WRITE_LAZ = os.environ.get("V2_INFER_LAZ", "1") != "0"
KEEP = int(os.environ.get("V2_INFER_KEEP", "24"))      # tiles held in memory (~300 MB each)
MAP_PX = float(os.environ.get("V2_INFER_MAP_PX", "0.25"))
PARAMS = T.MODELS / "detect_params.json"
SPLITS = {0: "train", 1: "val", 2: "test", 3: "buffer", 255: "outside"}
LOG = RESULTS / "stemnet_v2_infer.txt"


def log(msg):
    line = f"[v2-infer] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def checkpoint():
    name = os.environ.get("V2_CKPT", "")
    cands = [name] if name else ["head_tree_best.pth", "head_best.pth"]
    for c in cands:
        p = Path(c) if os.sep in c else T.MODELS / c
        if p.exists():
            return p
    raise SystemExit(f"[v2] no checkpoint ({', '.join(cands)}) in {T.MODELS} — run v2_train_tree")


def load_runner(log):
    """-> (Runner, checkpoint dict, checkpoint path), head in eval mode on V2_DEVICE."""
    dev = T.DEVICE
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[v2] V2_DEVICE=cuda but no GPU is visible in the container — check the NVIDIA runtime")
    path = checkpoint()
    ck = torch.load(path, map_location=dev, weights_only=False)
    head = M.Head(ck["d_in"], tree=ck.get("task") == "base+tree").to(dev)
    head.load_state_dict(ck["head"])
    head.eval()
    enc = M.load_sonata(dev) if T.ENCODER == "sonata" else M.FakeEncoder()
    ev = ck.get("eval") or {}
    log(f"checkpoint {path.name}: step {ck['step']}, task {ck.get('task', 'base')}, best val F1@0.5 "
        f"{ck['best']:.3f}" + (f", tree mIoU {ev['tree']['miou']:.3f}" if "tree" in ev else "") + f" | device {dev}")
    return T.Runner(enc, head, ck["mu"], ck["sd"], dev), ck, path


def detect_params(ckpt_path, log):
    """Val-tuned settings for this checkpoint, else vote.py's defaults."""
    default = dict(p_thresh=V.P_THRESH, peak_sep=V.PEAK_SEP, peak_min=V.PEAK_MIN)
    if PARAMS.exists():
        d = json.loads(PARAMS.read_text())
        if d.get("checkpoint") == ckpt_path.name:
            log(f"detection settings tuned on val (val F1@0.5 {d['val_f1']:.3f}): gate >= {d['p_thresh']:g}, "
                f"peaks {d['peak_sep']:g} m apart, vote mass >= {d['peak_min']:g}")
            return {k: d[k] for k in default}
        log(f"!! {PARAMS.name} is for {d.get('checkpoint')}, not {ckpt_path.name} — using the defaults")
    else:
        log(f"!! no {PARAMS.name} (run v2_validate) — using the default detection settings")
    log(f"   gate >= {default['p_thresh']:g}, peaks {default['peak_sep']:g} m apart, vote mass >= {default['peak_min']:g}")
    return default


def split_at(regions, xy):
    """Split id of the 1 m cell under each xy (255 outside the raster)."""
    split = regions["split"]
    i, j = np.floor(xy[:, 0]).astype(np.int64), np.floor(xy[:, 1]).astype(np.int64)
    ok = (i >= 0) & (j >= 0) & (i < split.shape[1]) & (j < split.shape[0])
    out = np.full(len(xy), 255, np.uint8)
    out[ok] = split[j[ok], i[ok]]
    return out


def area(regions):
    if BBOX:
        return tuple(float(v) for v in BBOX.split(","))
    jj, ii = np.nonzero(regions["occupancy"] > 0)
    return float(ii.min()), float(ii.max() + 1), float(jj.min()), float(jj.max() + 1)


def blocks(bbox, occ, block=None):
    """Block cores (half-open), serpentine so consecutive blocks share tiles; empty ones skipped."""
    block = block or BLOCK
    x0, x1, y0, y1 = bbox
    out = []
    for n, bx in enumerate(np.arange(x0, x1, block)):
        ys = np.arange(y0, y1, block)
        for by in (ys if n % 2 == 0 else ys[::-1]):
            core = (float(bx), float(min(bx + block, x1)), float(by), float(min(by + block, y1)))
            c = occ[max(int(np.floor(core[2])), 0):max(int(np.ceil(core[3])), 0),
                    max(int(np.floor(core[0])), 0):max(int(np.ceil(core[1])), 0)]
            if c.size and c.sum() > 0:
                out.append(core)
    return out


def iter_blocks(predict, store, cores, params, rng, margin=None, cap=None, detect=True):
    """Per block: (core, R, inside, bases, score, votes). R is predict_region's
    full dict for the block plus margin; `inside` marks its own points; the
    bases are those detected (on all of R) inside the core; none with detect=False."""
    margin = MARGIN if margin is None else margin
    cap = cap or T.CAP
    for core in cores:
        big = (core[0] - margin, core[1] + margin, core[2] - margin, core[3] + margin)
        R = V.predict_region(predict, store, big, cap, rng, full=True)
        xy = R["xy"]
        inside = (xy[:, 0] >= core[0]) & (xy[:, 0] < core[1]) & (xy[:, 1] >= core[2]) & (xy[:, 1] < core[3])
        det, score, nv = V.detect(xy, R["p"], R["off"], **params) if detect else V.detect(xy[:0], R["p"][:0], R["off"][:0])
        k = ((det[:, 0] >= core[0]) & (det[:, 0] < core[1]) & (det[:, 1] >= core[2]) & (det[:, 1] < core[3])
             if len(det) else np.zeros(0, bool))
        yield core, R, inside, det[k], score[k], nv[k]


class Rgb:
    """Scan RGB per grid tile row (TileStore rows are grid rows)."""

    def __init__(self):
        self.mem = {}

    def __call__(self, tile, row):
        out = np.zeros((len(tile), 3), np.uint8)
        for t in np.unique(tile):
            if t not in self.mem:
                q = D.GRID_DIR / f"tile_{int(t) // 100000}_{int(t) % 100000}.npz"
                self.mem[t] = np.load(q)["rgb"] if q.exists() else None
                while len(self.mem) > 32:
                    self.mem.pop(next(iter(self.mem)))
            if self.mem[t] is not None:
                m = tile == t
                out[m] = self.mem[t][row[m]]
        return out


def write_laz(path, R, k, rgb, split, origin, has_tree):
    import laspy
    hdr = laspy.LasHeader(point_format=7, version="1.4")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.floor(np.asarray(origin, np.float64))
    extra = [("gate_prob", np.float32), ("label_tree", np.int8), ("split", np.uint8)]
    if has_tree:
        extra = [("tree_prob", np.float32), ("pred_tree", np.uint8)] + extra
    hdr.add_extra_dims([laspy.ExtraBytesParams(name=n, type=t) for n, t in extra])
    las = laspy.LasData(hdr)
    las.x = R["xy"][k, 0].astype(np.float64) + origin[0]
    las.y = R["xy"][k, 1].astype(np.float64) + origin[1]
    las.z = R["z"][k].astype(np.float64) + origin[2]
    c = rgb.astype(np.uint16) * 257
    las.red, las.green, las.blue = c[:, 0], c[:, 1], c[:, 2]
    if has_tree:
        las.tree_prob = R["tree"][k].astype(np.float32)
        las.pred_tree = (R["tree"][k] >= 0.5).astype(np.uint8)
    las.gate_prob = R["p"][k].astype(np.float32)
    las.label_tree = R["sem"][k].astype(np.int8)
    las.split = split
    las.write(str(path))


def counts(pt, sem):
    """Confusion counts (tp, fp, fn, tn) of tree_prob >= 0.5 vs labelled points (sem >= 0, tree = 1)."""
    k = sem >= 0
    pred, y = pt[k] >= 0.5, sem[k] == 1
    return np.array([(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()], np.int64)


def from_counts(c):
    tp, fp, fn, tn = (int(v) for v in c)
    it, inn = tp / max(tp + fp + fn, 1), tn / max(tn + fp + fn, 1)
    n = tp + fp + fn + tn
    return dict(points=n, acc=(tp + tn) / max(n, 1), iou_tree=it, iou_non=inn, miou=(it + inn) / 2,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), tree_share=(tp + fn) / max(n, 1))


def tree_line(t):
    return (f"mIoU {t['miou']:.3f} (IoU tree {t['iou_tree']:.3f}, non-tree {t['iou_non']:.3f}) | acc {t['acc']:.3f} | "
            f"tree P {t['precision']:.3f} R {t['recall']:.3f} | {t['points']:,} pts, {t['tree_share'] * 100:.1f}% tree")


def figure_map(top_p, bases, bbox, path, has_tree):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x0, x1, y0, y1 = bbox
    fig, ax = plt.subplots(figsize=(14, 14 * (y1 - y0) / max(x1 - x0, 1)))
    cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_bad("white")
    im = ax.imshow(top_p, origin="lower", extent=bbox, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.6, label="tree probability" if has_tree else "gate probability")
    if len(bases):
        ax.scatter(bases[:, 0], bases[:, 1], s=3, c="tab:red", lw=0, label=f"detected bases ({len(bases):,})")
        ax.legend(loc="upper right")
    ax.set_title(f"StemNet v2 inference: highest point per {MAP_PX * 100:g} cm pixel (frame m), white = no point")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    frame = json.loads((OUT / "frame.json").read_text())
    origin = np.asarray(frame["origin_m"], np.float64)
    regions = np.load(OUT / "regions.npz")
    runner, ck, cpath = load_runner(log)
    has_tree = runner.head.has_tree
    params = detect_params(cpath, log)
    bbox = area(regions)
    cores = blocks(bbox, regions["occupancy"])
    out_dir = OUT / "infer" / cpath.stem
    if WRITE_LAZ:
        out_dir.mkdir(parents=True, exist_ok=True)
    log(f"area {bbox} (frame m) | {len(cores)} blocks of {BLOCK:g} m + {MARGIN:g} m margin"
        + (f" | LAZ -> {out_dir}" if WRITE_LAZ else " | no LAZ (V2_INFER_LAZ=0)"))
    store, rgb_of, rng = D.TileStore(keep=KEEP), Rgb(), np.random.default_rng(T.SEED)
    nx, ny = int(np.ceil((bbox[1] - bbox[0]) / MAP_PX)), int(np.ceil((bbox[3] - bbox[2]) / MAP_PX))
    top_z = np.full(ny * nx, -np.inf, np.float32); top_p = np.full(ny * nx, np.nan, np.float32)
    tc = {(s, v): np.zeros(4, np.int64) for s in SPLITS for v in ("all", "matched")}
    B, n_pts, t0 = [], 0, time.time()
    for k, (core, R, inside, det, score, nv) in enumerate(iter_blocks(runner.predict, store, cores, params, rng)):
        n = int(inside.sum()); n_pts += n
        sp = split_at(regions, R["xy"][inside])
        if n:
            pt = R["tree"][inside] if has_tree else R["p"][inside]
            if has_tree:
                sem = R["sem"][inside]
                for s in np.unique(sp):
                    m = sp == s
                    tc[(int(s), "all")] += counts(pt[m], sem[m])
                    tc[(int(s), "matched")] += counts(pt[m], np.where(sem[m] == 2, -1, sem[m]))
            xy = R["xy"][inside]
            i = np.clip(((xy[:, 0] - bbox[0]) / MAP_PX).astype(np.int64), 0, nx - 1)
            j = np.clip(((xy[:, 1] - bbox[2]) / MAP_PX).astype(np.int64), 0, ny - 1)
            key = j * nx + i
            o = np.lexsort((R["z"][inside], key))
            last = np.r_[key[o][1:] != key[o][:-1], True]
            kk, zz, pp = key[o][last], R["z"][inside][o][last], pt[o][last]
            up = zz > top_z[kk]                      # blocks align with pixels only when BLOCK / MAP_PX is whole
            top_z[kk[up]] = zz[up]; top_p[kk[up]] = pp[up]
            if WRITE_LAZ:
                write_laz(out_dir / f"block_{core[0]:.0f}_{core[2]:.0f}.laz", R, inside,
                          rgb_of(R["tile"][inside], R["row"][inside]), sp, origin, has_tree)
        for (x, y), s_, v_ in zip(det, score, nv):
            B.append((x, y, s_, int(v_)))
        el = time.time() - t0
        log(f"block {k + 1}/{len(cores)} x {core[0]:.0f}-{core[1]:.0f} y {core[2]:.0f}-{core[3]:.0f}: {n:,} pts, "
            f"{len(det)} bases" + (f", {(R['tree'][inside] >= 0.5).mean() * 100:.0f}% tree" if has_tree and n else "")
            + f" | {el / 60:.1f} min, eta {el / (k + 1) * (len(cores) - k - 1) / 60:.1f} min")

    bases = np.array([b[:2] for b in B]).reshape(-1, 2)
    bsplit = split_at(regions, bases)
    with open(RESULTS / "stemnet_v2_infer_bases.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "x", "y", "utm_x", "utm_y", "score", "votes", "split"])
        for n_, ((x, y, s_, v_), sid) in enumerate(zip(B, bsplit)):
            w.writerow([n_, f"{x:.3f}", f"{y:.3f}", f"{x + origin[0]:.3f}", f"{y + origin[1]:.3f}", f"{s_:.2f}", v_,
                        SPLITS[int(sid)]])
    figure_map(top_p.reshape(ny, nx), bases, bbox, RESULTS / "52_stemnet_v2_infer_map.png", has_tree)

    log(f"done: {n_pts:,} points, {len(B):,} bases in {(time.time() - t0) / 60:.1f} min")
    if has_tree:
        log("tree head vs the campus_pred_B24 labels (B24 is itself a model's output: agreement, not truth; "
            "train was trained on, val picked the checkpoint, test is held out):")
        for s, nm in SPLITS.items():
            for v, vn in (("all", "all labelled"), ("matched", "matched only")):
                if tc[(s, v)].sum():
                    log(f"  {nm:<7} {vn:<13} {tree_line(from_counts(tc[(s, v)]))}")
    gt = np.array([[float(r["x"]), float(r["y"])] for r in csv.DictReader(open(OUT / "label_bases.csv"))]).reshape(-1, 2)
    if len(gt):
        gsplit = split_at(regions, gt)
        x0, x1, y0, y1 = bbox
        inb = (gt[:, 0] >= x0) & (gt[:, 0] < x1) & (gt[:, 1] >= y0) & (gt[:, 1] < y1)
        mm = V.match(bases, np.array([b[2] for b in B]), gt[inb], 0.5)
        found = np.zeros(inb.sum(), bool); found[mm["pair"][mm["pair"] >= 0]] = True
        log("survey bases with a detection within 0.5 m (recall only: unsurveyed trees make precision meaningless "
            "outside val / test — v2_validate scores those):")
        for s, nm in SPLITS.items():
            m = gsplit[inb] == s
            if m.any():
                log(f"  {nm:<7} {found[m].mean():.3f} of {m.sum():,} bases | {(bsplit == s).sum():,} detections")
    log("-> results/stemnet_v2_infer_bases.csv, 52_stemnet_v2_infer_map.png"
        + (f", {out_dir}/block_*.laz" if WRITE_LAZ else ""))


if __name__ == "__main__":
    main()
