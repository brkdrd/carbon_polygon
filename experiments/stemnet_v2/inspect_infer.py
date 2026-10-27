#!/usr/bin/env python3
"""StemNet v2: where the inference disagrees with B24 and with the survey, and
what is there. CPU; reads v2_infer's LAZ blocks, nothing is predicted again.

1. Tree head vs B24, by height above B24's own ground (hag, from the sem
   tiles; joined to the LAZ points by their exact millimetre coordinates) and
   by the head's confidence (does tree_prob mean what it says?).
2. Trunks of surveyed trees: an independent check of the tree class. Points
   on the bark ring of every base with a measured radius (|distance to the
   centre - radius| <= V2_INSPECT_RING, 5 cm) between 0.5 and 2 m above
   B24's ground: share called tree by the head, and by B24.
3. Bases on the test and val blocks: every detection's distance to the
   nearest survey base (within 0.5 m = found; 0.5-1.5 m near; further = a tree
   nobody surveyed, or a false detection).
4. Close-ups, V2_INSPECT_SPOTS (3) of each kind, V2_INSPECT_SIZE (12 m) squares:
     tree_fp    most points said tree where B24 says non-tree (above its ground)
     tree_fn    most points said non-tree where B24 says tree
     trunk      surveyed trunks the head calls tree the least
     base_far   test detections > 1.5 m from any survey base, highest vote mass
     base_near  test detections 0.5-1.5 m from a survey base, highest vote mass
   Each: from above, coloured by head vs B24 / scan RGB; a 2 m side slab
   through the centre coloured by head vs B24 / tree probability. Survey bases
   drawn as black circles (their radius), detections as magenta crosses.

Inputs: data/stemnet_v2/infer/<V2_INSPECT_RUN, head_tree_best>/block_*.laz,
results/stemnet_v2_infer_bases.csv, label_bases.csv, grid + sem tiles.
Outputs:
  results/stemnet_v2_inspect.txt
  results/55_stemnet_v2_inspect.png             error by height, confidence, trunk shares
  results/56_stemnet_v2_inspect_<kind>_<n>.png  the close-ups
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS, TILE  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
GRID_DIR, SEM_DIR = OUT / f"grid{GRID:g}", OUT / f"sem{GRID:g}"
RUN = os.environ.get("V2_INSPECT_RUN", "head_tree_best")
SPOTS = int(os.environ.get("V2_INSPECT_SPOTS", "3"))
SIZE = float(os.environ.get("V2_INSPECT_SIZE", "12"))
RING = float(os.environ.get("V2_INSPECT_RING", "0.05"))
CELL = 2.0                                               # error raster for choosing spots
HAG_BINS = (-np.inf, 0.3, 1.0, 2.0, 5.0, 10.0, np.inf)
SPLITS = {0: "train", 1: "val", 2: "test", 3: "buffer", 255: "outside"}
PAL = np.array([[0.13, 0.55, 0.13], [0.75, 0.75, 0.75], [0.85, 0.1, 0.1], [0.1, 0.3, 0.9], [0.95, 0.85, 0.3]])
CLS = ("tree, agree", "non-tree, agree", "said tree, B24 non-tree", "said non-tree, B24 tree", "no B24 label")
LOG = RESULTS / "stemnet_v2_inspect.txt"


def log(msg):
    line = f"[v2-inspect] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def split_at(split, xy):
    i, j = np.floor(xy[:, 0]).astype(np.int64), np.floor(xy[:, 1]).astype(np.int64)
    ok = (i >= 0) & (j >= 0) & (i < split.shape[1]) & (j < split.shape[0])
    out = np.full(len(xy), 255, np.uint8)
    out[ok] = split[j[ok], i[ok]]
    return out


def classes(tree, sem):
    """0 TP, 1 TN, 2 FP, 3 FN, 4 unlabelled (B24 ground counts as non-tree)."""
    pred, y, lab = tree >= 0.5, sem == 1, sem >= 0
    c = np.full(len(sem), 4, np.int8)
    c[lab & pred & y] = 0; c[lab & ~pred & ~y] = 1; c[lab & pred & ~y] = 2; c[lab & ~pred & y] = 3
    return c


def read_block(path, origin):
    import laspy
    las = laspy.read(str(path))
    d = dict(I=np.column_stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)]).astype(np.int64),
             xyz=np.column_stack([np.asarray(las.x) - origin[0], np.asarray(las.y) - origin[1],
                                  np.asarray(las.z) - origin[2]]),
             tree=np.asarray(las["tree_prob"], np.float32), sem=np.asarray(las["label_tree"], np.int8),
             split=np.asarray(las["split"], np.uint8),
             rgb=(np.column_stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)]) // 257).astype(np.uint8))
    d["scale"], d["offset"] = np.asarray(las.header.scales), np.asarray(las.header.offsets)
    return d


class Tiles:
    """Grid xyz + B24 hag per storage tile, a few kept."""

    def __init__(self):
        self.mem = {}

    def __call__(self, ij):
        if ij not in self.mem:
            g, s = GRID_DIR / f"tile_{ij[0]}_{ij[1]}.npz", SEM_DIR / f"tile_{ij[0]}_{ij[1]}.npz"
            self.mem[ij] = ((np.load(g)["xyz"], np.load(s)["hag"].astype(np.float32)) if g.exists() and s.exists()
                            else None)
            while len(self.mem) > 16:
                self.mem.pop(next(iter(self.mem)))
        return self.mem[ij]


def join_hag(b, tiles, origin):
    """B24 height above ground for each LAZ point: the grid point with the same
    stored millimetre coordinates (as write_laz computed them); 2 mm nearest
    neighbour for any that do not match exactly. -> hag (nan: none / unknown), matched."""
    from scipy.spatial import cKDTree
    lo, hi = b["xyz"].min(0), b["xyz"].max(0)
    G, H = [], []
    for a in range(int(np.floor(lo[0] / TILE)), int(np.floor(hi[0] / TILE)) + 1):
        for c in range(int(np.floor(lo[1] / TILE)), int(np.floor(hi[1] / TILE)) + 1):
            t = tiles((a, c))
            if t is not None:
                G.append(t[0]); H.append(t[1])
    hag = np.full(len(b["xyz"]), np.nan, np.float32)
    if not G:
        return hag, np.zeros(len(hag), bool)
    g, h = np.concatenate(G).astype(np.float64), np.concatenate(H)
    gi = np.round((g + origin - b["offset"]) / b["scale"]).astype(np.int64)
    base = np.minimum(gi.min(0), b["I"].min(0))
    span = np.maximum(gi.max(0), b["I"].max(0)) - base + 1

    def key(v):
        v = v - base
        return (v[:, 0] * span[1] + v[:, 1]) * span[2] + v[:, 2]
    gk, bk = key(gi), key(b["I"])
    o = np.argsort(gk)
    pos = np.clip(np.searchsorted(gk[o], bk), 0, len(gk) - 1)
    ok = gk[o][pos] == bk
    hag[ok] = h[o][pos[ok]]
    if (~ok).any():
        d, i = cKDTree(g).query(b["xyz"][~ok], distance_upper_bound=0.002)
        f = np.isfinite(d)
        idx = np.flatnonzero(~ok)
        hag[idx[f]] = h[i[f]]
        ok[idx[f]] = True
    return hag, ok


def box(a, k=3):
    from scipy import ndimage
    return ndimage.uniform_filter(a.astype(np.float64), k, mode="constant") * k * k


def pick(score, n, sep_cells):
    """Top-n cells of a raster, kept apart."""
    out = []
    for f in np.argsort(-score, axis=None):
        if len(out) >= n or score.flat[f] <= 0:
            break
        j, i = np.unravel_index(f, score.shape)
        if all(max(abs(i - a), abs(j - b_)) >= sep_cells for a, b_ in out):
            out.append((i, j))
    return out


def load_square(blocks, cx, cy, origin, h):
    parts = []
    for p, (x0, x1, y0, y1) in blocks:
        if x1 < cx - h or x0 > cx + h or y1 < cy - h or y0 > cy + h:
            continue
        b = read_block(p, origin)
        m = (np.abs(b["xyz"][:, 0] - cx) <= h) & (np.abs(b["xyz"][:, 1] - cy) <= h)
        parts.append({k: b[k][m] for k in ("xyz", "tree", "sem", "rgb")})
    if not parts:
        return None
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


def spot_figure(path, title, P, cx, cy, gt, det, px=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch
    h = SIZE / 2
    x0, y0 = cx - h, cy - h
    n = int(round(SIZE / px))
    xyz, cls = P["xyz"], classes(P["tree"], P["sem"])
    i = np.clip(((xyz[:, 0] - x0) / px).astype(np.int64), 0, n - 1)
    j = np.clip(((xyz[:, 1] - y0) / px).astype(np.int64), 0, n - 1)
    key = j * n + i
    o = np.lexsort((xyz[:, 2], key))
    last = np.r_[key[o][1:] != key[o][:-1], True]
    top_c = np.ones((n * n, 3)); top_c[key[o][last]] = PAL[cls[o][last]]
    top_rgb = np.ones((n * n, 3)); top_rgb[key[o][last]] = P["rgb"][o][last] / 255.0
    ext = (x0, x0 + SIZE, y0, y0 + SIZE)
    fig, axs = plt.subplots(1, 4, figsize=(26, 7))
    for ax, img, t in ((axs[0], top_c, "from above: head vs B24"), (axs[1], top_rgb, "from above: scan RGB")):
        ax.imshow(img.reshape(n, n, 3), origin="lower", extent=ext, interpolation="nearest")
        for (gx, gy, gr) in gt:
            ax.add_patch(Circle((gx, gy), max(gr, 0.05), fill=False, ec="black", lw=1.5))
        if len(det):
            ax.scatter(det[:, 0], det[:, 1], marker="x", s=60, c="magenta", lw=2)
        ax.set_xlim(x0, x0 + SIZE); ax.set_ylim(y0, y0 + SIZE); ax.set_title(t)
    s = np.abs(xyz[:, 1] - cy) <= 1.0
    zs = xyz[s]
    for ax, t in ((axs[2], "side, slab y +/- 1 m: head vs B24"), (axs[3], "side, slab y +/- 1 m: tree probability")):
        if t.endswith("B24"):
            oo = np.argsort((cls[s] >= 2) & (cls[s] <= 3), kind="stable")        # disagreements drawn last
            ax.scatter(zs[oo, 0], zs[oo, 2], s=0.3, c=PAL[cls[s][oo]], lw=0)
        else:
            sc = ax.scatter(zs[:, 0], zs[:, 2], s=0.3, c=P["tree"][s], cmap="viridis", vmin=0, vmax=1, lw=0)
            fig.colorbar(sc, ax=ax, shrink=0.8)
        for (gx, gy, gr) in gt:
            if abs(gy - cy) <= 1.0 + gr:
                ax.axvline(gx, c="black", lw=0.8, ls="--")
        for dx, dy in det:
            if abs(dy - cy) <= 1.0:
                ax.axvline(dx, c="magenta", lw=0.8, ls=":")
        ax.set_xlim(x0, x0 + SIZE); ax.set_aspect("equal", adjustable="datalim"); ax.set_title(t)
    axs[0].legend(handles=[Patch(color=c, label=l) for c, l in zip(PAL, CLS)], loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def summary_figure(path, hb, cal, trunk_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    lab = [f"{a:g}-{b:g}" if np.isfinite(a) and np.isfinite(b) else (f"< {b:g}" if np.isfinite(b) else f">= {a:g}")
           for a, b in zip(HAG_BINS[:-1], HAG_BINS[1:])] + ["unknown"]
    x = np.arange(len(lab))
    tp, fp, fn, tn = hb[:, 0], hb[:, 1], hb[:, 2], hb[:, 3]
    axs[0].bar(x - 0.2, fp / np.maximum(fp + tn, 1) * 100, 0.4, color="tab:red", label="said tree / B24 non-tree (% of non-tree)")
    axs[0].bar(x + 0.2, fn / np.maximum(fn + tp, 1) * 100, 0.4, color="tab:blue", label="said non-tree / B24 tree (% of tree)")
    axs[0].set_xticks(x, lab, rotation=30); axs[0].set_xlabel("height above B24 ground (m)"); axs[0].set_ylabel("%")
    axs[0].legend(fontsize=8); axs[0].set_title("disagreement by height (all splits)")
    c = (np.arange(10) + 0.5) / 10
    share = cal[:, 0] / np.maximum(cal.sum(1), 1)
    axs[1].plot([0, 1], [0, 1], c="grey", ls="--"); axs[1].plot(c, share, "o-")
    axs[1].set_xlabel("head tree probability"); axs[1].set_ylabel("share B24 tree"); axs[1].set_title("calibration")
    ax2 = axs[1].twinx(); ax2.bar(c, cal.sum(1), 0.08, alpha=0.2, color="k"); ax2.set_yscale("log"); ax2.set_ylabel("points")
    if trunk_rows:
        a = np.array([r[0] for r in trunk_rows]); b = np.array([r[1] for r in trunk_rows])
        axs[2].hist([a, b], bins=np.linspace(0, 1, 21), label=["head", "B24"], color=["tab:green", "tab:orange"])
        axs[2].set_xlabel("share of a surveyed trunk's bark points called tree"); axs[2].set_ylabel("trunks")
        axs[2].legend(); axs[2].set_title(f"surveyed trunks, 0.5-2 m up ({len(a)})")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    from scipy.spatial import cKDTree
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    t0 = time.time()
    frame = json.loads((OUT / "frame.json").read_text())
    origin = np.asarray(frame["origin_m"], np.float64)
    split = np.load(OUT / "regions.npz")["split"]
    run_dir = OUT / "infer" / RUN
    paths = sorted(run_dir.glob("block_*.laz"))
    if not paths:
        raise SystemExit(f"[v2-inspect] no LAZ blocks in {run_dir} — run v2_infer")
    import laspy
    blocks = []
    for p in paths:
        with laspy.open(str(p)) as f:
            hd = f.header
            if "tree_prob" not in hd.point_format.extra_dimension_names:
                raise SystemExit(f"[v2-inspect] {p.name} has no tree_prob — a base-only head; nothing to inspect")
            blocks.append((p, (hd.mins[0] - origin[0], hd.maxs[0] - origin[0], hd.mins[1] - origin[1], hd.maxs[1] - origin[1])))
    lb = list(csv.DictReader(open(OUT / "label_bases.csv")))
    gxy = np.array([[float(r["x"]), float(r["y"])] for r in lb]).reshape(-1, 2)
    grad = np.array([float(r["radius"]) for r in lb])
    gmeas = np.array([r["radius_measured"] == "1" for r in lb])
    gsplit = split_at(split, gxy)
    dets = list(csv.DictReader(open(RESULTS / "stemnet_v2_infer_bases.csv")))
    dxy = np.array([[float(r["x"]), float(r["y"])] for r in dets]).reshape(-1, 2)
    dscore = np.array([float(r["score"]) for r in dets])
    dsplit = np.array([r["split"] for r in dets])
    log(f"{len(paths)} LAZ blocks from {run_dir} | {len(lb):,} survey bases | {len(dets):,} detections")

    # pass 1: every block
    tiles = Tiles()
    nx, ny = int(np.ceil(split.shape[1] / CELL)), int(np.ceil(split.shape[0] / CELL))
    FP, FN = np.zeros((ny, nx)), np.zeros((ny, nx))
    hb = {s: np.zeros((len(HAG_BINS), 4), np.int64) for s in SPLITS}         # rows: hag bins + unknown
    cal = np.zeros((10, 2), np.int64)                                       # tree prob decile x (B24 tree, non-tree)
    ring = np.zeros((len(lb), 3), np.int64)                                 # bark points, head tree, B24 tree
    kd = cKDTree(gxy) if len(gxy) else None
    n_pts = n_ok = 0
    for k, (p, _) in enumerate(blocks):
        b = read_block(p, origin)
        hag, ok = join_hag(b, tiles, origin)
        n_pts += len(hag); n_ok += int(ok.sum())
        pred, sem = b["tree"] >= 0.5, b["sem"]
        lab = sem >= 0
        hbin = np.where(np.isfinite(hag), np.digitize(hag, HAG_BINS[1:-1]), len(HAG_BINS) - 1)
        conf = np.stack([pred & (sem == 1), pred & lab & (sem != 1), ~pred & (sem == 1), ~pred & lab & (sem != 1)], 1)
        for s in np.unique(b["split"]):
            m = b["split"] == s
            np.add.at(hb[int(s)], hbin[m], conf[m].astype(np.int64))
        dec = np.clip((b["tree"] * 10).astype(np.int64), 0, 9)
        np.add.at(cal, (dec[lab], (sem[lab] != 1).astype(np.int64)), 1)
        ci = np.clip((b["xyz"][:, 0] / CELL).astype(np.int64), 0, nx - 1)
        cj = np.clip((b["xyz"][:, 1] / CELL).astype(np.int64), 0, ny - 1)
        fp = pred & (sem == 0); fn = ~pred & (sem == 1)                     # above B24's ground only
        np.add.at(FP, (cj[fp], ci[fp]), 1); np.add.at(FN, (cj[fn], ci[fn]), 1)
        if kd is not None:
            band = np.isfinite(hag) & (hag >= 0.5) & (hag <= 2.0)
            idx = np.flatnonzero(band)
            d, gi = kd.query(b["xyz"][idx, :2], distance_upper_bound=float(grad.max()) + RING + 0.01)
            f = np.isfinite(d)
            idx, d, gi = idx[f], d[f], gi[f]
            on = gmeas[gi] & (np.abs(d - grad[gi]) <= RING)
            idx, gi = idx[on], gi[on]
            np.add.at(ring, gi, np.stack([np.ones(len(idx), np.int64), pred[idx].astype(np.int64),
                                          (sem[idx] == 1).astype(np.int64)], 1))
        if (k + 1) % 10 == 0 or k + 1 == len(blocks):
            el = time.time() - t0
            log(f"block {k + 1}/{len(blocks)} | {n_pts:,} pts | {el / 60:.1f} min, eta {el / (k + 1) * (len(blocks) - k - 1) / 60:.1f} min")
    log(f"joined to B24's height above ground: {n_ok:,} of {n_pts:,} points ({n_ok / max(n_pts, 1) * 100:.2f}%)")

    # 1. by height, by confidence
    lab_names = [f"{a:g}..{b:g} m" for a, b in zip(HAG_BINS[:-1], HAG_BINS[1:])] + ["no B24 ground"]
    for s in (0, 1, 2):
        log(f"{SPLITS[s]}: head vs B24 by height above B24's ground (FP = said tree / B24 non-tree, FN = said non-tree / B24 tree):")
        for r, nm in enumerate(lab_names):
            tp, fp, fn, tn = (int(v) for v in hb[s][r])
            n = tp + fp + fn + tn
            if n:
                log(f"  {nm:>14}: {n:>12,} labelled, {tp + fn:>12,} B24 tree ({(tp + fn) / n * 100:5.1f}%) | "
                    f"FP {fp / max(fp + tn, 1) * 100:5.2f}% of non-tree | FN {fn / max(fn + tp, 1) * 100:5.2f}% of tree | "
                    f"disagree {(fp + fn) / n * 100:5.2f}%")
    log("calibration (all splits): head tree probability -> share of those points B24 calls tree")
    for d_ in range(10):
        n = int(cal[d_].sum())
        if n:
            log(f"  {d_ / 10:.1f}-{(d_ + 1) / 10:.1f}: {n:>12,} pts, {cal[d_, 0] / n * 100:5.1f}% B24 tree")

    # 2. surveyed trunks
    trunk_rows, worst = [], []
    for s in (0, 1, 2, 3):
        m = (gsplit == s) & (ring[:, 0] >= 20)
        if not m.any():
            continue
        r = ring[m]
        log(f"surveyed trunks, {SPLITS[s]}: {m.sum()} with >= 20 bark points 0.5-2 m up (of {(gmeas & (gsplit == s)).sum()} "
            f"with a measured radius) | head calls tree {r[:, 1].sum() / r[:, 0].sum() * 100:.1f}% of {r[:, 0].sum():,} "
            f"bark points, B24 {r[:, 2].sum() / r[:, 0].sum() * 100:.1f}% | trunks >= 90% tree: head "
            f"{(r[:, 1] / r[:, 0] >= 0.9).mean() * 100:.1f}%, B24 {(r[:, 2] / r[:, 0] >= 0.9).mean() * 100:.1f}%")
    m = ring[:, 0] >= 20
    trunk_rows = list(zip(ring[m, 1] / ring[m, 0], ring[m, 2] / ring[m, 0]))
    summary_figure(RESULTS / "55_stemnet_v2_inspect.png", hb[0] + hb[1] + hb[2] + hb[3] + hb[255], cal, trunk_rows)

    # 3. bases on val / test vs the survey
    for s in ("val", "test"):
        m = dsplit == s
        if not m.any() or kd is None:
            continue
        d, _ = kd.query(dxy[m])
        bins = (0, 0.5, 1.0, 1.5, 2.5, 5.0, np.inf)
        h, _ = np.histogram(d, bins)
        log(f"{s} detections ({m.sum()}), distance to the nearest survey base: " + " | ".join(
            f"{a:g}-{b:g} m {n} (vote mass median {np.median(dscore[m][(d >= a) & (d < b)]) if n else 0:.0f})"
            for a, b, n in zip(bins[:-1], bins[1:], h)))

    # 4. close-ups
    spots = []
    sep = int(np.ceil(SIZE / CELL))
    for kind, R_ in (("tree_fp", FP), ("tree_fn", FN)):
        for (i, j) in pick(box(R_), SPOTS, sep):
            spots.append((kind, (i + 0.5) * CELL, (j + 0.5) * CELL, f"{int(box(R_)[j, i]):,} disagreeing points within 3 m"))
    m = ring[:, 0] >= 20
    order = np.flatnonzero(m)[np.argsort(ring[m, 1] / ring[m, 0])]
    for g in order:
        if sum(1 for s_ in spots if s_[0] == "trunk") >= SPOTS:
            break
        if all(max(abs(gxy[g, 0] - s_[1]), abs(gxy[g, 1] - s_[2])) >= SIZE for s_ in spots if s_[0] == "trunk"):
            spots.append(("trunk", gxy[g, 0], gxy[g, 1], f"surveyed trunk r {grad[g] * 100:.0f} cm: head tree "
                          f"{ring[g, 1] / ring[g, 0] * 100:.0f}%, B24 {ring[g, 2] / ring[g, 0] * 100:.0f}% of {ring[g, 0]} bark points"))
    if kd is not None:
        t = np.flatnonzero(dsplit == "test")
        d, _ = kd.query(dxy[t]) if len(t) else (np.zeros(0), None)
        for kind, lo, hi in (("base_far", 1.5, np.inf), ("base_near", 0.5, 1.5)):
            c = t[(d >= lo) & (d < hi)]
            c = c[np.argsort(-dscore[c])]
            got = []
            for q in c:
                if len(got) >= SPOTS:
                    break
                if all(max(abs(dxy[q, 0] - dxy[g, 0]), abs(dxy[q, 1] - dxy[g, 1])) >= SIZE for g in got):
                    got.append(q)
                    spots.append((kind, dxy[q, 0], dxy[q, 1], f"test detection, vote mass {dscore[q]:.0f}, nearest survey "
                                  f"base {d[np.flatnonzero(t == q)[0]]:.2f} m"))
    h = SIZE / 2
    count = {}
    for kind, cx, cy, why in spots:
        count[kind] = count.get(kind, 0) + 1
        P = load_square(blocks, cx, cy, origin, h)
        if P is None:
            continue
        gm = (np.abs(gxy[:, 0] - cx) <= h) & (np.abs(gxy[:, 1] - cy) <= h)
        dm = (np.abs(dxy[:, 0] - cx) <= h) & (np.abs(dxy[:, 1] - cy) <= h)
        sp = SPLITS[int(split_at(split, np.array([[cx, cy]]))[0])]
        title = (f"{kind} {count[kind]}: centre ({cx:.1f}, {cy:.1f}) frame m = UTM ({cx + origin[0]:.1f}, {cy + origin[1]:.1f}), "
                 f"{sp} | {why} | {len(P['xyz']):,} points")
        name = f"56_stemnet_v2_inspect_{kind}_{count[kind]}.png"
        spot_figure(RESULTS / name, title, P, cx, cy, [(x, y, r) for (x, y), r in zip(gxy[gm], grad[gm])], dxy[dm])
        log(f"{name}: {title}")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/stemnet_v2_inspect.txt, 55_stemnet_v2_inspect.png, "
        f"56_stemnet_v2_inspect_*.png")


if __name__ == "__main__":
    main()
