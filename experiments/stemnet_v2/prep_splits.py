#!/usr/bin/env python3
"""StemNet v2 prep, part 4: where training may look, and where it may not.

A 1 m raster over the whole scan, every cell labelled:

  test    the girth-attributed patch: inside the hull of the 392 attributed
          trees, or within V2_TEST_MARGIN (3 m) of one. Never trained on,
          never validated on. Every survey row in it is test GT — attributed
          or not, since the surveyors walked the whole patch.
  val     a second held-out block (V2_VAL_MODE):
            match  (default) the smallest square, clear of the test zone and
                   its buffer, holding at least as many survey trees as the test
                   patch — so the two held-out scores have the same precision.
                   (The attributed patch is ~3x denser in trees than the campus
                   average, so a test-sized square elsewhere would not.)
            strip  the old way: the highest-x strip holding V2_VAL_FRAC (15%)
                   of the non-test trees
  buffer  within V2_BUFFER (10 m) of test or val. Belongs to nobody: a training
          crop that reached into it could see held-out trees. Its trees are kept
          and labelled `buffer`, never dropped.
  train   everything else.

Plus an occupancy raster (grid points per 1 m cell), and the fraction of the
training area where a V2_CROP-metre crop fits entirely inside `train` with at
least V2_OCC (45%) of its cells scanned — TreeLearn's gate — so the training
stage knows how much ground it really has.

Survey rows are assigned by the cell of their survey point. All of them are
kept; split is a label, never a filter.

Outputs (data/stemnet_v2/):
  regions.npz    split (uint8: 0 train, 1 val, 2 test, 3 buffer), occupancy
                 (int32), x0, y0, cell — the training crop sampler reads this
  splits.csv     one row per survey row: id, rtk x/y (shifted), split, and if part 3 has
                 run, the derived base, status and fork group
  results/stemnet_v2_splits.txt, 44_stemnet_v2_splits.png
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ATTRS, OUT, RESULTS, STEM_MAP, TILE, parse_stem_map, survey_to_local  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
CELL = 1.0
TEST_MARGIN = float(os.environ.get("V2_TEST_MARGIN", "3.0"))
BUFFER = float(os.environ.get("V2_BUFFER", "10.0"))
VAL_MODE = os.environ.get("V2_VAL_MODE", "match")
VAL_FRAC = float(os.environ.get("V2_VAL_FRAC", "0.15"))
VAL_STRIDE = float(os.environ.get("V2_VAL_STRIDE", "5.0"))
CROP = float(os.environ.get("V2_CROP", "8.0"))
OCC = float(os.environ.get("V2_OCC", "0.45"))
GRID_DIR = OUT / f"grid{GRID:g}"
FRAME_JSON = OUT / "frame.json"
TRAIN, VAL, TEST, BUF = 0, 1, 2, 3
NAMES = {TRAIN: "train", VAL: "val", TEST: "test", BUF: "buffer"}


def log(msg):
    print(f"[v2-split] {msg}", flush=True)


def occupancy():
    """Grid points per 1 m cell, from the storage tiles."""
    tiles = []
    for q in GRID_DIR.glob("tile_*.npz"):
        a, b = q.stem.split("_")[1:]
        tiles.append((int(a), int(b), q))
    if not tiles:
        raise SystemExit("[v2-split] no grid tiles — run v2_grid first")
    imax = max(t[0] for t in tiles) + 1
    jmax = max(t[1] for t in tiles) + 1
    k = int(round(TILE / CELL))
    occ = np.zeros((jmax * k, imax * k), np.int32)
    for n, (a, b, q) in enumerate(tiles):
        P = np.load(q)["xyz"]
        cx = np.floor(P[:, 0] / CELL).astype(np.int64)
        cy = np.floor(P[:, 1] / CELL).astype(np.int64)
        ok = (cx >= 0) & (cy >= 0) & (cx < occ.shape[1]) & (cy < occ.shape[0])
        np.add.at(occ, (cy[ok], cx[ok]), 1)
        if (n + 1) % 100 == 0 or n + 1 == len(tiles):
            log(f"occupancy: tile {n + 1}/{len(tiles)}")
    return occ


def test_mask(shape, att_xy):
    """Hull of the attributed trees, plus everything within TEST_MARGIN of one.
    The union, not the hull alone: the hull clips trees on the boundary."""
    from scipy.spatial import Delaunay, cKDTree
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    C = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5]) * CELL
    inside = Delaunay(att_xy).find_simplex(C) >= 0
    near = cKDTree(att_xy).query(C, distance_upper_bound=TEST_MARGIN)[0] <= TEST_MARGIN
    return (inside | near).reshape(shape)


def dilate(mask, metres):
    """Every cell within `metres` (Euclidean) of the mask."""
    from scipy import ndimage
    if not mask.any():
        return mask.copy()
    return ndimage.distance_transform_edt(~mask) * CELL <= metres


def box_sum(a, k):
    """Sum over every k x k window, value at the window's lower-left cell."""
    I = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
    I[1:, 1:] = a.cumsum(0).cumsum(1)
    return I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]


def cells_of(xy, shape):
    cx = np.clip(np.floor(xy[:, 0] / CELL).astype(int), 0, shape[1] - 1)
    cy = np.clip(np.floor(xy[:, 1] / CELL).astype(int), 0, shape[0] - 1)
    return cy, cx


def choose_val(split, occ, xy, n_test, test_area_cells):
    shape = split.shape
    blocked = dilate(split == TEST, BUFFER)
    if VAL_MODE == "strip":
        free = ~blocked[cells_of(xy, shape)]
        xs = np.sort(xy[free, 0])
        if not len(xs):
            raise SystemExit("[v2-split] no trees outside the test zone for a val strip")
        x_cut = xs[int((1 - VAL_FRAC) * (len(xs) - 1))]
        m = np.zeros(shape, bool)
        m[:, int(np.floor(x_cut / CELL)):] = True
        return m & ~blocked, dict(mode="strip", x_from_m=float(x_cut))
    # match: the SMALLEST square, clear of test + its buffer, holding at least
    # as many survey trees as the test patch. Tree count, not area, is what
    # sets a held-out score's precision, and the attributed patch is far denser
    # in trees than the campus average, so a test-sized square elsewhere would
    # hold a fraction of the trees and give a much noisier number.
    side0 = max(int(round(np.sqrt(test_area_cells))), 1)
    trees = np.zeros(shape, np.int64)
    cy, cx = cells_of(xy, shape)
    np.add.at(trees, (cy, cx), 1)
    step = max(int(round(VAL_STRIDE / CELL)), 1)
    for side in range(side0, min(shape) + 1, step):
        nb = box_sum(blocked.astype(np.int64), side)
        nt = box_sum(trees, side)
        no = box_sum((occ > 0).astype(np.int64), side)
        ok = (nb == 0) & (no >= 0.3 * side * side) & (nt >= n_test)
        sub = np.zeros_like(ok)
        sub[::step, ::step] = ok[::step, ::step]
        if sub.any():
            jj, ii = np.nonzero(sub)
            k = int(np.argmin(nt[jj, ii]))                 # closest from above
            j, i = int(jj[k]), int(ii[k])
            m = np.zeros(shape, bool)
            m[j:j + side, i:i + side] = True
            return m, dict(mode="match", side_m=side * CELL, x0_m=i * CELL, y0_m=j * CELL,
                           trees_in_square=int(nt[j, i]), test_trees=int(n_test))
    raise SystemExit(f"[v2-split] no square clear of the test zone holds {n_test} trees "
                     f"— use V2_VAL_MODE=strip")


def main():
    if not FRAME_JSON.exists():
        raise SystemExit("[v2-split] no frame.json — run v2_grid first")
    frame = json.loads(FRAME_JSON.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    sm = parse_stem_map(STEM_MAP)
    gt = survey_to_local(sm, frame["epsg"], frame["origin_m"])
    # in the scan's frame every survey point sits at survey + the measured
    # shift (v2_bases); the test zone and every tree's split are built there,
    # or the test patch's edges land 1.2 m away from its own trees
    shift = np.array(frame.get("bases", {}).get("shift_m", [0.0, 0.0]), np.float64)
    if "bases" not in frame:
        log("!! v2_bases has not run: using raw survey positions, which are ~1.2 m off the scan")
    else:
        log(f"survey positions shifted by ({shift[0]:+.2f}, {shift[1]:+.2f}) m, as v2_bases recorded")
    xy = gt[:, :2] + shift

    att = []
    with open(ATTRS) as f:
        for r in csv.DictReader(f):
            att.append((float(r["lat"]), float(r["lon"])))
    key = {(round(a, 8), round(b, 8)) for a, b in att}
    is_att = np.array([(round(float(a), 8), round(float(b), 8)) in key
                       for a, b in zip(sm.lat, sm.lon)])
    log(f"{len(xy):,} survey rows, {int(is_att.sum())} of them girth-attributed")

    occ = occupancy()
    split = np.full(occ.shape, TRAIN, np.uint8)
    tm = test_mask(occ.shape, xy[is_att])
    split[tm] = TEST
    cy, cx = cells_of(xy, occ.shape)
    n_test = int(tm[cy, cx].sum())
    vm, vinfo = choose_val(split, occ, xy, n_test, int(tm.sum()))
    split[vm & (split != TEST)] = VAL
    held = (split == TEST) | (split == VAL)
    split[dilate(held, BUFFER) & ~held] = BUF
    tree_split = split[cy, cx]

    # how much of `train` a training crop can actually use
    k = max(int(round(CROP / CELL)), 1)
    all_train = box_sum((split == TRAIN).astype(np.int64), k) == k * k
    scanned = box_sum((occ > 0).astype(np.int64), k) >= OCC * k * k
    eligible = all_train & scanned
    n_train_cells = int(((split == TRAIN) & (occ > 0)).sum())

    np.savez(OUT / "regions.npz", split=split, occupancy=occ,
             x0=0.0, y0=0.0, cell=CELL, names=np.array([NAMES[i] for i in range(4)]))

    derived = {}
    bp = OUT / "bases.csv"
    if bp.exists():
        with open(bp) as f:
            for r in csv.DictReader(f):
                derived[r["id"]] = r
    with open(OUT / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "block", "n", "rtk_x", "rtk_y", "attributed", "split",
                    "x", "y", "status", "fork_group"])
        for i in range(len(xy)):
            rid = f"{int(sm.block[i])}-{sm.n[i]}"
            d = derived.get(rid, {})
            w.writerow([rid, int(sm.block[i]), sm.n[i], f"{xy[i, 0]:.4f}", f"{xy[i, 1]:.4f}",
                        int(is_att[i]), NAMES[int(tree_split[i])],
                        d.get("x", ""), d.get("y", ""), d.get("status", ""), d.get("fork_group", "")])

    sc = occ > 0
    L = ["=" * 72, "StemNet v2 — splits", "=" * 72,
         f"survey rows {len(xy):,} (all kept) | raster {occ.shape[1]} x {occ.shape[0]} m, "
         f"{int(sc.sum()) / 1e4:.2f} ha scanned"]
    for s in (TRAIN, VAL, TEST, BUF):
        L.append(f"  {NAMES[s]:6s}  trees {int((tree_split == s).sum()):5,}   "
                 f"scanned area {int((sc & (split == s)).sum()) / 1e4:6.2f} ha")
    L.append(f"test = attributed hull + {TEST_MARGIN} m: {int((tree_split == TEST).sum())} trees, "
             f"{int((is_att & (tree_split == TEST)).sum())}/{int(is_att.sum())} attributed")
    L.append(f"val ({vinfo['mode']}): " + ", ".join(f"{k} {v}" for k, v in vinfo.items() if k != "mode"))
    L.append(f"buffer {BUFFER} m around test and val")
    L.append(f"training crops of {CROP:g} m that fit inside train with >= {OCC:.0%} scanned: "
             f"{int(eligible.sum()):,} positions (1 m stride) over {n_train_cells / 1e4:.2f} ha of "
             f"scanned train ground")
    txt = "\n".join(L)
    (RESULTS / "stemnet_v2_splits.txt").write_text(txt + "\n")
    print(txt, flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(12, 10))
    show = np.where(sc, split.astype(float), np.nan)
    ax.imshow(show, origin="lower", cmap=ListedColormap(["#9ecae1", "#fdae6b", "#e6550d", "#bdbdbd"]),
              vmin=-0.5, vmax=3.5, extent=[0, occ.shape[1], 0, occ.shape[0]], interpolation="nearest")
    for s, c in ((TRAIN, "#08519c"), (VAL, "#a63603"), (TEST, "k"), (BUF, "#525252")):
        m = tree_split == s
        ax.scatter(xy[m, 0], xy[m, 1], s=2, c=c, lw=0, label=f"{NAMES[s]} ({int(m.sum())})")
    ax.set_aspect("equal"); ax.legend(loc="upper right", markerscale=4)
    ax.set_title("splits: blue train, orange val, red test, grey buffer (scanned cells only)")
    fig.tight_layout(); fig.savefig(RESULTS / "44_stemnet_v2_splits.png", dpi=110); plt.close(fig)

    frame["splits"] = dict(file="splits.csv", regions="regions.npz", val=vinfo,
                           test_margin_m=TEST_MARGIN, buffer_m=BUFFER, crop_m=CROP, occ=OCC,
                           trees={NAMES[s]: int((tree_split == s).sum()) for s in NAMES},
                           eligible_crop_positions=int(eligible.sum()))
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log("-> data/stemnet_v2/regions.npz, splits.csv | results/stemnet_v2_splits.txt, 44_ png")


if __name__ == "__main__":
    main()
