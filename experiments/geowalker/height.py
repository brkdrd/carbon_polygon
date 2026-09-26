#!/usr/bin/env python3
"""Canopy height above the walker — the one thing the 2 m sphere cannot see.

The people who built the stem map decided what counted as a tree partly by
HEIGHT. GeoWalker has no access to that: its encoder crop is a 2 m sphere and
each crop is shifted by its own z.min(), so the model sees at most ~4 m of
vertical structure measured from the bottom of its own window. A 25 m oak, a
sapling, a hedge and a fence post are nearly the same object at that scale —
which is exactly the confusion a height criterion was invented to resolve.

This builds a canopy height model (highest cloud return per cell, minus the
ground DEM) and then does two things with it:

  * DERIVES the annotators' effective threshold, instead of guessing it. Every
    labelled base passed their test, so the low tail of "canopy height above a
    labelled base" is where their cut fell.
  * MEASURES how many current detections stand under something too short to
    have been labelled — i.e. how much of the false-positive count is this
    blind spot rather than genuine error.

Writes /data/geowalker/chm.npz, reusable as a model input or a post-hoc filter.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# this directory first (its siblings are model.py/geodesic.py), then /app
# for the shared modules — /app also holds BaseWalker's train.py/infer.py,
# which must never shadow ours.
sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402

BW = C.DATA / "basewalker"
GW = C.DATA / "geowalker"
CHM_NPZ = GW / "chm.npz"

CELL = float(os.environ.get("GW_CHM_CELL", "0.5"))    # raster resolution, m
SMOOTH_R = float(os.environ.get("GW_CHM_R", "1.5"))   # max-pool radius for lookup


def build_chm():
    """Highest return per CELL-metre cell, minus the ground DEM = canopy height.

    Streams the tiles so the full cloud never sits in memory at once.
    """
    dem = np.load(BW / "dem.npz", allow_pickle=True)
    dz, dx0, dy0, dcell = dem["z"], float(dem["x0"]), float(dem["y0"]), float(dem["cell"])

    tiles = sorted((BW / "tiles").glob("tile_*.npz"))
    if not tiles:
        raise SystemExit(f"[gw-chm] no tiles in {BW/'tiles'} — run bw_prep first")

    lo = np.array([np.inf, np.inf]); hi = np.array([-np.inf, -np.inf])
    for p in tiles:
        xy = np.load(p)["xyz"][:, :2]
        lo = np.minimum(lo, xy.min(0)); hi = np.maximum(hi, xy.max(0))
    nx = int((hi[0] - lo[0]) / CELL) + 1
    ny = int((hi[1] - lo[1]) / CELL) + 1
    top = np.full((ny, nx), -np.inf, np.float32)
    print(f"[gw-chm] raster {nx} x {ny} @ {CELL} m from {len(tiles)} tiles")

    for p in tiles:
        xyz = np.load(p)["xyz"].astype(np.float64)
        ix = ((xyz[:, 0] - lo[0]) / CELL).astype(np.int64).clip(0, nx - 1)
        iy = ((xyz[:, 1] - lo[1]) / CELL).astype(np.int64).clip(0, ny - 1)
        np.maximum.at(top, (iy, ix), xyz[:, 2].astype(np.float32))

    # ground under each cell, from the DEM raster
    cx = lo[0] + (np.arange(nx) + 0.5) * CELL
    cy = lo[1] + (np.arange(ny) + 0.5) * CELL
    gi = ((cx - dx0) / dcell).astype(np.int64).clip(0, dz.shape[1] - 1)
    gj = ((cy - dy0) / dcell).astype(np.int64).clip(0, dz.shape[0] - 1)
    ground = dz[np.ix_(gj, gi)]

    chm = np.where(np.isfinite(top), top - ground, np.nan).astype(np.float32)
    ok = np.isfinite(chm)
    print(f"[gw-chm] {ok.mean()*100:.1f}% of cells occupied | "
          f"height p50 {np.nanpercentile(chm,50):.1f} m "
          f"p95 {np.nanpercentile(chm,95):.1f} m "
          f"max {np.nanmax(chm):.1f} m")
    return dict(chm=chm, x0=float(lo[0]), y0=float(lo[1]), cell=CELL)


def height_at(c, xy, r=SMOOTH_R):
    """Tallest canopy within r metres of each xy — a walker standing beside a
    trunk should still be credited with the tree it is standing at."""
    chm, x0, y0, cell = c["chm"], float(c["x0"]), float(c["y0"]), float(c["cell"])
    ny, nx = chm.shape
    k = max(0, int(round(r / cell)))
    ix = ((xy[:, 0] - x0) / cell).astype(np.int64)
    iy = ((xy[:, 1] - y0) / cell).astype(np.int64)
    out = np.full(len(xy), np.nan, np.float32)
    for n, (i, j) in enumerate(zip(ix, iy)):
        i0, i1 = max(0, i - k), min(nx, i + k + 1)
        j0, j1 = max(0, j - k), min(ny, j + k + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        w = chm[j0:j1, i0:i1]
        if np.isfinite(w).any():
            out[n] = np.nanmax(w)
    return out


def main():
    GW.mkdir(parents=True, exist_ok=True)
    if CHM_NPZ.exists() and not os.environ.get("FORCE_GW_CHM"):
        print(f"[gw-chm] {CHM_NPZ} exists — reuse (FORCE_GW_CHM=1 rebuilds)")
        c = dict(np.load(CHM_NPZ))
    else:
        c = build_chm()
        np.savez_compressed(CHM_NPZ, **c)
        print(f"[gw-chm] wrote {CHM_NPZ}")

    lab = np.load(BW / "labels.npz")
    bases, is_val, in_margin = lab["bases"], lab["is_val"], lab["in_margin"]
    gt = bases[~in_margin]
    h_gt = height_at(c, gt[:, :2])
    h_gt = h_gt[np.isfinite(h_gt)]

    print(f"\n[gw-chm] ===== canopy height ABOVE LABELLED BASES ({len(h_gt)}) =====")
    qs = [1, 2, 5, 10, 25, 50, 75, 95]
    for q, v in zip(qs, np.percentile(h_gt, qs)):
        print(f"   p{q:<3d} {v:6.2f} m")
    cut = float(np.percentile(h_gt, 2))
    print(f"\n[gw-chm] the annotators' effective height cut is ~{cut:.1f} m")
    print(f"   (2nd percentile: 98% of labelled trees stand under canopy taller)")
    out = dict(cell=CELL, smooth_r=SMOOTH_R, n_gt=int(len(h_gt)),
               gt_height_pct={f"p{q}": float(v)
                              for q, v in zip(qs, np.percentile(h_gt, qs))},
               derived_cut_m=cut)

    ep = C.RESULTS / "geowalker_endpoints.csv"
    if ep.exists():
        d = np.loadtxt(ep, delimiter=",", skiprows=1)
        h_det = height_at(c, d[:, :2])
        short = np.isfinite(h_det) & (h_det < cut)
        print(f"\n[gw-chm] ===== current detections vs that cut =====")
        print(f"   {len(d)} endpoints | {int(short.sum())} "
              f"({short.mean()*100:.1f}%) stand under canopy SHORTER than "
              f"{cut:.1f} m")
        print(f"   those could not have been labelled trees whatever else is "
              f"true of them")
        out["endpoints"] = dict(n=int(len(d)), n_below_cut=int(short.sum()),
                                frac_below_cut=float(short.mean()))
        np.savetxt(C.RESULTS / "geowalker_endpoint_heights.csv",
                   np.column_stack([d[:, :4], h_det]), delimiter=",",
                   header="x,y,z,score,canopy_height", comments="")
    else:
        print(f"\n[gw-chm] {ep} not found — run gw_infer to compare detections")

    (C.RESULTS / "geowalker_height.json").write_text(json.dumps(out, indent=2))
    print(f"[gw-chm] -> results/geowalker_height.json")


if __name__ == "__main__":
    main()
