#!/usr/bin/env python3
"""StemNet v2 prep, part 6: the training cache — the tiles training and val
read, rewritten so a crop reads only its own points.

Why: the grid / feature / label tiles keep their points in scan order, so a
crop has to load whole 25 m tiles (~45 MB each) and sort them. Random crops
over ~500 training tiles missed the in-memory cache 4 times in 5: run 1 spent
2.6 s of every 3.4 s step loading tiles.

What: every tile touched by a train or val crop (the train and val cells,
grown by 5 m), its points sorted into 1 m cells (column by column, and within
a column by row), one plain .npy per array so it can be read in place:
  xyz     float32 (N, 3)
  feat    float16 (N, 12)   inf saturated at the float16 maximum (65,504)
  gate    int8 (N,)
  offset  float16 (N, 2)
  cells   int64 (626,)      cells[c] = first row of 1 m cell c = col * 25 + row
An 8 m crop is then 9 contiguous slices per tile. Same values as the source
tiles, reordered; nothing is recomputed.

Skips itself when frame.json's cache stamp matches the current labels and
features. Outputs data/stemnet_v2/cache0.02/tile_<i>_<j>/*.npy.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, TILE  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
GRID_DIR = OUT / f"grid{GRID:g}"
FEAT_DIR = OUT / f"feat{GRID:g}"
LAB_DIR = OUT / f"labels{GRID:g}"
CACHE_DIR = OUT / f"cache{GRID:g}"
FRAME_JSON = OUT / "frame.json"
GROW_M = 5
F16_MAX = 65504.0


def log(msg):
    print(f"[v2-cache] {msg}", flush=True)


def ncell():
    return int(round(TILE))


def stamp_of(frame):
    return json.dumps(dict(labels=frame.get("labels"), features=frame.get("features"), tile=TILE), sort_keys=True)


def needed_tiles(split):
    """Tiles holding any cell within GROW_M of a train (0) or val (1) cell."""
    from scipy import ndimage
    m = ndimage.binary_dilation((split == 0) | (split == 1), np.ones((2 * GROW_M + 1,) * 2, bool))
    jj, ii = np.nonzero(m)
    return set(zip((ii // TILE).astype(int).tolist(), (jj // TILE).astype(int).tolist()))


def sort_tile(xyz, i, j):
    """-> order, cells: rows sorted by 1 m cell (col-major), start row per cell."""
    n = ncell()
    cx = np.clip(np.floor(xyz[:, 0] - i * TILE).astype(np.int64), 0, n - 1)
    cy = np.clip(np.floor(xyz[:, 1] - j * TILE).astype(np.int64), 0, n - 1)
    key = cx * n + cy
    order = np.argsort(key, kind="stable")
    cells = np.searchsorted(key[order], np.arange(n * n + 1))
    return order, cells


def main():
    frame = json.loads(FRAME_JSON.read_text())
    if "labels" not in frame:
        raise SystemExit("[v2-cache] no labels — run v2_labels first")
    stamp = stamp_of(frame)
    if frame.get("cache", {}).get("stamp") == stamp and CACHE_DIR.exists():
        log(f"up to date ({frame['cache']['tiles']} tiles) — nothing to do")
        return
    split = np.load(OUT / "regions.npz")["split"]
    have = {tuple(int(v) for v in q.stem.split("_")[1:]) for q in GRID_DIR.glob("tile_*.npz")}
    tiles = sorted(needed_tiles(split) & have)
    log(f"{len(tiles)} of {len(have)} tiles are touched by train / val crops")
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    tmp = CACHE_DIR.with_name(CACHE_DIR.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    t0, n_pts, n_bytes, n_inf = time.time(), 0, 0, 0
    for k, (i, j) in enumerate(tiles):
        name = f"tile_{i}_{j}"
        xyz = np.load(GRID_DIR / f"{name}.npz")["xyz"].astype(np.float32)
        feat = np.load(FEAT_DIR / f"{name}.npz")["feat"]
        lb = np.load(LAB_DIR / f"{name}.npz")
        if not (len(feat) == len(lb["gate"]) == len(xyz)):
            raise SystemExit(f"[v2-cache] {name}: grid {len(xyz)} / feat {len(feat)} / labels "
                             f"{len(lb['gate'])} rows — rerun v2_features / v2_labels")
        bad = ~np.isfinite(feat)
        n_inf += int(bad.sum())
        feat = np.nan_to_num(feat, nan=0.0, posinf=F16_MAX, neginf=-F16_MAX).astype(np.float16)
        order, cells = sort_tile(xyz, i, j)
        d = tmp / name
        d.mkdir()
        arrays = dict(xyz=xyz[order], feat=feat[order], gate=lb["gate"].astype(np.int8)[order],
                      offset=lb["offset"].astype(np.float16)[order], cells=cells)
        for a, v in arrays.items():
            np.save(d / f"{a}.npy", np.ascontiguousarray(v))
            n_bytes += v.nbytes
        n_pts += len(xyz)
        if (k + 1) % 25 == 0 or k + 1 == len(tiles):
            el = time.time() - t0
            log(f"tile {k + 1}/{len(tiles)} | {n_pts:,} pts, {n_bytes / 2**30:.1f} GB | "
                f"{el / 60:.1f} min, eta {el / (k + 1) * (len(tiles) - k - 1) / 60:.1f} min")
    tmp.rename(CACHE_DIR)
    frame = json.loads(FRAME_JSON.read_text())
    frame["cache"] = dict(dir=CACHE_DIR.name, stamp=stamp, tiles=len(tiles), points=n_pts,
                          gb=round(n_bytes / 2**30, 2), feature_values_saturated=n_inf)
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log(f"-> {CACHE_DIR} | {len(tiles)} tiles, {n_pts:,} points, {n_bytes / 2**30:.1f} GB | "
        f"{n_inf:,} non-finite feature values saturated")


if __name__ == "__main__":
    main()
