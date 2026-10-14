#!/usr/bin/env python3
"""StemNet v2 prep, part 2: per-point geometric features, as in Liu et al.
2025 (Forests 17:36, sec. 2.2): "12 geometric features derived from the
covariance matrix of its 20 nearest neighbors".

For every point of the 0.02 m grid: the covariance of its 20 nearest
neighbours (the point itself included), eigenvalues l1 >= l2 >= l3, normalised
e_i = l_i / (l1 + l2 + l3), smallest eigenvector n. The paper names the
features but not the formulas, so these are the standard definitions it draws
on (Weinmann et al. 2015, ISPRS J. 105:286):

   0 linearity          (l1 - l2) / l1
   1 planarity          (l2 - l3) / l1
   2 sphericity          l3 / l1
   3 omnivariance       (e1 e2 e3)^(1/3)
   4 anisotropy         (l1 - l3) / l1
   5 eigenentropy       -sum e_i ln e_i
   6 sum_eig_cm2         l1 + l2 + l3, in cm^2 (so float16 keeps its precision)
   7 change_curvature    l3 / (l1 + l2 + l3)
   8 normal_x           n, sign chosen so normal_z >= 0
   9 normal_y
  10 normal_z
  11 verticality        1 - |normal_z|

The normal (8-10) is also what Sonata's normal channel receives.

Tile edges: a point's neighbours can sit in the next tile, so every tile is
processed with its neighbours' points within V2_FEAT_HALO metres added. The
test checks the result is identical to one KD-tree over the whole cloud.

Also measured and recorded, because "20 neighbours" is a count, not a size:
the distance to the 20th neighbour, i.e. how large a patch each feature
actually describes on this cloud.

Outputs (data/stemnet_v2/):
  feat0.02/tile_<i>_<j>.npz   feat float16 (N, 12), row-aligned with
                              grid0.02/tile_<i>_<j>.npz; names stored inside
  frame.json                  ["features"]: k, names, neighbour-distance stats
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
K = int(os.environ.get("V2_FEAT_K", "20"))
HALO = float(os.environ.get("V2_FEAT_HALO", "1.0"))
BATCH = int(float(os.environ.get("V2_FEAT_BATCH", "400000")))
FORCE = os.environ.get("V2_FORCE", "0") != "0"
GRID_DIR = OUT / f"grid{GRID:g}"
FEAT_DIR = OUT / f"feat{GRID:g}"
FRAME_JSON = OUT / "frame.json"
NAMES = ["linearity", "planarity", "sphericity", "omnivariance", "anisotropy",
         "eigenentropy", "sum_eig_cm2", "change_curvature",
         "normal_x", "normal_y", "normal_z", "verticality"]
EPS = 1e-12


def log(msg):
    print(f"[v2-feat] {msg}", flush=True)


def covariance_features(nbr):
    """nbr: (N, K, 3) neighbour coordinates -> (N, 12) float64."""
    c = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / nbr.shape[1]
    w, v = np.linalg.eigh(cov)                     # ascending
    l3, l2, l1 = (np.clip(w[:, i], 0, None) for i in range(3))
    s = l1 + l2 + l3
    e = np.stack([l1, l2, l3], 1) / (s[:, None] + EPS)
    n = v[:, :, 0]                                 # eigenvector of the smallest
    n = n * np.where(n[:, 2:3] < 0, -1.0, 1.0)
    d1 = l1 + EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(e * np.log(np.where(e > 0, e, 1.0))).sum(1)
    return np.column_stack([
        (l1 - l2) / d1, (l2 - l3) / d1, l3 / d1,
        np.cbrt(e[:, 0] * e[:, 1] * e[:, 2]),
        (l1 - l3) / d1, ent, s * 1e4, l3 / (s + EPS),
        n[:, 0], n[:, 1], n[:, 2], 1.0 - np.abs(n[:, 2])])


def tile_ij(p: Path):
    a, b = p.stem.split("_")[1:]
    return int(a), int(b)


def with_halo(ij, cache):
    """The tile's own points first, then neighbours' points within HALO."""
    i, j = ij
    own = cache(ij)
    parts = [own]
    x0, x1 = i * TILE - HALO, (i + 1) * TILE + HALO
    y0, y1 = j * TILE - HALO, (j + 1) * TILE + HALO
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0:
                continue
            P = cache((i + di, j + dj))
            if P is None or not len(P):
                continue
            m = (P[:, 0] >= x0) & (P[:, 0] < x1) & (P[:, 1] >= y0) & (P[:, 1] < y1)
            if m.any():
                parts.append(P[m])
    return own.astype(np.float64), np.concatenate(parts).astype(np.float64)


def main():
    from scipy.spatial import cKDTree
    if not FRAME_JSON.exists() or f"{GRID:g}" not in json.loads(FRAME_JSON.read_text()).get("grids", {}):
        raise SystemExit(f"[v2-feat] no {GRID:g} m grid — run v2_grid first")
    frame = json.loads(FRAME_JSON.read_text())
    stamp = dict(grid=frame["grids"][f"{GRID:g}"]["stamp"], k=K, halo_m=HALO, names=NAMES)
    if not FORCE and frame.get("features", {}).get(f"{GRID:g}", {}).get("stamp") == stamp:
        log("features already built for this grid — skipping (V2_FORCE=1 rebuilds)")
        return
    tiles = sorted(GRID_DIR.glob("tile_*.npz"), key=tile_ij)   # numeric: neighbours stay cached
    if FEAT_DIR.exists():
        shutil.rmtree(FEAT_DIR)
    FEAT_DIR.mkdir(parents=True)

    mem: dict = {}

    def cache(ij):
        # neighbours are needed by up to 9 tiles; keep a small window in memory
        if ij not in mem:
            q = GRID_DIR / f"tile_{ij[0]}_{ij[1]}.npz"
            mem[ij] = np.load(q)["xyz"] if q.exists() else None   # float32, ~12 B/pt
            if len(mem) > 64:
                mem.pop(next(iter(mem)))
        return mem[ij]

    rk_samples, n_done, short = [], 0, 0
    rng = np.random.default_rng(0)
    t0 = time.time()
    for t, q in enumerate(tiles):
        ij = tile_ij(q)
        own, allp = with_halo(ij, cache)
        F = np.empty((len(own), 12), np.float16)
        if len(allp) >= K:
            kd = cKDTree(allp)
            for a in range(0, len(own), BATCH):
                b = min(a + BATCH, len(own))
                d, idx = kd.query(own[a:b], k=K, workers=-1)
                F[a:b] = covariance_features(allp[idx]).astype(np.float16)
                rk = d[:, -1]
                short += int((rk > HALO).sum())
                rk_samples.append(rng.choice(rk, size=min(2000, len(rk)), replace=False))
        else:
            F[:] = np.nan
        np.savez(FEAT_DIR / q.name, feat=F, names=np.array(NAMES))
        n_done += len(own)
        if (t + 1) % 25 == 0 or t + 1 == len(tiles):
            el = time.time() - t0
            log(f"tile {t + 1}/{len(tiles)} | {n_done:,} pts | {el / 60:.1f} min, "
                f"eta {el / (t + 1) * (len(tiles) - t - 1) / 60:.1f} min")

    rk = np.concatenate(rk_samples) if rk_samples else np.zeros(1)
    stats = {f"p{p}": float(np.percentile(rk, p) * 100) for p in (10, 50, 90, 99)}
    frame.setdefault("features", {})[f"{GRID:g}"] = dict(
        stamp=stamp, dir=FEAT_DIR.name, k=K, names=NAMES, points=n_done,
        dist_to_kth_neighbour_cm=stats,
        points_whose_kth_neighbour_is_beyond_halo=short,
        formulas="Weinmann et al. 2015; see stemnet_v2/prep_features.py docstring")
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log(f"done: {n_done:,} points, {len(tiles)} tiles -> {FEAT_DIR}")
    log(f"the {K}-neighbour patch on this cloud: distance to the {K}th neighbour "
        + "  ".join(f"{k} {v:.1f} cm" for k, v in stats.items()))
    if short:
        log(f"!! {short:,} points have their {K}th neighbour beyond the {HALO} m halo "
            f"— their features near tile edges are approximate; raise V2_FEAT_HALO")


if __name__ == "__main__":
    main()
