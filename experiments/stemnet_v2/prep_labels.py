#!/usr/bin/env python3
"""StemNet v2 prep, part 5: per-point training labels.

There are no per-point annotations, only tree bases (prep_bases.py). So the
model is not taught tree / not-tree; it is taught "am I close to a tree base,
and if so, where is it":

  gate = 1   horizontal distance to the nearest base <= that base's radius +
             V2_NEAR_M (25 cm), and the point is between V2_BELOW (10 cm) below
             and V2_HIGH (2.5 m) above the ground at that base (the 11th-lowest
             grid point within 0.5 m of it). The trunk and the ground right at
             its foot; not the crown overhanging it.
  gate = 0   everything else.
  offset     base xy - point xy, for gate = 1 points (2-D; z of a base is not
             needed by anything downstream)
  base       index of that base, for gate = 1 points (-1 otherwise)

Bases are the tiered bases, with forks merged: survey rows sharing a
fork_group become one base at their mean position, radius = the largest
member's. Tier C bases have no measured radius and use V2_DEFAULT_R (5 cm,
the surveyed median).

Nothing else is detected or masked: no trunk finding, no ignore zones. A
tree nobody surveyed is labelled background like everything else.

Outputs (data/stemnet_v2/):
  labels0.02/tile_<i>_<j>.npz   gate int8 (0/1), offset float16 (N, 2), base int32,
                                row-aligned with grid0.02/tile_<i>_<j>.npz
  label_bases.csv               the merged bases the labels point at
  results/stemnet_v2_labels.txt
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS, TILE  # noqa: E402
import trunk_circles as TC  # noqa: E402  (R_MAX only)

GRID = float(os.environ.get("V2_GRID", "0.02"))
NEAR_M = float(os.environ.get("V2_NEAR_M", "0.25"))
HIGH = float(os.environ.get("V2_HIGH", "2.5"))
BELOW = float(os.environ.get("V2_BELOW", "0.10"))
DEFAULT_R = float(os.environ.get("V2_DEFAULT_R", "0.05"))
GROUND_R = 0.5
HALO = TC.R_MAX + NEAR_M + 0.1        # tile halo: a base's zone and ground can reach this far
GRID_DIR = OUT / f"grid{GRID:g}"
LAB_DIR = OUT / f"labels{GRID:g}"
FRAME_JSON = OUT / "frame.json"


def log(msg):
    print(f"[v2-labels] {msg}", flush=True)


def tile_ij(p: Path):
    a, b = p.stem.split("_")[1:]
    return int(a), int(b)


def load_bases():
    """Tiered bases, forks merged: one base per fork_group."""
    rows = list(csv.DictReader(open(OUT / "bases.csv")))
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["fork_group"], []).append(r)
    B = []
    for g, mem in groups.items():
        x = np.mean([float(r["x"]) for r in mem]); y = np.mean([float(r["y"]) for r in mem])
        rad = [float(r["radius"]) for r in mem if r.get("radius")]
        tiers = sorted(r.get("base_tier", "C") for r in mem)
        B.append(dict(x=x, y=y, radius=max(rad) if rad else DEFAULT_R, tier=tiers[0],
                      members=";".join(r["id"] for r in mem), measured=bool(rad)))
    return B


def ground_at(points, kd2, xy):
    """11th-lowest z within GROUND_R of each xy (nan where too few points)."""
    out = np.full(len(xy), np.nan)
    for k, idx in enumerate(kd2.query_ball_point(xy, GROUND_R)):
        if len(idx) >= 11:
            out[k] = np.partition(points[idx, 2], 10)[10]
    return out


def label_points(own, allp, bases_xy, bases_r):
    """gate / offset / base for the tile's own points; allp (own + halo) is
    only used to measure the ground at each base and trunk."""
    from scipy.spatial import cKDTree
    n = len(own)
    gate = np.zeros(n, np.int8); off = np.zeros((n, 2), np.float32); base = np.full(n, -1, np.int32)
    if not n:
        return gate, off, base
    kd2 = cKDTree(allp[:, :2])
    reach = TC.R_MAX + NEAR_M

    def zone(cxy, crad):
        """(point index, centre index) for points inside each centre's zone."""
        if not len(cxy):
            return np.zeros(0, int), np.zeros(0, int)
        lo, hi = own[:, :2].min(0) - reach, own[:, :2].max(0) + reach
        near = np.flatnonzero(np.all((cxy >= lo) & (cxy <= hi), axis=1))
        if not len(near):
            return np.zeros(0, int), np.zeros(0, int)
        gz = ground_at(allp, kd2, cxy[near])
        kdo = cKDTree(own[:, :2])
        pi, ci = [], []
        for k, c in enumerate(near):
            if not np.isfinite(gz[k]):
                continue
            idx = np.array(kdo.query_ball_point(cxy[c], crad[c] + NEAR_M), int)
            if not len(idx):
                continue
            h = own[idx, 2] - gz[k]
            idx = idx[(h >= -BELOW) & (h <= HIGH)]
            pi.append(idx); ci.append(np.full(len(idx), c))
        if not pi:
            return np.zeros(0, int), np.zeros(0, int)
        return np.concatenate(pi), np.concatenate(ci)

    pi, ci = zone(bases_xy, bases_r)
    if len(pi):
        # a point in two bases' zones belongs to the nearer one
        d = np.hypot(*(own[pi, :2] - bases_xy[ci]).T)
        o = np.lexsort((d, pi))
        first = np.r_[True, pi[o][1:] != pi[o][:-1]]
        pi, ci = pi[o][first], ci[o][first]
        gate[pi] = 1
        off[pi] = bases_xy[ci] - own[pi, :2]
        base[pi] = ci
    return gate, off, base


def main():
    from scipy.spatial import cKDTree
    frame = json.loads(FRAME_JSON.read_text())
    for need in ("grids", "bases", "splits"):
        if need not in frame:
            raise SystemExit(f"[v2-labels] frame.json has no '{need}' — run v2_grid, v2_bases, v2_splits first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(os.environ.get("V2_SEED", "0")))
    B = load_bases()
    bxy = np.array([[b["x"], b["y"]] for b in B]); brad = np.array([b["radius"] for b in B])
    log(f"{len(B):,} bases after merging forks ({sum(b['measured'] for b in B):,} with a measured radius)")
    tiles = sorted(GRID_DIR.glob("tile_*.npz"), key=tile_ij)

    mem: dict = {}

    def cache(ij):
        if ij not in mem:
            q = GRID_DIR / f"tile_{ij[0]}_{ij[1]}.npz"
            mem[ij] = np.load(q)["xyz"] if q.exists() else None
            if len(mem) > 64:
                mem.pop(next(iter(mem)))
        return mem[ij]

    def with_halo(ij, halo):
        i, j = ij
        own = cache(ij)
        parts = [own]
        x0, x1, y0, y1 = i * TILE - halo, (i + 1) * TILE + halo, j * TILE - halo, (j + 1) * TILE + halo
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

    # ---- labels, tile by tile ----------------------------------------
    if LAB_DIR.exists():
        shutil.rmtree(LAB_DIR)
    LAB_DIR.mkdir(parents=True)
    split = np.load(OUT / "regions.npz")["split"]
    names = ["train", "val", "test", "buffer"]
    tally = {s: np.zeros(2, np.int64) for s in names}             # gate 0, 1
    seen_base = np.zeros(len(B), bool)
    t0 = time.time()
    for k, q in enumerate(tiles):
        own, allp = with_halo(tile_ij(q), HALO)
        g, o, b = label_points(own, allp, bxy, brad)
        np.savez(LAB_DIR / q.name, gate=g, offset=o.astype(np.float16), base=b)
        seen_base[b[b >= 0]] = True
        cy = np.clip(np.floor(own[:, 1]).astype(int), 0, split.shape[0] - 1)
        cx = np.clip(np.floor(own[:, 0]).astype(int), 0, split.shape[1] - 1)
        sp = split[cy, cx]
        for si, s in enumerate(names):
            m = sp == si
            if m.any():
                tally[s] += np.bincount(g[m], minlength=2)
        if (k + 1) % 25 == 0 or k + 1 == len(tiles):
            el = time.time() - t0
            log(f"labels: tile {k + 1}/{len(tiles)} | {el / 60:.1f} min, eta {el / (k + 1) * (len(tiles) - k - 1) / 60:.1f} min")

    with open(OUT / "label_bases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base", "x", "y", "radius", "tier", "radius_measured", "has_points", "members"])
        for i, bb in enumerate(B):
            w.writerow([i, f"{bb['x']:.4f}", f"{bb['y']:.4f}", f"{bb['radius']:.4f}", bb["tier"],
                        int(bb["measured"]), int(seen_base[i]), bb["members"]])

    L = ["=" * 72, "StemNet v2 — per-point labels: close to a tree base?", "=" * 72,
         f"bases {len(B):,} (forks merged) | close = within radius + {NEAR_M:g} m horizontally, "
         f"{-BELOW:g} .. {HIGH:g} m above the ground at the base",
         f"bases with no labelled point (no scan points there): {int((~seen_base).sum()):,}",
         "", f"{'split':8s}{'points':>14s}{'gate=1':>12s}{'%':>7s}"]
    for s in names:
        ng, po = tally[s]
        tot = ng + po
        if tot:
            L.append(f"{s:8s}{tot:14,d}{po:12,d}{po / tot * 100:6.2f}%")
    txt = "\n".join(L)
    (RESULTS / "stemnet_v2_labels.txt").write_text(txt + "\n")
    print(txt, flush=True)
    frame["labels"] = dict(dir=LAB_DIR.name, bases="label_bases.csv", near_m=NEAR_M, high_m=HIGH,
                           below_m=BELOW, default_r=DEFAULT_R,
                           tally={s: dict(zip(("neg", "pos"), map(int, tally[s]))) for s in names})
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log("-> data/stemnet_v2/labels0.02/, label_bases.csv | results/stemnet_v2_labels.txt")


if __name__ == "__main__":
    main()
