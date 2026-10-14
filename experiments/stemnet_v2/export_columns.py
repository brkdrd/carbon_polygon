#!/usr/bin/env python3
"""StemNet v2: cut the scan out around a sample of survey points, for looking.

Why: whether a survey point touches its trunk is a question about the survey
and the scan, and it has to be answered without any trunk detector in between.
This writes the raw 0.02 m grid points around a sample of survey rows into one
file small enough to move and inspect.

Sample (V2_EXPORT_SEED fixed, so the same trees every time):
  - V2_EXPORT_GIRTH (40) girth-surveyed trees, spread evenly over girth
  - V2_EXPORT_RANDOM (40) random survey rows from the whole map
Per tree: every grid point within V2_EXPORT_R (3 m) of the survey point (wide
enough to include the ~1.2 m shift and the trunks around it), lowest
V2_EXPORT_H (5 m) above the column's own lowest points. Capped at
V2_EXPORT_CAP points per tree by random thinning, recorded if it happens.

Output: data/stemnet_v2/columns_sample.npz
  xyz      float32 (N, 3), shared local frame
  tree     int32 (N,)     which sample row each point belongs to
  ids      survey ids ("block-n"), survey (S, 3) local xyz, girth_cm, block
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ATTRS, OUT, STEM_MAP, TILE, parse_stem_map, survey_to_local  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
R = float(os.environ.get("V2_EXPORT_R", "3.0"))
H = float(os.environ.get("V2_EXPORT_H", "5.0"))
N_GIRTH = int(os.environ.get("V2_EXPORT_GIRTH", "40"))
N_RANDOM = int(os.environ.get("V2_EXPORT_RANDOM", "40"))
CAP = int(float(os.environ.get("V2_EXPORT_CAP", "400000")))
SEED = int(os.environ.get("V2_EXPORT_SEED", "0"))
GRID_DIR = OUT / f"grid{GRID:g}"


def log(msg):
    print(f"[v2-export] {msg}", flush=True)


def main():
    from scipy.spatial import cKDTree
    frame = json.loads((OUT / "frame.json").read_text())
    sm = parse_stem_map(STEM_MAP)
    gt = survey_to_local(sm, frame["epsg"], frame["origin_m"])
    girth = {}
    if ATTRS.exists():
        with open(ATTRS) as f:
            for r in csv.DictReader(f):
                g = (r.get("girth_cm") or "").strip()
                if g:
                    girth[(round(float(r["lat"]), 8), round(float(r["lon"]), 8))] = g
    g_of = np.array([girth.get((round(float(a), 8), round(float(b), 8)), "")
                     for a, b in zip(sm.lat, sm.lon)])
    rng = np.random.default_rng(SEED)

    def gval(s):
        try:
            return max(float(v) for v in s.split(";") if v.strip())
        except ValueError:
            return np.nan
    has = np.flatnonzero(np.array([np.isfinite(gval(s)) if s else False for s in g_of]))
    order = has[np.argsort([gval(g_of[i]) for i in has])]
    pick_g = order[np.linspace(0, len(order) - 1, min(N_GIRTH, len(order))).astype(int)] if len(order) else []
    rest = np.setdiff1d(np.arange(len(gt)), pick_g)
    pick_r = rng.choice(rest, size=min(N_RANDOM, len(rest)), replace=False)
    pick = np.r_[pick_g, pick_r].astype(int)
    log(f"sample: {len(pick_g)} girth-surveyed + {len(pick_r)} random survey rows")

    have = {tuple(int(v) for v in q.stem.split("_")[1:]) for q in GRID_DIR.glob("tile_*.npz")}
    need = {}
    for s, i in enumerate(pick):
        x, y = gt[i, :2]
        for a in range(int(np.floor((x - R) / TILE)), int(np.floor((x + R) / TILE)) + 1):
            for b in range(int(np.floor((y - R) / TILE)), int(np.floor((y + R) / TILE)) + 1):
                if (a, b) in have:
                    need.setdefault((a, b), []).append(s)
    parts = [[] for _ in pick]
    for k, q in enumerate(sorted(need)):
        P = np.load(GRID_DIR / f"tile_{q[0]}_{q[1]}.npz")["xyz"]
        if not len(P):
            continue
        ss = need[q]
        hits = cKDTree(P[:, :2]).query_ball_point(gt[pick[ss], :2], r=R)
        for s, h in zip(ss, hits):
            if h:
                parts[s].append(P[h])
        log(f"tile {k + 1}/{len(need)}")
    xyz, tree, capped = [], [], 0
    for s, p in enumerate(parts):
        P = np.concatenate(p) if p else np.zeros((0, 3), np.float32)
        if len(P):
            low = np.sort(P[:, 2])[min(10, len(P) - 1)]
            P = P[P[:, 2] <= low + H]
        if len(P) > CAP:
            P = P[rng.choice(len(P), CAP, replace=False)]
            capped += 1
        xyz.append(P.astype(np.float32))
        tree.append(np.full(len(P), s, np.int32))
    xyz = np.concatenate(xyz); tree = np.concatenate(tree)
    out = OUT / "columns_sample.npz"
    np.savez_compressed(out, xyz=xyz, tree=tree,
                        ids=np.array([f"{int(sm.block[i])}-{sm.n[i]}" for i in pick]),
                        survey=gt[pick].astype(np.float64), girth_cm=g_of[pick],
                        block=sm.block[pick].astype(np.int32), radius_m=R, height_m=H,
                        capped_trees=capped, origin_m=np.array(frame["origin_m"]))
    log(f"{len(pick)} trees, {len(xyz):,} points ({capped} capped) -> {out} "
        f"({out.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
