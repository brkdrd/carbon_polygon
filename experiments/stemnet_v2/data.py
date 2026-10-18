"""StemNet v2 training data: square crops cut from the prepared tiles, and the
augmentation applied to them.

Per point a crop carries: xyz (shared local frame), normal (from the feature
tiles), the 12 covariance features, gate (0/1) and the 2-D offset to its base,
plus which tile and row it came from (so inference can average the
predictions a point receives from overlapping crops).

Augmentation follows TreeLearn (dataset.py dataAugment / point_jitter, each
applied with probability 0.6, point jitter 0.25), with two deliberate changes:
  * the random 3x3 jitter matrix acts on x/y only. Mixing z into x/y would make
    the horizontal offset to the base depend on a point's height, and the
    target would no longer be exact.
  * point jitter is scaled by the grid ratio: TreeLearn's sigma 0.10 / clip
    0.20 m is for a 0.1 m grid; on this 0.02 m grid it would erase thin stems,
    so it is sigma 0.02 / clip 0.04 m.
Everything linear (rotation, flip, scale, xy jitter) is applied to the points
AND to the offsets and normals, so point + offset is still the transformed
base; point jitter moves a point but not its base, so its offset is reduced by
the same amount.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from core import OUT, TILE

GRID = float(os.environ.get("V2_GRID", "0.02"))
GRID_DIR = OUT / f"grid{GRID:g}"
FEAT_DIR = OUT / f"feat{GRID:g}"
LAB_DIR = OUT / f"labels{GRID:g}"
SEM_DIR = OUT / f"sem{GRID:g}"
NORMAL_COLS = (8, 9, 10)                     # normal_x/y/z in the 12-feature tiles
AUG_P, JIT_P = 0.6, 0.25
PT_SIGMA, PT_CLIP = 0.02, 0.04
F16_MAX = 65504.0


class TileStore:
    """Grid, feature and label tiles, loaded on demand, recently used kept."""

    def __init__(self, keep=96, labels=True):
        self.keep, self.labels, self.mem = keep, labels, {}
        self.have = {tuple(int(v) for v in q.stem.split("_")[1:]) for q in GRID_DIR.glob("tile_*.npz")}
        if not self.have:
            raise SystemExit(f"[v2-data] no grid tiles in {GRID_DIR} — run v2_grid")
        if not FEAT_DIR.exists():
            raise SystemExit(f"[v2-data] no feature tiles in {FEAT_DIR} — run v2_features")
        if labels and not LAB_DIR.exists():
            raise SystemExit(f"[v2-data] no label tiles in {LAB_DIR} — run v2_labels")

    def tile(self, ij):
        if ij not in self.mem:
            if ij not in self.have:
                self.mem[ij] = None
            else:
                name = f"tile_{ij[0]}_{ij[1]}.npz"
                g = np.load(GRID_DIR / name)
                # sum_eig_cm2 overflows float16 (65,504 cm^2) for isolated points
                # whose 20 neighbours spread over metres: saturate, don't feed inf
                f = np.nan_to_num(np.load(FEAT_DIR / name)["feat"].astype(np.float32),
                                  nan=0.0, posinf=F16_MAX, neginf=-F16_MAX)
                t = dict(xyz=g["xyz"].astype(np.float64), feat=f,
                         normal=f[:, NORMAL_COLS].astype(np.float32))
                if self.labels:
                    lb = np.load(LAB_DIR / name)
                    t.update(gate=lb["gate"].astype(np.int8), offset=lb["offset"].astype(np.float32))
                    q = SEM_DIR / name
                    t["sem"] = (np.load(q)["sem"].astype(np.int8) if q.exists()
                                else np.full(len(t["xyz"]), -1, np.int8))
                order = np.argsort(t["xyz"][:, 0], kind="stable")
                t["order"], t["xs"] = order, t["xyz"][order, 0]
                self.mem[ij] = t
            while len(self.mem) > self.keep:
                self.mem.pop(next(iter(self.mem)))
        return self.mem[ij]

    def crop(self, cx, cy, size):
        """Every point in the square of edge `size` centred on (cx, cy)."""
        h = size / 2.0
        out = {k: [] for k in ("xyz", "normal", "feat", "gate", "offset", "sem", "tile", "row")}
        for a in range(int(np.floor((cx - h) / TILE)), int(np.floor((cx + h) / TILE)) + 1):
            for b in range(int(np.floor((cy - h) / TILE)), int(np.floor((cy + h) / TILE)) + 1):
                t = self.tile((a, b))
                if t is None:
                    continue
                lo, hi = np.searchsorted(t["xs"], [cx - h, cx + h], side="left")
                idx = t["order"][lo:hi]
                idx = idx[np.abs(t["xyz"][idx, 1] - cy) <= h]
                idx = idx[np.abs(t["xyz"][idx, 0] - cx) <= h]
                if not len(idx):
                    continue
                for k in ("xyz", "normal", "feat"):
                    out[k].append(t[k][idx])
                if self.labels:
                    out["gate"].append(t["gate"][idx]); out["offset"].append(t["offset"][idx])
                    out["sem"].append(t["sem"][idx])
                out["tile"].append(np.full(len(idx), a * 100000 + b, np.int64)); out["row"].append(idx)
        if not out["xyz"]:
            return None
        c = {k: np.concatenate(v) for k, v in out.items() if v}
        return c


class CacheStore:
    """The sorted training cache (prep_cache.py), read in place: a crop reads
    one contiguous slice per 1 m column per tile, nothing else. Same crop()
    and tile() as TileStore; rows are rows of the sorted cache."""

    def __init__(self):
        self.dir = OUT / f"cache{GRID:g}"
        self.have = {tuple(int(v) for v in q.name.split("_")[1:]) for q in self.dir.glob("tile_*")}
        self.n = int(round(TILE))
        self.mem = {}

    def tile(self, ij):
        if ij not in self.mem:
            d = self.dir / f"tile_{ij[0]}_{ij[1]}"
            if ij in self.have:
                t = {a: np.load(d / f"{a}.npy", mmap_mode="r") for a in ("xyz", "feat", "gate", "offset", "cells")}
                t["sem"] = (np.load(d / "sem.npy", mmap_mode="r") if (d / "sem.npy").exists()
                            else np.full(len(t["xyz"]), -1, np.int8))
                self.mem[ij] = t
            else:
                self.mem[ij] = None
        return self.mem[ij]

    def crop(self, cx, cy, size):
        h, n = size / 2.0, self.n
        out = {k: [] for k in ("xyz", "feat", "gate", "offset", "sem", "tile", "row")}
        for a in range(int(np.floor((cx - h) / TILE)), int(np.floor((cx + h) / TILE)) + 1):
            for b in range(int(np.floor((cy - h) / TILE)), int(np.floor((cy + h) / TILE)) + 1):
                t = self.tile((a, b))
                if t is None:
                    continue
                c0, c1 = (int(np.clip(np.floor(v - a * TILE), 0, n - 1)) for v in (cx - h, cx + h))
                r0, r1 = (int(np.clip(np.floor(v - b * TILE), 0, n - 1)) for v in (cy - h, cy + h))
                cells = t["cells"]
                for col in range(c0, c1 + 1):
                    lo, hi = int(cells[col * n + r0]), int(cells[col * n + r1 + 1])
                    if hi <= lo:
                        continue
                    xyz = np.asarray(t["xyz"][lo:hi], np.float64)   # edges compared as TileStore does
                    k = np.flatnonzero((np.abs(xyz[:, 0] - cx) <= h) & (np.abs(xyz[:, 1] - cy) <= h))
                    if not len(k):
                        continue
                    out["xyz"].append(xyz[k]); out["row"].append(lo + k)
                    out["feat"].append(np.asarray(t["feat"][lo:hi])[k])
                    out["gate"].append(np.asarray(t["gate"][lo:hi])[k])
                    out["offset"].append(np.asarray(t["offset"][lo:hi])[k])
                    out["sem"].append(np.asarray(t["sem"][lo:hi])[k])
                    out["tile"].append(np.full(len(k), a * 100000 + b, np.int64))
        if not out["xyz"]:
            return None
        c = {k: np.concatenate(v) for k, v in out.items()}
        c["feat"] = c["feat"].astype(np.float32)
        c["offset"] = c["offset"].astype(np.float32)
        c["normal"] = c["feat"][:, NORMAL_COLS[0]:NORMAL_COLS[2] + 1].copy()
        return c


def open_store(log=print):
    """The sorted cache when it matches the current labels and features,
    else the plain tiles (slow: whole tiles per crop) with a warning."""
    import json
    import prep_cache
    frame = json.loads((OUT / "frame.json").read_text())
    if frame.get("cache", {}).get("stamp") == prep_cache.stamp_of(frame) and (OUT / f"cache{GRID:g}").exists():
        log(f"reading crops from the sorted cache ({frame['cache']['tiles']} tiles)")
        return CacheStore()
    log("!! no up-to-date sorted cache — reading whole tiles per crop (slow); run v2_cache")
    return TileStore()


def eligible_centres(regions, size=8.0, occ_min=0.45, split_id=0):
    """Centres of size x size squares lying wholly in one split, with at least
    occ_min of their 1 m cells scanned (TreeLearn's occupancy gate)."""
    split, occ = regions["split"], regions["occupancy"]
    k = int(round(size))

    def box(a):
        I = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
        I[1:, 1:] = a.cumsum(0).cumsum(1)
        return I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]
    ok = (box((split == split_id).astype(np.int64)) == k * k) & (box((occ > 0).astype(np.int64)) >= occ_min * k * k)
    jj, ii = np.nonzero(ok)
    return np.column_stack([ii + size / 2.0, jj + size / 2.0])


def augment(c, rng, centre):
    """In place. Returns the 2x2 linear map applied to x/y (for the record)."""
    A = np.eye(2)
    zs = 1.0
    if rng.random() < AUG_P:                                  # scale
        A = A * rng.uniform(0.8, 1.2, 2)
        zs = rng.uniform(0.95, 1.05)
    if rng.random() < AUG_P:                                  # xy jitter (no z mixing)
        A = A + rng.normal(0, 0.1, (2, 2))
    if rng.random() < AUG_P and rng.integers(0, 2):           # flip x
        A = A @ np.diag([-1.0, 1.0])
    if rng.random() < AUG_P:                                  # rotation about z
        t = rng.uniform(0, 2 * np.pi)
        A = A @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = c["xyz"][:, :2] - centre
    c["xyz"][:, :2] = xy @ A.T + centre
    c["xyz"][:, 2] *= zs
    if "offset" in c:
        c["offset"] = (c["offset"] @ A.T).astype(np.float32)
    # normals: transform as covectors (inverse-transpose), then renormalise
    Ainv = np.linalg.inv(A).T
    n = c["normal"].astype(np.float64)
    n[:, :2] = n[:, :2] @ Ainv.T
    n[:, 2] /= zs
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    c["normal"] = n.astype(np.float32)
    c["feat"][:, NORMAL_COLS[0]:NORMAL_COLS[2] + 1] = c["normal"]
    if rng.random() < JIT_P:                                  # point jitter: the base stays put
        d = np.clip(rng.normal(0, PT_SIGMA, (len(xy), 3)), -PT_CLIP, PT_CLIP)
        c["xyz"] += d
        if "offset" in c:
            c["offset"] = (c["offset"] - d[:, :2]).astype(np.float32)
    return A
