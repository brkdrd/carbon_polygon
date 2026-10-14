"""Shared by every StemNet v2 stage: paths, the source file, and the integer
frame that all grids, tiles and voxels are cut from.

One frame for the whole track. Its origin is the scene's minimum, floored to
the storage tile in x and y and to the metre in z, and every stage subtracts
that same origin — from the cloud and from the survey alike, on all three
axes. Nothing else ever shifts coordinates, so there is nothing to reconcile.

Indices are computed in the LAS file's own integer units. At UTM magnitudes a
float64 coordinate already carries ~1e-11 m of error, which divided by a 2 cm
voxel misplaces points that sit on a voxel wall; in integers there is no wall
to fall off.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", "/results"))
OUT = DATA / "stemnet_v2"
LAZ_NAME = os.environ.get(
    "SOURCE_LAZ_NAME", "point_cloud_campus_2025-03-14-12-59-27_result.laz")
TILE = float(os.environ.get("V2_TILE", "25.0"))     # storage tile edge, m


def find_laz() -> Path:
    raw = DATA / "raw"
    named = raw / LAZ_NAME
    if named.exists():
        return named
    hits = sorted(raw.rglob("*.laz")) + sorted(raw.rglob("*.las"))
    if not hits:
        raise SystemExit(f"[v2] no .laz/.las under {raw} — put the campus scan there")
    return hits[0]


class Frame:
    """Integer frame of one LAS file, origin aligned to the storage tile."""

    def __init__(self, scales, offsets, mins, tile=None):
        tile = TILE if tile is None else float(tile)
        self.tile = tile
        self.s = np.asarray(scales, np.float64)
        self.o = np.asarray(offsets, np.float64)
        mins = np.asarray(mins, np.float64)
        org = np.array([np.floor(mins[0] / tile) * tile,
                        np.floor(mins[1] / tile) * tile,
                        np.floor(mins[2])])
        self.oi = np.round((org - self.o) / self.s).astype(np.int64)
        self.origin_m = self.oi * self.s + self.o

    def units(self, size, axis):
        q = size / self.s[axis]
        return int(round(q)), abs(q - round(q)) < 1e-6

    def index(self, I, axis, size):
        """Cell index of raw integers I along `axis` for cells of `size` m."""
        d = np.asarray(I, np.int64) - self.oi[axis]
        q, exact = self.units(size, axis)
        if exact:
            return d // q
        return np.floor(d * self.s[axis] / size).astype(np.int64)

    def exact(self, size):
        return all(self.units(size, a)[1] for a in range(3))

    def metres(self, P):
        return P.astype(np.float64) * self.s + self.o

    def local(self, P):
        """Raw integers -> metres in the shared frame (origin subtracted in
        integers first, so the float carries only the small local value)."""
        return (np.asarray(P, np.int64) - self.oi).astype(np.float64) * self.s
