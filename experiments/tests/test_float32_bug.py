#!/usr/bin/env python3
"""Regression test for the float32 coordinate-quantization trap.

Verifies common.geo.center_float64 against the exact scenario and numbers in
FLOAT32_COORDINATE_BUG.md. CPU-only, no GPU / laspy needed:

    python tests/test_float32_bug.py      # or: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import geo


def _canopy_patch(n=20_000, seed=0):
    """1 m canopy patch at UTM 52N magnitudes (easting ~7.36e5, northing ~4.77e6)."""
    rng = np.random.default_rng(seed)
    x = 736_000.0 + rng.random(n)
    y = 4_767_000.0 + rng.random(n)   # the killer axis: float32 ULP ~0.5 m
    z = 40.0 + rng.random(n)
    return np.column_stack([x, y, z])


def test_ulp_report_flags_northing():
    xyz = _canopy_patch()
    rep = geo.float32_ulp_report(xyz)
    # northing lands in a band where the float32 step is a full 0.5 m
    assert rep["y"]["float32_ulp_m"] >= 0.4, rep


def test_bug_collapses_geometry():
    """cast-then-center shreds the cloud (bug doc: x=17 y=3 z~19k)."""
    xyz = _canopy_patch()
    bug = xyz.astype(np.float32)
    bug = bug - bug.mean(0, keepdims=True)
    distinct = [len(np.unique(bug[:, i])) for i in range(3)]
    assert distinct[1] < 50, distinct          # northing collapses to a handful


def test_fix_preserves_geometry():
    """center_float64 (float64, cast last) preserves detail, zero geometric error."""
    xyz = _canopy_patch()
    centered, origin = geo.center_float64(xyz)
    cast = geo.to_float32_local(centered)
    distinct = [len(np.unique(cast[:, i])) for i in range(3)]
    assert min(distinct) > 15_000, distinct
    max_err = float(np.max(np.abs((centered + origin) - xyz)))
    assert max_err < 1e-3, max_err             # bug doc: 0.0000 m


def test_fixed_origin_roundtrips():
    xyz = _canopy_patch()
    origin = np.array([736_000.0, 4_767_000.0, 40.0])
    centered, used = geo.center_float64(xyz, origin=origin)
    assert np.allclose(used, origin)
    assert np.max(np.abs((centered + used) - xyz)) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} float32-trap tests passed.")
