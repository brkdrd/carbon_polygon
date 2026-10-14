#!/usr/bin/env python3
"""StemNet v2 prep part 1 (grid): checked against a brute force on a synthetic
scan the scanner "walked past twice", so a voxel's points arrive in different
chunks — exactly the case a per-chunk reduction can get wrong."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "stemnet_v2" / "prep_grid.py"
TILE = 5.0
X0, Y0 = 362_401.0, 4_763_602.0


def _scene(tmp: Path):
    import laspy
    rng = np.random.default_rng(3)
    n = 60_000
    A = np.column_stack([X0 + rng.uniform(0, 12, n), Y0 + rng.uniform(0, 12, n),
                         50 + rng.uniform(0, 3, n)])
    B = A + rng.normal(0, 0.004, A.shape)            # second pass, a few mm off
    C = A[rng.choice(n, 5000, replace=False)]        # exact duplicates
    P = np.round(np.concatenate([A, B, C]) * 1000) / 1000
    rgb = rng.integers(0, 256, (len(P), 3)).astype(np.uint16)
    inten = rng.integers(0, 256, len(P)).astype(np.uint16)
    raw = tmp / "data" / "raw"; raw.mkdir(parents=True)
    hdr = laspy.LasHeader(point_format=7, version="1.4")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([X0, Y0, 0.0])
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = P[:, 0], P[:, 1], P[:, 2]
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.intensity = inten
    las.write(str(raw / "scan.laz"))
    return P, rgb, inten


def _run(tmp: Path, chunk, extra=None):
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"),
               V2_TILE=str(TILE), V2_GRID="0.02", V2_CHUNK=str(chunk))
    env.update(extra or {})
    r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def _load(tmp: Path):
    d = tmp / "data" / "stemnet_v2" / "grid0.02"
    parts = [np.load(q) for q in sorted(d.glob("tile_*.npz"))]
    return (np.concatenate([p["xyz"] for p in parts]),
            np.concatenate([p["rgb"] for p in parts]),
            np.concatenate([p["intensity"] for p in parts]))


def test_grid_keeps_the_real_point_nearest_each_voxel_centre():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        P, rgb, inten = _scene(tmp)
        out = _run(tmp, chunk=7_000)                 # many chunks, voxels split across them
        fj = json.loads((tmp / "data" / "stemnet_v2" / "frame.json").read_text())

        # shared origin: scene minimum floored to the tile (x, y) and metre (z)
        mm = np.round(P * 1000).astype(np.int64)
        org_mm = np.array([np.floor(P[:, 0].min() / TILE) * TILE * 1000,
                           np.floor(P[:, 1].min() / TILE) * TILE * 1000,
                           np.floor(P[:, 2].min()) * 1000]).round().astype(np.int64)
        assert np.allclose(np.array(fj["origin_m"]) * 1000, org_mm, atol=1e-3), fj["origin_m"]

        # brute force: per voxel, the source point nearest the centre (ties -> first)
        D = mm - org_mm
        V = D // 20
        E = 2 * (D - V * 20) - 20
        d2 = (E * E).sum(1)
        key = V[:, 0] + 10_000 * (V[:, 1] + 10_000 * V[:, 2])
        order = np.lexsort((np.arange(len(key)), d2, key))
        win = order[np.r_[True, key[order][1:] != key[order][:-1]]]

        xyz, c, i = _load(tmp)
        assert len(xyz) == len(win), (len(xyz), len(win))
        assert fj["grids"]["0.02"]["points_kept"] == len(win)
        # same points, same colour, same intensity — matched by position
        got = np.round(xyz.astype(np.float64) * 1000).astype(np.int64)
        want = D[win]
        g_o, w_o = np.lexsort(got.T[::-1]), np.lexsort(want.T[::-1])
        assert np.abs(got[g_o] - want[w_o]).max() <= 1, "kept points are not the voxel winners"
        assert (c[g_o] == rgb[win][w_o].astype(np.uint8)).all(), "colour did not travel with its point"
        assert (i[g_o] == inten[win][w_o].astype(np.uint8)).all()

        # chunking must not change the answer
        _run(tmp, chunk=10_000_000, extra=dict(V2_FORCE="1"))
        xyz2, c2, _ = _load(tmp)
        o1, o2 = np.lexsort(xyz.T[::-1]), np.lexsort(xyz2.T[::-1])
        assert np.array_equal(xyz[o1], xyz2[o2]) and np.array_equal(c[o1], c2[o2])

        # rebuilt once; a third run skips
        assert "already built" in _run(tmp, chunk=7_000)
        assert not (tmp / "data" / "stemnet_v2" / "scratch_grid0.02").exists()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 grid test(s) "
          f"{'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
