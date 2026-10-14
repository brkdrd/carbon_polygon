#!/usr/bin/env python3
"""StemNet v2 prep part 2 (features): the 12 covariance features must

  * equal one KD-tree over the whole cloud, even for points on a tile edge
    (their neighbours sit in the next tile), and
  * mean what their names say: a trunk is vertical, the ground is flat and
    faces up, a wire is linear.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "stemnet_v2" / "prep_features.py"
TILE = 5.0
sys.path.insert(0, str(HERE / "stemnet_v2"))


def _cloud(rng):
    n = 40_000
    ground = np.column_stack([rng.uniform(0, 12, n), rng.uniform(0, 12, n),
                              rng.normal(0, 0.002, n)])
    a = rng.uniform(0, 2 * np.pi, 20_000); h = rng.uniform(0.2, 3.0, 20_000)
    trunk = np.column_stack([5.0 + 0.2 * np.cos(a), 6.0 + 0.2 * np.sin(a), h])  # ON the x=5 tile edge
    t = rng.uniform(0, 8, 4_000)
    wire = np.column_stack([2 + t, np.full_like(t, 2.0), np.full_like(t, 2.5)])
    wire += rng.normal(0, 0.002, wire.shape)
    return [ground, trunk, wire]


def _write_grid(tmp: Path, P):
    d = tmp / "data" / "stemnet_v2" / "grid0.02"
    d.mkdir(parents=True)
    ti = np.floor(P[:, 0] / TILE).astype(int); tj = np.floor(P[:, 1] / TILE).astype(int)
    order = []
    for i, j in sorted(set(zip(ti, tj))):
        m = (ti == i) & (tj == j)
        np.savez(d / f"tile_{i}_{j}.npz", xyz=P[m].astype(np.float32),
                 rgb=np.zeros((m.sum(), 3), np.uint8), intensity=np.zeros(m.sum(), np.uint8))
        order.append(np.flatnonzero(m))
    (tmp / "data" / "stemnet_v2" / "frame.json").write_text(json.dumps(
        dict(origin_m=[0, 0, 0], tile_m=TILE, grids={"0.02": dict(stamp=dict(x=1))})))
    return order


def test_features_match_one_global_kdtree_and_mean_what_they_say():
    from scipy.spatial import cKDTree
    from prep_features import covariance_features, NAMES
    rng = np.random.default_rng(5)
    parts = _cloud(rng)
    P = np.concatenate(parts).astype(np.float32).astype(np.float64)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        order = _write_grid(tmp, P)
        env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"),
                   V2_TILE=str(TILE), V2_FEAT_BATCH="7000")
        r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]

        fd = tmp / "data" / "stemnet_v2" / "feat0.02"
        got = np.full((len(P), 12), np.nan)
        for idx, q in zip(order, sorted(fd.glob("tile_*.npz"),
                                        key=lambda q: tuple(map(int, q.stem.split("_")[1:])))):
            z = np.load(q)
            assert list(z["names"]) == NAMES
            got[idx] = z["feat"].astype(np.float64)

        # one KD-tree over everything: the answer tiling must reproduce
        _, nb = cKDTree(P).query(P, k=20)
        want = covariance_features(P[nb])
        err = np.abs(got - want)
        # float16 storage; sum_eig_cm2 is the only feature not bounded by ~1
        tol = np.array([2e-3] * 12); tol[6] = 2e-3 * max(1.0, np.abs(want[:, 6]).max())
        bad = (err > tol + 1e-3 * np.abs(want)).any(1)
        assert bad.sum() == 0, f"{bad.sum()} points differ from the global KD-tree"

        g = len(parts[0]); tr = slice(g, g + len(parts[1])); w = slice(g + len(parts[1]), None)
        v = NAMES.index("verticality"); nz = NAMES.index("normal_z"); lin = NAMES.index("linearity")
        assert np.median(got[tr, v]) > 0.95, np.median(got[tr, v])          # trunk: vertical
        assert np.median(got[:g, v]) < 0.05, np.median(got[:g, v])          # ground: flat
        assert np.median(got[:g, nz]) > 0.95                                # ...facing up
        assert (got[:, nz] >= 0).all()                                      # sign convention
        assert np.median(got[w, lin]) > 0.9, np.median(got[w, lin])         # wire: linear

        fj = json.loads((tmp / "data" / "stemnet_v2" / "frame.json").read_text())
        fe = fj["features"]["0.02"]
        assert fe["k"] == 20 and fe["points"] == len(P)
        assert fe["dist_to_kth_neighbour_cm"]["p50"] > 0
        r2 = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
        assert "already built" in r2.stdout


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 feature test(s) "
          f"{'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
