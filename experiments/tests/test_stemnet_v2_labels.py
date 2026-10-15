#!/usr/bin/env python3
"""StemNet v2 per-point labels: every rule, on a scene where the answer to
each is known, and the same answer whether the scene is one tile or four."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "stemnet_v2" / "prep_labels.py"
W = 20.0


def ground(x):
    return 0.05 * x


def _trunk(rng, cx, cy, r, h=6.0):
    na = max(int(2 * np.pi * r / 0.02), 8)
    a = np.linspace(0, 2 * np.pi, na, endpoint=False)
    z = np.arange(0.0, h, 0.02)
    A, Z = np.meshgrid(a, z)
    P = np.column_stack([cx + r * np.cos(A.ravel()), cy + r * np.sin(A.ravel()), ground(cx) + Z.ravel()])
    return P + rng.normal(0, 0.002, P.shape)


def _scene(rng):
    g = np.arange(0.02, W, 0.04)
    gx, gy = np.meshgrid(g, g)
    G = np.column_stack([gx.ravel(), gy.ravel(), ground(gx.ravel())])
    trees = {"edge": (10.0, 6.0, 0.15), "open": (4.0, 4.0, 0.10), "unsurveyed": (15.0, 15.0, 0.12),
             "fork_a": (5.0, 15.0, 0.08), "fork_b": (5.25, 15.0, 0.08)}       # x = 10 is a tile edge
    P = np.concatenate([G] + [_trunk(rng, x, y, r) for x, y, r in trees.values()]).astype(np.float32)
    return P, trees


def _write(tmp: Path, P, trees, tile):
    d = tmp / "data" / "stemnet_v2"
    (d / "grid0.02").mkdir(parents=True)
    ti = np.floor(P[:, 0] / tile).astype(int); tj = np.floor(P[:, 1] / tile).astype(int)
    for i, j in sorted(set(zip(ti, tj))):
        m = (ti == i) & (tj == j)
        np.savez(d / "grid0.02" / f"tile_{i}_{j}.npz", xyz=P[m], rgb=np.zeros((m.sum(), 3), np.uint8),
                 intensity=np.zeros(m.sum(), np.uint8))
    rows = [("1-1", *trees["edge"], "A", 0), ("1-2", *trees["open"], "B", 1),
            ("1-3", *trees["fork_a"], "A", 2), ("1-4", *trees["fork_b"], "A", 2)]
    with open(d / "bases.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "x", "y", "radius", "base_tier", "fork_group"])
        for r in rows:
            w.writerow(r)
    split = np.zeros((int(W), int(W)), np.uint8); split[:, 15:] = 1
    np.savez(d / "regions.npz", split=split, occupancy=np.ones_like(split, np.int32))
    (d / "frame.json").write_text(json.dumps(dict(grids={"0.02": {}}, bases={}, splits={}, origin_m=[0, 0, 0])))


def _run(tile):
    rng = np.random.default_rng(0)
    P, trees = _scene(rng)
    tmp = Path(tempfile.mkdtemp())
    _write(tmp, P, trees, tile)
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"), V2_TILE=str(tile))
    r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = tmp / "data" / "stemnet_v2"
    X, G, O, B = [], [], [], []
    for q in sorted((d / "grid0.02").glob("tile_*.npz")):
        L = np.load(d / "labels0.02" / q.name)
        X.append(np.load(q)["xyz"]); G.append(L["gate"]); O.append(L["offset"]); B.append(L["base"])
    X = np.concatenate(X).astype(np.float64)
    o = np.lexsort(X.T[::-1])
    bases = list(csv.DictReader(open(d / "label_bases.csv")))
    return X[o], np.concatenate(G)[o], np.concatenate(O)[o].astype(np.float64), np.concatenate(B)[o], trees, bases


def test_labels_follow_every_rule():
    X, g, off, b, trees, bases = _run(tile=W)          # one tile
    assert len(bases) == 3, bases                       # the fork pair merged into one base

    def at(name, hmin, hmax, rmin=0.0, rmax=0.02):
        x, y, r = trees[name]
        d = np.hypot(X[:, 0] - x, X[:, 1] - y) - r
        h = X[:, 2] - ground(x)
        return (d >= rmin - 0.01) & (d <= rmax) & (h >= hmin) & (h <= hmax)

    for name in ("edge", "open"):
        low = at(name, 0.1, 2.3)
        assert low.sum() > 100 and (g[low] == 1).all(), (name, np.unique(g[low], return_counts=True))
        x, y, _ = trees[name]                           # offset points at the base
        tgt = np.array([x, y]) - X[low, :2]
        assert np.abs(off[low] - tgt).max() < 0.01, np.abs(off[low] - tgt).max()
        high = at(name, 2.8, 6.0)                       # the trunk above 2.5 m: background
        assert (g[high] == 0).all(), name
        x, y, r = trees[name]
        dg = np.hypot(X[:, 0] - x, X[:, 1] - y)
        foot = (dg > r + 0.03) & (dg < r + 0.2) & (np.abs(X[:, 2] - ground(X[:, 0])) < 0.05)
        assert (g[foot] == 1).all(), name               # the ground at its foot is "close"
        away = (dg > r + 0.35) & (dg < r + 1.0) & (np.abs(X[:, 2] - ground(X[:, 0])) < 0.05)
        assert (g[away] == 0).all(), name
    un = at("unsurveyed", 0.1, 2.3)
    assert un.sum() > 100 and (g[un] == -1).all(), np.unique(g[un], return_counts=True)
    fk = at("fork_a", 0.1, 2.3) | at("fork_b", 0.1, 2.3)
    assert (g[fk] == 1).all() and len(set(b[fk].tolist())) == 1   # both stems -> one merged base


def test_labels_do_not_depend_on_tiling():
    """The 'edge' trunk sits on x = 10: with 10 m tiles its points and its
    ground come from two tiles, and every label must match the one-tile run."""
    X1, g1, o1, _, _, _ = _run(tile=W)
    X4, g4, o4, _, _, _ = _run(tile=10.0)
    assert np.array_equal(X1, X4)
    assert (g1 == g4).mean() > 0.9999, (g1 != g4).sum()
    assert np.abs(o1 - o4).max() < 1e-3


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 label test(s) {'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
