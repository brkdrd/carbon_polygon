#!/usr/bin/env python3
"""StemNet v2 tree / non-tree labels from a labelled, ground-removed,
height-above-ground copy of the scan: a scene where every answer is known,
and the same answer whether the scene is one tile or four."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "stemnet_v2" / "prep_semantic.py"
W = 20.0
ORIGIN = np.array([700000.0, 4700000.0, 20.0])
EMPTY = (12.5, 19.5, 0.5, 8.0)          # x0, x1, y0, y1: the file has no point here at all


def ground(x, y):
    return 0.05 * x + 0.3 * np.sin(y / 3.0)            # varies inside a metre: one offset per column is not enough


def _scene(rng):
    g = np.arange(0.02, W, 0.04)
    gx, gy = np.meshgrid(g, g)
    G = np.column_stack([gx.ravel(), gy.ravel(), ground(gx.ravel(), gy.ravel())])
    parts, kind = [G], [np.zeros(len(G), int)]                       # 0 ground, 1 trunk, 2 bench
    for cx, cy, r in [(4.0, 4.0, 0.12), (10.0, 10.0, 0.15), (6.0, 15.0, 0.1), (15.5, 14.0, 0.2)]:
        a = np.linspace(0, 2 * np.pi, max(int(2 * np.pi * r / 0.02), 8), endpoint=False)
        A, Z = np.meshgrid(a, np.arange(0.0, 6.0, 0.02))
        x, y = cx + r * np.cos(A.ravel()), cy + r * np.sin(A.ravel())
        parts.append(np.column_stack([x, y, ground(x, y) + Z.ravel()])); kind.append(np.ones(len(x), int))
    bx, by = np.meshgrid(np.arange(7.0, 8.5, 0.02), np.arange(5.0, 5.6, 0.02))   # a bench seat, 0.5 m up
    x, y = bx.ravel(), by.ravel()
    parts.append(np.column_stack([x, y, ground(x, y) + 0.5])); kind.append(np.full(len(x), 2))
    P = np.concatenate(parts) + rng.normal(0, 0.002, (sum(map(len, parts)), 3))
    return P.astype(np.float32), np.concatenate(kind)


def _write(tmp: Path, P, kind, tile, rng):
    import laspy
    d = tmp / "data" / "stemnet_v2"
    (d / "grid0.02").mkdir(parents=True)
    ti = np.floor(P[:, 0] / tile).astype(int); tj = np.floor(P[:, 1] / tile).astype(int)
    for i, j in sorted(set(zip(ti, tj))):
        m = (ti == i) & (tj == j)
        np.savez(d / "grid0.02" / f"tile_{i}_{j}.npz", xyz=P[m], rgb=np.zeros((m.sum(), 3), np.uint8),
                 intensity=np.zeros(m.sum(), np.uint8))
    split = np.zeros((int(W), int(W)), np.uint8); split[:, 15:] = 1
    np.savez(d / "regions.npz", split=split, occupancy=np.ones_like(split, np.int32))
    (d / "frame.json").write_text(json.dumps(dict(grids={"0.02": {"stamp": {}}}, origin_m=ORIGIN.tolist(), tile_m=tile)))
    # the labelled file: 3 cm thinning, ground mostly removed, Z = height above ground, 1 cm quantised
    vox = np.floor(P / 0.03).astype(np.int64)
    _, first = np.unique(vox, axis=0, return_index=True)
    keep = np.zeros(len(P), bool); keep[first] = True
    keep &= (kind != 0) | (rng.random(len(P)) < 0.3)
    keep &= ~((P[:, 0] >= EMPTY[0]) & (P[:, 0] < EMPTY[1]) & (P[:, 1] >= EMPTY[2]) & (P[:, 1] < EMPTY[3]))
    Q = P[keep].astype(np.float64)
    hdr = laspy.LasHeader(point_format=7, version="1.4")
    hdr.add_extra_dims([laspy.ExtraBytesParams("pred_tree", np.uint8), laspy.ExtraBytesParams("prob_tree", np.float32)])
    hdr.scales = [0.01, 0.01, 0.01]; hdr.offsets = [0, 0, 0]
    las = laspy.LasData(hdr)
    las.x = Q[:, 0] + ORIGIN[0]; las.y = Q[:, 1] + ORIGIN[1]
    las.z = Q[:, 2] - ground(Q[:, 0], Q[:, 1])                      # height above ground, not elevation
    las.pred_tree = (kind[keep] == 1).astype(np.uint8)
    las.prob_tree = (kind[keep] == 1).astype(np.float32)
    (tmp / "data" / "raw").mkdir(parents=True)
    las.write(str(tmp / "data" / "raw" / "campus_pred_B24.laz"))
    return dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"), V2_TILE=str(tile))


def _run(tile):
    rng = np.random.default_rng(0)
    P, kind = _scene(rng)
    with tempfile.TemporaryDirectory() as t:
        env = _write(Path(t), P, kind, tile, rng)
        r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        d = Path(t) / "data" / "stemnet_v2"
        xyz, sem = [], []
        for q in sorted((d / "grid0.02").glob("tile_*.npz")):
            xyz.append(np.load(q)["xyz"]); sem.append(np.load(d / "sem0.02" / q.name)["sem"])
        fr = json.loads((d / "frame.json").read_text())
        assert "semantic" in fr and (Path(t) / "res" / "50_stemnet_v2_semantic.png").exists()
        r2 = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
        assert "up to date" in r2.stdout, r2.stdout[-1500:]
        xyz, sem = np.concatenate(xyz), np.concatenate(sem)
        o = np.lexsort(xyz.T)
        # kind per grid point, recovered by matching back to the scene
        oP = np.lexsort(P.T)
        assert np.array_equal(xyz[o], P[oP])
        return sem[o], kind[oP], P[oP]


def test_labels_carry_over():
    sem, kind, P = _run(10.0)
    hag = P[:, 2] - ground(P[:, 0], P[:, 1])
    inside = (P[:, 0] >= EMPTY[0] + 2.5) & (P[:, 0] < EMPTY[1] - 2.5) & (P[:, 1] >= EMPTY[2] + 2.5) & (P[:, 1] < EMPTY[3] - 2.5)
    trunk = (kind == 1) & (hag >= 0.3)
    assert (sem[trunk] == 1).mean() > 0.99, np.unique(sem[trunk], return_counts=True)
    assert (sem[kind == 2] == 0).mean() > 0.99, np.unique(sem[kind == 2], return_counts=True)
    g = (kind == 0) & ~((P[:, 0] >= EMPTY[0] - 0.5) & (P[:, 0] < EMPTY[1] + 0.5) & (P[:, 1] >= EMPTY[2] - 0.5) & (P[:, 1] < EMPTY[3] + 0.5))
    assert np.isin(sem[g], (0, 2)).mean() > 0.99, np.unique(sem[g], return_counts=True)
    assert (sem[inside] == -1).all() and inside.sum() > 100, np.unique(sem[inside], return_counts=True)


def test_same_labels_one_tile_or_four():
    a, _, _ = _run(10.0)
    b, _, _ = _run(20.0)
    assert (a == b).mean() > 0.9999, (a != b).sum()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 semantic test(s) {'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
