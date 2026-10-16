#!/usr/bin/env python3
"""StemNet v2 training, on CPU with a stand-in encoder: the pieces that can be
silently wrong without a GPU ever noticing.

The scene is built through the real prep stages (features, labels) from
synthetic grid tiles, so the trainer reads exactly what it will read on the
campus."""
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
V2 = HERE / "stemnet_v2"
TILE, W = 10.0, 30.0


def _build(tmp: Path):
    rng = np.random.default_rng(1)
    pos = []
    while len(pos) < 40:
        p = rng.uniform(2, W - 2, 2)
        if all(np.hypot(*(p - q)) >= 1.2 for q in pos):
            pos.append(p)
    pos = np.array(pos); rad = rng.uniform(0.05, 0.15, len(pos))
    g = np.arange(0.02, W, 0.04)
    gx, gy = np.meshgrid(g, g)
    parts = [np.column_stack([gx.ravel(), gy.ravel(), 0.03 * gx.ravel()])]
    for (x, y), r in zip(pos, rad):
        a = np.linspace(0, 2 * np.pi, max(int(2 * np.pi * r / 0.02), 8), endpoint=False)
        z = np.arange(0, 4.0, 0.02)
        A, Z = np.meshgrid(a, z)
        parts.append(np.column_stack([x + r * np.cos(A.ravel()), y + r * np.sin(A.ravel()), 0.03 * x + Z.ravel()]))
    P = np.concatenate(parts).astype(np.float32)
    d = tmp / "data" / "stemnet_v2"
    (d / "grid0.02").mkdir(parents=True)
    ti = np.floor(P[:, 0] / TILE).astype(int); tj = np.floor(P[:, 1] / TILE).astype(int)
    for i, j in sorted(set(zip(ti, tj))):
        m = (ti == i) & (tj == j)
        np.savez(d / "grid0.02" / f"tile_{i}_{j}.npz", xyz=P[m], rgb=np.zeros((m.sum(), 3), np.uint8),
                 intensity=np.zeros(m.sum(), np.uint8))
    with open(d / "bases.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "x", "y", "radius", "base_tier", "fork_group"])
        for k, ((x, y), r) in enumerate(zip(pos, rad)):
            w.writerow([f"1-{k}", x, y, r, "A", k])
    split = np.full((int(W), int(W)), 0, np.uint8)
    split[:, 18:20] = 3; split[:, 20:] = 1                  # train | buffer | val
    np.savez(d / "regions.npz", split=split, occupancy=np.ones_like(split, np.int32))
    (d / "frame.json").write_text(json.dumps(dict(grids={"0.02": {"stamp": {}}}, bases={}, splits={},
                                                  origin_m=[0, 0, 0], tile_m=TILE)))
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"), V2_TILE=str(TILE))
    for stage in ("prep_features.py", "prep_labels.py"):
        r = subprocess.run([sys.executable, str(V2 / stage)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, (stage, r.stdout[-2000:], r.stderr[-2000:])
    return env, pos, rad


def _with_env(env):
    for k, v in env.items():
        os.environ[k] = v
    sys.path.insert(0, str(V2))
    for m in ("core", "data", "model", "vote", "train"):
        sys.modules.pop(m, None)


def test_crop_equals_brute_force_and_augmentation_keeps_bases():
    with tempfile.TemporaryDirectory() as t:
        env, pos, rad = _build(Path(t))
        _with_env(env)
        import data as D
        store = D.TileStore()
        allxyz = np.concatenate([store.tile(ij)["xyz"] for ij in sorted(store.have)])
        c = store.crop(10.0, 10.0, 8.0)                      # straddles four tiles
        want = allxyz[(np.abs(allxyz[:, 0] - 10) <= 4) & (np.abs(allxyz[:, 1] - 10) <= 4)]
        assert len(c["xyz"]) == len(want)
        assert c["gate"].sum() > 100
        base_before = c["xyz"][:, :2] + c["offset"]
        rng = np.random.default_rng(3)
        D.AUG_P, D.JIT_P = 1.0, 1.0                           # every augmentation, every time
        A = D.augment(c, rng, np.array([10.0, 10.0]))
        k = c["gate"] == 1
        want_base = (base_before[k] - [10, 10]) @ A.T + [10, 10]
        got_base = c["xyz"][k, :2] + c["offset"][k]
        assert np.abs(got_base - want_base).max() < 1e-3, np.abs(got_base - want_base).max()
        assert np.allclose(np.linalg.norm(c["normal"], axis=1), 1, atol=1e-4)


def test_inference_path_finds_every_base_from_perfect_predictions():
    """Feed the true gate and offset through predict_region -> detect -> match:
    whatever the model learns, this path must not lose bases on its own."""
    with tempfile.TemporaryDirectory() as t:
        env, pos, rad = _build(Path(t))
        _with_env(env)
        import data as D, vote as V
        store = D.TileStore()
        oracle = lambda c: (c["gate"].astype(np.float32), c["offset"].astype(np.float32))  # noqa: E731
        bbox = (20.0, 30.0, 0.0, 30.0)
        xy, p, off = V.predict_region(oracle, store, bbox, 300000, np.random.default_rng(0))
        det, score, nv = V.detect(xy, p, off)
        gt = pos[(pos[:, 0] >= 20)]
        m = V.match(det, score, gt, 0.3, bbox)
        assert m["recall"] == 1.0 and m["precision"] == 1.0, {k: v for k, v in m.items() if k != "pair"}
        assert m["rmse"] < 0.03, m["rmse"]
        rng = np.random.default_rng(5)                         # an imperfect model

        def noisy(c):
            g = c["gate"].astype(np.float32)
            g[(g == 0) & (rng.random(len(g)) < 0.05)] = 0.9
            return g, (c["offset"] + rng.normal(0, 0.05, c["offset"].shape)).astype(np.float32)
        xy, p, off = V.predict_region(noisy, store, bbox, 300000, rng)
        det, score, nv = V.detect(xy, p, off)
        m = V.match(det, score, gt, 0.3, bbox)
        assert m["recall"] == 1.0 and m["rmse"] < 0.05, {k: v for k, v in m.items() if k != "pair"}


def test_training_runs_evaluates_saves_and_resumes():
    import torch
    with tempfile.TemporaryDirectory() as t:
        env, pos, rad = _build(Path(t))
        for q in sorted((Path(t) / "data" / "stemnet_v2" / "feat0.02").glob("tile_*.npz")):
            d = dict(np.load(q))                              # run 1 died on this: an isolated
            d["feat"][::50, 6] = np.inf                       # point overflowing float16
            np.savez(q, **d)
        env.update(V2_ENCODER="fake", V2_DEVICE="cpu", V2_STEPS="12", V2_EVAL_EVERY="6", V2_ACCUM="1",
                   V2_CAP="40000", V2_PROGRESS_S="0")
        r = subprocess.run([sys.executable, str(V2 / "train.py")], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        log = (Path(t) / "res" / "stemnet_v2_train.txt").read_text()
        assert log.count("EVAL step") == 2 and "done: 12 steps" in log, log[-1500:]
        assert (Path(t) / "data" / "stemnet_v2" / "models" / "head_last.pth").exists()
        assert "non-finite" not in log and "not applied" not in log, log[-1500:]
        models = Path(t) / "data" / "stemnet_v2" / "models"
        ck = torch.load(models / "head_last.pth", weights_only=False)
        assert all(bool(torch.isfinite(v).all()) for v in ck["head"].values())
        env["V2_STEPS"] = "18"
        r = subprocess.run([sys.executable, str(V2 / "train.py")], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "resuming from step 12" in r.stdout, r.stdout[-1500:]
        best_step = torch.load(models / "head_best.pth", weights_only=False)["step"]
        ck = torch.load(models / "head_last.pth", weights_only=False)   # a NaN'd latest state
        ck["head"] = {k: v * float("nan") for k, v in ck["head"].items()}
        torch.save(ck, models / "head_last.pth")
        env["V2_STEPS"] = "20"
        r = subprocess.run([sys.executable, str(V2 / "train.py")], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "head_last.pth (step 18) has non-finite weights" in r.stdout, r.stdout[-1500:]
        assert f"resuming from step {best_step}" in r.stdout, r.stdout[-1500:]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 train test(s) {'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
