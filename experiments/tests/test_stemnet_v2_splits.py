#!/usr/bin/env python3
"""StemNet v2 prep part 4 (splits): no training ground within the buffer of a
held-out tree, every survey row labelled exactly once, nothing dropped."""
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
SCRIPT = HERE / "stemnet_v2" / "prep_splits.py"
TILE = 25.0
ORIGIN = (362_400.0, 4_763_600.0, 40.0)
EPSG = 32652
W, H = 200.0, 150.0


def _write(tmp: Path, rng):
    from pyproj import Transformer
    d = tmp / "data" / "stemnet_v2" / "grid0.02"
    d.mkdir(parents=True)
    g = np.arange(0.25, W, 0.5)
    gx, gy = np.meshgrid(g, np.arange(0.25, H, 0.5))
    P = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    P = P[~((P[:, 0] > 150) & (P[:, 1] > 110))]                # an unscanned corner
    ti = np.floor(P[:, 0] / TILE).astype(int); tj = np.floor(P[:, 1] / TILE).astype(int)
    for i, j in sorted(set(zip(ti, tj))):
        m = (ti == i) & (tj == j)
        np.savez(d / f"tile_{i}_{j}.npz", xyz=P[m].astype(np.float32),
                 rgb=np.zeros((m.sum(), 3), np.uint8), intensity=np.zeros(m.sum(), np.uint8))
    (tmp / "data" / "stemnet_v2" / "frame.json").write_text(json.dumps(dict(
        epsg=EPSG, origin_m=list(ORIGIN), tile_m=TILE, grids={"0.02": {}})))
    trees = np.column_stack([rng.uniform(2, W - 2, 700), rng.uniform(2, H - 2, 700)])
    att = np.column_stack([rng.uniform(20, 55, 90), rng.uniform(20, 55, 90)])   # the patch
    inv = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    rows, arows = ["base_1,43.1\t,131.8\t,114.2"], []
    for k, (x, y) in enumerate(np.concatenate([att, trees]), 1):
        lon, lat = inv.transform(ORIGIN[0] + x, ORIGIN[1] + y)
        rows.append(f"{k},{lat:.9f}\t,{lon:.9f}\t,50.000")
        if k <= len(att):
            arows.append(f"{k},{lat:.9f},{lon:.9f},50,60,1,Tilia,,0")
    (tmp / "stem.csv").write_text("\n".join(rows) + "\n")
    (tmp / "attrs.csv").write_text("n,lat,lon,z,girth_cm,code,species,comment,name_ambiguous\n"
                                   + "\n".join(arows) + "\n")
    return len(att), len(att) + len(trees)


def _run(tmp, **extra):
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"),
               V2_TILE=str(TILE), V2_STEM_MAP=str(tmp / "stem.csv"), V2_ATTRS=str(tmp / "attrs.csv"))
    env.update(extra)
    r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    R = np.load(tmp / "data" / "stemnet_v2" / "regions.npz")
    rows = list(csv.DictReader(open(tmp / "data" / "stemnet_v2" / "splits.csv")))
    return R, rows


def _check(R, rows, n_att, n_all, buffer=10.0, crop=8):
    from scipy import ndimage
    split, occ = R["split"], R["occupancy"]
    assert len(rows) == n_all and len({r["id"] for r in rows}) == n_all, "every row once"
    assert all(r["split"] == "test" for r in rows if r["attributed"] == "1")
    # no train cell within the buffer of any held-out cell
    held = (split == 1) | (split == 2)
    dist = ndimage.distance_transform_edt(~held)
    assert dist[split == 0].min() > buffer, dist[split == 0].min()
    # every tree's label matches its cell
    for r in rows:
        c = split[int(float(r["rtk_y"])), int(float(r["rtk_x"]))]
        assert ["train", "val", "test", "buffer"][c] == r["split"], r
    # a crop counted as eligible really is all-train
    k = crop
    I = np.zeros((split.shape[0] + 1, split.shape[1] + 1), np.int64)
    I[1:, 1:] = (split == 0).cumsum(0).cumsum(1)
    full = (I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]) == k * k
    assert full.any()


def test_splits_hold_out_cleanly_and_keep_every_row():
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        n_att, n_all = _write(tmp, rng)
        R, rows = _run(tmp)
        _check(R, rows, n_att, n_all)
        fj = json.loads((tmp / "data" / "stemnet_v2" / "frame.json").read_text())
        v = fj["splits"]["val"]
        assert v["mode"] == "match"
        # the val square holds at least as many trees as the test patch — the
        # same statistical power — and not wildly more
        assert v["test_trees"] <= v["trees_in_square"] <= 1.5 * v["test_trees"], v
        n = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test", "buffer")}
        assert n["val"] > 0 and n["train"] > n["val"], n
        assert (tmp / "res" / "44_stemnet_v2_splits.png").stat().st_size > 1000

        R2, rows2 = _run(tmp, V2_VAL_MODE="strip")
        _check(R2, rows2, n_att, n_all)
        xs = [float(r["rtk_x"]) for r in rows2 if r["split"] == "val"]
        assert min(xs) > W / 2, "strip val sits at the high-x end"


def test_splits_use_the_shift_v2_bases_recorded():
    """The test zone is built from shifted survey points: with a 1.2 m shift
    recorded, every split label must equal the label of the SHIFTED position."""
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        n_att, n_all = _write(tmp, rng)
        _, raw = _run(tmp)                                  # no v2_bases yet: raw positions
        fp = tmp / "data" / "stemnet_v2" / "frame.json"
        fj = json.loads(fp.read_text()); fj["bases"] = dict(shift_m=[0.68, 1.02]); fp.write_text(json.dumps(fj))
        R, rows = _run(tmp)
        for a, b in zip(raw, rows):                         # every position moved by the shift
            assert abs(float(b["rtk_x"]) - float(a["rtk_x"]) - 0.68) < 1e-3, (a, b)
            assert abs(float(b["rtk_y"]) - float(a["rtk_y"]) - 1.02) < 1e-3, (a, b)
        split = R["split"]
        for r in rows:
            c = split[int(float(r["rtk_y"])), int(float(r["rtk_x"]))]
            assert ["train", "val", "test", "buffer"][c] == r["split"], r


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 split test(s) "
          f"{'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
