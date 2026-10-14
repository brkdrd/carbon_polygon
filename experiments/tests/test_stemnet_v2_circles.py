#!/usr/bin/env python3
"""StemNet v2 prep part 3 (circle fit): a synthetic campus with every hard case
the real one has, the true stem axis of each tree known, and — like the real
scan — every survey point displaced from its trunk by one common shift.

  big       r 15 cm, seen all round, a crown above it
  oneside   r 30 cm, only 200 deg visible (mobile scanner), ON a tile edge
  sapling   r 3 cm
  fork_a/b  two r 8 cm stems 25 cm apart, both surveyed
  shrub     r 12 cm with a dense bush pressed against it
  hidden    surveyed, but no trunk anywhere near — only ground
on an 8% slope. Ground is NOT removed: the fit must keep it out on its own."""
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
SCRIPT = HERE / "stemnet_v2" / "prep_circles.py"
TILE = 5.0
ORIGIN = (362_400.0, 4_763_600.0, 40.0)        # UTM 52N, the campus area
EPSG = 32652
SHIFT = np.array([0.30, 0.40])                 # trunk lies NE of its survey point
sys.path.insert(0, str(HERE / "stemnet_v2"))


def ground(x):
    return 0.08 * x                            # 8% slope


def _trunk(rng, cx, cy, r, arc_deg=360.0, h=4.0, a0=0.0):
    """Points on a cylinder at ~2 cm spacing, like the 0.02 m grid."""
    na = max(int(np.radians(arc_deg) * r / 0.02), 6)
    a = a0 + np.linspace(0, np.radians(arc_deg), na, endpoint=False)
    z = np.arange(0.0, h, 0.02)
    A, Z = np.meshgrid(a, z)
    P = np.column_stack([cx + r * np.cos(A.ravel()), cy + r * np.sin(A.ravel()),
                         ground(cx) + Z.ravel()])
    return P + rng.normal(0, 0.003, P.shape)


def _scene(rng):
    g = np.arange(0.01, 14.0, 0.02)
    gx, gy = np.meshgrid(g, g)
    G = np.column_stack([gx.ravel(), gy.ravel(), ground(gx.ravel())])
    G[:, 2] += rng.normal(0, 0.005, len(G))
    trees = {                                   # name: (cx, cy, r, visible arc, bark angle)
        "big": (2.5, 2.5, 0.15, 360, 0.3),
        "oneside": (5.0, 7.0, 0.30, 200, 1.6),  # x = 5.0 is a tile edge
        "sapling": (8.0, 2.0, 0.03, 360, 2.0),
        "fork_a": (11.0, 11.0, 0.08, 360, np.pi),
        "fork_b": (11.25, 11.0, 0.08, 360, 0.0),
        "shrub": (3.0, 11.0, 0.12, 360, 4.0),
    }
    parts = [G]
    for cx, cy, r, arc, _ in trees.values():
        parts.append(_trunk(rng, cx, cy, r, arc, a0=0.5))
    b = rng.uniform(-0.45, 0.45, (40_000, 3))             # a dense bush against `shrub`
    b = b[np.linalg.norm(b[:, :2], axis=1) < 0.45]
    parts.append(np.column_stack([3.0 + 0.55 + b[:, 0], 11.0 + b[:, 1],
                                  ground(3.55) + 0.7 + b[:, 2] * 1.5]))
    c = rng.normal(0, 1.0, (30_000, 3))                   # a crown over `big`
    parts.append(np.column_stack([2.5 + c[:, 0] * 1.5, 2.5 + c[:, 1] * 1.5,
                                  ground(2.5) + 6.5 + c[:, 2]]))
    P = np.concatenate(parts)
    survey = {}
    for name, (cx, cy, r, _, ang) in trees.items():
        bark = np.array([cx + r * np.cos(ang), cy + r * np.sin(ang)])
        survey[name] = tuple(bark - SHIFT + rng.normal(0, 0.01, 2))
    survey["hidden"] = (7.0, 12.5)              # nothing within 1.5 m but ground
    return P, trees, survey


def _write(tmp: Path, P, survey, trees):
    from pyproj import Transformer
    d = tmp / "data" / "stemnet_v2" / "grid0.02"
    d.mkdir(parents=True)
    ti = np.floor(P[:, 0] / TILE).astype(int); tj = np.floor(P[:, 1] / TILE).astype(int)
    for i, j in sorted(set(zip(ti, tj))):
        m = (ti == i) & (tj == j)
        np.savez(d / f"tile_{i}_{j}.npz", xyz=P[m].astype(np.float32),
                 rgb=np.zeros((m.sum(), 3), np.uint8), intensity=np.zeros(m.sum(), np.uint8))
    (tmp / "data" / "stemnet_v2" / "frame.json").write_text(json.dumps(dict(
        epsg=EPSG, origin_m=list(ORIGIN), tile_m=TILE, grids={"0.02": {}})))
    inv = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    rows, attrs, names = ["base_1,43.118967237\t,131.889649040\t,114.216"], [], []
    for k, (name, (x, y)) in enumerate(survey.items(), 1):
        lon, lat = inv.transform(ORIGIN[0] + x, ORIGIN[1] + y)
        z = ORIGIN[2] + ground(x) - 0.4          # survey heights are off, as measured
        rows.append(f"{k},{lat:.9f}\t,{lon:.9f}\t,{z:.3f}")
        if name in ("big", "oneside"):
            girth = 2 * np.pi * trees[name][2] * 100
            attrs.append(f"{k},{lat:.9f},{lon:.9f},{z:.3f},{girth:.0f},1,Tilia,,0")
        names.append(name)
    (tmp / "stem.csv").write_text("\n".join(rows) + "\n")
    (tmp / "attrs.csv").write_text("n,lat,lon,z,girth_cm,code,species,comment,name_ambiguous\n"
                                   + "\n".join(attrs) + "\n")
    return names


def _env(tmp, **extra):
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"),
               V2_TILE=str(TILE), V2_STEM_MAP=str(tmp / "stem.csv"), V2_ATTRS=str(tmp / "attrs.csv"))
    env.update(extra)
    return env


def test_circle_fit_finds_the_nearest_trunk_and_measures_the_shift():
    rng = np.random.default_rng(11)
    P, trees, survey = _scene(rng)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        names = _write(tmp, P, survey, trees)
        r = subprocess.run([sys.executable, str(SCRIPT)], env=_env(tmp), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        rows = list(csv.DictReader(open(tmp / "data" / "stemnet_v2" / "bases.csv")))
        assert len(rows) == len(names), "every survey row must be kept"
        by = dict(zip(names, rows))
        axes = {n: np.array(v[:2]) for n, v in trees.items()}

        for name in trees:
            row = by[name]
            # the rule: the trunk axis nearest the survey point
            s = np.array(survey[name])
            want = min(axes, key=lambda n: np.hypot(*(axes[n] - s)))
            tol = 0.03 if want == "sapling" else 0.02
            assert row["status"] in (("ok", "weak") if want == "sapling" else ("ok",)), (name, row["status"])
            got = np.array([float(row["x"]), float(row["y"])])
            err = np.hypot(*(got - axes[want]))
            assert err <= tol, f"{name}: base {err * 100:.1f} cm from the {want} axis"
            rad = trees[want][2]
            assert abs(float(row["radius"]) - rad) <= max(0.02, 0.15 * rad), (name, row["radius"])

        h = by["hidden"]
        assert h["status"] == "no_fit", h["status"]
        assert abs(float(h["x"]) - survey["hidden"][0]) < 1e-3

        # the common shift is measured: median survey -> trunk vector ~ SHIFT
        fj = json.loads((tmp / "data" / "stemnet_v2" / "frame.json").read_text())
        o = fj["bases"]["offsets"]["ALL"]
        assert abs(o["median_dx_cm"] - SHIFT[0] * 100) < 8 and abs(o["median_dy_cm"] - SHIFT[1] * 100) < 8, o

        txt = (tmp / "res" / "stemnet_v2_circles.txt").read_text()
        assert "girth validation: 2 single-stem" in txt, txt
        for f in ("42_stemnet_v2_girth.png", "43_stemnet_v2_circle_examples.png",
                  "45_stemnet_v2_offsets.png"):
            assert (tmp / "res" / f).stat().st_size > 1000, f


def test_subset_run_fits_only_fully_covered_trees():
    rng = np.random.default_rng(11)
    P, trees, survey = _scene(rng)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        names = _write(tmp, P, survey, trees)
        (tmp / "data" / "stemnet_v2" / "grid0.02" / "tile_0_1.npz").unlink()
        r = subprocess.run([sys.executable, str(SCRIPT)], env=_env(tmp, V2_SUBSET="1"),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "SUBSET RUN" in r.stdout
        ids = {row["id"] for row in csv.DictReader(open(tmp / "data" / "stemnet_v2" / "bases.csv"))}
        k = {n: f"1-{i}" for i, n in enumerate(names, 1)}
        assert k["oneside"] not in ids, "a half-covered column was fitted"
        assert k["big"] in ids and k["sapling"] in ids


def test_ground_never_passes_as_a_trunk_and_a_lone_ring_does():
    from prep_circles import layer_circles
    rng = np.random.default_rng(2)
    g = np.arange(-1.5, 1.5, 0.02)
    gx, gy = np.meshgrid(g, g)
    D = np.column_stack([gx.ravel(), gy.ravel()])
    disc = D[np.hypot(D[:, 0], D[:, 1]) <= 1.5]
    assert layer_circles(disc, rng) == []                       # wide: skipped as ground
    patch = D[np.hypot(D[:, 0] - 0.5, D[:, 1]) <= 0.25]         # a small solid blob
    assert layer_circles(patch, rng) == [], "a solid patch passed as a hollow trunk"
    a = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ring = np.tile(np.column_stack([0.2 * np.cos(a), 0.2 * np.sin(a)]), (10, 1))
    ring += rng.normal(0, 0.003, ring.shape)
    f = layer_circles(np.concatenate([ring, patch + [1.0, 0.0]]), rng)
    assert len(f) == 1 and np.hypot(f[0]["cx"], f[0]["cy"]) < 0.01 and abs(f[0]["r"] - 0.2) < 0.01, f


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 circle test(s) "
          f"{'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
