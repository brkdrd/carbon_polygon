#!/usr/bin/env python3
"""StemNet v2 prep part 3 (circle fit): a synthetic campus with every hard case
the real one has, and the true stem axis of each tree known.

  big       r 15 cm, seen all round
  oneside   r 30 cm, only 200 deg visible (mobile scanner), ON a tile edge
  sapling   r 3 cm
  fork_a/b  two r 8 cm stems 25 cm apart, both surveyed
  shrub     r 12 cm with a dense bush pressed against it
  hidden    surveyed, but no trunk points at all — only ground around it
plus sloped ground everywhere and a crown above `big`. Ground is NOT removed:
the fit must keep it out on its own."""
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
    trees = {                                   # name: (cx, cy, r, visible arc, survey angle)
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
    # a dense bush pressed against `shrub`, 0-1.4 m high
    b = rng.uniform(-0.45, 0.45, (40_000, 3))
    b = b[np.linalg.norm(b[:, :2], axis=1) < 0.45]
    parts.append(np.column_stack([3.0 + 0.55 + b[:, 0], 11.0 + b[:, 1],
                                  ground(3.55) + 0.7 + b[:, 2] * 1.5]))
    # a crown over `big`
    c = rng.normal(0, 1.0, (30_000, 3))
    parts.append(np.column_stack([2.5 + c[:, 0] * 1.5, 2.5 + c[:, 1] * 1.5,
                                  ground(2.5) + 6.5 + c[:, 2]]))
    P = np.concatenate(parts)
    survey = {}
    for name, (cx, cy, r, _, ang) in trees.items():
        # the stick touched the bark, give or take a centimetre
        survey[name] = (cx + r * np.cos(ang) + rng.normal(0, 0.01),
                        cy + r * np.sin(ang) + rng.normal(0, 0.01))
    survey["hidden"] = (7.0, 12.0)              # nothing there but ground
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


def test_circle_fit_finds_every_axis_and_keeps_ground_out():
    rng = np.random.default_rng(11)
    P, trees, survey = _scene(rng)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        names = _write(tmp, P, survey, trees)
        env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"),
                   V2_TILE=str(TILE), V2_STEM_MAP=str(tmp / "stem.csv"),
                   V2_ATTRS=str(tmp / "attrs.csv"))
        r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        rows = list(csv.DictReader(open(tmp / "data" / "stemnet_v2" / "bases.csv")))
        assert len(rows) == len(names), "every survey row must be kept"
        by = dict(zip(names, rows))

        for name, (cx, cy, rad, _, _) in trees.items():
            row = by[name]
            tol = 0.03 if name == "sapling" else 0.02
            assert row["status"] in (("ok", "weak") if name == "sapling" else ("ok",)), (name, row["status"])
            err = np.hypot(float(row["x"]) - cx, float(row["y"]) - cy)
            assert err <= tol, f"{name}: derived base {err * 100:.1f} cm from the true axis"
            assert abs(float(row["radius"]) - rad) <= max(0.02, 0.15 * rad), (name, row["radius"])

        # the hidden tree: ground only, so no circle may be accepted, and the
        # row survives with the survey position
        h = by["hidden"]
        assert h["status"] == "no_fit", h["status"]
        assert abs(float(h["x"]) - survey["hidden"][0]) < 1e-3

        # forks share a group; the merged file has one row for them
        assert by["fork_a"]["fork_group"] == by["fork_b"]["fork_group"]
        merged = list(csv.DictReader(open(tmp / "data" / "stemnet_v2" / "bases_merged.csv")))
        assert len(merged) == len(names) - 1

        # girth validation ran on the two attributed trees
        txt = (tmp / "res" / "stemnet_v2_circles.txt").read_text()
        assert "girth validation: 2 single-stem" in txt, txt
        for f in ("42_stemnet_v2_girth.png", "43_stemnet_v2_circle_examples.png"):
            assert (tmp / "res" / f).stat().st_size > 1000, f


def test_flat_ground_alone_never_passes_as_a_trunk():
    """The empty-interior check: a dense disc of ground, survey point in the
    middle, every layer must be rejected."""
    sys.path.insert(0, str(HERE / "stemnet_v2"))
    from prep_circles import fit_layer
    rng = np.random.default_rng(2)
    g = np.arange(-1.0, 1.0, 0.02)
    gx, gy = np.meshgrid(g, g)
    D = np.column_stack([gx.ravel(), gy.ravel()])
    D = D[np.hypot(D[:, 0], D[:, 1]) <= 1.0]
    assert all(fit_layer(D, np.array([0.1, 0.0]), rng) is None for _ in range(20))
    # the same disc with a trunk standing in it is found: ground runs right up
    # to the bark and nothing is scanned inside the wood
    a = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ring = np.column_stack([0.2 * np.cos(a), 0.2 * np.sin(a)])
    # a 0.2 m layer holds ten 2 cm levels of trunk wall but only one of flat
    # ground, so in plan view the trunk is ten points deep per cell
    ring = np.tile(ring, (10, 1)) + rng.normal(0, 0.003, (600, 2))
    out = D[np.hypot(D[:, 0], D[:, 1]) > 0.21]
    f = fit_layer(np.concatenate([out, ring]), np.array([0.2, 0.0]), rng)
    assert f is not None and np.hypot(f["cx"], f["cy"]) < 0.01 and abs(f["r"] - 0.2) < 0.01, f


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
