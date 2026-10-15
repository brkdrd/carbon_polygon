#!/usr/bin/env python3
"""StemNet v2 prep part 3 (bases): base = survey point + fixed shift, and the
audit must (a) see that points sit on bark where the shift is right, and (b)
flag a block that sits somewhere else.

Synthetic campus: 60 trunks (3-20 cm radius, >= 0.8 m apart) on sloped
ground; three survey blocks, every survey point on its trunk's bark, then
displaced. Blocks 1 and 2 by exactly the fixed shift; block 3 by the fixed
shift plus a further (40, 0) cm — the case the audit exists for."""
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
SCRIPT = HERE / "stemnet_v2" / "prep_bases.py"
TILE = 5.0
ORIGIN = (362_400.0, 4_763_600.0, 40.0)
EPSG = 32652
FIXED = np.array([0.68, 1.02])
EXTRA = {1: np.zeros(2), 2: np.zeros(2), 3: np.array([0.40, 0.0])}
W = 30.0


def ground(x):
    return 0.06 * x


def _trunk(rng, cx, cy, r, h=3.0, arc=360.0, a0=None):
    a0 = rng.uniform(0, 1) if a0 is None else a0
    na = max(int(np.radians(arc) * r / 0.02), 8)
    a = np.linspace(0, np.radians(arc), na, endpoint=(arc < 360)) + a0
    z = np.arange(0.0, h, 0.02)
    A, Z = np.meshgrid(a, z)
    P = np.column_stack([cx + r * np.cos(A.ravel()), cy + r * np.sin(A.ravel()), ground(cx) + Z.ravel()])
    return P + rng.normal(0, 0.003, P.shape)


def _scene(rng):
    pos = []
    while len(pos) < 60:
        p = rng.uniform(3, W - 3, 2)
        if all(np.hypot(*(p - q)) >= 0.8 for q in pos):
            pos.append(p)
    pos = np.array(pos)
    rad = rng.uniform(0.03, 0.20, len(pos))
    g = np.arange(0.02, W, 0.04)
    gx, gy = np.meshgrid(g, g)
    G = np.column_stack([gx.ravel(), gy.ravel(), ground(gx.ravel())])
    # the last 12 trunks are seen from one side only (70 deg of bark): too
    # little arc for a circle, so with a girth they are tier B, without one C
    face = rng.uniform(0, 2 * np.pi, len(pos))
    parts = [G]
    for i, ((x, y), r) in enumerate(zip(pos, rad)):
        parts.append(_trunk(rng, x, y, r, arc=70.0, a0=face[i] - np.radians(35)) if i >= 48
                     else _trunk(rng, x, y, r))
    P = np.concatenate(parts)
    survey = []
    for i in range(60):
        b = 1 + i % 3
        ang = face[i] if i >= 48 else rng.uniform(0, 2 * np.pi)   # the stick touches visible bark
        bark = pos[i] + rad[i] * np.array([np.cos(ang), np.sin(ang)])
        s = bark - FIXED - EXTRA[b] + rng.normal(0, 0.005, 2)
        survey.append((b, s[0], s[1], i))
    return P, pos, rad, survey


def _write(tmp: Path, P, survey, girth=None):
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
    lines, arows = [], []
    girth = girth or {}
    for b in (1, 2, 3):
        lines.append(f"base_{b},43.1\t,131.8\t,100.0")
        for k, (bb, x, y, _) in enumerate(survey):
            if bb == b:
                lon, lat = inv.transform(ORIGIN[0] + x, ORIGIN[1] + y)
                lines.append(f"{k + 1},{lat:.9f}\t,{lon:.9f}\t,{ORIGIN[2] + ground(x) - 0.4:.3f}")
                if k in girth:
                    arows.append(f"{k + 1},{lat:.9f},{lon:.9f},50,{girth[k]},1,Tilia,,0")
    (tmp / "stem.csv").write_text("\n".join(lines) + "\n")
    (tmp / "attrs.csv").write_text("n,lat,lon,z,girth_cm,code,species,comment,name_ambiguous\n"
                                   + "".join(a + "\n" for a in arows))


def _env(tmp, **extra):
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"),
               V2_TILE=str(TILE), V2_STEM_MAP=str(tmp / "stem.csv"), V2_ATTRS=str(tmp / "attrs.csv"),
               V2_AREA="15", V2_AREA_MIN="8")
    env.update(extra)
    return env


def test_bases_are_shifted_survey_points_and_the_audit_catches_a_block_that_is_off():
    rng = np.random.default_rng(5)
    P, pos, rad, survey = _scene(rng)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _write(tmp, P, survey)
        r = subprocess.run([sys.executable, str(SCRIPT)], env=_env(tmp), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        rows = list(csv.DictReader(open(tmp / "data" / "stemnet_v2" / "bases.csv")))
        assert len(rows) == len(survey), "every survey row kept"
        for row in rows:                                   # bark point = survey + fixed shift, exactly
            assert abs(float(row["bark_x"]) - float(row["rtk_x"]) - FIXED[0]) < 1e-3
            assert abs(float(row["bark_y"]) - float(row["rtk_y"]) - FIXED[1]) < 1e-3
            assert row["quality"] in ("on_bark", "near", "far")
        A = json.loads((tmp / "data" / "stemnet_v2" / "frame.json").read_text())["bases"]["audit"]
        for b in ("1", "2"):
            a = A["blocks"][b]
            assert a["share_fixed"] >= 80, (b, a)            # on the bark where the shift is right
            assert a["best_offset_cm"] <= 6, (b, a)
        a3 = A["blocks"]["3"]
        assert a3["share_fixed"] <= 30, a3                   # block 3 is not on bark
        assert abs(a3["best_shift"][0] - (FIXED[0] + 0.40)) <= 0.06 and abs(a3["best_shift"][1] - FIXED[1]) <= 0.06, a3
        txt = (tmp / "res" / "stemnet_v2_bases.txt").read_text()
        assert "block 3" in txt and "WARNING" in txt.split("block 3")[1].split("\n")[0], txt
        for b in ("1", "2"):
            assert "WARNING" not in txt.split(f"block {b}")[1].split("\n")[0]
        assert (tmp / "res" / "45_stemnet_v2_bases_audit.png").stat().st_size > 1000


def test_tiered_bases_move_from_the_bark_to_the_axis():
    """A: a circle touching the shifted point -> its centre. B: no circle but a
    single-stem girth -> moved inward by the girth radius along the bark normal.
    C: neither -> the shifted point. A circle whose radius contradicts the
    tree's own girth is someone else's trunk and must not be used."""
    rng = np.random.default_rng(5)
    P, pos, rad, survey = _scene(rng)
    girth = {k: f"{2 * np.pi * rad[t] * 100:.1f}" for k, (b, _, _, t) in enumerate(survey)
             if b in (1, 2) and (k % 2 == 0 or t >= 48)}
    bad = next(k for k, (b, _, _, t) in enumerate(survey) if b in (1, 2) and k % 2 == 0 and t < 48 and rad[t] >= 0.12)
    girth[bad] = f"{2 * np.pi * rad[survey[bad][3]] * 100 * 0.25:.1f}"          # a quarter of the truth
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _write(tmp, P, survey, girth)
        r = subprocess.run([sys.executable, str(SCRIPT)], env=_env(tmp), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        rows = list(csv.DictReader(open(tmp / "data" / "stemnet_v2" / "bases.csv")))
        assert len(rows) == len(survey)
        by = {(int(r_["block"]), int(r_["n"])): r_ for r_ in rows}
        err = {"A": [], "B": [], "C": []}
        bark_err = []
        for k, (b, _, _, tr) in enumerate(survey):
            if b == 3:
                continue                                    # displaced block: not on bark
            row = by[(b, k + 1)]
            base = np.array([float(row["x"]), float(row["y"])])
            bark = np.array([float(row["bark_x"]), float(row["bark_y"])])
            e = np.hypot(*(base - pos[tr]))
            if k != bad:
                err[row["base_tier"]].append(e)
                if row["base_tier"] == "B":
                    bark_err.append(np.hypot(*(bark - pos[tr])))
            if row["base_tier"] == "C":
                assert np.allclose(base, bark), row
            if k == bad:
                assert row["base_tier"] != "A", "a circle contradicting the girth was accepted"
        assert len(err["A"]) >= 20, {t: len(v) for t, v in err.items()}
        assert np.max(err["A"]) <= 0.025, np.max(err["A"])            # circle centre = axis
        assert len(err["B"]) >= 4, {t: len(v) for t, v in err.items()}       # the one-sided trunks
        assert np.median(err["B"]) < 0.5 * np.median(bark_err), (np.median(err["B"]), np.median(bark_err))
        txt = (tmp / "res" / "stemnet_v2_bases.txt").read_text()
        assert "tier B" in txt and "audit of tier B" in txt, txt


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 bases test(s) {'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
