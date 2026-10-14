#!/usr/bin/env python3
"""StemNet v2 prep part 3: discover trunks, register each survey session,
match. The synthetic campus copies what the real one showed:

  * two survey sessions over the SAME ground — block 1 displaced by one common
    shift (70, 90) cm, block 2 on the trunks
  * crowded: 50 surveyed trunks plus 20 unsurveyed ones (trees nobody
    surveyed, or posts) interleaved with them, 3-20 cm radius
  * every survey point on its trunk's bark, give or take a centimetre
  * one survey point over empty ground
  * sloped ground left in the cloud
"""
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
ORIGIN = (362_400.0, 4_763_600.0, 40.0)
EPSG = 32652
# blocks 1 and 3 share one shift (block 3 is too small to register alone and
# must take the pooled one); block 2 is not shifted
SHIFT = {1: np.array([0.70, 0.90]), 2: np.array([0.0, 0.0]), 3: np.array([0.70, 0.90])}
W = 30.0
sys.path.insert(0, str(HERE / "stemnet_v2"))


def ground(x):
    return 0.06 * x


def _trunk(rng, cx, cy, r, h=4.0):
    na = max(int(2 * np.pi * r / 0.02), 8)
    a = np.linspace(0, 2 * np.pi, na, endpoint=False) + rng.uniform(0, 1)
    z = np.arange(0.0, h, 0.02)
    A, Z = np.meshgrid(a, z)
    P = np.column_stack([cx + r * np.cos(A.ravel()), cy + r * np.sin(A.ravel()), ground(cx) + Z.ravel()])
    return P + rng.normal(0, 0.003, P.shape)


def _scene(rng):
    pos = []
    while len(pos) < 70:                                   # >= 0.6 m apart, like a crowded campus
        p = rng.uniform(2.5, W - 2.5, 2)
        if all(np.hypot(*(p - q)) >= 0.6 for q in pos):
            pos.append(p)
    pos = np.array(pos)
    rad = rng.uniform(0.03, 0.20, len(pos))
    g = np.arange(0.02, W, 0.04)
    gx, gy = np.meshgrid(g, g)
    G = np.column_stack([gx.ravel(), gy.ravel(), ground(gx.ravel())])
    P = np.concatenate([G] + [_trunk(rng, x, y, r) for (x, y), r in zip(pos, rad)])
    survey = []                                            # (block, x, y, tree index or -1)
    for i in range(50):                                    # first 50 surveyed, interleaved:
        b = 2 if i % 5 in (1, 3) else 1                    # 30 in block 1, 20 in block 2
        ang = rng.uniform(0, 2 * np.pi)
        bark = pos[i] + rad[i] * np.array([np.cos(ang), np.sin(ang)])
        s = bark - SHIFT[b] + rng.normal(0, 0.01, 2)
        survey.append((b, s[0], s[1], i))
    for i in range(50, 55):                                # block 3: five points only
        ang = rng.uniform(0, 2 * np.pi)
        bark = pos[i] + rad[i] * np.array([np.cos(ang), np.sin(ang)])
        s = bark - SHIFT[3] + rng.normal(0, 0.01, 2)
        survey.append((3, s[0], s[1], i))
    while True:                                            # a survey point over empty ground
        p = rng.uniform(4, W - 4, 2)
        if np.min(np.hypot(*(pos - p).T)) > 1.5:
            survey.append((2, p[0], p[1], -1))
            break
    return P, pos, rad, survey


def _write(tmp: Path, P, survey, pos, rad):
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
    lines, attrs = [], []
    for b in (1, 2, 3):
        lines.append(f"base_{b},43.1\t,131.8\t,100.0")
        for k, (bb, x, y, tree) in enumerate(survey):
            if bb != b:
                continue
            lon, lat = inv.transform(ORIGIN[0] + x, ORIGIN[1] + y)
            lines.append(f"{k + 1},{lat:.9f}\t,{lon:.9f}\t,{ORIGIN[2] + ground(x) - 0.4:.3f}")
            if tree >= 0 and k < 10:
                attrs.append(f"{k + 1},{lat:.9f},{lon:.9f},50,{2 * np.pi * rad[tree] * 100:.0f},1,Tilia,,0")
    (tmp / "stem.csv").write_text("\n".join(lines) + "\n")
    (tmp / "attrs.csv").write_text("n,lat,lon,z,girth_cm,code,species,comment,name_ambiguous\n"
                                   + "\n".join(attrs) + "\n")


def _env(tmp, **extra):
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"),
               V2_TILE=str(TILE), V2_STEM_MAP=str(tmp / "stem.csv"), V2_ATTRS=str(tmp / "attrs.csv"),
               V2_SCAN_H="4.5")
    env.update(extra)
    return env


def test_register_each_session_then_match_every_survey_point_to_its_own_trunk():
    rng = np.random.default_rng(21)
    P, pos, rad, survey = _scene(rng)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _write(tmp, P, survey, pos, rad)
        r = subprocess.run([sys.executable, str(SCRIPT)], env=_env(tmp), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        fj = json.loads((tmp / "data" / "stemnet_v2" / "frame.json").read_text())
        reg = fj["bases"]["registration"]
        for b in (1, 2):
            got = np.array(reg[str(b)]["shift_m"])
            assert reg[str(b)]["applied"], reg[str(b)]
            assert np.hypot(*(got - SHIFT[b])) <= 0.03, f"block {b}: shift {got} vs {SHIFT[b]}"
        assert not reg["3"]["applied"] and reg["3"]["pooled_m"] is not None, reg["3"]

        rows = list(csv.DictReader(open(tmp / "data" / "stemnet_v2" / "bases.csv")))
        assert len(rows) == len(survey), "every survey row kept"
        key = {(int(r["block"]), r["n"]): r for r in rows}
        wrong = 0
        for k, (b, x, y, tree) in enumerate(survey):
            row = key[(b, str(k + 1))]
            if tree < 0:
                assert row["status"] == "no_fit", row
                continue
            assert row["status"] == "ok", (k, row["status"])
            err = np.hypot(float(row["x"]) - pos[tree, 0], float(row["y"]) - pos[tree, 1])
            wrong += err > 0.02
            assert abs(float(row["resid_m"])) <= 0.04, (k, row["resid_m"])
        assert wrong == 0, f"{wrong} survey points matched to the wrong trunk"

        trunks = list(csv.DictReader(open(tmp / "data" / "stemnet_v2" / "trunks.csv")))
        full = [t for t in trunks if int(t["n_layers"]) >= 3]
        assert len(full) >= 66, len(full)                  # nearly all 70 trunks found
        unclaimed = sum(int(t["survey_rows"]) == 0 for t in full)
        assert 12 <= unclaimed <= 17, unclaimed            # the 15 unsurveyed ones

        txt = (tmp / "res" / "stemnet_v2_circles.txt").read_text()
        assert "girth:" in txt, txt
        for f in ("42_stemnet_v2_girth.png", "43_stemnet_v2_circle_examples.png",
                  "45_stemnet_v2_registration.png"):
            assert (tmp / "res" / f).stat().st_size > 1000, f


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


def test_registration_finds_a_shift_only_when_there_is_one():
    """A clean shift gives a sharp peak; survey points scattered with no
    relation to the trunks give a flat surface, and must not be 'registered'."""
    from prep_circles import register, REG_SHARP, REG_SECOND

    def accepted(g):
        return g["sharpness"] >= REG_SHARP and g["second_ratio"] <= REG_SECOND
    rng = np.random.default_rng(4)
    T = rng.uniform(0, 30, (300, 2)); R = rng.uniform(0.03, 0.2, 300)
    ang = rng.uniform(0, 2 * np.pi, 80)
    on = T[:80] + R[:80, None] * np.column_stack([np.cos(ang), np.sin(ang)]) - [0.6, -0.4]
    good = register(on, T, R)
    assert np.hypot(*(good["shift"] - [0.6, -0.4])) <= 0.02 and accepted(good), good["shift"]
    # only a third of the points on a trunk, the rest scattered: still found
    mixed = register(np.concatenate([on[:25], rng.uniform(0, 30, (55, 2))]), T, R)
    assert np.hypot(*(mixed["shift"] - [0.6, -0.4])) <= 0.02 and accepted(mixed)
    for seed in range(4):                      # no relation at all: never accepted
        noise = register(np.random.default_rng(seed).uniform(0, 30, (80, 2)), T, R)
        assert not accepted(noise), (noise["sharpness"], noise["second_ratio"])


def test_registration_accepts_a_broad_peak_from_scattered_points():
    """The real survey scatters 10-35 cm around the common shift, which makes
    the peak broad. Its shoulder is not a rival answer."""
    from prep_circles import register, REG_SHARP, REG_SECOND
    # 200 points scattered by 18 cm pin the shift to a few cm, not to 1 cm: the
    # score maximum wanders 2-9 cm with the draw. Matching tolerates 40 cm, so
    # that precision is enough; what must never happen is rejecting the block.
    for seed in (9, 1, 2, 3, 4):
        rng = np.random.default_rng(seed)
        T = rng.uniform(0, 40, (500, 2)); R = rng.uniform(0.03, 0.2, 500)
        ang = rng.uniform(0, 2 * np.pi, 200)
        S = (T[:200] + R[:200, None] * np.column_stack([np.cos(ang), np.sin(ang)])
             - [0.7, 1.0] + rng.normal(0, 0.18, (200, 2)))
        g = register(S, T, R)
        assert np.hypot(*(g["shift"] - [0.7, 1.0])) <= 0.10, (seed, g["shift"])
        assert g["sharpness"] >= REG_SHARP and g["second_ratio"] <= REG_SECOND, \
            (seed, g["sharpness"], g["second_ratio"])


def test_one_failed_layer_does_not_split_a_trunk():
    from prep_circles import chains
    c = dict(cx=1.0, cy=2.0, r=0.1, rms=0.005, arc=300.0)
    layers = [(k, 0.2 * k, [] if k in (6, 7) else [dict(c)]) for k in range(15)]
    ch = [x for x in chains(layers) if len(x) >= 2]
    assert len(ch) == 1 and len(ch[0]) == 13, [len(x) for x in ch]
    layers = [(k, 0.2 * k, [] if k in (5, 6, 7) else [dict(c)]) for k in range(15)]
    assert len([x for x in chains(layers) if len(x) >= 2]) == 2      # 3 missing: two pieces


def test_matching_is_one_to_one():
    """Two survey points both closest to one trunk's bark: the better one gets
    it, the other its own next trunk — never both on the same stem."""
    from prep_circles import match
    trunks = [dict(x=0.0, y=0.0, radius=0.1, n_layers=5), dict(x=0.5, y=0.0, radius=0.1, n_layers=5)]
    S = np.array([[0.11, 0.0], [0.13, 0.05]])
    m = match(S, trunks)
    assert m[0][0] == 0 and m[1][0] == 1, m


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
