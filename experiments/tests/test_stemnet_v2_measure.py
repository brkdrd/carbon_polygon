#!/usr/bin/env python3
"""StemNet v2 measure stage: every number it reports must match a brute-force
count on a synthetic scan whose answers are known in advance.

Needs laspy + pyproj (in the stemnet_v2 image; `run_experiments v2_test`)."""
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
SCRIPT = HERE / "stemnet_v2" / "measure.py"
TILE = 5.0
LAT0, LON0 = 43.02, 131.89                     # the campus, roughly


def _scene(tmp: Path):
    """A sloped ground grid, two trunks (one straddling a tile edge), 1,000
    exact duplicates, and a stem map with a tree the scanner never saw."""
    import laspy
    from pyproj import Transformer
    rng = np.random.default_rng(0)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
    x0, y0 = tr.transform(LON0, LAT0)
    x0, y0 = np.floor(x0 / TILE) * TILE + 1.0, np.floor(y0 / TILE) * TILE + 1.0

    def ground(x):
        return 50.0 + 0.1 * (x - x0)

    g = np.arange(0.005, 12.0, 0.02)           # off the voxel walls on purpose
    gx, gy = np.meshgrid(g, g)
    G = np.column_stack([x0 + gx.ravel(), y0 + gy.ravel(), np.zeros(gx.size)])
    G[:, 2] = ground(G[:, 0])
    trunks = [(x0 + 3.0, y0 + 3.0, 0.15),
              (x0 + 4.0, y0 + 6.0, 0.20)]          # x0+4.0 = 1 m below a tile edge...
    # ...and a third one centred ON a tile edge so its points come from two tiles
    edge_x = (np.floor((x0 + 9.0) / TILE) * TILE)
    trunks.append((edge_x, y0 + 8.0, 0.18))
    T = []
    for cx, cy, r in trunks:
        a = np.arange(0, 2 * np.pi, 0.02 / r)
        h = np.arange(0.05, 6.0, 0.02)
        A, H = np.meshgrid(a, h)
        T.append(np.column_stack([cx + r * np.cos(A.ravel()), cy + r * np.sin(A.ravel()),
                                  ground(cx) + H.ravel()]))
    P = np.concatenate([G] + T)
    dup = P[rng.choice(len(P), 1000, replace=False)]
    P = np.concatenate([P, dup])

    raw = tmp / "data" / "raw"; raw.mkdir(parents=True)
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([np.floor(x0), np.floor(y0), 0.0])
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = P[:, 0], P[:, 1], P[:, 2]
    las.intensity = rng.integers(0, 4000, len(P)).astype(np.uint16)
    las.write(str(raw / "scan.laz"))

    inv = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
    rows = ["base_1,43.118967237\t,131.889649040\t,114.216"]
    gt = []
    for k, (cx, cy, r) in enumerate(trunks, 1):
        bx, by = cx + r, cy                     # the stick touches the bark
        lon, lat = inv.transform(bx, by)
        rows.append(f"{k},{lat:.9f}\t,{lon:.9f}\t,{ground(bx):.3f}")
        gt.append((bx, by, ground(bx)))
    lon, lat = inv.transform(x0 + 100.0, y0 + 100.0)   # never scanned
    rows.append(f"{len(trunks) + 1},{lat:.9f}\t,{lon:.9f}\t,60.000")
    gt.append((x0 + 100.0, y0 + 100.0, 60.0))
    (tmp / "stem.csv").write_text("\n".join(rows) + "\n")
    (tmp / "attrs.csv").write_text("n,lat,lon,z,girth_cm,code,species,comment,name_ambiguous\n"
                                   "1,0,0,0,94,1,Tilia,,0\n2,0,0,0,29;22,1,Acer,,0\n")
    return P, np.array(gt)


def _run(tmp: Path, extra=None):
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"),
               V2_STEM_MAP=str(tmp / "stem.csv"), V2_ATTRS=str(tmp / "attrs.csv"),
               V2_TILE=str(TILE), V2_CROPS="2,4", V2_CHUNK="200000")
    env.update(extra or {})
    r = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout, json.loads((tmp / "res" / "stemnet_v2_measure.json").read_text())


def test_measure_counts_match_brute_force():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        P, gt = _scene(tmp)
        # what the file actually stores: millimetre integers. Every brute-force
        # count below uses these, as the stage does — the unrounded floats put
        # points that round onto a voxel wall on the wrong side of it.
        P = np.round(P * 1000) / 1000
        out, R = _run(tmp)

        # every point streamed, duplicates exact
        assert R["header"]["point_count"] == len(P)
        assert R["pass1"]["points_streamed"] == len(P)
        Pi = np.round(P * 1000).astype(np.int64)
        n_dup = len(Pi) - len(np.unique(Pi, axis=0))
        assert R["pass2"]["exact_duplicates"] == n_dup, (R["pass2"]["exact_duplicates"], n_dup)

        # voxel counts exact at every grid
        org = np.array([np.floor(P[:, 0].min() / TILE) * TILE,
                        np.floor(P[:, 1].min() / TILE) * TILE, np.floor(P[:, 2].min())])
        # in integer millimetres: at UTM magnitudes a float64 coordinate is off by
        # ~1e-11 m, which divided by a 2 cm voxel already misplaces wall points
        Pi_mm = np.round(P * 1000).astype(np.int64)
        org_mm = np.round(org * 1000).astype(np.int64)
        for g in (0.02, 0.05, 0.1):
            want = len(np.unique((Pi_mm - org_mm) // int(round(g * 1000)), axis=0))
            got = R["pass2"]["voxels"][f"{g:g}"]["points"]
            assert got == want, (g, got, want)
            assert R["pass2"]["voxels"][f"{g:g}"]["exact"]

        # spacing: the scene is built on a 2 cm lattice
        sp = R["pass2"]["spacing_cm"]["p50"]
        assert 1.0 <= sp <= 2.5, sp

        # stem map: nothing dropped, base station kept aside
        gtR = R["stem_map"]
        assert gtR["tree_rows"] == 4 and len(gtR["base_stations"]) == 1
        assert gtR["with_points_2m"] == 3
        with open(tmp / "res" / "stemnet_v2_trees.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert int(rows[3]["n_r2"]) == 0                    # the unscanned tree, still listed
        for i in range(3):                                  # per-tree counts = brute force,
            d2 = (P[:, 0] - gt[i, 0]) ** 2 + (P[:, 1] - gt[i, 1]) ** 2   # incl. the edge trunk
            assert int(rows[i]["n_r1"]) == int((d2 <= 1.0).sum()), (i, rows[i]["n_r1"])
            near = (d2 <= 1.0) & (P[:, 2] >= gt[i, 2] + 0.3) & (P[:, 2] <= gt[i, 2] + 1.5)
            assert int(rows[i]["n_slice_gt"]) == int(near.sum()), (i, rows[i]["n_slice_gt"])
        # survey z and scan z share a frame in this scene
        assert abs(gtR["z_minus_low3"]["p50"]) < 0.1, gtR["z_minus_low3"]
        assert gtR["attributes"] == dict(rows=2, with_girth=2, multi_stem=1)

        for f in ("stemnet_v2_measure.txt", "40_stemnet_v2_density.png", "41_stemnet_v2_hist.png"):
            assert (tmp / "res" / f).stat().st_size > 500, f

        # a second run reuses the tile cache and reports the same numbers
        out2, R2 = _run(tmp)
        assert "matches this file" in out2
        assert R2["pass2"]["voxels"] == R["pass2"]["voxels"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 measure test(s) "
          f"{'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
