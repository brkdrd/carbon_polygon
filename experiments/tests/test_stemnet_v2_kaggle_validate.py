#!/usr/bin/env python3
"""CPU checks for the independent Kaggle validation adapter."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "stemnet_v2"))
import data as D  # noqa: E402
import validate_kaggle as K  # noqa: E402


def _cloud(path, instances=False):
    hdr = laspy.LasHeader(point_format=3, version="1.4")
    hdr.scales = np.array([0.01, 0.01, 0.01])
    hdr.offsets = np.array([736000.0, 4767000.0, 50.0])
    if instances:
        hdr.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.int32))
    las = laspy.LasData(hdr)
    rng = np.random.default_rng(2)
    ids = np.repeat(np.array([0, 1, 2]), [20, 50, 50])
    centres = np.array([[5, 5], [10, 10], [20, 20]], np.float64)
    xy = centres[ids] + rng.normal(0, 0.02, (len(ids), 2))
    z = np.r_[rng.uniform(0, 1, 20), np.linspace(0, 8, 50), np.linspace(0, 10, 50)]
    las.x = 736000 + xy[:, 0]
    las.y = 4767000 + xy[:, 1]
    las.z = 50 + z
    if instances:
        las.tree_id = ids
    las.write(path)
    return centres


def test_csv_truth_accepts_global_and_local_coordinates():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cloud = t / "plot.laz"
        centres = _cloud(cloud)
        frame = dict(origin_m=[736000.0, 4767000.0, 50.0])
        for local in (False, True):
            q = t / f"bases_{local}.csv"
            with open(q, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tree_id", "easting", "northing"])
                for i, (x, y) in enumerate(centres[1:]):
                    w.writerow([i + 1, x if local else x + 736000, y if local else y + 4767000])
            xy, ids, source = K.csv_truth(q, cloud, frame)
            assert np.allclose(xy, centres[1:])
            assert ids == ["1", "2"]
            assert ("local" in source) == local


def test_instance_truth_uses_low_points_and_ignores_background():
    with tempfile.TemporaryDirectory() as t:
        cloud = Path(t) / "instances.laz"
        centres = _cloud(cloud, instances=True)
        xy, ids, source = K.instance_truth(cloud, dict(origin_m=[736000.0, 4767000.0, 50.0]))
        assert ids == ["1", "2"]
        assert np.max(np.abs(xy - centres[1:])) < 0.03, (xy, centres)
        assert "tree_id" in source


def test_external_tile_store_is_isolated_and_label_free():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        gd, fd = root / "grid0.02", root / "feat0.02"
        gd.mkdir()
        fd.mkdir()
        xyz = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0]], np.float32)
        np.savez(gd / "tile_0_0.npz", xyz=xyz, rgb=np.zeros((2, 3), np.uint8), intensity=np.zeros(2, np.uint8))
        feat = np.zeros((2, 12), np.float16)
        feat[:, 10] = 1
        np.savez(fd / "tile_0_0.npz", feat=feat)
        store = D.TileStore(root=root, labels=False)
        crop = store.crop(1.5, 1.5, 4)
        assert len(crop["xyz"]) == 2
        assert "gate" not in crop and np.array_equal(crop["normal"][:, 2], np.ones(2))


def test_external_cloud_uses_the_real_grid_and_feature_pipeline():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        cloud = root / "download" / "plot.laz"
        cloud.parent.mkdir()
        _cloud(cloud)
        K.LOG = root / "validate.log"
        prepared = K.prepared_cloud(root, cloud)
        frame = json.loads((prepared / "frame.json").read_text())
        assert frame["source"].endswith("plot.laz")
        store = D.TileStore(root=prepared, labels=False)
        assert sum(len(store.tile(ij)["xyz"]) for ij in store.have) > 0
        bbox, occ = K.bbox_and_occupancy(store)
        assert bbox[0] == bbox[2] == 0 and occ.sum() > 0


def test_file_selection_and_edge_band_are_explicit():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "plot.laz").touch()
        (t / "tree_bases.csv").write_text("x,y\n1,2\n")
        cloud, gt = K.select_files(t)
        assert cloud.name == "plot.laz" and gt.name == "tree_bases.csv"
    xy = np.array([[1, 1], [4, 4], [16, 16], [19, 19]], float)
    assert K.inset_mask(xy, (0, 20, 0, 20), 4).tolist() == [False, True, False, False]
    assert K.safe_slug("owner/dataset") == "owner__dataset"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} Kaggle validation test(s) {'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
