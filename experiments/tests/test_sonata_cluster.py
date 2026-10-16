#!/usr/bin/env python3
"""cluster_square on CPU with a stand-in encoder: everything except Sonata.

A synthetic scan (ground + upright cylinders) at UTM-sized coordinates goes
through square picking, the streaming read, grid cells, tiled embedding,
k-means and the write. Checks: the square is read exactly, coordinates come
back bit-identical, every point gets one embedding even with several passes,
and clusters land on the geometry the embedding separates."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sonata_cluster"))
import cluster_square as cs  # noqa: E402

E0, N0, W = 412_345.0, 6_178_901.0, 40.0


def _scan(path: Path) -> None:
    rng = np.random.default_rng(3)
    g = np.c_[rng.uniform(0, W, 400_000), rng.uniform(0, W, 400_000),
              rng.normal(0, 0.01, 400_000)]
    trunks = []
    for cx, cy in rng.uniform(3, W - 3, (60, 2)):
        a, z = rng.uniform(0, 2 * np.pi, 4000), rng.uniform(0.05, 6, 4000)
        trunks.append(np.c_[cx + 0.2 * np.cos(a), cy + 0.2 * np.sin(a), z])
    t = np.concatenate(trunks)
    xyz = np.concatenate([g, t])
    kind = np.r_[np.zeros(len(g)), np.ones(len(t))].astype(np.uint8)
    o = rng.permutation(len(xyz))                     # scan order is not spatial
    h = laspy.LasHeader(point_format=1, version="1.2")
    h.offsets, h.scales = [E0, N0, 0], [0.001] * 3
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[o, 0] + E0, xyz[o, 1] + N0, xyz[o, 2]
    las.classification = kind[o]                      # ground truth for the test
    las.write(path)


def _stand_in(calls):
    def enc(coord, color, normal):
        calls.append(len(coord))
        nz = np.abs(normal[:, 2:3])
        return np.c_[nz, 1 - nz, np.zeros((len(coord), 6))].astype(np.float16)
    return enc


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "scan.laz"
        _scan(src)
        calls = []
        cfg = dict(source=str(src), square=15.0, center=None, seed=7, k=2,
                   grid=0.02, tile=6.0, margin=1.0, cover=0.5)
        meta = cs.run(cfg, encoder=_stand_in(calls), out_dir=d / "out")

        full = laspy.read(src)
        x0, y0 = meta["corner"]
        inside = ((full.x >= x0) & (full.x < x0 + 15) &
                  (full.y >= y0) & (full.y < y0 + 15))
        out = laspy.read(d / "out" / "square_seed7_15m_k2.laz")
        assert len(out.points) == inside.sum() == meta["points"], \
            (len(out.points), inside.sum())
        a = np.sort(np.c_[full.X[inside], full.Y[inside], full.Z[inside]].view(
            "i4,i4,i4").ravel())
        b = np.sort(np.c_[out.X, out.Y, out.Z].view("i4,i4,i4").ravel())
        assert (a == b).all(), "coordinates changed on the way through"
        assert len(calls) >= 4, f"expected several passes, got {calls}"

        lab = np.asarray(out.cluster)
        assert set(np.unique(lab)) <= {0, 1}
        pal = (cs.palette(2)[lab] * 65535).round().astype(np.uint16)
        assert (np.c_[out.red, out.green, out.blue] == pal).all()
        kind = np.asarray(out.classification)
        agree = max(np.mean(lab == kind), np.mean(lab != kind))
        assert agree > 0.95, f"ground/trunk split only {agree:.1%}"
        assert lab.tolist().count(0) >= lab.tolist().count(1)   # 0 = largest
        assert (d / "out" / "square_seed7_15m_k2.png").stat().st_size > 10_000
        print(f"ok: {meta['points']:,} pts, {len(calls)} passes, "
              f"cluster/geometry agreement {agree:.1%}, coverage {meta['coverage']:.0%}")


if __name__ == "__main__":
    main()
