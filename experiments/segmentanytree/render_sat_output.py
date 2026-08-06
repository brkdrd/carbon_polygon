#!/usr/bin/env python3
"""Render SegmentAnyTree output into a result image (CPU-only postprocess).

SegmentAnyTree runs as its own official Docker image (maciekwielgosz/segment-any-tree)
and writes segmented clouds to <out>/final_results/*.laz with a per-point
`PredInstance` extra dim (uint16; 0 = unassigned, 1..N = tree instances). This
step reads that and colours it consistently with the TreeLearn results.

Used for two experiments, selected by EXPERIMENT (`raw` -> exp2, `masked` -> exp4).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from common import config as C
from common import viz

SAT_DIR = C.DATA / "sat"


def read_sat(out_dir: Path):
    import laspy
    hits = sorted(out_dir.rglob("final_results/*.la[sz]")) or sorted(out_dir.rglob("*.la[sz]"))
    if not hits:
        raise SystemExit(f"[sat] no output cloud under {out_dir} — did the SAT container run?")
    las = laspy.read(str(hits[0]))
    names = [d.name for d in las.point_format.dimensions]
    inst_dim = next((n for n in ("PredInstance", "preds_instance", "treeID", "instance")
                     if n in names), None)
    if inst_dim is None:
        raise SystemExit(f"[sat] no instance dim in {hits[0].name}; dims={names}")
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    inst = np.asarray(las[inst_dim], dtype=np.int64)
    print(f"[sat] read {hits[0].name} | instance dim '{inst_dim}'")
    return xyz, inst


def main():
    exp = os.environ.get("EXPERIMENT", "raw")
    if exp == "raw":
        out_dir, img, title = (SAT_DIR / "raw_out", C.IMG_EXP2,
                               "exp2 — SegmentAnyTree on raw chunk")
    elif exp == "masked":
        out_dir, img, title = (SAT_DIR / "masked_out", C.IMG_EXP4,
                               "exp4 — SegmentAnyTree on Sonata vegetation mask")
    else:
        raise SystemExit(f"[sat] unknown EXPERIMENT={exp}")

    xyz, inst = read_sat(out_dir)
    n = len(set(inst.tolist()) - {0, -1})
    print(f"[sat] {exp}: {n} tree instances")
    viz.plot_instances(xyz, inst, img, f"{title}  |  {n} trees", unassigned=(-1, 0))
    print(f"[sat] image -> {img}")


if __name__ == "__main__":
    main()
