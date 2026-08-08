#!/usr/bin/env python3
"""Run TreeLearn instance segmentation on one chunk and render the result.

Used for two of the four experiments:
  * exp1 : raw chunk        (chunk_local.laz)
  * exp3 : Sonata-masked    (chunk_masked_local.laz)

Both feed the SAME upstream pipeline; only the input file and output image differ,
selected by the EXPERIMENT env var (`raw` | `masked`).

TreeLearn (github.com/ecker-lab/TreeLearn) centers coordinates internally, and our
chunk is already local, so the float32 trap is doubly avoided. Output carries a
per-point `treeID` extra dim (0 = non-tree, 1..N = instances).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "/app")
from common import config as C
from common import viz

TL_REPO = Path(os.environ.get("TL_REPO", "/opt/TreeLearn"))
CKPT_DATASET = os.environ.get("TL_WEIGHTS", "model_weights_20241213")


def ensure_weights() -> Path:
    """Download the pretrained checkpoint into the persistent models volume."""
    dst_dir = C.MODELS / "treelearn"
    ckpt = dst_dir / f"{CKPT_DATASET}.pth"
    if ckpt.exists():
        print(f"[treelearn] weights present: {ckpt}")
        return ckpt
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[treelearn] downloading {CKPT_DATASET} ...")
    r = subprocess.run(
        [sys.executable, "tree_learn/util/download.py",
         "--dataset_name", CKPT_DATASET, "--root_folder", str(dst_dir)],
        cwd=str(TL_REPO), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("[treelearn] weight download failed")
    if not ckpt.exists():
        hits = list(dst_dir.rglob("*.pth"))
        if not hits:
            raise SystemExit("[treelearn] no .pth after download")
        ckpt = hits[0]
    return ckpt


def build_config(forest_path: Path, ckpt: Path, results_dir: Path) -> Path:
    """Clone the repo's pipeline.yaml and override the three paths we control.
    Kept in the repo tree so its relative `default_args` includes still resolve."""
    base = TL_REPO / "configs" / "pipeline" / "pipeline.yaml"
    cfg = yaml.safe_load(base.read_text())
    cfg["forest_path"] = str(forest_path)
    cfg["pretrain"] = str(ckpt)
    cfg["tile_generation"] = True
    cfg.setdefault("save_cfg", {})
    cfg["save_cfg"]["results_dir"] = str(results_dir)
    cfg["save_cfg"]["save_formats"] = ["laz"]
    cfg["save_cfg"]["save_pointwise"] = True
    cfg["save_cfg"]["return_type"] = "original"
    # We only consume the full-forest laz; the treewise branch also breaks on
    # zero-instance results (indexes per-instance arrays that are then empty).
    cfg["save_cfg"]["save_treewise"] = False
    out = TL_REPO / "configs" / "pipeline" / "_run.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def report_filter_diagnostics(results_dir: Path):
    """Show how many points pass each of TreeLearn's clustering filters
    (thresholds from configs/_modular/grouping.yaml: 0.5 / 0.6 / 4). Explains
    WHY an instance count is low/zero — e.g. a vegetation mask that kept canopy
    but dropped the trunk points TreeLearn clusters on."""
    p = next(results_dir.rglob("pointwise_results.npz"), None)
    if p is None:
        return
    d = np.load(p)
    logits = d["semantic_prediction_logits"].astype(np.float64)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    tree = probs[:, 0] >= 0.5           # tree class = 0
    vert = d["input_feats"][:, -1] > 0.6
    off = np.abs(d["offset_predictions"][:, 2]) < 4
    print(f"[treelearn] clustering filters on {len(tree):,} voxelized pts: "
          f"tree-conf {tree.sum():,} | verticality {vert.sum():,} | "
          f"z-offset {off.sum():,} | all three {(tree & vert & off).sum():,}")


def read_instances(results_dir: Path):
    """Return (xyz, instance_labels) from the TreeLearn result. Prefers the
    full-forest laz (`treeID`); falls back to the pointwise npz."""
    import laspy

    lazs = sorted(results_dir.rglob("full_forest/*.la[sz]")) \
        or sorted(results_dir.rglob("*.la[sz]"))
    for p in lazs:
        las = laspy.read(str(p))
        names = [d.name for d in las.point_format.dimensions]
        if "treeID" in names:
            xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
            return xyz, np.asarray(las["treeID"], dtype=np.int64)

    npzs = sorted(results_dir.rglob("*pointwise*.npz")) or sorted(results_dir.rglob("*.npz"))
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        if "instance_preds" in d:
            key = "coords" if "coords" in d else ("xyz" if "xyz" in d else None)
            xyz = d[key].astype(np.float64) if key else None
            return xyz, np.asarray(d["instance_preds"], dtype=np.int64)
    raise SystemExit(f"[treelearn] no readable result under {results_dir}")


def main():
    exp = os.environ.get("EXPERIMENT", "raw")
    if exp == "raw":
        forest, img, title = C.CHUNK_RAW_LAZ, C.IMG_EXP1, "exp1 — TreeLearn on raw chunk"
    elif exp == "masked":
        forest, img, title = (C.CHUNK_MASKED_LAZ, C.IMG_EXP3,
                              "exp3 — TreeLearn on Sonata vegetation mask")
    else:
        raise SystemExit(f"[treelearn] unknown EXPERIMENT={exp}")

    if not forest.exists():
        raise SystemExit(f"[treelearn] input missing: {forest} (run prep/sonata first)")

    results_dir = C.DATA / "treelearn" / exp
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # TreeLearn tiles land in a shared dir (/data/tiles) and its dataset loads
    # EVERY file there (os.listdir, no plot filter) — a previous run's tiles for
    # a different chunk would join this inference and then blow up the
    # hash-based back-propagation to original points (KeyError). Tiles are
    # regenerated every run (tile_generation: True), so wiping loses nothing.
    tiles_dir = C.DATA / "tiles"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)

    ckpt = ensure_weights()
    cfg = build_config(forest, ckpt, results_dir)
    print(f"[treelearn] running pipeline on {forest.name} -> {results_dir}")
    r = subprocess.run([sys.executable, "tools/pipeline/pipeline.py",
                        "--config", str(cfg)], cwd=str(TL_REPO))
    if r.returncode != 0:
        raise SystemExit(f"[treelearn] pipeline exited {r.returncode}")

    report_filter_diagnostics(results_dir)
    xyz, inst = read_instances(results_dir)
    n = len(set(inst.tolist()) - {0, -1})
    print(f"[treelearn] {exp}: {n} tree instances")
    viz.plot_instances(xyz, inst, img, f"{title}  |  {n} trees", unassigned=(-1, 0))
    print(f"[treelearn] image -> {img}")


if __name__ == "__main__":
    main()
