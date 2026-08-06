"""Single source of truth for paths and the shared chunk definition.

The whole point of the study is that all four experiments run on *the same chunk*.
That is enforced here: `prepare_chunk.py` writes the canonical chunk files listed
below once, and every downstream stage reads them — nobody re-extracts. The chunk
geometry (center + size) is fixed and reproducible, overridable only via env.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- mounted volumes (same in every container) ----------------------------
DATA = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", "/results"))
MODELS = Path(os.environ.get("MODELS_DIR", str(DATA / "models")))

CHUNK_DIR = DATA / "chunk"
RAW_DIR = DATA / "raw"           # downloaded source point cloud + WHU scenes
CACHE_DIR = DATA / "cache"

# ---- Kaggle source dataset (private) --------------------------------------
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", "samsuperman12/my-private-lidar-dataset")
# filename of the .laz inside that dataset (matches the notebook)
SOURCE_LAZ_NAME = os.environ.get(
    "SOURCE_LAZ_NAME",
    "point_cloud_campus_2025-03-14-12-59-27_result.laz",
)

# ---- the shared chunk ------------------------------------------------------
# Center: if unset, prepare_chunk uses the source cloud's centroid (deterministic).
# Size: a single square tile in metres. 40 m is a sensible forest-plot size for
# TreeLearn / SegmentAnyTree; the notebook used 100 m. Override via CHUNK_SIZE.
CHUNK_SIZE = float(os.environ.get("CHUNK_SIZE", "40.0"))
_cx = os.environ.get("CHUNK_CENTER_X")
_cy = os.environ.get("CHUNK_CENTER_Y")
CHUNK_CENTER = (float(_cx), float(_cy)) if (_cx and _cy) else None

# base voxel grid used when reading the WHU training scenes (metres)
GRID_SIZE = float(os.environ.get("GRID_SIZE", "0.05"))

# ---- canonical chunk artifacts (written by prep, read by everyone) --------
# All coordinates in these files are LOCAL (origin subtracted in float64), so no
# downstream tool can trip the float32 quantization trap. The origin is in meta.
CHUNK_RAW_LAZ = CHUNK_DIR / "chunk_local.laz"          # exp1 (TreeLearn) + exp2 (SAT) input
CHUNK_RAW_NPZ = CHUNK_DIR / "chunk_local.npz"          # xyz(float64 local) + intensity
CHUNK_MASKED_LAZ = CHUNK_DIR / "chunk_masked_local.laz"  # exp3 + exp4 input (veg only)
CHUNK_MASKED_NPZ = CHUNK_DIR / "chunk_masked.npz"
CHUNK_META = CHUNK_DIR / "chunk_meta.json"             # origin, bbox, CRS, density

# ---- result images ---------------------------------------------------------
IMG_RAW_CHUNK = RESULTS / "00_chunk_raw.png"
IMG_SONATA_MASK = RESULTS / "03_sonata_vegetation_mask.png"
IMG_EXP1 = RESULTS / "exp1_treelearn_raw.png"
IMG_EXP2 = RESULTS / "exp2_segmentanytree_raw.png"
IMG_EXP3 = RESULTS / "exp3_treelearn_on_sonata_mask.png"
IMG_EXP4 = RESULTS / "exp4_segmentanytree_on_sonata_mask.png"

# ---- Sonata vegetation head ------------------------------------------------
SONATA_HEAD = MODELS / "sonata_vegetation_head.pth"
# WHU-STree source used to train the head (same Drive folder as the notebook)
WHU_DRIVE_URL = os.environ.get(
    "WHU_DRIVE_URL",
    "https://drive.google.com/drive/folders/18braC_G3RAi5fn8TZNRe-Vt8q1bYRZqk",
)
WHU_CITY = os.environ.get("WHU_CITY", "whu-stree-nj")
WHU_MAX_PLY = int(os.environ.get("WHU_MAX_PLY", "2"))
