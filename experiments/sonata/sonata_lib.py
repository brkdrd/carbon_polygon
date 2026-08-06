"""Sonata (frozen PTv3) feature extraction, shared by head-training and masking.

Ported from the reference notebook. The one line that matters for correctness is
in `make_point`: coordinates are centered in float64 and cast to float32 LAST
(FLOAT32_COORDINATE_BUG.md). Inputs from the prep stage are already local, but we
re-center here defensively so this function is safe on any input.
"""
from __future__ import annotations

import os

# JIT guards must be set before cumm/spconv import (see notebook cells 1/2).
os.environ.setdefault("CUMM_DISABLE_JIT", "1")
os.environ.setdefault("SPCONV_DISABLE_JIT", "1")

import gc
import numpy as np
import torch


def gpu_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_normals(xyz, radius=0.5, max_nn=30):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.normalize_normals()
    return np.asarray(pcd.normals, dtype=np.float32)


def make_point(xyz, inten=None, segment=None, use_intensity=False):
    """Build a Sonata input dict. THE FIX lives here: center in float64, cast
    float32 last. `estimate_normals` runs on the centered float64 coords."""
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz - xyz.mean(0, keepdims=True)      # centered while still float64
    normal = estimate_normals(xyz)               # normals on clean geometry
    xyz32 = xyz.astype(np.float32)               # cast the SMALL centered values
    if use_intensity and inten is not None:
        color = np.repeat((inten.astype(np.float32) / 65535.0)[:, None], 3, axis=1) * 255.0
    else:
        color = np.full((len(xyz32), 3), 127.5, dtype=np.float32)
    d = dict(coord=xyz32, color=color.astype(np.float32), normal=normal)
    if segment is not None:
        d["segment"] = np.asarray(segment, dtype=np.int64)
    return d


def load_model_and_transform(grid_size):
    import sonata
    custom_config = dict(enc_patch_size=[1024] * 5, enable_flash=False)
    model = sonata.load("sonata", repo_id="facebook/sonata",
                        custom_config=custom_config).cuda()
    model.eval()
    transform = sonata.transform.Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"),
             feat_keys=("coord", "color", "normal")),
    ])
    return model, transform


@torch.no_grad()
def sonata_feat(model, transform, pd):
    """Encoder ladder -> per-point features via successive un-pooling (upcast)."""
    point = transform(dict(pd))
    for k in list(point.keys()):
        if isinstance(point[k], torch.Tensor):
            point[k] = point[k].cuda(non_blocking=True)
    point = model(point)
    for _ in range(2):
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = point.feat[inv]
        point = parent
    return point.feat[point.inverse].float().cpu()


def feat_recursive(model, transform, xyz, inten, lab, depth=0):
    """Extract features; on CUDA OOM split the tile along its longer axis and
    recurse (notebook's try/except splitter)."""
    try:
        f = sonata_feat(model, transform, make_point(xyz, inten, segment=lab)).numpy()
        return [(f, lab)]
    except torch.cuda.OutOfMemoryError:
        gpu_free()
        if depth >= 6 or len(xyz) < 2000:
            print(f"   [sonata] dropped {len(xyz):,} pts (OOM)")
            return []
        ax = 0 if np.ptp(xyz[:, 0]) >= np.ptp(xyz[:, 1]) else 1
        mid = np.median(xyz[:, ax])
        m = xyz[:, ax] < mid
        if m.all() or (~m).all():
            return []
        return (feat_recursive(model, transform, xyz[m], inten[m], lab[m], depth + 1)
                + feat_recursive(model, transform, xyz[~m], inten[~m], lab[~m], depth + 1))
