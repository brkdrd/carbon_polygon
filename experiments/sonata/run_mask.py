#!/usr/bin/env python3
"""Sonata stage, part B — vegetation mask on the shared chunk.

Loads the canonical chunk, density-matches it to the WHU training regime, runs
the frozen Sonata encoder + trained vegetation head over overlapping sub-chunks,
and writes:
  * chunk_masked_local.laz / .npz  — vegetation-only points (input for exp3/exp4)
  * 03_sonata_vegetation_mask.png  — the mask visualization

All coordinates stay local (the chunk is already centered); make_point re-centers
in float64 regardless, so the float32 trap cannot reappear.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/app")
from common import config as C
from common import geo
from common import viz
import sonata_lib as SL
from train_head import LinearHead, reference_density


def density_match(tile_xyz, tile_int, target_density):
    """Match campus density to the head's training regime so the encoder sees a
    familiar sampling. `target_density` is the common density the head was
    trained at (carried in the checkpoint) — see train_head.build_tiles."""
    camp_d = geo.occupied_density(tile_xyz)
    grid = C.GRID_SIZE * np.sqrt(max(camp_d, 1) / max(target_density, 1))
    print(f"[sonata] density target {target_density:.0f} vs campus {camp_d:.0f} "
          f"pts/m^2 -> grid {grid:.3f} m")
    xyz_m, (int_m,) = geo.grid_subsample(tile_xyz, [tile_int], grid)
    print(f"[sonata] {len(tile_xyz):,} -> {len(xyz_m):,} pts "
          f"(matched {geo.occupied_density(xyz_m):.0f} pts/m^2)")
    return xyz_m, int_m


def chunk_bboxes(xyz, size=25.0, ov=3.0):
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    x1, y1 = xyz[:, 0].max(), xyz[:, 1].max()
    step, out, x = size - ov, [], x0
    while x < x1:
        y = y0
        while y < y1:
            out.append((x, min(x + size, x1 + 1e-6), y, min(y + size, y1 + 1e-6)))
            y += step
        x += step
    return out


def main():
    if not C.CHUNK_RAW_NPZ.exists():
        raise SystemExit("[sonata] chunk not prepared — run the prep stage first")
    if not C.SONATA_HEAD.exists():
        raise SystemExit("[sonata] vegetation head missing — run train_head.py first")

    d = np.load(C.CHUNK_RAW_NPZ)
    tile_xyz, tile_int = d["xyz"].astype(np.float64), d["intensity"].astype(np.float32)
    print(f"[sonata] chunk {tile_xyz.shape} loaded")

    model, transform = SL.load_model_and_transform(C.GRID_SIZE)
    ckpt = torch.load(str(C.SONATA_HEAD), map_location="cuda")
    head = LinearHead(ckpt["feat_dim"]).cuda()
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    print(f"[sonata] head loaded (val veg IoU {ckpt.get('iou', float('nan')):.4f} "
          f"| sources {ckpt.get('sources', ['whu'])})")

    # Match the campus to the exact density the head was trained at. Newer heads
    # carry it in the checkpoint; older ones fall back to the WHU reference.
    target_density = ckpt.get("target_density") or reference_density()
    tile_xyz, tile_int = density_match(tile_xyz, tile_int, target_density)

    # inference with OOM-safe recursive splitting (notebook infer_box logic)
    lsum = np.zeros((len(tile_xyz), 2), np.float32)
    lcnt = np.zeros(len(tile_xyz), np.float32)

    def infer_box(idx, depth=0):
        if len(idx) < 500:
            return
        try:
            feat = SL.sonata_feat(model, transform,
                                  SL.make_point(tile_xyz[idx], tile_int[idx])).cuda()
            with torch.no_grad():
                lg = head(feat).cpu().numpy()
            del feat
            lsum[idx] += lg
            lcnt[idx] += 1
            return
        except torch.cuda.OutOfMemoryError:
            SL.gpu_free()
            if depth >= 6:
                print(f"   [sonata] giving up on {len(idx):,} pts (OOM)")
                return
            pts = tile_xyz[idx]
            ax = 0 if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 1]) else 1
            mid = np.median(pts[:, ax])
            left, right = idx[pts[:, ax] < mid], idx[pts[:, ax] >= mid]
            if len(left) == 0 or len(right) == 0:
                return
            infer_box(left, depth + 1)
            infer_box(right, depth + 1)

    from tqdm.auto import tqdm
    boxes = chunk_bboxes(tile_xyz)
    print(f"[sonata] {len(boxes)} sub-chunks")
    for bx0, bx1, by0, by1 in tqdm(boxes):
        m = ((tile_xyz[:, 0] >= bx0) & (tile_xyz[:, 0] < bx1) &
             (tile_xyz[:, 1] >= by0) & (tile_xyz[:, 1] < by1))
        infer_box(np.where(m)[0])

    cov = lcnt > 0
    pred = np.zeros(len(tile_xyz), np.int64)
    pred[cov] = lsum[cov].argmax(1)
    veg = pred == 1
    print(f"[sonata] covered {cov.mean()*100:.1f}% | vegetation {veg.mean()*100:.2f}%")

    # ---- write masked chunk (vegetation only) for exp3/exp4 ----
    C.CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    veg_xyz, veg_int = tile_xyz[veg], tile_int[veg]
    geo.write_las(C.CHUNK_MASKED_LAZ, veg_xyz, intensity=veg_int)
    np.savez_compressed(C.CHUNK_MASKED_NPZ,
                        xyz=veg_xyz.astype(np.float64),
                        intensity=veg_int.astype(np.float32),
                        full_xyz=tile_xyz.astype(np.float64),
                        veg_mask=veg)
    print(f"[sonata] wrote {C.CHUNK_MASKED_LAZ.name} ({veg.sum():,} veg pts)")

    viz.plot_semantic_mask(
        tile_xyz, veg, C.IMG_SONATA_MASK,
        "Sonata vegetation mask (trees + bushes vs everything)")
    print(f"[sonata] mask image -> {C.IMG_SONATA_MASK}")


if __name__ == "__main__":
    main()
