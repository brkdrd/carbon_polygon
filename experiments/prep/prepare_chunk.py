#!/usr/bin/env python3
"""Stage 0 — build the ONE shared chunk that all four experiments consume.

Steps:
  1. Download the private Kaggle LiDAR dataset (needs kaggle.json / env creds).
  2. Cut a fixed square chunk (center + size from common.config) out of the LAZ.
  3. Voxel-grid subsample to a sane density.
  4. Center to a LOCAL frame in float64 (FLOAT32_COORDINATE_BUG.md fix), record
     the origin, then write local coords. Downstream tools therefore never see
     UTM-magnitude coordinates and cannot trip the float32 quantization trap.
  5. Write chunk_local.laz / .npz / chunk_meta.json and a preview PNG.

Idempotent: if the canonical chunk already exists it is reused, so re-running the
four experiments always hits the identical chunk.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")  # common/ is mounted/copied here
from common import config as C
from common import geo
from common import viz


# ---------------------------------------------------------------------------
# 1. Kaggle download
# ---------------------------------------------------------------------------
def ensure_source_laz() -> Path:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = C.RAW_DIR / C.SOURCE_LAZ_NAME
    if target.exists():
        print(f"[prep] source LAZ already present: {target}")
        return target

    # locate any pre-mounted copy first (lets users bypass Kaggle entirely)
    for cand in C.RAW_DIR.rglob("*.laz"):
        print(f"[prep] using pre-mounted LAZ: {cand}")
        return cand

    print(f"[prep] downloading Kaggle dataset {C.KAGGLE_DATASET} ...")
    # kaggle reads creds from KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json
    # (KAGGLE_CONFIG_DIR is set to /root/.kaggle by the Dockerfile).
    cmd = ["kaggle", "datasets", "download", "-d", C.KAGGLE_DATASET,
           "-p", str(C.RAW_DIR), "--unzip"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(
            "[prep] Kaggle download failed. Provide credentials (mount "
            "kaggle.json to /root/.kaggle or set KAGGLE_USERNAME/KAGGLE_KEY), "
            "or drop the .laz into experiments/data/raw/ manually.")

    hits = sorted(C.RAW_DIR.rglob("*.laz")) + sorted(C.RAW_DIR.rglob("*.las"))
    if not hits:
        raise SystemExit("[prep] no .laz/.las found after Kaggle download")
    # prefer the configured name if present
    for h in hits:
        if h.name == C.SOURCE_LAZ_NAME:
            return h
    print(f"[prep] configured name not found; using {hits[0].name}")
    return hits[0]


# ---------------------------------------------------------------------------
# 2. + 3. cut the chunk and subsample
# ---------------------------------------------------------------------------
def extract_chunk(path: Path):
    import laspy

    with laspy.open(str(path)) as f:
        h = f.header
        mins, maxs = np.asarray(h.mins, np.float64), np.asarray(h.maxs, np.float64)
        n_pts = h.point_count
        has_i = "intensity" in [d.name for d in h.point_format.dimensions]
    print(f"[prep] source: {n_pts:,} pts | extent "
          f"{np.round((maxs - mins)[:2], 1)} m | intensity={has_i}")

    center = C.CHUNK_CENTER or ((mins[:2] + maxs[:2]) / 2.0)
    center = np.asarray(center, np.float64)
    half = C.CHUNK_SIZE / 2.0
    x0, x1 = center[0] - half, center[0] + half
    y0, y1 = center[1] - half, center[1] + half
    print(f"[prep] chunk center {np.round(center, 2)} | size {C.CHUNK_SIZE} m | "
          f"bbox x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]")

    xs, ys, zs, its, kept = [], [], [], [], 0
    with laspy.open(str(path)) as f:
        for pts in f.chunk_iterator(8_000_000):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            m = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
            if not m.any():
                continue
            xs.append(X[m]); ys.append(Y[m]); zs.append(np.asarray(pts.z, np.float64)[m])
            its.append(np.asarray(pts.intensity, np.float32)[m] if has_i
                       else np.zeros(int(m.sum()), np.float32))
            kept += int(m.sum())
    if not kept:
        raise SystemExit("[prep] empty chunk — adjust CHUNK_CENTER / CHUNK_SIZE")

    xyz = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
    inten = np.concatenate(its)
    print(f"[prep] kept {kept:,} pts in chunk")

    # This is the exact magnitude regime the bug doc warns about — audit it.
    geo.warn_if_quantizing(xyz, feature_size_m=C.GRID_SIZE)

    xyz, (inten,) = geo.grid_subsample(xyz, [inten], C.GRID_SIZE)
    print(f"[prep] after {C.GRID_SIZE} m grid: {len(xyz):,} pts "
          f"| density {geo.occupied_density(xyz):.0f} pts/m^2")
    return xyz, inten, center, (x0, x1, y0, y1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    C.CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)

    if C.CHUNK_RAW_NPZ.exists() and C.CHUNK_META.exists() and not os.environ.get("FORCE_PREP"):
        print(f"[prep] canonical chunk already exists at {C.CHUNK_RAW_NPZ} — reuse. "
              "Set FORCE_PREP=1 to rebuild.")
        return

    src = ensure_source_laz()
    xyz_utm, inten, center, bbox = extract_chunk(src)

    # ---- THE FIX: center in float64, cast float32 last -------------------
    xyz_local, origin = geo.center_float64(xyz_utm)  # origin = centroid, float64
    print(f"[prep] local frame origin (UTM): {np.round(origin, 3).tolist()}")
    print(f"[prep] local coord range: {np.round(xyz_local.min(0), 2).tolist()} .. "
          f"{np.round(xyz_local.max(0), 2).tolist()}")

    # write canonical artifacts (local coords everywhere)
    geo.write_las(C.CHUNK_RAW_LAZ, xyz_local, intensity=inten)
    np.savez_compressed(C.CHUNK_RAW_NPZ,
                        xyz=xyz_local.astype(np.float64),
                        intensity=inten.astype(np.float32))
    geo.save_meta(
        C.CHUNK_META,
        source_laz=str(src),
        kaggle_dataset=C.KAGGLE_DATASET,
        chunk_center_utm=center,
        chunk_size_m=C.CHUNK_SIZE,
        bbox_utm=list(bbox),
        local_origin_utm=origin,          # add back to recover UTM
        crs_note="source ~UTM 52N (easting ~7.36e5, northing ~4.77e6)",
        grid_size_m=C.GRID_SIZE,
        n_points=int(len(xyz_local)),
        density_pts_per_m2=geo.occupied_density(xyz_local),
        coords_frame="local (origin subtracted in float64; safe to cast float32)",
    )
    print(f"[prep] wrote {C.CHUNK_RAW_LAZ.name}, {C.CHUNK_RAW_NPZ.name}, {C.CHUNK_META.name}")

    viz.plot_semantic_mask(
        xyz_local, np.zeros(len(xyz_local), bool),
        C.IMG_RAW_CHUNK,
        f"Shared chunk (raw) — {len(xyz_local):,} pts, {C.CHUNK_SIZE:.0f} m tile")
    print(f"[prep] preview -> {C.IMG_RAW_CHUNK}")
    print("[prep] done.")


if __name__ == "__main__":
    main()
