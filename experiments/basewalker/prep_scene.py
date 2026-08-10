#!/usr/bin/env python3
"""BaseWalker stage 0 — build the training scene from the full campus cloud.

Artifacts (all coordinates LOCAL to one scene origin, centered in float64,
FLOAT32_COORDINATE_BUG.md discipline):
  /data/basewalker/scene_meta.json   origin, EPSG, bbox, counts
  /data/basewalker/tiles/*.npz       xyz float32 + normals float16, 30 m tiles
  /data/basewalker/dem.npz           ground DEM (CSF cloth, percentile fallback)
  /data/basewalker/labels.npz        tree bases: local xyz, fork-merged, val flag
  /data/basewalker/seeds.npz         candidate start points on the DEM grid

Idempotent: skips whatever already exists (FORCE_BW_PREP=1 rebuilds all).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from common import config as C
from common import geo

BW = C.DATA / "basewalker"
TILES = BW / "tiles"
SCENE_META = BW / "scene_meta.json"
DEM_NPZ = BW / "dem.npz"
LABELS_NPZ = BW / "labels.npz"
SEEDS_NPZ = BW / "seeds.npz"

GRID = C.GRID_SIZE                                   # 0.05 m base subsample
TILE = float(os.environ.get("BW_TILE", "30.0"))      # tile edge, m
HALO = 2.0                                           # normals halo, m
UTM_EPSG = int(os.environ.get("BW_UTM_EPSG", "32652"))
MERGE_R = float(os.environ.get("BW_MERGE_R", "0.3"))   # fork-merge radius, m
SEED_GRID = float(os.environ.get("BW_SEED_GRID", "2.0"))
DEM_CELL = float(os.environ.get("BW_DEM_CELL", "0.5"))
VAL_QUANT = float(os.environ.get("BW_VAL_QUANT", "0.8"))  # x-quantile: val block
VAL_MARGIN = float(os.environ.get("BW_VAL_MARGIN", "10.0"))  # leakage buffer, m
LABEL_COVER_R = 2.0  # label needs cloud points within this xy radius to be usable


def ensure_source_laz() -> Path:
    """Same contract as prep: reuse anything in /data/raw, else Kaggle."""
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = C.RAW_DIR / C.SOURCE_LAZ_NAME
    if target.exists():
        return target
    for cand in sorted(C.RAW_DIR.rglob("*.laz")) + sorted(C.RAW_DIR.rglob("*.las")):
        print(f"[bw-prep] using pre-mounted cloud: {cand}")
        return cand
    print(f"[bw-prep] downloading Kaggle dataset {C.KAGGLE_DATASET} ...")
    r = subprocess.run(["kaggle", "datasets", "download", "-d", C.KAGGLE_DATASET,
                        "-p", str(C.RAW_DIR), "--unzip"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("[bw-prep] Kaggle download failed and no .laz in /data/raw")
    hits = sorted(C.RAW_DIR.rglob("*.laz")) + sorted(C.RAW_DIR.rglob("*.las"))
    if not hits:
        raise SystemExit("[bw-prep] no .laz/.las after download")
    return next((h for h in hits if h.name == C.SOURCE_LAZ_NAME), hits[0])


def load_full_cloud(path: Path):
    """Stream the whole cloud, grid-subsampled to GRID. Returns float64 UTM."""
    import laspy
    parts = []
    with laspy.open(str(path)) as f:
        print(f"[bw-prep] source: {f.header.point_count:,} pts")
        for pts in f.chunk_iterator(8_000_000):
            xyz = np.column_stack([np.asarray(pts.x, np.float64),
                                   np.asarray(pts.y, np.float64),
                                   np.asarray(pts.z, np.float64)])
            xyz, _ = geo.grid_subsample(xyz, [], GRID)
            parts.append(xyz)
    xyz = np.concatenate(parts)
    del parts
    xyz, _ = geo.grid_subsample(xyz, [], GRID)  # dedup chunk boundaries
    print(f"[bw-prep] {len(xyz):,} pts after {GRID} m grid")
    return xyz


def build_dem(xyz_local):
    """Ground DEM on a DEM_CELL grid. CSF cloth if importable, else per-cell
    5th-percentile z. Holes filled by iterative 3x3 nan-mean dilation."""
    sub, _ = geo.grid_subsample(xyz_local, [], 0.2)
    ground = None
    method = os.environ.get("BW_DEM_METHOD", "csf")
    if method == "csf":
        try:
            import CSF
            csf = CSF.CSF()
            csf.params.bSloopSmooth = True
            csf.params.cloth_resolution = 0.5
            csf.params.rigidness = 2
            csf.setPointCloud(sub.astype(np.float64))
            gi, ngi = CSF.VecInt(), CSF.VecInt()
            csf.do_filtering(gi, ngi)
            ground = sub[np.asarray(gi, dtype=np.int64)]
            print(f"[bw-prep] CSF ground: {len(ground):,}/{len(sub):,} pts")
        except Exception as e:  # noqa: BLE001 — any CSF failure falls back
            print(f"[bw-prep] CSF unavailable ({e!r}) — percentile fallback")
    if ground is None:
        method = "percentile"

    x0, y0 = xyz_local[:, 0].min(), xyz_local[:, 1].min()
    nx = int(np.ceil((xyz_local[:, 0].max() - x0) / DEM_CELL)) + 1
    ny = int(np.ceil((xyz_local[:, 1].max() - y0) / DEM_CELL)) + 1
    dem = np.full((ny, nx), np.nan, np.float32)

    src = ground if ground is not None else sub
    ix = ((src[:, 0] - x0) / DEM_CELL).astype(np.int64).clip(0, nx - 1)
    iy = ((src[:, 1] - y0) / DEM_CELL).astype(np.int64).clip(0, ny - 1)
    cell = iy * nx + ix
    order = np.argsort(cell, kind="stable")
    cell_s, z_s = cell[order], src[order, 2]
    starts = np.flatnonzero(np.r_[True, np.diff(cell_s) > 0])
    ends = np.r_[starts[1:], len(cell_s)]
    for s, e in zip(starts, ends):
        c = cell_s[s]
        zs = z_s[s:e]
        dem[c // nx, c % nx] = (np.median(zs) if ground is not None
                                else np.percentile(zs, 5))
    valid = ~np.isnan(dem)
    print(f"[bw-prep] DEM ({method}): {nx}x{ny} cells, {valid.mean()*100:.1f}% valid")

    # fill holes by dilated nan-mean so every cell is defined (walkers project
    # here); keep the original validity mask for seeding.
    filled = dem.copy()
    for _ in range(200):
        nan = np.isnan(filled)
        if not nan.any():
            break
        p = np.pad(filled, 1, constant_values=np.nan)
        stack = np.stack([p[dy:dy + ny, dx:dx + nx]
                          for dy in (0, 1, 2) for dx in (0, 1, 2)])
        with np.errstate(invalid="ignore"):
            m = np.nanmean(stack, 0)
        filled[nan] = m[nan]
    filled = np.nan_to_num(filled, nan=float(np.nanmedian(dem)))
    return dict(z=filled.astype(np.float32), valid=valid,
                x0=float(x0), y0=float(y0), cell=float(DEM_CELL), method=method)


def dem_lookup(dem, xy):
    ix = ((xy[:, 0] - dem["x0"]) / dem["cell"]).astype(np.int64).clip(0, dem["z"].shape[1] - 1)
    iy = ((xy[:, 1] - dem["y0"]) / dem["cell"]).astype(np.int64).clip(0, dem["z"].shape[0] - 1)
    return dem["z"][iy, ix]


def prep_labels(origin, bbox, dem, xyz_local):
    """CSV (WGS84) -> local frame; drop uncovered; merge forks; z from DEM;
    spatial train/val split on x."""
    from pyproj import Transformer
    rows = []
    with open("/app/poderevka.csv") as f:
        for line in f:
            p = [s.strip() for s in line.split(",")]
            if len(p) < 4 or p[0].startswith("base"):
                continue
            rows.append([float(p[1]), float(p[2])])
    lat, lon = np.array(rows).T
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    ex, ny_ = tr.transform(lon, lat)                       # float64 UTM
    xy = np.column_stack([ex, ny_]) - origin[None, :2]     # float64 local
    print(f"[bw-prep] labels: {len(xy)} parsed | scene bbox x[{bbox[0]:.0f},{bbox[1]:.0f}] "
          f"y[{bbox[2]:.0f},{bbox[3]:.0f}]")

    inside = ((xy[:, 0] > bbox[0]) & (xy[:, 0] < bbox[1]) &
              (xy[:, 1] > bbox[2]) & (xy[:, 1] < bbox[3]))
    xy = xy[inside]
    print(f"[bw-prep] labels inside scene bbox: {len(xy)}")

    # coverage: require cloud points near the base (the scanner did not cover
    # the whole campus; unscanned labels would poison the loss as unreachable)
    from scipy.spatial import cKDTree
    kd = cKDTree(xyz_local[:, :2])
    n_near = kd.query_ball_point(xy, r=LABEL_COVER_R, return_length=True)
    covered = n_near >= 50
    xy = xy[covered]
    print(f"[bw-prep] labels with cloud coverage (>=50 pts within "
          f"{LABEL_COVER_R} m): {len(xy)}")

    # fork merge: greedy union of pairs closer than MERGE_R -> centroids
    kd = cKDTree(xy)
    pairs = kd.query_pairs(MERGE_R)
    parent = np.arange(len(xy))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(len(xy))])
    merged = np.stack([xy[roots == r0].mean(0) for r0 in np.unique(roots)])
    print(f"[bw-prep] after fork-merge (<{MERGE_R} m): {len(merged)} bases")

    z = dem_lookup(dem, merged)
    bases = np.column_stack([merged, z])

    # spatial split: rightmost x-block = validation, with a dead margin band
    xq = np.quantile(bases[:, 0], VAL_QUANT)
    is_val = bases[:, 0] > xq + VAL_MARGIN / 2
    in_margin = np.abs(bases[:, 0] - xq) <= VAL_MARGIN / 2
    print(f"[bw-prep] split at x={xq:.1f}: train {int((~is_val & ~in_margin).sum())} "
          f"| val {int(is_val.sum())} | margin (unused) {int(in_margin.sum())}")
    return bases.astype(np.float64), is_val, in_margin, float(xq)


def write_tiles(xyz_local):
    """30 m tiles with open3d normals (computed with a 2 m halo, halo dropped)."""
    import sonata_lib as SL
    TILES.mkdir(parents=True, exist_ok=True)
    x0, y0 = xyz_local[:, 0].min(), xyz_local[:, 1].min()
    tx = ((xyz_local[:, 0] - x0) / TILE).astype(np.int64)
    ty = ((xyz_local[:, 1] - y0) / TILE).astype(np.int64)
    n_written = 0
    for t in np.unique(tx * 100000 + ty):
        i, j = t // 100000, t % 100000
        out = TILES / f"tile_{i}_{j}.npz"
        if out.exists():
            n_written += 1
            continue
        cx0, cy0 = x0 + i * TILE, y0 + j * TILE
        m_halo = ((xyz_local[:, 0] >= cx0 - HALO) & (xyz_local[:, 0] < cx0 + TILE + HALO) &
                  (xyz_local[:, 1] >= cy0 - HALO) & (xyz_local[:, 1] < cy0 + TILE + HALO))
        pts = xyz_local[m_halo]
        if len(pts) < 100:
            continue
        normals = SL.estimate_normals(pts)
        inner = ((pts[:, 0] >= cx0) & (pts[:, 0] < cx0 + TILE) &
                 (pts[:, 1] >= cy0) & (pts[:, 1] < cy0 + TILE))
        np.savez_compressed(out,
                            xyz=pts[inner].astype(np.float32),
                            normal=normals[inner].astype(np.float16))
        n_written += 1
    print(f"[bw-prep] {n_written} tiles in {TILES}")


def main():
    BW.mkdir(parents=True, exist_ok=True)
    force = bool(os.environ.get("FORCE_BW_PREP"))
    if SCENE_META.exists() and DEM_NPZ.exists() and LABELS_NPZ.exists() \
            and SEEDS_NPZ.exists() and not force:
        print("[bw-prep] artifacts exist — reuse (FORCE_BW_PREP=1 rebuilds)")
        return

    src = ensure_source_laz()
    xyz_utm = load_full_cloud(src)
    geo.warn_if_quantizing(xyz_utm, feature_size_m=GRID)
    xyz_local, origin = geo.center_float64(xyz_utm)
    del xyz_utm
    xyz_local32 = xyz_local.astype(np.float32)
    bbox = (float(xyz_local[:, 0].min()), float(xyz_local[:, 0].max()),
            float(xyz_local[:, 1].min()), float(xyz_local[:, 1].max()))

    dem = build_dem(xyz_local)
    np.savez_compressed(DEM_NPZ, **{k: v for k, v in dem.items() if k != "method"},
                        method=np.array(dem["method"]))

    bases, is_val, in_margin, xq = prep_labels(origin, bbox, dem, xyz_local)
    np.savez_compressed(LABELS_NPZ, bases=bases, is_val=is_val,
                        in_margin=in_margin, split_x=np.array(xq))

    # seeds: DEM grid over VALID (observed-ground) cells
    zv, valid = dem["z"], dem["valid"]
    step = max(1, int(round(SEED_GRID / dem["cell"])))
    iy, ix = np.nonzero(valid)
    keep = (iy % step == 0) & (ix % step == 0)
    sx = dem["x0"] + ix[keep] * dem["cell"]
    sy = dem["y0"] + iy[keep] * dem["cell"]
    sz = zv[iy[keep], ix[keep]]
    seeds = np.column_stack([sx, sy, sz]).astype(np.float64)
    seed_is_val = seeds[:, 0] > xq + VAL_MARGIN / 2
    seed_in_margin = np.abs(seeds[:, 0] - xq) <= VAL_MARGIN / 2
    np.savez_compressed(SEEDS_NPZ, seeds=seeds, is_val=seed_is_val,
                        in_margin=seed_in_margin)
    print(f"[bw-prep] {len(seeds):,} seeds ({SEED_GRID} m grid) | "
          f"val {int(seed_is_val.sum()):,}")

    write_tiles(xyz_local32)

    with open(SCENE_META, "w") as f:
        json.dump(dict(origin_utm=list(map(float, origin)), utm_epsg=UTM_EPSG,
                       bbox_local=list(bbox), n_points=int(len(xyz_local)),
                       grid_m=GRID, tile_m=TILE, dem_method=dem["method"],
                       n_bases=int(len(bases)), merge_r=MERGE_R,
                       seed_grid_m=SEED_GRID, split_x=xq), f, indent=2)
    print("[bw-prep] done.")


if __name__ == "__main__":
    main()
