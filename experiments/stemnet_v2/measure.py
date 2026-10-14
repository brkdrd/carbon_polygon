#!/usr/bin/env python3
"""StemNet v2, step 0: measure the raw cloud and the stem map. Nothing is
modelled, filtered or thrown away here — this stage only counts, so every later
prep decision (grid, crop size, slice height) rests on a number instead of a
guess.

Two passes over the LAZ:

  pass 1  streams the file once. Header facts, attribute statistics, a 1 m
          plan-view point-count map, and a tile cache: every point's raw LAS
          integers appended to one file per V2_TILE-metre tile.
  pass 2  reads the cache one tile at a time. Exact duplicate count, exact
          voxel counts at each V2_GRIDS size, point spacing from local
          nearest neighbours, and the points around every surveyed tree.

Exact, not sampled: voxels are computed in the file's own integer units, and
tiles are aligned to the voxel grid, so per-tile unique counts add up to the
global count. If a grid size is not a whole number of LAS units the report says
`exact: false` for it.

The stem map is used as-is. Every tree row is kept and reported; nothing is
merged and nothing is deleted. The six `base_N` rows are the RTK base
stations, not trees, and are reported separately.

Outputs
  /results/stemnet_v2_measure.json    every number
  /results/stemnet_v2_measure.txt     the same, readable
  /results/stemnet_v2_trees.csv       one row per surveyed tree
  /results/40_stemnet_v2_density.png  points/m2 map with the stem map on top
  /results/41_stemnet_v2_hist.png     spacing, crop sizes, per-tree counts
  /data/stemnet_v2/cache/             the tile cache (reused by later stages;
                                      V2_FORCE=1 rebuilds it)
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
DATA = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", "/results"))
OUT = DATA / "stemnet_v2"
CACHE = OUT / "cache"
STEM_MAP = Path(os.environ.get("V2_STEM_MAP", "/app/poderevka.csv"))
ATTRS = Path(os.environ.get("V2_ATTRS", "/app/poderevka_attributes.csv"))
LAZ_NAME = os.environ.get(
    "SOURCE_LAZ_NAME", "point_cloud_campus_2025-03-14-12-59-27_result.laz")
TILE = float(os.environ.get("V2_TILE", "25.0"))
GRIDS = [float(g) for g in os.environ.get("V2_GRIDS", "0.02,0.05,0.1").split(",")]
CHUNK = int(float(os.environ.get("V2_CHUNK", "5000000")))
CROPS = [float(c) for c in os.environ.get("V2_CROPS", "6,8,10,12,15,20,25").split(",")]
SPACING_BOXES = int(os.environ.get("V2_SPACING_BOXES", "8"))   # per tile
FORCE = os.environ.get("V2_FORCE", "0") != "0"
SEED = int(os.environ.get("V2_SEED", "0"))
TREE_R = (0.5, 1.0, 2.0)          # plan-view radii counted around each tree
SLICE = (0.3, 1.5)                # metres above the tree's ground, circle-fit slice


def log(msg):
    print(f"[v2-measure] {msg}", flush=True)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def find_laz() -> Path:
    raw = DATA / "raw"
    named = raw / LAZ_NAME
    if named.exists():
        return named
    hits = sorted(raw.rglob("*.laz")) + sorted(raw.rglob("*.las"))
    if not hits:
        raise SystemExit(f"[v2-measure] no .laz/.las under {raw} — put the "
                         f"campus scan there (the old bw_prep downloads it)")
    return hits[0]


@dataclass
class StemMap:
    block: np.ndarray            # which base_N block a tree row sits under
    n: np.ndarray                # tree number as written (restarts per block)
    lat: np.ndarray
    lon: np.ndarray
    z: np.ndarray
    stations: list = field(default_factory=list)   # (name, lat, lon, z)


def parse_stem_map(path: Path) -> StemMap:
    """Every tree row, in file order. `base_N` rows open a block and are RTK
    base stations; they are kept aside, not dropped."""
    block, n, lat, lon, z, stations = [], [], [], [], [], []
    cur = 0
    with open(path) as f:
        for raw in f:
            p = [s.strip() for s in raw.split(",")]
            if len(p) < 4 or not p[0]:
                continue
            if p[0].startswith("base"):
                cur = int(p[0].split("_")[1]) if "_" in p[0] else cur + 1
                stations.append((p[0], float(p[1]), float(p[2]), float(p[3])))
                continue
            block.append(cur); n.append(p[0])
            lat.append(float(p[1])); lon.append(float(p[2])); z.append(float(p[3]))
    return StemMap(np.array(block), np.array(n), np.array(lat), np.array(lon),
                   np.array(z), stations)


def girth_rows(path: Path) -> dict:
    out = dict(rows=0, with_girth=0, multi_stem=0)
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out["rows"] += 1
            g = (r.get("girth_cm") or "").strip()
            if g:
                out["with_girth"] += 1
                if ";" in g:
                    out["multi_stem"] += 1
    return out


# the integer frame lives in core.py, shared with every later v2 stage
from core import Frame  # noqa: E402


# ---------------------------------------------------------------------------
# pass 1: stream the file
# ---------------------------------------------------------------------------
def header_facts(laz_path: Path, hdr) -> dict:
    pf = hdr.point_format
    crs, epsg = None, None
    try:
        c = hdr.parse_crs()
        if c is not None:
            crs = c.name
            epsg = c.to_epsg()
    except Exception as e:                     # noqa: BLE001 - reported, not fatal
        crs = f"unparseable: {type(e).__name__}"
    mx = np.max(np.abs(np.r_[hdr.mins, hdr.maxs]))
    return dict(
        file=str(laz_path), file_bytes=laz_path.stat().st_size,
        version=f"{hdr.version.major}.{hdr.version.minor}",
        point_format=int(pf.id), dimensions=list(pf.dimension_names),
        point_count=int(hdr.point_count),
        scales=list(map(float, hdr.scales)), offsets=list(map(float, hdr.offsets)),
        mins=list(map(float, hdr.mins)), maxs=list(map(float, hdr.maxs)),
        extent_m=list(map(float, np.asarray(hdr.maxs) - np.asarray(hdr.mins))),
        crs=crs, epsg=epsg,
        system_identifier=str(getattr(hdr, "system_identifier", "")).strip("\x00 "),
        generating_software=str(getattr(hdr, "generating_software", "")).strip("\x00 "),
        creation_date=str(getattr(hdr, "creation_date", "")),
        vlrs=[str(v.description) for v in hdr.vlrs],
        float32_ulp_at_max_coord_m=float(np.spacing(np.float32(mx))),
    )


class Pass1:
    """Everything that can be counted while streaming."""

    def __init__(self, fr: Frame, hdr, dims):
        self.fr = fr
        self.dims = set(dims)
        imax = np.round((np.asarray(hdr.maxs) - fr.o) / fr.s).astype(np.int64)
        nx = int(fr.index([imax[0]], 0, 1.0)[0]) + 1
        ny = int(fr.index([imax[1]], 1, 1.0)[0]) + 1
        self.occ = np.zeros((ny, nx), np.int64)          # points per 1 m column
        self.zbins = np.arange(np.floor(hdr.mins[2]) - 1, np.ceil(hdr.maxs[2]) + 1.5, 0.5)
        self.zhist = np.zeros(len(self.zbins) - 1, np.int64)
        self.n = 0
        self.intensity = np.zeros(65536, np.int64) if "intensity" in self.dims else None
        self.cls = np.zeros(256, np.int64) if "classification" in self.dims else None
        self.ret = np.zeros(16, np.int64) if "return_number" in self.dims else None
        self.nret = np.zeros(16, np.int64) if "number_of_returns" in self.dims else None
        self.rgb_nonzero = 0
        self.rgb_max = 0
        self.gps = [np.inf, -np.inf]
        self.tiles: dict[tuple, int] = {}

    def add(self, pts, X, Y, Z, writer=None):
        fr = self.fr
        self.n += len(X)
        ix = fr.index(X, 0, 1.0); iy = fr.index(Y, 1, 1.0)
        ok = (ix >= 0) & (iy >= 0) & (ix < self.occ.shape[1]) & (iy < self.occ.shape[0])
        np.add.at(self.occ, (iy[ok], ix[ok]), 1)
        zm = Z * fr.s[2] + fr.o[2]
        self.zhist += np.histogram(zm, self.zbins)[0]
        if self.intensity is not None:
            self.intensity += np.bincount(np.asarray(pts["intensity"], np.int64),
                                          minlength=65536)[:65536]
        if self.cls is not None:
            self.cls += np.bincount(np.asarray(pts["classification"], np.int64),
                                    minlength=256)[:256]
        if self.ret is not None:
            self.ret += np.bincount(np.clip(np.asarray(pts["return_number"], np.int64), 0, 15),
                                    minlength=16)
        if self.nret is not None:
            self.nret += np.bincount(np.clip(np.asarray(pts["number_of_returns"], np.int64), 0, 15),
                                     minlength=16)
        if "red" in self.dims:
            r = np.asarray(pts["red"]); g = np.asarray(pts["green"]); b = np.asarray(pts["blue"])
            self.rgb_nonzero += int(((r > 0) | (g > 0) | (b > 0)).sum())
            self.rgb_max = max(self.rgb_max, int(max(r.max(), g.max(), b.max())))
        if "gps_time" in self.dims:
            t = np.asarray(pts["gps_time"])
            self.gps = [min(self.gps[0], float(t.min())), max(self.gps[1], float(t.max()))]

        tx = fr.index(X, 0, TILE); ty = fr.index(Y, 1, TILE)
        key = tx * 100000 + ty
        order = np.argsort(key, kind="stable")
        ks = key[order]
        cuts = np.flatnonzero(np.r_[True, np.diff(ks) != 0, True])
        P = np.column_stack([X, Y, Z]).astype(np.int32)[order]
        for a, b in zip(cuts[:-1], cuts[1:]):
            k = int(ks[a]); t = (k // 100000, k % 100000)
            self.tiles[t] = self.tiles.get(t, 0) + int(b - a)
            if writer is not None:
                writer(t, P[a:b])

    def facts(self):
        out = dict(points_streamed=int(self.n))
        occ = self.occ[self.occ > 0]
        out["plan_view_1m"] = dict(
            cells_total=int(self.occ.size), cells_with_points=int(occ.size),
            area_with_points_ha=float(occ.size / 1e4),
            points_per_m2_p10=float(np.percentile(occ, 10)) if occ.size else 0.0,
            points_per_m2_p50=float(np.percentile(occ, 50)) if occ.size else 0.0,
            points_per_m2_p90=float(np.percentile(occ, 90)) if occ.size else 0.0,
            points_per_m2_p99=float(np.percentile(occ, 99)) if occ.size else 0.0,
            points_per_m2_max=int(occ.max()) if occ.size else 0)
        out["z_m"] = dict(p01=_hq(self.zbins, self.zhist, 0.01), p50=_hq(self.zbins, self.zhist, 0.5),
                          p99=_hq(self.zbins, self.zhist, 0.99))
        if self.intensity is not None:
            out["intensity"] = dict(nonzero=int(self.intensity[1:].sum()),
                                    p01=_bq(self.intensity, 0.01), p50=_bq(self.intensity, 0.5),
                                    p99=_bq(self.intensity, 0.99), max=int(np.flatnonzero(self.intensity).max())
                                    if self.intensity.any() else 0)
        if self.cls is not None:
            out["classification"] = {str(i): int(v) for i, v in enumerate(self.cls) if v}
        if self.ret is not None:
            out["return_number"] = {str(i): int(v) for i, v in enumerate(self.ret) if v}
        if self.nret is not None:
            out["number_of_returns"] = {str(i): int(v) for i, v in enumerate(self.nret) if v}
        if "red" in self.dims:
            out["rgb"] = dict(points_with_colour=int(self.rgb_nonzero), max_value=int(self.rgb_max))
        if "gps_time" in self.dims and np.isfinite(self.gps[0]):
            out["gps_time"] = dict(min=self.gps[0], max=self.gps[1], span_s=self.gps[1] - self.gps[0])
        tc = np.array(list(self.tiles.values()))
        out["tiles"] = dict(size_m=TILE, count=int(len(tc)),
                            points_p50=int(np.median(tc)) if len(tc) else 0,
                            points_max=int(tc.max()) if len(tc) else 0)
        return out


def _hq(bins, hist, q):
    c = np.cumsum(hist)
    if not len(c) or c[-1] == 0:
        return float("nan")
    i = int(np.searchsorted(c, q * c[-1]))
    return float(bins[min(i + 1, len(bins) - 1)])


def _bq(counts, q):
    c = np.cumsum(counts)
    return int(np.searchsorted(c, q * c[-1])) if c[-1] else 0


class TileCache:
    """One raw int32 (X, Y, Z) file per tile, appended chunk by chunk."""

    def __init__(self, root: Path):
        self.root = root

    def path(self, t):
        return self.root / f"tile_{t[0]}_{t[1]}.bin"

    def write(self, t, P):
        with open(self.path(t), "ab") as f:
            f.write(np.ascontiguousarray(P, np.int32).tobytes())

    def read(self, t):
        return np.fromfile(self.path(t), dtype=np.int32).reshape(-1, 3)

    def tiles(self):
        out = []
        for q in sorted(self.root.glob("tile_*.bin")):
            a, b = q.stem.split("_")[1:]
            out.append((int(a), int(b)))
        return out


def cache_marker(laz: Path):
    st = laz.stat()
    return dict(file=str(laz), bytes=st.st_size, mtime=int(st.st_mtime), tile_m=TILE)


# ---------------------------------------------------------------------------
# pass 2: one tile at a time from the cache
# ---------------------------------------------------------------------------
class Trees:
    """Points around every surveyed tree, gathered as each tile is read.

    A tree near a tile edge needs points from up to four tiles; it is finished
    (and its points freed) when the last of them has been read."""

    def __init__(self, fr: Frame, x, y, z):
        self.fr, self.x, self.y, self.z = fr, x, y, z
        R = max(TREE_R)
        self.need: dict[tuple, list] = {}
        self.left = np.zeros(len(x), np.int64)
        for i in range(len(x)):
            ts = set()
            for dx in (-R, R):
                for dy in (-R, R):
                    I = (x[i] + dx - fr.o[0]) / fr.s[0]
                    J = (y[i] + dy - fr.o[1]) / fr.s[1]
                    ts.add((int(fr.index([np.floor(I)], 0, TILE)[0]),
                            int(fr.index([np.floor(J)], 1, TILE)[0])))
            for t in ts:
                self.need.setdefault(t, []).append(i)
            self.left[i] = len(ts)
        self.count = {r: np.zeros(len(x), np.int64) for r in TREE_R}
        self.buf: dict[int, list] = {}
        self.stats: dict[int, dict] = {}

    def visit(self, t, xm, ym, zm):
        idx = self.need.get(t)
        if not idx:
            return
        order = np.argsort(xm, kind="stable")
        xs = xm[order]
        R = max(TREE_R)
        for i in idx:
            lo, hi = np.searchsorted(xs, [self.x[i] - R, self.x[i] + R])
            if hi > lo:
                c = order[lo:hi]
                dx = xm[c] - self.x[i]; dy = ym[c] - self.y[i]
                d2 = dx * dx + dy * dy
                for r in TREE_R:
                    self.count[r][i] += int((d2 <= r * r).sum())
                k = c[d2 <= 1.0]
                if len(k):
                    self.buf.setdefault(i, []).append(
                        np.column_stack([xm[k], ym[k], zm[k], d2[d2 <= 1.0]]))
            self.left[i] -= 1
            if self.left[i] == 0:
                self._finish(i)

    def missing_tile(self, t):
        """A tile the tree needed has no points at all (outside the scan)."""
        for i in self.need.get(t, []):
            self.left[i] -= 1
            if self.left[i] == 0:
                self._finish(i)

    def _finish(self, i):
        parts = self.buf.pop(i, [])
        P = np.concatenate(parts) if parts else np.zeros((0, 4))
        d2 = P[:, 3]
        z05 = np.sort(P[d2 <= 0.25, 2])
        low3 = float(z05[2]) if len(z05) >= 3 else (float(z05[0]) if len(z05) else np.nan)
        gz = float(self.z[i])
        sl_gt = P[(P[:, 2] >= gz + SLICE[0]) & (P[:, 2] <= gz + SLICE[1])]
        sl_low = (P[(P[:, 2] >= low3 + SLICE[0]) & (P[:, 2] <= low3 + SLICE[1])]
                  if np.isfinite(low3) else P[:0])
        out = dict(low3=low3, n_slice_gt=int(len(sl_gt)), n_slice_low=int(len(sl_low)))
        for g in GRIDS:
            v = np.floor(sl_gt[:, :3] / g).astype(np.int64)
            out[f"n_slice_gt_g{g:g}"] = int(len(np.unique(v, axis=0))) if len(v) else 0
        self.stats[i] = out


def spacing_boxes(rng, P_m, n_boxes, half=0.5, min_pts=10):
    """Nearest-neighbour distance inside small cubes centred on random points.

    Local, not global: nearest neighbours over a random subsample of the whole
    cloud would measure the subsample's spacing, not the scanner's."""
    from scipy.spatial import cKDTree
    meds, pool, dup = [], [], 0
    if len(P_m) < min_pts:
        return meds, pool, dup
    for c in rng.choice(len(P_m), size=min(n_boxes, len(P_m)), replace=False):
        m = np.all(np.abs(P_m - P_m[c]) <= half, axis=1)
        B = P_m[m]
        if len(B) < min_pts:
            continue
        d, _ = cKDTree(B).query(B, k=2)
        nn = d[:, 1]
        dup += int((nn == 0).sum())
        nn = nn[nn > 0]
        if len(nn):
            meds.append(float(np.median(nn)))
            pool.append(rng.choice(nn, size=min(2000, len(nn)), replace=False))
    return meds, pool, dup


def pass2(cache: TileCache, fr: Frame, trees: Trees, occ_shape, rng):
    tiles = cache.tiles()
    exact = {g: fr.exact(g) for g in GRIDS}
    vox = {g: 0 for g in GRIDS}
    col = {g: np.zeros(occ_shape, np.int64) for g in GRIDS}
    n_pts = n_unique = 0
    meds, pool, zero_nn = [], [], 0
    seen = set()
    t0 = time.time()
    for k, t in enumerate(tiles):
        P = cache.read(t)
        seen.add(t)
        n_pts += len(P)
        # exact duplicates, at the file's own resolution
        D = P.astype(np.int64) - P.min(0)
        span = D.max(0) + 1
        key = D[:, 0] + span[0] * (D[:, 1] + span[1] * D[:, 2])
        n_unique += int(len(np.unique(key)))
        for g in GRIDS:
            vx = fr.index(P[:, 0], 0, g); vy = fr.index(P[:, 1], 1, g)
            vz = fr.index(P[:, 2], 2, g)
            V = np.column_stack([vx, vy, vz]); V0 = V.min(0); V = V - V0
            sp = V.max(0) + 1
            u = np.unique(V[:, 0] + sp[0] * (V[:, 1] + sp[1] * V[:, 2]))
            vox[g] += int(len(u))
            ux = u % sp[0] + V0[0]
            uy = (u // sp[0]) % sp[1] + V0[1]
            per_m = int(round(1.0 / g))
            cx = ux // per_m; cy = uy // per_m
            ok = (cx >= 0) & (cy >= 0) & (cx < occ_shape[1]) & (cy < occ_shape[0])
            np.add.at(col[g], (cy[ok], cx[ok]), 1)
        Pm = fr.metres(P)
        m, p, z = spacing_boxes(rng, Pm, SPACING_BOXES)
        meds += m; pool += p; zero_nn += z
        trees.visit(t, Pm[:, 0], Pm[:, 1], Pm[:, 2])
        if (k + 1) % 10 == 0 or k + 1 == len(tiles):
            el = time.time() - t0
            log(f"pass 2: tile {k + 1}/{len(tiles)} | {n_pts:,} pts | "
                f"{el / 60:.1f} min, eta {el / (k + 1) * (len(tiles) - k - 1) / 60:.1f} min")
    for t in list(trees.need):
        if t not in seen:
            trees.missing_tile(t)
    pool = np.concatenate(pool) if pool else np.zeros(0)
    return dict(n_pts=n_pts, n_unique=n_unique, vox=vox, exact=exact, col=col,
                spacing_box_medians=np.array(meds), spacing_pool=pool,
                zero_nn_in_boxes=zero_nn)


def crop_table(col, sizes):
    """Points a square crop of each size would hold, over every 1 m position
    where the crop contains any points. Box sums from an integral image."""
    I = np.zeros((col.shape[0] + 1, col.shape[1] + 1), np.int64)
    I[1:, 1:] = col.cumsum(0).cumsum(1)
    rows = []
    for L in sizes:
        k = int(round(L))
        if k > min(col.shape):
            continue
        S = I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]
        S = S[S > 0]
        if not len(S):
            continue
        rows.append(dict(crop_m=L, positions=int(len(S)),
                         p50=int(np.percentile(S, 50)), p90=int(np.percentile(S, 90)),
                         p99=int(np.percentile(S, 99)), max=int(S.max())))
    return rows


# ---------------------------------------------------------------------------
# stem map in the cloud's frame
# ---------------------------------------------------------------------------
def to_cloud_crs(sm: StemMap, epsg_hdr):
    from pyproj import Transformer
    if epsg_hdr:
        epsg, src = int(epsg_hdr), "LAZ header"
    else:
        zone = int(np.floor((np.median(sm.lon) + 180) / 6) + 1)
        epsg = (32600 if np.median(sm.lat) >= 0 else 32700) + zone
        src = f"no CRS in the LAZ header; UTM zone {zone} from the trees' longitude"
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tr.transform(sm.lon, sm.lat)
    return np.asarray(x), np.asarray(y), epsg, src


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def q(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def figures(occ, trees_xy, inside, fr, sp_pool, crops, slice_counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    show = np.where(occ > 0, np.log10(np.maximum(occ, 1)), np.nan)
    ext = [fr.origin_m[0], fr.origin_m[0] + occ.shape[1],
           fr.origin_m[1], fr.origin_m[1] + occ.shape[0]]
    im = ax.imshow(show, origin="lower", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=ax, label="log10 points per m2 (raw)")
    ax.scatter(trees_xy[inside, 0], trees_xy[inside, 1], s=3, c="red", lw=0,
               label=f"tree inside scan ({int(inside.sum())})")
    if (~inside).any():
        ax.scatter(trees_xy[~inside, 0], trees_xy[~inside, 1], s=6, c="white",
                   edgecolors="k", lw=0.3, label=f"tree with no points within 2 m ({int((~inside).sum())})")
    ax.set_aspect("equal"); ax.legend(loc="upper right"); ax.set_title("raw cloud density + stem map")
    fig.tight_layout(); fig.savefig(RESULTS / "40_stemnet_v2_density.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    if len(sp_pool):
        ax[0].hist(np.clip(sp_pool * 100, 0, 10), bins=100)
    ax[0].set_xlabel("nearest-neighbour distance (cm, clipped at 10)"); ax[0].set_title("raw point spacing")
    for g, rows in crops.items():
        if rows:
            ax[1].plot([r["crop_m"] for r in rows], [r["p90"] for r in rows], "o-", label=f"{g}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("crop edge (m)"); ax[1].set_ylabel("points in crop, p90")
    ax[1].legend(title="grid"); ax[1].set_title("crop size vs points")
    ax[2].hist(np.clip(slice_counts, 0, 2000), bins=80)
    ax[2].set_xlabel(f"points within 1 m, {SLICE[0]}-{SLICE[1]} m above the tree's own z (clipped 2000)")
    ax[2].set_title("per-tree points available for a circle fit")
    fig.tight_layout(); fig.savefig(RESULTS / "41_stemnet_v2_hist.png", dpi=120); plt.close(fig)


def write_trees_csv(sm, x, y, tr: Trees):
    cols = ["block", "n", "lat", "lon", "x", "y", "z"] + [f"n_r{r:g}" for r in TREE_R] + \
           ["low3_z", "z_minus_low3", "n_slice_gt", "n_slice_low"] + [f"n_slice_gt_g{g:g}" for g in GRIDS]
    with open(RESULTS / "stemnet_v2_trees.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(x)):
            s = tr.stats.get(i, {})
            lo = s.get("low3", np.nan)
            w.writerow([int(sm.block[i]), sm.n[i], f"{sm.lat[i]:.9f}", f"{sm.lon[i]:.9f}",
                        f"{x[i]:.3f}", f"{y[i]:.3f}", f"{sm.z[i]:.3f}"]
                       + [int(tr.count[r][i]) for r in TREE_R]
                       + ["" if not np.isfinite(lo) else f"{lo:.3f}",
                          "" if not np.isfinite(lo) else f"{sm.z[i] - lo:.3f}",
                          s.get("n_slice_gt", 0), s.get("n_slice_low", 0)]
                       + [s.get(f"n_slice_gt_g{g:g}", 0) for g in GRIDS])


def text_report(R):
    h, p1, p2, gt = R["header"], R["pass1"], R["pass2"], R["stem_map"]
    L = []
    a = L.append
    a("=" * 72); a("StemNet v2 — raw cloud and stem map, measured"); a("=" * 72)
    a(f"file           {h['file']}  ({h['file_bytes'] / 2**30:.2f} GB)")
    a(f"LAS            v{h['version']}, point format {h['point_format']}")
    a(f"dimensions     {', '.join(h['dimensions'])}")
    a(f"points         {h['point_count']:,} in header | {p1['points_streamed']:,} streamed")
    a(f"scale          {h['scales']}  (smallest representable step, m)")
    a(f"extent         {h['extent_m'][0]:.1f} x {h['extent_m'][1]:.1f} m, z {h['mins'][2]:.2f} .. {h['maxs'][2]:.2f} m")
    a(f"CRS            {h['crs']} (EPSG {h['epsg']})")
    a(f"float32 ulp    {h['float32_ulp_at_max_coord_m'] * 1000:.3f} mm at the largest raw coordinate")
    a("")
    a("-- density --")
    pv = p1["plan_view_1m"]
    a(f"scanned area   {pv['area_with_points_ha']:.2f} ha of 1 m cells with any point")
    a(f"points / m2    p10 {pv['points_per_m2_p10']:,.0f}  p50 {pv['points_per_m2_p50']:,.0f}  "
      f"p90 {pv['points_per_m2_p90']:,.0f}  p99 {pv['points_per_m2_p99']:,.0f}  max {pv['points_per_m2_max']:,}")
    if p2:
        sp = p2["spacing_cm"]
        a(f"NN spacing     p10 {sp.get('p10', float('nan')):.2f}  p50 {sp.get('p50', float('nan')):.2f}  "
          f"p90 {sp.get('p90', float('nan')):.2f} cm   (local nearest neighbour, {p2['spacing_boxes']} boxes)")
        a(f"duplicates     {p2['exact_duplicates']:,} points share an exact position with another "
          f"({p2['exact_duplicates'] / max(p2['points'], 1) * 100:.2f}%)")
        a("")
        a("-- after voxel grids (exact unless marked) --")
        for g, v in p2["voxels"].items():
            a(f"grid {g:>5} m   {v['points']:>14,} pts  ({v['fraction_kept'] * 100:5.1f}% kept)"
              + ("" if v["exact"] else "  [NOT exact: grid is not a whole number of LAS units]"))
        a("")
        a("-- points in a square crop, over every 1 m position --")
        for g, rows in p2["crops"].items():
            a(f"grid {g}:")
            for r in rows:
                a(f"   {r['crop_m']:>4.0f} m  p50 {r['p50']:>12,}  p90 {r['p90']:>12,}  "
                  f"p99 {r['p99']:>12,}  max {r['max']:>12,}")
    a("")
    a("-- stem map --")
    a(f"tree rows      {gt['tree_rows']:,} in {gt['blocks']} blocks (+ {len(gt['base_stations'])} base-station rows, not trees)")
    a(f"in the cloud   {gt['crs_used']} — {gt['crs_source']}")
    a(f"inside bbox    {gt['inside_bbox']:,} | with any point within 2 m: {gt['with_points_2m']:,} "
      f"| none within 2 m: {gt['tree_rows'] - gt['with_points_2m']:,}")
    a(f"attributes     {gt['attributes']['rows']} rows, {gt['attributes']['with_girth']} with girth, "
      f"{gt['attributes']['multi_stem']} multi-stem")
    for r in TREE_R:
        a(f"pts within {r:g} m  " + "  ".join(f"{k} {v:,.0f}" for k, v in gt[f"pts_within_{r:g}m"].items()))
    a(f"survey z - cloud's 3rd-lowest z within 0.5 m  "
      + "  ".join(f"{k} {v:+.3f}" for k, v in gt["z_minus_low3"].items()) + " m")
    a("   (~0 => survey z and scan z share a vertical frame and the slice can use the GT z directly)")
    a(f"circle-fit slice, {SLICE[0]}-{SLICE[1]} m above the GT z, within 1 m:  "
      + "  ".join(f"{k} {v:,.0f}" for k, v in gt["slice_points_gt_z"].items()))
    for g in GRIDS:
        a(f"   same slice at grid {g:g} m:  "
          + "  ".join(f"{k} {v:,.0f}" for k, v in gt[f"slice_points_gt_z_g{g:g}"].items()))
    a(f"   trees with < 20 slice points: {gt['slice_lt20']:,}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main():
    import laspy
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    laz = find_laz()
    log(f"source: {laz}")

    with laspy.open(str(laz)) as f:
        hdr = f.header
        H = header_facts(laz, hdr)
        fr = Frame(hdr.scales, hdr.offsets, hdr.mins)
        log(f"header: {H['point_count']:,} pts, format {H['point_format']}, "
            f"scale {H['scales']}, CRS {H['crs']}")

        marker = cache_marker(laz)
        mpath = CACHE / "complete.json"
        reuse = (not FORCE and mpath.exists()
                 and json.loads(mpath.read_text()).get("marker") == marker)
        need = H["point_count"] * 12
        free = shutil.disk_usage(OUT).free
        write_cache = True
        if not reuse and need > 0.8 * free:
            write_cache = False
            log(f"!! tile cache needs {need / 2**30:.1f} GB, only {free / 2**30:.1f} GB "
                f"free — pass 2 (exact voxel counts, spacing, per-tree points) skipped")
        p1 = Pass1(fr, hdr, H["dimensions"])
        if reuse:
            log("tile cache from a previous run matches this file — pass 1 counts only")
        elif write_cache:
            if CACHE.exists():
                shutil.rmtree(CACHE)
            CACHE.mkdir(parents=True)
        tc = TileCache(CACHE)
        writer = tc.write if (write_cache and not reuse) else None
        t0 = time.time()
        for k, pts in enumerate(f.chunk_iterator(CHUNK)):
            X = np.asarray(pts.X, np.int64); Y = np.asarray(pts.Y, np.int64)
            Z = np.asarray(pts.Z, np.int64)
            p1.add(pts, X, Y, Z, writer)
            el = time.time() - t0
            log(f"pass 1: {p1.n:,}/{H['point_count']:,} | {el / 60:.1f} min, "
                f"eta {el / p1.n * (H['point_count'] - p1.n) / 60:.1f} min")
        if writer is not None:
            mpath.write_text(json.dumps(dict(marker=marker)))
    P1 = p1.facts()

    sm = parse_stem_map(STEM_MAP)
    tx_, ty_, epsg, src = to_cloud_crs(sm, H["epsg"])
    trees = Trees(fr, tx_, ty_, sm.z)
    inside = ((tx_ >= H["mins"][0]) & (tx_ <= H["maxs"][0])
              & (ty_ >= H["mins"][1]) & (ty_ <= H["maxs"][1]))

    P2 = {}
    if write_cache or reuse:
        r = pass2(tc, fr, trees, p1.occ.shape, rng)
        crops = {f"{g:g}": crop_table(r["col"][g], CROPS) for g in GRIDS}
        crops["raw"] = crop_table(p1.occ, CROPS)
        P2 = dict(
            points=r["n_pts"], exact_duplicates=r["n_pts"] - r["n_unique"],
            spacing_cm={k: v * 100 for k, v in q(r["spacing_pool"], (10, 25, 50, 75, 90)).items()},
            spacing_box_median_cm={k: v * 100 for k, v in q(r["spacing_box_medians"], (10, 50, 90)).items()},
            spacing_boxes=int(len(r["spacing_box_medians"])),
            zero_distance_in_boxes=int(r["zero_nn_in_boxes"]),
            voxels={f"{g:g}": dict(points=r["vox"][g], fraction_kept=r["vox"][g] / max(r["n_pts"], 1),
                                    exact=bool(r["exact"][g])) for g in GRIDS},
            crops=crops)
    else:
        r = dict(spacing_pool=np.zeros(0))
        crops = {}

    st = trees.stats
    low3 = np.array([st.get(i, {}).get("low3", np.nan) for i in range(len(tx_))])
    sl = np.array([st.get(i, {}).get("n_slice_gt", 0) for i in range(len(tx_))])
    has = trees.count[2.0] > 0
    GT = dict(
        tree_rows=int(len(sm.n)), blocks=int(len(set(sm.block.tolist()))),
        base_stations=[dict(name=s[0], lat=s[1], lon=s[2], z=s[3]) for s in sm.stations],
        crs_used=f"EPSG:{epsg}", crs_source=src,
        inside_bbox=int(inside.sum()), with_points_2m=int(has.sum()),
        attributes=girth_rows(ATTRS),
        z_minus_low3=q((sm.z - low3)[has]),
        slice_points_gt_z=q(sl[has]),
        slice_lt20=int(((sl < 20) & has).sum()),
    )
    for rr in TREE_R:
        GT[f"pts_within_{rr:g}m"] = q(trees.count[rr][has])
    for g in GRIDS:
        GT[f"slice_points_gt_z_g{g:g}"] = q(
            np.array([st.get(i, {}).get(f"n_slice_gt_g{g:g}", 0) for i in range(len(tx_))])[has])

    R = dict(header=H, pass1=P1, pass2=P2, stem_map=GT,
             config=dict(tile_m=TILE, grids=GRIDS, crops=CROPS, slice_m=SLICE,
                         tree_radii_m=TREE_R, spacing_boxes_per_tile=SPACING_BOXES))
    (RESULTS / "stemnet_v2_measure.json").write_text(json.dumps(R, indent=2, default=float))
    txt = text_report(R)
    (RESULTS / "stemnet_v2_measure.txt").write_text(txt + "\n")
    write_trees_csv(sm, tx_, ty_, trees)
    figures(p1.occ, np.column_stack([tx_, ty_]), has, fr, r["spacing_pool"], crops, sl[has])
    print(txt, flush=True)
    log("-> results/stemnet_v2_measure.txt | .json | stemnet_v2_trees.csv | 40_/41_ png")


if __name__ == "__main__":
    main()
