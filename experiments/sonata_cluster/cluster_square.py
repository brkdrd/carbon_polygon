#!/usr/bin/env python3
"""Unsupervised segmentation of one random square of the campus scan.

1. Pick a random square that the scan actually covers. Point density comes
   from a few thousand seeks into the .laz, not a full read.
2. Read every point inside that square in one streaming pass over the file.
3. Run frozen, pretrained Sonata to get an embedding for every point.
4. Run k-means in that embedding space (cosine / spherical).
5. Write the square back out as a .laz in the scan's own coordinates. Every
   point is coloured by its cluster, and the cluster id is stored as an extra
   dimension. A preview PNG and a JSON with the run's numbers sit next to it.

Settings (environment):
  SC_SOURCE   path to the .laz      (default: the campus scan under DATA_DIR/raw)
  SC_SQUARE   square side, m        (15)
  SC_CENTER   "x,y" in scan coords  (unset = random)
  SC_SEED     int                   (unset = fresh; printed so a run can be repeated)
  SC_K        number of clusters    (8)
  SC_GRID     Sonata grid, m        (0.02)
  SC_TILE     Sonata pass size, m   (10; passes that OOM split themselves)
  SC_MARGIN   context around a pass (2)
  SC_COVER    min fraction of the square the scan must cover (0.5)

Coordinates stay float64 until each Sonata pass has been centred; only then
are they cast to float32.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import laspy
import numpy as np

DATA = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", "/results"))
SOURCE_NAME = "point_cloud_campus_2025-03-14-12-59-27_result.laz"


def log(msg: str) -> None:
    print(f"[cluster] {msg}", flush=True)


def settings() -> dict:
    env = os.environ.get
    seed = env("SC_SEED")
    center = env("SC_CENTER")
    return dict(
        source=env("SC_SOURCE"),
        square=float(env("SC_SQUARE", "15")),
        center=tuple(float(v) for v in center.split(",")) if center else None,
        seed=int(seed) if seed else int.from_bytes(os.urandom(4), "little"),
        k=int(env("SC_K", "8")),
        grid=float(env("SC_GRID", "0.02")),
        tile=float(env("SC_TILE", "10")),
        margin=float(env("SC_MARGIN", "2")),
        cover=float(env("SC_COVER", "0.5")),
    )


# ---- 1. where the scan is ----------------------------------------------------

def find_source(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    raw = DATA / "raw"
    named = sorted(raw.rglob(SOURCE_NAME))
    if named:
        return named[0]
    hits = sorted(raw.rglob("*.laz")) + sorted(raw.rglob("*.las"))
    if not hits:
        raise SystemExit(f"no .laz/.las under {raw}. Put the campus scan there "
                         "(./run_experiments.sh prep downloads it from Kaggle).")
    return hits[0]


def sample_xy(path: Path, rng: np.random.Generator,
              seeks: int = 4000, per_seek: int = 500) -> np.ndarray:
    """XY of short runs of points read at random offsets: a cheap density map."""
    with laspy.open(path) as r:
        n = r.header.point_count
        if n <= seeks * per_seek:
            p = r.read_points(n)
            return np.c_[np.asarray(p.x), np.asarray(p.y)]
        out = []
        for s in np.sort(rng.integers(0, n - per_seek, seeks)):
            r.seek(int(s))
            p = r.read_points(per_seek)
            out.append(np.c_[np.asarray(p.x), np.asarray(p.y)])
    return np.concatenate(out)


def pick_square(xy: np.ndarray, size: float, rng: np.random.Generator,
                min_cover: float, cell: float = 2.0, tries: int = 500):
    """Corner (x0, y0) of a size x size square centred on a sampled point; the
    first one whose 2 m cells are at least `min_cover` occupied, else the best."""
    lo = xy.min(0)
    ij = np.floor((xy - lo) / cell).astype(np.int64)
    occ = np.zeros(ij.max(0) + 1, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    n = max(int(round(size / cell)), 1)
    best, best_cover = None, -1.0
    for _ in range(tries):
        c = xy[rng.integers(len(xy))]
        x0, y0 = c - size / 2
        i0, j0 = np.floor((np.array([x0, y0]) - lo) / cell).astype(np.int64)
        win = occ[max(i0, 0):max(i0 + n, 0), max(j0, 0):max(j0 + n, 0)]
        cover = win.sum() / (n * n)
        if cover > best_cover:
            best, best_cover = (float(x0), float(y0)), float(cover)
        if cover >= min_cover:
            break
    return best, best_cover


# ---- 2. the points of the square -------------------------------------------

def read_square(path: Path, x0: float, y0: float, size: float):
    """Every point with x0 <= x < x0+size, y0 <= y < y0+size, as the file's own
    point records (all dimensions kept, stored integers copied untouched). The
    test uses laspy's own float64 x/y, so a point on the edge falls the same
    side as it would for anyone reading the file."""
    backend = (laspy.LazBackend.LazrsParallel
               if laspy.LazBackend.LazrsParallel.is_available() else None)
    with laspy.open(path, laz_backend=backend) as r:
        h = r.header
        keep, seen, t = [], 0, time.time()
        for pts in r.chunk_iterator(5_000_000):
            x, y = np.asarray(pts.x), np.asarray(pts.y)
            m = (x >= x0) & (x < x0 + size) & (y >= y0) & (y < y0 + size)
            a = pts.array
            if m.any():
                keep.append(a[m].copy())
            seen += len(a)
            if (seen // 5_000_000) % 40 == 0:
                log(f"  read {seen / 1e6:,.0f}M / {h.point_count / 1e6:,.0f}M points "
                    f"({time.time() - t:.0f} s), kept {sum(map(len, keep)):,}")
        if not keep:
            raise SystemExit("the square holds no points")
        rec = laspy.PackedPointRecord(np.concatenate(keep), h.point_format)
    las = laspy.LasData(h, points=rec)
    return las


def colour_of(las) -> np.ndarray:
    """0..255 RGB if the scan carries colour, else zeros."""
    if "red" not in las.point_format.dimension_names:
        return np.zeros((len(las.points), 3), np.float32)
    rgb = np.c_[las.red, las.green, las.blue].astype(np.float32)
    if rgb.max() > 255:
        rgb /= 257.0
    return rgb


# ---- 3. per-point embeddings ------------------------------------------------

def voxelize(xyz: np.ndarray, grid: float):
    """One point per grid cell (mean position). Returns the cell index of every
    input point (`inverse`) and the cell count."""
    k = np.floor((xyz - xyz.min(0)) / grid).astype(np.int64)
    dims = k.max(0) + 1
    key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inverse = np.unique(key, return_inverse=True)
    return inverse.ravel(), int(inverse.max()) + 1


def cell_mean(values: np.ndarray, inverse: np.ndarray, n: int) -> np.ndarray:
    cnt = np.bincount(inverse, minlength=n).astype(np.float64)
    return np.stack([np.bincount(inverse, values[:, d], minlength=n) / cnt
                     for d in range(values.shape[1])], 1)


def normals(xyz: np.ndarray, k: int = 16, batch: int = 500_000) -> np.ndarray:
    """PCA normals from k nearest neighbours, flipped to point up."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    out = np.empty_like(xyz, dtype=np.float32)
    for s in range(0, len(xyz), batch):
        _, idx = tree.query(xyz[s:s + batch], k=min(k, len(xyz)), workers=-1)
        nb = xyz[idx]
        nb = nb - nb.mean(1, keepdims=True)
        _, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
        n = vec[:, :, 0]
        n[n[:, 2] < 0] *= -1
        out[s:s + batch] = n
    return out


class SonataEncoder:
    """Frozen pretrained Sonata, per-point read-out at full resolution."""

    def __init__(self, grid: float):
        import sonata
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            import flash_attn  # noqa: F401
            custom = None
        except ImportError:
            custom = dict(enc_patch_size=[1024] * 5, enable_flash=False)
        self.model = sonata.load("sonata", repo_id="facebook/sonata",
                                 download_root=str(DATA / "models" / "sonata"),
                                 custom_config=custom).to(self.device).eval()
        self.transform = sonata.transform.Compose([
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=grid, hash_type="fnv", mode="train",
                 return_grid_coord=True, return_inverse=True),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"),
                 feat_keys=("coord", "color", "normal")),
        ])

    def __call__(self, coord, color, normal, keep):
        """Embeddings of the points flagged in `keep` (the rest is context).
        Raises PassTooBig, with the GPU memory released, on CUDA OOM."""
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            return self._forward(coord, color, normal, keep)
        except torch.cuda.OutOfMemoryError:
            pass
        # out here the exception, and the tensors its frames held, are gone
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        raise PassTooBig()

    def _forward(self, coord, color, normal, keep):
        torch = self.torch
        local = (coord - coord.mean(0)).astype(np.float32)   # centre in f64, then cast
        point = self.transform(dict(coord=local, color=color.astype(np.float32),
                                    normal=normal.astype(np.float32)))
        with torch.inference_mode():
            for key, v in point.items():
                if isinstance(v, torch.Tensor):
                    point[key] = v.to(self.device, non_blocking=True)
            point = self.model(point)
            for _ in range(2):                  # the two coarsest levels: concatenate
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            # the finer levels only copy features down; follow the indices
            # instead of building full-resolution copies of the whole pass
            feat, chain = point.feat, []
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                chain.append(point.pop("pooling_inverse"))
                point = parent
            sel = point.inverse[torch.from_numpy(np.flatnonzero(keep)).to(self.device)]
            for inverse in reversed(chain):
                sel = inverse[sel]
            out = feat[sel].half().cpu().numpy()
        if self.device == "cuda":
            self.peak_gb = torch.cuda.max_memory_allocated() / 1e9
            del point, feat, sel, chain
            torch.cuda.empty_cache()
        return out


class PassTooBig(Exception):
    """An encoder pass did not fit on the GPU."""


def embed(encoder, coord, color, normal, tile: float, margin: float,
          min_tile: float = 1.0) -> np.ndarray:
    """Encoder over XY tiles: each pass sees its tile plus `margin` of context
    and keeps the features of its own tile only, so every point gets exactly
    one. A pass that does not fit on the GPU is split in four and retried, with
    half the context: at small sizes the margin, not the tile, is the bulk."""
    xy = coord[:, :2]
    lo = xy.min(0)
    span = xy.max(0) - lo + 1e-3                     # top edge inside the last tile
    nt = np.maximum(np.ceil(span / tile).astype(int), 1)
    step = span / nt
    todo = [(lo + step * [i, j], lo + step * [i + 1, j + 1], margin)
            for i in range(nt[0]) for j in range(nt[1])]
    feat, done, passes = None, 0, 0
    while todo:
        clo, chi, m = todo.pop(0)
        idx = np.flatnonzero(np.all((xy >= clo - m) & (xy < chi + m), 1))
        mine = np.all((xy[idx] >= clo) & (xy[idx] < chi), 1)
        if not mine.any():
            continue
        t = time.time()
        try:
            f = encoder(coord[idx], color[idx], normal[idx], mine)
        except PassTooBig:
            if (chi - clo).max() / 2 < min_tile:
                raise SystemExit(f"a {(chi - clo).max():.1f} m pass with "
                                 f"{len(idx):,} points still does not fit on the GPU")
            mid = (clo + chi) / 2
            todo[:0] = [(np.array([a0, b0]), np.array([a1, b1]), m / 2)
                        for a0, a1 in ((clo[0], mid[0]), (mid[0], chi[0]))
                        for b0, b1 in ((clo[1], mid[1]), (mid[1], chi[1]))]
            log(f"  {len(idx):,} pts did not fit, splitting a "
                f"{(chi - clo).max():.1f} m pass (+{m:.2f} m context) in four")
            continue
        if feat is None:
            feat = np.full((len(coord), f.shape[1]), np.nan, np.float16)
        feat[idx[mine]] = f
        done += int(mine.sum())
        passes += 1
        peak = getattr(encoder, "peak_gb", None)
        log(f"  pass {passes} ({(chi - clo).max():.1f} m +{m:.2f} m): "
            f"{len(idx):,} pts in, {mine.sum():,} kept, "
            f"{f.shape[1]}-d, {time.time() - t:.1f} s"
            + (f", GPU peak {peak:.1f} GB" if peak else "")
            + f" — {done / len(coord):.0%} of cells done")
    assert done == len(coord) and not np.isnan(feat).any(), "a point got no embedding"
    return feat


# ---- 4. clustering in embedding space ---------------------------------------

def kmeans(feat: np.ndarray, k: int, seed: int, iters: int = 50,
           batch: int = 200_000) -> np.ndarray:
    """Spherical k-means (cosine), k-means++ seeding on a sample. Runs on the
    GPU when there is one, holding the features there if they fit."""
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    src = torch.from_numpy(feat)
    if dev == "cuda" and feat.nbytes < 0.5 * torch.cuda.mem_get_info()[0]:
        src = src.to(dev)

    def rows(a, b=None):
        x = src[a] if b is None else src[a:b]
        return F.normalize(x.to(dev).float(), dim=1)

    n = len(feat)
    S = rows(torch.from_numpy(np.sort(rng.choice(n, min(n, 50_000), replace=False))))
    C = [S[rng.integers(len(S))]]
    d = (1 - S @ C[0]).clamp_min(0)
    for _ in range(1, k):
        p = (d ** 2).double().cpu().numpy()
        i = rng.choice(len(S), p=p / p.sum()) if p.sum() > 0 else rng.integers(len(S))
        C.append(S[i])
        d = torch.minimum(d, (1 - S @ S[i]).clamp_min(0))
    C = torch.stack(C)

    def assign():
        return torch.cat([(rows(b, b + batch) @ C.T).argmax(1)
                          for b in range(0, n, batch)])

    for it in range(iters):
        sums = torch.zeros_like(C)
        cnt = torch.zeros(k, device=dev)
        for b in range(0, n, batch):
            x = rows(b, b + batch)
            lab = (x @ C.T).argmax(1)
            sums.index_add_(0, lab, x)
            cnt += torch.bincount(lab, minlength=k)
        new = torch.where(cnt[:, None] > 0, F.normalize(sums, dim=1), C)
        shift = float((1 - (new * C).sum(1)).max())
        C = new
        if shift < 1e-6:
            break
    log(f"  k-means: {it + 1} iterations, last centre shift {shift:.2e}")
    lab = assign().cpu().numpy()
    order = np.argsort(-np.bincount(lab, minlength=k))   # cluster 0 = largest
    return np.argsort(order)[lab]


# ---- 5. output --------------------------------------------------------------

def palette(k: int) -> np.ndarray:
    import matplotlib
    cmap = matplotlib.colormaps["tab10" if k <= 10 else "tab20" if k <= 20 else "hsv"]
    return np.array([cmap(i / max(k - 1, 1) if k > 20 else i)[:3] for i in range(k)])


def write_scene(las, labels: np.ndarray, k: int, path: Path) -> None:
    if "red" not in las.point_format.dimension_names:
        las = laspy.convert(las, point_format_id=7 if las.point_format.id >= 6 else 3)
    rgb = (palette(k)[labels] * 65535).round().astype(np.uint16)
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.add_extra_dim(laspy.ExtraBytesParams(name="cluster", type=np.uint16,
                                             description="k-means id, Sonata space"))
    las.cluster = labels.astype(np.uint16)
    las.write(path)


def write_preview(xyz: np.ndarray, labels: np.ndarray, k: int, title: str,
                  path: Path, n: int = 400_000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    s = rng.choice(len(xyz), min(n, len(xyz)), replace=False)
    s = s[np.argsort(xyz[s, 2])]                     # high points drawn last
    p, col = xyz[s] - xyz.min(0), palette(k)[labels[s]]
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax[0].scatter(p[:, 0], p[:, 1], c=col, s=0.2, lw=0)
    ax[0].set(aspect="equal", title="top", xlabel="x, m", ylabel="y, m")
    s2 = np.argsort(-p[:, 1])                        # side view, near points last
    ax[1].scatter(p[s2, 0], p[s2, 2], c=col[s2], s=0.2, lw=0)
    ax[1].set(aspect="equal", title="side (looking +y)", xlabel="x, m", ylabel="z, m")
    for c in range(k):
        ax[1].scatter([], [], c=[palette(k)[c]], s=40,
                      label=f"{c}: {np.mean(labels == c):.1%}")
    ax[1].legend(loc="upper right", fontsize=9, markerscale=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(cfg: dict, encoder=None, out_dir: Path | None = None) -> dict:
    t0 = time.time()
    out_dir = out_dir or RESULTS / "sonata_cluster"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    src = find_source(cfg["source"])
    log(f"source {src} ({src.stat().st_size / 1e9:.1f} GB), seed {cfg['seed']}")

    size = cfg["square"]
    if cfg["center"]:
        x0, y0 = cfg["center"][0] - size / 2, cfg["center"][1] - size / 2
        cover = None
    else:
        xy = sample_xy(src, rng)
        (x0, y0), cover = pick_square(xy, size, rng, cfg["cover"])
        log(f"square from {len(xy):,} sampled points, coverage {cover:.0%}")
    log(f"square {size:g} m, corner ({x0:.2f}, {y0:.2f})")

    las = read_square(src, x0, y0, size)
    xyz = np.c_[las.x, las.y, las.z]                 # float64, scan coordinates
    log(f"{len(xyz):,} points in the square, z {xyz[:, 2].min():.1f}..{xyz[:, 2].max():.1f}")

    inv, nv = voxelize(xyz, cfg["grid"])
    vox = cell_mean(xyz, inv, nv)
    vcol = cell_mean(colour_of(las), inv, nv).astype(np.float32)
    log(f"{nv:,} cells at {cfg['grid']:g} m; estimating normals")
    vnorm = normals(vox)

    encoder = encoder or SonataEncoder(cfg["grid"])
    log("embedding")
    feat = embed(encoder, vox, vcol, vnorm, cfg["tile"], cfg["margin"])
    log(f"clustering {nv:,} x {feat.shape[1]} into k={cfg['k']}")
    labels = kmeans(feat, cfg["k"], cfg["seed"])[inv]  # cell label -> every point

    stem = f"square_seed{cfg['seed']}_{size:g}m_k{cfg['k']}"
    write_scene(las, labels, cfg["k"], out_dir / f"{stem}.laz")
    title = (f"Sonata embeddings + k-means (k={cfg['k']}), {size:g} m square at "
             f"({x0 + size / 2:.1f}, {y0 + size / 2:.1f}), {len(xyz):,} points")
    write_preview(xyz, labels, cfg["k"], title, out_dir / f"{stem}.png")
    meta = dict(source=str(src), seed=cfg["seed"], square_m=size,
                corner=[x0, y0], center=[x0 + size / 2, y0 + size / 2],
                coverage=cover, points=int(len(xyz)), cells=nv, grid_m=cfg["grid"],
                embedding_dim=int(feat.shape[1]), k=cfg["k"],
                cluster_share=[float(np.mean(labels == c)) for c in range(cfg["k"])],
                seconds=round(time.time() - t0, 1))
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {out_dir / stem}.laz / .png / .json ({meta['seconds']:.0f} s). "
        f"Repeat with SC_SEED={cfg['seed']}")
    return meta


if __name__ == "__main__":
    try:
        run(settings())
    except Exception as e:  # CUDA OOM outside the encoder gets a hint too
        if "out of memory" in str(e).lower():
            import traceback
            traceback.print_exc()
            sys.exit("CUDA out of memory outside the encoder passes: "
                     "lower SC_SQUARE")
        raise
