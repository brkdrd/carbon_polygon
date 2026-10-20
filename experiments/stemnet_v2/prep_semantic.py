#!/usr/bin/env python3
"""StemNet v2 prep, part 5b: tree / non-tree labels per grid point, carried
over from campus_pred_B24.laz.

That file is the campus scan thinned to ~3 cm with a label per point
(pred_tree, 1 = tree), the ground removed, and Z replaced by height above its
own ground model. Measured on 80 real grid columns: that ground varies even
inside a 3 m column (median spread 0.30 m between quadrants), so the height
offset (frame z - Z) is estimated per 0.5 m cell; after it, 96% of grid points
>= 0.3 m above the ground have a labelled point within 5 cm.

0. If no campus_pred_B24.laz is under data/raw, download it from the public
   Kaggle dataset V2_SEM_KAGGLE (brkdrd/campus-data-tree-no-tree-split,
   1.76 GB, public: the single-file download needs no credentials). It lands
   in a temporary folder first and is moved into data/raw only when complete.
1. Read the file once, into the shared frame (x, y minus origin_m; Z as is).
2. Offset per 0.5 m cell: pair every grid point with the labelled points
   within V2_SEM_PAIR_M (1.5 cm) horizontally; offset = the most common
   (z_grid - Z) in 1 cm bins, from >= V2_SEM_MIN_PAIRS (30) pairs. On a trunk
   the pairs spread over its height; the mode is where both clouds coincide.
   Cells without enough pairs take the nearest cell's offset within
   V2_SEM_FILL_M (2 m); further away the offset is unknown.
3. Per grid point, with the labelled points lifted by their cell's offset:
     sem = 1   tree       nearest labelled point within V2_SEM_MATCH_M (5 cm) is tree
     sem = 0   non-tree   ... is non-tree
     sem = 2   ground     no labelled point within 5 cm and < V2_SEM_GROUND_M
                          (0.3 m) above the file's ground: the ground it removed,
                          non-tree (user's choice)
     sem = -1  unknown    otherwise (no match higher up, or no offset): no loss
   Training target: tree = (sem == 1); loss only where sem >= 0.

Outputs (data/stemnet_v2/):
  sem0.02/tile_<i>_<j>.npz   sem int8, hag float16 (height above the file's
                             ground, nan where unknown), row-aligned with the grid
  frame.json                 ["semantic"]: stamp, counts per split
  results/stemnet_v2_semantic.txt, 50_stemnet_v2_semantic.png
"""
from __future__ import annotations

import json
import os
import zipfile
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import DATA, OUT, RESULTS, TILE  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
SEM_NAME = os.environ.get("V2_SEM_LAZ", "campus_pred_B24.laz")
KAGGLE_SEM = os.environ.get("V2_SEM_KAGGLE", "brkdrd/campus-data-tree-no-tree-split")
CELL = float(os.environ.get("V2_SEM_CELL", "0.5"))
PAIR_M = float(os.environ.get("V2_SEM_PAIR_M", "0.015"))
MIN_PAIRS = int(os.environ.get("V2_SEM_MIN_PAIRS", "30"))
FILL_M = float(os.environ.get("V2_SEM_FILL_M", "2.0"))
MATCH_M = float(os.environ.get("V2_SEM_MATCH_M", "0.05"))
GROUND_M = float(os.environ.get("V2_SEM_GROUND_M", "0.3"))
HALO = 0.1
GRID_DIR = OUT / f"grid{GRID:g}"
SEM_DIR = OUT / f"sem{GRID:g}"
FRAME_JSON = OUT / "frame.json"
CODES = {1: "tree", 0: "non-tree", 2: "ground", -1: "unknown"}


def log(msg):
    print(f"[v2-semantic] {msg}", flush=True)


def tile_ij(p: Path):
    a, b = p.stem.split("_")[1:]
    return int(a), int(b)


def find_source():
    raw = DATA / "raw"
    hits = sorted(raw.rglob(SEM_NAME)) if raw.exists() else []
    if hits:
        log(f"labelled file: {hits[0]} ({hits[0].stat().st_size / 1e9:.2f} GB)")
        return hits[0]
    return download_source(raw)


def download_source(raw):
    """Kaggle -> a temporary folder in data/raw -> data/raw/SEM_NAME, so a broken download never looks done."""
    log(f"no {SEM_NAME} under {raw} — downloading Kaggle dataset {KAGGLE_SEM} (1.76 GB)")
    raw.mkdir(parents=True, exist_ok=True)
    tmp = raw / ".campus_pred_download"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    t0 = time.time()
    # the whole-dataset zip is 404 without credentials; the single file is not
    r = subprocess.run(["kaggle", "datasets", "download", KAGGLE_SEM, "-f", SEM_NAME, "-p", str(tmp), "--force"],
                       capture_output=True, text=True)
    for z in tmp.rglob(SEM_NAME + ".zip"):          # older CLIs zip a large single file
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
        z.unlink()
    hits = sorted(tmp.rglob(SEM_NAME))
    if r.returncode != 0 or not hits:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"[v2-semantic] Kaggle download of {KAGGLE_SEM} failed"
                         + ("" if r.returncode else f" (no {SEM_NAME} in it)")
                         + ". Check KAGGLE_USERNAME / KAGGLE_KEY in .env, or put the file in data/raw by hand.")
    dst = raw / SEM_NAME
    os.replace(hits[0], dst)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"downloaded -> {dst} ({dst.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f} s)")
    return dst


def read_source(path, origin):
    """-> x, y (frame, float32), Z (height above its ground), tree (int8), sorted by storage tile."""
    import laspy
    with laspy.open(path) as f:
        n = f.header.point_count
        x = np.empty(n, np.float32); y = np.empty(n, np.float32); z = np.empty(n, np.float32)
        lab = np.empty(n, np.int8)
        k = 0
        for ch in f.chunk_iterator(10_000_000):
            m = len(ch)
            x[k:k + m] = np.asarray(ch.x) - origin[0]
            y[k:k + m] = np.asarray(ch.y) - origin[1]
            z[k:k + m] = np.asarray(ch.z)
            lab[k:k + m] = np.asarray(ch["pred_tree"]).astype(np.int8)
            k += m
    key = (np.floor(x / TILE).astype(np.int64) * 100000 + np.floor(y / TILE).astype(np.int64))
    o = np.argsort(key, kind="stable")
    return x[o], y[o], z[o], lab[o], key[o]


class Source:
    def __init__(self, x, y, z, lab, key):
        self.x, self.y, self.z, self.lab, self.key = x, y, z, lab, key

    def around(self, i, j, halo=HALO):
        """Labelled points of tile (i, j) and its neighbours within halo."""
        parts = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                kk = (i + di) * 100000 + (j + dj)
                a, b = np.searchsorted(self.key, [kk, kk + 1])
                if b > a:
                    parts.append(np.arange(a, b))
        if not parts:
            return np.zeros(0, np.int64)
        idx = np.concatenate(parts)
        x, y = self.x[idx], self.y[idx]
        m = (x >= i * TILE - halo) & (x < (i + 1) * TILE + halo) & (y >= j * TILE - halo) & (y < (j + 1) * TILE + halo)
        return idx[m]


def cell_index(x, y, shape):
    ci = np.clip(np.floor(np.asarray(x) / CELL).astype(np.int64), 0, shape[1] - 1)
    cj = np.clip(np.floor(np.asarray(y) / CELL).astype(np.int64), 0, shape[0] - 1)
    return cj, ci


def tile_offsets(g, src, idx, shape):
    """-> (cell_j, cell_i, offset, pairs) for the cells of one tile with enough pairs."""
    from scipy.spatial import cKDTree
    if len(idx) == 0:
        return None
    bxy = np.column_stack([src.x[idx], src.y[idx]]).astype(np.float64)
    d, nb = cKDTree(bxy).query(g[:, :2], k=4, distance_upper_bound=PAIR_M, workers=-1)
    ok = np.isfinite(d)
    gi = np.nonzero(ok)[0]
    if not len(gi):
        return None
    dz = g[gi, 2] - src.z[idx[nb[ok]]]
    cj, ci = cell_index(g[gi, 0], g[gi, 1], shape)
    cell = cj * shape[1] + ci
    zb = np.round(dz / 0.01).astype(np.int64) + 50000            # 1 cm bins, offsets within +-500 m
    u, cnt = np.unique(cell * 100000 + zb, return_counts=True)
    uc, ub = u // 100000, u % 100000
    o = np.lexsort((-cnt, uc))                                   # per cell, most common bin first
    first = np.r_[True, uc[o][1:] != uc[o][:-1]]
    tot = np.bincount(np.searchsorted(np.unique(uc), uc), weights=cnt)
    cells = uc[o][first]
    mode = (ub[o][first] - 50000) * 0.01
    pairs = tot[np.searchsorted(np.unique(uc), cells)]
    keep = pairs >= MIN_PAIRS
    return cells[keep] // shape[1], cells[keep] % shape[1], mode[keep], pairs[keep]


def fill(D):
    """Cells without an offset take the nearest cell's within FILL_M."""
    from scipy import ndimage
    have = np.isfinite(D)
    if not have.any():
        return D, 0
    dist, (jj, ii) = ndimage.distance_transform_edt(~have, return_indices=True)
    F = D[jj, ii].copy()
    F[dist * CELL > FILL_M] = np.nan
    return F, int((np.isfinite(F) & ~have).sum())


def label_tile(g, src, idx, D):
    """-> sem (int8), hag (float32) for the grid points g of one tile."""
    from scipy.spatial import cKDTree
    n = len(g)
    sem = np.full(n, -1, np.int8)
    gj, gi = cell_index(g[:, 0], g[:, 1], D.shape)
    off = D[gj, gi]
    hag = g[:, 2] - off
    matched = np.zeros(n, bool)
    if len(idx):
        bj, bi = cell_index(src.x[idx], src.y[idx], D.shape)
        boff = D[bj, bi]
        ok = np.isfinite(boff)
        if ok.any():
            B = np.column_stack([src.x[idx][ok], src.y[idx][ok], src.z[idx][ok] + boff[ok]]).astype(np.float64)
            dist, nn = cKDTree(B).query(g, k=1, distance_upper_bound=MATCH_M, workers=-1)
            matched = np.isfinite(dist)
            sem[matched] = src.lab[idx][ok][nn[matched]]
    ground = ~matched & np.isfinite(hag) & (hag < GROUND_M)
    sem[ground] = 2
    return sem, hag


def figure(cnt, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tot = cnt.sum(-1).astype(np.float64)
    lab = tot - cnt[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        tree = np.where(lab > 0, cnt[..., 0] / lab, np.nan)
        unk = np.where(tot > 0, cnt[..., 3] / tot, np.nan)
        grd = np.where(tot > 0, cnt[..., 2] / tot, np.nan)
    fig, axs = plt.subplots(1, 3, figsize=(24, 8.5))
    for ax, a, t, cm in ((axs[0], tree, "share TREE among labelled points", "Greens"),
                         (axs[1], grd, "share GROUND (removed in the file -> non-tree)", "Oranges"),
                         (axs[2], unk, "share UNKNOWN (no loss)", "Reds")):
        c = matplotlib.colormaps[cm].copy(); c.set_bad("white")
        im = ax.imshow(a, origin="lower", cmap=c, vmin=0, vmax=1, interpolation="nearest")
        ax.contour(split, levels=[0.5, 1.5, 2.5], colors="k", linewidths=0.5)
        ax.set_title(t + " per 1 m cell"); fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Tree / non-tree labels carried over from campus_pred_B24 onto the 0.02 m grid "
                 "(black lines: train / val / test borders)")
    fig.tight_layout(); fig.savefig(RESULTS / "50_stemnet_v2_semantic.png", dpi=90); plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = json.loads(FRAME_JSON.read_text())
    src_path = find_source()
    st = src_path.stat()
    stamp = dict(grid=frame.get("grids", {}).get(f"{GRID:g}", {}).get("stamp"), source=src_path.name,
                 size=st.st_size, mtime=int(st.st_mtime), cell=CELL, pair_m=PAIR_M, min_pairs=MIN_PAIRS,
                 fill_m=FILL_M, match_m=MATCH_M, ground_m=GROUND_M, tile=TILE)
    if frame.get("semantic", {}).get("stamp") == stamp and SEM_DIR.exists():
        log("up to date — nothing to do"); return
    t0 = time.time()
    origin = np.asarray(frame["origin_m"], np.float64)
    src = Source(*read_source(src_path, origin))
    log(f"{src_path.name}: {len(src.x):,} labelled points, {(src.lab == 1).mean() * 100:.1f}% tree | "
        f"{time.time() - t0:.0f} s")
    tiles = sorted(GRID_DIR.glob("tile_*.npz"), key=tile_ij)
    ij = np.array([tile_ij(q) for q in tiles])
    shape = (int(round((ij[:, 1].max() + 1) * TILE / CELL)), int(round((ij[:, 0].max() + 1) * TILE / CELL)))

    # ---- pass 1: height offset per cell -------------------------------
    D = np.full(shape, np.nan, np.float32)
    P = np.zeros(shape, np.int32)
    for k, q in enumerate(tiles):
        i, j = tile_ij(q)
        g = np.load(q)["xyz"].astype(np.float64)
        r = tile_offsets(g, src, src.around(i, j), shape)
        if r is not None:
            cj, ci, m, p = r
            D[cj, ci] = m; P[cj, ci] = p
        if (k + 1) % 50 == 0 or k + 1 == len(tiles):
            log(f"offsets: tile {k + 1}/{len(tiles)} | {(time.time() - t0) / 60:.1f} min")
    n_direct = int(np.isfinite(D).sum())
    D, n_filled = fill(D)
    log(f"offset cells: {n_direct:,} measured (median {np.median(P[P >= MIN_PAIRS]) if n_direct else 0:.0f} "
        f"pairs), {n_filled:,} filled from within {FILL_M:g} m")

    # ---- pass 2: labels -----------------------------------------------
    tmp = SEM_DIR.with_name(SEM_DIR.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    split = np.load(OUT / "regions.npz")["split"]
    names = ["train", "val", "test", "buffer"]
    tally = {s: np.zeros(4, np.int64) for s in names}
    cnt = np.zeros(split.shape + (4,), np.int64)                   # per 1 m cell: tree, non-tree, ground, unknown
    for k, q in enumerate(tiles):
        i, j = tile_ij(q)
        g = np.load(q)["xyz"].astype(np.float64)
        sem, hag = label_tile(g, src, src.around(i, j), D)
        np.savez(tmp / q.name, sem=sem, hag=hag.astype(np.float16))
        cy = np.clip(np.floor(g[:, 1]).astype(int), 0, split.shape[0] - 1)
        cx = np.clip(np.floor(g[:, 0]).astype(int), 0, split.shape[1] - 1)
        col = np.searchsorted(np.array([-1, 0, 1, 2]), sem)          # -1,0,1,2 -> 0..3
        col = np.array([3, 1, 0, 2])[col]                             # -> tree 0, non-tree 1, ground 2, unknown 3
        np.add.at(cnt, (cy, cx, col), 1)
        sp = split[cy, cx]
        for si, s in enumerate(names):
            m = sp == si
            if m.any():
                tally[s] += np.bincount(col[m], minlength=4)
        if (k + 1) % 50 == 0 or k + 1 == len(tiles):
            el = time.time() - t0
            log(f"labels: tile {k + 1}/{len(tiles)} | {el / 60:.1f} min")
    if SEM_DIR.exists():
        shutil.rmtree(SEM_DIR)
    tmp.rename(SEM_DIR)

    L = ["=" * 72, "StemNet v2 — tree / non-tree labels from " + src_path.name, "=" * 72,
         f"offset per {CELL:g} m cell from pairs within {PAIR_M * 100:g} cm (>= {MIN_PAIRS}), filled within "
         f"{FILL_M:g} m | match within {MATCH_M * 100:g} cm | ground = unmatched and < {GROUND_M:g} m above",
         f"offset cells measured {n_direct:,}, filled {n_filled:,}", "",
         f"{'split':8s}{'points':>15s}{'tree%':>8s}{'non-tree%':>11s}{'ground%':>9s}{'unknown%':>10s}"
         f"{'tree% of labelled':>19s}"]
    for s in names:
        t = tally[s]; tot = t.sum()
        if tot:
            L.append(f"{s:8s}{tot:15,d}" + "".join(f"{v / tot * 100:{w}.2f}" for v, w in zip(t, (8, 11, 9, 10)))
                     + f"{t[0] / max(t[:3].sum(), 1) * 100:19.2f}")
    txt = "\n".join(L)
    (RESULTS / "stemnet_v2_semantic.txt").write_text(txt + "\n")
    print(txt, flush=True)
    figure(cnt, split)
    frame = json.loads(FRAME_JSON.read_text())
    frame["semantic"] = dict(dir=SEM_DIR.name, stamp=stamp, codes={str(k): v for k, v in CODES.items()},
                             tally={s: dict(zip(("tree", "non_tree", "ground", "unknown"), map(int, tally[s])))
                                    for s in names})
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log(f"-> data/stemnet_v2/{SEM_DIR.name}/ | results/stemnet_v2_semantic.txt, 50_ png | "
        f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
