#!/usr/bin/env python3
"""StemNet v2: look at the points in squares across the campus. No model, no
clustering, no labels: the 0.02 m grid tiles drawn as images.

Squares (V2_VIEW_SIZE m, default 24), wholly scanned (>= 90% of their 1 m
cells hold points):
  - V2_VIEW_N (8) spread over point density: the squares whose mean density
    sits at the 5th ... 95th percentile of all such squares, kept apart
  - plus every centre in V2_VIEW_XY ("x,y;x,y", frame metres; default the
    probe square, 162,274)

Per square, four images at V2_VIEW_PX (5 cm) per pixel. Every pixel is
computed from the points in it, nothing is drawn as a marker, so WHITE MEANS NO
POINT AT ALL in that pixel:
  1. from above: height of the highest point in the pixel
  2. from below: height of the lowest point in the pixel (where the ground
     was scanned, this is the ground)
  3. colour: the scan's RGB of the highest point in the pixel
  4. side view: points per pixel (log) in a 2 m slab through the square's
     middle, x against z
Heights are metres above the square's lowest 1% of points.

Outputs:
  results/49_stemnet_v2_view_map.png       campus occupancy, squares numbered
  results/49_stemnet_v2_view_<n>.png       one per square
  results/stemnet_v2_view.txt              per square: points, points/m^2,
                                           share of pixels with a point from above
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS, TILE  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
GRID_DIR = OUT / f"grid{GRID:g}"
SIZE = float(os.environ.get("V2_VIEW_SIZE", "24"))
N = int(os.environ.get("V2_VIEW_N", "8"))
XY = os.environ.get("V2_VIEW_XY", "162,274")
PX = float(os.environ.get("V2_VIEW_PX", "0.05"))
SLAB = 2.0
LOG = RESULTS / "stemnet_v2_view.txt"


def log(msg):
    line = f"[v2-view] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def box_sum(a, k):
    I = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    I[1:, 1:] = a.cumsum(0).cumsum(1)
    return I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]


def pick_squares(occ):
    """-> list of (cx, cy, why)."""
    k = int(round(SIZE))
    out = []
    for s in XY.split(";"):
        if s.strip():
            x, y = (float(v) for v in s.split(","))
            out.append((x, y, "chosen"))
    full = box_sum((occ > 0).astype(np.float64), k) >= 0.9 * k * k
    dens = box_sum(occ.astype(np.float64), k) / (k * k)
    jj, ii = np.nonzero(full)
    if not len(jj) or N <= 0:
        return out
    d = dens[jj, ii]
    order = np.argsort(d)
    for q in np.linspace(5, 95, N):
        # walk outward from the q-th percentile to the nearest square not overlapping a chosen one
        start = int(round(q / 100 * (len(order) - 1)))
        for step in range(len(order)):
            for t in (start + step, start - step):
                if 0 <= t < len(order):
                    j, i = jj[order[t]], ii[order[t]]
                    cx, cy = i + SIZE / 2, j + SIZE / 2
                    if all(max(abs(cx - a), abs(cy - b)) >= SIZE for a, b, _ in out):
                        out.append((cx, cy, f"density p{q:.0f}: {d[order[t]]:,.0f} pts/m2"))
                        break
            else:
                continue
            break
    return out


def load_square(cx, cy):
    h = SIZE / 2
    xyz, rgb = [], []
    for a in range(int(np.floor((cx - h) / TILE)), int(np.floor((cx + h) / TILE)) + 1):
        for b in range(int(np.floor((cy - h) / TILE)), int(np.floor((cy + h) / TILE)) + 1):
            q = GRID_DIR / f"tile_{a}_{b}.npz"
            if not q.exists():
                continue
            g = np.load(q)
            p = g["xyz"]
            m = (np.abs(p[:, 0] - cx) < h) & (np.abs(p[:, 1] - cy) < h)
            xyz.append(p[m].astype(np.float64)); rgb.append(g["rgb"][m])
    if not xyz:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return np.concatenate(xyz), np.concatenate(rgb)


def raster(xyz, rgb, x0, y0):
    """-> top (highest z), bottom (lowest z), colour of the highest point, per pixel."""
    n = int(round(SIZE / PX))
    i = np.clip(((xyz[:, 0] - x0) / PX).astype(np.int64), 0, n - 1)
    j = np.clip(((xyz[:, 1] - y0) / PX).astype(np.int64), 0, n - 1)
    key = j * n + i
    top = np.full(n * n, -np.inf); np.maximum.at(top, key, xyz[:, 2])
    bot = np.full(n * n, np.inf); np.minimum.at(bot, key, xyz[:, 2])
    o = np.lexsort((xyz[:, 2], key))                  # last row per pixel = highest point
    last = np.r_[key[o][1:] != key[o][:-1], True]
    col = np.ones((n * n, 3))
    col[key[o][last]] = rgb[o][last] / 255.0
    empty = ~np.isfinite(top)
    top[empty] = np.nan; bot[empty] = np.nan
    return top.reshape(n, n), bot.reshape(n, n), col.reshape(n, n, 3), empty.reshape(n, n)


def figure_square(k, cx, cy, why, xyz, rgb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    x0, y0 = cx - SIZE / 2, cy - SIZE / 2
    z0 = np.percentile(xyz[:, 2], 1)
    z = xyz.copy(); z[:, 2] -= z0
    top, bot, col, empty = raster(z, rgb, x0, y0)
    zmax = max(np.nanpercentile(top, 99), 1.0)
    cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_bad("white")
    ext = (x0, x0 + SIZE, y0, y0 + SIZE)
    fig, axs = plt.subplots(1, 4, figsize=(24, 6.6))
    im = axs[0].imshow(top, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=zmax, interpolation="nearest")
    axs[0].set_title("from above: height of the highest point (m)"); fig.colorbar(im, ax=axs[0], shrink=0.8)
    im = axs[1].imshow(bot, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=min(zmax, 3.0), interpolation="nearest")
    axs[1].set_title("from below: height of the lowest point (m, 0-3)"); fig.colorbar(im, ax=axs[1], shrink=0.8)
    axs[2].imshow(col, origin="lower", extent=ext, interpolation="nearest")
    axs[2].set_title("scan RGB of the highest point")
    s = np.abs(z[:, 1] - cy) <= SLAB / 2
    zt = max(np.percentile(z[s, 2], 99.9) if s.any() else 1.0, 1.0)
    nz = int(np.ceil((zt + 0.5) / PX))
    H, _, _ = np.histogram2d(z[s, 0], z[s, 2], bins=[int(round(SIZE / PX)), nz], range=[[x0, x0 + SIZE], [-0.5, zt]])
    H[H == 0] = np.nan
    cm2 = matplotlib.colormaps["magma_r"].copy(); cm2.set_bad("white")
    im = axs[3].imshow(H.T, origin="lower", extent=(x0, x0 + SIZE, -0.5, zt), cmap=cm2, norm=LogNorm(),
                       interpolation="nearest", aspect="equal")
    axs[3].set_title(f"side view, slab y {cy - SLAB / 2:.0f}-{cy + SLAB / 2:.0f}: points per pixel")
    fig.colorbar(im, ax=axs[3], shrink=0.8)
    for ax in axs[:3]:
        ax.set_xlim(x0, x0 + SIZE); ax.set_ylim(y0, y0 + SIZE)
    fig.suptitle(f"square {k}: centre ({cx:.0f}, {cy:.0f}) frame m, {SIZE:g} m | {why} | {len(xyz):,} points, "
                 f"{len(xyz) / SIZE ** 2:,.0f} per m2 | {(~empty).mean() * 100:.0f}% of {PX * 100:g} cm pixels "
                 f"hold a point | white = no point", fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / f"49_stemnet_v2_view_{k}.png", dpi=100); plt.close(fig)
    return (~empty).mean()


def figure_map(occ, squares):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    o = occ.astype(np.float64); o[o == 0] = np.nan
    fig, ax = plt.subplots(figsize=(12, 12 * occ.shape[0] / max(occ.shape[1], 1)))
    im = ax.imshow(o, origin="lower", cmap="Greys", norm=LogNorm(), interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.6, label="grid points per 1 m cell")
    for k, (cx, cy, _) in enumerate(squares):
        ax.add_patch(Rectangle((cx - SIZE / 2, cy - SIZE / 2), SIZE, SIZE, fill=False, ec="tab:red", lw=1.5))
        ax.text(cx, cy, str(k), color="tab:red", ha="center", va="center", fontsize=12, weight="bold")
    ax.set_title("Viewed squares on the campus (frame m)")
    fig.tight_layout(); fig.savefig(RESULTS / "49_stemnet_v2_view_map.png", dpi=100); plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    regions = np.load(OUT / "regions.npz")
    occ = regions["occupancy"]
    squares = pick_squares(occ)
    log(f"{len(squares)} squares of {SIZE:g} m, {PX * 100:g} cm pixels")
    figure_map(occ, squares)
    for k, (cx, cy, why) in enumerate(squares):
        xyz, rgb = load_square(cx, cy)
        if not len(xyz):
            log(f"square {k} ({cx:.0f}, {cy:.0f}): no points"); continue
        cov = figure_square(k, cx, cy, why, xyz, rgb)
        log(f"square {k} ({cx:.0f}, {cy:.0f}) {why}: {len(xyz):,} points, {len(xyz) / SIZE ** 2:,.0f} per m2, "
            f"{cov * 100:.0f}% of pixels hold a point from above")
    log(f"-> results/49_stemnet_v2_view_map.png, 49_stemnet_v2_view_<n>.png (0-{len(squares) - 1})")


if __name__ == "__main__":
    main()
