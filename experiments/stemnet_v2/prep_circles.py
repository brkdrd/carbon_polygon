#!/usr/bin/env python3
"""StemNet v2 prep, part 3: the stem axis of every surveyed tree, found in the
cloud near the survey point.

Why the survey point is only a seed. The first version demanded a circle that
passes THROUGH the survey point (the stick touched the bark). On the real scan
that produced 47 cm median radii against girths of 5-15 cm: in the scan's
frame the survey points are not on the bark. They sit beside the trunks, and
the direction from survey point to trunk was the same (north-east, +52..+58
deg) in every survey block. So here the cloud finds the trunks on its own and
the survey point only says which one:

  1. column    every grid point within V2_SEARCH_R (1.5 m) of the survey point
  2. layers    the lowest V2_SCAN_H (8 m) cut into V2_LAYER (0.2 m) slices
  3. clusters  per layer, points split where gaps exceed V2_GAP (5 cm). A trunk
               cross-section is a compact cluster; ground and crown are huge
               ones and are skipped (wider than 2 x V2_R_MAX + 10 cm)
  4. circle    per cluster: RANSAC over 3-point circles + least-squares refit.
               Accepted only if
                 - radius V2_R_MIN..V2_R_MAX (2-60 cm)
                 - inliers cover >= V2_ARC (90) deg; >= V2_MIN_IN (12) inliers
                 - HOLLOW: point density inside the circle (or, below 6 cm,
                   in a 5 cm shell outside it) at most V2_INSIDE (0.25) x the
                   density on the ring — measured against ALL the layer's
                   points, so a ring drawn on ground still fails
  5. trunks    circles in consecutive layers whose centres agree within
               V2_RUN_TOL (10 cm) are chained; a chain of >= V2_RUN (3) layers
               is a trunk
  6. choice    the trunk whose median centre is nearest the survey point.
               Also recorded: how many trunks the column held and how far the
               second-nearest was, so an ambiguous pick is visible.

No ground estimate and no survey height anywhere. Every survey row is kept:
  ok      a trunk (chain >= V2_RUN layers) within V2_SEARCH_R
  weak    only shorter chains — the nearest used, flagged
  no_fit  points in the column, no circle accepted
  sparse  fewer than V2_MIN_COL (50) points in the column
Anything not ok/weak keeps the survey point as its base.

Then:
  offsets  survey -> derived base, per survey block and over the campus:
           median vector, spread, and how aligned the directions are. One
           common direction means scan and survey are shifted against each
           other; random directions mean RTK noise.
  girth    for the girth-attributed trees, the fitted radius 1.0-1.6 m above
           the trunk's lowest layer against girth / 2 pi. This is the test of
           the trunk finder itself: if radii match, the trunks are real.
  forks    rows whose DERIVED bases lie within V2_MERGE_R (0.3 m) share a
           fork_group; bases_merged.csv has one row per group for training.

Outputs:
  data/stemnet_v2/bases.csv, bases_merged.csv
  results/stemnet_v2_circles.txt
  results/42_stemnet_v2_girth.png             fitted vs surveyed radius
  results/43_stemnet_v2_circle_examples.png   trunks, circles, survey point
  results/45_stemnet_v2_offsets.png           survey -> trunk arrows, campus
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ATTRS, OUT, RESULTS, STEM_MAP, TILE, parse_stem_map, survey_to_local  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
SEARCH_R = float(os.environ.get("V2_SEARCH_R", "1.5"))
SCAN_H = float(os.environ.get("V2_SCAN_H", "8.0"))
LAYER = float(os.environ.get("V2_LAYER", "0.2"))
GAP = float(os.environ.get("V2_GAP", "0.05"))
R_MIN = float(os.environ.get("V2_R_MIN", "0.02"))
R_MAX = float(os.environ.get("V2_R_MAX", "0.6"))
ARC = float(os.environ.get("V2_ARC", "90"))
MIN_IN = int(os.environ.get("V2_MIN_IN", "12"))
IN_TOL = float(os.environ.get("V2_IN_TOL", "0.02"))      # inlier band, m (~ one voxel)
RUN = int(os.environ.get("V2_RUN", "3"))
RUN_TOL = float(os.environ.get("V2_RUN_TOL", "0.10"))
MIN_COL = int(os.environ.get("V2_MIN_COL", "50"))
MERGE_R = float(os.environ.get("V2_MERGE_R", "0.3"))
ITERS = int(os.environ.get("V2_RANSAC", "200"))
INSIDE = float(os.environ.get("V2_INSIDE", "0.25"))     # max density inside / density on the ring
SMALL_R = float(os.environ.get("V2_SMALL_R", "0.06"))   # below this there is no inside to test
SHELL = 0.05                                            # outside band for small circles, m
BH = (1.0, 1.6)
# V2_SUBSET=1: only some grid tiles are present (a trial on a few real tiles).
# Fit only the trees whose whole column lies in tiles that exist; a column cut
# off by a missing tile would give a confident, wrong fit.
SUBSET = os.environ.get("V2_SUBSET", "0") != "0"
KEEP_FOR_PLOT = 400
GRID_DIR = OUT / f"grid{GRID:g}"
FRAME_JSON = OUT / "frame.json"


def log(msg):
    print(f"[v2-circle] {msg}", flush=True)


# ---------------------------------------------------------------------------
# circle geometry
# ---------------------------------------------------------------------------
def circles_3pt(A, B, C):
    """Vectorised circumcircles of point triples (H, 2) each -> centres, radii."""
    ax, ay, bx, by, cx, cy = A[:, 0], A[:, 1], B[:, 0], B[:, 1], C[:, 0], C[:, 1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ok = np.abs(d) > 1e-12
    d = np.where(ok, d, 1.0)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    U = np.column_stack([ux, uy])
    r = np.hypot(ax - ux, ay - uy)
    return U, np.where(ok, r, np.inf)


def kasa(P):
    """Algebraic least-squares circle."""
    A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(max(sol[2] + c @ c, 0.0)))


def arc_deg(P, c):
    """Degrees of the circle the points cover: 360 minus the biggest gap."""
    a = np.sort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
    return float(np.degrees(2 * np.pi - gaps.max()))


def crowding(S, r):
    """How crowded the place a trunk ring must leave empty is, as a DENSITY
    relative to the ring itself. Densities, not counts: a small circle's inside
    is a tiny disc that holds few points even on solid ground, so a count would
    wave small circles through. Flat ground scores ~1 at any radius; a trunk ~0.

    S (H, n): signed distance of every point to each circle, r (H,) radii.
    Large circles are judged on their inside; below SMALL_R there is no inside
    left, so on a shell just outside the ring."""
    ring_n = (np.abs(S) <= IN_TOL).sum(1)
    ring_a = 2 * np.pi * r * 2 * IN_TOL
    inner_n = (S < -2 * IN_TOL).sum(1)
    inner_a = np.pi * np.clip(r - 2 * IN_TOL, 0, None) ** 2
    shell_n = ((S > 2 * IN_TOL) & (S <= 2 * IN_TOL + SHELL)).sum(1)
    ro = r + 2 * IN_TOL
    shell_a = np.pi * ((ro + SHELL) ** 2 - ro ** 2)
    ring_d = ring_n / np.maximum(ring_a, 1e-9)
    other = np.where(r < SMALL_R, shell_n / shell_a, inner_n / np.maximum(inner_a, 1e-9))
    return other / np.maximum(ring_d, 1e-9)


# ---------------------------------------------------------------------------
# one layer: every trunk cross-section in it
# ---------------------------------------------------------------------------
def clusters(P):
    """Connected components of a layer's points, linked when closer than GAP."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if len(P) < 2:
        return [np.arange(len(P))]
    pairs = cKDTree(P).query_pairs(GAP, output_type="ndarray")
    if not len(pairs):
        return [np.array([i]) for i in range(len(P))]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    _, lab = connected_components(g, directed=False)
    order = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.r_[True, np.diff(lab[order]) != 0, True])
    return [order[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def fit_cluster(C, P, rng):
    """Best accepted circle for one cluster C (m, 2); the hollow test is made
    against all the layer's points P, not just the cluster's."""
    m = len(C)
    if m < MIN_IN:
        return None
    idx = rng.integers(0, m, size=(ITERS, 3))
    U, r = circles_3pt(C[idx[:, 0]], C[idx[:, 1]], C[idx[:, 2]])
    keep = (r >= R_MIN) & (r <= R_MAX)
    if not keep.any():
        return None
    U, r = U[keep], r[keep]
    S = np.hypot(C[None, :, 0] - U[:, None, 0], C[None, :, 1] - U[:, None, 1]) - r[:, None]
    cnt = (np.abs(S) <= IN_TOL).sum(1)
    best = int(np.argmax(cnt))
    if cnt[best] < MIN_IN:
        return None
    c, rr = U[best], float(r[best])
    for _ in range(3):                                   # refit on the refit's inliers
        d = np.abs(np.hypot(C[:, 0] - c[0], C[:, 1] - c[1]) - rr)
        inl = C[d <= IN_TOL]
        if len(inl) < MIN_IN:
            return None
        c, rr = kasa(inl)
    if not (R_MIN <= rr <= R_MAX):
        return None
    arc = arc_deg(inl, c)
    if arc < ARC:
        return None
    inside = float(crowding((np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) - rr)[None, :],
                            np.array([rr]))[0])
    if inside > INSIDE:
        return None
    res = float(np.sqrt(np.mean((np.hypot(inl[:, 0] - c[0], inl[:, 1] - c[1]) - rr) ** 2)))
    return dict(cx=float(c[0]), cy=float(c[1]), r=float(rr), n_in=int(len(inl)),
                arc=arc, rms=res, inside=inside)


def layer_circles(P, rng):
    """Every accepted trunk cross-section in one layer P (n, 2)."""
    out = []
    if len(P) < MIN_IN:
        return out
    wide = 2 * R_MAX + 0.1
    for ix in clusters(P):
        if len(ix) < MIN_IN:
            continue
        C = P[ix]
        if (C.max(0) - C.min(0)).max() > wide:            # ground, crown, a wall
            continue
        f = fit_cluster(C, P, rng)
        if f is not None:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# one tree
# ---------------------------------------------------------------------------
def chains(layers):
    """Link circles in consecutive layers whose centres agree within RUN_TOL.
    layers: list of (k, z, [circles]). Returns lists of (k, z, circle)."""
    done, open_ = [], []
    prev_k = None
    for k, z, circs in layers:
        nxt, used = [], set()
        if prev_k is not None and k == prev_k + 1:
            for ch in open_:
                m = np.median([[c["cx"], c["cy"]] for _, _, c in ch], axis=0)
                best, bd = None, RUN_TOL
                for j, c in enumerate(circs):
                    if j in used:
                        continue
                    d = np.hypot(c["cx"] - m[0], c["cy"] - m[1])
                    if d <= bd:
                        best, bd = j, d
                if best is None:
                    done.append(ch)
                else:
                    used.add(best)
                    nxt.append(ch + [(k, z, circs[best])])
        else:
            done.extend(open_)
        for j, c in enumerate(circs):
            if j not in used:
                nxt.append([(k, z, c)])
        open_ = nxt
        prev_k = k
    return done + open_


def fit_tree(col, g, rng):
    """col (n, 3) column points, g survey xy -> (result dict, layers)."""
    if len(col) < MIN_COL:
        return dict(status="sparse", n_trunks=0), []
    z0 = float(col[:, 2].min())
    col = col[col[:, 2] < z0 + SCAN_H]
    li = np.floor((col[:, 2] - z0) / LAYER).astype(int)
    layers = [(k, z0 + k * LAYER, layer_circles(col[li == k, :2], rng))
              for k in range(int(li.max()) + 1)]
    ch = chains(layers)
    if not ch:
        return dict(status="no_fit", n_trunks=0), layers

    def centre(c):
        return np.median([[x["cx"], x["cy"]] for _, _, x in c], axis=0)
    trunks = [c for c in ch if len(c) >= RUN]
    pool = trunks if trunks else ch
    dist = np.array([np.hypot(*(centre(c) - g)) for c in pool])
    order = np.argsort(dist)
    best = pool[int(order[0])]
    c = centre(best)
    zb = best[0][1]
    rad = float(np.median([x["r"] for _, _, x in best]))
    bh = [x["r"] for _, z, x in best if BH[0] <= z - zb + LAYER / 2 <= BH[1]]
    return dict(status="ok" if trunks else "weak",
                x=float(c[0]), y=float(c[1]), z=float(zb), radius=rad,
                r_bh=float(np.median(bh)) if bh else np.nan, n_bh=len(bh),
                run_layers=len(best), n_trunks=len(trunks),
                second_m=float(dist[order[1]]) if len(order) > 1 else np.nan,
                rms=float(np.median([x["rms"] for _, _, x in best])),
                arc=float(np.median([x["arc"] for _, _, x in best]))), layers


# ---------------------------------------------------------------------------
def column_tiles(xy):
    lo = np.floor((xy - SEARCH_R) / TILE).astype(int)
    hi = np.floor((xy + SEARCH_R) / TILE).astype(int)
    return [{(a, b) for a in range(lo[i, 0], hi[i, 0] + 1) for b in range(lo[i, 1], hi[i, 1] + 1)}
            for i in range(len(xy))]


def attributes():
    """lat/lon -> (girth string, species). The attribute rows are the stem
    map's own rows, so they join on position to 1e-8 degrees."""
    out = {}
    if not ATTRS.exists():
        return out
    with open(ATTRS) as f:
        for r in csv.DictReader(f):
            out[(round(float(r["lat"]), 8), round(float(r["lon"]), 8))] = (
                (r.get("girth_cm") or "").strip(), (r.get("species") or "").strip())
    return out


def fork_groups(xy):
    from scipy.spatial import cKDTree
    parent = np.arange(len(xy))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in cKDTree(xy).query_pairs(MERGE_R):
        parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(len(xy))])
    _, g = np.unique(roots, return_inverse=True)
    return g


def girth_radius(g):
    """'29;22' -> thickest stem, as a radius in metres."""
    try:
        return max(float(v) for v in g.split(";") if v.strip()) / (2 * np.pi) / 100
    except ValueError:
        return np.nan


def offset_stats(v):
    """Survey -> trunk vectors (n, 2): median vector, spread, alignment."""
    if len(v) < 3:
        return {}
    d = np.linalg.norm(v, axis=1)
    u = v / np.maximum(d[:, None], 1e-9)
    med = np.median(v, axis=0)
    resid = np.linalg.norm(v - med, axis=1)
    return dict(n=int(len(v)), median_dx_cm=float(med[0] * 100), median_dy_cm=float(med[1] * 100),
                median_shift_cm=float(np.hypot(*med) * 100),
                direction_deg=float(np.degrees(np.arctan2(med[1], med[0]))),
                alignment_R=float(np.linalg.norm(u.mean(0))),
                distance_p50_cm=float(np.median(d) * 100),
                residual_after_median_p50_cm=float(np.median(resid) * 100),
                residual_after_median_p90_cm=float(np.percentile(resid, 90) * 100))


def figures(rows, examples, gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r["status"] == "ok" and np.isfinite(r.get("r_bh", np.nan))
          and np.isfinite(r["r_girth"]) and ";" not in r["girth_cm"]]
    fig, ax = plt.subplots(figsize=(7, 7))
    if ok:
        a = np.array([[r["r_girth"] * 100, r["r_bh"] * 100] for r in ok])
        ax.scatter(a[:, 0], a[:, 1], s=10)
        m = max(a.max(), 1)
        ax.plot([0, m], [0, m], "k--", lw=0.8, label="fitted = surveyed")
        ax.legend()
    ax.set_xlabel("surveyed radius, girth / 2 pi (cm)"); ax.set_ylabel("fitted radius at 1.0-1.6 m (cm)")
    ax.set_title(f"circle fit vs survey girth, {len(ok)} single-stem trees")
    fig.tight_layout(); fig.savefig(RESULTS / "42_stemnet_v2_girth.png", dpi=120); plt.close(fig)

    v = np.array([[r["x"] - r["rtk_x"], r["y"] - r["rtk_y"], r["rtk_x"], r["rtk_y"]]
                  for r in rows if r["status"] == "ok"])
    if len(v):
        fig, ax = plt.subplots(1, 2, figsize=(18, 8.5))
        ax[0].quiver(v[:, 2], v[:, 3], v[:, 0], v[:, 1], np.hypot(v[:, 0], v[:, 1]) * 100,
                     angles="xy", scale_units="xy", scale=1 / 15, width=0.0015, cmap="viridis")
        ax[0].set_aspect("equal"); ax[0].set_title("survey point -> trunk (arrows x15)")
        ax[1].scatter(v[:, 0] * 100, v[:, 1] * 100, s=3, alpha=0.4)
        med = np.median(v[:, :2], axis=0) * 100
        ax[1].plot(*med, "r+", ms=18, mew=3, label=f"median ({med[0]:+.0f}, {med[1]:+.0f}) cm")
        ax[1].axhline(0, c="k", lw=0.5); ax[1].axvline(0, c="k", lw=0.5)
        ax[1].set_xlim(-SEARCH_R * 100, SEARCH_R * 100); ax[1].set_ylim(-SEARCH_R * 100, SEARCH_R * 100)
        ax[1].set_aspect("equal"); ax[1].legend()
        ax[1].set_xlabel("dx east (cm)"); ax[1].set_ylabel("dy north (cm)")
        ax[1].set_title("every offset, survey point at the origin")
        fig.tight_layout(); fig.savefig(RESULTS / "45_stemnet_v2_offsets.png", dpi=110); plt.close(fig)

    n = len(examples)
    if not n:
        return
    fig, axs = plt.subplots(3, (n + 2) // 3, figsize=(4 * ((n + 2) // 3), 12), squeeze=False)
    for a, (i, col, layers) in zip(axs.ravel(), examples):
        r = rows[i]
        g = gt[i, :2]
        if len(col):
            a.scatter(col[:, 0] - g[0], col[:, 1] - g[1], s=0.4, c=col[:, 2] - col[:, 2].min(),
                      cmap="viridis", lw=0)
        for _, _, circs in layers:
            for f in circs:
                a.add_patch(plt.Circle((f["cx"] - g[0], f["cy"] - g[1]), f["r"],
                                       fill=False, lw=0.5, color="tab:orange"))
        a.plot(0, 0, "r+", ms=12, mew=2, label="survey point")
        if r["status"] in ("ok", "weak"):
            a.add_patch(plt.Circle((r["x"] - g[0], r["y"] - g[1]), r["radius"], fill=False,
                                   lw=1.6, color="k"))
            a.plot(r["x"] - g[0], r["y"] - g[1], "kx", ms=9, mew=2, label="derived base")
        a.set_xlim(-SEARCH_R, SEARCH_R); a.set_ylim(-SEARCH_R, SEARCH_R); a.set_aspect("equal")
        rad = r.get("radius", np.nan)
        a.set_title(f"{r['id']} {r['status']} r={rad * 100:.0f} cm trunks={r.get('n_trunks', 0)}"
                    + (f" girth-r={r['r_girth'] * 100:.0f}" if np.isfinite(r["r_girth"]) else ""),
                    fontsize=8)
    axs.ravel()[0].legend(fontsize=7)
    for a in axs.ravel()[n:]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / "43_stemnet_v2_circle_examples.png", dpi=110)
    plt.close(fig)


def main():
    from scipy.spatial import cKDTree
    if not FRAME_JSON.exists():
        raise SystemExit("[v2-circle] no frame.json — run v2_grid first")
    frame = json.loads(FRAME_JSON.read_text())
    if f"{GRID:g}" not in frame.get("grids", {}):
        raise SystemExit(f"[v2-circle] no {GRID:g} m grid — run v2_grid first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(os.environ.get("V2_SEED", "0")))

    sm = parse_stem_map(STEM_MAP)
    gt = survey_to_local(sm, frame["epsg"], frame["origin_m"])
    log(f"{len(gt):,} survey rows in {len(set(sm.block.tolist()))} blocks")
    have = {tuple(int(v) for v in q.stem.split("_")[1:]) for q in GRID_DIR.glob("tile_*.npz")}
    need = column_tiles(gt[:, :2])
    if SUBSET:
        cov = np.array([n <= have for n in need])
        log(f"!! SUBSET RUN on {len(have)} tiles: fitting the {int(cov.sum())} trees whose "
            f"whole column is covered; {int((~cov).sum()):,} others are not in this run")
        k = np.flatnonzero(cov)
        sm = type(sm)(sm.block[k], sm.n[k], sm.lat[k], sm.lon[k], sm.z[k], sm.stations)
        gt = gt[k]
        need = [need[i] for i in k]
    attrs = attributes()

    # stream the tiles; fit each tree the moment its last tile has been read,
    # so only the columns in progress are ever in memory
    left = [len(n & have) for n in need]
    by_tile: dict = {}
    for i, n in enumerate(need):
        for q in n & have:
            by_tile.setdefault(q, []).append(i)
    parts = [[] for _ in range(len(gt))]
    plot_ids = set(np.random.default_rng(3).choice(len(gt), size=min(KEEP_FOR_PLOT, len(gt)),
                                                   replace=False).tolist())
    kept_cols, results = {}, [None] * len(gt)

    def finish(i):
        col = np.concatenate(parts[i]) if parts[i] else np.zeros((0, 3))
        parts[i] = None
        results[i] = fit_tree(col, gt[i, :2], rng)
        if i in plot_ids:
            kept_cols[i] = col

    for i in range(len(gt)):                              # trees in no existing tile
        if left[i] == 0:
            finish(i)
    t0 = time.time()
    tiles = sorted(by_tile)
    done = 0
    for k, q in enumerate(tiles):
        P = np.load(GRID_DIR / f"tile_{q[0]}_{q[1]}.npz")["xyz"].astype(np.float64)
        ids = by_tile[q]
        if len(P):
            hits = cKDTree(P[:, :2]).query_ball_point(gt[ids, :2], r=SEARCH_R)
            for i, h in zip(ids, hits):
                if h:
                    parts[i].append(P[h])
        for i in ids:
            left[i] -= 1
            if left[i] == 0:
                finish(i)
                done += 1
        if (k + 1) % 25 == 0 or k + 1 == len(tiles):
            el = time.time() - t0
            log(f"tile {k + 1}/{len(tiles)} | {done:,}/{len(gt):,} trees fitted | "
                f"{el / 60:.1f} min, eta {el / (k + 1) * (len(tiles) - k - 1) / 60:.1f} min")

    rows = []
    for i in range(len(gt)):
        res, _ = results[i]
        g_str, species = attrs.get((round(float(sm.lat[i]), 8), round(float(sm.lon[i]), 8)), ("", ""))
        row = dict(id=f"{int(sm.block[i])}-{sm.n[i]}", block=int(sm.block[i]), n=sm.n[i],
                   rtk_x=gt[i, 0], rtk_y=gt[i, 1], rtk_z=gt[i, 2],
                   species=species, girth_cm=g_str, r_girth=girth_radius(g_str) if g_str else np.nan)
        row.update(res)
        if row["status"] not in ("ok", "weak"):
            row.update(x=gt[i, 0], y=gt[i, 1], z=np.nan, radius=np.nan, r_bh=np.nan)
        row["shift_m"] = float(np.hypot(row["x"] - gt[i, 0], row["y"] - gt[i, 1]))
        rows.append(row)

    groups = fork_groups(np.array([[r["x"], r["y"]] for r in rows]))
    for r, gid in zip(rows, groups):
        r["fork_group"] = int(gid)

    cols_out = ["id", "block", "n", "rtk_x", "rtk_y", "rtk_z", "x", "y", "z", "shift_m", "radius",
                "r_bh", "n_bh", "status", "run_layers", "n_trunks", "second_m", "rms", "arc",
                "fork_group", "species", "girth_cm", "r_girth"]
    with open(OUT / "bases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else
                            ("" if isinstance(v, float) else v)) for k, v in r.items()})
    with open(OUT / "bases_merged.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fork_group", "x", "y", "members", "status_best", "member_ids"])
        rank = {"ok": 0, "weak": 1, "no_fit": 2, "sparse": 3}
        for gid in np.unique(groups):
            mem = [r for r in rows if r["fork_group"] == gid]
            w.writerow([int(gid), f"{np.mean([r['x'] for r in mem]):.4f}",
                        f"{np.mean([r['y'] for r in mem]):.4f}", len(mem),
                        min((r["status"] for r in mem), key=rank.get),
                        ";".join(r["id"] for r in mem)])

    # ---- report ---------------------------------------------------------
    st = {s: sum(r["status"] == s for r in rows) for s in ("ok", "weak", "no_fit", "sparse")}
    okr = [r for r in rows if r["status"] == "ok"]
    L = ["=" * 72, "StemNet v2 — derived stem bases (trunks found near the survey point)"
         + ("  [SUBSET RUN]" if SUBSET else ""), "=" * 72,
         f"survey rows     {len(rows):,}  (all kept) | search radius {SEARCH_R} m",
         "status          " + "  ".join(f"{k} {v:,} ({v / len(rows) * 100:.1f}%)" for k, v in st.items()),
         f"fork groups     {len(np.unique(groups)):,} after merging derived bases < {MERGE_R} m apart"]
    if okr:
        rad = np.array([r["radius"] for r in okr]) * 100
        nt = np.array([r["n_trunks"] for r in okr])
        sec = np.array([r["second_m"] for r in okr])
        L += [f"fitted radius (ok), cm   p10 {np.percentile(rad, 10):.1f}  p50 {np.median(rad):.1f}  "
              f"p90 {np.percentile(rad, 90):.1f}",
              f"trunks in the column     1: {int((nt == 1).sum())}  2: {int((nt == 2).sum())}  "
              f"3+: {int((nt >= 3).sum())}"
              + (f"  | 2nd-nearest trunk p50 {np.nanmedian(sec):.2f} m" if np.isfinite(sec).any() else "")]
    val = [(r["r_bh"], r["r_girth"]) for r in okr
           if np.isfinite(r["r_bh"]) and np.isfinite(r["r_girth"]) and ";" not in r["girth_cm"]]
    if val:
        v = np.array(val) * 100
        err = v[:, 0] - v[:, 1]
        L += ["", f"-- girth validation: {len(v)} single-stem surveyed trees with an ok fit --",
              f"fitted r at {BH[0]}-{BH[1]} m minus girth/2pi, cm:  p10 {np.percentile(err, 10):+.1f}  "
              f"p50 {np.median(err):+.1f}  p90 {np.percentile(err, 90):+.1f}",
              f"|error| median {np.median(np.abs(err)):.1f} cm | within 2 cm: "
              f"{(np.abs(err) <= 2).mean() * 100:.0f}% | within 5 cm: {(np.abs(err) <= 5).mean() * 100:.0f}%"]
    att = [r for r in rows if r["girth_cm"]]
    if att:
        L.append("attributed trees by status: " + "  ".join(
            f"{s} {sum(r['status'] == s for r in att)}" for s in st))
    offs = {}
    if okr:
        L += ["", "-- survey point -> trunk offset (ok trees) --",
              "   R: direction alignment, 0 = random, 1 = all the same way",
              "   residual: how far each offset is from the group's median vector"]
        sets = {"ALL": okr}
        for b in sorted({r["block"] for r in okr}):
            sets[f"block {b}"] = [r for r in okr if r["block"] == b]
        for name, rr in sets.items():
            s = offset_stats(np.array([[r["x"] - r["rtk_x"], r["y"] - r["rtk_y"]] for r in rr]))
            if not s:
                continue
            offs[name] = s
            L.append(f"   {name:8s} n={s['n']:5d}  median ({s['median_dx_cm']:+5.1f}, {s['median_dy_cm']:+5.1f}) cm"
                     f" = {s['median_shift_cm']:4.1f} cm @ {s['direction_deg']:+4.0f} deg  R={s['alignment_R']:.2f}"
                     f"  |offset| p50 {s['distance_p50_cm']:4.1f}  residual p50 {s['residual_after_median_p50_cm']:4.1f}"
                     f" p90 {s['residual_after_median_p90_cm']:4.1f} cm")
    txt = "\n".join(L)
    (RESULTS / "stemnet_v2_circles.txt").write_text(txt + "\n")
    print(txt, flush=True)

    rng2 = np.random.default_rng(1)
    ex = []
    for s, k in (("ok", 6), ("weak", 3), ("no_fit", 3)):
        ids = [i for i in kept_cols if rows[i]["status"] == s]
        ex += [int(i) for i in rng2.choice(ids, size=min(k, len(ids)), replace=False)] if ids else []
    figures(rows, [(i, kept_cols[i], results[i][1]) for i in ex], gt)

    frame["bases"] = dict(file="bases.csv", merged="bases_merged.csv", status=st, offsets=offs,
                          subset=SUBSET,
                          params=dict(search_r=SEARCH_R, scan_h=SCAN_H, layer=LAYER, gap=GAP,
                                      r_min=R_MIN, r_max=R_MAX, arc=ARC, min_in=MIN_IN, in_tol=IN_TOL,
                                      run=RUN, run_tol=RUN_TOL, merge_r=MERGE_R, inside=INSIDE,
                                      small_r=SMALL_R, ransac=ITERS))
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log("-> data/stemnet_v2/bases.csv, bases_merged.csv | results/stemnet_v2_circles.txt, "
        "42_/43_/45_ png")


if __name__ == "__main__":
    main()
