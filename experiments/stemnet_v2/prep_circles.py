#!/usr/bin/env python3
"""StemNet v2 prep, part 3: the stem axis of every surveyed tree — trunks found
in the cloud, survey sessions registered onto them, then matched.

What the real data showed, and why this stage has three steps:
  - the trunk finder works: fitted radii against the surveyed girths, median
    error -0.3 cm, 87% within 5 cm
  - but survey and scan disagree by ONE OFFSET PER SURVEY SESSION. In the same
    patch of ground, block 1's survey points sit ~1.2 m south-west of their
    trunks while block 2's sit on them (~4 cm). RTK is precise relative to its
    base station; each session's absolute position rides on the coordinates
    entered for that base, and that error moves the whole session together
  - with a ~1.1 m offset and several stems within 1.5 m, "nearest trunk" picks
    the wrong stem; the median offset said 51 cm while unambiguous columns said
    ~1.1 m

  1. DISCOVER  every trunk within V2_DISCOVER_R (2.5 m) of any survey point, once.
               The area is cut into V2_CELL (2 m) cells, each read with a
               V2_HALO (0.7 m) halo; a trunk belongs to the cell holding its
               centre, so nothing is found twice. Per cell:
                 layers   the lowest V2_SCAN_H (8 m), V2_LAYER (0.2 m) slices
                 clusters split where gaps exceed V2_GAP (5 cm); clusters wider
                          than 2 x V2_R_MAX + 10 cm (ground, crown) skipped
                 circle   RANSAC + refit; radius V2_R_MIN..V2_R_MAX (2-60 cm),
                          arc >= V2_ARC (90 deg), >= V2_MIN_IN (12) inliers, and
                          HOLLOW against all the layer's points
                 trunk    circles chained across layers within V2_RUN_TOL
                          (10 cm), bridging up to V2_MAX_GAP (2) failed layers;
                          >= V2_RUN (3) layers is a trunk, 2 a weak candidate
  2. REGISTER  per survey block, the horizontal shift that puts the most survey
               points ON a trunk's bark (|distance to axis - radius| small):
               a coarse search over +-V2_REG_R (2 m) at 5 cm, refined at 1 cm.
               Applied only when the peak is a single clear answer: >=
               V2_REG_SHARP (6) x the median score AND the best score more than
               30 cm away <= V2_REG_SECOND (0.6) of the peak, with >= V2_REG_MIN
               survey points. Sharpness alone is not enough on a crowded campus:
               survey points with no relation to the trunks still reach ~3.5x by
               chance, but their runner-up is 0.87-0.99 of the peak, while a real
               shift's runner-up stays <= 0.4 even when only a third of the
               points match. Otherwise the block keeps zero shift and is flagged.
               Blocks too small, or with no single answer, take the shift pooled
               over the clean blocks (on the real scan all blocks peaked at the
               same ~1.2 m NE, and what is left after it is per-point scatter
               uncorrelated between neighbours: one survey-wide offset).
  3. MATCH     one-to-one, best first: each registered survey point and each
               trunk used once, smallest |distance - radius| within
               V2_MATCH_TOL (40 cm), full trunks before weak ones.

The label is always the trunk axis from the cloud. The shift only decides
which trunk a survey row points at; it never moves a label. Survey heights are
never used. Every survey row is kept:
  ok      matched to a trunk (>= 3 layers)
  weak    matched to a 2-layer candidate only
  no_fit  no trunk bark within tolerance; x, y = the registered survey point

Audit, all in the report:
  - per block: shift, how many survey points it puts on bark, peak sharpness
  - residual |distance - radius| after matching: a survey point should sit on
    the bark, so this should be centimetres if registration is right
  - girth: fitted radius 1.0-1.6 m up the trunk vs girth / 2 pi
  - trunks no survey row claims (unsurveyed trees, or posts) and trunks two
    rows claim

Outputs:
  data/stemnet_v2/trunks.csv         every trunk discovered
  data/stemnet_v2/bases.csv          one row per survey row
  data/stemnet_v2/bases_merged.csv   one row per fork group, for training
  results/stemnet_v2_circles.txt     the audit
  results/42_stemnet_v2_girth.png    fitted vs surveyed radius
  results/43_stemnet_v2_circle_examples.png
  results/45_stemnet_v2_registration.png   score surface per block, residuals
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
DISCOVER_R = float(os.environ.get("V2_DISCOVER_R", "2.5"))
CELL = float(os.environ.get("V2_CELL", "2.0"))
HALO = float(os.environ.get("V2_HALO", "0.7"))
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
MAX_GAP = int(os.environ.get("V2_MAX_GAP", "2"))       # failed layers a trunk may bridge
ITERS = int(os.environ.get("V2_RANSAC", "200"))
INSIDE = float(os.environ.get("V2_INSIDE", "0.25"))     # max density inside / density on the ring
SMALL_R = float(os.environ.get("V2_SMALL_R", "0.06"))   # below this there is no inside to test
SHELL = 0.05                                            # outside band for small circles, m
REG_R = float(os.environ.get("V2_REG_R", "2.0"))
REG_SHARP = float(os.environ.get("V2_REG_SHARP", "6.0"))
REG_SECOND = float(os.environ.get("V2_REG_SECOND", "0.6"))
REG_MIN = int(os.environ.get("V2_REG_MIN", "20"))
MATCH_TOL = float(os.environ.get("V2_MATCH_TOL", "0.40"))
MERGE_R = float(os.environ.get("V2_MERGE_R", "0.3"))
BH = (1.0, 1.6)
SUBSET = os.environ.get("V2_SUBSET", "0") != "0"
N_PLOT = 200
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


def chains(layers):
    """Link circles across layers whose centres agree within RUN_TOL. A chain
    may bridge up to MAX_GAP layers with no accepted circle: a single layer
    that fails its tests (a branch stub, an occluded band) must not split one
    trunk into two — the first real run found 729 such split pairs, typically
    1.4 m apart vertically. layers: list of (k, z, [circles])."""
    done, open_ = [], []
    for k, z, circs in layers:
        alive = [ch for ch in open_ if k - ch[-1][0] <= MAX_GAP + 1]
        done.extend(ch for ch in open_ if k - ch[-1][0] > MAX_GAP + 1)
        nxt, used = [], set()
        for ch in alive:
            m = np.median([[c["cx"], c["cy"]] for _, _, c in ch], axis=0)
            best, bd = None, RUN_TOL
            for j, c in enumerate(circs):
                if j in used:
                    continue
                d = np.hypot(c["cx"] - m[0], c["cy"] - m[1])
                if d <= bd:
                    best, bd = j, d
            if best is None:
                nxt.append(ch)                       # carried over the gap
            else:
                used.add(best)
                nxt.append(ch + [(k, z, circs[best])])
        for j, c in enumerate(circs):
            if j not in used:
                nxt.append([(k, z, c)])
        open_ = nxt
    return done + open_


def summarise(ch):
    c = np.median([[x["cx"], x["cy"]] for _, _, x in ch], axis=0)
    zb = ch[0][1]
    bh = [x["r"] for _, z, x in ch if BH[0] <= z - zb + LAYER / 2 <= BH[1]]
    return dict(x=float(c[0]), y=float(c[1]), z=float(zb),
                radius=float(np.median([x["r"] for _, _, x in ch])),
                r_bh=float(np.median(bh)) if bh else np.nan, n_layers=len(ch),
                rms=float(np.median([x["rms"] for _, _, x in ch])),
                arc=float(np.median([x["arc"] for _, _, x in ch])))


def cell_trunks(P, x0, y0, rng):
    """Every trunk (>= 2 layers) whose centre lies in the cell [x0, x0+CELL) x
    [y0, y0+CELL); P holds the cell's points plus its halo."""
    if len(P) < MIN_IN:
        return [], []
    z0 = float(P[:, 2].min())
    P = P[P[:, 2] < z0 + SCAN_H]
    li = np.floor((P[:, 2] - z0) / LAYER).astype(int)
    layers = [(k, z0 + k * LAYER, layer_circles(P[li == k, :2], rng))
              for k in range(int(li.max()) + 1)]
    out = []
    for ch in chains(layers):
        if len(ch) < 2:
            continue
        s = summarise(ch)
        if x0 <= s["x"] < x0 + CELL and y0 <= s["y"] < y0 + CELL:
            out.append(s)
    return out, layers


# ---------------------------------------------------------------------------
# registration and matching
# ---------------------------------------------------------------------------
def bark_score(S, kd, R, d, tol):
    """Soft count of survey points S + d lying on a trunk's bark."""
    dist, j = kd.query(S + d)
    e = np.abs(dist - R[j])
    return float(np.clip(1.0 - e / tol, 0, None).sum())


def register(S, T, R):
    """Shift that puts the most survey points S on trunk bark. Returns the
    shift, the coarse score surface, and how clear the peak is."""
    from scipy.spatial import cKDTree
    kd = cKDTree(T)
    g = np.arange(-REG_R, REG_R + 1e-9, 0.05)
    surf = np.array([[bark_score(S, kd, R, np.array([dx, dy]), 0.15) for dx in g] for dy in g])
    j, i = np.unravel_index(np.argmax(surf), surf.shape)
    best = np.array([g[i], g[j]])
    fine = np.arange(-0.1, 0.1 + 1e-9, 0.01)
    # the fine step scores with the same 15 cm tolerance as the coarse one:
    # the survey scatters 10-35 cm around the shift, and a tolerance narrower
    # than that makes the refinement chase individual points instead of the
    # bulk (a 6 cm version landed 9 cm off on 18 cm scatter)
    fs = [(bark_score(S, kd, R, best + [a, b], 0.15), a, b) for a in fine for b in fine]
    _, a, b = max(fs)
    shift = best + [a, b]
    med = float(np.median(surf))
    # a competing answer is a SEPARATE local maximum at least 50 cm away. Not
    # "the best score more than 30 cm away": on the real scan the peak is
    # broad (per-point scatter of 10-35 cm), so that test read the shoulder of
    # the one peak as a rival and rejected three blocks whose surfaces had a
    # single clear maximum
    from scipy import ndimage
    G = np.stack(np.meshgrid(g, g), -1)
    loc = (surf == ndimage.maximum_filter(surf, size=5, mode="nearest")) & (surf > 0)
    far = loc & (np.hypot(G[..., 0] - best[0], G[..., 1] - best[1]) >= 0.5)
    second = float(surf[far].max()) if far.any() else 0.0
    dist, jj = kd.query(S + shift)
    on_bark = float((np.abs(dist - R[jj]) <= 0.08).mean())
    return dict(shift=shift, surface=surf, grid=g, peak=float(surf.max()), median=med,
                sharpness=float(surf.max() / max(med, 1e-9)),
                second_ratio=float(second / max(surf.max(), 1e-9)), on_bark=on_bark)


def match(S, trunks):
    """One-to-one: every (survey point, trunk) pair whose bark is within
    MATCH_TOL of the point, taken best-first (smallest |distance - radius|),
    each trunk and each point used once. Full trunks are assigned before
    2-layer candidates. One-to-one because the per-point scatter left after
    registration (10-35 cm) is comparable to the spacing of stems in a dense
    stand: a point allowed to take any nearby trunk would take a neighbour's."""
    from scipy.spatial import cKDTree
    out = [None] * len(S)
    taken = set()
    for tier in (True, False):
        idx = [k for k, t in enumerate(trunks) if (t["n_layers"] >= RUN) == tier]
        if not idx:
            continue
        T = np.array([[trunks[k]["x"], trunks[k]["y"]] for k in idx])
        R = np.array([trunks[k]["radius"] for k in idx])
        kd = cKDTree(T)
        cand = []
        for i, s in enumerate(S):
            if out[i] is not None:
                continue
            for q in kd.query_ball_point(s, R_MAX + MATCH_TOL):
                d = float(np.hypot(*(T[q] - s)))
                e = abs(d - R[q])
                if e <= MATCH_TOL:
                    cand.append((e, i, q, d))
        for e, i, q, d in sorted(cand):
            if out[i] is None and idx[q] not in taken:
                out[i] = (idx[q], d - R[q], d)
                taken.add(idx[q])
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
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


def needed_cells(xy):
    """Cells within DISCOVER_R of any survey point."""
    cells = set()
    k = int(np.ceil(DISCOVER_R / CELL))
    base = np.floor(xy / CELL).astype(int)
    for (a, b), (x, y) in zip(base, xy):
        for i in range(a - k, a + k + 1):
            for j in range(b - k, b + k + 1):
                # nearest point of the cell to the survey point
                nx = min(max(x, i * CELL), (i + 1) * CELL)
                ny = min(max(y, j * CELL), (j + 1) * CELL)
                if np.hypot(nx - x, ny - y) <= DISCOVER_R:
                    cells.add((i, j))
    return sorted(cells)


def tiles_of_box(x0, x1, y0, y1):
    return {(a, b) for a in range(int(np.floor(x0 / TILE)), int(np.floor(x1 / TILE)) + 1)
            for b in range(int(np.floor(y0 / TILE)), int(np.floor(y1 / TILE)) + 1)}


# ---------------------------------------------------------------------------
def figures(rows, trunks, reg, examples):
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
    ax.set_title(f"matched trunk vs survey girth, {len(ok)} single-stem trees")
    fig.tight_layout(); fig.savefig(RESULTS / "42_stemnet_v2_girth.png", dpi=120); plt.close(fig)

    blocks = [b for b in reg if "surface" in reg[b]]
    if blocks:
        n = len(blocks) + 1
        fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
        for a, b in zip(axs[0], blocks):
            R = reg[b]
            g = R["grid"] * 100
            a.imshow(R["surface"], origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap="magma")
            a.plot(R["shift"][0] * 100, R["shift"][1] * 100, "c+", ms=14, mew=2)
            a.set_title(f"block {b}: ({R['shift'][0] * 100:+.0f}, {R['shift'][1] * 100:+.0f}) cm\n"
                        f"sharp x{R['sharpness']:.1f}, on bark {R['on_bark'] * 100:.0f}%"
                        + ("" if R["applied"] else " NOT APPLIED"), fontsize=9)
            a.set_xlabel("dx cm"); a.set_ylabel("dy cm")
        res = np.array([r["resid_m"] for r in rows if r["status"] in ("ok", "weak")]) * 100
        axs[0][-1].hist(np.clip(res, -25, 25), bins=50)
        axs[0][-1].set_title("survey point - bark after matching (cm)")
        fig.tight_layout(); fig.savefig(RESULTS / "45_stemnet_v2_registration.png", dpi=110)
        plt.close(fig)

    n = len(examples)
    if not n:
        return
    T = np.array([[t["x"], t["y"], t["radius"]] for t in trunks]) if trunks else np.zeros((0, 3))
    fig, axs = plt.subplots(3, (n + 2) // 3, figsize=(4 * ((n + 2) // 3), 12), squeeze=False)
    for a, (i, col) in zip(axs.ravel(), examples):
        r = rows[i]
        g = np.array([r["reg_x"], r["reg_y"]])
        if len(col):
            a.scatter(col[:, 0] - g[0], col[:, 1] - g[1], s=0.4, c=col[:, 2] - col[:, 2].min(),
                      cmap="viridis", lw=0)
        near = T[np.hypot(T[:, 0] - g[0], T[:, 1] - g[1]) <= DISCOVER_R] if len(T) else T
        for x, y, rad in near:
            a.add_patch(plt.Circle((x - g[0], y - g[1]), rad, fill=False, lw=0.8, color="tab:orange"))
        a.plot(r["rtk_x"] - g[0], r["rtk_y"] - g[1], "r+", ms=10, mew=1.5, label="survey (raw)")
        a.plot(0, 0, "m+", ms=12, mew=2, label="survey (registered)")
        if r["status"] in ("ok", "weak"):
            a.add_patch(plt.Circle((r["x"] - g[0], r["y"] - g[1]), r["radius"], fill=False, lw=1.8, color="k"))
            a.plot(r["x"] - g[0], r["y"] - g[1], "kx", ms=9, mew=2, label="matched trunk")
        a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_aspect("equal")
        a.set_title(f"{r['id']} {r['status']} r={r.get('radius', np.nan) * 100:.0f} cm "
                    f"resid={r.get('resid_m', np.nan) * 100:+.0f}"
                    + (f" girth-r={r['r_girth'] * 100:.0f}" if np.isfinite(r["r_girth"]) else ""), fontsize=8)
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
    if SUBSET:
        reach = DISCOVER_R + REG_R
        cov = np.array([tiles_of_box(x - reach, x + reach, y - reach, y + reach) <= have
                        for x, y in gt[:, :2]])
        log(f"!! SUBSET RUN on {len(have)} tiles: {int(cov.sum())} trees fully covered, "
            f"{int((~cov).sum()):,} others not in this run")
        k = np.flatnonzero(cov)
        sm = type(sm)(sm.block[k], sm.n[k], sm.lat[k], sm.lon[k], sm.z[k], sm.stations)
        gt = gt[k]
    attrs = attributes()

    # ---- 1. discover ----------------------------------------------------
    cells = needed_cells(gt[:, :2])
    need = {c: tiles_of_box(c[0] * CELL - HALO, (c[0] + 1) * CELL + HALO,
                            c[1] * CELL - HALO, (c[1] + 1) * CELL + HALO) & have for c in cells}
    plot_ids = set(np.random.default_rng(3).choice(len(gt), size=min(N_PLOT, len(gt)),
                                                   replace=False).tolist())
    PR = 1.5 + REG_R                      # the registered point can land anywhere within REG_R
    plot_need = {i: tiles_of_box(gt[i, 0] - PR, gt[i, 0] + PR,
                                 gt[i, 1] - PR, gt[i, 1] + PR) & have for i in plot_ids}
    by_tile: dict = {}
    for c, ts in need.items():
        for q in ts:
            by_tile.setdefault(q, []).append(("c", c))
    for i, ts in plot_need.items():
        for q in ts:
            by_tile.setdefault(q, []).append(("p", i))
    left = {("c", c): len(ts) for c, ts in need.items()}
    left.update({("p", i): len(ts) for i, ts in plot_need.items()})
    parts: dict = {}
    trunks, plot_cols = [], {}
    log(f"discovery: {len(cells):,} cells of {CELL:g} m within {DISCOVER_R:g} m of a survey point")

    def finish(key):
        P = np.concatenate(parts.pop(key)) if key in parts else np.zeros((0, 3))
        if key[0] == "c":
            c = key[1]
            ts, _ = cell_trunks(P, c[0] * CELL, c[1] * CELL, rng)
            trunks.extend(ts)
        else:
            plot_cols[key[1]] = P

    for key, n in list(left.items()):
        if n == 0:
            finish(key)
    t0 = time.time()
    tiles = sorted(by_tile)
    for k, q in enumerate(tiles):
        P = np.load(GRID_DIR / f"tile_{q[0]}_{q[1]}.npz")["xyz"].astype(np.float64)
        jobs = by_tile[q]
        if len(P):
            order = np.argsort(P[:, 0], kind="stable")
            xs = P[order, 0]
            for key in jobs:
                if key[0] == "c":
                    c = key[1]
                    x0, x1 = c[0] * CELL - HALO, (c[0] + 1) * CELL + HALO
                    y0, y1 = c[1] * CELL - HALO, (c[1] + 1) * CELL + HALO
                else:
                    i = key[1]
                    x0, x1 = gt[i, 0] - PR, gt[i, 0] + PR
                    y0, y1 = gt[i, 1] - PR, gt[i, 1] + PR
                lo, hi = np.searchsorted(xs, [x0, x1])
                ix = order[lo:hi]
                m = (P[ix, 1] >= y0) & (P[ix, 1] < y1)
                if m.any():
                    parts.setdefault(key, []).append(P[ix[m]])
        for key in jobs:
            left[key] -= 1
            if left[key] == 0:
                finish(key)
        if (k + 1) % 25 == 0 or k + 1 == len(tiles):
            el = time.time() - t0
            log(f"discovery: tile {k + 1}/{len(tiles)} | {len(trunks):,} trunk candidates | "
                f"{el / 60:.1f} min, eta {el / (k + 1) * (len(tiles) - k - 1) / 60:.1f} min")
    for t, tr in enumerate(trunks):
        tr["id"] = t
    full = [t for t in trunks if t["n_layers"] >= RUN]
    log(f"discovered {len(full):,} trunks (>= {RUN} layers) + {len(trunks) - len(full):,} 2-layer candidates")

    # ---- 2. register, per survey block ------------------------------------
    T = np.array([[t["x"], t["y"]] for t in full]) if full else np.zeros((0, 2))
    R = np.array([t["radius"] for t in full]) if full else np.zeros(0)
    reg, shift = {}, np.zeros((len(gt), 2))
    for b in sorted(set(sm.block.tolist())):
        k = sm.block == b
        if k.sum() < REG_MIN or len(T) < REG_MIN:
            reg[b] = dict(n=int(k.sum()), applied=False, why=f"fewer than {REG_MIN} survey points",
                          shift=np.zeros(2))
            continue
        R_ = register(gt[k, :2], T, R)
        ok = R_["sharpness"] >= REG_SHARP and R_["second_ratio"] <= REG_SECOND
        R_.update(n=int(k.sum()), applied=bool(ok),
                  why="" if ok else (f"peak x{R_['sharpness']:.1f} the median, runner-up "
                                     f"{R_['second_ratio']:.2f} of the peak — no single answer"))
        reg[b] = R_
        if ok:
            shift[k] = R_["shift"]
        log(f"block {b}: shift ({R_['shift'][0] * 100:+.1f}, {R_['shift'][1] * 100:+.1f}) cm, "
            f"sharpness x{R_['sharpness']:.1f}, {R_['on_bark'] * 100:.0f}% on bark"
            + ("" if ok else "  -> NOT applied"))
    # blocks too small to register, or with no single answer, take the shift
    # pooled over the blocks that registered cleanly (n-weighted median). On
    # the real scan every block's own peak sat at the same ~1.2 m NE and the
    # residual left after it is uncorrelated between neighbours, so the shift
    # is common to the whole survey, not per session
    good = [R_ for R_ in reg.values() if R_.get("applied")]
    if good:
        w = np.array([R_["n"] for R_ in good], float)
        V = np.array([R_["shift"] for R_ in good])
        pooled = np.array([np.median(np.repeat(V[:, a], w.astype(int))) for a in (0, 1)])
        for b, R_ in reg.items():
            if not R_.get("applied"):
                shift[sm.block == b] = pooled
                R_["pooled"] = pooled
        log(f"pooled shift ({pooled[0] * 100:+.1f}, {pooled[1] * 100:+.1f}) cm for "
            f"{sum(1 for R_ in reg.values() if 'pooled' in R_)} block(s) that could not stand alone")
    reg_xy = gt[:, :2] + shift

    # ---- 3. match ---------------------------------------------------------
    m = match(reg_xy, trunks)
    claims = np.zeros(len(trunks), int)
    for x in m:
        if x is not None:
            claims[x[0]] += 1
    rows = []
    for i in range(len(gt)):
        g_str, species = attrs.get((round(float(sm.lat[i]), 8), round(float(sm.lon[i]), 8)), ("", ""))
        row = dict(id=f"{int(sm.block[i])}-{sm.n[i]}", block=int(sm.block[i]), n=sm.n[i],
                   rtk_x=gt[i, 0], rtk_y=gt[i, 1], rtk_z=gt[i, 2],
                   shift_dx=shift[i, 0], shift_dy=shift[i, 1], reg_x=reg_xy[i, 0], reg_y=reg_xy[i, 1],
                   species=species, girth_cm=g_str, r_girth=girth_radius(g_str) if g_str else np.nan)
        if m[i] is None:
            row.update(status="no_fit", x=reg_xy[i, 0], y=reg_xy[i, 1], z=np.nan, radius=np.nan,
                       r_bh=np.nan, resid_m=np.nan, dist_m=np.nan, trunk_id="", n_layers=0, shared=0)
        else:
            tid, resid, d = m[i]
            t = trunks[tid]
            row.update(status="ok" if t["n_layers"] >= RUN else "weak", x=t["x"], y=t["y"], z=t["z"],
                       radius=t["radius"], r_bh=t["r_bh"], resid_m=resid, dist_m=d, trunk_id=tid,
                       n_layers=t["n_layers"], shared=int(claims[tid]), rms=t["rms"], arc=t["arc"])
        rows.append(row)
    groups = fork_groups(np.array([[r["x"], r["y"]] for r in rows]))
    for r, gid in zip(rows, groups):
        r["fork_group"] = int(gid)

    # ---- write ------------------------------------------------------------
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}" if np.isfinite(v) else ""
        return v
    with open(OUT / "trunks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trunk_id", "x", "y", "z", "radius", "r_bh", "n_layers", "rms", "arc", "survey_rows"])
        for t in trunks:
            w.writerow([t["id"]] + [fmt(t[k]) for k in ("x", "y", "z", "radius", "r_bh")]
                       + [t["n_layers"], fmt(t["rms"]), fmt(t["arc"]), int(claims[t["id"]])])
    cols_out = ["id", "block", "n", "rtk_x", "rtk_y", "rtk_z", "shift_dx", "shift_dy", "reg_x", "reg_y",
                "x", "y", "z", "radius", "r_bh", "status", "trunk_id", "resid_m", "dist_m", "n_layers",
                "shared", "rms", "arc", "fork_group", "species", "girth_cm", "r_girth"]
    with open(OUT / "bases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})
    with open(OUT / "bases_merged.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fork_group", "x", "y", "members", "status_best", "member_ids"])
        rank = {"ok": 0, "weak": 1, "no_fit": 2}
        for gid in np.unique(groups):
            mem = [r for r in rows if r["fork_group"] == gid]
            w.writerow([int(gid), f"{np.mean([r['x'] for r in mem]):.4f}",
                        f"{np.mean([r['y'] for r in mem]):.4f}", len(mem),
                        min((r["status"] for r in mem), key=rank.get), ";".join(r["id"] for r in mem)])

    # ---- audit ------------------------------------------------------------
    st = {s: sum(r["status"] == s for r in rows) for s in ("ok", "weak", "no_fit")}
    L = ["=" * 72, "StemNet v2 — trunks discovered, survey registered per block, matched"
         + ("  [SUBSET RUN]" if SUBSET else ""), "=" * 72,
         f"trunks discovered   {len(full):,} (>= {RUN} layers) + {len(trunks) - len(full):,} two-layer "
         f"candidates, within {DISCOVER_R:g} m of {len(gt):,} survey points",
         "", "-- 1. registration: one horizontal shift per survey block --",
         "   on bark: survey points within 8 cm of a trunk's bark after the shift",
         "   sharp: peak score / median score over the +-2 m search; 2nd: best score >30 cm away / peak"]
    for b, R_ in reg.items():
        if "surface" not in R_:
            L.append(f"   block {b}  n={R_['n']:5d}  not registered ({R_['why']})"
                     + (f"  -> pooled ({R_['pooled'][0] * 100:+.0f}, {R_['pooled'][1] * 100:+.0f}) cm"
                        if "pooled" in R_ else ""))
            continue
        L.append(f"   block {b}  n={R_['n']:5d}  shift ({R_['shift'][0] * 100:+6.1f}, {R_['shift'][1] * 100:+6.1f}) cm"
                 f" = {np.hypot(*R_['shift']) * 100:5.1f} cm  on bark {R_['on_bark'] * 100:4.1f}%  "
                 f"sharp x{R_['sharpness']:4.1f}  2nd {R_['second_ratio']:.2f}"
                 + ("" if R_["applied"] else f"  OWN PEAK REJECTED: {R_['why']}")
                 + (f"  -> pooled ({R_['pooled'][0] * 100:+.0f}, {R_['pooled'][1] * 100:+.0f}) cm"
                    if "pooled" in R_ else ""))
    L += ["", "-- 2. matching --",
          "status              " + "  ".join(f"{k} {v:,} ({v / len(rows) * 100:.1f}%)" for k, v in st.items())]
    mt = [r for r in rows if r["status"] in ("ok", "weak")]
    if mt:
        res = np.array([r["resid_m"] for r in mt]) * 100
        L.append(f"survey point - bark (cm, + outside the trunk):  p10 {np.percentile(res, 10):+.1f}  "
                 f"p50 {np.median(res):+.1f}  p90 {np.percentile(res, 90):+.1f} | within 3 cm "
                 f"{(np.abs(res) <= 3).mean() * 100:.0f}%, 5 cm {(np.abs(res) <= 5).mean() * 100:.0f}%, "
                 f"10 cm {(np.abs(res) <= 10).mean() * 100:.0f}%")
        for b in sorted({r["block"] for r in mt}):
            rb = np.array([r["resid_m"] for r in mt if r["block"] == b]) * 100
            nb = sum(r["block"] == b for r in rows)
            L.append(f"   block {b}: matched {len(rb):4d}/{nb:4d}  |resid| p50 {np.median(np.abs(rb)):4.1f} cm  "
                     f"within 5 cm {(np.abs(rb) <= 5).mean() * 100:3.0f}%")
        rad = np.array([r["radius"] for r in mt if r["status"] == "ok"]) * 100
        if len(rad):
            L.append(f"matched trunk radius (ok), cm  p10 {np.percentile(rad, 10):.1f}  p50 {np.median(rad):.1f}  "
                     f"p90 {np.percentile(rad, 90):.1f}")
    shared = int((claims >= 2).sum())
    unclaimed = int(((claims == 0) & np.array([t["n_layers"] >= RUN for t in trunks], bool)).sum()) if trunks else 0
    L += [f"trunks claimed by 2+ survey rows: {shared:,} (forks, or two rows for one stem)",
          f"trunks no survey row claims:     {unclaimed:,} of {len(full):,} (unsurveyed trees, or posts)",
          f"fork groups: {len(np.unique(groups)):,} after merging bases < {MERGE_R} m apart"]
    val = [(r["r_bh"], r["r_girth"]) for r in rows if r["status"] == "ok" and np.isfinite(r["r_bh"])
           and np.isfinite(r["r_girth"]) and ";" not in r["girth_cm"]]
    if val:
        v = np.array(val) * 100
        err = v[:, 0] - v[:, 1]
        L += ["", f"-- 3. girth: {len(v)} single-stem surveyed trees matched to a trunk --",
              f"fitted r at {BH[0]}-{BH[1]} m minus girth/2pi, cm:  p10 {np.percentile(err, 10):+.1f}  "
              f"p50 {np.median(err):+.1f}  p90 {np.percentile(err, 90):+.1f}",
              f"|error| median {np.median(np.abs(err)):.1f} cm | within 2 cm "
              f"{(np.abs(err) <= 2).mean() * 100:.0f}% | within 5 cm {(np.abs(err) <= 5).mean() * 100:.0f}%"]
    att = [r for r in rows if r["girth_cm"]]
    if att:
        L.append("attributed trees by status: " + "  ".join(
            f"{s} {sum(r['status'] == s for r in att)}" for s in st))
    txt = "\n".join(L)
    (RESULTS / "stemnet_v2_circles.txt").write_text(txt + "\n")
    print(txt, flush=True)

    rng2 = np.random.default_rng(1)
    ex = []
    for s, k in (("ok", 6), ("weak", 3), ("no_fit", 3)):
        ids = [i for i in plot_cols if rows[i]["status"] == s]
        ex += [int(i) for i in rng2.choice(ids, size=min(k, len(ids)), replace=False)] if ids else []
    figures(rows, full, reg, [(i, plot_cols[i]) for i in ex])

    frame["bases"] = dict(
        file="bases.csv", merged="bases_merged.csv", trunks="trunks.csv", status=st, subset=SUBSET,
        registration={str(b): dict(n=R_["n"], applied=R_["applied"],
                                   shift_m=[float(v) for v in R_["shift"]],
                                   pooled_m=[float(v) for v in R_["pooled"]] if "pooled" in R_ else None,
                                   **({k: R_[k] for k in ("on_bark", "sharpness", "second_ratio")}
                                      if "surface" in R_ else {"why": R_["why"]}))
                      for b, R_ in reg.items()},
        params=dict(discover_r=DISCOVER_R, cell=CELL, halo=HALO, scan_h=SCAN_H, layer=LAYER, gap=GAP,
                    r_min=R_MIN, r_max=R_MAX, arc=ARC, min_in=MIN_IN, in_tol=IN_TOL, run=RUN,
                    run_tol=RUN_TOL, max_gap=MAX_GAP, inside=INSIDE, small_r=SMALL_R, ransac=ITERS, reg_r=REG_R,
                    reg_sharp=REG_SHARP, reg_second=REG_SECOND, match_tol=MATCH_TOL,
                    merge_r=MERGE_R))
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log("-> data/stemnet_v2/trunks.csv, bases.csv, bases_merged.csv | results/stemnet_v2_circles.txt, "
        "42_/43_/45_ png")


if __name__ == "__main__":
    main()
