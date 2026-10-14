#!/usr/bin/env python3
"""StemNet v2 prep, part 3: the derived stem base of every surveyed tree.

The RTK stick touched the bark, not the stem axis, so each survey point sits
about one stem radius off-centre in an unrecorded direction. Here the cloud
itself says where the axis is:

  1. column   every grid point within V2_COL_R (1 m) of the survey point, plan
              view, all heights
  2. layers   the lowest V2_SCAN_H (8 m) of the column cut into V2_LAYER (0.2 m)
              slices, bottom up
  3. circle   per layer: RANSAC over 3-point circles, then a least-squares
              refit on the inliers. A layer is accepted only if
                - the survey point lies on the circle within V2_ON_TOL (8 cm):
                  the stick touched this trunk, not the one next to it
                - the radius is V2_R_MIN..V2_R_MAX (1-80 cm)
                - the inliers cover >= V2_ARC (90) degrees of the circle
                - at least V2_MIN_IN (12) inliers
                - the inside is EMPTY: point density inside the circle at
                  most V2_INSIDE (0.25) x the density on the ring. A trunk is hollow to a scanner;
                  a slice through flat ground is a solid disc, and without this
                  test any circle drawn on it would pass — ground stays in the
                  cloud, so this is the check that keeps it out of the fit.
                  Below V2_SMALL_R (6 cm) a circle has no inside left once its
                  own 2 cm band is taken away, so for saplings the test looks
                  at a 5 cm SHELL just outside the ring instead: a sapling
                  stands in empty air, flat ground has points all round
  4. trunk    the LOWEST run of >= V2_RUN (3) consecutive accepted layers whose
              centres agree within V2_RUN_TOL (10 cm). Base = the median centre
              of that run; z = the bottom of its lowest layer.

No ground estimate and no survey height are used anywhere. Crown, shrub and
neighbouring trunks fail the "circle through the survey point" test on their
own, so the scan needs no height reference.

Every survey row is kept. status:
  ok      a run of >= V2_RUN layers
  weak    accepted layers, but no run that long — best run used, flagged
  no_fit  points in the column, no layer passed
  sparse  fewer than V2_MIN_COL (50) points in the column
Anything not `ok` keeps the survey point as its base, so `x, y` is always a
usable position and `status` says how much to trust it.

Forks: rows whose DERIVED bases lie within V2_MERGE_R (0.3 m) share a
fork_group, and the group's merged base (for training) is the mean of its
members' bases. Grouping happens after the fit, on the fitted axes: two stems
25 cm apart, each poked on its outer side, have survey points 41 cm apart, so
grouping the survey points would never join them. Each member row keeps its
own fit.

Validation: for the girth-attributed trees, the radius of the accepted layers
1.0-1.6 m above the trunk bottom (breast height) against girth / 2 pi.

Outputs:
  data/stemnet_v2/bases.csv                  one row per survey row
  data/stemnet_v2/bases_merged.csv           one row per fork group (training)
  results/stemnet_v2_circles.txt             summary + girth validation
  results/42_stemnet_v2_girth.png            fitted vs surveyed radius
  results/43_stemnet_v2_circle_examples.png  layers, circles, survey point
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
COL_R = float(os.environ.get("V2_COL_R", "1.0"))
SCAN_H = float(os.environ.get("V2_SCAN_H", "8.0"))
LAYER = float(os.environ.get("V2_LAYER", "0.2"))
ON_TOL = float(os.environ.get("V2_ON_TOL", "0.08"))
R_MIN = float(os.environ.get("V2_R_MIN", "0.01"))
R_MAX = float(os.environ.get("V2_R_MAX", "0.8"))
ARC = float(os.environ.get("V2_ARC", "90"))
MIN_IN = int(os.environ.get("V2_MIN_IN", "12"))
IN_TOL = float(os.environ.get("V2_IN_TOL", "0.02"))      # inlier band, m (~ one voxel)
RUN = int(os.environ.get("V2_RUN", "3"))
RUN_TOL = float(os.environ.get("V2_RUN_TOL", "0.10"))
MIN_COL = int(os.environ.get("V2_MIN_COL", "50"))
MERGE_R = float(os.environ.get("V2_MERGE_R", "0.3"))
ITERS = int(os.environ.get("V2_RANSAC", "500"))
INSIDE = float(os.environ.get("V2_INSIDE", "0.25"))     # max density inside / density on the ring
SMALL_R = float(os.environ.get("V2_SMALL_R", "0.06"))   # below this there is no inside to test
SHELL = 0.05                                            # outside band for small circles, m
BH = (1.0, 1.6)
# V2_SUBSET=1: only some grid tiles are present (a trial on a few real tiles).
# Fit only the trees whose whole column lies in tiles that exist; a column cut
# off by a missing tile would give a confident, wrong fit.
SUBSET = os.environ.get("V2_SUBSET", "0") != "0"
GRID_DIR = OUT / f"grid{GRID:g}"
FRAME_JSON = OUT / "frame.json"


def log(msg):
    print(f"[v2-circle] {msg}", flush=True)


# ---------------------------------------------------------------------------
# one layer
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
    relative to the ring itself (points per m2 there / points per m2 on the
    ring). Densities, not counts: a small circle's inside is a tiny disc that
    holds few points even when it is solid ground, so a count would wave
    small circles through. Flat ground scores ~1 at any radius; a trunk ~0.

    S (H, n): signed distance of every point to each hypothesis' circle,
    r (H,) radii. Large circles are judged on their inside; below SMALL_R
    there is no inside left, so on a shell just outside the ring."""
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


def fit_layer(P, g, rng):
    """Best accepted circle in one layer, or None. P (n, 2), g survey xy."""
    n = len(P)
    if n < MIN_IN:
        return None
    # half the hypotheses use the survey point as one of the three: when the
    # trunk is 10% of a layer, two trunk points in a row (1%) are far likelier
    # than three (0.1%). The refit below removes the survey point's error.
    # and points near the survey point are drawn more often: a trunk touching
    # it lies within two radii of it, while ground and shrub are spread over
    # the whole metre-wide column
    d = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1])
    w = 1.0 / (d + 0.05)
    w /= w.sum()
    idx = np.column_stack([rng.integers(0, n, ITERS),
                           rng.choice(n, ITERS, p=w), rng.choice(n, ITERS, p=w)])
    A = P[idx[:, 0]].copy()
    A[: ITERS // 2] = g
    U, r = circles_3pt(A, P[idx[:, 1]], P[idx[:, 2]])
    on = np.abs(np.hypot(U[:, 0] - g[0], U[:, 1] - g[1]) - r)
    keep = (r >= R_MIN) & (r <= R_MAX) & (on <= ON_TOL)
    if not keep.any():
        return None
    U, r = U[keep], r[keep]
    # inliers AND inside counts for all surviving hypotheses at once. The
    # hollow-inside test has to act here, while choosing: a large circle drawn
    # through dense ground always has more inliers than a real trunk ring, and
    # picking it first would throw the whole layer away
    S = np.hypot(P[None, :, 0] - U[:, None, 0], P[None, :, 1] - U[:, None, 1]) - r[:, None]
    cnt = (np.abs(S) <= IN_TOL).sum(1)
    ins = crowding(S, r)
    cnt = np.where(ins <= INSIDE, cnt, -1)
    best = int(np.argmax(cnt))
    if cnt[best] < MIN_IN:
        return None
    inl = P[np.abs(S[best]) <= IN_TOL]
    if len(inl) < MIN_IN:
        return None
    c, rr = kasa(inl)
    for _ in range(2):                                   # refit on the refit's inliers
        d = np.abs(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) - rr)
        inl = P[d <= IN_TOL]
        if len(inl) < MIN_IN:
            return None
        c, rr = kasa(inl)
    on = abs(np.hypot(c[0] - g[0], c[1] - g[1]) - rr)
    if not (R_MIN <= rr <= R_MAX) or on > ON_TOL:
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
                arc=arc, rms=res, on=float(on), inside=inside)


# ---------------------------------------------------------------------------
# one tree
# ---------------------------------------------------------------------------
def runs(layers):
    """Consecutive accepted layers whose centres agree within RUN_TOL."""
    out, cur = [], []
    for L in layers:
        if L["fit"] is None:
            if cur:
                out.append(cur)
            cur = []
            continue
        if cur:
            m = np.median([[c["fit"]["cx"], c["fit"]["cy"]] for c in cur], axis=0)
            if np.hypot(L["fit"]["cx"] - m[0], L["fit"]["cy"] - m[1]) > RUN_TOL:
                out.append(cur)
                cur = []
        cur.append(L)
    if cur:
        out.append(cur)
    return out


def fit_tree(col, g, rng):
    """col (n, 3) column points, g survey xy -> result dict + per-layer list."""
    if len(col) < MIN_COL:
        return dict(status="sparse"), []
    z0 = float(col[:, 2].min())
    nl = int(np.ceil(min(SCAN_H, col[:, 2].max() - z0 + 1e-9) / LAYER))
    li = np.floor((col[:, 2] - z0) / LAYER).astype(int)
    layers = []
    for k in range(nl):
        P = col[li == k, :2]
        layers.append(dict(k=k, z=z0 + k * LAYER, n=len(P), fit=fit_layer(P, g, rng)))
    rs = runs(layers)
    if not rs:
        return dict(status="no_fit", n_layers_ok=0), layers
    long = [r for r in rs if len(r) >= RUN]
    run = long[0] if long else max(rs, key=len)          # lowest long run, else the longest
    C = np.array([[L["fit"]["cx"], L["fit"]["cy"]] for L in run])
    c = np.median(C, axis=0)
    zb = run[0]["z"]
    rad = np.median([L["fit"]["r"] for L in run])
    bh = [L["fit"]["r"] for L in layers
          if L["fit"] is not None and BH[0] <= L["z"] - zb + LAYER / 2 <= BH[1]
          and np.hypot(L["fit"]["cx"] - c[0], L["fit"]["cy"] - c[1]) <= RUN_TOL]
    return dict(status="ok" if len(run) >= RUN else "weak",
                x=float(c[0]), y=float(c[1]), z=float(zb), radius=float(rad),
                r_bh=float(np.median(bh)) if bh else np.nan, n_bh=len(bh),
                run_layers=len(run), n_layers_ok=sum(L["fit"] is not None for L in layers),
                rms=float(np.median([L["fit"]["rms"] for L in run])),
                arc=float(np.median([L["fit"]["arc"] for L in run])),
                on=float(np.median([L["fit"]["on"] for L in run]))), layers


# ---------------------------------------------------------------------------
def load_columns(xy, frame):
    """Column points around every survey point, gathered tile by tile."""
    from scipy.spatial import cKDTree
    t = np.floor(xy / TILE).astype(np.int64)
    cols = [[] for _ in range(len(xy))]
    tiles = sorted({(int(a), int(b)) for a, b in t})
    need = {}
    for i, (a, b) in enumerate(t):                     # a column can cross a tile edge
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                need.setdefault((int(a) + da, int(b) + db), []).append(i)
    t0 = time.time()
    present = sorted(q for q in need if (GRID_DIR / f"tile_{q[0]}_{q[1]}.npz").exists())
    for k, q in enumerate(present):
        P = np.load(GRID_DIR / f"tile_{q[0]}_{q[1]}.npz")["xyz"].astype(np.float64)
        if not len(P):
            continue
        ids = need[q]
        kd = cKDTree(P[:, :2])
        hits = kd.query_ball_point(xy[ids], r=COL_R)
        for i, h in zip(ids, hits):
            if h:
                cols[i].append(P[h])
        if (k + 1) % 50 == 0 or k + 1 == len(present):
            log(f"columns: tile {k + 1}/{len(present)} | {(time.time() - t0) / 60:.1f} min")
    return [np.concatenate(c) if c else np.zeros((0, 3)) for c in cols]


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


def figures(rows, examples, cols, gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r["status"] == "ok" and np.isfinite(r["r_bh"]) and np.isfinite(r["r_girth"])]
    fig, ax = plt.subplots(figsize=(7, 7))
    if ok:
        a = np.array([[r["r_girth"] * 100, r["r_bh"] * 100] for r in ok])
        ax.scatter(a[:, 0], a[:, 1], s=10)
        m = max(a.max(), 1)
        ax.plot([0, m], [0, m], "k--", lw=0.8, label="fitted = surveyed")
    ax.set_xlabel("surveyed radius, girth / 2 pi (cm)"); ax.set_ylabel("fitted radius at 1.0-1.6 m (cm)")
    ax.set_title(f"circle fit vs survey girth, {len(ok)} trees"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "42_stemnet_v2_girth.png", dpi=120); plt.close(fig)

    n = len(examples)
    if not n:
        return
    fig, axs = plt.subplots(3, (n + 2) // 3, figsize=(4 * ((n + 2) // 3), 12), squeeze=False)
    for a, (i, layers) in zip(axs.ravel(), examples):
        r = rows[i]
        C = cols[i]
        a.scatter(C[:, 0] - gt[i, 0], C[:, 1] - gt[i, 1], s=0.5, c=C[:, 2] - C[:, 2].min(),
                  cmap="viridis", lw=0)
        for L in layers:
            f = L["fit"]
            if f is not None:
                a.add_patch(plt.Circle((f["cx"] - gt[i, 0], f["cy"] - gt[i, 1]), f["r"],
                                       fill=False, lw=0.6, color="tab:orange"))
        a.plot(0, 0, "r+", ms=12, mew=2, label="survey point")
        if r["status"] in ("ok", "weak"):
            a.plot(r["x"] - gt[i, 0], r["y"] - gt[i, 1], "kx", ms=10, mew=2, label="derived base")
        a.set_xlim(-COL_R, COL_R); a.set_ylim(-COL_R, COL_R); a.set_aspect("equal")
        a.set_title(f"{r['id']} {r['status']} r={r.get('radius', np.nan) * 100:.0f} cm "
                    f"run={r.get('run_layers', 0)}", fontsize=9)
    axs.ravel()[0].legend(fontsize=7)
    for a in axs.ravel()[n:]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / "43_stemnet_v2_circle_examples.png", dpi=110)
    plt.close(fig)


def main():
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
    if SUBSET:
        have = {tuple(int(v) for v in q.stem.split("_")[1:]) for q in GRID_DIR.glob("tile_*.npz")}
        lo = np.floor((gt[:, :2] - COL_R) / TILE).astype(int)
        hi = np.floor((gt[:, :2] + COL_R) / TILE).astype(int)
        cov = np.array([all((a, b) in have for a in range(lo[i, 0], hi[i, 0] + 1)
                            for b in range(lo[i, 1], hi[i, 1] + 1)) for i in range(len(gt))])
        log(f"!! SUBSET RUN on {len(have)} tiles: fitting the {int(cov.sum())} trees whose "
            f"whole column is covered; {int((~cov).sum()):,} others are not in this run")
        k = np.flatnonzero(cov)
        sm = type(sm)(sm.block[k], sm.n[k], sm.lat[k], sm.lon[k], sm.z[k], sm.stations)
        gt = gt[k]
    cols = load_columns(gt[:, :2], frame)
    attrs = attributes()

    rows, layer_log = [], {}
    t0 = time.time()
    for i in range(len(gt)):
        res, layers = fit_tree(cols[i], gt[i, :2], rng)
        layer_log[i] = layers
        g_str, species = attrs.get((round(float(sm.lat[i]), 8), round(float(sm.lon[i]), 8)), ("", ""))
        row = dict(id=f"{int(sm.block[i])}-{sm.n[i]}", block=int(sm.block[i]), n=sm.n[i],
                   rtk_x=gt[i, 0], rtk_y=gt[i, 1], rtk_z=gt[i, 2],
                   n_column=int(len(cols[i])),
                   species=species, girth_cm=g_str, r_girth=girth_radius(g_str) if g_str else np.nan)
        row.update(res)
        if row["status"] not in ("ok", "weak"):
            row.update(x=gt[i, 0], y=gt[i, 1], z=np.nan, radius=np.nan, r_bh=np.nan)
        row["shift_m"] = float(np.hypot(row["x"] - gt[i, 0], row["y"] - gt[i, 1]))
        rows.append(row)
        if (i + 1) % 500 == 0 or i + 1 == len(gt):
            el = time.time() - t0
            log(f"fit: {i + 1}/{len(gt)} | {el / 60:.1f} min, eta {el / (i + 1) * (len(gt) - i - 1) / 60:.1f} min")

    groups = fork_groups(np.array([[r["x"], r["y"]] for r in rows]))
    for r, gid in zip(rows, groups):
        r["fork_group"] = int(gid)

    cols_out = ["id", "block", "n", "rtk_x", "rtk_y", "rtk_z", "x", "y", "z", "shift_m", "radius",
                "r_bh", "n_bh", "status", "run_layers", "n_layers_ok", "rms", "arc", "on",
                "n_column", "fork_group", "species", "girth_cm", "r_girth"]
    with open(OUT / "bases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else
                            ("" if isinstance(v, float) else v)) for k, v in r.items()})

    # one base per fork group, for training: mean of the members' bases
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

    # report
    st = {s: sum(r["status"] == s for r in rows) for s in ("ok", "weak", "no_fit", "sparse")}
    okr = [r for r in rows if r["status"] == "ok"]
    sh = np.array([r["shift_m"] for r in okr]) * 100
    rad = np.array([r["radius"] for r in okr]) * 100
    val = [(r["r_bh"], r["r_girth"]) for r in okr
           if np.isfinite(r["r_bh"]) and np.isfinite(r["r_girth"]) and ";" not in r["girth_cm"]]
    L = ["=" * 72, "StemNet v2 — derived stem bases (circle fit, layer scan)"
         + ("  [SUBSET RUN]" if SUBSET else ""), "=" * 72,
         f"survey rows     {len(rows):,}  (all kept)",
         "status          " + "  ".join(f"{k} {v:,} ({v / len(rows) * 100:.1f}%)" for k, v in st.items()),
         f"fork groups     {len(np.unique(groups)):,} after merging survey points < {MERGE_R} m apart"]
    if len(okr):
        L += [f"survey -> derived shift (ok trees), cm:  p10 {np.percentile(sh, 10):.1f}  "
              f"p50 {np.median(sh):.1f}  p90 {np.percentile(sh, 90):.1f}",
              f"fitted radius (ok trees), cm:            p10 {np.percentile(rad, 10):.1f}  "
              f"p50 {np.median(rad):.1f}  p90 {np.percentile(rad, 90):.1f}",
              "   (if the stick touched the bark, shift ~= radius)"]
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
        L.append(f"attributed trees by status: " + "  ".join(
            f"{s} {sum(r['status'] == s for r in att)}" for s in st))
    txt = "\n".join(L)
    (RESULTS / "stemnet_v2_circles.txt").write_text(txt + "\n")
    print(txt, flush=True)

    rng2 = np.random.default_rng(1)
    ex = []
    for s, k in (("ok", 6), ("weak", 3), ("no_fit", 3)):
        ids = [i for i, r in enumerate(rows) if r["status"] == s]
        ex += [int(i) for i in rng2.choice(ids, size=min(k, len(ids)), replace=False)] if ids else []
    figures(rows, [(i, layer_log[i]) for i in ex], cols, gt)

    frame["bases"] = dict(file="bases.csv", merged="bases_merged.csv", status=st,
                          params=dict(col_r=COL_R, scan_h=SCAN_H, layer=LAYER, on_tol=ON_TOL,
                                      r_min=R_MIN, r_max=R_MAX, arc=ARC, min_in=MIN_IN,
                                      in_tol=IN_TOL, run=RUN, run_tol=RUN_TOL, merge_r=MERGE_R,
                                      inside=INSIDE, small_r=SMALL_R, ransac=ITERS))
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log("-> data/stemnet_v2/bases.csv, bases_merged.csv | results/stemnet_v2_circles.txt, 42_/43_ png")


if __name__ == "__main__":
    main()
