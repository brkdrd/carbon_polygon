#!/usr/bin/env python3
"""StemNet v2 prep, part 3: the tree bases — survey points, shifted into the
scan's frame, and an audit that the shift holds everywhere.

What was found (2026-10-15, on 80 columns cut out around survey points): the
survey points are fine, the whole survey is just offset from the scan. Shifted
by (+0.68, +1.02) m (1.23 m toward north-east), the survey points sit a median
2.4 cm from the nearest vertical surface in the scan — the bark the stick
touched: 76% within 5 cm, 85% within 10 cm, against 6-25% for wrong shifts of
the same length. Block 2 agreed at 2.0 cm. No trunk detector was involved.

So the base of every surveyed tree is:

    x, y = survey point + V2_SHIFT

Nothing is fitted, nothing is moved per tree, no survey height is used, and
every survey row is kept.

The audit — the shift was measured on 80 trees, mostly one block, so before
anything trains on it, it is checked on every survey point:
  * per point: distance from the shifted survey point to the nearest
    vertical-surface point in the scan (20-neighbour covariance verticality
    > V2_VERT, the paper's definition), in a band 0.1-1.3 m above the lowest
    scan points within 1 m. Same for the raw point and for the shift rotated
    90/180/270 deg (the chance level).
  * per survey block and per V2_AREA (100 m) square of campus: the share
    within 5 cm at the fixed shift, and the shift within +-V2_SEARCH (50 cm)
    of it that would do best for that group alone. If that local best is more
    than V2_WARN_M (15 cm) away AND gains more than V2_WARN_GAIN (10) points
    of within-5-cm share, the report says so loudly: that block or area does
    not sit where the fixed shift puts it.

Every row gets a quality flag, never a filter:
  on_bark  <= 5 cm from a vertical surface
  near     5-15 cm
  far      > 15 cm, or no vertical surface found within 1 m

Forks: shifted points within V2_MERGE_R (0.3 m) share a fork_group;
bases_merged.csv has one row per group, for training.

Outputs:
  data/stemnet_v2/bases.csv, bases_merged.csv
  results/stemnet_v2_bases.txt                   the audit
  results/45_stemnet_v2_bases_audit.png          distances, and per-area map
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
SHIFT = np.array([float(v) for v in os.environ.get("V2_SHIFT", "0.68,1.02").split(",")])
VERT = float(os.environ.get("V2_VERT", "0.8"))
K = int(os.environ.get("V2_FEAT_K", "20"))
BAND = (0.1, 1.3)
NEAR_R = 1.0                                        # how far around a point to look
SEARCH = float(os.environ.get("V2_SEARCH", "0.5"))
STEP = 0.05
AREA = float(os.environ.get("V2_AREA", "100"))
AREA_MIN = int(os.environ.get("V2_AREA_MIN", "30"))
WARN_M = float(os.environ.get("V2_WARN_M", "0.15"))
WARN_GAIN = float(os.environ.get("V2_WARN_GAIN", "10"))
MERGE_R = float(os.environ.get("V2_MERGE_R", "0.3"))
ON_BARK, NEAR = 0.05, 0.15
GRID_DIR = OUT / f"grid{GRID:g}"
FRAME_JSON = OUT / "frame.json"


def log(msg):
    print(f"[v2-bases] {msg}", flush=True)


def vertical_points(P, centre):
    """Vertical-surface points (xy) of the low band around `centre`: points
    0.1-1.3 m above the lowest scan points within 1 m, whose 20-neighbour
    covariance normal is near-horizontal (verticality > VERT)."""
    from scipy.spatial import cKDTree
    if len(P) < K + 10:
        return np.zeros((0, 2))
    near = np.hypot(P[:, 0] - centre[0], P[:, 1] - centre[1]) <= NEAR_R
    if near.sum() < 20:
        return np.zeros((0, 2))
    low = np.sort(P[near, 2])[min(10, near.sum() - 1)]
    Q = P[(P[:, 2] >= low + BAND[0]) & (P[:, 2] <= low + BAND[1])]
    if len(Q) < K:
        return np.zeros((0, 2))
    _, nb = cKDTree(Q).query(Q, k=K)
    C = Q[nb] - Q[nb].mean(1, keepdims=True)
    _, v = np.linalg.eigh(np.einsum("nki,nkj->nij", C, C))
    return Q[1 - np.abs(v[:, 2, 0]) > VERT, :2]


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


def attributes():
    out = {}
    if ATTRS.exists():
        with open(ATTRS) as f:
            for r in csv.DictReader(f):
                out[(round(float(r["lat"]), 8), round(float(r["lon"]), 8))] = (
                    (r.get("girth_cm") or "").strip(), (r.get("species") or "").strip())
    return out


def main():
    from scipy.spatial import cKDTree
    if not FRAME_JSON.exists():
        raise SystemExit("[v2-bases] no frame.json — run v2_grid first")
    frame = json.loads(FRAME_JSON.read_text())
    if f"{GRID:g}" not in frame.get("grids", {}):
        raise SystemExit(f"[v2-bases] no {GRID:g} m grid — run v2_grid first")
    RESULTS.mkdir(parents=True, exist_ok=True)

    sm = parse_stem_map(STEM_MAP)
    gt = survey_to_local(sm, frame["epsg"], frame["origin_m"])
    reg = gt[:, :2] + SHIFT
    n = len(gt)
    log(f"{n:,} survey rows | fixed shift ({SHIFT[0]:+.2f}, {SHIFT[1]:+.2f}) m")

    # candidate shifts: the local search grid around the fixed shift, plus the
    # raw position and the chance-level rotations
    g = np.arange(-SEARCH, SEARCH + 1e-9, STEP)
    GX, GY = np.meshgrid(g, g)
    cand = np.column_stack([GX.ravel(), GY.ravel()]) + SHIFT
    i0 = int(np.argmin(np.hypot(*(cand - SHIFT).T)))
    rots = []
    for ang in (90, 180, 270):
        a = np.radians(ang)
        rots.append(np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ SHIFT)
    extra = np.vstack([np.zeros(2)] + rots)
    allc = np.vstack([cand, extra])                       # (C, 2)
    reach = SEARCH + NEAR_R + 0.2

    # gather each point's surroundings tile by tile; finish a point when its
    # last tile is read (centres: the raw point, the fixed shift and the decoys
    # all fall inside a box of +-(|shift| + reach))
    have = {tuple(int(v) for v in q.stem.split("_")[1:]) for q in GRID_DIR.glob("tile_*.npz")}
    span = np.hypot(*SHIFT) + reach
    need = []
    for x, y in gt[:, :2]:
        need.append({(a, b) for a in range(int(np.floor((x - span) / TILE)), int(np.floor((x + span) / TILE)) + 1)
                     for b in range(int(np.floor((y - span) / TILE)), int(np.floor((y + span) / TILE)) + 1)} & have)
    by_tile: dict = {}
    for i, ts in enumerate(need):
        for q in ts:
            by_tile.setdefault(q, []).append(i)
    left = [len(t) for t in need]
    parts = [[] for _ in range(n)]
    D = np.full((n, len(allc)), np.inf)

    def finish(i):
        P = np.concatenate(parts[i]) if parts[i] else np.zeros((0, 3))
        parts[i] = None
        cen = gt[i, :2] + allc                            # every candidate centre
        # vertical points around the fixed-shift centre cover the whole local
        # search; the raw point and the decoys each get their own
        groups = [(reg[i], np.arange(len(cand)))] + [(gt[i, :2] + e, [len(cand) + k]) for k, e in enumerate(extra)]
        for c0, cols in groups:
            box = (np.abs(P[:, 0] - c0[0]) <= reach) & (np.abs(P[:, 1] - c0[1]) <= reach)
            V = vertical_points(P[box], c0)
            if len(V):
                D[i, cols] = cKDTree(V).query(cen[cols])[0]

    for i in range(n):
        if left[i] == 0:
            finish(i)
    t0 = time.time()
    tiles = sorted(by_tile)
    for k, q in enumerate(tiles):
        P = np.load(GRID_DIR / f"tile_{q[0]}_{q[1]}.npz")["xyz"].astype(np.float64)
        ids = by_tile[q]
        if len(P):
            order = np.argsort(P[:, 0], kind="stable"); xs = P[order, 0]
            for i in ids:
                x, y = gt[i, :2]
                lo, hi = np.searchsorted(xs, [x - span, x + span])
                ix = order[lo:hi]
                m = np.abs(P[ix, 1] - y) <= span
                if m.any():
                    parts[i].append(P[ix[m]])
        for i in ids:
            left[i] -= 1
            if left[i] == 0:
                finish(i)
        if (k + 1) % 25 == 0 or k + 1 == len(tiles):
            el = time.time() - t0
            log(f"tile {k + 1}/{len(tiles)} | {el / 60:.1f} min, eta {el / (k + 1) * (len(tiles) - k - 1) / 60:.1f} min")

    d_fix = D[:, i0]
    d_raw = D[:, len(cand)]
    d_rot = D[:, len(cand) + 1:]
    quality = np.where(d_fix <= ON_BARK, "on_bark", np.where(d_fix <= NEAR, "near", "far"))

    # ---- audit per block and per area -------------------------------------
    def audit(members):
        share = (D[members][:, :len(cand)] <= ON_BARK).mean(0) * 100
        b = int(np.argmax(share))
        return dict(n=int(len(members)), share_fixed=float(share[i0]), share_best=float(share[b]),
                    best_shift=[float(v) for v in cand[b]],
                    best_offset_cm=float(np.hypot(*(cand[b] - SHIFT)) * 100),
                    median_cm=float(np.median(np.where(np.isfinite(d_fix[members]), d_fix[members], 9.99)) * 100))

    def warn(a):
        return a["best_offset_cm"] > WARN_M * 100 and a["share_best"] - a["share_fixed"] > WARN_GAIN

    blocks = {}
    for b in sorted(set(sm.block.tolist())):
        m = np.flatnonzero(sm.block == b)
        if len(m):
            blocks[int(b)] = audit(m)
    areas = {}
    cell = np.floor(reg / AREA).astype(int)
    for c in sorted({tuple(v) for v in cell}):
        m = np.flatnonzero((cell[:, 0] == c[0]) & (cell[:, 1] == c[1]))
        if len(m) >= AREA_MIN:
            areas[c] = audit(m)
    all_ = audit(np.arange(n))

    # ---- write ------------------------------------------------------------
    attrs = attributes()
    groups = fork_groups(reg)
    rows = []
    for i in range(n):
        g_str, species = attrs.get((round(float(sm.lat[i]), 8), round(float(sm.lon[i]), 8)), ("", ""))
        rows.append(dict(id=f"{int(sm.block[i])}-{sm.n[i]}", block=int(sm.block[i]), n=sm.n[i],
                         rtk_x=gt[i, 0], rtk_y=gt[i, 1], rtk_z=gt[i, 2], x=reg[i, 0], y=reg[i, 1],
                         shift_dx=SHIFT[0], shift_dy=SHIFT[1],
                         bark_dist_m=d_fix[i], bark_dist_raw_m=d_raw[i], quality=quality[i],
                         status=quality[i], fork_group=int(groups[i]), species=species, girth_cm=g_str))

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}" if np.isfinite(v) else ""
        return v
    cols = ["id", "block", "n", "rtk_x", "rtk_y", "rtk_z", "x", "y", "shift_dx", "shift_dy",
            "bark_dist_m", "bark_dist_raw_m", "quality", "status", "fork_group", "species", "girth_cm"]
    with open(OUT / "bases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) for k in cols})
    with open(OUT / "bases_merged.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fork_group", "x", "y", "members", "quality_best", "member_ids"])
        rank = {"on_bark": 0, "near": 1, "far": 2}
        for gid in np.unique(groups):
            mem = [r for r in rows if r["fork_group"] == gid]
            w.writerow([int(gid), f"{np.mean([r['x'] for r in mem]):.4f}", f"{np.mean([r['y'] for r in mem]):.4f}",
                        len(mem), min((r["quality"] for r in mem), key=rank.get), ";".join(r["id"] for r in mem)])

    # ---- report -----------------------------------------------------------
    def pct(d, t):
        return float((d <= t).mean() * 100)
    fin = np.isfinite(d_fix)
    L = ["=" * 72, "StemNet v2 — tree bases = survey point + fixed shift, audited", "=" * 72,
         f"survey rows {n:,} (all kept) | shift ({SHIFT[0]:+.2f}, {SHIFT[1]:+.2f}) m = {np.hypot(*SHIFT):.2f} m "
         f"@ {np.degrees(np.arctan2(SHIFT[1], SHIFT[0])):.0f} deg",
         f"no vertical surface within {NEAR_R:g} m of the shifted point: {int((~fin).sum()):,}",
         "", "-- distance from the survey point to the nearest vertical surface (bark) --",
         f"   {'':22s}{'p50':>8s}{'<=5 cm':>9s}{'<=10 cm':>9s}{'<=20 cm':>9s}"]
    for lab, d in (("fixed shift", d_fix), ("raw survey point", d_raw)):
        L.append(f"   {lab:22s}{np.median(np.where(np.isfinite(d), d, 9.99)) * 100:7.1f}{pct(d, .05):8.0f}%"
                 f"{pct(d, .10):8.0f}%{pct(d, .20):8.0f}%")
    for k, ang in enumerate((90, 180, 270)):
        d = d_rot[:, k]
        L.append(f"   {f'chance: rotated {ang} deg':22s}{np.median(np.where(np.isfinite(d), d, 9.99)) * 100:7.1f}"
                 f"{pct(d, .05):8.0f}%{pct(d, .10):8.0f}%{pct(d, .20):8.0f}%")
    L.append("quality             " + "  ".join(f"{q} {int((quality == q).sum()):,} "
                                               f"({(quality == q).mean() * 100:.0f}%)" for q in ("on_bark", "near", "far")))
    L += ["", f"-- does the fixed shift hold everywhere? best local shift within +-{SEARCH * 100:.0f} cm --",
          f"   WARNING when the local best is > {WARN_M * 100:.0f} cm away AND gains > {WARN_GAIN:.0f} points of <=5 cm share",
          f"   {'group':16s}{'n':>6s}{'<=5cm fixed':>13s}{'<=5cm best':>12s}{'best shift (cm)':>18s}{'away':>7s}"]

    def line(name, a):
        flag = "  <-- WARNING" if warn(a) else ""
        return (f"   {name:16s}{a['n']:6d}{a['share_fixed']:12.0f}%{a['share_best']:11.0f}%"
                f"   ({a['best_shift'][0] * 100:+5.0f},{a['best_shift'][1] * 100:+5.0f}){a['best_offset_cm']:7.0f}{flag}")
    L.append(line("ALL", all_))
    for b, a in blocks.items():
        L.append(line(f"block {b}", a))
    for c, a in areas.items():
        L.append(line(f"area {c[0] * AREA:.0f},{c[1] * AREA:.0f}", a))
    n_warn = sum(warn(a) for a in list(blocks.values()) + list(areas.values()))
    L.append(f"   -> {n_warn} group(s) flagged" + ("" if n_warn else ": the fixed shift holds everywhere it was checked"))
    txt = "\n".join(L)
    (RESULTS / "stemnet_v2_bases.txt").write_text(txt + "\n")
    print(txt, flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    bins = np.linspace(0, 60, 61)
    for lab, d, c in (("fixed shift", d_fix, "tab:green"), ("raw", d_raw, "tab:red"),
                      ("chance (rotated 180)", d_rot[:, 1], "0.5")):
        ax[0].hist(np.clip(np.where(np.isfinite(d), d, 0.6) * 100, 0, 60), bins=bins, histtype="step", lw=2,
                   label=lab, color=c)
    ax[0].set_xlabel("survey point -> nearest vertical surface (cm, clipped at 60)"); ax[0].legend()
    ax[0].set_title("survey points on the bark?")
    if areas:
        A = np.array([[c[0] * AREA + AREA / 2, c[1] * AREA + AREA / 2, a["share_fixed"],
                       a["best_shift"][0] - SHIFT[0], a["best_shift"][1] - SHIFT[1]] for c, a in areas.items()])
        sc = ax[1].scatter(A[:, 0], A[:, 1], c=A[:, 2], s=900, marker="s", cmap="RdYlGn", vmin=0, vmax=100)
        ax[1].quiver(A[:, 0], A[:, 1], A[:, 3], A[:, 4], angles="xy", scale_units="xy", scale=1 / 100, width=0.004)
        plt.colorbar(sc, ax=ax[1], label="% within 5 cm at the fixed shift")
        ax[1].set_aspect("equal")
        ax[1].set_title("per 100 m area: share on bark; arrow = local best minus fixed shift (x100)")
    fig.tight_layout(); fig.savefig(RESULTS / "45_stemnet_v2_bases_audit.png", dpi=110); plt.close(fig)

    frame["bases"] = dict(file="bases.csv", merged="bases_merged.csv", shift_m=[float(v) for v in SHIFT],
                          quality={q: int((quality == q).sum()) for q in ("on_bark", "near", "far")},
                          audit=dict(all=all_, blocks={str(b): a for b, a in blocks.items()},
                                     areas={f"{c[0]},{c[1]}": a for c, a in areas.items()}, warnings=n_warn),
                          params=dict(vert=VERT, k=K, band=BAND, near_r=NEAR_R, search=SEARCH, step=STEP,
                                      area=AREA, warn_m=WARN_M, warn_gain=WARN_GAIN, merge_r=MERGE_R))
    FRAME_JSON.write_text(json.dumps(frame, indent=2))
    log("-> data/stemnet_v2/bases.csv, bases_merged.csv | results/stemnet_v2_bases.txt, 45_ png")


if __name__ == "__main__":
    main()
