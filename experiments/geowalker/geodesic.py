#!/usr/bin/env python3
"""GeoWalker stage A — the through-cloud distance field to the nearest tree base.

For every point of the scene this computes the distance to the nearest labelled
tree base measured **along the point cloud**, never through open air: a
multi-source Dijkstra over a kNN graph whose edges are capped at GW_MAX_EDGE
metres. That is exactly the "start from the bases, repeatedly absorb the closest
unreached point and carry its edge length along" construction — Dijkstra is that
procedure with the frontier kept in a heap.

The result is >= the straight-line distance everywhere (a path through the cloud
cannot be shorter than the chord), and it is the quantity the walker descends:
it encodes *how far along real structure* a base is, so the model can orient
even when the base is hidden behind a wall or on the other side of a hedge.

Resolution: the field is built on a VOXEL-metre occupancy grid of the scene, not
on the raw 0.05 m cloud (a campus at 0.05 m is ~10^8 points — a kNN graph on
that does not fit in RAM, and the walker never needs the field at finer than
voxel resolution). Nodes are voxel centres; edge weights are the true 3D
distances between them.

Inputs  (from the BaseWalker prep stage — same scene, same local frame):
  /data/basewalker/tiles/*.npz     xyz float32, local coordinates
  /data/basewalker/labels.npz      tree bases + train/val/margin flags
Outputs:
  /data/geowalker/field.npz        node xyz + g_train + g_all + build params
  /results/20_geowalker_field.png  what the field looks like (audit figure)

Idempotent: skips if field.npz exists (FORCE_GW_FIELD=1 rebuilds).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# this directory first (its siblings are model.py/geodesic.py), then /app
# for the shared modules — /app also holds BaseWalker's train.py/infer.py,
# which must never shadow ours.
sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C

BW = C.DATA / "basewalker"
GW = C.DATA / "geowalker"
FIELD_NPZ = GW / "field.npz"
FIELD_META = GW / "field_meta.json"

VOXEL = float(os.environ.get("GW_VOXEL", "0.25"))        # graph resolution, m
KNN = int(os.environ.get("GW_KNN", "16"))                # candidate neighbours
# Edge cap. 2 x VOXEL spans the 26-neighbourhood of the occupancy grid plus a
# one-voxel hole, so the path follows surfaces but still crosses the small gaps
# a real scan leaves. Raise it and the path starts short-cutting through air.
MAX_EDGE = float(os.environ.get("GW_MAX_EDGE", str(2.0 * VOXEL)))
BASE_R = float(os.environ.get("GW_BASE_R", "0.5"))       # seed ball around a base
MAX_NODES = int(float(os.environ.get("GW_MAX_NODES", "12e6")))
QUERY_CHUNK = 1_000_000


# ---------------------------------------------------------------------------
# voxel occupancy grid
# ---------------------------------------------------------------------------
def _unique_voxels(k: np.ndarray) -> np.ndarray:
    """Unique rows of an integer voxel-index array, via an int64 flat key
    (np.unique(axis=0) is an order of magnitude slower)."""
    k = np.asarray(k, dtype=np.int64)
    lo = k.min(0)
    dim = (k.max(0) - lo + 1).astype(np.int64)
    flat = ((k[:, 0] - lo[0]) * dim[1] + (k[:, 1] - lo[1])) * dim[2] + (k[:, 2] - lo[2])
    keep = np.unique(flat, return_index=True)[1]
    keep.sort()
    return k[keep]


def build_nodes():
    """Scene tiles -> occupancy-grid nodes (voxel centres, local frame).

    Voxel indices use one GLOBAL origin (floor(x / v)), so per-tile dedup and
    the final global dedup agree and tiles cannot double-count a shared voxel.
    """
    tiles = sorted((BW / "tiles").glob("tile_*.npz"))
    if not tiles:
        raise SystemExit(f"[gw-field] no tiles in {BW / 'tiles'} — run bw_prep first")
    v = VOXEL
    parts, n_raw = [], 0
    for p in tiles:
        xyz = np.load(p)["xyz"].astype(np.float64)
        n_raw += len(xyz)
        parts.append(_unique_voxels(np.floor(xyz / v)))
    k = _unique_voxels(np.concatenate(parts))
    del parts
    print(f"[gw-field] {n_raw:,} cloud pts -> {len(k):,} occupied {v} m voxels "
          f"({len(tiles)} tiles)")

    # Guard the RAM cliff: the kNN graph is ~KNN edges per node and scipy holds
    # it in float64 + int32. Coarsen rather than die half way through a build.
    while len(k) > MAX_NODES:
        old = v
        v *= 1.5
        k = _unique_voxels(np.floor(((k + 0.5) * old) / v))
        print(f"[gw-field] !! > {MAX_NODES:,} nodes — coarsening to {v:.3f} m "
              f"-> {len(k):,} nodes (raise GW_MAX_NODES to keep {old:.3f} m)")

    nodes = (k.astype(np.float64) + 0.5) * v      # voxel centres, local frame
    return nodes, v, n_raw


# ---------------------------------------------------------------------------
# kNN graph + multi-source Dijkstra
# ---------------------------------------------------------------------------
def build_graph(nodes, kd):
    """Symmetric kNN graph, edges longer than MAX_EDGE dropped.

    Built straight into CSR: the kNN query already yields the rows in order, so
    a row-count cumsum is the indptr — no COO round trip, no sort of ~10^8 edges.
    """
    from scipy.sparse import csr_matrix

    n = len(nodes)
    ind_l, dat_l, cnt_l = [], [], []
    t0 = time.time()
    for s in range(0, n, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, n)
        # distance_upper_bound: missing neighbours come back as (inf, n)
        d, j = kd.query(nodes[s:e], k=KNN + 1, workers=-1,
                        distance_upper_bound=MAX_EDGE)
        m = np.isfinite(d) & (j != np.arange(s, e)[:, None])
        cnt_l.append(m.sum(1))
        ind_l.append(j[m].astype(np.int32))
        dat_l.append(d[m])
        print(f"[gw-field]   kNN {e:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)
    counts = np.concatenate(cnt_l)
    cum = np.cumsum(counts, dtype=np.int64)
    if cum[-1] >= 2 ** 31:
        raise SystemExit(f"[gw-field] {cum[-1]:,} edges overflow 32-bit CSR — "
                         f"raise GW_VOXEL or lower GW_KNN")
    indptr = np.zeros(n + 1, np.int32)
    indptr[1:] = cum
    g = csr_matrix((np.concatenate(dat_l), np.concatenate(ind_l), indptr),
                   shape=(n, n))
    deg = counts.mean()
    print(f"[gw-field] graph: {g.nnz:,} directed edges | mean degree {deg:.1f} "
          f"| edge cap {MAX_EDGE:.2f} m | {time.time()-t0:.0f}s")
    if deg < 3:
        print("[gw-field] !! mean degree < 3 — the cloud is sparser than "
              "GW_MAX_EDGE; raise GW_MAX_EDGE or GW_VOXEL or the field will "
              "be mostly unreachable")
    return g


def seed_nodes(kd, bases, name):
    """Graph nodes that ARE a tree base: everything within BASE_R of a label.

    A label with no node in range (a base the scanner never covered) falls back
    to its single nearest node so it still seeds the field instead of silently
    dropping out.
    """
    ids, n_fallback = set(), 0
    for bi, hit in enumerate(kd.query_ball_point(bases, r=BASE_R, workers=-1)):
        if hit:
            ids.update(hit)
        else:
            ids.add(int(kd.query(bases[bi])[1]))
            n_fallback += 1
    out = np.fromiter(sorted(ids), np.int64)
    print(f"[gw-field] {name}: {len(bases)} bases -> {len(out):,} seed nodes "
          f"(within {BASE_R} m; {n_fallback} fell back to nearest node)")
    return out


def geodesic(graph, sources, name):
    """Multi-source Dijkstra: distance from every node to its nearest source,
    travelling only along graph edges. Unreached nodes stay at +inf."""
    from scipy.sparse.csgraph import dijkstra

    t0 = time.time()
    # min_only=True runs ONE pass over the whole graph with every source
    # pre-loaded in the heap — the additive frontier expansion, not |sources|
    # separate searches. directed=False makes the kNN graph symmetric on the fly.
    d = dijkstra(graph, directed=False, indices=sources, min_only=True)
    d = d.astype(np.float32)
    ok = np.isfinite(d)
    q = np.percentile(d[ok], [50, 90, 99]) if ok.any() else [np.nan] * 3
    print(f"[gw-field] {name}: reachable {ok.mean()*100:.1f}% of nodes | "
          f"median {q[0]:.1f} m p90 {q[1]:.1f} m p99 {q[2]:.1f} m | "
          f"{time.time()-t0:.0f}s", flush=True)
    return d


def audit(nodes, g, sources, name, rng):
    """The field must never be SHORTER than the straight line to its own source
    set — a shorter value means the graph short-cut through something it should
    not have. Reported, not asserted, so a build still finishes."""
    from scipy.spatial import cKDTree

    ok = np.isfinite(g)
    if not ok.any():
        print(f"[gw-field] {name}: nothing reachable — cannot audit")
        return {}
    pick = rng.choice(np.flatnonzero(ok), size=min(200_000, int(ok.sum())),
                      replace=False)
    euc = cKDTree(nodes[sources]).query(nodes[pick], workers=-1)[0]
    ratio = g[pick] / np.maximum(euc, 1e-3)
    n_short = int((g[pick] < euc - 1e-3).sum())
    r = np.percentile(ratio[euc > 1.0], [50, 90, 99]) if (euc > 1.0).any() \
        else [np.nan] * 3
    print(f"[gw-field] {name}: geodesic/euclidean ratio (beyond 1 m) "
          f"median {r[0]:.2f} p90 {r[1]:.2f} p99 {r[2]:.2f} | "
          f"{n_short} of {len(pick):,} sampled nodes shorter than the chord "
          f"({'OK' if n_short == 0 else 'INVESTIGATE'})")
    return dict(ratio_median=float(r[0]), ratio_p90=float(r[1]),
                ratio_p99=float(r[2]), n_shorter_than_chord=n_short)


# ---------------------------------------------------------------------------
# audit figure
# ---------------------------------------------------------------------------
def render(nodes, g_all, bases, v, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    cap = 40.0
    sub = rng.permutation(len(nodes))[:600_000]
    p, gv = nodes[sub], g_all[sub]
    fin = np.isfinite(gv)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])

    ax = fig.add_subplot(gs[0, :])
    ax.scatter(p[~fin, 0], p[~fin, 1], s=0.12, c="0.85", marker=".",
               linewidths=0, label="unreachable (no path through the cloud)")
    sc = ax.scatter(p[fin, 0], p[fin, 1], s=0.12, c=np.minimum(gv[fin], cap),
                    cmap="viridis_r", vmin=0, vmax=cap, marker=".", linewidths=0)
    ax.scatter(bases[:, 0], bases[:, 1], s=9, marker="x", c="red", lw=0.6,
               label="tree base (RTK)")
    ax.set_aspect("equal")
    ax.set_title(f"through-cloud distance to the nearest tree base "
                 f"({v:.2f} m graph, edge cap {MAX_EDGE:.2f} m)")
    ax.legend(loc="upper right", markerscale=4, fontsize=8)
    fig.colorbar(sc, ax=ax, label=f"geodesic distance (m, capped at {cap:.0f})",
                 fraction=0.025)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(np.minimum(gv[fin], 120), bins=120, color="#4c78a8")
    ax.set_xlabel("geodesic distance to nearest base (m)")
    ax.set_ylabel("nodes")
    ax.set_title(f"field distribution — {fin.mean()*100:.1f}% of nodes reachable")

    # vertical slice: the field should run DOWN trunks to the base, so a slab
    # cut through a few trees shows dark (near) at ground level and brighter up.
    ax = fig.add_subplot(gs[1, 1])
    if len(bases):
        # Slice the FULL node set, then subsample — slicing the already
        # subsampled 600k leaves a few thousand points in a 3 m slab and the
        # panel comes out nearly empty.
        y0 = float(np.median(bases[:, 1]))
        m_all = np.abs(nodes[:, 1] - y0) < 1.5
        x0 = float(np.median(nodes[m_all, 0])) if m_all.any() else 0.0
        m_all &= np.abs(nodes[:, 0] - x0) < 25
        sl = np.flatnonzero(m_all)
        if len(sl) > 300_000:
            sl = rng.choice(sl, 300_000, replace=False)
        p, gv = nodes[sl], g_all[sl]
        fin = np.isfinite(gv)
        m = np.ones(len(p), bool)
        ax.scatter(p[m & ~fin, 0], p[m & ~fin, 2], s=1.2, c="0.85", marker=".",
                   linewidths=0)
        mm = m & fin
        ax.scatter(p[mm, 0], p[mm, 2], s=1.2, c=np.minimum(gv[mm], cap),
                   cmap="viridis_r", vmin=0, vmax=cap, marker=".", linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"vertical slice at y={y0:.0f} m (x-z)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[gw-field] figure -> {out}")


def main():
    GW.mkdir(parents=True, exist_ok=True)
    if FIELD_NPZ.exists() and not os.environ.get("FORCE_GW_FIELD"):
        print(f"[gw-field] {FIELD_NPZ} exists — reuse (FORCE_GW_FIELD=1 rebuilds)")
        return
    rng = np.random.default_rng(0)

    nodes, v, n_raw = build_nodes()
    from scipy.spatial import cKDTree
    kd = cKDTree(nodes)
    graph = build_graph(nodes, kd)

    lab = np.load(BW / "labels.npz")
    bases, is_val, in_margin = lab["bases"], lab["is_val"], lab["in_margin"]
    sel_train = ~is_val & ~in_margin
    sel_all = ~in_margin
    print(f"[gw-field] bases: {int(sel_train.sum())} train | "
          f"{int(is_val.sum())} val | {int(in_margin.sum())} margin (unused)")

    src_train = seed_nodes(kd, bases[sel_train], "train")
    src_all = seed_nodes(kd, bases[sel_all], "all")
    g_train = geodesic(graph, src_train, "g_train")
    g_all = geodesic(graph, src_all, "g_all")

    stats = dict(train=audit(nodes, g_train, src_train, "g_train", rng),
                 all=audit(nodes, g_all, src_all, "g_all", rng))

    np.savez_compressed(FIELD_NPZ, xyz=nodes.astype(np.float32),
                        g_train=g_train, g_all=g_all,
                        voxel=np.float32(v), max_edge=np.float32(MAX_EDGE),
                        knn=np.int32(KNN), base_r=np.float32(BASE_R))
    meta = dict(voxel_m=float(v), knn=KNN, max_edge_m=MAX_EDGE, base_r_m=BASE_R,
                n_nodes=int(len(nodes)), n_cloud_pts=int(n_raw),
                n_edges=int(graph.nnz),
                reachable_train=float(np.isfinite(g_train).mean()),
                reachable_all=float(np.isfinite(g_all).mean()), audit=stats)
    FIELD_META.write_text(json.dumps(meta, indent=2))
    print(f"[gw-field] wrote {FIELD_NPZ} ({FIELD_NPZ.stat().st_size/1e6:.0f} MB)")

    render(nodes, g_all, bases[sel_all], v, C.RESULTS / "20_geowalker_field.png")
    print("[gw-field] done.")


if __name__ == "__main__":
    main()
