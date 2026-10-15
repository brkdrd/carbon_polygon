"""Trunk cross-sections in the cloud: per 0.2 m layer, circles fitted to
compact point clusters, chained across layers into trunks.

This is the trunk finder from the circle-fit stage (commit b914631), kept as a
library: on the real scan, when it is on the right trunk its radius matches the
surveyed girth to a median 0.0-0.3 cm (92-97% within 5 cm). What failed was
using it on its own to decide WHICH trunk a survey point meant — it fires on
shrub stems and branches too (0.71 candidates per m2 against 0.067 trees).
prep_bases.py now lets the shifted survey point make that choice and only asks
this module for the circle at the point.

Per layer: points split where gaps exceed GAP; clusters wider than 2 x R_MAX
+ 10 cm (ground, crown, walls) skipped; per cluster RANSAC over 3-point
circles + least-squares refit, accepted only if the radius is R_MIN..R_MAX,
the inliers cover >= ARC degrees and are >= MIN_IN, and the circle is HOLLOW
(point density inside, or in a shell outside for small circles, <= INSIDE x
the density on the ring, against all the layer's points). Chains link
circles whose centres agree within RUN_TOL, bridging up to MAX_GAP failed
layers.
"""
from __future__ import annotations

import os

import numpy as np

GAP = float(os.environ.get("V2_GAP", "0.05"))
R_MIN = float(os.environ.get("V2_R_MIN", "0.02"))
R_MAX = float(os.environ.get("V2_R_MAX", "0.6"))
ARC = float(os.environ.get("V2_ARC", "90"))
MIN_IN = int(os.environ.get("V2_MIN_IN", "12"))
IN_TOL = float(os.environ.get("V2_IN_TOL", "0.02"))
RUN_TOL = float(os.environ.get("V2_RUN_TOL", "0.10"))
MAX_GAP = int(os.environ.get("V2_MAX_GAP", "2"))
ITERS = int(os.environ.get("V2_RANSAC", "200"))
INSIDE = float(os.environ.get("V2_INSIDE", "0.25"))
SMALL_R = float(os.environ.get("V2_SMALL_R", "0.06"))
SHELL = 0.05
LAYER = float(os.environ.get("V2_LAYER", "0.2"))
BH = (1.0, 1.6)


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
