"""StemNet v2: from per-point predictions to tree bases, and scoring them.

Predict a region: overlapping 8 m crops on a 4 m stride; each crop's
predictions count for the points in its inner 6 m only (the rim is context);
a point covered by several crops gets the mean of its predictions.
A crop denser than the point cap is split into disjoint random parts of at
most the cap (training subsamples a dense crop to the cap, so each part looks
like a training crop); every point is predicted. (It used to be subsampled
once, leaving the rest of its points unpredicted.)

Detect (a vote histogram; HDBSCAN was tried and dropped, see below):
  1. points with gate probability >= V2_P shift by their offset (a vote,
     weighted by the probability)
  2. votes summed on a V2_VOTE_CELL (3 cm) grid, smoothed with a Gaussian of
     V2_VOTE_SIGMA (6 cm)
  3. peaks = cells that are the maximum within V2_PEAK_SEP (0.3 m)
  4. each peak's mass = the vote weight within V2_PEAK_R (0.15 m); peaks
     lighter than V2_PEAK_MIN (10) are dropped; the base is the weighted mean
     of those votes, its score the mass
HDBSCAN on the votes failed in two measured ways: a confident model puts all
of a tree's votes in one spot, which HDBSCAN calls noise (perfect predictions
scored 0 bases) or, with the votes counted, splits into one "tree" per cell;
and it slows faster than linearly (20k votes 0.8 s, 50k 5.5 s, 200k > 10 min).

Score: greedy one-to-one matching, highest score first, each detection to the
nearest unmatched truth base within the radius. With a bbox, an unmatched
detection outside it is ignored rather than counted false: the truth list
stops at the bbox edge, and a tree just outside it (its base in the buffer,
its bark inside) is found correctly from the points the region does hold.
"""
from __future__ import annotations

import os

import numpy as np

P_THRESH = float(os.environ.get("V2_P", "0.5"))
VOTE_CELL = float(os.environ.get("V2_VOTE_CELL", "0.03"))
VOTE_SIGMA = float(os.environ.get("V2_VOTE_SIGMA", "0.06"))
PEAK_SEP = float(os.environ.get("V2_PEAK_SEP", "0.3"))
PEAK_R = float(os.environ.get("V2_PEAK_R", "0.15"))
PEAK_MIN = float(os.environ.get("V2_PEAK_MIN", "10"))
CROP, STRIDE, INNER = 8.0, 4.0, 6.0


def detect(xy, p, off, p_thresh=P_THRESH, cell=VOTE_CELL, peak_sep=PEAK_SEP, peak_min=PEAK_MIN):
    """-> (bases (k, 2), score (k,), n_votes (k,))."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    none = np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
    k = p >= p_thresh
    if not k.any():
        return none
    v = xy[k] + off[k]
    w = p[k].astype(np.float64)
    inb = np.all((v >= xy.min(0) - 2) & (v <= xy.max(0) + 2), 1)   # wild votes cannot grow the grid
    v, w = v[inb], w[inb]
    if not len(v):
        return none
    lo = v.min(0) - 3 * VOTE_SIGMA
    ij = np.floor((v - lo) / cell).astype(np.int64)
    shape = tuple(ij.max(0) + 1 + int(np.ceil(3 * VOTE_SIGMA / cell)))
    H = np.zeros(shape, np.float32)
    np.add.at(H, (ij[:, 0], ij[:, 1]), w)
    S = ndimage.gaussian_filter(H, VOTE_SIGMA / cell, mode="constant")
    sep = 2 * int(round(peak_sep / cell)) + 1
    lab, n = ndimage.label((S == ndimage.maximum_filter(S, size=sep, mode="constant")) & (S > 0),
                           structure=np.ones((3, 3)))                  # a flat top is one peak
    if not n:
        return none
    pk = np.array(ndimage.center_of_mass(S, lab, np.arange(1, n + 1)))
    height = ndimage.maximum(S, lab, np.arange(1, n + 1))
    centre = lo + (pk + 0.5) * cell
    kd = cKDTree(v)
    out = []
    for c in centre[np.argsort(-height)]:
        idx = kd.query_ball_point(c, PEAK_R)
        mass = w[idx].sum()
        if mass >= peak_min:
            out.append((*(v[idx] * w[idx, None]).sum(0) / mass, mass, len(idx)))
    if not out:
        return none
    o = np.array(out)
    return o[:, :2], o[:, 2], o[:, 3].astype(int)


def match(det, score, gt, radius, bbox=None, ignore=None):
    """Greedy one-to-one. -> dict tp, fp, fn, precision, recall, f1, rmse, and
    the matched gt index per detection (-1 unmatched, -2 unmatched outside
    bbox, or flagged in `ignore` (per detection), and ignored)."""
    from scipy.spatial import cKDTree
    pair = np.full(len(det), -1)
    used = np.zeros(len(gt), bool)
    if len(det) and len(gt):
        kd = cKDTree(gt)
        for i in np.argsort(-score):
            for j in sorted(kd.query_ball_point(det[i], radius), key=lambda j: np.hypot(*(gt[j] - det[i]))):
                if not used[j]:
                    used[j] = True; pair[i] = j
                    break
    if bbox is not None and len(det):
        out = (det[:, 0] < bbox[0]) | (det[:, 0] > bbox[1]) | (det[:, 1] < bbox[2]) | (det[:, 1] > bbox[3])
        pair[(pair < 0) & out] = -2
    if ignore is not None and len(det):
        pair[(pair < 0) & ignore] = -2
    tp = int((pair >= 0).sum()); fp = int((pair == -1).sum()); fn = len(gt) - tp
    p = tp / max(tp + fp, 1); r = tp / max(len(gt), 1)
    err = [np.hypot(*(det[i] - gt[pair[i]])) for i in range(len(det)) if pair[i] >= 0]
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9),
                rmse=float(np.sqrt(np.mean(np.square(err)))) if err else float("nan"), pair=pair)


def predict_region(predict_crop, store, bbox, cap, rng, full=False):
    """Mean predictions for every point in bbox covered by an inner square.
    predict_crop(crop) -> (p (n,), off (n, 2)[, tree (n,)]). Returns xy, p, off;
    with full=True a dict that also carries z, the mean tree probability (when
    predict_crop gives one), the points' tree / non-tree labels (sem), and
    each point's storage tile id and row in it (tile, row).

    Accumulates into one slot per tile point (sum of p, of offsets, count), so
    a region of tens of millions of points costs ~13 bytes per point and no
    per-point Python."""
    x0, x1, y0, y1 = bbox
    acc = {}

    def accumulate(c, cx, cy):
        out = predict_crop(c)
        p, off = out[0], out[1]
        pt = out[2] if len(out) > 2 else None
        inner = (np.abs(c["xyz"][:, 0] - cx) <= INNER / 2) & (np.abs(c["xyz"][:, 1] - cy) <= INNER / 2)
        for t in np.unique(c["tile"][inner]):
            m = inner & (c["tile"] == t)
            if t not in acc:
                n = len(store.tile((int(t) // 100000, int(t) % 100000))["xyz"])
                acc[t] = [np.zeros(n, np.float32), np.zeros((n, 2), np.float32), np.zeros(n, np.uint16),
                          np.zeros(n, np.float32) if pt is not None else None]
            r = c["row"][m]                          # unique within one crop
            acc[t][0][r] += p[m]; acc[t][1][r] += off[m]; acc[t][2][r] += 1
            if pt is not None:
                acc[t][3][r] += pt[m]

    xs = np.arange(x0 + INNER / 2, x1 + STRIDE / 2, STRIDE)
    ys = np.arange(y0 + INNER / 2, y1 + STRIDE / 2, STRIDE)
    for cx in xs:
        for cy in ys:
            c0 = store.crop(cx, cy, CROP)
            if c0 is None or len(c0["xyz"]) < 64:
                continue
            n0 = len(c0["xyz"])
            # denser than the cap: disjoint random parts, each one like a training crop, so every point is predicted
            parts = [None] if n0 <= cap else np.array_split(rng.permutation(n0), int(np.ceil(n0 / cap)))
            for sel in parts:
                c = c0 if sel is None else {k: v[sel] for k, v in c0.items()}
                accumulate(c, cx, cy)
    XY, Z, P, O, PT, SEM, TI, RW = [], [], [], [], [], [], [], []
    for t, (sp, so, n, spt) in acc.items():
        k = n > 0
        tile = store.tile((int(t) // 100000, int(t) % 100000))
        xyz = tile["xyz"]
        XY.append(xyz[k, :2]); P.append(sp[k] / n[k]); O.append(so[k] / n[k, None])
        if full:
            Z.append(np.asarray(xyz[k, 2], np.float32))
            PT.append(spt[k] / n[k] if spt is not None else np.full(int(k.sum()), np.nan, np.float32))
            SEM.append(np.asarray(tile["sem"])[k] if "sem" in tile else np.full(int(k.sum()), -1, np.int8))
            TI.append(np.full(int(k.sum()), t, np.int64)); RW.append(np.flatnonzero(k))
    if not XY:
        if full:
            return dict(xy=np.zeros((0, 2)), z=np.zeros(0), p=np.zeros(0), off=np.zeros((0, 2)),
                        tree=np.zeros(0), sem=np.zeros(0, np.int8), tile=np.zeros(0, np.int64),
                        row=np.zeros(0, np.int64))
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2))
    xy, p, off = np.concatenate(XY), np.concatenate(P), np.concatenate(O)
    inb = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    if full:
        return dict(xy=xy[inb], z=np.concatenate(Z)[inb], p=p[inb], off=off[inb],
                    tree=np.concatenate(PT)[inb], sem=np.concatenate(SEM)[inb],
                    tile=np.concatenate(TI)[inb], row=np.concatenate(RW)[inb])
    return xy[inb], p[inb], off[inb]
