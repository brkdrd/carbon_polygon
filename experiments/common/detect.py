"""Detection primitives: suppression, and the matcher every score here uses.

These live in `common` because the NUMBER has to be the same one. BaseWalker,
GeoWalker, the classical control and the dense stem head are all scored by
`match_metrics`, and a second copy of a greedy matcher that drifted by one
tie-break would silently make those four results incomparable — which is the
only reason any of them is worth reporting.

Nothing here knows about spheres, walks or chunks: detections in, metrics out.
"""
from __future__ import annotations

import numpy as np

def nms(points: np.ndarray, scores: np.ndarray, radius: float = 0.5):
    order = np.argsort(-scores)
    kept = []
    for i in order:
        p = points[i]
        if all(((p - points[j]) ** 2).sum() > radius * radius for j in kept):
            kept.append(i)
    return np.array(kept, dtype=np.int64)


def match_pairs(pred_xyz, pred_score, gt_xyz, radius):
    """Greedy confidence-ordered matching, one GT per detection.

    Returns (pairs, fp_idx, fn_idx): pairs is (K,2) of [pred_i, gt_j], fp_idx
    indexes unmatched detections, fn_idx indexes unmatched GT bases.
    """
    used = np.zeros(len(gt_xyz), bool)
    hit = np.zeros(len(pred_xyz), bool)
    pairs = []
    for i in np.argsort(-pred_score):
        if used.all():          # also covers the no-GT case
            break
        d = np.linalg.norm(gt_xyz - pred_xyz[i][None, :], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= radius:
            used[j] = hit[i] = True
            pairs.append((i, j))
    return (np.array(pairs, np.int64).reshape(-1, 2),
            np.flatnonzero(~hit), np.flatnonzero(~used))


def match_metrics(pred_xyz, pred_score, gt_xyz, radius):
    """Greedy confidence-ordered matching. Returns dict with P/R/F1/RMSE."""
    if len(pred_xyz) == 0:
        return dict(tp=0, fp=0, fn=len(gt_xyz), precision=0.0, recall=0.0,
                    f1=0.0, rmse=float("nan"))
    pairs, fp_i, fn_i = match_pairs(pred_xyz, pred_score, gt_xyz, radius)
    tp, fp, fn = len(pairs), len(fp_i), len(fn_i)
    errs = (np.linalg.norm(pred_xyz[pairs[:, 0]] - gt_xyz[pairs[:, 1]], axis=1)
            if tp else np.zeros(0))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                rmse=float(np.sqrt(np.mean(np.square(errs)))) if tp else float("nan"))
