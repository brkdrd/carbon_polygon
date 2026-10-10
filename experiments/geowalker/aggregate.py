#!/usr/bin/env python3
"""What to do with a walk after it has finished — the evidence NMS throws away.

Three things are known at the end of inference and none of them reach the
detection list:

  1. WHERE THE WALK WAS, not just where it stopped. The read-out scores every
     one of the GW_STEPS+1 positions; only the last survived. A walker that
     stood on a base at step 5 and drifted off by step 8 was recorded as
     wherever it drifted to, and counted as a miss.

  2. HOW MANY WALKERS AGREED. Thousands of seeds converge onto hundreds of
     bases, and NMS keeps the single highest-scoring endpoint of each pile and
     deletes the rest. But the size of the pile is itself the measurement: a
     stem pulls walkers in from every direction, a bush catches the two that
     started next to it. Keeping one endpoint per cluster throws away a vote
     count and keeps a single noisy score in its place.

  3. WHAT IS STANDING ABOVE. gw_height derived the annotators' effective height
     cut from the labels themselves (the 2nd percentile of canopy height over a
     labelled base) and showed the 2 m sphere cannot see it. The CHM it leaves
     behind answers the question directly, per candidate, for free.

None of this needs the walker retrained, and none of it is a new model. It is
the same walk, read properly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402

GW = C.DATA / "geowalker"
CHM_NPZ = GW / "chm.npz"

CLUSTER_R = float(os.environ.get("GW_AGG_R", "0.5"))      # vote radius, m
V0 = float(os.environ.get("GW_AGG_V0", "4"))              # vote half-saturation
Z_Q = float(os.environ.get("GW_AGG_ZQ", "25"))            # z percentile of a pile
CUT_PCT = float(os.environ.get("GW_AGG_CUT_PCT", "2"))    # height cut percentile
HEIGHT_R = float(os.environ.get("GW_AGG_HEIGHT_R", "1.5"))  # CHM max-pool radius


# ---------------------------------------------------------------------------
def best_along_walk(traj, traj_score):
    """The best position each walker ever stood at, not the one it ended at.

    traj (K,N,3), traj_score (K,N) -> (N,3), (N,). The walk is a search; its
    last step is not privileged, and the read-out already judged every step.
    """
    traj = np.asarray(traj)
    traj_score = np.asarray(traj_score)
    k = traj_score.argmax(axis=0)                       # (N,)
    n = np.arange(traj.shape[1])
    return traj[k, n, :].copy(), traj_score[k, n].copy()


def cluster_votes(xyz, score, radius=CLUSTER_R, z_q=Z_Q):
    """Greedy agglomeration in PLAN VIEW: every walker in a pile votes.

    Same sweep as NMS — highest score first, absorb everything within `radius` —
    but the absorbed walkers are counted instead of deleted. Plan view, because
    walkers converging on one stem can stop at different heights on it and a 3D
    radius would split the pile in two.

    Returns a dict of per-cluster arrays: xyz, n_votes, score_max, score_mean,
    spread (mean horizontal distance of members to the centre).

    The centre is the score-weighted mean of the members in xy; in z it is a low
    percentile, because a base sits at the bottom of the pile and the mean of a
    column of walkers floats above it.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, np.float64)
    score = np.asarray(score, np.float64)
    if len(xyz) == 0:
        return dict(xyz=np.zeros((0, 3)), n_votes=np.zeros(0, int),
                    score_max=np.zeros(0), score_mean=np.zeros(0),
                    spread=np.zeros(0))

    tree = cKDTree(xyz[:, :2])
    taken = np.zeros(len(xyz), bool)
    out_xyz, n_votes, s_max, s_mean, spread = [], [], [], [], []
    for i in np.argsort(-score):
        if taken[i]:
            continue
        m = np.fromiter((j for j in tree.query_ball_point(xyz[i, :2], radius)
                         if not taken[j]), int)
        if len(m) == 0:                                  # i itself is always in
            m = np.array([i])
        taken[m] = True
        w = score[m]
        wsum = w.sum()
        cxy = (xyz[m, :2] * w[:, None]).sum(0) / wsum if wsum > 0 \
            else xyz[m, :2].mean(0)
        cz = np.percentile(xyz[m, 2], z_q)
        out_xyz.append([cxy[0], cxy[1], cz])
        n_votes.append(len(m))
        s_max.append(w.max())
        s_mean.append(w.mean())
        spread.append(float(np.linalg.norm(xyz[m, :2] - cxy, axis=1).mean()))
    return dict(xyz=np.array(out_xyz), n_votes=np.array(n_votes, int),
                score_max=np.array(s_max), score_mean=np.array(s_mean),
                spread=np.array(spread))


def vote_confidence(cl, v0=V0):
    """One number per cluster, from its score and how many walkers agreed.

        conf = score_max * n / (n + v0)

    Monotone in both, with one interpretable knob: v0 is the vote count at which
    a cluster keeps half its score. Nothing here is fitted — a fitted weighting
    would need the val labels, and then the threshold sweep that follows would
    be choosing on data it had already seen.
    """
    n = cl["n_votes"].astype(np.float64)
    return cl["score_max"] * n / (n + v0)


# ---------------------------------------------------------------------------
def load_chm():
    """The canopy height raster gw_height leaves behind, or None."""
    if not CHM_NPZ.exists():
        return None
    return dict(np.load(CHM_NPZ))


def canopy_height(chm, xy, r=HEIGHT_R):
    """Tallest canopy within r metres of each xy (gw_height's own lookup)."""
    import height as H
    return H.height_at(chm, np.asarray(xy, np.float64), r=r)


def derive_cut(chm, gt_xy, pct=CUT_PCT, r=HEIGHT_R):
    """The annotators' effective height cut, from the labels and nothing else.

    Every labelled base passed their test, so the low tail of canopy height over
    a labelled base is where the cut fell. Derived here rather than read from
    results/ so the gate cannot be applied with a threshold from another run.
    """
    h = canopy_height(chm, gt_xy, r=r)
    h = h[np.isfinite(h)]
    return float(np.percentile(h, pct)) if len(h) else float("nan")


def height_gate(chm, xy, cut, r=HEIGHT_R):
    """Keep what stands under canopy at least `cut` tall.

    Unknown height (outside the raster) is kept: the gate removes candidates it
    can prove are too short, and nothing else.
    """
    h = canopy_height(chm, xy, r=r)
    return (~np.isfinite(h)) | (h >= cut), h


# ---------------------------------------------------------------------------
def aggregate_walks(traj, traj_score, gt_xy=None, chm=None, radius=CLUSTER_R,
                    v0=V0, use_best_step=True, verbose=True):
    """The whole stage: best step -> votes -> height gate.

    Returns (xyz, conf, info). `info` carries the per-cluster arrays and what
    each stage did, so a run can report where its detections went rather than
    only how many came out.
    """
    if use_best_step:
        xyz, sc = best_along_walk(traj, traj_score)
    else:
        xyz, sc = np.asarray(traj[-1]), np.asarray(traj_score[-1])
    info = dict(n_walkers=int(len(xyz)), use_best_step=bool(use_best_step))

    cl = cluster_votes(xyz, sc, radius=radius)
    conf = vote_confidence(cl, v0=v0)
    info.update(n_clusters=int(len(conf)),
                votes_p50=float(np.median(cl["n_votes"])) if len(conf) else 0.0,
                votes_max=int(cl["n_votes"].max()) if len(conf) else 0)
    if verbose:
        print(f"[gw-agg] {len(xyz):,} walkers -> {len(conf):,} clusters "
              f"(votes: median {info['votes_p50']:.0f}, max {info['votes_max']})",
              flush=True)

    xyz_out = cl["xyz"]
    keep = np.ones(len(xyz_out), bool)
    if chm is not None and gt_xy is not None and len(xyz_out):
        cut = derive_cut(chm, gt_xy)
        keep, h = height_gate(chm, xyz_out[:, :2], cut)
        info.update(height_cut_m=cut, n_dropped_short=int((~keep).sum()),
                    height_known=float(np.isfinite(h).mean()))
        if verbose:
            print(f"[gw-agg] canopy cut {cut:.1f} m (p{CUT_PCT:g} over labelled "
                  f"bases): drops {int((~keep).sum()):,} of {len(keep):,} "
                  f"clusters standing under something too short to be labelled",
                  flush=True)
    elif verbose:
        print("[gw-agg] no chm.npz — height gate skipped (run gw_height)",
              flush=True)

    info["clusters"] = {k: v[keep] for k, v in cl.items()}
    return xyz_out[keep], conf[keep], info
