#!/usr/bin/env python3
"""GeoWalker diagnostics — where does recall actually go?

Training showed the walk working (median through-cloud distance 3.2 m -> 0.5 m,
density penalty ~0, no escapes) while detection F1 sat at ~0.36 with
precision 0.53 and recall 0.27. Recall is the binding constraint, and it can be
lost in exactly three places. This measures each one separately, on a trained
checkpoint, without retraining:

  1. WALKING   — does any walker endpoint land within MATCH_R of the base at all?
                 This is the ceiling: no later stage can recover a base that no
                 walker ever reached.
  2. NMS       — of the bases that were reached, how many survive the greedy
                 radius suppression? A base loses here when a higher-scoring
                 endpoint elsewhere absorbs the one that found it.
  3. THRESHOLD — of those, how many clear the score cut? A base lost here means
                 the read-out head under-rates a walker that actually arrived.

It also profiles the walk itself per step (does the walker decelerate on arrival
or orbit the base forever?) and scores the read-out head as a ranker (AUC of
"this endpoint is within MATCH_R of a base"), which says whether the head is
worth improving or is already close to the information it has.

Read-only: loads the checkpoint, writes a JSON + a figure, touches no weights.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# this directory first (its siblings are model.py/geodesic.py), then /app
# for the shared modules — /app also holds BaseWalker's train.py/infer.py,
# which must never shadow ours.
sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402
import model as M  # noqa: E402

BW = C.DATA / "basewalker"
GW = C.DATA / "geowalker"
CKPT = C.MODELS / "geowalker_decoder.pth"

MATCH_R = float(os.environ.get("GW_DIAG_MATCH_R", "0.5"))
N_SEEDS = int(os.environ.get("GW_DIAG_SEEDS", "8192"))    # 0 = every val seed
N_TRACE = int(os.environ.get("GW_DIAG_TRACE", "512"))     # seeds for the profile


def auc(pos_scores, neg_scores):
    """ROC-AUC of the read-out head as a ranker, without sklearn."""
    from scipy.stats import rankdata
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos_scores, neg_scores]))
    n_p, n_n = len(pos_scores), len(neg_scores)
    return float((r[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def covered(points, gt, r):
    """Fraction of gt bases with at least one of `points` within r."""
    from scipy.spatial import cKDTree
    if len(points) == 0 or len(gt) == 0:
        return 0.0, np.zeros(len(gt), bool)
    hit = cKDTree(points[:, :3]).query_ball_point(gt[:, :3], r=r,
                                                  return_length=True) > 0
    return float(hit.mean()), hit


def step_profile(dec, enc, scene, dem, field, seeds, out_json):
    """Per-step displacement and field value. A walker that arrives and settles
    shows a decaying step norm; one that orbits shows a flat one."""
    s = torch.from_numpy(seeds.astype(np.float32)).to(scene.device)
    with torch.no_grad():
        _, _, _, info = M.rollout(dec, enc, scene, None, dem,
                                  dem.project(s).detach(), train=False,
                                  record=True)
    traj = info["traj"]                                    # (steps+1, B, 3)
    disp = np.linalg.norm(np.diff(traj, axis=0), axis=-1)  # (steps, B)
    prof = []
    for k in range(traj.shape[0]):
        pos = torch.from_numpy(traj[k]).to(scene.device)
        with torch.no_grad():
            gbar, dens, _, fin, _ = M.field_terms(field, pos)
        f = gbar[fin].median().item() if int(fin.sum()) else float("nan")
        prof.append(dict(step=k, field_median=f,
                         dens_mean=float(dens.mean()),
                         disp_mean=float(disp[k].mean()) if k < len(disp) else 0.0))
        print(f"[gw-diag]   step {k}: field {f:5.2f} m | "
              f"moved {prof[-1]['disp_mean']:5.3f} m | dens {prof[-1]['dens_mean']:.3f}")
    out_json["step_profile"] = prof
    return traj, disp, prof


def main():
    dev = "cuda"
    if not CKPT.exists():
        raise SystemExit(f"[gw-diag] {CKPT} missing — run gw_train first")
    scene = M.load_scene(BW / "tiles", dev)
    dem = M.Dem(BW / "dem.npz", dev)
    field = M.load_field(GW / "field.npz", "g_all", dev)

    lab = np.load(BW / "labels.npz")
    gt = lab["bases"][lab["is_val"]].astype(np.float32)
    sd = np.load(BW / "seeds.npz")
    va_seeds = sd["seeds"][sd["is_val"]].astype(np.float32)

    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    dec = M.GeoWalkerDecoder().to(dev)
    dec.load_state_dict(ck["state_dict"])
    dec.eval()
    thresh = float(os.environ.get("GW_CONF_THRESH",
                                  str(ck.get("val", {}).get("thresh", 0.5))))
    print(f"[gw-diag] checkpoint it {ck.get('it')} | val F1 "
          f"{ck.get('val', {}).get('f1', float('nan')):.3f} | thresh {thresh}")

    if N_SEEDS and len(va_seeds) > N_SEEDS:
        va_seeds = va_seeds[np.random.RandomState(0).permutation(len(va_seeds))[:N_SEEDS]]
    print(f"[gw-diag] {len(va_seeds):,} val seeds | {len(gt)} val GT bases")

    enc = M.load_encoder()
    ends, scores = M.detect(dec, enc, scene, dem, va_seeds)

    out = dict(checkpoint=dict(it=ck.get("it"), val=ck.get("val")),
               match_r=MATCH_R, n_seeds=int(len(va_seeds)), n_gt=int(len(gt)))

    # ---- the three-stage recall decomposition -----------------------------
    ceil_r, ceil_hit = covered(ends, gt, MATCH_R)
    keep = M.nms(ends, scores, radius=M.NMS_R)
    nms_r, nms_hit = covered(ends[keep], gt, MATCH_R)
    det = keep[scores[keep] >= thresh]
    thr_r, thr_hit = covered(ends[det], gt, MATCH_R)
    final = M.match_metrics(ends[det], scores[det], gt, radius=MATCH_R)

    print(f"\n[gw-diag] ===== where recall goes =====")
    print(f"  reached by ANY walker        {ceil_r*100:5.1f}%   <- ceiling set by WALKING")
    print(f"  still covered after NMS      {nms_r*100:5.1f}%   (lost {(ceil_r-nms_r)*100:4.1f} pts)")
    print(f"  still covered after thresh   {thr_r*100:5.1f}%   (lost {(nms_r-thr_r)*100:4.1f} pts)")
    print(f"  final greedy-matched recall  {final['recall']*100:5.1f}%   "
          f"(lost {(thr_r-final['recall'])*100:4.1f} pts to one-GT-per-detection)")
    print(f"  precision {final['precision']:.3f} | F1 {final['f1']:.3f} | "
          f"rmse {final['rmse']:.3f} m")
    out["recall_stages"] = dict(walking_ceiling=ceil_r, after_nms=nms_r,
                                after_thresh=thr_r, final=final["recall"],
                                precision=final["precision"], f1=final["f1"],
                                rmse=final["rmse"])

    # ---- is the read-out head a useful ranker? ----------------------------
    from scipy.spatial import cKDTree
    d_end = cKDTree(gt[:, :3]).query(ends[:, :3])[0]
    near = d_end <= MATCH_R
    a = auc(scores[near], scores[~near])
    print(f"\n[gw-diag] endpoints within {MATCH_R} m of a base: "
          f"{near.mean()*100:.1f}% of {len(ends):,}")
    print(f"  score AUC (near vs far endpoints): {a:.3f}   "
          f"({'useful ranker' if a > 0.7 else 'WEAK — the head is a bottleneck'})")
    print(f"  median score  near {np.median(scores[near]):.3f} | "
          f"far {np.median(scores[~near]):.3f}")
    out["readout"] = dict(auc=a, frac_near=float(near.mean()),
                          median_score_near=float(np.median(scores[near])),
                          median_score_far=float(np.median(scores[~near])))

    # ---- pile-up: are walkers wasted on bases already found? --------------
    per_gt = cKDTree(ends[:, :3]).query_ball_point(gt[:, :3], r=MATCH_R,
                                                   return_length=True)
    reached = per_gt[per_gt > 0]
    print(f"\n[gw-diag] endpoints per REACHED base: median {np.median(reached):.0f} "
          f"| p90 {np.percentile(reached, 90):.0f} | max {per_gt.max()}")
    print(f"  bases with zero endpoints nearby: {int((per_gt == 0).sum())} "
          f"of {len(gt)}")
    out["pileup"] = dict(median_per_reached=float(np.median(reached)) if len(reached) else 0.0,
                         p90=float(np.percentile(reached, 90)) if len(reached) else 0.0,
                         n_unreached=int((per_gt == 0).sum()))

    # ---- does the walk settle, or orbit? ----------------------------------
    print(f"\n[gw-diag] ===== per-step walk profile ({N_TRACE} seeds) =====")
    trace = va_seeds[np.random.RandomState(1).permutation(len(va_seeds))[:N_TRACE]]
    traj, disp, prof = step_profile(dec, enc, scene, dem, field, trace, out)
    settle = prof[-1]["disp_mean"] / max(prof[0]["disp_mean"], 1e-6)
    print(f"  last step / first step displacement ratio: {settle:.2f}   "
          f"({'settles on arrival' if settle < 0.6 else 'STILL MOVING — no stop behaviour'})")
    out["settle_ratio"] = float(settle)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].hist(np.minimum(d_end, 5.0), bins=100, color="#4c78a8")
    ax[0].axvline(MATCH_R, c="red", ls="--", label=f"match radius {MATCH_R} m")
    ax[0].set_xlabel("endpoint -> nearest GT base (m)")
    ax[0].set_ylabel("endpoints")
    ax[0].set_title(f"where walkers stop ({near.mean()*100:.0f}% inside the radius)")
    ax[0].legend()

    bins = np.linspace(0, 1, 40)
    ax[1].hist(scores[near], bins=bins, alpha=0.65, label=f"within {MATCH_R} m",
               color="#2ca02c", density=True)
    ax[1].hist(scores[~near], bins=bins, alpha=0.65, label="further",
               color="#d62728", density=True)
    ax[1].axvline(thresh, c="black", ls="--", label=f"thresh {thresh}")
    ax[1].set_xlabel("read-out score")
    ax[1].set_title(f"is the score separable?  AUC {a:.3f}")
    ax[1].legend()

    k = np.arange(len(prof))
    ax[2].plot(k, [p["field_median"] for p in prof], "o-", c="#4c78a8",
               label="median field value (m)")
    ax[2].plot(k, [p["disp_mean"] for p in prof], "s-", c="#ff7f0e",
               label="mean displacement (m)")
    ax[2].set_xlabel("step")
    ax[2].set_title("does the walk converge and settle?")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.RESULTS / "23_geowalker_diagnostics.png", dpi=130,
                bbox_inches="tight")
    (C.RESULTS / "geowalker_diagnostics.json").write_text(json.dumps(out, indent=2))
    print(f"\n[gw-diag] -> results/23_geowalker_diagnostics.png")
    print(f"[gw-diag] -> results/geowalker_diagnostics.json")

    # ---- the verdict -------------------------------------------------------
    print("\n[gw-diag] ===== verdict =====")
    if ceil_r < 0.5:
        print(f"  WALKING is the ceiling ({ceil_r*100:.0f}% of bases ever reached).")
        print( "  Tuning NMS or the score head cannot get past it. Look at step")
        print( "  count/length (GW_STEPS, GW_MAX_STEP), seed density, and whether")
        print( "  the loss minimum actually sits ON the base (GW_GEO_SIGMA).")
    elif ceil_r - nms_r > 0.10:
        print(f"  NMS is eating {(ceil_r-nms_r)*100:.0f} points of recall.")
        print( "  Walkers do arrive; suppression discards them. Try a smaller")
        print( "  GW_NMS_R, or score-aware clustering instead of greedy NMS.")
    elif nms_r - thr_r > 0.10:
        print(f"  THE SCORE HEAD is eating {(nms_r-thr_r)*100:.0f} points of recall.")
        print( "  Walkers arrive and survive NMS but are under-rated. Raise")
        print( "  GW_W_AUX, or lower the threshold / widen the sweep.")
    else:
        print( "  Recall is spread evenly across the three stages — no single")
        print( "  dominant loss. The walk itself is the thing to improve.")


if __name__ == "__main__":
    main()
