#!/usr/bin/env python3
"""GeoWalker stage C — walk every seed, NMS, score against the RTK bases.

No distance field is used here: at inference the model only ever sees the sphere
around its current position (plus its last GW_MEM spheres), which is the whole
point. Detections are walker endpoints, scored by the read-out head's own
estimate of how far it still is from a base through the cloud.

Outputs:
  /results/geowalker_detections.csv       x,y,z,score (local frame), post-NMS
  /results/geowalker_metrics.json         P/R/F1/RMSE at 0.5 m and 1.0 m
  /results/21_geowalker_detections.png    top-down: GT vs detections
  /results/22_geowalker_walks.png         sample walks over the distance field
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
N_WALKS = int(os.environ.get("GW_N_WALKS", "300"))
# surveyed-region grid: cell size and how far to dilate around a labelled base
SURVEY_CELL = float(os.environ.get("GW_SURVEY_CELL", "10.0"))
SURVEY_DILATE = int(os.environ.get("GW_SURVEY_DILATE", "2"))


def detections_figure(scene, seeds, gt, det_xyz, f1, region, out):
    xyz = scene.xyz.cpu().numpy()
    x0, x1 = seeds[:, 0].min() - 5, seeds[:, 0].max() + 5
    y0, y1 = seeds[:, 1].min() - 5, seeds[:, 1].max() + 5
    m = (xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1)
    sub = xyz[m][np.random.permutation(int(m.sum()))[:400_000]]
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(sub[:, 0], sub[:, 1], s=0.05, c="0.75", marker=".", linewidths=0)
    ax.scatter(gt[:, 0], gt[:, 1], s=42, marker="x", c="green", label="GT base")
    ax.scatter(det_xyz[:, 0], det_xyz[:, 1], s=26, marker="o",
               facecolors="none", edgecolors="red", label="detection")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"GeoWalker detections ({region}) | F1@0.5m {f1:.3f} | "
                 f"{len(det_xyz)} det / {len(gt)} GT")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[gw-infer] image -> {out}")


def walks_figure(dec, enc, scene, dem, field_npz, seeds, gt, out):
    """A handful of walks drawn on top of the distance field they descend.

    Restricted to one WIN-metre window around the densest patch of ground truth,
    or the walks would be single pixels spread over the whole campus."""
    win_m = float(os.environ.get("GW_WALK_WIN", "40.0"))
    if len(gt):
        # densest GT window: the base whose WIN-neighbourhood holds the most
        from scipy.spatial import cKDTree
        n = cKDTree(gt[:, :2]).query_ball_point(gt[:, :2], r=win_m / 2,
                                                return_length=True)
        cx, cy = gt[int(np.argmax(n)), :2]
    else:
        cx, cy = seeds[:, :2].mean(0)
    inw = ((np.abs(seeds[:, 0] - cx) < win_m / 2)
           & (np.abs(seeds[:, 1] - cy) < win_m / 2))
    pool = seeds[inw] if inw.sum() >= 10 else seeds
    gt = gt[(np.abs(gt[:, 0] - cx) < win_m / 2) & (np.abs(gt[:, 1] - cy) < win_m / 2)] \
        if len(gt) else gt
    sub = pool[np.random.permutation(len(pool))[:N_WALKS]].astype(np.float32)
    s = torch.from_numpy(sub).to(scene.device)
    with torch.no_grad():
        _, _, score, info = M.rollout(dec, enc, scene, None, dem,
                                      dem.project(s).detach(), train=False,
                                      record=True)
    traj = info["traj"]                                   # (steps+1, B, 3)
    sc = score.cpu().numpy()

    d = np.load(field_npz)
    nodes, g = d["xyz"], d["g_all"]
    pad = 6.0
    win = ((nodes[:, 0] > sub[:, 0].min() - pad) & (nodes[:, 0] < sub[:, 0].max() + pad)
           & (nodes[:, 1] > sub[:, 1].min() - pad) & (nodes[:, 1] < sub[:, 1].max() + pad))
    nw, gw = nodes[win], g[win]
    if len(nw) > 400_000:
        keep = np.random.permutation(len(nw))[:400_000]
        nw, gw = nw[keep], gw[keep]
    fin = np.isfinite(gw)

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(nw[~fin, 0], nw[~fin, 1], s=0.2, c="0.9", marker=".", linewidths=0)
    f = ax.scatter(nw[fin, 0], nw[fin, 1], s=0.2, c=np.minimum(gw[fin], 20),
                   cmap="viridis_r", vmin=0, vmax=20, marker=".", linewidths=0)
    fig.colorbar(f, ax=ax, label="through-cloud distance to a base (m)",
                 fraction=0.03)
    ax.plot(traj[:, :, 0], traj[:, :, 1], lw=0.7, c="0.25", alpha=0.7, zorder=3)
    ax.scatter(traj[0, :, 0], traj[0, :, 1], s=7, c="black", marker=".",
               zorder=4, label="seed")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=34, c=sc, cmap="autumn_r",
               vmin=0, vmax=1, marker="o", edgecolors="k", lw=0.3, zorder=5,
               label="endpoint (colour = score)")
    ax.scatter(gt[:, 0], gt[:, 1], s=60, marker="x", c="red", lw=1.2, zorder=6,
               label="GT base")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=2, fontsize=9)
    ax.set_title(f"{len(sub)} walks ({M.N_STEPS} steps, {M.MAX_STEP} m cap) "
                 f"over the field they were trained to descend")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[gw-infer] image -> {out}")


def main():
    dev = "cuda"
    scene = M.load_scene(BW / "tiles", dev)
    dem = M.Dem(BW / "dem.npz", dev)
    lab = np.load(BW / "labels.npz")
    bases = lab["bases"]
    sd = np.load(BW / "seeds.npz")

    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    dec = M.GeoWalkerDecoder().to(dev)
    dec.load_state_dict(ck["state_dict"])
    dec.eval()
    thresh = float(os.environ.get("GW_CONF_THRESH",
                                  str(ck.get("val", {}).get("thresh", 0.5))))
    print(f"[gw-infer] decoder from it {ck.get('it')} | score thresh {thresh}")

    region = os.environ.get("GW_INFER_REGION", "val")      # val | all
    if region == "val":
        seeds = sd["seeds"][sd["is_val"]]
        gt = bases[lab["is_val"]]
    else:
        seeds = sd["seeds"]
        gt = bases[~lab["in_margin"]]
    print(f"[gw-infer] region={region}: {len(seeds):,} seeds | {len(gt)} GT bases")

    enc = M.load_encoder()
    ends, scores = M.detect(dec, enc, scene, dem, seeds.astype(np.float32))
    keep = M.nms(ends, scores, radius=M.NMS_R)
    kept_xyz, kept_s = ends[keep], scores[keep]
    print(f"[gw-infer] {len(kept_xyz)} endpoints survive NMS "
          f"(scores {kept_s.min():.3f}-{kept_s.max():.3f})")

    # Which detections are even judgeable: the RTK survey does not cover the
    # whole scanned scene, and a real tree nobody walked to is not a model
    # error. Scored both ways so the unmasked number stays comparable to
    # BaseWalker's.
    inside = M.surveyed_mask(gt[:, :2], cell=SURVEY_CELL, dilate=SURVEY_DILATE)
    in_survey = inside(kept_xyz[:, :2])
    print(f"[gw-infer] inside the surveyed region: {in_survey.sum()} of "
          f"{len(kept_xyz)} endpoints ({in_survey.mean()*100:.0f}%)")

    # Re-tune the cut HERE: the checkpoint's threshold was chosen at eval-seed
    # density, and the number of detections scales with the seed count, so it
    # does not transfer to a full run.
    sweep = []
    for thr in M.sweep_thresholds(kept_s):
        k = kept_s >= thr
        sweep.append(dict(
            thresh=float(thr), n=int(k.sum()),
            all=M.match_metrics(kept_xyz[k], kept_s[k], gt, radius=0.5),
            surveyed=M.match_metrics(kept_xyz[k & in_survey], kept_s[k & in_survey],
                                     gt, radius=0.5)))
    best = max(sweep, key=lambda r: r["all"]["f1"])
    best_s = max(sweep, key=lambda r: r["surveyed"]["f1"])
    print(f"[gw-infer] best F1@0.5m over the sweep: {best['all']['f1']:.3f} "
          f"at thresh {best['thresh']:.3f} (checkpoint said {thresh:.3f})")
    print(f"[gw-infer]   inside the surveyed region: "
          f"{best_s['surveyed']['f1']:.3f} at thresh {best_s['thresh']:.3f}")

    if os.environ.get("GW_USE_BEST_THRESH", "1") != "0":
        thresh = best["thresh"]
    det = keep[scores[keep] >= thresh]
    det_xyz, det_s = ends[det], scores[det]
    print(f"[gw-infer] {len(det)} detections at thresh {thresh:.3f}")

    metrics = {}
    for r in (0.5, 1.0):
        metrics[f"@{r}m"] = m = M.match_metrics(det_xyz, det_s, gt, radius=r)
        ins = inside(det_xyz[:, :2])
        ms = M.match_metrics(det_xyz[ins], det_s[ins], gt, radius=r)
        metrics[f"@{r}m_surveyed"] = ms
        print(f"[gw-infer] @{r}m: P {m['precision']:.3f} R {m['recall']:.3f} "
              f"F1 {m['f1']:.3f} rmse {m['rmse']:.2f} m "
              f"(tp {m['tp']} fp {m['fp']} fn {m['fn']})")
        print(f"[gw-infer]   surveyed-only: P {ms['precision']:.3f} "
              f"R {ms['recall']:.3f} F1 {ms['f1']:.3f}")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    # EVERY post-NMS endpoint, with its score and whether it is judgeable — so
    # the threshold can be re-tuned offline without another GPU run.
    np.savetxt(C.RESULTS / "geowalker_endpoints.csv",
               np.column_stack([kept_xyz, kept_s, in_survey.astype(np.float32)]),
               delimiter=",", header="x,y,z,score,in_surveyed", comments="")
    np.savetxt(C.RESULTS / "geowalker_detections.csv",
               np.column_stack([det_xyz, det_s]), delimiter=",",
               header="x,y,z,score", comments="")
    (C.RESULTS / "geowalker_metrics.json").write_text(json.dumps(
        dict(region=region, thresh=thresh, n_det=int(len(det)), n_gt=int(len(gt)),
             n_seeds=int(len(seeds)), n_endpoints=int(len(kept_xyz)),
             metrics=metrics, threshold_sweep=sweep,
             surveyed=dict(cell=SURVEY_CELL, dilate=SURVEY_DILATE,
                           frac_endpoints_inside=float(in_survey.mean())),
             checkpoint=dict(it=ck.get("it"), val=ck.get("val"))), indent=2))

    detections_figure(scene, seeds, gt, det_xyz, metrics["@0.5m"]["f1"], region,
                      C.RESULTS / "21_geowalker_detections.png")
    field_npz = GW / "field.npz"
    if field_npz.exists():
        walks_figure(dec, enc, scene, dem, field_npz, seeds, gt,
                     C.RESULTS / "22_geowalker_walks.png")
    else:
        print(f"[gw-infer] {field_npz} missing — skipping the walks figure")


if __name__ == "__main__":
    main()
