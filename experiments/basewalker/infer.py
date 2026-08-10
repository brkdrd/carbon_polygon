#!/usr/bin/env python3
"""BaseWalker inference: walk every seed, NMS, score against the labels.

Outputs:
  /results/basewalker_detections.csv     x,y,z,score (local frame), post-NMS
  /results/basewalker_metrics.json       P/R/F1/RMSE at 0.5 m and 1.0 m
  /results/10_basewalker_detections.png  top-down: GT vs detections (val block)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/app")
from common import config as C
import walker as W

BW = C.DATA / "basewalker"
CKPT = C.MODELS / "basewalker_decoder.pth"


def main():
    dev = "cuda"
    scene = W.load_scene(BW / "tiles", dev)
    dem = W.Dem(BW / "dem.npz", dev)
    lab = np.load(BW / "labels.npz")
    bases = lab["bases"]
    sd = np.load(BW / "seeds.npz")

    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    dec = W.WalkerDecoder().to(dev)
    dec.load_state_dict(ck["state_dict"])
    dec.eval()
    thresh = float(os.environ.get("BW_CONF_THRESH",
                                  str(ck.get("val", {}).get("thresh", 0.5))))
    print(f"[bw-infer] decoder from it {ck.get('it')} | conf thresh {thresh}")

    region = os.environ.get("BW_INFER_REGION", "val")  # val | all
    if region == "val":
        seeds = sd["seeds"][sd["is_val"]]
        gt = bases[lab["is_val"]]
    else:
        seeds = sd["seeds"]
        gt = bases[~lab["in_margin"]]
    print(f"[bw-infer] region={region}: {len(seeds):,} seeds | {len(gt)} GT bases")

    ends, scores = W.detect(model=W.load_encoder(), dec=dec, scene=scene,
                            dem=dem, seeds_np=seeds.astype(np.float32))
    keep = W.nms(ends, scores)
    det = keep[scores[keep] >= thresh]
    det_xyz, det_s = ends[det], scores[det]
    print(f"[bw-infer] {len(det)} detections after NMS+thresh")

    metrics = {}
    for r in (0.5, 1.0):
        metrics[f"@{r}m"] = W.match_metrics(det_xyz, det_s, gt, radius=r)
        m = metrics[f"@{r}m"]
        print(f"[bw-infer] @{r}m: P {m['precision']:.3f} R {m['recall']:.3f} "
              f"F1 {m['f1']:.3f} rmse {m['rmse']:.2f} m "
              f"(tp {m['tp']} fp {m['fp']} fn {m['fn']})")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    np.savetxt(C.RESULTS / "basewalker_detections.csv",
               np.column_stack([det_xyz, det_s]), delimiter=",",
               header="x,y,z,score", comments="")
    with open(C.RESULTS / "basewalker_metrics.json", "w") as f:
        json.dump(dict(region=region, thresh=thresh, n_det=int(len(det)),
                       n_gt=int(len(gt)), metrics=metrics), f, indent=2)

    # top-down figure: scene silhouette + GT + detections
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz = scene.xyz.cpu().numpy()
    x0, x1 = seeds[:, 0].min() - 5, seeds[:, 0].max() + 5
    y0, y1 = seeds[:, 1].min() - 5, seeds[:, 1].max() + 5
    m = (xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1)
    sub = xyz[m][np.random.permutation(int(m.sum()))[:400_000]]
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(sub[:, 0], sub[:, 1], s=0.05, c="0.75", marker=".")
    ax.scatter(gt[:, 0], gt[:, 1], s=42, marker="x", c="green", label="GT base")
    ax.scatter(det_xyz[:, 0], det_xyz[:, 1], s=26, marker="o",
               facecolors="none", edgecolors="red", label="detection")
    ax.set_aspect("equal")
    ax.legend()
    f1 = metrics["@0.5m"]["f1"]
    ax.set_title(f"BaseWalker detections ({region}) | F1@0.5m {f1:.3f} | "
                 f"{len(det)} det / {len(gt)} GT")
    fig.savefig(C.RESULTS / "10_basewalker_detections.png", dpi=130,
                bbox_inches="tight")
    print(f"[bw-infer] image -> {C.RESULTS / '10_basewalker_detections.png'}")


if __name__ == "__main__":
    main()
