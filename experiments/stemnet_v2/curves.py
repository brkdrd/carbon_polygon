#!/usr/bin/env python3
"""StemNet v2: a threshold-free score for tree detection.

ROC-AUC needs true negatives, correctly rejected non-trees, and detection has
no countable set of those. The detection equivalents use exactly
true positive = a detection matched to a survey base (one-to-one, within the
radius) and false positive = any other detection:

  precision-recall curve and AP (average precision, the area under it with
      precision made non-increasing, as COCO / VOC do): detections ranked by
      vote mass, every threshold at once. 1.0 = every tree found, with nothing
      false ranked above any true one.
  FROC: recall against false detections per hectare of scanned ground; read
      off at 1, 2, 5, 10, 20 and 50 FP / ha, and their mean (the CPM, as
      lung-nodule detection challenges report it).

Both at 0.5 m (the headline) and 0.3 m. Matching is greedy by vote mass, the
same as v2_validate's, so a prefix of the ranking is matched exactly as if
the lower-ranked detections did not exist (the test checks this against
re-matching at every threshold). An unmatched detection outside the
split's cells is ignored (the survey stops there).

The ranking is by vote mass only; the gate threshold changes which points
vote, so it is a setting, not a rank. Every gate threshold in V2_CURVE_P is
scored on val; the one with the best AP@0.5 is the one test is reported with
(nothing is chosen on test). Peak separation from detect_params.json (v2_validate),
else 0.3 m; minimum vote mass V2_CURVE_MIN_MASS (1), so the curve runs to
the lowest-ranked peak. The settings v2_infer uses are marked on the curves.

Reads the predictions v2_validate saved (data/stemnet_v2/predictions/<V2_CKPT
stem, head_tree_best>/val|test/); CPU, nothing is predicted.
Outputs:
  results/stemnet_v2_curves.txt
  results/57_stemnet_v2_curves.png     PR and FROC curves, val and test
  results/stemnet_v2_curves.csv        the test curves, one row per detection rank
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS  # noqa: E402
import vote as V  # noqa: E402

RUN = Path(os.environ.get("V2_CKPT", "head_tree_best.pth")).stem
P_GRID = [float(v) for v in os.environ.get("V2_CURVE_P", "0.3,0.4,0.5,0.6,0.7,0.8,0.9").split(",")]
MIN_MASS = float(os.environ.get("V2_CURVE_MIN_MASS", "1"))
FP_HA = (1, 2, 5, 10, 20, 50)
SPLITS = {1: "val", 2: "test"}
PARAMS = OUT / "models" / "detect_params.json"
LOG = RESULTS / "stemnet_v2_curves.txt"


def log(msg):
    line = f"[v2-curves] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def split_at(split, xy):
    i, j = np.floor(xy[:, 0]).astype(np.int64), np.floor(xy[:, 1]).astype(np.int64)
    ok = (i >= 0) & (j >= 0) & (i < split.shape[1]) & (j < split.shape[0])
    out = np.full(len(xy), 255, np.uint8)
    out[ok] = split[j[ok], i[ok]]
    return out


def curve(det, score, gt, radius, ignore=None):
    """Ranked by score: -> dict score, tp, fp (cumulative counts per rank, ignored
    detections dropped), n_gt."""
    m = V.match(det, score, gt, radius, ignore=ignore)
    o = np.argsort(-score, kind="stable")
    pair = m["pair"][o]
    keep = pair != -2
    hit = (pair[keep] >= 0).astype(np.int64)
    return dict(score=score[o][keep], tp=np.cumsum(hit), fp=np.cumsum(1 - hit), n_gt=len(gt))


def ap(c):
    """Area under the precision-recall curve, precision made non-increasing in recall."""
    if not len(c["tp"]) or not c["n_gt"]:
        return 0.0
    rec = c["tp"] / c["n_gt"]
    prec = c["tp"] / (c["tp"] + c["fp"])
    env = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * env))


def froc(c, ha):
    """Recall at each FP / ha in FP_HA (the best recall with at most that many false per ha)."""
    out = []
    for f in FP_HA:
        k = np.flatnonzero(c["fp"] <= f * ha)
        out.append(float(c["tp"][k[-1]] / c["n_gt"]) if len(k) and c["n_gt"] else 0.0)
    return out


def at_mass(c, mass):
    """Precision / recall of the detections with score >= mass."""
    k = np.flatnonzero(c["score"] >= mass)
    if not len(k):
        return 0.0, 0.0
    tp, fp = c["tp"][k[-1]], c["fp"][k[-1]]
    return float(tp / max(tp + fp, 1)), float(tp / max(c["n_gt"], 1))


def load(sid):
    d = OUT / "predictions" / RUN / SPLITS[sid]
    if not (d / "stamp.json").exists():
        raise SystemExit(f"[v2-curves] no saved predictions in {d} — run v2_validate (it saves them)")
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("xy", "p", "off")}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    regions = np.load(OUT / "regions.npz")
    split = regions["split"]
    occ = regions["occupancy"]
    rows = list(csv.DictReader(open(OUT / "label_bases.csv")))
    gxy = np.array([[float(r["x"]), float(r["y"])] for r in rows]).reshape(-1, 2)
    gs = split_at(split, gxy)
    prm = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
    if prm and prm.get("checkpoint", "").replace(".pth", "") != RUN:
        prm = {}
    sep = float(prm.get("peak_sep", V.PEAK_SEP))
    op = (float(prm["p_thresh"]), float(prm["peak_min"])) if prm else None
    log(f"predictions of {RUN} | ranking by vote mass, peaks {sep:g} m apart, mass >= {MIN_MASS:g} | "
        + (f"v2_infer's settings: gate >= {op[0]:g}, mass >= {op[1]:g}" if op else "no detect_params.json for it"))
    res = {}
    for sid in (1, 2):
        R = load(sid)
        xy, p, off = np.asarray(R["xy"], np.float64), np.asarray(R["p"]), np.asarray(R["off"], np.float64)
        gt = gxy[gs == sid]
        ha = float(((split == sid) & (occ > 0)).sum()) / 1e4
        log(f"{SPLITS[sid]}: {len(gt)} survey bases, {ha:.2f} ha scanned, {len(xy):,} points")
        for pt in P_GRID:
            det, sc, _ = V.detect(xy, p, off, p_thresh=pt, peak_sep=sep, peak_min=MIN_MASS)
            ign = split_at(split, det) != sid if len(det) else None
            c5, c3 = curve(det, sc, gt, 0.5, ign), curve(det, sc, gt, 0.3, ign)
            fr = froc(c5, ha)
            res[(sid, pt)] = dict(c5=c5, c3=c3, ap5=ap(c5), ap3=ap(c3), froc=fr, ha=ha)
            log(f"  gate >= {pt:.1f}: AP@0.5 m {ap(c5):.3f} | AP@0.3 m {ap(c3):.3f} | FROC recall at "
                + " ".join(f"{f}:{r:.2f}" for f, r in zip(FP_HA, fr)) + f" FP/ha, CPM {np.mean(fr):.3f} | "
                f"max recall {c5['tp'][-1] / max(len(gt), 1) if len(c5['tp']) else 0:.3f} from {len(c5['tp'])} detections")
        del R, xy, p, off
    best = max(P_GRID, key=lambda pt: res[(1, pt)]["ap5"])
    v, t = res[(1, best)], res[(2, best)]
    log(f"gate threshold chosen on val by AP@0.5: {best:g} (val AP {v['ap5']:.3f})")
    log(f"TEST (held out): AP@0.5 m {t['ap5']:.3f} | AP@0.3 m {t['ap3']:.3f} | CPM {np.mean(t['froc']):.3f} "
        f"(recall at 1/2/5/10/20/50 FP per ha: " + " / ".join(f"{r:.2f}" for r in t["froc"]) + ")")
    if op and (2, op[0]) in res:
        c = res[(2, op[0])]["c5"]
        pr, rc = at_mass(c, op[1])
        log(f"  v2_infer's settings on test, from the same curve: precision {pr:.3f} recall {rc:.3f}")
    with open(RESULTS / "stemnet_v2_curves.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "vote_mass", "tp", "fp", "precision", "recall", "fp_per_ha"])
        c = t["c5"]
        for k in range(len(c["tp"])):
            w.writerow([k + 1, f"{c['score'][k]:.2f}", int(c["tp"][k]), int(c["fp"][k]),
                        f"{c['tp'][k] / (k + 1):.4f}", f"{c['tp'][k] / max(c['n_gt'], 1):.4f}", f"{c['fp'][k] / t['ha']:.3f}"])
    figure(res, best, op)
    log("-> results/stemnet_v2_curves.txt, 57_stemnet_v2_curves.png, stemnet_v2_curves.csv (test, 0.5 m)")


def figure(res, best, op):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    for sid, ls in ((1, "--"), (2, "-")):
        r = res[(sid, best)]
        for key, col, rad in (("c5", "tab:blue", 0.5), ("c3", "tab:orange", 0.3)):
            c = r[key]
            if not len(c["tp"]):
                continue
            rec, prec = c["tp"] / max(c["n_gt"], 1), c["tp"] / (c["tp"] + c["fp"])
            axs[0].plot(rec, prec, ls, c=col, label=f"{SPLITS[sid]} @{rad} m: AP {r['ap5' if rad == 0.5 else 'ap3']:.3f}")
        c = r["c5"]
        if len(c["tp"]):
            axs[1].plot(c["fp"] / r["ha"], c["tp"] / max(c["n_gt"], 1), ls, c="tab:blue",
                        label=f"{SPLITS[sid]} @0.5 m: CPM {np.mean(r['froc']):.3f}")
        if op and (sid, op[0]) in res:
            c = res[(sid, op[0])]["c5"]
            k = np.flatnonzero(c["score"] >= op[1])
            if len(k):
                k = k[-1]
                axs[0].plot(c["tp"][k] / max(c["n_gt"], 1), c["tp"][k] / (k + 1), "k*", ms=12)
                axs[1].plot(c["fp"][k] / res[(sid, op[0])]["ha"], c["tp"][k] / max(c["n_gt"], 1), "k*", ms=12)
    axs[0].set_xlabel("recall (share of survey bases found)"); axs[0].set_ylabel("precision (share of detections that are a base)")
    axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1.02); axs[0].legend(loc="lower left")
    axs[0].set_title(f"precision-recall, ranked by vote mass, gate >= {best:g} (chosen on val)")
    axs[1].set_xscale("log"); axs[1].set_xlabel("false detections per hectare"); axs[1].set_ylabel("recall @0.5 m")
    axs[1].set_ylim(0, 1.02); axs[1].legend(loc="upper left")
    for f in FP_HA:
        axs[1].axvline(f, c="grey", lw=0.5, ls=":")
    axs[1].set_title("FROC (* = v2_infer's settings; dashed = val, solid = test)")
    fig.tight_layout(); fig.savefig(RESULTS / "57_stemnet_v2_curves.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
