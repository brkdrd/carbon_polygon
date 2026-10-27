#!/usr/bin/env python3
"""StemNet v2 validation: the trained head on the two held-out blocks.

1. val: the block training evaluated every 5,000 steps and picked the
   checkpoint with, so not independent. Predicted once (blocks, as v2_infer
   does), then the detection settings are swept on it:
     gate threshold      V2_VAL_P     (0.3 ... 0.9)
     peak separation     V2_VAL_SEP   (0.3, 0.5, 0.8 m)
     minimum vote mass   V2_VAL_MASS  (5 ... 320)
   The best F1 @0.5 m -> models/detect_params.json, which v2_infer reads.
2. test: the girth-attributed patch, never trained on and never used to
   choose anything. Predicted once, scored with the default settings and with
   the val-tuned ones. An unmatched detection outside the test cells is
   ignored, not counted false: the survey stops there.
   Reported: F1 / precision / recall at 0.5 m and 0.3 m, placement RMSE,
   recall by tier, by girth (DBH = 2 x the measured radius) and for bases with
   scan points under them; the tree head vs the campus_pred_B24 labels on the
   test cells (all labelled, and matched only = without the removed ground).

Checkpoint: as v2_infer (V2_CKPT, default head_tree_best.pth).

Predictions are saved (data/stemnet_v2/predictions/<checkpoint>/val|test/,
~1.5 GB + ~3.5 GB) and reused by a rerun with the same checkpoint and by
v2_curves.

Outputs:
  results/stemnet_v2_validate.txt
  results/stemnet_v2_test_detections.csv   val-tuned settings: x, y, utm, score,
                                           votes, matched base (-1 false, -2 ignored)
  results/53_stemnet_v2_test.png           test: bases found / missed, detections
  results/54_stemnet_v2_test_tree.png      test: tree head right / wrong vs B24
  data/stemnet_v2/models/detect_params.json
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
from core import OUT, RESULTS  # noqa: E402
import data as D  # noqa: E402
import infer as I  # noqa: E402
import train as T  # noqa: E402
import vote as V  # noqa: E402


def floats(name, default):
    return [float(v) for v in os.environ.get(name, default).split(",")]


P_GRID = floats("V2_VAL_P", "0.3,0.4,0.5,0.6,0.7,0.8,0.9")
SEP_GRID = floats("V2_VAL_SEP", "0.3,0.5,0.8")
MASS_GRID = floats("V2_VAL_MASS", "5,10,20,40,80,160,320")
DBH_BINS = (0.0, 0.2, 0.4, 0.6, np.inf)                 # m
DEFAULT = dict(p_thresh=V.P_THRESH, peak_sep=V.PEAK_SEP, peak_min=V.PEAK_MIN)
LOG = RESULTS / "stemnet_v2_validate.txt"


def log(msg):
    line = f"[v2-validate] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def bases_in(regions, sid):
    rows = list(csv.DictReader(open(OUT / "label_bases.csv")))
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows]).reshape(-1, 2)
    m = I.split_at(regions, xy) == sid
    col = lambda k, f=str: np.array([f(r[k]) for r in rows])[m]  # noqa: E731
    return dict(xy=xy[m], id=col("base", int), tier=col("tier"), r=col("radius", float),
                measured=col("radius_measured") == "1", has_points=col("has_points") == "1")


def bbox_of(regions, sid):
    jj, ii = np.nonzero(regions["split"] == sid)
    return float(ii.min()), float(ii.max() + 1), float(jj.min()), float(jj.max() + 1)


PRED_KEYS = dict(xy=np.float32, z=np.float32, p=np.float32, off=np.float32, tree=np.float16, sem=np.int8)


def pred_stamp(cpath, ck, bbox):
    st = cpath.stat()
    return dict(checkpoint=cpath.name, step=int(ck["step"]), size=st.st_size, mtime=int(st.st_mtime), cap=T.CAP,
                block=I.BLOCK, margin=I.MARGIN, bbox=list(bbox), seed=T.SEED)


def predict_split(runner, store, regions, sid, rng, cpath=None, ck=None):
    """Every point of the split's bounding box, predicted block by block. -> (R, bbox).
    Saved to data/stemnet_v2/predictions/<checkpoint>/<split>/ and reused while
    the checkpoint and settings are the same (v2_curves reads them too)."""
    t0 = time.time()
    bbox = bbox_of(regions, sid)
    d = OUT / "predictions" / cpath.stem / I.SPLITS[sid] if cpath is not None else None
    stamp = pred_stamp(cpath, ck, bbox) if d is not None else None
    if d is not None and (d / "stamp.json").exists() and json.loads((d / "stamp.json").read_text()) == stamp:
        R = {k: np.load(d / f"{k}.npy") for k in PRED_KEYS}
        log(f"{I.SPLITS[sid]}: {len(R['xy']):,} saved predictions reused ({d})")
        return R, bbox
    cores = I.blocks(bbox, regions["occupancy"])
    parts = [{k: v[inside] for k, v in R.items()}
             for _, R, inside, *_ in I.iter_blocks(runner.predict, store, cores, DEFAULT, rng, detect=False)]
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    log(f"{I.SPLITS[sid]}: {len(R['xy']):,} points in {bbox} predicted in {len(cores)} blocks, "
        f"{(time.time() - t0) / 60:.1f} min")
    R = {k: R[k].astype(t) for k, t in PRED_KEYS.items()}
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "stamp.json").unlink(missing_ok=True)                  # written last: a half-saved set is never reused
        for k, v in R.items():
            np.save(d / f"{k}.npy", v)
        (d / "stamp.json").write_text(json.dumps(stamp))
        log(f"  saved -> {d} ({sum(v.nbytes for v in R.values()) / 2**30:.1f} GB)")
    return R, bbox


def score(det, sc, G, regions, sid, radius):
    ignore = I.split_at(regions, det) != sid if len(det) else None
    mm = V.match(det, sc, G["xy"], radius, ignore=ignore)
    mm["found"] = np.zeros(len(G["xy"]), bool)
    mm["found"][mm["pair"][mm["pair"] >= 0]] = True
    return mm


def det_line(name, a, b):
    return (f"{name}: @0.5 m F1 {a['f1']:.3f} (P {a['precision']:.3f} R {a['recall']:.3f}, {a['tp']} TP / "
            f"{a['fp']} FP / {a['fn']} FN, RMSE {a['rmse'] * 100:.1f} cm) | @0.3 m F1 {b['f1']:.3f} "
            f"(P {b['precision']:.3f} R {b['recall']:.3f})")


def breakdown(G, found):
    out = ["recall@0.5 by tier: " + " | ".join(f"{t} {found[G['tier'] == t].mean():.2f} of {(G['tier'] == t).sum()}"
                                              for t in ("A", "B", "C") if (G["tier"] == t).any())]
    dbh = 2 * G["r"]
    parts = []
    for lo, hi in zip(DBH_BINS[:-1], DBH_BINS[1:]):
        m = G["measured"] & (dbh >= lo) & (dbh < hi)
        if m.any():
            parts.append(f"{lo * 100:.0f}-{hi * 100:.0f} cm {found[m].mean():.2f} of {m.sum()}"
                         if np.isfinite(hi) else f">= {lo * 100:.0f} cm {found[m].mean():.2f} of {m.sum()}")
    if parts:
        out.append("recall@0.5 by DBH (measured girths): " + " | ".join(parts))
    hp = G["has_points"]
    out.append(f"recall@0.5, bases with scan points under them: {found[hp].mean() if hp.any() else float('nan'):.2f} "
               f"of {hp.sum()} | without: {found[~hp].mean() if (~hp).any() else float('nan'):.2f} of {(~hp).sum()}")
    return out


def tree_report(name, R, regions, sid):
    m = I.split_at(regions, R["xy"]) == sid
    t = I.from_counts(I.counts(R["tree"][m], R["sem"][m]))
    tm = I.from_counts(I.counts(R["tree"][m], np.where(R["sem"][m] == 2, -1, R["sem"][m])))
    log(f"  TREE {name} all labelled: {I.tree_line(t)}")
    log(f"  TREE {name} matched only (no removed ground): {I.tree_line(tm)}")
    return t, m


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    frame = json.loads((OUT / "frame.json").read_text())
    origin = np.asarray(frame["origin_m"], np.float64)
    regions = np.load(OUT / "regions.npz")
    runner, ck, cpath = I.load_runner(log)
    has_tree = runner.head.has_tree
    store, rng = D.TileStore(keep=I.KEEP), np.random.default_rng(T.SEED)

    # 1. val: sweep the detection settings
    Gv = bases_in(regions, 1)
    Rv, _ = predict_split(runner, store, regions, 1, rng, cpath, ck)
    xy, p, off = Rv["xy"], Rv["p"], Rv["off"]
    d0, s0, _ = V.detect(xy, p, off, **DEFAULT)
    base_v = score(d0, s0, Gv, regions, 1, 0.5)
    log(det_line(f"VAL default settings ({len(Gv['xy'])} bases)", base_v, score(d0, s0, Gv, regions, 1, 0.3))
        + f" | training's own eval of this checkpoint: F1 {ck['best']:.3f}")
    if has_tree:
        tree_report("val", Rv, regions, 1)
    rows, t0 = [], time.time()
    for pt in P_GRID:
        for sep in SEP_GRID:
            det, sc, _ = V.detect(xy, p, off, p_thresh=pt, peak_sep=sep, peak_min=min(MASS_GRID))
            for mass in MASS_GRID:
                k = sc >= mass
                mm = score(det[k], sc[k], Gv, regions, 1, 0.5)
                rows.append(dict(p_thresh=pt, peak_sep=sep, peak_min=mass, f1=mm["f1"], precision=mm["precision"],
                                 recall=mm["recall"], det=int(k.sum())))
    rows.sort(key=lambda r: -r["f1"])
    best = rows[0]
    log(f"sweep on val: {len(rows)} settings in {(time.time() - t0) / 60:.1f} min; top 10 by F1 @0.5 m:")
    for r in rows[:10]:
        log(f"  gate >= {r['p_thresh']:.1f} | peaks {r['peak_sep']:.1f} m apart | mass >= {r['peak_min']:>5g} -> "
            f"F1 {r['f1']:.3f} (P {r['precision']:.3f} R {r['recall']:.3f}, {r['det']} det)")
    tuned = {k: best[k] for k in DEFAULT}
    T.MODELS.mkdir(parents=True, exist_ok=True)
    I.PARAMS.write_text(json.dumps(dict(checkpoint=cpath.name, step=int(ck["step"]), **tuned, val_f1=best["f1"],
                                        val_f1_default=base_v["f1"], swept=dict(p=P_GRID, sep=SEP_GRID, mass=MASS_GRID)),
                                   indent=1))
    log(f"-> {I.PARAMS} (v2_infer uses these)")
    del Rv, xy, p, off

    # 2. test: scored once, with the defaults and with the val-tuned settings
    Gt = bases_in(regions, 2)
    Rt, bbox = predict_split(runner, store, regions, 2, rng, cpath, ck)
    log(f"TEST: the girth-attributed patch, {len(Gt['xy'])} survey bases; nothing below was chosen on it")
    res = {}
    for name, prm in (("default", DEFAULT), ("val-tuned", tuned)):
        det, sc, nv = V.detect(Rt["xy"], Rt["p"], Rt["off"], **prm)
        a, b = score(det, sc, Gt, regions, 2, 0.5), score(det, sc, Gt, regions, 2, 0.3)
        log(det_line(f"TEST {name} settings", a, b) + f" | {len(det)} detections, "
            f"{int((a['pair'] == -2).sum())} outside the test cells ignored")
        res[name] = (det, sc, nv, a)
    det, sc, nv, a = res["val-tuned"]
    for line in breakdown(Gt, a["found"]):
        log(f"  {line} (val-tuned)")
    if has_tree:
        t, m = tree_report("test", Rt, regions, 2)
        T.tree_figure({k: v[m] for k, v in Rt.items()}, bbox, ck["step"], t, RESULTS / "54_stemnet_v2_test_tree.png",
                      name="Test patch")
    with open(RESULTS / "stemnet_v2_test_detections.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x", "y", "utm_x", "utm_y", "score", "votes", "matched_base"])
        for (x, y), s_, n_, pr in zip(det, sc, nv, a["pair"]):
            w.writerow([f"{x:.3f}", f"{y:.3f}", f"{x + origin[0]:.3f}", f"{y + origin[1]:.3f}", f"{s_:.2f}", int(n_),
                        int(Gt["id"][pr]) if pr >= 0 else int(pr)])
    T.figure(Gt["xy"], det, a["pair"], bbox, RESULTS / "53_stemnet_v2_test.png",
             title=f"StemNet v2 on the test patch (val-tuned settings), match 0.5 m: F1 {a['f1']:.3f}")
    log("-> results/stemnet_v2_validate.txt, stemnet_v2_test_detections.csv, 53_stemnet_v2_test.png"
        + (", 54_stemnet_v2_test_tree.png" if has_tree else ""))


if __name__ == "__main__":
    main()
