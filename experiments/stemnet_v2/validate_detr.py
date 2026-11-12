#!/usr/bin/env python3
"""StemNet v2: validate ONE existing DETR checkpoint, on measured DBH only.

The stage is standalone: it loads a checkpoint written by v2_train_detr, reads
only the checkpoint's own architecture, feature statistics, encoder flag and
prediction settings (voxel, cap, token limit, crop, NMS radius, minimum
probability, feature grid, seed), predicts the selected split with
detr.predict_region, and scores it. A checkpoint from before those settings
were saved gets a documented module-default fallback and a warning in the
metrics. Nothing is trained, no threshold is chosen here, and the training
artifacts are never touched. A requested threshold below the checkpoint's
min_p lowers min_p for the run (logged and recorded) rather than silently
dropping every kept detection.

Eligibility: survey bases whose measured diameter is >= 10 cm. Diameters come
from the attributed CSV (V2_DETR_ATTRS) joined to label_bases.csv members by
detr_truth.load_truth, which also returns every base of the split with the
reason it was kept or excluded. Height filtering is PENDING by explicit user
decision (the attributed CSV has no tree heights); the audit therefore records
height_filter = "pending_not_applied" and this script applies none. Only
TARGETS are filtered: the point cloud is never masked with truth.

Scoring. Known but sub-threshold stems stay negatives: an unmatched detection
on one is a false positive. An unknown-diameter base cannot label a detection
false, so after eligible matching (greedy, one-to-one, highest score first) an
unmatched detection within the matching radius of an unknown base is ignored
(-2). A duplicate next to an already-matched eligible base stays false even
when an unknown base is also nearby. An unmatched detection outside the
split's cells is ignored and counted apart (the survey stops there).

Reported at a FIXED threshold, V2_DETR_THRESHOLD when set, else the
checkpoint's own eval cut (chosen on val during training): TP / FP / FN,
precision / recall / F1 and placement RMSE (null when nothing matched) at
0.5 m and 0.3 m; AP@0.5 m and AP@0.3 m use the full ranking. Ignored and
out-of-split detections are counted explicitly, never as false. Outside
predictions are removed before matching, so an outside high score cannot
consume an eligible base inside the split.

Splits: V2_DETR_SPLIT=test (default) or val. The attributed CSV covers the
test patch (sid 2); val (sid 1) has no attributes, so val requires an
explicit V2_DETR_ATTRS that does cover it.

Encoder: the checkpoint says which one trained it. cfg.encoder="fake" is a
test-only synthetic encoder; the run says so in every log line, in the metric
notes and in the figure title. cfg.encoder="sonata" loads the real frozen
Sonata.

Outputs (RESULTS_DIR):
  stemnet_v2_<split>_detr_metrics.json     finite JSON, AP / fixed-threshold
                                           metrics, target audit, accounting
  stemnet_v2_<split>_detr_detections.csv   x, y, UTM x, UTM y (origin added in
                                           float64), score, above_threshold,
                                           status, matched_base, distance_m
  stemnet_v2_<split>_detr_bases.csv        every base of the split, including
                                           the excluded unknown ones;
                                           detected = found at the fixed
                                           threshold, detected_any_score =
                                           ranked match below it
  stemnet_v2_<split>_detr.png              found / missed bases and detections
                                           (V2_DETR_FIGURE=0 skips it)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ATTRS as DEFAULT_ATTRS, OUT, RESULTS  # noqa: E402
import curves as C  # noqa: E402
import data as D  # noqa: E402
import detr as R  # noqa: E402
import model as M  # noqa: E402
import train as T  # noqa: E402
import train_detr as TR  # noqa: E402
import vote as V  # noqa: E402
from detr_truth import load_truth  # noqa: E402

SPLITS = {1: "val", 2: "test"}
SPLIT_NAME = os.environ.get("V2_DETR_SPLIT", "test").strip().lower()
CKPT = os.environ.get("V2_DETR_CKPT", "detr_best.pth").strip()
ATTRS = os.environ.get("V2_DETR_ATTRS", "").strip()
THRESHOLD_ENV = os.environ.get("V2_DETR_THRESHOLD", "").strip()
FIGURE = os.environ.get("V2_DETR_FIGURE", "1").strip() != "0"
RADII = (0.5, 0.3)
SONATA_DIM = 1232          # name in - the dense read-out of model.dense_features
OPTIONAL = ("max_tokens", "crop", "own", "nms_r", "min_p", "grid", "seed")


def log(msg):
    print(f"[v2-detr-val] {msg}", flush=True)


def positive_int(value, name):
    """A real positive int; never truncate a float or accept a bool."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"checkpoint cfg.{name} {value!r} is not an int")
    v = int(value)
    if v <= 0:
        raise ValueError(f"checkpoint cfg.{name} {v!r} is not positive")
    return v


def non_negative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"checkpoint cfg.{name} {value!r} is not an int")
    v = int(value)
    if v < 0:
        raise ValueError(f"checkpoint cfg.{name} {v!r} is negative")
    return v


def positive_float(value, name):
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint cfg.{name} {value!r} is not a number") from exc
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"checkpoint cfg.{name} {v!r} is not a positive finite size")
    return v


def split_id(name):
    for k, v in SPLITS.items():
        if v == name:
            return k
    raise SystemExit(f"[v2-detr-val] V2_DETR_SPLIT={name!r}: use val or test")


def checkpoint_path(name):
    """V2_DETR_CKPT is models/<name> unless it is an absolute path."""
    p = Path(name)
    return p if p.is_absolute() else T.MODELS / p


# --------------------------------------------------------------------------
# checkpoint policy
# --------------------------------------------------------------------------
def checkpoint_plan(ck):
    """Validate a saved DETR checkpoint -> the settings prediction must use.

    Everything comes from the checkpoint, never from the current training
    configuration: architecture (hcfg), input width (d_in), standardisation
    (mu / sd), encoder, voxel size, cap and token limit. Raises ValueError
    when the checkpoint cannot support a faithful validation.
    """
    if not isinstance(ck, dict):
        raise ValueError("checkpoint is not a dict")
    if ck.get("task") != "detr":
        raise ValueError(f"checkpoint task is {ck.get('task')!r}, not 'detr'; run v2_train_detr first")
    for key in ("head", "d_in", "hcfg", "mu", "sd", "cfg"):
        if key not in ck:
            raise ValueError(f"checkpoint has no {key!r}")
    d_in = ck["d_in"]
    if not isinstance(d_in, int) or isinstance(d_in, bool) or d_in <= 0:
        raise ValueError(f"checkpoint d_in {d_in!r} is not a positive int")
    hcfg = ck["hcfg"]
    if not isinstance(hcfg, dict):
        raise ValueError("checkpoint hcfg is not a dict")
    for key in ("d", "n_query", "layers", "heads", "ff"):
        v = hcfg.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"checkpoint hcfg[{key!r}] = {v!r}: need a positive int")
    mu = np.asarray(ck["mu"], np.float64)
    sd = np.asarray(ck["sd"], np.float64)
    if mu.ndim != 1 or mu.shape != sd.shape or mu.size != 12:
        raise ValueError(f"checkpoint mu / sd shapes {mu.shape} / {sd.shape}: data.py always carries "
                         f"exactly 12 covariance features")
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("checkpoint mu / sd are not finite with a positive sd")
    if d_in != mu.size + SONATA_DIM:
        raise ValueError(f"checkpoint d_in {d_in} != {mu.size} covariance features + {SONATA_DIM} Sonata values")
    head = ck["head"]
    if not isinstance(head, dict) or not head:
        raise ValueError("checkpoint head is empty")
    for name, value in head.items():
        if not hasattr(value, "isfinite") or not bool(value.isfinite().all()):
            raise ValueError(f"checkpoint head[{name!r}] is missing or not finite")
    cfg = ck["cfg"]
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint cfg is not a dict")
    encoder = cfg.get("encoder")
    if encoder not in ("sonata", "fake"):
        raise ValueError(f"checkpoint cfg.encoder {encoder!r}: expected sonata or fake")
    voxel = positive_float(cfg.get("voxel"), "voxel")
    cap = positive_int(cfg.get("cap"), "cap")

    plan: dict = dict(d_in=d_in, hcfg=dict(hcfg), mu=mu, sd=sd, encoder=encoder, voxel=voxel, cap=cap,
                      sources=dict(voxel="checkpoint", cap="checkpoint"), warnings=[])
    plan["sources"].update({k: None for k in OPTIONAL})
    defaults = dict(max_tokens=R.MAX_TOKENS, crop=R.CROP, own=R.OWN, nms_r=R.NMS_R,
                    min_p=R.MIN_P, grid=M.GRID, seed=T.SEED)
    for key in OPTIONAL:
        value = cfg.get(key)
        if value is None:
            plan[key] = defaults[key]
            plan["sources"][key] = "module_default"
            plan["warnings"].append(
                f"{key} is not recorded in the checkpoint cfg; using the current module default "
                f"{defaults[key]!r} (v2_train_detr now saves it)")
            continue
        plan["sources"][key] = "checkpoint"
        if key == "max_tokens":
            plan[key] = positive_int(value, key)
        elif key == "seed":
            plan[key] = non_negative_int(value, key)
        elif key in ("crop", "own", "grid"):
            plan[key] = positive_float(value, key)
        elif key == "nms_r":
            plan[key] = float(value)
            if not math.isfinite(plan[key]) or plan[key] < 0:
                raise ValueError(f"checkpoint cfg.nms_r {value!r}: need a non-negative finite radius")
        else:                                        # min_p
            plan[key] = float(value)
            if not math.isfinite(plan[key]) or not 0.0 <= plan[key] < 1.0:
                raise ValueError(f"checkpoint cfg.min_p {value!r}: need a probability in [0, 1)")
    if plan["crop"] <= plan["own"]:
        raise ValueError(f"checkpoint geometry crop {plan['crop']} <= own {plan['own']}")
    return plan


def apply_geometry(plan):
    """Put the checkpoint's prediction settings into detr.py / model.py."""
    R.VOXEL = float(plan["voxel"])
    R.MAX_TOKENS = int(plan["max_tokens"])
    R.CROP = float(plan["crop"])
    R.HALF = R.CROP / 2.0
    R.OWN = float(plan["own"])
    R.NMS_R = float(plan["nms_r"])
    R.MIN_P = float(plan["min_p"])
    M.GRID = float(plan["grid"])


def resolve_threshold(env_value, ck):
    """Fixed detection threshold: V2_DETR_THRESHOLD, else the checkpoint's eval cut."""
    if env_value != "":
        try:
            value = float(env_value)
        except ValueError as exc:
            raise ValueError(f"V2_DETR_THRESHOLD={env_value!r} is not a number") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"V2_DETR_THRESHOLD={value!r}: need a probability in [0, 1]")
        return value, "environment"
    ev = ck.get("eval") or {}
    cut = ev.get("cut")
    if cut is None:
        raise ValueError("checkpoint carries no eval.cut; set V2_DETR_THRESHOLD (no threshold is tuned here)")
    value = float(cut)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"checkpoint eval.cut {cut!r}: need a probability in [0, 1]")
    return value, "checkpoint_eval_cut"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def match_split(det, score, truth, split, sid, radius):
    """Greedy one-to-one matching with the measured eligibility rules.

    -> pair (>= 0 matched truth index, -1 false, -2 ignored), dist (m, NaN
    unmatched), outside / unknown / duplicates masks, n_gt. A duplicate near
    an eligible base stays -1 even if an unknown base is also nearby.
    """
    from scipy.spatial import cKDTree
    det = np.asarray(det, np.float64).reshape(-1, 2)
    score = np.asarray(score, np.float64).reshape(-1)
    gt = np.asarray(truth["xy"], np.float64).reshape(-1, 2)
    unknown = np.array([[r["x"], r["y"]] for r in truth["rows"] if r.get("reason") == "unknown"],
                       np.float64).reshape(-1, 2)
    empty = dict(pair=np.zeros(0, np.int64), dist=np.zeros(0), outside=np.zeros(0, bool),
                 unknown=np.zeros(0, bool), duplicates=np.zeros(0, bool), n_gt=len(gt))
    if not len(det):
        return empty
    outside = C.split_at(split, det) != sid
    inside = ~outside
    # Predictions outside the split's cells are ignored BEFORE matching, so an
    # outside high score cannot consume an eligible inside base.
    pair = np.full(len(det), -2, np.int64)
    if inside.any():
        pair[inside] = (np.asarray(V.match(det[inside], score[inside], gt, radius)["pair"], np.int64)
                        if len(gt) else -1)
    near_t = np.zeros(len(det), bool)
    near_u = np.zeros(len(det), bool)
    if len(gt) and inside.any():
        kt = cKDTree(gt)
        near_t = np.zeros(len(det), bool)
        near_t[inside] = np.array([len(v) > 0 for v in kt.query_ball_point(det[inside], radius)], bool)
    if len(unknown) and inside.any():
        ku = cKDTree(unknown)
        near_u = np.zeros(len(det), bool)
        near_u[inside] = np.array([len(v) > 0 for v in ku.query_ball_point(det[inside], radius)], bool)
    unmatched = pair == -1
    duplicates = unmatched & inside & near_t                    # stays a false positive
    unknown_ign = unmatched & inside & ~near_t & near_u        # cannot be labelled false
    pair[unknown_ign] = -2
    dist = np.full(len(det), np.nan)
    hit = pair >= 0
    if hit.any():
        dist[hit] = np.hypot(*(det[hit] - gt[pair[hit]]).T)
    return dict(pair=pair, dist=dist, outside=outside, unknown=unknown_ign,
                duplicates=duplicates, n_gt=len(gt))


def pr_curve(score, pair):
    """Ranked by score, ignored detections dropped: cumulative TP / FP counts."""
    o = np.argsort(-np.asarray(score, np.float64), kind="stable")
    pair = np.asarray(pair)[o]
    keep = pair != -2
    hit = pair[keep] >= 0
    return dict(score=np.asarray(score, np.float64)[o][keep],
                tp=np.cumsum(hit).astype(np.int64), fp=np.cumsum(~hit).astype(np.int64))


def average_precision(score, pair, n_gt):
    c = pr_curve(score, pair)
    return float(C.ap(dict(tp=c["tp"], fp=c["fp"], n_gt=int(n_gt))))


def fixed_metrics(det, score, match, cut):
    """P / R / F1 / TP / FP / FN / RMSE at one fixed probability threshold."""
    score = np.asarray(score, np.float64)
    kept = score >= cut
    pair = np.asarray(match["pair"])[kept]
    dist = np.asarray(match["dist"])[kept]
    outside = np.asarray(match["outside"], bool)[kept]
    tp = int((pair >= 0).sum())
    fp = int((pair == -1).sum())
    ignored = int(((pair == -2) & ~outside).sum())
    outside_n = int(((pair == -2) & outside).sum())
    duplicates = int(np.asarray(match["duplicates"], bool)[kept].sum())
    n_gt = int(match["n_gt"])
    fn = n_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    hits = dist[pair >= 0]
    rmse = float(np.sqrt(np.mean(hits ** 2))) if len(hits) else None
    return dict(threshold=float(cut), kept=int(kept.sum()), tp=tp, fp=fp, fn=fn,
                ignored=ignored, outside=outside_n, duplicates=duplicates,
                precision=float(precision), recall=float(recall), f1=float(f1), rmse=rmse)


def fixed_payload(metric, radius):
    """fixed_metrics -> the JSON block, with the radius and RMSE in metres."""
    return dict(radius_m=float(radius), threshold=metric["threshold"], kept=metric["kept"],
                tp=metric["tp"], fp=metric["fp"], fn=metric["fn"], ignored=metric["ignored"],
                outside=metric["outside"], duplicates=metric["duplicates"],
                precision=metric["precision"], recall=metric["recall"], f1=metric["f1"],
                rmse_m=metric["rmse"])


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def finite_json(value):
    """numpy -> JSON, non-finite floats -> null, and no NaN/Infinity tokens."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def write_json(path, payload):
    path.write_text(json.dumps(finite_json(payload), indent=1, allow_nan=False) + "\n", encoding="utf-8")


def write_detections(path, det, score, match, cut, ids, origin):
    pair = match["pair"]
    outside = np.asarray(match["outside"], bool)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "utm_x", "utm_y", "score", "above_threshold", "status",
                    "matched_base", "distance_m"])
        for i, ((x, y), s, p, d) in enumerate(zip(det, score, pair, match["dist"])):
            if p >= 0:
                status, base = "tp", int(ids[p])
            elif p == -1:
                status, base = "false", -1
            elif outside[i]:
                status, base = "ignored_outside", -2
            else:
                status, base = "ignored_unknown", -2
            utm_x = float(x) + float(origin[0])
            utm_y = float(y) + float(origin[1])
            w.writerow([float(x), float(y), utm_x, utm_y, float(s), int(s >= cut), status, base,
                        float(d) if np.isfinite(d) else ""])


def write_bases(path, truth, match, score, cut):
    """Per-base audit: every base of the split, eligible or excluded.

    `detected` means found at the FIXED threshold used for the metrics;
    `detected_any_score` records a ranked match below that threshold without
    calling it a detection.
    """
    ids = np.asarray(truth["id"], np.int64)
    pair = match["pair"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["base", "members", "x", "y", "tier", "eligible", "reason",
                    "detected", "detected_any_score", "score", "distance_m"])
        for row in truth["rows"]:
            detected, ranked, s, d = 0, 0, "", ""
            if row["reason"] == "eligible":
                j = np.flatnonzero(ids == int(row["base"]))
                hit = np.flatnonzero(pair == int(j[0])) if len(j) else np.zeros(0, np.int64)
                if len(hit):
                    ranked = 1
                    s, d = float(score[hit[0]]), float(match["dist"][hit[0]])
                    detected = int(s >= cut)
            w.writerow([int(row["base"]), row["members"], float(row["x"]), float(row["y"]),
                        row.get("tier", ""), int(bool(row["eligible"])), row["reason"],
                        detected, ranked, s, d])


# --------------------------------------------------------------------------
def main():
    sid = split_id(SPLIT_NAME)
    if sid == 1 and not ATTRS:
        raise SystemExit(
            "[v2-detr-val] split val: the default attributed survey CSV covers the test patch only. "
            "Set V2_DETR_ATTRS to an attributes file that covers val (sid 1).")
    if not ATTRS and not Path(DEFAULT_ATTRS).exists():
        raise SystemExit(f"[v2-detr-val] no attributes: set V2_DETR_ATTRS (default {DEFAULT_ATTRS} is missing)")
    frame = json.loads((OUT / "frame.json").read_text())
    origin = np.asarray(frame["origin_m"], np.float64).reshape(-1)
    if origin.size != 3 or not np.isfinite(origin).all():
        raise SystemExit(f"[v2-detr-val] frame origin_m {frame.get('origin_m')!r} is not 3 finite metres")
    regions = np.load(OUT / "regions.npz")
    split = regions["split"]
    truth = load_truth(regions, sid, attrs_path=ATTRS or None)
    audit = dict(truth["audit"])
    measured = int(audit["eligible"]) + int(audit["below_threshold"])
    if measured == 0:
        raise SystemExit(
            f"[v2-detr-val] split {SPLIT_NAME}: {audit['total']} survey bases but none has a measured "
            f"diameter (all unknown) — there is nothing to score against; refusing to load the "
            f"checkpoint or model. Check V2_DETR_ATTRS joins to label_bases members.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    cpath = checkpoint_path(CKPT)
    if not cpath.exists():
        raise SystemExit(f"[v2-detr-val] no checkpoint {cpath} — set V2_DETR_CKPT (models/ or absolute)")
    import torch
    ck = torch.load(cpath, map_location="cpu", weights_only=False)
    try:
        plan = checkpoint_plan(ck)
        threshold, tsource = resolve_threshold(THRESHOLD_ENV, ck)
    except ValueError as exc:
        raise SystemExit(f"[v2-detr-val] {cpath.name}: {exc}") from exc
    apply_geometry(plan)
    # A threshold below the stored min_p would have its detections dropped
    # silently by predict_region; lower min_p explicitly for this run instead.
    plan["min_p_applied"] = plan["min_p"]
    if threshold < plan["min_p"]:
        plan["min_p_applied"] = float(threshold)
        R.MIN_P = plan["min_p_applied"]
        warning = (f"threshold {threshold:g} is below the checkpoint min_p {plan['min_p']:g}; "
                   f"min_p lowered to {threshold:g} for this run so its detections are not dropped")
        plan["warnings"].append(warning)

    synthetic = plan["encoder"] == "fake"
    log(f"{SPLIT_NAME} split (sid {sid}): {audit['total']} survey bases | eligible DBH >= 10 cm: "
        f"{audit['eligible']} | below threshold: {audit['below_threshold']} | unknown: {audit['unknown']} "
        f"| height filter {audit['height_filter']}")
    log(f"checkpoint {cpath} step {ck.get('step')} | head {plan['hcfg']} | d_in {plan['d_in']} | "
        f"encoder {plan['encoder']} | threshold {threshold:.4f} ({tsource}; not tuned here)")
    log(f"geometry from checkpoint: voxel {plan['voxel']:g} m, cap {plan['cap']:,} points, "
        f"tokens <= {plan['max_tokens']:,}, crop {plan['crop']:g} m, own {plan['own']:g} m, "
        f"nms {plan['nms_r']:g} m, min_p {plan['min_p']:g}, grid {plan['grid']:g} m, seed {plan['seed']}")
    for warning in plan["warnings"]:
        log(f"!! {warning}")
    filt = ck.get("validation_filter")
    if filt != audit:
        log("!! the checkpoint's training validation_filter differs from this audit "
            f"(training: {filt!r}); reporting against the measured-DBH filter")
    if synthetic:
        log("!! SYNTHETIC ENCODER: this checkpoint was trained with models.FakeEncoder — "
            "TEST-ONLY; these numbers are not a model result")

    dev = T.DEVICE
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[v2-detr-val] V2_DEVICE=cuda but no GPU is visible — check the runtime")
    head = R.DetrHead(plan["d_in"], **plan["hcfg"]).to(dev)
    try:
        head.load_state_dict(ck["head"])
    except RuntimeError as exc:
        raise SystemExit(f"[v2-detr-val] checkpoint head does not match its recorded architecture: {exc}") from exc
    head.eval()
    enc = M.FakeEncoder() if synthetic else M.load_sonata(dev)
    runner = TR.DetrRunner(enc, head, plan["mu"], plan["sd"], dev)
    store = D.TileStore(labels=False)
    torch.manual_seed(int(plan["seed"]))
    det, sc = R.predict_region(runner.predict, store, tuple(truth["bbox"]), plan["cap"],
                               np.random.default_rng(int(plan["seed"])))
    det = np.asarray(det, np.float64).reshape(-1, 2)
    sc = np.asarray(sc, np.float64).reshape(-1)
    if not np.isfinite(det).all() or not np.isfinite(sc).all():
        raise SystemExit("[v2-detr-val] non-finite detections or scores")
    log(f"{len(det):,} detections (p >= {R.MIN_P:g}) in bbox {tuple(truth['bbox'])}, "
        f"{int((split == sid).sum()):,} split cells")

    matches = {r: match_split(det, sc, truth, split, sid, r) for r in RADII}
    m5 = matches[0.5]
    fixed_by = {f"{r:.1f}": fixed_payload(fixed_metrics(det, sc, matches[r], threshold), r) for r in RADII}
    fixed = fixed_by["0.5"]
    rmse = "null" if fixed["rmse_m"] is None else f"{fixed['rmse_m'] * 100:.1f} cm"
    log(f"@0.5 m, p >= {threshold:.4f}: TP {fixed['tp']} | FP {fixed['fp']} | FN {fixed['fn']} | "
        f"P {fixed['precision']:.3f} R {fixed['recall']:.3f} F1 {fixed['f1']:.3f} | RMSE {rmse} | "
        f"ignored {fixed['ignored']} (unknown base) + {fixed['outside']} (outside split)"
        + (f" | {fixed['duplicates']} duplicate(s) near an eligible base kept as FP"
           if fixed["duplicates"] else ""))
    metrics = dict(
        version=1, split=SPLIT_NAME, split_id=sid,
        checkpoint=dict(file=cpath.name, path=str(cpath), step=int(ck.get("step", -1)),
                        best=ck.get("best"), task=ck.get("task"), architecture=plan["hcfg"],
                        encoder=plan["encoder"], training_validation_filter=filt,
                        filter_matches_training=filt == audit),
        encoder=plan["encoder"], synthetic_encoder=synthetic,
        notes=[("height filter pending_not_applied: eligibility is measured DBH >= 10 cm only"
                if audit["height_filter"] == "pending_not_applied" else
                f"height filter {audit['height_filter']}"),
               "only targets are filtered; the scan is never masked with truth",
               "threshold fixed before scoring; nothing is tuned on the split"]
        + (["test-only synthetic encoder: these numbers are not a model result"] if synthetic else []),
        threshold=dict(value=float(threshold), source=tsource),
        geometry=dict(voxel=plan["voxel"], max_tokens=int(plan["max_tokens"]), cap=int(plan["cap"]),
                      crop=plan["crop"], own=plan["own"], nms_r=plan["nms_r"],
                      min_p=plan["min_p"], min_p_applied=plan["min_p_applied"],
                      grid=plan["grid"], seed=int(plan["seed"]),
                      sources=plan["sources"], warnings=plan["warnings"]),
        targets=dict(total=audit["total"], eligible=audit["eligible"],
                     unknown=audit["unknown"], below_threshold=audit["below_threshold"],
                     measured=measured, height_filter=audit["height_filter"], audit=audit),
        detections=dict(total=int(len(det)), at_threshold=int((sc >= threshold).sum()),
                        below_threshold=int((sc < threshold).sum())),
        fixed=fixed_by,
        fixed_threshold=fixed,
        ap={f"{r:.1f}": average_precision(sc, matches[r]["pair"], matches[r]["n_gt"]) for r in RADII},
        ignored=dict(unknown_base=fixed["ignored"], outside_split=fixed["outside"],
                     duplicates_kept_false=fixed["duplicates"]),
        frame=dict(origin_m=origin.tolist(), bbox=[float(v) for v in truth["bbox"]]),
    )
    write_json(RESULTS / f"stemnet_v2_{SPLIT_NAME}_detr_metrics.json", metrics)
    write_detections(RESULTS / f"stemnet_v2_{SPLIT_NAME}_detr_detections.csv", det, sc, m5,
                     threshold, truth["id"], origin)
    write_bases(RESULTS / f"stemnet_v2_{SPLIT_NAME}_detr_bases.csv", truth, m5, sc, threshold)
    log(f"AP@0.5 m {metrics['ap']['0.5']:.3f} | AP@0.3 m {metrics['ap']['0.3']:.3f}")

    if FIGURE:
        title = (f"StemNet v2 DETR, {SPLIT_NAME} split (measured DBH >= 10 cm), "
                 f"p >= {threshold:.2f}: F1 {fixed['f1']:.3f}, AP@0.5 {metrics['ap']['0.5']:.3f}"
                 + (" [SYNTHETIC ENCODER — test only]" if synthetic else ""))
        kept = sc >= threshold
        T.figure(truth["xy"], det[kept], m5["pair"][kept], tuple(truth["bbox"]),
                 RESULTS / f"stemnet_v2_{SPLIT_NAME}_detr.png", title=title)
    log(f"-> {RESULTS / f'stemnet_v2_{SPLIT_NAME}_detr_metrics.json'}, "
        f"stemnet_v2_{SPLIT_NAME}_detr_detections.csv, stemnet_v2_{SPLIT_NAME}_detr_bases.csv"
        + (f", stemnet_v2_{SPLIT_NAME}_detr.png" if FIGURE else ""))


if __name__ == "__main__":
    main()
