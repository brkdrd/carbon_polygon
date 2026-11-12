#!/usr/bin/env python3
"""Checks for the standalone DETR validation stage (v2_validate_detr).

Two kinds of checks, both run in a subprocess so the flat `import core` /
`import detr` module layout cannot leak between test modules and the scratch
TMPDIR is this test's own directory:

  * oracle scoring: hand-built detections over a known truth set. Eligible
    bases score; a known but sub-threshold stem stays a negative (an unmatched
    detection on it is false); a detection near an unknown-diameter base is
    ignored only after eligible matching, so a duplicate near an eligible base
    stays false; an unmatched detection outside the split is ignored and
    counted apart.
  * the real script: tiny prepared tiles (grid + 12 covariance features, no
    per-point labels), a fake checkpoint with a NON-default architecture, and
    real subprocess runs. It must use the checkpoint's architecture, encoder,
    voxel/cap/token geometry and eval cut, write finite JSON and the CSVs, and
    fail before the model loads when the split has no measured diameter.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
V2 = HERE / "stemnet_v2"
SCRIPT = V2 / "validate_detr.py"
ORIGIN = (736000.0, 4767000.0, 50.0)
SPLIT = "test"


# --------------------------------------------------------------------------
# subprocess plumbing
# --------------------------------------------------------------------------
def clean_env(tmp: Path, **extra) -> dict:
    """No inherited V2_* / DATA_DIR / RESULTS_DIR; scratch inside `tmp`."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("V2_", "DATA_DIR", "RESULTS_DIR")) and k != "PYTHONPATH"}
    scratch = tmp / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    env.update(DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "results"),
               TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch), V2_DEVICE="cpu")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_code(code: str, tmp: Path, **extra) -> subprocess.CompletedProcess:
    head = f"import sys\nsys.path.insert(0, {str(V2)!r})\n"
    return subprocess.run([sys.executable, "-c", head + code], cwd=str(HERE),
                          env=clean_env(tmp, **extra), capture_output=True, text=True)


def run_script(tmp: Path, **extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=str(HERE),
                          env=clean_env(tmp, **extra), capture_output=True, text=True)


def out_of(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# oracle scoring
# --------------------------------------------------------------------------
SCORING = r'''
import json
import numpy as np
import validate_detr as V

truth = dict(
    xy=np.array([[1.0, 1.0], [5.0, 5.0]]),
    id=np.array([7, 9]),
    rows=[dict(base=7, x=1.0, y=1.0, eligible=True, reason='eligible'),
          dict(base=9, x=5.0, y=5.0, eligible=True, reason='eligible'),
          dict(base=11, x=1.8, y=1.0, eligible=False, reason='unknown'),
          dict(base=12, x=3.0, y=3.0, eligible=False, reason='unknown'),
          dict(base=13, x=7.0, y=7.0, eligible=False, reason='below_threshold')],
)
det = np.array([[1.05, 1.00],   # A tp, 5 cm
                [1.35, 1.00],   # B duplicate near eligible 7 AND unknown 11
                [1.95, 1.00],   # C near unknown 11 only
                [3.00, 3.00],   # D on unknown 12
                [7.00, 7.00],   # E on the below-threshold stem 13
                [9.50, 5.00],   # F outside the split cells
                [5.02, 5.00]])  # G tp, 2 cm, below the 0.5 cut
score = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.95, 0.30])
split = np.full((10, 10), 2, np.uint8)
split[:, 9:] = 0
m5 = V.match_split(det, score, truth, split, 2, 0.5)
res = dict(
    pair=m5['pair'].tolist(),
    outside=m5['outside'].tolist(),
    unknown=m5['unknown'].tolist(),
    duplicates=m5['duplicates'].tolist(),
    dist=[None if not np.isfinite(v) else float(v) for v in m5['dist']],
    ap5=V.average_precision(score, m5['pair'], 2),
    fixed=V.fixed_metrics(det, score, m5, 0.5),
    m3=V.match_split(det, score, truth, split, 2, 0.3),
    ap3=None,
)
res['ap3'] = V.average_precision(score, res['m3']['pair'], 2)
res['pair3'] = res['m3']['pair'].tolist()
res.pop('m3')
tiny = dict(xy=np.zeros((0, 2)), id=np.zeros(0, np.int64),
            rows=[dict(base=1, x=2.0, y=2.0, eligible=False, reason='below_threshold')])
mt = V.match_split(np.array([[2.0, 2.0], [4.0, 4.0]]), np.array([0.4, 0.9]), tiny,
                   np.full((8, 8), 2, np.uint8), 2, 0.5)
res['tiny'] = V.fixed_metrics(np.array([[2.0, 2.0], [4.0, 4.0]]),
                              np.array([0.4, 0.9]), mt, 0.5)
res['tiny_ap'] = V.average_precision(np.array([0.4, 0.9]), mt['pair'], 0)
# A high-scoring detection in another split's cell must not steal the base
# from an inside detection: outside detections are ignored before matching.
steal_truth = dict(xy=np.array([[1.0, 1.0]]), id=np.array([5]),
                   rows=[dict(base=5, x=1.0, y=1.0, eligible=True, reason='eligible')])
steal_split = np.full((4, 4), 2, np.uint8)
steal_split[1, 0] = 0
ms = V.match_split(np.array([[0.95, 1.0], [1.4, 1.0]]), np.array([0.95, 0.5]),
                   steal_truth, steal_split, 2, 0.5)
res['steal'] = dict(pair=ms['pair'].tolist(), outside=ms['outside'].tolist(),
                    fixed=V.fixed_metrics(np.array([[0.95, 1.0], [1.4, 1.0]]),
                                          np.array([0.95, 0.5]), ms, 0.5))
print(json.dumps(res))
'''


def test_oracle_scoring_ignores_only_unknown_bases_and_counts_outside(tmp_path):
    tmp = tmp_path
    p = run_code(SCORING, tmp)
    assert p.returncode == 0, out_of(p)
    r = json.loads(p.stdout.strip().splitlines()[-1])

    pair = r["pair"]
    assert pair[0] == 0, "A matched eligible base 0"                     # tp
    assert pair[1] == -1, "B duplicates an eligible base -> stays FP"    # dup, near unknown too
    assert pair[2] == -2, "C is near an unknown base only -> ignored"
    assert pair[3] == -2, "D sits on an unknown base -> ignored"
    assert pair[4] == -1, "E sits on a known below-threshold stem -> FP"
    assert pair[5] == -2 and r["outside"][5], "F is outside the split -> ignored, counted apart"
    assert pair[6] == 1, "G matched eligible base 1"
    assert r["duplicates"][1] and not r["duplicates"][0]
    assert sum(r["outside"]) == 1 and sum(r["unknown"]) == 2

    assert abs(r["dist"][0] - 0.05) < 1e-9 and abs(r["dist"][6] - 0.02) < 1e-9
    assert r["dist"][1] is None and r["dist"][5] is None

    fixed = r["fixed"]
    # at cut 0.5: A,E,B,C,D,F kept; G below. TP 1, FP 2 (B,E), ignored 2, outside 1.
    assert (fixed["tp"], fixed["fp"], fixed["fn"]) == (1, 2, 1)
    assert fixed["ignored"] == 2 and fixed["outside"] == 1
    assert fixed["kept"] == fixed["tp"] + fixed["fp"] + fixed["ignored"] + fixed["outside"] == 6
    assert fixed["duplicates"] == 1
    assert abs(fixed["rmse"] - 0.05) < 1e-9
    assert fixed["threshold"] == 0.5

    # ranked A(.9) B(.8) E(.5) G(.3) after ignored F/C/D; the envelope lifts
    # the lower-recall points to G's precision .5 -> AP = .5*1 + .5*.5 = .75
    assert abs(r["ap5"] - 0.75) < 1e-9, r["ap5"]
    # At 0.3 m the duplicate and the unknown-base detections change sides, never eligibility.
    assert abs(r["ap3"] - 0.75) < 1e-9, (r["ap3"], r["pair3"])
    assert r["pair3"][1] == -1 and r["pair3"][2] == -2 and r["pair3"][4] == -1


def test_oracle_outside_detection_cannot_steal_an_eligible_base(tmp_path):
    p = run_code(SCORING, tmp_path)
    assert p.returncode == 0, out_of(p)
    r = json.loads(p.stdout.strip().splitlines()[-1])["steal"]
    assert r["pair"] == [-2, 0], "outside det ignored; the inside det keeps the base"
    assert r["outside"] == [True, False]
    assert (r["fixed"]["tp"], r["fixed"]["fp"], r["fixed"]["outside"]) == (1, 0, 1)


def test_oracle_zero_eligible_is_valid_and_scores_detections_as_false(tmp_path):
    p = run_code(SCORING, tmp_path)
    assert p.returncode == 0, out_of(p)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    t = r["tiny"]
    assert (t["tp"], t["fp"], t["fn"], t["ignored"], t["outside"]) == (0, 1, 0, 0, 0)
    assert t["precision"] == 0.0 and t["recall"] == 0.0 and t["f1"] == 0.0
    assert t["rmse"] is None, "no hits -> null RMSE, not NaN"
    assert r["tiny_ap"] == 0.0


# --------------------------------------------------------------------------
# checkpoint policy
# --------------------------------------------------------------------------
CHECKPOINT = r'''
import json
import numpy as np
import torch
import validate_detr as V

CFG = dict(encoder='fake', voxel=0.25, cap=1000, max_tokens=500, crop=8.0, own=4.0,
           nms_r=0.3, min_p=0.01, grid=0.02, seed=0)

def ck(**over):
    d = dict(task='detr', d_in=1244,
             hcfg=dict(d=8, n_query=9, layers=1, heads=2, ff=16),
             mu=np.zeros(12, np.float32), sd=np.ones(12, np.float32),
             head=dict(a=torch.zeros(2)), step=7, cfg=dict(CFG))
    d.update(over)
    return d

plan = V.checkpoint_plan(ck())
keys = ('max_tokens', 'crop', 'own', 'nms_r', 'min_p', 'grid', 'seed')
out = dict(plan=dict(d_in=plan['d_in'], hcfg=plan['hcfg'], encoder=plan['encoder'],
                     voxel=plan['voxel'], cap=plan['cap'],
                     **{k: plan[k] for k in keys},
                     sources={k: plan['sources'][k] for k in keys},
                     warnings=plan['warnings']))
bad, ok = {}, 0
for name, c in (('task', ck(task='base')),
                ('sd_zero', ck(sd=np.zeros(12, np.float32))),
                ('mu_nan', ck(mu=np.full(12, np.nan, np.float32))),
                ('mu_size', ck(mu=np.zeros(8, np.float32))),
                ('dim', ck(d_in=1232)),
                ('encoder', ck(cfg=dict(CFG, encoder='other'))),
                ('no_cfg', {k: v for k, v in ck().items() if k != 'cfg'}),
                ('no_hcfg_layer', ck(hcfg=dict(d=8, n_query=9, heads=2, ff=16))),
                ('hcfg_zero', ck(hcfg=dict(d=0, n_query=9, layers=1, heads=2, ff=16))),
                ('voxel', ck(cfg=dict(CFG, voxel=0.0))),
                ('cap', ck(cfg=dict(CFG, cap=0))),
                ('cap_float', ck(cfg=dict(CFG, cap=1000.5))),
                ('cap_bool', ck(cfg=dict(CFG, cap=True))),
                ('tokens_float', ck(cfg=dict(CFG, max_tokens=500.5))),
                ('tokens_bool', ck(cfg=dict(CFG, max_tokens=True))),
                ('nan_weight', ck(head=dict(a=torch.tensor([float('nan')])))),
                ('no_task', {k: v for k, v in ck().items() if k != 'task'})):
    try:
        V.checkpoint_plan(c)
    except ValueError:
        ok += 1
    else:
        bad[name] = True

old = dict(encoder='fake', voxel=0.5, cap=123)
legacy = V.checkpoint_plan(ck(cfg=old))
print(json.dumps(dict(plan=out['plan'], bad=bad, rejected=ok,
                      legacy=dict(voxel=legacy['voxel'], max_tokens=legacy['max_tokens'],
                                  sources={k: legacy['sources'][k] for k in keys},
                                  warnings=legacy['warnings']))))
'''


def test_checkpoint_plan_requires_detr_policy_and_consistent_geometry(tmp_path):
    p = run_code(CHECKPOINT, tmp_path)
    assert p.returncode == 0, out_of(p)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r["bad"] == {}, f"accepted invalid checkpoints: {r['bad']}"
    assert r["rejected"] == 17
    plan = r["plan"]
    assert plan["hcfg"] == dict(d=8, n_query=9, layers=1, heads=2, ff=16)
    assert plan["encoder"] == "fake" and plan["voxel"] == 0.25
    assert (plan["cap"], plan["max_tokens"]) == (1000, 500)
    assert (plan["nms_r"], plan["min_p"], plan["grid"], plan["seed"]) == (0.3, 0.01, 0.02, 0)
    keys = ("max_tokens", "crop", "own", "nms_r", "min_p", "grid", "seed")
    assert all(plan["sources"][k] == "checkpoint" for k in keys)
    assert plan["warnings"] == []


def test_checkpoint_plan_flags_geometry_missing_from_older_checkpoints(tmp_path):
    p = run_code(CHECKPOINT, tmp_path)
    assert p.returncode == 0, out_of(p)
    r = json.loads(p.stdout.strip().splitlines()[-1])["legacy"]
    assert r["voxel"] == 0.5
    keys = ("max_tokens", "crop", "own", "nms_r", "min_p", "grid", "seed")
    assert all(r["sources"][k] == "module_default" for k in keys)
    assert len(r["warnings"]) == len(keys), r["warnings"]
    assert all("checkpoint" in w for w in r["warnings"])


# --------------------------------------------------------------------------
# prepared tiles + checkpoint for the real script
# --------------------------------------------------------------------------
def prepare(tmp: Path, diameters=("20", "8", "15", "")):
    root = tmp / "data" / "stemnet_v2"
    g, f = root / "grid0.02", root / "feat0.02"
    g.mkdir(parents=True)
    f.mkdir(parents=True)
    ax = np.arange(0.0, 10.001, 0.1)
    gx, gy = np.meshgrid(ax, ax)
    xyz = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    np.savez(g / "tile_0_0.npz", xyz=xyz.astype(np.float32),
             rgb=np.zeros((len(xyz), 3), np.uint8), intensity=np.zeros(len(xyz), np.uint8))
    feat = np.zeros((len(xyz), 12), np.float16)
    feat[:, 10] = 1.0                                      # normals (0, 0, 1), cols 8..10
    np.savez(f / "tile_0_0.npz", feat=feat)

    bases = [(0, 2.0, 2.0, "A", "1-1"), (1, 4.0, 4.0, "C", "1-2"),
             (2, 6.0, 6.0, "B", "1-3"), (3, 6.2, 2.0, "C", "1-4")]
    with (root / "label_bases.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["base", "x", "y", "tier", "members", "radius"])
        for i, x, y, tier, member in bases:
            w.writerow([i, x, y, tier, member, "0.05"])
    with (root / "attrs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "diameter_cm"])
        for (i, *_), d in zip(bases, diameters):
            w.writerow([f"1-{i + 1}", d])

    split = np.full((12, 12), 2, np.uint8)
    split[10:, :] = 0
    split[:, 10:] = 0
    np.savez(root / "regions.npz", split=split, occupancy=np.ones_like(split))
    (root / "frame.json").write_text(json.dumps(dict(origin_m=list(ORIGIN), labels={})))
    return root


def make_checkpoint(tmp: Path, hcfg=None, cut=0.5, cfg_extra=None) -> Path:
    path = tmp / "data" / "stemnet_v2" / "models" / "detr_best.pth"
    code = f'''
import numpy as np
import torch
import detr as R
import train_detr as train
h = {hcfg or dict(d=8, n_query=9, layers=1, heads=2, ff=16)}
head = R.DetrHead(1244, **h)
with torch.no_grad():
    head.cls.bias.fill_(2.0)   # p ~ 0.88 at the initial reference grid: real kept detections
opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
train.save({str(path)!r}, head, opt, 7, 0.0,
           np.zeros(12, np.float32), np.ones(12, np.float32),
           dict(cut={cut!r}, best_f1=0.0))
# the geometry keys v2_train_detr now records (restored by the validator)
ck = torch.load({str(path)!r}, map_location='cpu', weights_only=False)
ck['cfg'].update(dict(nms_r=0.3, min_p=0.01, grid=0.02, seed=0))
ck['cfg'].update({cfg_extra or {}})
torch.save(ck, {str(path)!r})
print("checkpoint written")
'''
    p = run_code(code, tmp, V2_ENCODER="fake", V2_CAP="1000", V2_DETR_VOXEL="0.25",
                 V2_DETR_MAX_TOKENS="500")
    assert p.returncode == 0, out_of(p)
    return path


def read_metrics(tmp: Path, split=SPLIT) -> dict:
    return json.loads((tmp / "results" / f"stemnet_v2_{split}_detr_metrics.json").read_text())


def rows_of(path: Path) -> list:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_script_scores_the_checkpoint_end_to_end(tmp_path):
    root = prepare(tmp_path)
    ckpt = make_checkpoint(tmp_path)
    assert ckpt.exists()
    p = run_script(tmp_path, V2_DETR_SPLIT=SPLIT, V2_DETR_CKPT="detr_best.pth",
                   V2_DETR_ATTRS=str(root / "attrs.csv"),
                   V2_DETR_D="256", V2_DETR_QUERIES="99", V2_DETR_LAYERS="6",
                   V2_DETR_NMS="9", V2_DETR_MIN_P="0.9")
    assert p.returncode == 0, out_of(p)
    assert "synthetic" in p.stdout.lower(), "fake encoder must be flagged unmistakably"

    res = tmp_path / "results"
    text = (res / f"stemnet_v2_{SPLIT}_detr_metrics.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    m = json.loads(text)
    assert m["split"] == SPLIT and m["split_id"] == 2
    # checkpoint, not the training/env config
    assert m["checkpoint"]["architecture"] == dict(d=8, n_query=9, layers=1, heads=2, ff=16)
    assert m["encoder"] == "fake" and m["synthetic_encoder"] is True
    assert m["geometry"]["voxel"] == 0.25 and m["geometry"]["max_tokens"] == 500
    assert m["geometry"]["cap"] == 1000
    # restored from the checkpoint, not the inherited V2_DETR_NMS / V2_DETR_MIN_P
    assert (m["geometry"]["nms_r"], m["geometry"]["min_p"]) == (0.3, 0.01)
    assert (m["geometry"]["grid"], m["geometry"]["seed"]) == (0.02, 0)
    assert m["geometry"]["min_p_applied"] == 0.01
    assert m["detections"]["total"] > 0, "V2_DETR_MIN_P=0.9 must not leak into scoring"
    t = m["targets"]
    assert (t["total"], t["eligible"], t["unknown"], t["below_threshold"]) == (4, 2, 1, 1)
    assert t["height_filter"] == "pending_not_applied"
    assert m["threshold"] == dict(value=0.5, source="checkpoint_eval_cut")
    assert "ap" in m and set(m["ap"]) == {"0.5", "0.3"}
    assert 0.0 <= m["ap"]["0.5"] <= 1.0 and 0.0 <= m["ap"]["0.3"] <= 1.0
    fixed = m["fixed_threshold"]
    assert set(m["fixed"]) == {"0.5", "0.3"}
    assert m["fixed"]["0.5"]["radius_m"] == 0.5 and m["fixed"]["0.3"]["radius_m"] == 0.3
    assert fixed["radius_m"] == 0.5 and fixed["threshold"] == 0.5
    assert fixed["kept"] == fixed["tp"] + fixed["fp"] + fixed["ignored"] + fixed["outside"]
    assert fixed["tp"] + fixed["fn"] == t["eligible"]
    assert fixed["rmse_m"] is None or fixed["rmse_m"] >= 0
    # Non-vacuous end to end: kept detections exist, both measured bases are
    # found, the detection near the unknown base is ignored (not false), and
    # the ones on the split edge are counted apart from false positives.
    assert fixed["kept"] == m["detections"]["at_threshold"] >= 5
    assert (fixed["tp"], fixed["fn"]) == (2, 0)
    assert fixed["ignored"] == 1 and fixed["outside"] == 5
    assert fixed["fp"] == fixed["kept"] - 2 - 1 - 5
    assert (res / f"stemnet_v2_{SPLIT}_detr_detections.csv").exists()
    assert (res / f"stemnet_v2_{SPLIT}_detr_bases.csv").exists()
    assert not (res / "stemnet_v2_val_detections_detr.csv").exists(), "training artifact name"
    assert (res / f"stemnet_v2_{SPLIT}_detr.png").stat().st_size > 1000

    dets = rows_of(res / f"stemnet_v2_{SPLIT}_detr_detections.csv")
    assert dets, "the fake checkpoint must produce some detections"
    bases = {int(r["base"]): r for r in rows_of(res / f"stemnet_v2_{SPLIT}_detr_bases.csv")}
    assert set(bases) == {0, 1, 2, 3}
    assert [int(bases[i]["eligible"]) for i in (0, 1, 2, 3)] == [1, 0, 1, 0]
    assert [bases[i]["reason"] for i in (0, 1, 2, 3)] == ["eligible", "below_threshold",
                                                          "eligible", "unknown"]
    ids = {int(r["base"]) for r in bases.values() if r["eligible"] == "1"}
    kept = [r for r in dets if r["above_threshold"] == "1"]
    for r in dets:
        x, y = float(r["x"]), float(r["y"])
        assert abs(float(r["utm_x"]) - (x + ORIGIN[0])) < 1e-9
        assert abs(float(r["utm_y"]) - (y + ORIGIN[1])) < 1e-9
        mb = int(r["matched_base"])
        assert mb in ids or mb in (-1, -2), f"matched_base {mb} is not a stable base id"
        if mb >= 0:
            assert r["status"] == "tp" and abs(float(r["distance_m"])) >= 0
        else:
            assert r["status"] in ("false", "ignored_unknown", "ignored_outside")
    counts = {k: sum(r["status"] == k for r in kept)
              for k in ("tp", "false", "ignored_unknown", "ignored_outside")}
    assert counts["tp"] == fixed["tp"] and counts["false"] == fixed["fp"]
    assert counts["ignored_unknown"] == fixed["ignored"]
    assert counts["ignored_outside"] == fixed["outside"]
    assert len(kept) == fixed["kept"] == m["detections"]["at_threshold"]


def test_threshold_override_is_used_verbatim_and_never_searched(tmp_path):
    root = prepare(tmp_path)
    make_checkpoint(tmp_path)
    kw = dict(V2_DETR_SPLIT=SPLIT, V2_DETR_CKPT="detr_best.pth",
              V2_DETR_ATTRS=str(root / "attrs.csv"), V2_DETR_FIGURE="0")
    p = run_script(tmp_path, V2_DETR_THRESHOLD="0.9", **kw)
    assert p.returncode == 0, out_of(p)
    m = read_metrics(tmp_path)
    assert m["threshold"] == dict(value=0.9, source="environment")
    assert m["fixed_threshold"]["threshold"] == 0.9
    dump = json.dumps(m).lower()
    assert "best cut" not in dump and "tuned on test" not in dump
    assert "not tuned here" in p.stdout.lower()


def test_per_base_detection_is_reported_at_the_fixed_threshold(tmp_path):
    root = prepare(tmp_path)
    make_checkpoint(tmp_path)
    p = run_script(tmp_path, V2_DETR_SPLIT=SPLIT, V2_DETR_CKPT="detr_best.pth",
                   V2_DETR_ATTRS=str(root / "attrs.csv"), V2_DETR_THRESHOLD="0.99",
                   V2_DETR_FIGURE="0")
    assert p.returncode == 0, out_of(p)
    m = read_metrics(tmp_path)
    assert m["fixed_threshold"]["kept"] == 0 and m["fixed_threshold"]["tp"] == 0
    assert m["fixed_threshold"]["fn"] == m["targets"]["eligible"] == 2
    bases = rows_of(tmp_path / "results" / f"stemnet_v2_{SPLIT}_detr_bases.csv")
    eligible = [r for r in bases if r["eligible"] == "1"]
    assert len(eligible) == 2
    for r in eligible:
        assert r["detected"] == "0", "matched but below the fixed threshold is not detected"
        assert r["detected_any_score"] == "1", "ranked match stays visible in its own column"
    assert all(r["above_threshold"] == "0" for r in
               rows_of(tmp_path / "results" / f"stemnet_v2_{SPLIT}_detr_detections.csv"))


def test_threshold_below_the_stored_min_p_lowers_it_explicitly(tmp_path):
    root = prepare(tmp_path)
    make_checkpoint(tmp_path, cfg_extra={"min_p": 0.95})
    p = run_script(tmp_path, V2_DETR_SPLIT=SPLIT, V2_DETR_CKPT="detr_best.pth",
                   V2_DETR_ATTRS=str(root / "attrs.csv"), V2_DETR_THRESHOLD="0.005",
                   V2_DETR_FIGURE="0")
    assert p.returncode == 0, out_of(p)
    m = read_metrics(tmp_path)
    assert m["geometry"]["min_p"] == 0.95
    assert m["geometry"]["min_p_applied"] == 0.005
    assert m["detections"]["total"] > 0, "the threshold's detections must not be dropped silently"
    assert "min_p" in p.stdout.lower()


def test_val_split_requires_supplied_attributes(tmp_path):
    p = run_script(tmp_path, V2_DETR_SPLIT="val")
    assert p.returncode != 0
    assert "V2_DETR_ATTRS" in out_of(p), out_of(p)


def test_unknown_split_name_is_rejected(tmp_path):
    p = run_script(tmp_path, V2_DETR_SPLIT="validation")
    assert p.returncode != 0
    assert "val or test" in out_of(p), out_of(p)


def test_zero_measured_diameters_fail_before_the_checkpoint_or_model_is_loaded(tmp_path):
    root = prepare(tmp_path, diameters=("", "", "", ""))
    p = run_script(tmp_path, V2_DETR_SPLIT=SPLIT, V2_DETR_CKPT="does_not_exist.pth",
                   V2_DETR_ATTRS=str(root / "attrs.csv"))
    assert p.returncode != 0
    text = out_of(p)
    assert "measured diameter" in text, text
    assert "does_not_exist" not in text, "must fail before resolving/loading the checkpoint"
    assert "synthetic" not in text.lower() and "sonata" not in text.lower()
    assert not (tmp_path / "results" / f"stemnet_v2_{SPLIT}_detr_metrics.json").exists()


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
