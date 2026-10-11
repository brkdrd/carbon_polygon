#!/usr/bin/env python3
"""GeoWalker stage C — walk every seed, NMS, score against the RTK bases.

No distance field is used here: at inference the model only ever sees the sphere
around its current position (plus its last GW_MEM spheres), which is the whole
point. Detections are walker endpoints, scored by the read-out head's own
estimate of how far it still is from a base through the cloud.

GW_INFER_HEAD=1 ranks those same endpoints with the separately trained tree head
(gw_head) instead, and reports both rankings side by side. NMS keeps the
highest-scoring endpoint of a cluster, so the two scorers do not merely
re-threshold the same detections — they choose different ones, which is why the
comparison redoes NMS rather than reusing it.

GW_AGGREGATE=1 adds the post-walk evidence stage (aggregate.py): the best step
of each walk instead of its last, converging walkers counted as votes instead of
suppressed, and the canopy-height cut gw_height derived from the labels. Every
candidate pipeline is scored the same way and reported in one table; the first
one is what the figures and CSVs describe.

GW_INFER_REGION=patch restricts the walk to the densely attributed survey
patch (patch.py): 392 trees in 2 ha with a species and a girth on 388 of them,
which is the only region where a per-tree and per-species answer sheet is
possible. It writes its own files so it cannot overwrite a full run, and it
reports loudly that the patch is TRAINING ground — see patch.py.

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
USE_HEAD = os.environ.get("GW_INFER_HEAD", "0") == "1"
AGG = os.environ.get("GW_AGGREGATE", "0") == "1"
# surveyed-region grid: cell size and how far to dilate around a labelled base
SURVEY_CELL = float(os.environ.get("GW_SURVEY_CELL", "10.0"))
SURVEY_DILATE = int(os.environ.get("GW_SURVEY_DILATE", "2"))


def detections_figure(scene, seeds, gt, det_xyz, f1, region, out,
                      inside_mask=None, f1_surveyed=None):
    """Top-down GT vs detections.

    Detections outside the surveyed region are drawn apart from the rest: the
    RTK survey does not cover the whole scanned scene, so a detection out there
    has no green cross to match and reads as a false positive in the picture
    whether or not a tree is standing at it. Keeping them one colour made the
    map look far worse than the surveyed-only metric says it is.
    """
    xyz = scene.xyz.cpu().numpy()
    x0, x1 = seeds[:, 0].min() - 5, seeds[:, 0].max() + 5
    y0, y1 = seeds[:, 1].min() - 5, seeds[:, 1].max() + 5
    m = (xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1)
    sub = xyz[m][np.random.permutation(int(m.sum()))[:400_000]]
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(sub[:, 0], sub[:, 1], s=0.05, c="0.75", marker=".", linewidths=0)
    ax.scatter(gt[:, 0], gt[:, 1], s=42, marker="x", c="green", label="GT base")
    if inside_mask is None:
        ax.scatter(det_xyz[:, 0], det_xyz[:, 1], s=26, marker="o",
                   facecolors="none", edgecolors="red", label="detection")
    else:
        ins = np.asarray(inside_mask, bool)
        ax.scatter(det_xyz[~ins, 0], det_xyz[~ins, 1], s=26, marker="o",
                   facecolors="none", edgecolors="darkorange", alpha=0.7,
                   label=f"detection, unsurveyed ({int((~ins).sum())})")
        ax.scatter(det_xyz[ins, 0], det_xyz[ins, 1], s=26, marker="o",
                   facecolors="none", edgecolors="red",
                   label=f"detection, judgeable ({int(ins.sum())})")
    ax.set_aspect("equal")
    ax.legend()
    title = (f"GeoWalker detections ({region}) | F1@0.5m {f1:.3f}"
             + (f" | surveyed only {f1_surveyed:.3f}"
                if f1_surveyed is not None else "")
             + f" | {len(det_xyz)} det / {len(gt)} GT")
    ax.set_title(title)
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


def load_tree_head(dev):
    """The statically trained tree/not-tree head from gw_head.

    Its geometry comes from the checkpoint, never the environment: it was trained
    on a sphere of one radius with one block count and means nothing at another.
    It is read off the walk's own final sphere, which is only the same geometry
    when that radius IS the walker's — so a mismatch is an error here, not a
    silent re-interpretation of somebody else's tokens.
    """
    import treehead as TH
    if not TH.HEAD_CKPT.exists():
        raise SystemExit(f"[gw-infer] GW_INFER_HEAD=1 but {TH.HEAD_CKPT} is "
                         f"missing — run gw_head first")
    ck = torch.load(TH.HEAD_CKPT, map_location=dev, weights_only=False)
    sd = ck["state_dict"]
    blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("q_proj."))
    head = TH.TreeHead(blocks=blocks).to(dev)
    head.load_state_dict(sd)
    head.eval()
    r = float(ck.get("head_r", M.SPHERE_R))
    if abs(r - M.SPHERE_R) > 1e-6:
        raise SystemExit(
            f"[gw-infer] the head was trained on a {r} m sphere but the walker "
            f"encodes {M.SPHERE_R} m, so it cannot be read off the walk's own "
            f"tokens. Retrain it with GW_HEAD_R={M.SPHERE_R}, or leave "
            f"GW_INFER_HEAD unset.")
    print(f"[gw-infer] tree head: {blocks} blocks | sphere {r} m | trained to "
          f"val AUC {float(ck.get('auc', float('nan'))):.4f} at epoch "
          f"{ck.get('epoch')}")
    return head, r


def sweep_scorer(ends, sc, gt, inside, nms=True):
    """NMS + the threshold sweep for ONE ranking of candidates.

    Returns (keep, in_survey, sweep, best_all, best_surveyed). Every pipeline
    goes through this, so a comparison cannot accidentally be run two different
    ways. `nms=False` is for candidates whose own clustering already did the
    suppression — running it twice would quietly merge neighbouring stems.
    """
    keep = (M.nms(ends, sc, radius=M.NMS_R) if nms
            else np.arange(len(ends)))
    xyz, s = ends[keep], sc[keep]
    ins = inside(xyz[:, :2])
    sweep = []
    for thr in M.sweep_thresholds(s):
        k = s >= thr
        sweep.append(dict(
            thresh=float(thr), n=int(k.sum()),
            all=M.match_metrics(xyz[k], s[k], gt, radius=0.5),
            surveyed=M.match_metrics(xyz[k & ins], s[k & ins], gt, radius=0.5)))
    return (keep, ins, sweep, max(sweep, key=lambda r: r["all"]["f1"]),
            max(sweep, key=lambda r: r["surveyed"]["f1"]))


def patch_tables(gt, det_xyz, det_s, attrs, tag, radius=0.5):
    """Per-TREE and per-DETECTION answer sheets for the attributed patch.

    The full run reports aggregates; on this patch the survey is complete
    enough to say which individual tree was found and which was missed, with
    the species beside it. That is the table the people who commissioned the
    survey can actually check, and the one place a per-species recall is
    meaningful — so a miss gets a row of its own rather than only a count.
    """
    import patch as P
    pairs, fp_i, fn_i = M.match_pairs(det_xyz, det_s, gt, radius)
    sp, known = P.join_species(gt[:, :2], attrs=attrs)

    found = np.zeros(len(gt), bool)
    dist = np.full(len(gt), np.nan)
    score = np.full(len(gt), np.nan)
    for i, j in pairs:
        found[j] = True
        dist[j] = float(np.linalg.norm(det_xyz[i] - gt[j]))
        score[j] = float(det_s[i])

    trees = C.RESULTS / f"geowalker{tag}_trees.csv"
    with trees.open("w", newline="") as f:
        f.write("survey_n,species,girth_cm,x,y,z,found,err_m,score\n")
        for j in range(len(gt)):
            f.write(f"{sp['n'][j]},{sp['species'][j]},{sp['girth_cm'][j]},"
                    f"{gt[j,0]:.3f},{gt[j,1]:.3f},{gt[j,2]:.3f},"
                    f"{int(found[j])},"
                    f"{'' if not found[j] else f'{dist[j]:.3f}'},"
                    f"{'' if not found[j] else f'{score[j]:.4f}'}\n")

    gt_of = {int(i): int(j) for i, j in pairs}
    dets = C.RESULTS / f"geowalker{tag}_detections_status.csv"
    with dets.open("w", newline="") as f:
        f.write("x,y,z,score,status,survey_n,species,err_m\n")
        for i in range(len(det_xyz)):
            j = gt_of.get(i)
            st = "tp" if j is not None else "fp"
            f.write(f"{det_xyz[i,0]:.3f},{det_xyz[i,1]:.3f},{det_xyz[i,2]:.3f},"
                    f"{det_s[i]:.4f},{st},"
                    f"{'' if j is None else sp['n'][j]},"
                    f"{'' if j is None else sp['species'][j]},"
                    f"{'' if j is None else f'{dist[j]:.3f}'}\n")

    # per-species recall, which is the whole reason this patch is worth running
    import collections
    per = collections.defaultdict(lambda: [0, 0])
    for j in range(len(gt)):
        name = sp["species"][j]
        if not name:
            continue
        per[name][1] += 1
        per[name][0] += int(found[j])
    print(f"[gw-infer] patch: {int(found.sum())}/{len(gt)} GT bases found "
          f"@{radius} m | {len(fp_i)} false positives | {len(fn_i)} missed")
    print(f"[gw-infer] of those, {int(known.sum())} carry a surveyed species "
          f"(the rest are stem-map trees nobody attributed, and still count)")
    att = known & found
    print(f"[gw-infer] attributed trees found: {int(att.sum())}/"
          f"{int(known.sum())} = {att.sum()/max(known.sum(),1)*100:.1f}%")
    print(f"[gw-infer] ===== recall by species (n >= 3) =====")
    for name, (h, n) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        if n >= 3:
            print(f"   {name:28s} {h:3d}/{n:3d} = {h/n*100:5.1f}%")
    print(f"[gw-infer] -> {trees.name}, {dets.name}")


def main():
    dev = M.DEVICE
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

    region = os.environ.get("GW_INFER_REGION", "val")   # val | all | patch
    patch_info = patch_attrs = None
    if region == "val":
        seeds = sd["seeds"][sd["is_val"]]
        gt = bases[lab["is_val"]]
    elif region == "patch":
        import patch as P
        patch_attrs = P.load_attributes()
        seeds, gt, _rows, patch_info = P.select(
            sd["seeds"], bases, lab["is_val"], lab["in_margin"],
            attrs=patch_attrs)
        print(f"[gw-infer] patch: {patch_info['n_surveyed_trees']} surveyed "
              f"trees | {patch_info['n_species']} species | "
              f"{patch_info['area_ha']:.2f} ha | margin "
              f"{patch_info['margin_m']:.0f} m")
        # say what this will cost before spending it: the walk is ~4.6 sphere
        # encodes/s on one card and every seed costs GW_STEPS+1 of them.
        hrs = patch_info["n_seeds"] * (M.N_STEPS + 1) / 4.6 / 3600
        print(f"[gw-infer] patch: {patch_info['n_seeds']:,} seeds x "
              f"{M.N_STEPS + 1} positions -> roughly {hrs:.1f} h at the rate "
              f"the full run measured")
        if patch_info["trained_on"]:
            print(f"[gw-infer] *** {patch_info['n_gt_in_train']} of "
                  f"{patch_info['n_gt']} GT bases here are TRAINING labels "
                  f"({patch_info['frac_gt_in_train']*100:.0f}%) — the score "
                  f"below is an upper bound, not a held-out estimate ***")
    else:
        seeds = sd["seeds"]
        gt = bases[~lab["in_margin"]]
    if len(seeds) == 0:
        raise SystemExit(f"[gw-infer] region={region} selected no seeds")
    # a patch run must not overwrite the full run it is compared against
    tag = "_patch" if region == "patch" else ""
    print(f"[gw-infer] region={region}: {len(seeds):,} seeds | {len(gt)} GT bases")

    enc = M.load_encoder()
    head, head_r = load_tree_head(dev) if USE_HEAD else (None, None)
    extra = {}
    # One walk feeds every candidate pipeline below. The head reads the endpoint
    # sphere the walk already encoded; the trajectory is what the walk already
    # computed and used to discard.
    ends, scores = M.detect(dec, enc, scene, dem, seeds.astype(np.float32),
                            label="infer", head=head, head_r=head_r,
                            traj=AGG, out=extra)

    # (name, xyz, score, needs_nms) — the FIRST is what this run reports as its
    # detections; the rest are scored identically and printed beside it.
    cands = [("readout", ends, scores, True)]
    if head is not None:
        cands.append(("treehead", ends, extra["head"], True))
    agg_info = None
    if AGG:
        import aggregate as A
        a_xyz, a_conf, agg_info = A.aggregate_walks(
            extra["traj"], extra["traj_score"], gt_xy=gt[:, :2], chm=A.load_chm())
        # clustering already suppressed; NMS again would merge real neighbours
        cands.insert(0, ("aggregate", a_xyz, a_conf, False))
    elif head is not None:
        cands.insert(0, cands.pop(1))                 # the head leads if asked

    # Which detections are even judgeable: the RTK survey does not cover the
    # whole scanned scene, and a real tree nobody walked to is not a model
    # error. Scored both ways so the unmasked number stays comparable to
    # BaseWalker's.
    inside = M.surveyed_mask(gt[:, :2], cell=SURVEY_CELL, dilate=SURVEY_DILATE)

    # Re-tune the cut HERE: the checkpoint's threshold was chosen at eval-seed
    # density, and the number of detections scales with the seed count, so it
    # does not transfer to a full run.
    scored = {nm: sweep_scorer(xyz, sc, gt, inside, nms=nms)
              for nm, xyz, sc, nms in cands}
    scorer, pxyz, psc, _ = cands[0]
    keep, in_survey, sweep, best, best_s = scored[scorer]
    kept_xyz, kept_s = pxyz[keep], psc[keep]
    print(f"[gw-infer] [{scorer}] {len(kept_xyz)} candidates after suppression "
          f"(scores {kept_s.min():.3f}-{kept_s.max():.3f})")
    print(f"[gw-infer] inside the surveyed region: {in_survey.sum()} of "
          f"{len(kept_xyz)} ({in_survey.mean()*100:.0f}%)")
    print(f"[gw-infer] best F1@0.5m over the sweep: {best['all']['f1']:.3f} "
          f"at thresh {best['thresh']:.3f} (checkpoint said {thresh:.3f})")
    print(f"[gw-infer]   inside the surveyed region: "
          f"{best_s['surveyed']['f1']:.3f} at thresh {best_s['thresh']:.3f}")

    comparison = None
    if len(cands) > 1:
        comparison = {nm: dict(best=scored[nm][3], best_surveyed=scored[nm][4])
                      for nm, *_ in cands[1:]}
        print(f"\n[gw-infer] ===== one walk, {len(cands)} pipelines "
              f"(best F1@0.5m) =====")
        for nm, *_ in cands:
            b, bs = scored[nm][3], scored[nm][4]
            print(f"  {nm:10s} F1 {b['all']['f1']:.3f} "
                  f"(P {b['all']['precision']:.3f} R {b['all']['recall']:.3f}) "
                  f"| {b['n']} det @ thr {b['thresh']:.3f} "
                  f"| surveyed {bs['surveyed']['f1']:.3f}")
        base = scored["readout"][3]["all"]["f1"]
        d = best["all"]["f1"] - base
        print(f"  {scorer} vs the incumbent read-out: {d:+.3f} F1 on the full "
              f"{region} region\n")

    if os.environ.get("GW_USE_BEST_THRESH", "1") != "0":
        thresh = best["thresh"]
    sel = kept_s >= thresh
    det_xyz, det_s = kept_xyz[sel], kept_s[sel]
    print(f"[gw-infer] {len(det_xyz)} detections at thresh {thresh:.3f}")

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
    cols = [kept_xyz, kept_s, in_survey.astype(np.float32)]
    hdr = "x,y,z,score,in_surveyed"
    if scorer == "aggregate":
        cl = agg_info["clusters"]
        cols += [cl["n_votes"][keep], cl["spread"][keep]]
        hdr += ",n_votes,spread"
    else:
        for nm, _xyz, sc, _n in cands[1:]:
            if len(sc) == len(ends):              # per-endpoint, same rows
                cols.append(sc[keep])
                hdr += f",score_{nm}"
    np.savetxt(C.RESULTS / f"geowalker{tag}_endpoints.csv", np.column_stack(cols),
               delimiter=",", header=hdr, comments="")
    np.savetxt(C.RESULTS / f"geowalker{tag}_detections.csv",
               np.column_stack([det_xyz, det_s]), delimiter=",",
               header="x,y,z,score", comments="")
    (C.RESULTS / f"geowalker{tag}_metrics.json").write_text(json.dumps(
        dict(region=region, scorer=scorer, thresh=thresh,
             n_det=int(len(det_xyz)), n_gt=int(len(gt)),
             n_seeds=int(len(seeds)), n_endpoints=int(len(kept_xyz)),
             metrics=metrics, threshold_sweep=sweep, comparison=comparison,
             aggregate=agg_info and {k: v for k, v in agg_info.items()
                                     if k != "clusters"},
             surveyed=dict(cell=SURVEY_CELL, dilate=SURVEY_DILATE,
                           frac_endpoints_inside=float(in_survey.mean())),
             patch=patch_info,
             checkpoint=dict(it=ck.get("it"), val=ck.get("val"))), indent=2))

    if region == "patch":
        patch_tables(gt, det_xyz, det_s, patch_attrs, tag)

    detections_figure(scene, seeds, gt, det_xyz, metrics["@0.5m"]["f1"],
                      f"{region} | {scorer}",
                      C.RESULTS / f"21_geowalker{tag}_detections.png",
                      inside_mask=inside(det_xyz[:, :2]),
                      f1_surveyed=metrics["@0.5m_surveyed"]["f1"])
    field_npz = GW / "field.npz"
    if field_npz.exists():
        walks_figure(dec, enc, scene, dem, field_npz, seeds, gt,
                     C.RESULTS / f"22_geowalker{tag}_walks.png")
    else:
        print(f"[gw-infer] {field_npz} missing — skipping the walks figure")


if __name__ == "__main__":
    main()
