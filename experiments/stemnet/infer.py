#!/usr/bin/env python3
"""Dense stem inference: one encoder pass per window, every point votes.

Tiles the region in SN_WIN-metre squares (each encoded with a context ring
it does not predict on, so the inner squares tile exactly and no point is
counted twice), runs the frozen encoder and the trained head, and turns the
p-weighted cloud of SHIFTED points into detections.

The comparison this exists to make is printed at the end: hectares per hour,
beside the walker's 1.35. The walker spent 9 Sonata encodes per 2 m seed; this
spends one per window over 5-10% of the points, and the vote count per detection
goes from a median of 1 to the hundreds.

SN_REGION picks the ground: patch (the attributed survey patch, with a
per-tree answer sheet and recall by stem size), val (the held-out block — the
only honest score), or all.

Outputs (TAG is _patch / _val / _all):
  /results/stemnetTAG_detections.csv   x,y,z,score,n_votes
  /results/stemnetTAG_metrics.json     P/R/F1/RMSE at 0.5 m and 1.0 m
  /results/stemnetTAG_trees.csv        per surveyed tree (patch only)
  /results/30_stemnetTAG.png           detections, and the vote cloud
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402
import net as S  # noqa: E402
from common import patch as P  # noqa: E402
from common.patch import girth_to_dbh  # noqa: E402

REGION = os.environ.get("SN_REGION", "val")
PROGRESS_S = float(os.environ.get("SN_PROGRESS_S", "30"))


def run_region(enc, head, chunk, band, centres):
    """Every window's inner points, with p and predicted offset."""
    xs, ps, os_, t0, last = [], [], [], time.time(), time.time()
    n_enc = 0
    for k, (cx, cy) in enumerate(centres):
        pos, inner = band.window(cx, cy)
        if len(pos) < 64 or not inner.any():
            continue
        if len(pos) > S.PTS_CAP:
            # subsample, never truncate: `pos` is in bucket order, so a slice
            # would drop one spatial corner of the window and every tree in it
            print(f"[sn] window {k} has {len(pos):,} pts > SN_PTS "
                  f"({S.PTS_CAP:,}) — subsampling; raise SN_VOXEL to keep "
                  f"the whole window", flush=True)
            sel = np.random.default_rng(k).permutation(len(pos))[:S.PTS_CAP]
            pos, inner = pos[sel], inner[sel]
        idx, zcol = band.crop(pos)
        with torch.no_grad():
            feat = S.encode_points(enc, chunk, idx, zcol)
            logit, off = head(feat)
        m = torch.as_tensor(inner, device=chunk.device)
        xs.append(chunk.xyz[idx][m][:, :2].cpu().numpy())
        ps.append(torch.sigmoid(logit[m]).cpu().numpy())
        os_.append(off[m].cpu().numpy())
        n_enc += 1
        if PROGRESS_S and time.time() - last > PROGRESS_S:
            last = time.time()
            eta = (time.time() - t0) / (k + 1) * (len(centres) - k - 1)
            print(f"[sn] window {k+1}/{len(centres)} | "
                  f"{sum(len(a) for a in xs):,} pts | eta {eta/60:.0f}m",
                  flush=True)
    if not xs:
        raise SystemExit("[sn] no window held any band points")
    return (np.concatenate(xs), np.concatenate(ps), np.concatenate(os_),
            n_enc, time.time() - t0)


def figure(xy, p, off, det, gt, out, title):
    fig, ax = plt.subplots(1, 2, figsize=(17, 8))
    k = p >= S.P_THRESH
    ax[0].scatter(xy[~k, 0], xy[~k, 1], s=0.3, c="0.85", lw=0,
                  label=f"band, p<{S.P_THRESH}")
    ax[0].scatter(xy[k, 0], xy[k, 1], s=0.6, c=p[k], cmap="viridis", lw=0,
                  label="stem points")
    if len(gt):
        ax[0].scatter(gt[:, 0], gt[:, 1], s=44, facecolors="none",
                      edgecolors="tab:blue", lw=1.0, label="RTK base")
    if len(det):
        ax[0].scatter(det[:, 0], det[:, 1], s=16, c="tab:red", marker="x",
                      label="detection")
    ax[0].set_title(title)
    v = xy[k] + off[k]
    ax[1].scatter(v[:, 0], v[:, 1], s=0.5, c="tab:orange", lw=0, alpha=0.4,
                  label="shifted point (a vote)")
    if len(gt):
        ax[1].scatter(gt[:, 0], gt[:, 1], s=44, facecolors="none",
                      edgecolors="tab:blue", lw=1.0, label="RTK base")
    ax[1].set_title("votes: every stem point shifted onto the base it chose")
    for a in ax:
        a.set_aspect("equal")
        a.legend(loc="upper right", fontsize=8, markerscale=3)
        a.set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[sn] figure -> {out}")


def per_tree(gt, det, mass, nv, attrs, tag, radius=0.5):
    """One row per GT base, same columns as geowalker_patch_trees.csv so the
    two can be diffed line for line."""
    jn, _ok = P.join_species(gt[:, :2], attrs=attrs)
    found = np.zeros(len(gt), np.int8)
    err = np.full(len(gt), np.nan)
    score = np.full(len(gt), np.nan)
    if len(det):
        pairs, _fp, _fn = S.match_pairs(det, mass, gt[:, :2], radius)
        for i, j in pairs:
            found[j] = 1
            err[j] = float(np.linalg.norm(det[i] - gt[j, :2]))
            score[j] = float(mass[i])
    lines = ["survey_n,species,girth_cm,x,y,z,found,err_m,score"]
    for j in range(len(gt)):
        lines.append(",".join([
            str(jn["n"][j]), str(jn["species"][j]).replace(",", " "),
            str(jn["girth_cm"][j]), f"{gt[j,0]:.3f}", f"{gt[j,1]:.3f}",
            f"{gt[j,2]:.3f}", str(int(found[j])),
            "" if not np.isfinite(err[j]) else f"{err[j]:.3f}",
            "" if not np.isfinite(score[j]) else f"{score[j]:.4f}"]))
    p = C.RESULTS / f"stemnet{tag}_trees.csv"
    p.write_text("\n".join(lines) + "\n")
    print(f"[sn] -> results/{p.name}")

    dbh = np.array([girth_to_dbh(v) for v in jn["girth_cm"]])
    known = np.isfinite(dbh)
    print(f"\n[sn] ===== recall by stem size (match {radius} m) =====")
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        sel = known & (dbh >= lo) & (dbh < hi)
        if sel.sum() < 3:
            continue
        band = f"{lo:>3.0f}-{'inf' if hi > 1e8 else f'{hi:.0f}':>3} cm"
        print(f"   {band}  n={int(sel.sum()):4d}  recall {found[sel].mean():.1%}")
    if (~known).any():
        print(f"   unmeasured  n={int((~known).sum()):4d}  "
              f"recall {found[~known].mean():.1%}")
    print(f"   ALL         n={len(gt):4d}  recall {found.mean():.1%}")
    ba = np.where(known, (dbh / 200.0) ** 2 * np.pi, 0.0)
    if ba.sum() > 0:
        print(f"[sn] basal-area-weighted recall "
              f"{(ba*found).sum()/ba.sum():.1%} of {ba.sum():.2f} m2")
    return dict(recall_all=float(found.mean()),
                recall_ge10cm=float(found[known & (dbh >= 10)].mean())
                if (known & (dbh >= 10)).any() else None,
                recall_basal_area=float((ba * found).sum() / ba.sum())
                if ba.sum() > 0 else None)


def main():
    dev = S.DEVICE
    if not S.CKPT.exists():
        raise SystemExit(f"[sn] no checkpoint at {S.CKPT} — run "
                         f"stem_train first")

    lab = np.load(S.PREP / "labels.npz")
    bases, is_val, in_margin = lab["bases"], lab["is_val"], lab["in_margin"]
    xq = S.split_x(lab)

    attrs = inside = info = None
    if REGION == "patch":
        attrs = P.load_attributes()
        sp = S.PREP / "seeds.npz"
        seeds = np.load(sp)["seeds"] if sp.exists() else np.zeros((0, 3))
        _s, gt, _rows, info = P.select(seeds, bases, is_val, in_margin,
                                       attrs=attrs)
        inside = P.footprint(P.to_local(attrs["lat"], attrs["lon"]))
        tag = "_patch"
    elif REGION == "val":
        gt = bases[is_val & ~in_margin]
        lo = xq + S.VAL_MARGIN / 2
        inside = lambda xy: xy[:, 0] > lo          # noqa: E731
        tag = "_val"
    else:
        gt = bases[~in_margin]
        tag = "_all"
    if not len(gt):
        raise SystemExit(f"[sn] region={REGION} selected no GT")

    pad = 3.0
    bbox = (gt[:, 0].min() - pad, gt[:, 0].max() + pad,
            gt[:, 1].min() - pad, gt[:, 1].max() + pad)
    centres = S.tile_centres(*bbox)
    area_ha = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) / 1e4
    print(f"[sn] region={REGION}: {len(gt)} GT bases | "
          f"{bbox[1]-bbox[0]:.0f} x {bbox[3]-bbox[2]:.0f} m | "
          f"{len(centres)} windows of {S.WIN} m"
          + (f" | {info['area_ha']:.2f} ha hull" if info else ""))
    if info and info.get("trained_on"):
        print(f"[sn] !! THE PATCH IS TRAINING GROUND: "
              f"{info['frac_gt_in_train']*100:.0f}% of its GT was trained on. "
              f"This is a ceiling, not a score — read region=val for that.")

    chunk = S.load_chunk(device=dev)
    dem = S.Dem(S.PREP / "dem.npz", device=dev)
    terrain = dict(relief=S.relief(), dem_vs_survey=S.dem_check(dem, device=dev))
    band = S.Band(chunk, dem)
    enc = S.load_encoder(device=dev)

    ck = torch.load(S.CKPT, map_location=dev, weights_only=False)
    head = S.StemHead(int(ck["feat_dim"])).to(dev)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    print(f"[sn] head from {S.CKPT} @ it {ck.get('it','?')} "
          f"(train-time val F1 {ck.get('val',{}).get('f1',float('nan')):.3f})")
    trained_detrend = ck.get("cfg", {}).get("detrend")
    if trained_detrend is not None and bool(trained_detrend) != S.DETREND:
        raise SystemExit(
            f"[sn] this head was trained with detrend={trained_detrend} and "
            f"inference has SN_DETREND={int(S.DETREND)} — it would be fed a "
            f"different z convention than it learned. Set SN_DETREND="
            f"{int(bool(trained_detrend))} or retrain.")

    xy, p, off, n_enc, secs = run_region(enc, head, chunk, band, centres)
    print(f"[sn] {len(xy):,} inner band points | "
          f"{int((p>=S.P_THRESH).sum()):,} above p={S.P_THRESH} "
          f"({(p>=S.P_THRESH).mean()*100:.1f}%)")

    det, mass, nv = S.vote(xy, p, off)
    if len(det) and inside is not None:
        k = inside(det)
        print(f"[sn] {len(det)} clusters | {int(k.sum())} inside the "
              f"{REGION} footprint")
        det, mass, nv = det[k], mass[k], nv[k]
    if not len(det):
        raise SystemExit("[sn] no detections — lower SN_VOTE_MIN or "
                         "SN_P, or the head has not learned yet")
    print(f"[sn] votes per detection: p50 {np.percentile(nv,50):.0f} "
          f"p90 {np.percentile(nv,90):.0f} max {nv.max()}")

    out = dict(region=REGION, n_det=int(len(det)), n_gt=int(len(gt)),
               n_windows=int(n_enc), n_band_points=int(len(xy)),
               p_thresh=S.P_THRESH, votes_p50=float(np.percentile(nv, 50)),
               votes_max=int(nv.max()), bbox_area_ha=float(area_ha),
               seconds=float(secs), ha_per_hour=float(area_ha / (secs / 3600.0)),
               checkpoint=dict(it=int(ck.get("it", 0)),
                               val=ck.get("val", {}), feat_dim=int(ck["feat_dim"])),
               cfg=dict(band_lo=S.BAND_LO, band_hi=S.BAND_HI, near_r=S.NEAR_R,
                        win=S.WIN, pad=S.PAD, voxel=S.VOXEL, detrend=S.DETREND,
                        vote_cell=S.VOTE_CELL, vote_sep=S.VOTE_SEP,
                        vote_r=S.VOTE_R, vote_min=S.VOTE_MIN),
               terrain=terrain,
               patch=info, metrics={})
    for r in (0.5, 1.0):
        m = S.match_metrics(det, mass, gt[:, :2], r)
        out["metrics"][f"@{r}m"] = m
        print(f"[sn] @{r} m  tp {m['tp']:4d} fp {m['fp']:4d} fn {m['fn']:4d}"
              f"  P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}"
              f"  rmse {m['rmse']:.2f} m")
    sweep, best = [], dict(f1=-1.0)
    for q in np.linspace(0, 95, 20):
        t = float(np.percentile(mass, q))
        k = mass >= t
        if k.sum() < 1:
            continue
        m = S.match_metrics(det[k], mass[k], gt[:, :2], 0.5)
        sweep.append(dict(thresh=t, n=int(k.sum()), **m))
        if m["f1"] > best["f1"]:
            best = sweep[-1]
    out["threshold_sweep"] = sweep
    out["best_f1@0.5m"] = best
    if best["f1"] > 0:
        print(f"[sn] best F1@0.5 m {best['f1']:.3f} keeping {best['n']} "
              f"(mass >= {best['thresh']:.1f}): P {best['precision']:.3f} "
              f"R {best['recall']:.3f}")

    np.savetxt(C.RESULTS / f"stemnet{tag}_detections.csv",
               np.column_stack([det, dem.height(torch.from_numpy(
                   det.astype(np.float32)).to(dev)).cpu().numpy(), mass, nv]),
               delimiter=",", header="x,y,z,score,n_votes", comments="")
    if REGION == "patch":
        out["per_tree"] = per_tree(gt, det, mass, nv, attrs, tag)
    figure(xy, p, off, det, gt[:, :2],
           C.RESULTS / f"23_stemnet{tag}.png",
           f"dense stem head — {REGION}: {len(det)} detections, "
           f"{len(gt)} RTK bases")
    (C.RESULTS / f"stemnet{tag}_metrics.json").write_text(
        json.dumps(out, indent=2))
    print(f"\n[sn] {area_ha:.2f} ha in {secs/60:.1f} min = "
          f"{out['ha_per_hour']:.1f} ha/h "
          f"({1/max(out['ha_per_hour'],1e-9):.2f} h/ha; the walker: 1.35 h/ha)")
    print(f"[sn] -> results/stemnet{tag}_metrics.json")


if __name__ == "__main__":
    main()
