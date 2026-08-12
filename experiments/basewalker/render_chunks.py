#!/usr/bin/env python3
"""BaseWalker stage 3 — qualitative chunk renders (detections vs ground truth).

Takes the trained walker decoder and draws what it actually found on four square
chunks cut out of the full campus scene, each panel showing the cloud, the RTK
tree bases and the detections, colour-coded TP / FP / missed. Chunks are picked
deterministically: the densest non-overlapping windows, balanced across the
spatial split (two from the held-out VAL block, two from the TRAIN block) so the
pictures separate "what it memorised" from "what it generalises to".

Per chunk, walkers are seeded on the stored seed grid over the window plus a
small pad (a base near the edge is reached from outside), then endpoints are
NMS'd, thresholded, and clipped back to the window before matching against the
GT bases inside that same window.

Outputs:
  /results/11_basewalker_chunks.png          2x2 overview of the four chunks
  /results/11_basewalker_chunk{1..4}.png     one figure per chunk
  /results/basewalker_chunk_metrics.json     per-chunk P/R/F1/RMSE @0.5 m / 1 m
  /results/basewalker_chunk_detections.csv   chunk,x,y,z,score,status
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless container
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/app")
from common import config as C  # noqa: E402
import walker as W  # noqa: E402

BW = C.DATA / "basewalker"
CKPT = C.MODELS / "basewalker_decoder.pth"

CHUNK_M = float(os.environ.get("BW_CHUNK_M", "40.0"))     # chunk edge, m
N_CHUNKS = int(os.environ.get("BW_N_CHUNKS", "4"))
CHUNK_SPLIT = os.environ.get("BW_CHUNK_SPLIT", "mix")     # mix | val | train
SEED_PAD = float(os.environ.get("BW_CHUNK_PAD", "4.0"))   # seed halo, m
MIN_GT = int(os.environ.get("BW_CHUNK_MIN_GT", "5"))      # skip empty windows
VAL_MARGIN = float(os.environ.get("BW_VAL_MARGIN", "10.0"))  # same as prep
SLAB = (0.3, 3.0)          # trunk slab above the DEM, m — where stems live
PLOT_CAP = 250_000         # points drawn per panel
MATCH_R = 0.5              # radius the panels are scored/coloured at

# colours (kept distinct at small marker sizes)
CTX, STEM = "#dcdcdc", "#4d6b3f"
C_GT, C_TP, C_FP, C_FN = "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"


def in_window(pts, cx, cy, half) -> np.ndarray:
    return ((np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 1] - cy) <= half))


def side_of(cx, half, split_x) -> str:
    """Which side of the spatial train/val split a window sits on."""
    if cx - half >= split_x + VAL_MARGIN / 2:
        return "val"
    if cx + half <= split_x - VAL_MARGIN / 2:
        return "train"
    return "straddles"


def pick_chunks(bases, seeds, split_x):
    """Densest non-overlapping windows, balanced over the split. Deterministic.

    BW_CHUNKS="cx,cy;cx,cy;..." (local frame) overrides the automatic choice.
    """
    half = CHUNK_M / 2
    explicit = os.environ.get("BW_CHUNKS", "").strip()
    if explicit:
        out = []
        for part in (p for p in explicit.split(";") if p.strip()):
            cx, cy = (float(v) for v in part.split(",")[:2])
            out.append(dict(cx=cx, cy=cy, n_gt=int(in_window(bases, cx, cy, half).sum()),
                            side=side_of(cx, half, split_x)))
        print(f"[bw-render] {len(out)} chunk(s) from BW_CHUNKS")
        return out[:N_CHUNKS]

    # candidate windows on a half-overlapping grid over the SEEDED area (seeds
    # only exist where the DEM saw real ground, i.e. where the scanner covered)
    def axis(lo, hi):
        a = np.arange(lo + half, hi - half + 1e-6, CHUNK_M / 2)
        return a if len(a) else np.array([(lo + hi) / 2])

    cands = []
    for cx in axis(seeds[:, 0].min(), seeds[:, 0].max()):
        for cy in axis(seeds[:, 1].min(), seeds[:, 1].max()):
            n_gt = int(in_window(bases, cx, cy, half).sum())
            n_seed = int(in_window(seeds, cx, cy, half).sum())
            if n_gt < MIN_GT or n_seed < 20:
                continue
            cands.append(dict(cx=float(cx), cy=float(cy), n_gt=n_gt,
                              side=side_of(cx, half, split_x)))
    cands.sort(key=lambda c: (-c["n_gt"], c["cx"], c["cy"]))
    print(f"[bw-render] {len(cands)} candidate windows "
          f"({sum(c['side'] == 'val' for c in cands)} val / "
          f"{sum(c['side'] == 'train' for c in cands)} train) "
          f"with >={MIN_GT} GT bases")

    if CHUNK_SPLIT in ("val", "train"):
        want = [CHUNK_SPLIT] * N_CHUNKS
    else:                                    # mix: alternate val / train
        want = [("val", "train")[i % 2] for i in range(N_CHUNKS)]

    picked = []
    for side in want:
        free = [c for c in cands if c not in picked
                and all(abs(c["cx"] - p["cx"]) >= CHUNK_M or
                        abs(c["cy"] - p["cy"]) >= CHUNK_M for p in picked)]
        hit = next((c for c in free if c["side"] == side), None) \
            or next((c for c in free if c["side"] != "straddles"), None)
        if hit is None:
            print(f"[bw-render] no window left for side={side} — stopping at "
                  f"{len(picked)} chunk(s)")
            break
        picked.append(hit)
    return picked


def window_points(scene, dem, cx, cy, half):
    """Cloud points inside the window (capped) + their height above the DEM."""
    xyz = scene.xyz
    m = ((xyz[:, 0] >= cx - half) & (xyz[:, 0] <= cx + half) &
         (xyz[:, 1] >= cy - half) & (xyz[:, 1] <= cy + half))
    pts = xyz[m]
    if len(pts) > PLOT_CAP:
        pts = pts[torch.randperm(len(pts), device=pts.device)[:PLOT_CAP]]
    hag = pts[:, 2] - dem.height(pts[:, :2])
    return pts.cpu().numpy(), hag.cpu().numpy()


def run_chunk(model, dec, scene, dem, ch, seeds, bases, thresh):
    """Walk the chunk's seeds, NMS + threshold, match against its GT bases."""
    half = CHUNK_M / 2
    cx, cy = ch["cx"], ch["cy"]
    sub = seeds[in_window(seeds, cx, cy, half + SEED_PAD)]
    if len(sub):
        ends, scores = W.detect(model=model, dec=dec, scene=scene, dem=dem,
                                seeds_np=sub.astype(np.float32))
        keep = W.nms(ends, scores)
        keep = keep[scores[keep] >= thresh]
        keep = keep[in_window(ends[keep], cx, cy, half)]  # judge inside the window
        det_xyz, det_s = ends[keep], scores[keep]
    else:
        print(f"[bw-render] chunk @({cx:.0f},{cy:.0f}): no seeds — unscanned area")
        det_xyz, det_s = np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    gt = bases[in_window(bases, cx, cy, half)]

    metrics = {f"@{r}m": W.match_metrics(det_xyz, det_s, gt, radius=r)
               for r in (MATCH_R, 1.0)}
    pairs, fp, fn = W.match_pairs(det_xyz, det_s, gt, radius=MATCH_R)
    m = metrics[f"@{MATCH_R}m"]
    print(f"[bw-render] chunk @({cx:.0f},{cy:.0f}) {ch['side']:>9} | "
          f"{len(sub):,} seeds -> {len(det_xyz)} det / {len(gt)} GT | "
          f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
          f"rmse {m['rmse']:.2f} m", flush=True)

    pts, hag = window_points(scene, dem, cx, cy, half)
    return dict(ch, n_seeds=int(len(sub)), det_xyz=det_xyz, det_s=det_s, gt=gt,
                pairs=pairs, fp=fp, fn=fn, metrics=metrics, pts=pts, hag=hag)


def legend_handles():
    """Fixed key — proxies, so a panel without FPs still explains the colour."""
    from matplotlib.lines import Line2D

    def mk(marker, edge, face, label, ms=9, lw=1.6):
        return Line2D([], [], ls="none", marker=marker, ms=ms, mew=lw,
                      markeredgecolor=edge, markerfacecolor=face, label=label)

    return [mk(".", STEM, STEM, f"cloud {SLAB[0]}–{SLAB[1]} m above ground",
               ms=10, lw=0),
            mk("x", C_GT, C_GT, "GT tree base (RTK)"),
            mk("o", C_TP, "none", f"detection (TP, ≤{MATCH_R} m)"),
            mk("o", C_FP, "none", "detection (FP)"),
            mk("s", C_FN, "none", "missed GT (FN)", ms=11, lw=1.4)]


def draw_panel(ax, i, ch):
    half = CHUNK_M / 2
    p, hag, gt, det = ch["pts"], ch["hag"], ch["gt"], ch["det_xyz"]
    ax.scatter(p[:, 0], p[:, 1], s=0.08, c=CTX, marker=".", linewidths=0)
    slab = (hag > SLAB[0]) & (hag < SLAB[1])
    ax.scatter(p[slab, 0], p[slab, 1], s=0.5, c=STEM, marker=".", linewidths=0)
    if len(ch["pairs"]):
        a, b = ch["pairs"][:, 0], ch["pairs"][:, 1]
        ax.plot(np.stack([det[a, 0], gt[b, 0]]), np.stack([det[a, 1], gt[b, 1]]),
                c=C_TP, lw=0.9, zorder=3)
        ax.scatter(det[a, 0], det[a, 1], s=46, marker="o", facecolors="none",
                   edgecolors=C_TP, lw=1.6, zorder=5)
    if len(ch["fp"]):
        ax.scatter(det[ch["fp"], 0], det[ch["fp"], 1], s=46, marker="o",
                   facecolors="none", edgecolors=C_FP, lw=1.6, zorder=5)
    ax.scatter(gt[:, 0], gt[:, 1], s=55, marker="x", c=C_GT, lw=1.5, zorder=4)
    if len(ch["fn"]):
        ax.scatter(gt[ch["fn"], 0], gt[ch["fn"], 1], s=170, marker="s",
                   facecolors="none", edgecolors=C_FN, lw=1.4, zorder=4)
    m = ch["metrics"][f"@{MATCH_R}m"]
    ax.set_title(f"chunk {i} · {ch['side']} block · center "
                 f"({ch['cx']:.0f}, {ch['cy']:.0f}) m\n"
                 f"{len(det)} det / {len(gt)} GT · P {m['precision']:.2f} "
                 f"R {m['recall']:.2f} F1 {m['f1']:.2f} · "
                 f"rmse {m['rmse']:.2f} m @{MATCH_R} m", fontsize=10)
    ax.set_xlim(ch["cx"] - half, ch["cx"] + half)
    ax.set_ylim(ch["cy"] - half, ch["cy"] + half)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda"

    scene = W.load_scene(BW / "tiles", dev)
    dem = W.Dem(BW / "dem.npz", dev)
    lab = np.load(BW / "labels.npz")
    bases = lab["bases"][~lab["in_margin"]]      # margin bases train nothing
    split_x = float(lab["split_x"])
    seeds = np.load(BW / "seeds.npz")["seeds"]

    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    dec = W.WalkerDecoder().to(dev)
    dec.load_state_dict(ck["state_dict"])
    dec.eval()
    thresh = float(os.environ.get("BW_CONF_THRESH",
                                  str(ck.get("val", {}).get("thresh", 0.5))))
    print(f"[bw-render] decoder from it {ck.get('it')} | conf thresh {thresh} | "
          f"{CHUNK_M:.0f} m chunks | split x={split_x:.1f}")

    chunks = pick_chunks(bases, seeds, split_x)
    if not chunks:
        raise SystemExit("[bw-render] no usable chunk found — lower "
                         "BW_CHUNK_MIN_GT or set BW_CHUNKS explicitly")
    model = W.load_encoder()
    chunks = [run_chunk(model, dec, scene, dem, ch, seeds, bases, thresh)
              for ch in chunks]

    C.RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- 2x2 overview ----------------------------------------------------
    ncol = 2 if len(chunks) > 1 else 1
    nrow = int(np.ceil(len(chunks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8 * ncol, 8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, ch in enumerate(chunks):
        draw_panel(axes[k], k + 1, ch)
    for ax in axes[len(chunks):]:
        ax.axis("off")
    tp = sum(c["metrics"][f"@{MATCH_R}m"]["tp"] for c in chunks)
    n_det = sum(len(c["det_xyz"]) for c in chunks)
    n_gt = sum(len(c["gt"]) for c in chunks)
    h = legend_handles()
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"BaseWalker tree bases vs RTK ground truth — {len(chunks)} × "
                 f"{CHUNK_M:.0f} m chunks (decoder it {ck.get('it')}, "
                 f"thresh {thresh:.2f})\n"
                 f"{tp}/{n_gt} GT bases hit within {MATCH_R} m · "
                 f"{n_det} detections total", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    out = C.RESULTS / "11_basewalker_chunks.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[bw-render] overview -> {out}")

    # ---- one figure per chunk -------------------------------------------
    for k, ch in enumerate(chunks, 1):
        fig, ax = plt.subplots(figsize=(11, 11))
        draw_panel(ax, k, ch)
        ax.legend(handles=legend_handles(), loc="upper right", fontsize=9,
                  framealpha=0.9)
        p = C.RESULTS / f"11_basewalker_chunk{k}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[bw-render] chunk {k} -> {p}")

    # ---- numbers ---------------------------------------------------------
    with open(C.RESULTS / "basewalker_chunk_metrics.json", "w") as f:
        json.dump(dict(
            chunk_m=CHUNK_M, thresh=thresh, it=ck.get("it"), split_x=split_x,
            chunks=[dict(id=k, cx=c["cx"], cy=c["cy"], side=c["side"],
                         n_seeds=c["n_seeds"], n_det=int(len(c["det_xyz"])),
                         n_gt=int(len(c["gt"])), metrics=c["metrics"])
                    for k, c in enumerate(chunks, 1)]), f, indent=2)
    with open(C.RESULTS / "basewalker_chunk_detections.csv", "w") as f:
        f.write("chunk,x,y,z,score,status\n")
        for k, c in enumerate(chunks, 1):
            status = np.full(len(c["det_xyz"]), "fp", dtype=object)
            if len(c["pairs"]):
                status[c["pairs"][:, 0]] = "tp"
            for (x, y, z), s, st in zip(c["det_xyz"], c["det_s"], status):
                f.write(f"{k},{x:.3f},{y:.3f},{z:.3f},{s:.4f},{st}\n")
    print("[bw-render] done.")


if __name__ == "__main__":
    main()
