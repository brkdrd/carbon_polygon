#!/usr/bin/env python3
"""Train the dense stem head on the frozen Sonata encoder.

One iteration = one window. The encoder runs under no_grad (it is frozen, and
its activations are by far the largest thing on the card), the head trains on
the per-point features, and the loss is two terms:

  * BCE on "this point belongs to a stem", pos_weight set from the actual
    class balance of the window so the head is not simply taught to say no;
  * Smooth-L1 on the horizontal offset to that stem's base, on positives only
    and scaled by SN_NEAR_R, because an offset target is meaningless for a
    point that belongs to no stem.

Validation is the thing that matters and the thing the walker's training loop
could not afford: the full detection metric — vote, cluster, match against the
val bases — on a fixed sample of val windows, every SN_EVAL iterations. The
best-F1 head is what gets checkpointed, so a run that overfits cannot damage
what inference reads.

Resumes by default from data/models/stemnet.pth with its AdamW state
and best F1; SN_RESUME=0 starts cold. SN_ITERS counts THIS run.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402
import net as S  # noqa: E402

ITERS = int(os.environ.get("SN_ITERS", "4000"))
LR = float(os.environ.get("SN_LR", "1e-3"))
WD = float(os.environ.get("SN_WD", "1e-4"))
W_OFF = float(os.environ.get("SN_W_OFF", "1.0"))
EVAL_EVERY = int(os.environ.get("SN_EVAL", "250"))
# How many val windows the training-time metric is computed over. This is what
# BEST-F1 CHECKPOINTING IS SELECTED ON, so it has to be wide enough that the
# selection is not picking noise: 12 windows held 85 val bases, which puts the
# standard error on recall near 0.05 — wider than the entire spread of a run's
# eval scores, so `best` was the luckiest draw rather than the best head.
EVAL_WINDOWS = int(os.environ.get("SN_EVAL_WIN", "48"))
MATCH_R = float(os.environ.get("SN_MATCH_R", "0.5"))
RESUME = os.environ.get("SN_RESUME", "1") != "0"
RESUME_LR = float(os.environ.get("SN_RESUME_LR", str(LR)))
SEED = int(os.environ.get("SN_SEED", "0"))
POS_W_CAP = float(os.environ.get("SN_POS_W_CAP", "20.0"))
PROGRESS_S = float(os.environ.get("SN_PROGRESS_S", "30"))


def save(path, head, opt, it, val, feat_dim):
    torch.save(dict(state_dict=head.state_dict(), opt=opt.state_dict(), it=it,
                    val=val, feat_dim=feat_dim,
                    cfg=dict(band_lo=S.BAND_LO, band_hi=S.BAND_HI,
                             near_r=S.NEAR_R, voxel=S.VOXEL, win=S.WIN,
                             pad=S.PAD, detrend=S.DETREND)), path)


def window_batch(band, centres, rng, cap=None):
    """Pick one window and return its crop positions + inner mask."""
    cx, cy = centres[rng.integers(len(centres))]
    pos, inner = band.window(cx, cy)
    cap = cap or S.PTS_CAP
    if len(pos) > cap:
        # subsample the CROP, keeping the inner/outer proportion — a window
        # denser than the cap is a window the encoder cannot hold whole
        sel = rng.permutation(len(pos))[:cap]
        pos, inner = pos[sel], inner[sel]
    return pos, inner, (cx, cy)


def forward_window(enc, head, band, pos, inner, bases_xy, train=True):
    """Encode one window, predict on its inner square, and return the losses."""
    idx, zcol = band.crop(pos)
    feat = S.encode_points(enc, band.chunk, idx, zcol)
    m = torch.as_tensor(inner, device=band.device)
    # The pad ring is CONTEXT: the encoder has to see it so a trunk at a window
    # edge still sits inside somebody's neighbourhood, but nothing is ever
    # predicted there. Dropping it BEFORE the head rather than after keeps 36%
    # of a 20 m crop out of the head entirely ((20^2 - 16^2) / 20^2), and those
    # are exactly the activations autograd would hold until backward.
    feat = feat[m]
    xy = band.chunk.xyz[idx][m][:, :2]
    y, off_t = S.targets(xy, bases_xy)

    with torch.set_grad_enabled(train):
        logit, off_p = head(feat)

    n_pos = int(y.sum())
    pw = torch.tensor(
        min(POS_W_CAP, max(1.0, (len(y) - n_pos) / max(n_pos, 1))),
        device=logit.device)
    l_cls = F.binary_cross_entropy_with_logits(logit, y.float(), pos_weight=pw)
    if n_pos:
        l_off = F.smooth_l1_loss(off_p[y] / S.NEAR_R, off_t[y] / S.NEAR_R)
    else:
        l_off = logit.sum() * 0.0
    return l_cls, l_off, n_pos, len(y), (xy, logit, off_p)


OOM = getattr(torch, "OutOfMemoryError",
              getattr(torch.cuda, "OutOfMemoryError", None))


def is_oom(e):
    """An exhausted card and nothing else.

    A bare RuntimeError must NOT be swallowed as one, or a real bug gets
    retried, skipped and counted as a memory problem. In particular an
    `illegal memory access` is a RuntimeError and is not recoverable — the CUDA
    context is dead — so it has to propagate.
    """
    if OOM is not None and isinstance(e, OOM):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def free_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def oom_retry(fn, pos, inner, rng, what=""):
    """Run `fn(pos, inner)`, and on an exhausted card halve the crop and retry.

    A 4,000-iteration run must not die because one window in a thicket is
    twice as dense as the rest. Subsampling a crop is what SN_PTS already does,
    so a retry at half of it is the same operation and not a different
    experiment — but it is logged, and counted, so a run full of them is never
    read as a clean one.
    """
    try:
        return fn(pos, inner)
    except RuntimeError as e:
        if not is_oom(e):
            raise
        free_cache()
        keep = max(64, len(pos) // 2)
        sel = rng.permutation(len(pos))[:keep]
        print(f"[sn] card exhausted on a {len(pos):,}-point crop{what} — "
              f"retrying at {keep:,}; lower SN_PTS (now {S.PTS_CAP:,}) or "
              f"raise SN_VOXEL to stop this happening", flush=True)
        try:
            return fn(pos[sel], inner[sel])
        except RuntimeError as e2:
            if not is_oom(e2):
                raise
            free_cache()
            print(f"[sn] still exhausted at {keep:,} — window skipped",
                  flush=True)
            return None


@torch.no_grad()
def evaluate(enc, head, band, centres, val_bases, match_r=MATCH_R):
    """The real metric: vote, cluster, match. Pooled over the eval windows so
    one sparse window cannot dominate the number."""
    head.eval()
    xs, ps, os_ = [], [], []
    for cx, cy in centres:
        pos, inner = band.window(cx, cy)
        if len(pos) < 64 or not inner.any():
            continue
        if len(pos) > S.PTS_CAP:
            sel = np.random.default_rng(0).permutation(len(pos))[:S.PTS_CAP]
            pos, inner = pos[sel], inner[sel]
        idx, zcol = band.crop(pos)
        feat = S.encode_points(enc, band.chunk, idx, zcol)
        m = torch.as_tensor(inner, device=band.device)
        logit, off = head(feat[m])
        xs.append(band.chunk.xyz[idx][m][:, :2].cpu().numpy())
        ps.append(torch.sigmoid(logit).cpu().numpy())
        os_.append(off.cpu().numpy())
    head.train()
    if not xs:
        return dict(f1=-1.0, precision=0.0, recall=0.0, n_det=0, thresh=0.0)

    xy = np.concatenate(xs); p = np.concatenate(ps); off = np.concatenate(os_)
    det, mass, _n = S.vote(xy, p, off)
    if not len(det):
        return dict(f1=0.0, precision=0.0, recall=0.0, n_det=0, thresh=0.0)
    # which val bases are judgeable from these windows at all
    keep = np.zeros(len(val_bases), bool)
    for cx, cy in centres:
        keep |= ((np.abs(val_bases[:, 0] - cx) <= S.WIN / 2)
                 & (np.abs(val_bases[:, 1] - cy) <= S.WIN / 2))
    gt = val_bases[keep][:, :2]
    if not len(gt):
        return dict(f1=-1.0, precision=0.0, recall=0.0, n_det=0, thresh=0.0)
    best = dict(f1=-1.0, thresh=0.0)
    for q in np.linspace(0, 90, 10):
        t = float(np.percentile(mass, q))
        k = mass >= t
        if k.sum() < 1:
            continue
        m = S.match_metrics(det[k], mass[k], gt, match_r)
        if m["f1"] > best["f1"]:
            best = dict(m, thresh=t, n_det=int(k.sum()), n_gt=int(len(gt)))
    return best


def main():
    dev = S.DEVICE
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    lab = np.load(S.PREP / "labels.npz")
    bases, is_val, in_margin = lab["bases"], lab["is_val"], lab["in_margin"]
    xq = S.split_x(lab)
    tr_bases = bases[~is_val & ~in_margin]
    va_bases = bases[is_val & ~in_margin]
    print(f"[sn] split x={xq:.1f} | train bases {len(tr_bases)} | "
          f"val bases {len(va_bases)}")

    dem = S.Dem(S.PREP / "dem.npz", device=dev)
    # the terrain, before anything is trained on it: how much ground elevation
    # sits inside one crop, and whether the DEM is really the ground
    S.relief()
    S.dem_check(dem, device=dev)
    # streamed tile by tile: only the band is ever uploaded
    band = S.Band(S.PREP / "tiles", dem, device=dev)
    enc = S.load_encoder(device=dev)
    S.gpu_report(" after setup")

    # training windows: centred on train bases, jittered, and never allowed to
    # reach across the margin into the val block
    lim = xq - S.VAL_MARGIN / 2 - S.WIN / 2 - S.PAD
    jit = rng.uniform(-S.WIN / 3, S.WIN / 3, size=(len(tr_bases), 2))
    tr_centres = tr_bases[:, :2] + jit
    tr_centres = tr_centres[tr_centres[:, 0] < lim]
    if not len(tr_centres):
        raise SystemExit("[sn] no train windows fit left of the margin")
    va_lo = xq + S.VAL_MARGIN / 2 + S.WIN / 2
    va_c = va_bases[va_bases[:, 0] > va_lo][:, :2]
    if not len(va_c):
        raise SystemExit("[sn] no val windows fit right of the margin")
    ev_centres = va_c[np.linspace(0, len(va_c) - 1,
                                  min(EVAL_WINDOWS, len(va_c))).astype(int)]
    print(f"[sn] {len(tr_centres)} train window centres (x < {lim:.1f}) | "
          f"{len(ev_centres)} eval windows")

    tr_xy = torch.from_numpy(tr_bases[:, :2].astype(np.float32)).to(dev)

    # the head's width is whatever the encoder's dense read-out turns out to be
    pos, inner, _ = window_batch(band, tr_centres, rng)
    idx, zcol = band.crop(pos)
    with torch.no_grad():
        feat_dim = int(S.encode_points(enc, band.chunk, idx, zcol).shape[1])
    print(f"[sn] dense feature dim {feat_dim}")
    del idx, zcol

    head = S.StemHead(feat_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    it0, best_f1 = 0, -1.0
    if RESUME and S.CKPT.exists():
        ck = torch.load(S.CKPT, map_location=dev, weights_only=False)
        if int(ck.get("feat_dim", feat_dim)) != feat_dim:
            raise SystemExit(
                f"[sn] {S.CKPT} was trained on {ck['feat_dim']}-d features, "
                f"this encoder gives {feat_dim} — delete it or fix the read-out")
        head.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        it0 = int(ck.get("it", 0))
        best_f1 = float(ck.get("val", {}).get("f1", -1.0))
        for g in opt.param_groups:
            g["lr"] = RESUME_LR
        print(f"[sn] resuming {S.CKPT} @ it {it0} (val F1 {best_f1:.3f}) | "
              f"+{ITERS} iters at peak LR {RESUME_LR:g}")
    elif RESUME:
        print(f"[sn] no checkpoint at {S.CKPT} — training from scratch")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)

    def step(pos, inner):
        l_cls, l_off, n_pos, n_pts, _ = forward_window(
            enc, head, band, pos, inner, tr_xy, train=True)
        loss = l_cls + W_OFF * l_off
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        return float(l_cls), float(l_off), n_pos, n_pts

    head.train()
    t0, last, run = time.time(), time.time(), {}
    n_oom, n_big, crop_max = 0, 0, 0
    for it in range(it0 + 1, it0 + ITERS + 1):
        pos, inner, _c = window_batch(band, tr_centres, rng)
        if len(pos) < 64 or not inner.any():
            continue
        crop_max = max(crop_max, len(pos))
        got = oom_retry(step, pos, inner, rng, f" at it {it}")
        sched.step()
        if got is None:
            n_oom += 1
            continue
        c_cls, c_off, n_pos, n_pts = got

        for k, v in (("cls", c_cls), ("off", c_off),
                     ("pos", n_pos / max(n_pts, 1)), ("pts", n_pts),
                     ("crop", len(pos))):
            run[k] = 0.9 * run.get(k, v) + 0.1 * v
        if PROGRESS_S and time.time() - last > PROGRESS_S:
            last = time.time()
            done = it - it0
            eta = (time.time() - t0) / max(done, 1) * (ITERS - done)
            peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                    if str(dev).startswith("cuda") and torch.cuda.is_available()
                    else 0.0)
            print(f"[sn] it {it}/{it0+ITERS} cls {run['cls']:.4f} "
                  f"off {run['off']:.4f} pos {run['pos']*100:.1f}% "
                  f"pts {run['pts']:,.0f} crop {run['crop']:,.0f}/{crop_max:,} "
                  f"gpu {peak:.1f}G lr {sched.get_last_lr()[0]:.2e} "
                  f"eta {eta/60:.0f}m", flush=True)

        if it % EVAL_EVERY == 0 or it == it0 + ITERS:
            v = evaluate(enc, head, band, ev_centres, va_bases)
            print(f"[sn] EVAL it {it}: F1@{MATCH_R} {v['f1']:.3f} "
                  f"(P {v.get('precision',0):.3f} R {v.get('recall',0):.3f}) "
                  f"{v.get('n_det',0)} det / {v.get('n_gt',0)} GT", flush=True)
            save(S.LAST_CKPT, head, opt, it, v, feat_dim)
            if v["f1"] > best_f1:
                best_f1 = v["f1"]
                save(S.CKPT, head, opt, it, v, feat_dim)
                print(f"[sn]   new best -> {S.CKPT}")

    S.gpu_report(" high-water")
    print(f"[sn] done in {(time.time()-t0)/60:.1f} min | best val F1 "
          f"{best_f1:.3f} -> {S.CKPT}"
          + (f" | {n_oom} window(s) SKIPPED for want of memory — the card is "
             f"the limit, lower SN_PTS" if n_oom else ""))


if __name__ == "__main__":
    main()
