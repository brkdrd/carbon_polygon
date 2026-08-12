#!/usr/bin/env python3
"""BaseWalker training: frozen Sonata encoder + walker decoder rollout.

Seeds are drawn exactly like inference (DEM-grid seeds + xy jitter) from the
TRAIN spatial block; loss is the discounted per-step squared distance to the
nearest labeled base + per-step BCE on "base within CONF_R of current center"
(pos-weighted). Periodic detection eval (NMS + greedy matching) on the VAL
block picks the checkpoint.

Resumes by default: if a decoder checkpoint already exists it is loaded (with
its AdamW state and best val F1) and BW_ITERS more iterations are trained on
top of it. BW_ITERS is per-run, not cumulative, and defaults to 9000 — 3x the
3000 of the original from-scratch run. BW_RESUME=0 forces a cold start.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/app")
from common import config as C
import walker as W

BW = C.DATA / "basewalker"
CKPT = C.MODELS / "basewalker_decoder.pth"
# the decoder as of the last iteration, saved regardless of val F1 — a long
# fine-tune that never beats the resumed best would otherwise leave nothing.
LAST_CKPT = C.MODELS / "basewalker_decoder_last.pth"

# iterations to run in THIS invocation (added on top of a resumed checkpoint)
ITERS = int(os.environ.get("BW_ITERS", "9000"))
BATCH = int(os.environ.get("BW_BATCH", "48"))
LR = float(os.environ.get("BW_LR", "3e-4"))
CLIP = float(os.environ.get("BW_CLIP", "1.0"))
EVAL_EVERY = int(os.environ.get("BW_EVAL_EVERY", "500"))
EVAL_SEEDS = int(os.environ.get("BW_EVAL_SEEDS", "2048"))
NEAR_FRAC = float(os.environ.get("BW_NEAR_FRAC", "0.7"))
NEAR_R = float(os.environ.get("BW_NEAR_R", "8.0"))
SEED_JIT = float(os.environ.get("BW_SEED_JIT", "1.0"))
RESUME = os.environ.get("BW_RESUME", "1") != "0"
# peak LR for the resumed cosine cycle. Same as LR by default (an SGDR-style
# warm restart, which the restored AdamW moments damp); lower it if the extra
# iterations knock a converged decoder off its optimum.
RESUME_LR = float(os.environ.get("BW_RESUME_LR", str(LR)))
SEED = int(os.environ.get("BW_SEED", "0"))


def save_ckpt(path, dec, opt, it, val):
    torch.save(dict(state_dict=dec.state_dict(), opt=opt.state_dict(), it=it,
                    val=val, sphere_r=W.SPHERE_R, n_steps=W.N_STEPS,
                    conf_r=W.CONF_R), path)


def main():
    dev = "cuda"
    ck = None
    if RESUME and CKPT.exists():
        ck = torch.load(CKPT, map_location=dev, weights_only=False)
    elif RESUME:
        print(f"[bw-train] no checkpoint at {CKPT} — training from scratch")
    it0 = int(ck.get("it", 0)) if ck is not None else 0

    # offset the RNG by the iterations already trained, or a resumed run would
    # replay the first run's seed batches verbatim
    torch.manual_seed(SEED + it0)
    np.random.seed(SEED + it0)

    scene = W.load_scene(BW / "tiles", dev)
    dem = W.Dem(BW / "dem.npz", dev)
    lab = np.load(BW / "labels.npz")
    bases_np = lab["bases"].astype(np.float32)
    train_bases = bases_np[~lab["is_val"] & ~lab["in_margin"]]
    val_bases = bases_np[lab["is_val"]]
    bases_t = torch.from_numpy(train_bases).to(dev)
    print(f"[bw-train] bases: train {len(train_bases)} | val {len(val_bases)}")

    sd = np.load(BW / "seeds.npz")
    seeds_all = sd["seeds"].astype(np.float32)
    tr_seeds = seeds_all[~sd["is_val"] & ~sd["in_margin"]]
    va_seeds = seeds_all[sd["is_val"]]
    print(f"[bw-train] seeds: train {len(tr_seeds):,} | val {len(va_seeds):,}")

    # near/far seed pools: walkers must both refine near-base starts and cover
    # open ground, in inference-matching proportions
    from scipy.spatial import cKDTree
    d_near = cKDTree(train_bases[:, :2]).query(tr_seeds[:, :2], k=1)[0]
    pool_near = tr_seeds[d_near < W.SPHERE_R + NEAR_R]
    pool_far = tr_seeds
    print(f"[bw-train] near-pool {len(pool_near):,} (within {NEAR_R} m of a base)")

    model = W.load_encoder()
    dec = W.WalkerDecoder().to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=LR, weight_decay=1e-4)

    best_f1 = -1.0
    if ck is not None:
        for k, cur in (("sphere_r", W.SPHERE_R), ("n_steps", W.N_STEPS),
                       ("conf_r", W.CONF_R)):
            if k in ck and ck[k] != cur:
                print(f"[bw-train] !! checkpoint {k}={ck[k]} != current {cur} — "
                      f"fine-tuning under changed geometry")
        dec.load_state_dict(ck["state_dict"])
        if "opt" in ck:                      # absent in pre-resume checkpoints
            opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:
            g["lr"] = RESUME_LR
            # the loaded state carries the old cycle's initial_lr, and the
            # scheduler's setdefault would keep it as the new base
            g.pop("initial_lr", None)
        # keep the best-F1 bar: the extra iterations must beat the resumed model
        # to replace it (LAST_CKPT holds the final decoder either way)
        best_f1 = float(ck.get("val", {}).get("f1", -1.0))
        print(f"[bw-train] resuming {CKPT} @ it {it0} (val F1 {best_f1:.3f}) | "
              f"+{ITERS} iters at peak LR {RESUME_LR:g}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)

    def sample_batch():
        n_near = int(BATCH * NEAR_FRAC)
        pick_n = pool_near[np.random.randint(0, len(pool_near), n_near)]
        pick_f = pool_far[np.random.randint(0, len(pool_far), BATCH - n_near)]
        s = np.concatenate([pick_n, pick_f]).copy()
        s[:, :2] += np.random.uniform(-SEED_JIT, SEED_JIT, (len(s), 2))
        seeds = torch.from_numpy(s.astype(np.float32)).to(dev)
        return dem.project(seeds).detach()

    t0, it_end, last_val = time.time(), it0 + ITERS, {}
    C.MODELS.mkdir(parents=True, exist_ok=True)
    for it in range(it0 + 1, it_end + 1):
        dec.train()
        loss, c_end, _, info = W.rollout(model, dec, scene, dem, sample_batch(),
                                         bases_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(dec.parameters(), CLIP)
        opt.step()
        sched.step()

        if it % 50 == 0 or it == it0 + 1:
            with torch.no_grad():
                d_end = torch.cdist(c_end, bases_t).min(1).values
            n_near = int(BATCH * NEAR_FRAC)  # sample_batch puts near seeds first
            d_near_end = d_end[:n_near]
            print(f"[bw-train] it {it:5d} | dist {info['dist']:7.2f} "
                  f"conf {info['conf']:5.2f} | "
                  f"near end-dist {d_near_end.median().item():5.2f} m | "
                  f"near hit@{W.CONF_R} "
                  f"{(d_near_end < W.CONF_R).float().mean().item()*100:4.1f}% | "
                  f"reach {info['n_reach']}/{info['n']} | "
                  f"|step| {info['step']:5.2f} m | grad {gnorm:7.2f} | "
                  f"{(time.time()-t0)/(it-it0):.2f} s/it", flush=True)

        if it % EVAL_EVERY == 0 or it == it_end:
            dec.eval()
            sub = va_seeds[np.random.permutation(len(va_seeds))[:EVAL_SEEDS]]
            ends, scores = W.detect(model, dec, scene, dem, sub)
            keep = W.nms(ends, scores)
            best = dict(f1=-1.0, thresh=0.5)
            for thr in np.linspace(0.1, 0.9, 17):
                k = keep[scores[keep] >= thr]
                m = W.match_metrics(ends[k], scores[k], val_bases, radius=W.NMS_R)
                if m["f1"] > best["f1"]:
                    best = dict(m, thresh=float(thr))
            print(f"[bw-train] EVAL it {it}: F1@{W.NMS_R} {best['f1']:.3f} "
                  f"(P {best['precision']:.3f} R {best['recall']:.3f} "
                  f"thr {best['thresh']:.2f} rmse {best['rmse']:.2f} m)", flush=True)
            last_val = best
            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                save_ckpt(CKPT, dec, opt, it, best)
                print(f"[bw-train] saved {CKPT} (F1 {best_f1:.3f})")

    save_ckpt(LAST_CKPT, dec, opt, it_end, last_val)
    if not CKPT.exists():  # eval never ran => still leave a usable decoder
        save_ckpt(CKPT, dec, opt, it_end, last_val)
    with open(BW / "train_summary.json", "w") as f:
        json.dump(dict(iters=ITERS, start_it=it0, end_it=it_end,
                       resumed=ck is not None, best_val_f1=best_f1,
                       last_val_f1=last_val.get("f1")), f)
    print(f"[bw-train] done at it {it_end}. best val F1 {best_f1:.3f} "
          f"({CKPT}) | last {LAST_CKPT}")


if __name__ == "__main__":
    main()
