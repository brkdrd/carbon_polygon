#!/usr/bin/env python3
"""BaseWalker training: frozen Sonata encoder + walker decoder rollout.

Seeds are drawn exactly like inference (DEM-grid seeds + xy jitter) from the
TRAIN spatial block; loss is the discounted per-step squared distance to the
nearest labeled base + per-step BCE on "base within CONF_R of current center"
(pos-weighted). Periodic detection eval (NMS + greedy matching) on the VAL
block picks the checkpoint.
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

ITERS = int(os.environ.get("BW_ITERS", "3000"))
BATCH = int(os.environ.get("BW_BATCH", "48"))
LR = float(os.environ.get("BW_LR", "3e-4"))
CLIP = float(os.environ.get("BW_CLIP", "1.0"))
EVAL_EVERY = int(os.environ.get("BW_EVAL_EVERY", "500"))
EVAL_SEEDS = int(os.environ.get("BW_EVAL_SEEDS", "2048"))
NEAR_FRAC = float(os.environ.get("BW_NEAR_FRAC", "0.7"))
NEAR_R = float(os.environ.get("BW_NEAR_R", "8.0"))
SEED_JIT = float(os.environ.get("BW_SEED_JIT", "1.0"))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda"

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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)

    def sample_batch():
        n_near = int(BATCH * NEAR_FRAC)
        pick_n = pool_near[np.random.randint(0, len(pool_near), n_near)]
        pick_f = pool_far[np.random.randint(0, len(pool_far), BATCH - n_near)]
        s = np.concatenate([pick_n, pick_f]).copy()
        s[:, :2] += np.random.uniform(-SEED_JIT, SEED_JIT, (len(s), 2))
        seeds = torch.from_numpy(s.astype(np.float32)).to(dev)
        return dem.project(seeds).detach()

    best_f1, t0 = -1.0, time.time()
    C.MODELS.mkdir(parents=True, exist_ok=True)
    for it in range(1, ITERS + 1):
        dec.train()
        loss, c_end, _, info = W.rollout(model, dec, scene, dem, sample_batch(),
                                         bases_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(dec.parameters(), CLIP)
        opt.step()
        sched.step()

        if it % 50 == 0 or it == 1:
            with torch.no_grad():
                d_end = torch.cdist(c_end, bases_t).min(1).values
            n_near = int(BATCH * NEAR_FRAC)  # sample_batch puts near seeds first
            d_near_end = d_end[:n_near]
            print(f"[bw-train] it {it:5d} | loss {loss.item():9.3f} | "
                  f"near end-dist {d_near_end.median().item():5.2f} m | "
                  f"near hit@{W.CONF_R} "
                  f"{(d_near_end < W.CONF_R).float().mean().item()*100:4.1f}% | "
                  f"all median {d_end.median().item():6.2f} m | "
                  f"|step| {info['step']:5.2f} m | grad {gnorm:7.2f} | "
                  f"{(time.time()-t0)/it:.2f} s/it", flush=True)

        if it % EVAL_EVERY == 0 or it == ITERS:
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
            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                torch.save(dict(state_dict=dec.state_dict(), it=it,
                                val=best, sphere_r=W.SPHERE_R,
                                n_steps=W.N_STEPS, conf_r=W.CONF_R), CKPT)
                print(f"[bw-train] saved {CKPT} (F1 {best_f1:.3f})")

    if not CKPT.exists():  # eval never beat -1 => still save the final decoder
        torch.save(dict(state_dict=dec.state_dict(), it=ITERS, val={},
                        sphere_r=W.SPHERE_R, n_steps=W.N_STEPS,
                        conf_r=W.CONF_R), CKPT)
    with open(BW / "train_summary.json", "w") as f:
        json.dump(dict(iters=ITERS, best_val_f1=best_f1), f)
    print(f"[bw-train] done. best val F1 {best_f1:.3f}")


if __name__ == "__main__":
    main()
