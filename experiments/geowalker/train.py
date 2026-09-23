#!/usr/bin/env python3
"""GeoWalker stage B — train the decoder by simulated walking.

Every iteration draws a batch of seeds from the TRAIN spatial block, rolls the
walker GW_STEPS steps, and scores each step with the two terms the field makes
possible:

    l_geo   the kernel-weighted mean THROUGH-CLOUD distance to a tree base of
            the points around the newly predicted center
    l_dens  a penalty for landing somewhere with too few lidar points around it

plus the detached read-out term that lets an endpoint be scored at inference.
The encoder stays frozen; only the decoder is optimised.

Every GW_EVAL_EVERY iterations the walker is run over the held-out VAL block
exactly as inference does it (walk -> NMS -> threshold sweep -> greedy matching
against the RTK bases) and the best-F1 decoder is checkpointed.

Resumes by default, same contract as BaseWalker: an existing checkpoint is
loaded with its AdamW state and best F1, and GW_ITERS *more* iterations are
trained on a fresh cosine cycle with the RNG offset past the iterations already
done. GW_RESUME=0 starts cold.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# this directory first (its siblings are model.py/geodesic.py), then /app
# for the shared modules — /app also holds BaseWalker's train.py/infer.py,
# which must never shadow ours.
sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C
import model as M

BW = C.DATA / "basewalker"
GW = C.DATA / "geowalker"
FIELD = GW / "field.npz"
CKPT = C.MODELS / "geowalker_decoder.pth"
LAST_CKPT = C.MODELS / "geowalker_decoder_last.pth"

ITERS = int(os.environ.get("GW_ITERS", "9000"))       # THIS run, not cumulative
BATCH = int(os.environ.get("GW_BATCH", "48"))
LR = float(os.environ.get("GW_LR", "3e-4"))
CLIP = float(os.environ.get("GW_CLIP", "1.0"))
EVAL_EVERY = int(os.environ.get("GW_EVAL_EVERY", "500"))
EVAL_SEEDS = int(os.environ.get("GW_EVAL_SEEDS", "2048"))
NEAR_FRAC = float(os.environ.get("GW_NEAR_FRAC", "0.7"))
NEAR_R = float(os.environ.get("GW_NEAR_R", str(M.N_STEPS * M.MAX_STEP)))
SEED_JIT = float(os.environ.get("GW_SEED_JIT", "1.0"))
RESUME = os.environ.get("GW_RESUME", "1") != "0"
RESUME_LR = float(os.environ.get("GW_RESUME_LR", str(LR)))
SEED = int(os.environ.get("GW_SEED", "0"))

# geometry the checkpoint is only valid under — warned about on a resume
GEOM = dict(sphere_r=M.SPHERE_R, n_steps=M.N_STEPS, mem=M.MEM,
            max_step=M.MAX_STEP, field_r=M.FIELD_R, geo_sigma=M.GEO_SIGMA)


def save_ckpt(path, dec, opt, it, val):
    torch.save(dict(state_dict=dec.state_dict(), opt=opt.state_dict(), it=it,
                    val=val, **GEOM), path)


def seed_field_value(field, seeds):
    """Field value at each seed, via its nearest graph node. Seeds further than
    two voxels from any node sit off the cloud and get +inf (no supervision)."""
    from scipy.spatial import cKDTree
    d, i = cKDTree(field.xyz_np).query(seeds[:, :3].astype(np.float64), workers=-1)
    g = field.g_np[i].astype(np.float64)
    g[d > 2.0 * field.voxel] = np.inf
    return g


def evaluate(dec, enc, scene, dem, va_seeds, val_bases, n=EVAL_SEEDS):
    """Full detection eval on the val block: walk -> NMS -> best threshold."""
    sub = va_seeds[np.random.permutation(len(va_seeds))[:n]]
    ends, scores = M.detect(dec, enc, scene, dem, sub.astype(np.float32))
    keep = M.nms(ends, scores, radius=M.NMS_R)
    best = dict(f1=-1.0, thresh=0.5)
    for thr in np.linspace(0.05, 0.95, 19):
        k = keep[scores[keep] >= thr]
        m = M.match_metrics(ends[k], scores[k], val_bases, radius=M.NMS_R)
        if m["f1"] > best["f1"]:
            best = dict(m, thresh=float(thr))
    return best


def main():
    dev = "cuda"
    if not FIELD.exists():
        raise SystemExit(f"[gw-train] {FIELD} missing — run gw_field first")

    ck = None
    if RESUME and CKPT.exists():
        ck = torch.load(CKPT, map_location=dev, weights_only=False)
    elif RESUME:
        print(f"[gw-train] no checkpoint at {CKPT} — training from scratch")
    it0 = int(ck.get("it", 0)) if ck is not None else 0
    torch.manual_seed(SEED + it0)
    np.random.seed(SEED + it0)

    scene = M.load_scene(BW / "tiles", dev)
    dem = M.Dem(BW / "dem.npz", dev)
    field = M.load_field(FIELD, "g_train", dev)

    lab = np.load(BW / "labels.npz")
    bases = lab["bases"].astype(np.float32)
    val_bases = bases[lab["is_val"]]
    print(f"[gw-train] bases: train {int((~lab['is_val'] & ~lab['in_margin']).sum())} "
          f"| val {len(val_bases)}")

    sd = np.load(BW / "seeds.npz")
    seeds_all = sd["seeds"].astype(np.float32)
    tr_seeds = seeds_all[~sd["is_val"] & ~sd["in_margin"]]
    va_seeds = seeds_all[sd["is_val"]]
    print(f"[gw-train] seeds: train {len(tr_seeds):,} | val {len(va_seeds):,}")

    # near/far pools, split on the FIELD (not the straight line): a seed 3 m from
    # a base across a wall is genuinely far and belongs in the far pool.
    g_seed = seed_field_value(field, tr_seeds)
    near = np.isfinite(g_seed) & (g_seed <= NEAR_R)
    pool_near = tr_seeds[near] if near.any() else tr_seeds
    pool_far = tr_seeds
    print(f"[gw-train] near-pool {int(near.sum()):,} seeds "
          f"(field distance <= {NEAR_R} m) | far-pool {len(pool_far):,}")

    enc = M.load_encoder()
    dec = M.GeoWalkerDecoder().to(dev)
    n_par = sum(p.numel() for p in dec.parameters())
    print(f"[gw-train] decoder: {n_par/1e6:.2f} M params "
          f"({M.MEM} steps of memory, context radius {M.R_CTX:.1f} m)")
    opt = torch.optim.AdamW(dec.parameters(), lr=LR, weight_decay=1e-4)

    best_f1 = -1.0
    if ck is not None:
        for k, cur in GEOM.items():
            if k in ck and ck[k] != cur:
                print(f"[gw-train] !! checkpoint {k}={ck[k]} != current {cur} — "
                      f"fine-tuning under changed geometry")
        dec.load_state_dict(ck["state_dict"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:
            g["lr"] = RESUME_LR
            g.pop("initial_lr", None)   # or the scheduler keeps the old cycle
        best_f1 = float(ck.get("val", {}).get("f1", -1.0))
        print(f"[gw-train] resuming {CKPT} @ it {it0} (val F1 {best_f1:.3f}) | "
              f"+{ITERS} iters at peak LR {RESUME_LR:g}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)

    def sample_batch():
        n_near = int(BATCH * NEAR_FRAC)
        s = np.concatenate([pool_near[np.random.randint(0, len(pool_near), n_near)],
                            pool_far[np.random.randint(0, len(pool_far),
                                                       BATCH - n_near)]]).copy()
        s[:, :2] += np.random.uniform(-SEED_JIT, SEED_JIT, (len(s), 2))
        seeds = torch.from_numpy(s.astype(np.float32)).to(dev)
        return dem.project(seeds).detach()      # start on the ground, like infer

    t0, it_end, last_val = time.time(), it0 + ITERS, {}
    C.MODELS.mkdir(parents=True, exist_ok=True)
    GW.mkdir(parents=True, exist_ok=True)
    for it in range(it0 + 1, it_end + 1):
        dec.train()
        loss, _, _, info = M.rollout(dec, enc, scene, field, dem, sample_batch())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(dec.parameters(), CLIP)
        opt.step()
        sched.step()

        if it % 50 == 0 or it == it0 + 1:
            print(f"[gw-train] it {it:5d} | loss {float(loss.detach()):7.3f} "
                  f"(geo {info['geo']:6.2f} dens {info['dens']:5.3f} "
                  f"aux {info['aux']:5.3f}) | field {info['g0']:5.2f} -> "
                  f"{info['g_end']:5.2f} m | reach {info['n_reach']}/{info['n']} "
                  f"| ball {info['ball']:5.0f} pts (empty {info['n_empty']}) | "
                  f"|step| {info['step']:4.2f} m | grad {gnorm:7.2f} | "
                  f"{(time.time()-t0)/(it-it0):.2f} s/it", flush=True)

        if it % EVAL_EVERY == 0 or it == it_end:
            dec.eval()
            best = evaluate(dec, enc, scene, dem, va_seeds, val_bases)
            print(f"[gw-train] EVAL it {it}: F1@{M.NMS_R} {best['f1']:.3f} "
                  f"(P {best['precision']:.3f} R {best['recall']:.3f} "
                  f"thr {best['thresh']:.2f} rmse {best['rmse']:.2f} m)", flush=True)
            last_val = best
            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                save_ckpt(CKPT, dec, opt, it, best)
                print(f"[gw-train] saved {CKPT} (F1 {best_f1:.3f})")

    save_ckpt(LAST_CKPT, dec, opt, it_end, last_val)
    if not CKPT.exists():          # eval never ran -> still leave a usable model
        save_ckpt(CKPT, dec, opt, it_end, last_val)
    (GW / "train_summary.json").write_text(json.dumps(
        dict(iters=ITERS, start_it=it0, end_it=it_end, resumed=ck is not None,
             best_val_f1=best_f1, last_val_f1=last_val.get("f1"), **GEOM), indent=2))
    print(f"[gw-train] done at it {it_end}. best val F1 {best_f1:.3f} "
          f"({CKPT}) | last {LAST_CKPT}")


if __name__ == "__main__":
    main()
