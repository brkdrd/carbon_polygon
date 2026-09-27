#!/usr/bin/env python3
"""Can a dedicated head decide "is this a tree base?" well enough on its own?

The walker arrives: it drops the through-cloud distance 3.2 m -> 0.5 m and its
endpoints sit on stems. What it cannot do is tell a stem from a bush once it is
standing there — precision 0.41, and the false positives cluster on low woody
clutter. That decision is made by the read-out head, which is currently a
by-product: it regresses log1p of the field value, trained as a detached
auxiliary at weight 0.5 on whatever positions the rollout happened to visit.

This trains that decision DIRECTLY, as the binary question it actually is, on a
curated set of positions:

  positives     walker endpoints that landed on a base, plus the labelled bases
                themselves with jitter
  hard negatives the model's own false positives — the bushes and low clutter it
                currently calls trees, mined from a real inference pass
  easy negatives random seed positions far from any base

and then answers one question: does it separate them better than the incumbent
score does? The acceptance test is not AUC in a vacuum — it is what happens to
F1 when the EXISTING val endpoints are re-scored with this head and the
threshold re-swept. If that clears the bar, the head goes into the walker. If it
does not, the next step is a separate model, and this run says so plainly.

Nothing here touches the walker's weights. It is a static classifier over the
frozen Sonata encoding of one sphere, so it trains in minutes, not hours.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# this directory first (its siblings are model.py/geodesic.py), then /app
# for the shared modules — /app also holds BaseWalker's train.py/infer.py,
# which must never shadow ours.
sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402
import model as M  # noqa: E402

BW = C.DATA / "basewalker"
GW = C.DATA / "geowalker"
WALKER_CKPT = C.MODELS / "geowalker_decoder.pth"
HEAD_CKPT = C.MODELS / "geowalker_treehead.pth"
CACHE = GW / "treehead_dataset.npz"

HEAD_R = float(os.environ.get("GW_HEAD_R", str(M.SPHERE_R)))   # sphere it judges
POS_R = float(os.environ.get("GW_HEAD_POS_R", "0.5"))          # <= this: a tree
NEG_R = float(os.environ.get("GW_HEAD_NEG_R", "1.5"))          # >= this: not
                                                # (the band between is dropped)
EASY_N = int(os.environ.get("GW_HEAD_EASY", "6000"))   # random far negatives
JITTER_N = int(os.environ.get("GW_HEAD_JITTER", "4"))  # positives per base
JITTER_M = float(os.environ.get("GW_HEAD_JITTER_M", "0.25"))
EPOCHS = int(os.environ.get("GW_HEAD_EPOCHS", "12"))
BATCH = int(os.environ.get("GW_HEAD_BATCH", "64"))
LR = float(os.environ.get("GW_HEAD_LR", "3e-4"))
BLOCKS = int(os.environ.get("GW_HEAD_BLOCKS", "2"))
TARGET_F1 = float(os.environ.get("GW_HEAD_TARGET_F1", "0.60"))  # the bar
SEED = int(os.environ.get("GW_HEAD_SEED", "0"))


# ---------------------------------------------------------------------------
class TreeHead(nn.Module):
    """One learned query, cross-attention over the sphere's Sonata tokens, one
    logit. Same trunk shape as the walker decoder so a win here transfers, but
    trained on the binary question instead of inheriting it from a regression.
    """

    def __init__(self, feat_dim=512, d=256, heads=8, blocks=BLOCKS):
        super().__init__()
        self.d, self.heads, self.blocks = d, heads, blocks
        self.tok = nn.Linear(feat_dim, d)
        self.pos = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        mk = lambda: nn.ModuleList(nn.Linear(d, d) for _ in range(blocks))  # noqa: E731
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = mk(), mk(), mk(), mk()
        self.ln_q = nn.ModuleList(nn.LayerNorm(d) for _ in range(blocks))
        self.ln_kv = nn.ModuleList(nn.LayerNorm(d) for _ in range(blocks))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                          nn.Linear(4 * d, d)) for _ in range(blocks))
        self.out_ln = nn.LayerNorm(d)
        self.cls = nn.Linear(d, 1)

    def forward(self, tok_feat, tok_rel, gate, mask):
        B, T, _ = tok_feat.shape
        rel_n = tok_rel.norm(dim=-1, keepdim=True)
        x = self.tok(tok_feat) + self.pos(torch.cat([tok_rel, rel_n,
                                                     gate[..., None]], -1))
        bias = torch.log(gate.clamp_min(1e-6))[:, None, None, :]
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
        q = self.query.expand(B, 1, self.d)
        hd = self.d // self.heads
        for i in range(self.blocks):
            kv = self.ln_kv[i](x)
            qq = self.q_proj[i](self.ln_q[i](q)).view(B, 1, self.heads, hd).transpose(1, 2)
            kk = self.k_proj[i](kv).view(B, T, self.heads, hd).transpose(1, 2)
            vv = self.v_proj[i](kv).view(B, T, self.heads, hd).transpose(1, 2)
            att = ((qq @ kk.transpose(-1, -2)) / math.sqrt(hd) + bias).softmax(-1)
            q = q + self.o_proj[i]((att @ vv).transpose(1, 2).reshape(B, 1, self.d))
            q = q + self.ffn[i](q)
        return self.cls(self.out_ln(q[:, 0]))[:, 0]


def encode_at(enc, scene, xyz_np, dev):
    """Frozen-Sonata tokens for the sphere at each position, padded."""
    c = torch.from_numpy(np.ascontiguousarray(xyz_np, np.float32)).to(dev)
    feats, coords = M.encode_spheres(enc, scene, c, r=HEAD_R)
    tf, tr, gate, _age, mask = M.pad_context([(feats, coords)], c, r_ctx=HEAD_R)
    return tf, tr, gate, mask


# ---------------------------------------------------------------------------
def mine(dec, enc, scene, dem, dev):
    """Build the position dataset, hard negatives and all.

    Endpoints come from a real inference pass over every seed, so the negatives
    are the exact clutter the deployed walker stops on — not synthetic points
    that happen to be empty.
    """
    from scipy.spatial import cKDTree

    lab = np.load(BW / "labels.npz")
    bases = lab["bases"].astype(np.float32)
    is_val, in_margin = lab["is_val"], lab["in_margin"]
    sd = np.load(BW / "seeds.npz")
    seeds = sd["seeds"].astype(np.float32)
    s_val, s_margin = sd["is_val"], sd["in_margin"]

    print(f"[gw-head] walking {len(seeds):,} seeds to mine real endpoints ...",
          flush=True)
    t0 = time.time()
    ends, scores = M.detect(dec, enc, scene, dem, seeds[~s_margin])
    # Split on where the walker ENDED, not where it started: a train seed can
    # walk into the val block, and judging that position as training data would
    # leak the val block into the head.
    split_x = float(lab["split_x"])
    e_val = ends[:, 0] > split_x + 5.0
    e_keep = np.abs(ends[:, 0] - split_x) > 5.0          # drop the margin band
    ends, scores, e_val = ends[e_keep], scores[e_keep], e_val[e_keep]
    print(f"[gw-head]   {len(ends):,} endpoints in {time.time()-t0:.0f}s "
          f"({int(e_val.sum()):,} ended in the val block)")

    keep = M.nms(ends, scores, radius=M.NMS_R)
    ends, scores, e_val = ends[keep], scores[keep], e_val[keep]
    print(f"[gw-head]   {len(ends):,} after NMS")

    judge = bases[~in_margin]
    j_val = is_val[~in_margin]
    d_end = cKDTree(judge[:, :3]).query(ends[:, :3])[0]

    pos_m, neg_m = d_end <= POS_R, d_end >= NEG_R
    print(f"[gw-head]   endpoints: {int(pos_m.sum()):,} on a base (positive) | "
          f"{int(neg_m.sum()):,} hard negatives | "
          f"{int((~pos_m & ~neg_m).sum()):,} in the {POS_R}-{NEG_R} m band, dropped")

    X, y, blk, src = [], [], [], []
    X.append(ends[pos_m]); y.append(np.ones(pos_m.sum())); blk.append(e_val[pos_m])
    src.append(np.zeros(pos_m.sum()))                      # 0 = endpoint-positive
    X.append(ends[neg_m]); y.append(np.zeros(neg_m.sum())); blk.append(e_val[neg_m])
    src.append(np.ones(neg_m.sum()))                       # 1 = HARD negative

    # the labelled bases themselves, jittered — guarantees positive coverage
    # even where the walker never reached
    rng = np.random.default_rng(SEED)
    jit = np.repeat(judge, JITTER_N, axis=0).copy()
    jit[:, :2] += rng.uniform(-JITTER_M, JITTER_M, (len(jit), 2))
    X.append(jit); y.append(np.ones(len(jit)))
    blk.append(np.repeat(j_val, JITTER_N)); src.append(np.full(len(jit), 2.0))

    # easy negatives: seeds far from everything
    free = seeds[~s_margin]
    d_seed = cKDTree(judge[:, :3]).query(free[:, :3])[0]
    far = free[d_seed >= max(NEG_R, 3.0)]
    far_val = far[:, 0] > split_x + 5.0
    pick = rng.permutation(len(far))[:EASY_N]
    X.append(far[pick]); y.append(np.zeros(len(pick)))
    blk.append(far_val[pick]); src.append(np.full(len(pick), 3.0))

    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.float32)
    blk = np.concatenate(blk).astype(bool)
    src = np.concatenate(src).astype(np.float32)
    print(f"[gw-head] dataset {len(X):,} positions | "
          f"train {int((~blk).sum()):,} (pos {int(y[~blk].sum()):,}) | "
          f"val {int(blk.sum()):,} (pos {int(y[blk].sum()):,})")
    return dict(X=X, y=y, is_val=blk, src=src,
                val_ends=ends[e_val], val_scores=scores[e_val],
                val_gt=judge[j_val])


def evaluate(head, enc, scene, X, y, dev, batch=128):
    head.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            tf, tr, g, m = encode_at(enc, scene, X[s:s + batch], dev)
            out.append(torch.sigmoid(head(tf, tr, g, m)).cpu().numpy())
    p = np.concatenate(out)
    from scipy.stats import rankdata
    pos, neg = p[y > 0.5], p[y < 0.5]
    r = rankdata(np.concatenate([pos, neg]))
    a = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
              / max(len(pos) * len(neg), 1))
    return p, a


def main():
    dev = "cuda"
    if not WALKER_CKPT.exists():
        raise SystemExit(f"[gw-head] {WALKER_CKPT} missing — run gw_train first")
    torch.manual_seed(SEED); np.random.seed(SEED)

    scene = M.load_scene(BW / "tiles", dev)
    dem = M.Dem(BW / "dem.npz", dev)
    enc = M.load_encoder()
    ck = torch.load(WALKER_CKPT, map_location=dev, weights_only=False)
    dec = M.GeoWalkerDecoder().to(dev)
    dec.load_state_dict(ck["state_dict"]); dec.eval()
    print(f"[gw-head] walker from it {ck.get('it')} | sphere {HEAD_R} m")

    d = mine(dec, enc, scene, dem, dev)
    X, y, blk = d["X"], d["y"], d["is_val"]
    Xtr, ytr = X[~blk], y[~blk]
    Xva, yva = X[blk], y[blk]

    head = TreeHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    n_steps = max(1, len(Xtr) // BATCH) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    pw = torch.tensor(float((ytr < 0.5).sum()) / max(float((ytr > 0.5).sum()), 1.0),
                      device=dev)
    print(f"[gw-head] head {sum(p.numel() for p in head.parameters())/1e6:.2f} M "
          f"params | pos_weight {float(pw):.2f} | {EPOCHS} epochs")

    best = dict(auc=-1.0)
    for ep in range(1, EPOCHS + 1):
        head.train()
        order = np.random.permutation(len(Xtr))
        tot, n, t0 = 0.0, 0, time.time()
        for s in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[s:s + BATCH]
            tf, tr, g, m = encode_at(enc, scene, Xtr[idx], dev)
            logit = head(tf, tr, g, m)
            loss = F.binary_cross_entropy_with_logits(
                logit, torch.from_numpy(ytr[idx]).to(dev), pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach()); n += 1
        p_va, auc = evaluate(head, enc, scene, Xva, yva, dev)
        print(f"[gw-head] epoch {ep:2d} | loss {tot/max(n,1):.4f} | "
              f"val AUC {auc:.4f} | {time.time()-t0:.0f}s", flush=True)
        if auc > best["auc"]:
            best = dict(auc=auc, epoch=ep)
            torch.save(dict(state_dict=head.state_dict(), auc=auc, epoch=ep,
                            head_r=HEAD_R, pos_r=POS_R, neg_r=NEG_R), HEAD_CKPT)

    print(f"\n[gw-head] best val AUC {best['auc']:.4f} (epoch {best['epoch']}) "
          f"-> {HEAD_CKPT}")

    # ---- the acceptance test: re-score the real val endpoints ---------------
    head.load_state_dict(torch.load(HEAD_CKPT, map_location=dev,
                                    weights_only=False)["state_dict"])
    ends, gt = d["val_ends"], d["val_gt"]
    old = d["val_scores"]
    new, _ = evaluate(head, enc, scene, ends, np.zeros(len(ends)), dev)

    def best_f1(score):
        b = dict(f1=-1.0)
        for thr in M.sweep_thresholds(score):
            k = score >= thr
            m = M.match_metrics(ends[k], score[k], gt, radius=POS_R)
            if m["f1"] > b["f1"]:
                b = dict(m, thresh=float(thr), n=int(k.sum()))
        return b

    b_old, b_new = best_f1(old), best_f1(new)
    print(f"\n[gw-head] ===== acceptance test: same endpoints, new score =====")
    for name, b in (("incumbent read-out", b_old), ("trained tree head", b_new)):
        print(f"  {name:20s} F1 {b['f1']:.3f} | P {b['precision']:.3f} "
              f"R {b['recall']:.3f} | {b['n']} kept @ thr {b['thresh']:.3f}")
    gain = b_new["f1"] - b_old["f1"]
    print(f"  change: {gain:+.3f} F1")

    verdict = ("SUFFICIENT — fold this head into the walker"
               if b_new["f1"] >= TARGET_F1 else
               "NOT SUFFICIENT at the current bar — a separate model is the "
               "next step")
    print(f"\n[gw-head] target F1 {TARGET_F1:.2f} -> {verdict}")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "geowalker_treehead.json").write_text(json.dumps(dict(
        head_r=HEAD_R, pos_r=POS_R, neg_r=NEG_R, epochs=EPOCHS,
        n_train=int(len(Xtr)), n_val=int(len(Xva)),
        n_hard_negatives=int((d["src"] == 1).sum()),
        best_val_auc=best["auc"], best_epoch=best["epoch"],
        incumbent=b_old, tree_head=b_new, delta_f1=gain,
        target_f1=TARGET_F1, sufficient=bool(b_new["f1"] >= TARGET_F1)),
        indent=2, default=float))
    print(f"[gw-head] -> results/geowalker_treehead.json")


if __name__ == "__main__":
    main()
