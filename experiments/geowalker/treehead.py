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
# Mining cost control. The walk over every seed is ~9 encoder passes per batch
# and dwarfs the training it feeds; the head needs ENOUGH hard negatives, not
# all of them, so this caps the walk. 0 = every seed (the honest full pass).
SEEDS_N = int(os.environ.get("GW_HEAD_SEEDS", "0"))
CHUNK = int(os.environ.get("GW_HEAD_CHUNK", "8192"))     # walk-resume granularity
CACHE_GB = float(os.environ.get("GW_HEAD_CACHE_GB", "6"))  # token cache budget
FORCE = os.environ.get("GW_HEAD_FORCE", "0") == "1"      # ignore both caches
ENDS = GW / "treehead_endpoints.npz"                     # partial walk, resumable


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


def _sig(seeds, ck) -> str:
    """What a cached walk is only valid for: these seeds, this checkpoint, this
    geometry. Anything else and the cache is silently wrong, not stale."""
    import hashlib
    h = hashlib.sha1(np.ascontiguousarray(seeds, np.float32).tobytes())
    return (f"{h.hexdigest()[:16]}|n{len(seeds)}|it{ck.get('it')}|"
            f"steps{M.N_STEPS}|r{M.SPHERE_R}|amp{int(M.ENC_AMP)}|tf32{int(M.TF32)}")


def walk(dec, enc, scene, dem, seeds, ck):
    """Walk every seed to its endpoint, in chunks, saving as it goes.

    This is the most expensive thing in the experiment. It used to be a single
    uninterruptible call: 24 h in, a Ctrl-C threw away every endpoint it had
    found and the next run started from zero. Now each chunk lands on disk and a
    rerun picks up where it stopped.
    """
    GW.mkdir(parents=True, exist_ok=True)
    sig = _sig(seeds, ck)
    ends = np.zeros((0, 3), np.float32)
    scores = np.zeros(0, np.float32)
    if ENDS.exists() and not FORCE:
        z = np.load(ENDS, allow_pickle=False)
        if str(z["sig"]) == sig:
            ends, scores = z["ends"], z["scores"]
            print(f"[gw-head]   resuming {ENDS.name}: {len(ends):,} of "
                  f"{len(seeds):,} seeds already walked", flush=True)
        else:
            print(f"[gw-head]   {ENDS.name} belongs to a different walk "
                  f"(seeds/checkpoint/geometry changed) — walking again",
                  flush=True)
    start, t0 = len(ends), time.time()
    while len(ends) < len(seeds):
        hi = min(len(ends) + CHUNK, len(seeds))
        e, s = M.detect(dec, enc, scene, dem, seeds[len(ends):hi],
                        label="head-walk", base=len(ends), total=len(seeds))
        ends = np.concatenate([ends, e])
        scores = np.concatenate([scores, s])
        np.savez(ENDS, ends=ends, scores=scores, sig=sig)
        el = time.time() - t0
        rate = max(len(ends) - start, 1) / max(el, 1e-9)
        print(f"[gw-head]   checkpointed {len(ends):,}/{len(seeds):,} endpoints "
              f"| {M.hms(el)} elapsed | ETA "
              f"{M.hms((len(seeds) - len(ends)) / rate)}", flush=True)
    return ends, scores


class Tokens:
    """Frozen-Sonata tokens for a FIXED set of positions, encoded once.

    The encoder does not train here and the positions do not move, so calling it
    per batch re-did the same ~108 M-param forward EPOCHS times over identical
    geometry — as much encoder work as the mining walk itself, for nothing. Held
    in host RAM as fp16 and cast back on the way to the head; if that would not
    fit in GW_HEAD_CACHE_GB the cache turns itself off and the run falls back to
    encoding per batch (correct, just as slow as before).
    """

    def __init__(self, enc, scene, X, dev, batch=None):
        self.enc, self.scene, self.X, self.dev = enc, scene, X, dev
        self.on, self.feat, self.coord = False, [], []
        bs = int(batch or M.DETECT_BATCH)
        nbytes, t0, last = 0, time.time(), time.time()
        print(f"[gw-head] encoding {len(X):,} positions once "
              f"(fp16 token cache, budget {CACHE_GB:.1f} GB) ...", flush=True)
        for s in range(0, len(X), bs):
            c = torch.from_numpy(np.ascontiguousarray(X[s:s + bs],
                                                     np.float32)).to(dev)
            feats, coords = M.encode_spheres(enc, scene, c, r=HEAD_R)
            for f, co in zip(feats, coords):
                self.feat.append(f.half().cpu())
                self.coord.append(co.float().cpu())
                nbytes += f.numel() * 2 + co.numel() * 4
            done = min(s + bs, len(X))
            if nbytes / done * len(X) > CACHE_GB * 1e9:
                print(f"[gw-head]   projected {nbytes / done * len(X) / 1e9:.2f} GB "
                      f"> {CACHE_GB:.1f} GB budget — encoding per batch instead "
                      f"(GW_HEAD_CACHE_GB trades RAM for {EPOCHS}x less encoder "
                      f"time)", flush=True)
                self.feat, self.coord = [], []
                return
            if M.PROGRESS_S and time.time() - last >= M.PROGRESS_S:
                el = time.time() - t0
                print(f"[gw-head]   {done:,}/{len(X):,} encoded | "
                      f"{done / el:.0f} pos/s | ETA "
                      f"{M.hms((len(X) - done) / (done / el))} | "
                      f"{nbytes / 1e9:.2f} GB", flush=True)
                last = time.time()
        self.on = True
        print(f"[gw-head]   cached {nbytes / 1e9:.2f} GB of tokens in "
              f"{M.hms(time.time() - t0)} — the {EPOCHS} epochs now cost no "
              f"encoder time", flush=True)

    def get(self, rows):
        """(tok_feat, tok_rel, gate, mask) for these rows of X."""
        if not self.on:
            return encode_at(self.enc, self.scene, self.X[rows], self.dev)
        lens = [len(self.feat[i]) for i in rows]
        f = torch.cat([self.feat[i] for i in rows]).to(self.dev, torch.float32)
        co = torch.cat([self.coord[i] for i in rows]).to(self.dev, torch.float32)
        c = torch.from_numpy(np.ascontiguousarray(self.X[rows],
                                                 np.float32)).to(self.dev)
        tf, tr, g, _age, m = M.pad_context(
            [(list(torch.split(f, lens)), list(torch.split(co, lens)))],
            c, r_ctx=HEAD_R)
        return tf, tr, g, m


# ---------------------------------------------------------------------------
def mine(dec, enc, scene, dem, dev, ck):
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

    free = seeds[~s_margin]
    if SEEDS_N and len(free) > SEEDS_N:
        pick = np.sort(np.random.default_rng(SEED)
                       .permutation(len(free))[:SEEDS_N])
        walk_seeds = free[pick]
        print(f"[gw-head] walking a FIXED {len(walk_seeds):,} of {len(free):,} "
              f"non-margin seeds (GW_HEAD_SEEDS) — this thins the hard "
              f"negatives only; the labelled positives are unaffected",
              flush=True)
    else:
        walk_seeds = free
        print(f"[gw-head] walking all {len(walk_seeds):,} non-margin seeds to "
              f"mine real endpoints ...", flush=True)
    t0 = time.time()
    ends, scores = walk(dec, enc, scene, dem, walk_seeds, ck)
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

    # easy negatives: seeds far from everything (no walking, so the full
    # non-margin set regardless of how much of it was walked)
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


def evaluate(head, tok, rows, y, batch=128):
    head.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            tf, tr, g, m = tok.get(rows[s:s + batch])
            out.append(torch.sigmoid(head(tf, tr, g, m)).cpu().numpy())
    p = np.concatenate(out)
    from scipy.stats import rankdata
    pos, neg = p[y > 0.5], p[y < 0.5]
    r = rankdata(np.concatenate([pos, neg]))
    a = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
              / max(len(pos) * len(neg), 1))
    return p, a


def main():
    dev = M.DEVICE
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

    if CACHE.exists() and not FORCE:
        z = np.load(CACHE, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        print(f"[gw-head] dataset from {CACHE.name}: {len(d['X']):,} positions "
              f"| {int((d['src'] == 1).sum()):,} hard negatives "
              f"(GW_HEAD_FORCE=1 to mine it again)", flush=True)
    else:
        d = mine(dec, enc, scene, dem, dev, ck)
        np.savez(CACHE, **d)
        print(f"[gw-head] dataset -> {CACHE.name} (a rerun skips the walk)",
              flush=True)

    X, y, blk = d["X"], d["y"], d["is_val"]
    ends, gt, old = d["val_ends"], d["val_gt"], d["val_scores"]
    # One token cache over everything the head ever sees: the dataset positions
    # and the val endpoints the acceptance test re-scores. Same fp16 tokens for
    # training and for the test, so the comparison is not also a precision
    # change.
    tok = Tokens(enc, scene, np.concatenate([X, ends]), dev)
    rows_tr = np.where(~blk)[0]
    rows_va = np.where(blk)[0]
    rows_end = len(X) + np.arange(len(ends))
    ytr, yva = y[rows_tr], y[rows_va]

    head = TreeHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    n_steps = max(1, len(rows_tr) // BATCH) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    pw = torch.tensor(float((ytr < 0.5).sum()) / max(float((ytr > 0.5).sum()), 1.0),
                      device=dev)
    print(f"[gw-head] head {sum(p.numel() for p in head.parameters())/1e6:.2f} M "
          f"params | pos_weight {float(pw):.2f} | {EPOCHS} epochs")

    best = dict(auc=-1.0)
    steps_ep = len(range(0, len(rows_tr) - BATCH + 1, BATCH))
    for ep in range(1, EPOCHS + 1):
        head.train()
        order = np.random.permutation(rows_tr)
        tot, n, t0, last = 0.0, 0, time.time(), time.time()
        for s in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[s:s + BATCH]
            tf, tr, g, m = tok.get(idx)
            logit = head(tf, tr, g, m)
            loss = F.binary_cross_entropy_with_logits(
                logit, torch.from_numpy(y[idx]).to(dev), pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach()); n += 1
            if M.PROGRESS_S and time.time() - last >= M.PROGRESS_S:
                el = time.time() - t0
                print(f"[gw-head]   epoch {ep} step {n:,}/{steps_ep:,} | "
                      f"loss {tot / n:.4f} | {n / el:.1f} steps/s | "
                      f"ETA {M.hms((steps_ep - n) / (n / el))}", flush=True)
                last = time.time()
        p_va, auc = evaluate(head, tok, rows_va, yva)
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
    new, _ = evaluate(head, tok, rows_end, np.zeros(len(ends)))

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
        seeds_walked=SEEDS_N or None, token_cache=bool(tok.on),
        detect_batch=M.DETECT_BATCH, tf32=M.TF32, enc_amp=M.ENC_AMP,
        n_train=int(len(rows_tr)), n_val=int(len(rows_va)),
        n_hard_negatives=int((d["src"] == 1).sum()),
        best_val_auc=best["auc"], best_epoch=best["epoch"],
        incumbent=b_old, tree_head=b_new, delta_f1=gain,
        target_f1=TARGET_F1, sufficient=bool(b_new["f1"] >= TARGET_F1)),
        indent=2, default=float))
    print(f"[gw-head] -> results/geowalker_treehead.json")


if __name__ == "__main__":
    main()
