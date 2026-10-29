"""StemNet v2, DETR-style head: tree bases as a SET, one prediction per tree.

Per 8 m crop:
  1. frozen Sonata (1,232 per point) + the 12 standardised covariance features,
     exactly what the gate / offset head sees
  2. averaged into V2_DETR_VOXEL (0.25 m) voxels: the memory, each voxel with
     its position (x, y from the crop centre / 4 m, z from the crop's lowest
     point / 10 m) through a small MLP. No encoder self-attention: Sonata
     already is the encoder; the decoder only cross-attends (cost queries x
     voxels, not voxels^2).
  3. V2_DETR_QUERIES (49) queries, each with a learned feature and a learned
     2-D reference point (initialised on a 7 x 7 grid over the crop). Each of
     V2_DETR_LAYERS (6) decoder layers: self-attention between queries (so
     two cannot claim the same tree), cross-attention to the voxels, a
     feed-forward block; then every query moves its reference point by a
     predicted step (iterative refinement, as Deformable / DAB-DETR) and says
     how sure it is that it holds a tree base.
Loss (every layer, as DETR's auxiliary losses): Hungarian matching of queries
to the crop's survey bases (cost: -probability x 2 + L1 position in crop units
x 5), focal loss on "base / no object" (alpha 0.25, gamma 2), L1 on the
matched positions. Nothing else: no gate, no offsets, no vote threshold, no
peak finder; duplicates are penalised by the matching itself.

Region inference: crops centred on a 4 m grid; a crop keeps only the
predictions inside its own central 4 x 4 m square, so every spot belongs to
exactly one crop (2 m of context on every side); then predictions closer
than V2_DETR_NMS (0.3 m) to a stronger one are dropped (a tree on the border
of two squares can be claimed from both sides). Score = probability.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 8.0
OWN = 4.0
VOXEL = float(os.environ.get("V2_DETR_VOXEL", "0.25"))
NMS_R = float(os.environ.get("V2_DETR_NMS", "0.3"))
MIN_P = float(os.environ.get("V2_DETR_MIN_P", "0.01"))
MAX_TOKENS = int(float(os.environ.get("V2_DETR_MAX_TOKENS", "30000")))
W_CLS, W_XY = 2.0, 5.0
ALPHA, GAMMA = 0.25, 2.0
HALF = CROP / 2


class MLP(nn.Module):
    def __init__(self, i, h, o, n=2):
        super().__init__()
        dims = [i] + [h] * (n - 1) + [o]
        self.net = nn.Sequential(*[m for a, b in zip(dims[:-1], dims[1:]) for m in (nn.Linear(a, b), nn.GELU())][:-1])

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, q, qpos, mem, mpos):
        x = q + qpos
        q = self.n1(q + self.sa(x, x, q, need_weights=False)[0])
        q = self.n2(q + self.ca(q + qpos, mem + mpos, mem, need_weights=False)[0])
        return self.n3(q + self.ff(q))


class DetrHead(nn.Module):
    def __init__(self, d_in, d=256, n_query=49, layers=6, heads=8, ff=1024):
        super().__init__()
        self.d_in, self.cfg = d_in, dict(d=d, n_query=n_query, layers=layers, heads=heads, ff=ff)
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.pos = MLP(3, d, d)
        self.ref_pos = MLP(2, d, d)
        self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
        k = int(round(np.sqrt(n_query)))
        if k * k == n_query:
            g = np.linspace(-0.85, 0.85, k)
            ref = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        else:
            ref = np.random.default_rng(0).uniform(-0.85, 0.85, (n_query, 2))
        self.ref0 = nn.Parameter(torch.tensor(ref, dtype=torch.float32))
        self.layers = nn.ModuleList(DecoderLayer(d, heads, ff) for _ in range(layers))
        self.cls = nn.Linear(d, 1)
        self.step = MLP(d, d, 2)
        nn.init.constant_(self.cls.bias, -float(np.log((1 - 0.01) / 0.01)))   # start at p = 0.01, as focal-loss detectors do
        nn.init.zeros_(self.step.net[-1].weight)
        nn.init.zeros_(self.step.net[-1].bias)

    def forward(self, feat, pos):
        """feat (M, d_in) voxel features, pos (M, 3) normalised voxel positions ->
        per layer (logits (Q,), ref (Q, 2) in crop units: metres / 4 from the centre)."""
        mem, mpos = self.inp(feat)[None], self.pos(pos)[None]
        q, ref, outs = self.query[None], self.ref0, []
        for layer in self.layers:
            q = layer(q, self.ref_pos(ref)[None], mem, mpos)
            ref = ref + self.step(q[0])
            outs.append((self.cls(q[0])[:, 0], ref))
            ref = ref.detach()
        return outs


def voxelise(f, xyz, centre):
    """Point features -> voxel means and normalised voxel positions (torch)."""
    rel = torch.as_tensor(xyz - np.r_[centre, xyz[:, 2].min()], dtype=torch.float32, device=f.device)
    key = torch.floor(rel / VOXEL).long()
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    n = len(uniq)
    s = torch.zeros(n, f.shape[1], device=f.device).index_add_(0, inv, f)
    cnt = torch.zeros(n, device=f.device).index_add_(0, inv, torch.ones(len(inv), device=f.device))
    pos = torch.zeros(n, 3, device=f.device).index_add_(0, inv, rel)
    feat, pos = s / cnt[:, None], pos / cnt[:, None]
    if n > MAX_TOKENS:
        keep = torch.randperm(n, device=f.device)[:MAX_TOKENS]
        feat, pos = feat[keep], pos[keep]
    return feat, pos / torch.tensor([HALF, HALF, 10.0], device=f.device)


def focal(logits, target):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    return (ce * (1 - pt) ** GAMMA * (ALPHA * target + (1 - ALPHA) * (1 - target))).sum()


def set_loss(outs, gt):
    """gt (G, 2) in crop units. -> total loss, stats of the last layer."""
    from scipy.optimize import linear_sum_assignment
    total, n = 0.0, max(len(gt), 1)
    for logits, ref in outs:
        target = torch.zeros_like(logits)
        if len(gt):
            cost = -W_CLS * torch.sigmoid(logits)[:, None] + W_XY * torch.cdist(ref, gt, p=1)
            qi, gi = linear_sum_assignment(cost.detach().float().cpu().numpy())
            qi, gi = torch.as_tensor(qi, device=logits.device), torch.as_tensor(gi, device=logits.device)
            target[qi] = 1.0
            l_xy = (ref[qi] - gt[gi]).abs().sum() / n
        else:
            qi = gi = torch.zeros(0, dtype=torch.long, device=logits.device)
            l_xy = ref.sum() * 0.0
        total = total + W_CLS * focal(logits, target) / n + W_XY * l_xy
    with torch.no_grad():
        p = torch.sigmoid(outs[-1][0])
        err = ((outs[-1][1][qi] - gt[gi]) * HALF).norm(dim=1).mean().item() if len(gt) else float("nan")
        st = dict(gt=len(gt), err_m=err, p_matched=float(p[qi].mean()) if len(gt) else float("nan"),
                  p_other=float(p[~target.bool()].mean()) if len(gt) < len(p) else float("nan"),
                  confident=int((p >= 0.5).sum()))
    return total / len(outs), st


def crop_bases(bases_kd, bases_xy, cx, cy):
    """Survey bases inside the 8 m square around (cx, cy)."""
    idx = bases_kd.query_ball_point([cx, cy], HALF * np.sqrt(2) + 1e-6)
    b = bases_xy[idx] if len(idx) else np.zeros((0, 2))
    return b[(np.abs(b[:, 0] - cx) <= HALF) & (np.abs(b[:, 1] - cy) <= HALF)]


def nms(det, score, r=None):
    from scipy.spatial import cKDTree
    r = NMS_R if r is None else r
    if not len(det) or r <= 0:
        return det, score
    o = np.argsort(-score, kind="stable")
    det, score = det[o], score[o]
    kd, keep, dead = cKDTree(det), [], np.zeros(len(det), bool)
    for i in range(len(det)):
        if dead[i]:
            continue
        keep.append(i)
        dead[kd.query_ball_point(det[i], r)] = True
    return det[keep], score[keep]


def predict_region(predict_crop, store, bbox, cap, rng):
    """predict_crop(crop, centre) -> (xy (Q, 2) metres, score (Q,)). Crops on a 4 m
    grid, each keeping its own central 4 x 4 m square, then NMS. -> det, score."""
    x0, x1, y0, y1 = bbox
    D, S = [], []
    for cx in np.arange(x0 + OWN / 2, x1 + OWN / 2, OWN):
        for cy in np.arange(y0 + OWN / 2, y1 + OWN / 2, OWN):
            c = store.crop(cx, cy, CROP)
            if c is None or len(c["xyz"]) < 64:
                continue
            if len(c["xyz"]) > cap:
                sel = rng.choice(len(c["xyz"]), cap, replace=False)
                c = {k: v[sel] for k, v in c.items()}
            xy, s = predict_crop(c, np.array([cx, cy]))
            own = ((xy[:, 0] >= cx - OWN / 2) & (xy[:, 0] < cx + OWN / 2) & (xy[:, 1] >= cy - OWN / 2)
                   & (xy[:, 1] < cy + OWN / 2) & (s >= MIN_P))
            D.append(xy[own]); S.append(s[own])
    if not D:
        return np.zeros((0, 2)), np.zeros(0)
    det, score = np.concatenate(D), np.concatenate(S)
    inb = (det[:, 0] >= x0) & (det[:, 0] <= x1) & (det[:, 1] >= y0) & (det[:, 1] <= y1)
    return nms(det[inb], score[inb])
