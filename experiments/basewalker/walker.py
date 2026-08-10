"""BaseWalker core: GPU scene store, batched frozen-Sonata encoding of sphere
crops, the gated walker decoder, and the differentiable rollout.

Gradient design (agreed with the walk-through-time discussion):
  * The frozen encoder runs under torch.no_grad() on DETACHED crops — its output
    tokens are content only. No backward ever touches spconv (which also avoids
    the known Blackwell spconv-backward issue).
  * All geometric dependence on the current center c is re-attached at the
    decoder: token rel-positions (tok_abs - c) and the soft membership gate
    relu(1 - d/r) both carry gradient to c, so the through-time path
    loss_k -> c_k -> decoder_{k-1} -> c_{k-1} exists without encoder backward.
    The gate is the "relu(1m - distance)" logic applied at token level: as the
    center moves toward informative structure near the sphere boundary, that
    structure's attention weight rises smoothly.
  * DEM projection (bilinear) is differentiable in xy; z is the cloth height.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = float(os.environ.get("GRID_SIZE", "0.05"))
SPHERE_R = float(os.environ.get("BW_SPHERE_R", "2.0"))
N_STEPS = int(os.environ.get("BW_STEPS", "8"))
GAMMA = float(os.environ.get("BW_GAMMA", "0.85"))
CONF_R = float(os.environ.get("BW_CONF_R", "0.5"))
PTS_CAP = int(os.environ.get("BW_PTS_CAP", "4096"))
NMS_R = float(os.environ.get("BW_NMS_R", "0.5"))


# ---------------------------------------------------------------------------
# scene store: whole campus resident on GPU + 2D bucket index for ball queries
# ---------------------------------------------------------------------------
class SceneStore:
    def __init__(self, xyz: np.ndarray, normal: np.ndarray, cell: float = 1.0,
                 device="cuda"):
        self.device = device
        self.cell = cell
        self.xyz = torch.from_numpy(xyz.astype(np.float32)).to(device)
        self.normal = torch.from_numpy(normal.astype(np.float32)).to(device)
        x0 = float(xyz[:, 0].min()); y0 = float(xyz[:, 1].min())
        self.x0, self.y0 = x0, y0
        self.nx = int(np.floor((xyz[:, 0].max() - x0) / cell)) + 1
        self.ny = int(np.floor((xyz[:, 1].max() - y0) / cell)) + 1
        ix = ((xyz[:, 0] - x0) / cell).astype(np.int64).clip(0, self.nx - 1)
        iy = ((xyz[:, 1] - y0) / cell).astype(np.int64).clip(0, self.ny - 1)
        cid = iy * self.nx + ix
        order = np.argsort(cid, kind="stable")
        self.order = torch.from_numpy(order).to(device)
        cid_sorted = cid[order]
        starts = np.searchsorted(cid_sorted, np.arange(self.nx * self.ny))
        self.cell_start = torch.from_numpy(
            np.r_[starts, len(cid)].astype(np.int64)).to(device)

    def query_ball(self, center: torch.Tensor, r: float, cap: int = PTS_CAP):
        """Indices of scene points within 3D radius r of `center` (detached),
        random-subsampled to `cap`."""
        cx, cy = float(center[0]), float(center[1])
        ix0 = max(0, int((cx - r - self.x0) / self.cell))
        ix1 = min(self.nx - 1, int((cx + r - self.x0) / self.cell))
        iy0 = max(0, int((cy - r - self.y0) / self.cell))
        iy1 = min(self.ny - 1, int((cy + r - self.y0) / self.cell))
        if ix1 < ix0 or iy1 < iy0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        chunks = []
        for iy in range(iy0, iy1 + 1):
            c0 = iy * self.nx + ix0
            c1 = iy * self.nx + ix1 + 1
            s = self.cell_start[c0]
            e = self.cell_start[c1]
            if e > s:
                chunks.append(self.order[s:e])
        if not chunks:
            return torch.empty(0, dtype=torch.long, device=self.device)
        idx = torch.cat(chunks)
        d2 = ((self.xyz[idx] - center[None, :]) ** 2).sum(-1)
        idx = idx[d2 <= r * r]
        if len(idx) > cap:
            idx = idx[torch.randperm(len(idx), device=self.device)[:cap]]
        return idx


def load_scene(tiles_dir, device="cuda"):
    import glob
    xs, ns = [], []
    for p in sorted(glob.glob(str(tiles_dir / "tile_*.npz"))):
        d = np.load(p)
        xs.append(d["xyz"].astype(np.float32))
        ns.append(d["normal"].astype(np.float32))
    xyz = np.concatenate(xs); normal = np.concatenate(ns)
    print(f"[bw] scene: {len(xyz):,} pts from {len(xs)} tiles")
    return SceneStore(xyz, normal, device=device)


# ---------------------------------------------------------------------------
# DEM (differentiable bilinear lookup)
# ---------------------------------------------------------------------------
class Dem:
    def __init__(self, npz_path, device="cuda"):
        d = np.load(npz_path, allow_pickle=True)
        self.z = torch.from_numpy(d["z"].astype(np.float32)).to(device)
        self.x0 = float(d["x0"]); self.y0 = float(d["y0"])
        self.cell = float(d["cell"])
        self.ny, self.nx = self.z.shape

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear ground height at xy (B,2); differentiable in xy."""
        u = ((xy[:, 0] - self.x0) / self.cell).clamp(0, self.nx - 1.001)
        v = ((xy[:, 1] - self.y0) / self.cell).clamp(0, self.ny - 1.001)
        u0 = u.detach().floor().long(); v0 = v.detach().floor().long()
        du = (u - u0.float()).clamp(0, 1); dv = (v - v0.float()).clamp(0, 1)
        z00 = self.z[v0, u0]; z01 = self.z[v0, u0 + 1]
        z10 = self.z[v0 + 1, u0]; z11 = self.z[v0 + 1, u0 + 1]
        return (z00 * (1 - du) * (1 - dv) + z01 * du * (1 - dv)
                + z10 * (1 - du) * dv + z11 * du * dv)

    def project(self, c: torch.Tensor) -> torch.Tensor:
        """Drop centers (B,3) onto the cloth: z := dem(x, y)."""
        return torch.cat([c[:, :2], self.height(c[:, :2])[:, None]], dim=1)


# ---------------------------------------------------------------------------
# frozen Sonata encoding of a batch of sphere crops -> coarse tokens
# ---------------------------------------------------------------------------
def load_encoder():
    import sonata
    model = sonata.load("sonata", repo_id="facebook/sonata",
                        custom_config=dict(enc_patch_size=[1024] * 5,
                                           enable_flash=False)).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_spheres(model, scene: SceneStore, centers: torch.Tensor, r: float):
    """Encode the sphere crop around each (detached) center.

    Returns per-walker lists: token features [T,512] and token ABSOLUTE coords
    [T,3] (float32, no grad). Walkers whose sphere is empty get empty tensors.
    """
    B = len(centers)
    crops, owners = [], []
    for b in range(B):
        idx = scene.query_ball(centers[b].detach(), r)
        if len(idx) >= 16:
            crops.append(idx)
            owners.append(b)
    feats = [torch.zeros(0, 512, device=scene.device) for _ in range(B)]
    coords = [torch.zeros(0, 3, device=scene.device) for _ in range(B)]
    if not owners:
        return feats, coords

    coord_l, feat_l, shift_l, counts = [], [], [], []
    for idx in crops:
        pts = scene.xyz[idx]
        # pretrained CenterShift(apply_z=True): xy to mean, z to min
        shift = torch.cat([pts[:, :2].mean(0), pts[:, 2].min()[None]])
        local = pts - shift[None, :]
        # feat layout matches sonata_lib feat_keys=(coord,color,normal);
        # NormalizeColor maps the constant grey to 0.
        feat = torch.cat([local, torch.zeros_like(local), scene.normal[idx]], 1)
        coord_l.append(local); feat_l.append(feat); shift_l.append(shift)
        counts.append(len(idx))

    coord = torch.cat(coord_l)
    data = dict(
        coord=coord,
        grid_coord=((coord - coord.min(0).values[None, :]) / GRID).floor().long(),
        feat=torch.cat(feat_l),
        offset=torch.cumsum(torch.tensor(counts, device=scene.device), 0),
    )
    point = model(data)
    assert point.feat.shape[1] == 512, \
        f"expected 512-d coarse tokens, got {point.feat.shape[1]} — decoder dims must match"
    # point is the coarsest level; its offset gives per-sphere token ranges
    off = point.offset.tolist()
    lo = 0
    for k, hi in enumerate(off):
        b = owners[k]
        feats[b] = point.feat[lo:hi].float()
        coords[b] = point.coord[lo:hi] + shift_l[k][None, :]
        lo = hi
    return feats, coords


# ---------------------------------------------------------------------------
# walker decoder: gated cross-attention over coarse tokens
# ---------------------------------------------------------------------------
class WalkerDecoder(nn.Module):
    def __init__(self, feat_dim=512, d=256, heads=8, blocks=2):
        super().__init__()
        self.d, self.heads, self.blocks = d, heads, blocks
        self.tok = nn.Linear(feat_dim, d)
        self.pos = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.q_proj = nn.ModuleList(nn.Linear(d, d) for _ in range(blocks))
        self.k_proj = nn.ModuleList(nn.Linear(d, d) for _ in range(blocks))
        self.v_proj = nn.ModuleList(nn.Linear(d, d) for _ in range(blocks))
        self.o_proj = nn.ModuleList(nn.Linear(d, d) for _ in range(blocks))
        self.ln_q = nn.ModuleList(nn.LayerNorm(d) for _ in range(blocks))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                          nn.Linear(4 * d, d)) for _ in range(blocks))
        self.offset_head = nn.Linear(d, 3)
        self.conf_head = nn.Linear(d, 1)
        nn.init.zeros_(self.offset_head.weight)   # start as a stationary walker
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, tok_feat, tok_rel, gate, mask):
        """tok_feat (B,T,512) no-grad content; tok_rel (B,T,3) and gate (B,T)
        carry gradient to the center; mask (B,T) True = real token."""
        B, T, _ = tok_feat.shape
        x = self.tok(tok_feat) + self.pos(torch.cat([tok_rel, gate[..., None]], -1))
        # gate as additive attention bias: log w -> influence fades to zero
        # smoothly as a token approaches the sphere boundary
        bias = torch.log(gate.clamp_min(1e-6))[:, None, None, :]  # B,1,1,T
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
        q = self.query.expand(B, 1, self.d)
        hd = self.d // self.heads
        for i in range(self.blocks):
            qq = self.q_proj[i](self.ln_q[i](q)).view(B, 1, self.heads, hd).transpose(1, 2)
            kk = self.k_proj[i](x).view(B, T, self.heads, hd).transpose(1, 2)
            vv = self.v_proj[i](x).view(B, T, self.heads, hd).transpose(1, 2)
            att = (qq @ kk.transpose(-1, -2)) / math.sqrt(hd) + bias
            att = att.softmax(-1)
            out = (att @ vv).transpose(1, 2).reshape(B, 1, self.d)
            q = q + self.o_proj[i](out)
            q = q + self.ffn[i](q)
        q = q[:, 0]
        return self.offset_head(q), self.conf_head(q)[:, 0]


def pad_tokens(feats, coords, centers, r):
    """Per-walker token lists -> padded batch with rel-coords and gate.
    A dummy token (zero feat, rel 0, gate 1) guarantees >=1 valid key."""
    device = centers.device
    B = len(feats)
    T = max(1, max(len(f) for f in feats)) + 1
    tok_feat = torch.zeros(B, T, 512, device=device)
    tok_rel = torch.zeros(B, T, 3, device=device)
    gate = torch.zeros(B, T, device=device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, 0] = True
    gate[:, 0] = 1.0
    for b in range(B):
        n = len(feats[b])
        if n == 0:
            continue
        tok_feat[b, 1:n + 1] = feats[b]
        rel = coords[b] - centers[b][None, :]          # grad path to center
        tok_rel[b, 1:n + 1] = rel
        gate[b, 1:n + 1] = (1.0 - rel.norm(dim=-1) / r).clamp_min(0.0)
        mask[b, 1:n + 1] = True
    return tok_feat, tok_rel, gate, mask


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
def rollout(model, dec, scene, dem, seeds, bases, n_steps=N_STEPS, r=SPHERE_R,
            train=True):
    """Walk `seeds` (B,3) for n_steps. Returns (loss terms, trajectory info).
    bases: (M,3) GPU tensor of tree-base targets (None at pure inference)."""
    c = seeds.clone()
    conf_terms, dist_terms = [], []
    confs = []
    for k in range(n_steps + 1):
        feats, coords = encode_spheres(model, scene, c, r)
        tok_feat, tok_rel, gate, mask = pad_tokens(feats, coords, c, r)
        delta, conf = dec(tok_feat, tok_rel, gate, mask)
        confs.append(conf)
        if bases is not None:
            d_now = torch.cdist(c, bases).min(dim=1).values
            flag = (d_now < CONF_R).float()
            w_conf = GAMMA ** (n_steps - k)
            pos_w = torch.tensor(float(os.environ.get("BW_POS_WEIGHT", "5.0")),
                                 device=c.device)
            conf_terms.append(w_conf * F.binary_cross_entropy_with_logits(
                conf, flag, pos_weight=pos_w))
        if k == n_steps:
            break
        c = dem.project(c + delta)
        if bases is not None:
            d_new = torch.cdist(c, bases).min(dim=1).values
            w_dist = GAMMA ** (n_steps - 1 - k)
            dist_terms.append(w_dist * (d_new ** 2).mean())
        if not train:
            c = c.detach()
    loss = None
    if bases is not None:
        loss = torch.stack(dist_terms).sum() + torch.stack(conf_terms).sum()
    return loss, c, confs[-1]


# ---------------------------------------------------------------------------
# detection: NMS + matching metrics
# ---------------------------------------------------------------------------
def nms(points: np.ndarray, scores: np.ndarray, radius: float = NMS_R):
    order = np.argsort(-scores)
    kept = []
    for i in order:
        p = points[i]
        if all(((p - points[j]) ** 2).sum() > radius * radius for j in kept):
            kept.append(i)
    return np.array(kept, dtype=np.int64)


def match_metrics(pred_xyz, pred_score, gt_xyz, radius):
    """Greedy confidence-ordered matching. Returns dict with P/R/F1/RMSE."""
    if len(pred_xyz) == 0:
        return dict(tp=0, fp=0, fn=len(gt_xyz), precision=0.0, recall=0.0,
                    f1=0.0, rmse=float("nan"))
    used = np.zeros(len(gt_xyz), bool)
    order = np.argsort(-pred_score)
    tp, errs = 0, []
    for i in order:
        d = np.linalg.norm(gt_xyz - pred_xyz[i][None, :], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= radius:
            used[j] = True
            tp += 1
            errs.append(d[j])
    fp = len(pred_xyz) - tp
    fn = int((~used).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                rmse=float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan"))


@torch.no_grad()
def detect(model, dec, scene, dem, seeds_np, batch=256, n_steps=N_STEPS,
           r=SPHERE_R, device="cuda"):
    """Walk all seeds, return final positions + confidence (numpy)."""
    ends, scores = [], []
    for s in range(0, len(seeds_np), batch):
        seeds = torch.from_numpy(seeds_np[s:s + batch].astype(np.float32)).to(device)
        _, c, conf = rollout(model, dec, scene, dem, seeds, bases=None,
                             n_steps=n_steps, r=r, train=False)
        ends.append(c.cpu().numpy())
        scores.append(torch.sigmoid(conf).cpu().numpy())
    return np.concatenate(ends), np.concatenate(scores)
