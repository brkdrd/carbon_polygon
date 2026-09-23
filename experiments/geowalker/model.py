"""GeoWalker core — frozen Sonata encoder + a small transformer decoder that
walks one point toward the nearest tree base, supervised by the through-cloud
distance field built in `geodesic.py`.

The task, in one line: *given a small sphere of lidar points (plus the spheres
from the previous MEM steps), output the single point to move to next.*

What the model sees
  * the sphere of radius SPHERE_R around the current center, encoded ONCE by the
    frozen Sonata (PTv3) encoder into coarse tokens;
  * the tokens of the previous MEM spheres, kept verbatim — the dynamics context,
    so the decoder knows where it came from and does not oscillate.
  Every token is re-expressed relative to the CURRENT center, so all of them
  carry gradient to it even though the encoder itself never sees a gradient.

What it is scored on, per simulated step
  * l_geo  — the kernel-weighted mean THROUGH-CLOUD distance to a tree base of
             the cloud around the newly predicted center. Lower = the step
             landed somewhere better connected to a base. Linearised far away
             (smooth-L1, beta = HUBER_M) so a walker 40 m out cannot drown the
             gradient of one that is 1 m out.
  * l_dens — the mean distance to the DENS_K nearest lidar points, hinged at
             DENS_D0. This is "is there actually structure here": stepping into
             open air costs, stepping along a trunk is free.
  * l_aux  — read-out only: the decoder also predicts log1p of the field value
             at its current center. It is trained on the measured value
             (DETACHED, so it never steers the walk) and exists so an endpoint
             can be turned into a scored detection at inference, where the
             field does not exist.

Gradient design (inherited from the BaseWalker post-mortem, see walker.py)
  * the encoder runs under no_grad on detached crops — no backward ever touches
    spconv;
  * every geometric dependence on the center is re-attached at the decoder
    (token rel-positions, the membership gate) and in the loss (distances to a
    detached point set), so loss_k -> c_k -> decoder_{k-1} exists anyway;
  * the step is norm-clamped to MAX_STEP and truncated every TBPTT steps, which
    is what stopped the 100 m escapes and the through-time blow-up last time.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# BaseWalker owns the scene primitives (GPU bucket index, DEM, Sonata loader,
# NMS/matching). Same scene, same frame — reuse rather than fork them.
from walker import (SceneStore, Dem, load_scene, load_encoder, nms,  # noqa: F401
                    match_metrics, match_pairs)

GRID = float(os.environ.get("GRID_SIZE", "0.05"))

SPHERE_R = float(os.environ.get("GW_SPHERE_R", "2.0"))    # encoder crop radius
N_STEPS = int(os.environ.get("GW_STEPS", "8"))            # simulated steps
MEM = int(os.environ.get("GW_MEM", "3"))                  # previous steps kept
MAX_STEP = float(os.environ.get("GW_MAX_STEP", "1.0"))    # metres per step
GAMMA = float(os.environ.get("GW_GAMMA", "0.85"))         # step discount
PTS_CAP = int(os.environ.get("GW_PTS_CAP", "4096"))       # points fed to Sonata

FIELD_R = float(os.environ.get("GW_FIELD_R", "2.0"))      # loss ball radius
GEO_SIGMA = float(os.environ.get("GW_GEO_SIGMA", "0.4"))  # loss kernel width
HUBER_M = float(os.environ.get("GW_HUBER_M", "2.0"))      # geodesic linear knee
DENS_K = int(os.environ.get("GW_DENS_K", "12"))           # "enough points" = K
DENS_D0 = float(os.environ.get("GW_DENS_D0", "0.5"))      # free radius, m
W_DENS = float(os.environ.get("GW_W_DENS", "2.0"))        # density weight
W_AUX = float(os.environ.get("GW_W_AUX", "0.5"))          # read-out weight
REACH_M = float(os.environ.get("GW_REACH_M", str(1.5 * N_STEPS * MAX_STEP)))
FIELD_CAP = int(os.environ.get("GW_FIELD_CAP", "4096"))   # safety valve

TBPTT = int(os.environ.get("GW_TBPTT", "2"))
DEM_PROJECT = os.environ.get("GW_DEM_PROJECT", "0") == "1"
SCORE_TAU = float(os.environ.get("GW_SCORE_TAU", "0.5"))  # score = exp(-g/tau)
NMS_R = float(os.environ.get("GW_NMS_R", "0.5"))

# context radius: the oldest memory sphere can be MEM full steps behind
R_CTX = SPHERE_R + MEM * MAX_STEP


# ---------------------------------------------------------------------------
# stores
# ---------------------------------------------------------------------------
class FieldStore(SceneStore):
    """The geodesic field as a queryable point set: voxel centres + their
    through-cloud distance to the nearest tree base (+inf where no path exists).

    Reuses SceneStore purely for its 2D bucket index; normals are not needed, so
    it is given a zero-width normal array (no memory)."""

    def __init__(self, xyz: np.ndarray, g: np.ndarray, cell: float = 1.0,
                 device="cuda"):
        super().__init__(xyz, np.zeros((len(xyz), 0), np.float32), cell, device)
        self.g = torch.from_numpy(np.asarray(g, np.float32)).to(device)


def load_field(path, which="g_train", device="cuda"):
    d = np.load(path)
    g = d[which]
    xyz = d["xyz"].astype(np.float32)
    fin = np.isfinite(g)
    med = float(np.median(g[fin])) if fin.any() else float("nan")
    print(f"[gw] field {path.name}:{which}: {len(xyz):,} nodes @ "
          f"{float(d['voxel']):.2f} m | {fin.mean()*100:.1f}% reachable | "
          f"median {med:.1f} m")
    if not fin.any():
        raise SystemExit(f"[gw] {path.name}:{which} has no reachable node — the "
                         f"graph never connected to a base (raise GW_MAX_EDGE "
                         f"or GW_KNN, or check the labels)")
    store = FieldStore(xyz, g, device=device)
    # kept for CPU-side lookups (which seeds are near a base); the GPU copy is
    # the one the loss queries.
    store.xyz_np, store.g_np = xyz, g
    store.voxel = float(d["voxel"])
    return store


def _cell_start_np(store):
    cs = getattr(store, "_cs_np", None)
    if cs is None:
        cs = store.cell_start.cpu().numpy()
        store._cs_np = cs
    return cs


def batch_ball(store, centers: torch.Tensor, r: float):
    """Points of `store` within 3D radius r of each center.

    Returns flat (idx, owner), grouped by owner. One GPU->CPU sync for the whole
    batch instead of one per walker: the bucket arithmetic runs in numpy and the
    radius test runs once, batched, on the GPU.
    """
    dev = store.device
    cs = _cell_start_np(store)
    cnp = centers.detach().cpu().numpy()
    rows, own_b, own_n = [], [], []
    for b in range(len(cnp)):
        cx, cy = float(cnp[b][0]), float(cnp[b][1])
        ix0 = max(0, int((cx - r - store.x0) / store.cell))
        ix1 = min(store.nx - 1, int((cx + r - store.x0) / store.cell))
        iy0 = max(0, int((cy - r - store.y0) / store.cell))
        iy1 = min(store.ny - 1, int((cy + r - store.y0) / store.cell))
        if ix1 < ix0 or iy1 < iy0:
            continue
        for iy in range(iy0, iy1 + 1):
            s = int(cs[iy * store.nx + ix0])
            e = int(cs[iy * store.nx + ix1 + 1])
            if e > s:
                rows.append(store.order[s:e])
                own_b.append(b)
                own_n.append(e - s)
    if not rows:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z
    idx = torch.cat(rows)
    owner = torch.repeat_interleave(
        torch.tensor(own_b, dtype=torch.long, device=dev),
        torch.tensor(own_n, dtype=torch.long, device=dev))
    d2 = ((store.xyz[idx] - centers.detach()[owner]) ** 2).sum(-1)
    keep = d2 <= r * r
    return idx[keep], owner[keep]


def ball_lists(store, centers, r, cap=PTS_CAP):
    """batch_ball split per walker, random-subsampled to `cap`."""
    B = len(centers)
    idx, owner = batch_ball(store, centers, r)
    if len(idx) == 0:
        return [idx[:0] for _ in range(B)]
    counts = torch.bincount(owner, minlength=B).cpu().tolist()
    out, pos = [], 0
    for n in counts:
        sub = idx[pos:pos + n]
        pos += n
        if cap and n > cap:
            sub = sub[torch.randperm(n, device=idx.device)[:cap]]
        out.append(sub)
    return out


# ---------------------------------------------------------------------------
# the two loss terms (both differentiable in the center, point set detached)
# ---------------------------------------------------------------------------
def field_terms(field: FieldStore, centers: torch.Tensor, r=FIELD_R,
                sigma=GEO_SIGMA, dens_k=DENS_K, dens_d0=DENS_D0):
    """Geometry of the loss at `centers` (B,3).

    Returns (gbar, dens, has_pts, finite, n_pts):
      gbar    (B,) kernel-weighted mean through-cloud distance to a tree base of
              the cloud AROUND the center. exp(-d^2/2sigma^2) weights, so the
              gradient is a mean-shift toward the better-connected side.
      dens    (B,) mean distance to the DENS_K nearest points, hinged at
              dens_d0 and normalised by r: 0 inside real structure, ->1 in air.
      has_pts (B,) the ball held any point at all
      finite  (B,) the ball held any point with a finite field value
      n_pts   (B,) points in the ball (diagnostics)
    """
    B = len(centers)
    dev = centers.device
    idx, owner = batch_ball(field, centers, r)
    zeros = centers.new_zeros(B)
    no = torch.zeros(B, dtype=torch.bool, device=dev)
    if len(idx) == 0:
        return zeros, zeros + 1.0, no, no, torch.zeros(B, dtype=torch.long, device=dev)

    counts = torch.bincount(owner, minlength=B)
    n_pts = counts.clone()
    nmax = min(int(counts.max().item()), FIELD_CAP)
    off = torch.cumsum(counts, 0) - counts                    # exclusive scan
    slot = torch.arange(len(idx), device=dev) - off[owner]
    keep = slot < nmax                                        # safety valve
    idx, owner, slot = idx[keep], owner[keep], slot[keep]

    pts = centers.new_zeros(B, nmax, 3)
    gv = centers.new_zeros(B, nmax)
    msk = torch.zeros(B, nmax, dtype=torch.bool, device=dev)
    pts[owner, slot] = field.xyz[idx]
    gv[owner, slot] = field.g[idx]
    msk[owner, slot] = True

    # +eps: the norm is not differentiable at 0, and a center can land exactly
    # on a voxel centre.
    d = torch.sqrt(((pts - centers[:, None, :]) ** 2).sum(-1) + 1e-12)
    # Padded slots are pinned at the ball radius: torch.where routes no gradient
    # into them, and r is exactly the right stand-in for "no neighbour here" in
    # the density term below.
    d = torch.where(msk, d, torch.full_like(d, r))

    fin = msk & torch.isfinite(gv)
    w = torch.exp(-(d * d) / (2.0 * sigma * sigma)) * fin
    gbar = ((w * torch.where(fin, gv, torch.zeros_like(gv))).sum(1)
            / w.sum(1).clamp_min(1e-12))

    kk = min(dens_k, nmax)
    dk = torch.topk(d, kk, dim=1, largest=False).values
    mean_k = (dk.sum(1) + r * (dens_k - kk)) / dens_k
    dens = torch.relu(mean_k - dens_d0) / r

    return gbar, dens, counts > 0, fin.any(1), n_pts


# ---------------------------------------------------------------------------
# frozen Sonata encoding of the sphere crops
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_spheres(enc, scene: SceneStore, centers: torch.Tensor, r=SPHERE_R):
    """Encode the sphere around each (detached) center with the frozen encoder.

    Returns per-walker lists of token features [T,512] and token ABSOLUTE coords
    [T,3]. A walker whose sphere is (near) empty gets empty tensors.
    """
    B = len(centers)
    dev = scene.device
    crops = ball_lists(scene, centers, r)
    feats = [torch.zeros(0, 512, device=dev) for _ in range(B)]
    coords = [torch.zeros(0, 3, device=dev) for _ in range(B)]
    owners = [b for b in range(B) if len(crops[b]) >= 16]
    if not owners:
        return feats, coords

    coord_l, feat_l, shift_l, counts = [], [], [], []
    for b in owners:
        pts = scene.xyz[crops[b]]
        # pretrained CenterShift(apply_z=True): xy to mean, z to min
        shift = torch.cat([pts[:, :2].mean(0), pts[:, 2].min()[None]])
        local = pts - shift[None, :]
        # feat layout matches sonata_lib feat_keys=(coord,color,normal);
        # NormalizeColor maps the constant grey to 0.
        feat = torch.cat([local, torch.zeros_like(local), scene.normal[crops[b]]], 1)
        coord_l.append(local)
        feat_l.append(feat)
        shift_l.append(shift)
        counts.append(len(crops[b]))

    coord = torch.cat(coord_l)
    point = enc(dict(
        coord=coord,
        grid_coord=((coord - coord.min(0).values[None, :]) / GRID).floor().long(),
        feat=torch.cat(feat_l),
        offset=torch.cumsum(torch.tensor(counts, device=dev), 0),
    ))
    assert point.feat.shape[1] == 512, \
        f"expected 512-d coarse tokens, got {point.feat.shape[1]}"
    lo = 0
    for k, hi in enumerate(point.offset.tolist()):
        b = owners[k]
        feats[b] = point.feat[lo:hi].float()
        coords[b] = point.coord[lo:hi] + shift_l[k][None, :]
        lo = hi
    return feats, coords


def pad_context(hist, centers: torch.Tensor, r_ctx=R_CTX):
    """Tokens of the current sphere + the MEM previous ones -> one padded batch.

    `hist` is oldest-first; age 0 is the current sphere. Every token's position
    is expressed relative to the CURRENT center (so the whole context carries
    gradient to it) and gated by the soft membership relu(1 - |rel| / r_ctx).
    Slot 0 is a dummy key (zero feature, gate 1) so attention always has at
    least one valid key even for a walker standing in empty air.
    """
    dev = centers.device
    B = len(centers)
    per_b = [sum(len(h[0][b]) for h in hist) for b in range(B)]
    T = max(per_b + [0]) + 1

    tok_feat = torch.zeros(B, T, 512, device=dev)
    tok_rel = torch.zeros(B, T, 3, device=dev)
    gate = torch.zeros(B, T, device=dev)
    age = torch.zeros(B, T, dtype=torch.long, device=dev)
    mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
    mask[:, 0] = True
    gate[:, 0] = 1.0

    for b in range(B):
        o = 1
        for a, (feats, coords) in enumerate(reversed(hist)):   # a=0 -> newest
            n = len(feats[b])
            if n == 0:
                continue
            rel = coords[b] - centers[b][None, :]              # grad -> center
            tok_feat[b, o:o + n] = feats[b]
            tok_rel[b, o:o + n] = rel
            gate[b, o:o + n] = (1.0 - rel.norm(dim=-1) / r_ctx).clamp_min(0.0)
            age[b, o:o + n] = a
            mask[b, o:o + n] = True
            o += n
    return tok_feat, tok_rel, gate, age, mask


# ---------------------------------------------------------------------------
# the decoder
# ---------------------------------------------------------------------------
class GeoWalkerDecoder(nn.Module):
    """One learned query, `blocks` rounds of cross-attention over the context
    tokens, two read-outs:

      offset — the single point the model predicts, as a displacement from the
               current center (zero-initialised: a fresh walker stands still);
      dist   — log1p of the through-cloud distance at the current center, which
               is what scores an endpoint at inference time.

    The membership gate enters as an additive attention bias log(gate), so a
    token's influence fades smoothly to zero as it leaves the context radius
    instead of popping in and out.
    """

    def __init__(self, feat_dim=512, d=256, heads=8, blocks=3, mem=MEM,
                 n_steps=N_STEPS):
        super().__init__()
        self.d, self.heads, self.blocks = d, heads, blocks
        self.tok = nn.Linear(feat_dim, d)
        self.pos = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.age = nn.Embedding(mem + 1, d)
        self.step = nn.Embedding(n_steps + 1, d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        mk = lambda: nn.ModuleList(nn.Linear(d, d) for _ in range(blocks))  # noqa: E731
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = mk(), mk(), mk(), mk()
        self.ln_q = nn.ModuleList(nn.LayerNorm(d) for _ in range(blocks))
        self.ln_kv = nn.ModuleList(nn.LayerNorm(d) for _ in range(blocks))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                          nn.Linear(4 * d, d)) for _ in range(blocks))
        self.out_ln = nn.LayerNorm(d)
        self.offset_head = nn.Linear(d, 3)
        self.dist_head = nn.Linear(d, 1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, tok_feat, tok_rel, gate, age, mask, step):
        B, T, _ = tok_feat.shape
        rel_n = tok_rel.norm(dim=-1, keepdim=True)
        x = (self.tok(tok_feat)
             + self.pos(torch.cat([tok_rel, rel_n, gate[..., None]], -1))
             + self.age(age))
        bias = torch.log(gate.clamp_min(1e-6))[:, None, None, :]      # B,1,1,T
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
        step_id = torch.full((B,), int(step), dtype=torch.long, device=x.device)
        q = self.query.expand(B, 1, self.d) + self.step(step_id)[:, None, :]
        hd = self.d // self.heads
        for i in range(self.blocks):
            kv = self.ln_kv[i](x)
            qq = self.q_proj[i](self.ln_q[i](q)).view(B, 1, self.heads, hd).transpose(1, 2)
            kk = self.k_proj[i](kv).view(B, T, self.heads, hd).transpose(1, 2)
            vv = self.v_proj[i](kv).view(B, T, self.heads, hd).transpose(1, 2)
            att = ((qq @ kk.transpose(-1, -2)) / math.sqrt(hd) + bias).softmax(-1)
            q = q + self.o_proj[i]((att @ vv).transpose(1, 2).reshape(B, 1, self.d))
            q = q + self.ffn[i](q)
        h = self.out_ln(q[:, 0])
        return self.offset_head(h), self.dist_head(h)[:, 0]


def clamp_step(delta, max_step=MAX_STEP):
    """Cap the step length without killing its direction (identity below the
    cap). An uncapped offset head is how walkers escaped 100 m last time."""
    n = delta.norm(dim=-1, keepdim=True)
    return delta * (max_step / n.clamp_min(max_step))


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
def rollout(dec, enc, scene, field, dem, seeds, n_steps=N_STEPS, train=True,
            record=False):
    """Walk `seeds` (B,3) for n_steps, scoring every simulated step.

    field=None -> pure inference (no labels needed, no loss).
    record=True -> info["traj"] holds the (n_steps+1, B, 3) path, for figures.
    Returns (loss, end_centers, end_score, info).
    """
    c = seeds.clone()
    B = len(seeds)
    hist, l_move, l_aux = [], [], []
    d_geo, d_dens, reach, g0 = [], [], None, None
    step_norms, ball_n, n_empty = [], [], 0
    g_pred = torch.zeros(B, device=seeds.device)

    traj = []
    for k in range(n_steps + 1):
        if record:
            traj.append(c.detach().cpu().numpy().copy())
        if field is not None:
            gbar, dens, has_pts, finite, npts = field_terms(field, c)
            n_empty += int((~has_pts).sum().item())
            ball_n.append(float(npts.float().mean()))
            if k == 0:
                # A walker whose nearest base is further than n_steps of walking
                # cannot reach it, and a 2 m sphere holds no hint of which way to
                # go at 40 m. Its geodesic term is unlearnable noise that would
                # set the loss magnitude; it still trains density + read-out.
                reach = finite & (gbar.detach() <= REACH_M)
                g0 = gbar.detach()
            else:
                w = GAMMA ** (n_steps - k)
                m_g = (reach & finite).float()
                m_d = has_pts.float()
                # smooth-L1 against 0: quadratic inside HUBER_M, then linear in
                # metres, so a far walker contributes a bounded gradient.
                hub = F.smooth_l1_loss(gbar, torch.zeros_like(gbar),
                                       beta=HUBER_M, reduction="none")
                t_g = (hub * m_g).sum() / m_g.sum().clamp_min(1.0)
                t_d = (dens * m_d).sum() / m_d.sum().clamp_min(1.0)
                l_move.append(w * (t_g + W_DENS * t_d))
                d_geo.append(t_g.detach())
                d_dens.append(t_d.detach())

        feats, coords = encode_spheres(enc, scene, c)
        hist.append((feats, coords))
        if len(hist) > MEM + 1:
            hist.pop(0)
        tok_feat, tok_rel, gate, age, mask = pad_context(hist, c)
        delta, g_pred = dec(tok_feat, tok_rel, gate, age, mask, step=k)

        if field is not None:
            # read-out head: regress the MEASURED field value, detached, so this
            # term cannot steer the walk — it only learns to report where it is.
            m = finite.float()
            tgt = torch.log1p(gbar.detach().clamp_min(0.0))
            per = F.smooth_l1_loss(g_pred, tgt, reduction="none")
            l_aux.append((GAMMA ** (n_steps - k))
                         * (per * m).sum() / m.sum().clamp_min(1.0))

        if k == n_steps:
            break

        # No tokens -> no information about where a base is -> no movement.
        # (Letting the decoder emit its constant "empty" offset every step is
        # how BaseWalker produced a fixed drift that compounded into escapes.)
        has_tok = mask[:, 1:].any(dim=1).float()[:, None]
        delta = clamp_step(delta * has_tok)
        step_norms.append(delta.detach().norm(dim=-1).mean())
        c = c + delta
        if DEM_PROJECT:
            c = dem.project(c)
        # Bound the through-time recurrence: dc_{k+1}/dc_k can have norm > 1 and
        # an 8-long product explodes. Truncating keeps the local credit the
        # membership gate makes meaningful and caps the blow-up.
        if train and TBPTT > 0 and (k + 1) % TBPTT == 0:
            c = c.detach()
        if not train:
            c = c.detach()

    score = torch.exp(-torch.expm1(g_pred).clamp_min(0.0) / SCORE_TAU)
    info = dict(step=float(torch.stack(step_norms).mean()) if step_norms else 0.0,
                n=B, n_empty=n_empty)
    if record:
        # the k == n_steps pass already recorded the final center
        info["traj"] = np.stack(traj)
    loss = None
    if field is not None:
        zero = seeds.new_zeros(())
        l_m = torch.stack(l_move).sum() if l_move else zero      # n_steps == 0
        l_a = torch.stack(l_aux).sum() if l_aux else zero
        loss = l_m + W_AUX * l_a
        with torch.no_grad():
            gend, _, _, fin_end, _ = field_terms(field, c)
        info.update(ball=float(np.mean(ball_n)) if ball_n else 0.0,
                    geo=float(torch.stack(d_geo).sum()) if d_geo else 0.0,
                    dens=float(torch.stack(d_dens).sum()) if d_dens else 0.0,
                    aux=float(l_a.detach()),
                    n_reach=int(reach.sum()),
                    g0=float(g0[reach].median()) if int(reach.sum()) else float("nan"),
                    g_end=float(gend[reach & fin_end].median())
                    if int((reach & fin_end).sum()) else float("nan"))
    return loss, c, score, info


@torch.no_grad()
def detect(dec, enc, scene, dem, seeds_np, batch=256, n_steps=N_STEPS,
           device="cuda"):
    """Walk every seed to its endpoint. Returns (endpoints, scores) as numpy.

    The score is the read-out head's own estimate of the through-cloud distance
    at the endpoint, squashed to (0,1]: 1 means "I am standing on a tree base".
    """
    ends, scores = [], []
    for s in range(0, len(seeds_np), batch):
        seeds = torch.from_numpy(seeds_np[s:s + batch].astype(np.float32)).to(device)
        seeds = dem.project(seeds).detach()   # same start as training
        _, c, sc, _ = rollout(dec, enc, scene, None, dem, seeds,
                              n_steps=n_steps, train=False)
        ends.append(c.cpu().numpy())
        scores.append(sc.cpu().numpy())
    return np.concatenate(ends), np.concatenate(scores)
