#!/usr/bin/env python3
"""Dense stem head on frozen Sonata: per point, is this a tree, and where is
its base?

The walker's problem was never perception. Its misses on the attributed patch
reached within a metre of 94% of the trees they missed and stopped short, and
the median detection cluster had ONE vote out of the five walkers that passed
near each tree. Eight sequential sphere encodes per walker buy a single opinion
about one trajectory; the information to localise a stem is already in the
points around it.

So: one forward pass, every point votes.

  * the encoder is the same frozen Sonata, but read densely — the coarse tokens
    are upcast back to the input points through the pooling chain, which is the
    standard Sonata linear-probe read-out.
  * two heads per point: a logit for "I belong to a tree stem" and a 2-D
    horizontal offset to that stem's base.
  * detections are peaks in the p-weighted histogram of SHIFTED points. A
    cluster's vote count is the number of points that agreed, so consensus is
    dense by construction instead of something to hope for.

Why these targets and not something more ambitious: the only per-tree truth in
this project is a set of RTK stem bases. There is no per-point instance label
and no semantic label on this cloud, so "is a tree" is operationalised as "lies
within GW_SN_NEAR_R metres HORIZONTALLY of a surveyed base" — which, restricted
to the breast-height band, is a trunk and its immediate surroundings. That is a
narrower claim than tree/not-tree, and it is the one the
data can actually support — and it is exactly what we need, since a stem base
is what gets reported.

Horizontal, not 3-D: a trunk point 3 m up is 3 m from its own base in 3-D but
~0.15 m from it in plan view. The base's z comes from the DEM, so the head
never has to predict it, and the target is a short 2-D vector with far less
variance than the walker's 8 m trajectory.

Window geometry: the scene is processed in WIN-metre squares with a PAD-metre
context ring that is encoded but never predicted on, so a trunk at a window
edge is still seen inside somebody's neighbourhood. Inner squares tile exactly,
so no point is counted twice.

Shared by stem_train.py and stem_infer.py; nothing here touches the filesystem
at import.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402
from walker import SceneStore, Dem, load_scene, load_encoder  # noqa: E402,F401
from walker import match_metrics, match_pairs  # noqa: E402,F401

BW = C.DATA / "basewalker"
GW = C.DATA / "geowalker"
CKPT = C.MODELS / "geowalker_stemnet.pth"
LAST_CKPT = C.MODELS / "geowalker_stemnet_last.pth"

GRID = float(os.environ.get("GRID_SIZE", "0.05"))
VOXEL = float(os.environ.get("GW_SN_VOXEL", str(GRID)))   # band dedup resolution
BAND_LO = float(os.environ.get("GW_SN_BAND_LO", "0.3"))   # m above the DEM
BAND_HI = float(os.environ.get("GW_SN_BAND_HI", "4.0"))
NEAR_R = float(os.environ.get("GW_SN_NEAR_R", "1.0"))     # positive-label radius
WIN = float(os.environ.get("GW_SN_WIN", "16.0"))          # predicted square, m
PAD = float(os.environ.get("GW_SN_PAD", "2.0"))           # encoded context ring
PTS_CAP = int(float(os.environ.get("GW_SN_PTS", "250000")))
P_THRESH = float(os.environ.get("GW_SN_P", "0.5"))        # a point votes above this
VOTE_CELL = float(os.environ.get("GW_SN_VOTE_CELL", "0.1"))
VOTE_SEP = float(os.environ.get("GW_SN_VOTE_SEP", "0.6"))  # two stems this far apart
VOTE_R = float(os.environ.get("GW_SN_VOTE_R", "0.3"))     # voters pooled per peak
VOTE_MIN = float(os.environ.get("GW_SN_VOTE_MIN", "6.0"))  # minimum vote mass
VAL_MARGIN = float(os.environ.get("BW_VAL_MARGIN", "10.0"))
DEVICE = os.environ.get("GW_DEVICE", "cuda")


# ---------------------------------------------------------------------------
# reading Sonata densely
# ---------------------------------------------------------------------------
_UPCAST_NOTE = []


def upcast(point, want_xyz):
    """Coarse Sonata tokens -> one feature vector per input point.

    The pooling chain is the documented Sonata read-out: each level's features
    are concatenated onto its parent through the inverse pooling map, ending at
    the input points. If a build does not expose the chain, or returns a
    different number of points than it was given, we fall back to the nearest
    token — coarser, but it keeps the stage runnable instead of crashing on a
    library detail. Which path ran is printed once.
    """
    depth = 0
    while hasattr(point, "keys") and "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
        depth += 1

    feat, coord = point.feat.float(), point.coord.float()
    if len(feat) == len(want_xyz) and torch.allclose(coord, want_xyz, atol=1e-4):
        if not _UPCAST_NOTE:
            _UPCAST_NOTE.append("chain")
            print(f"[gw-sn] dense read-out: pooling chain, depth {depth}, "
                  f"{feat.shape[1]}-d per point")
        return feat

    # nearest-token fallback (scipy on the CPU: 1-NN over a window's tokens is
    # milliseconds, and this path is not expected to run at all)
    from scipy.spatial import cKDTree
    if not _UPCAST_NOTE:
        _UPCAST_NOTE.append("nn")
        print(f"[gw-sn] dense read-out: NEAREST-TOKEN fallback "
              f"(chain depth {depth}, {len(feat):,} tokens for "
              f"{len(want_xyz):,} points, {feat.shape[1]}-d) — the pooling "
              f"chain was unusable, features are token-resolution")
    j = cKDTree(coord.detach().cpu().numpy()).query(
        want_xyz.detach().cpu().numpy())[1]
    return feat[torch.as_tensor(j, device=feat.device)]


@torch.no_grad()
def encode_points(enc, scene: SceneStore, idx: torch.Tensor):
    """Per-point features for the scene points `idx`, as one Sonata crop.

    The crop is centre-shifted exactly as the pretrained CenterShift does (xy to
    the mean, z to the minimum) — the same convention the walker's sphere crops
    use, so a checkpoint trained here sees the inputs it was trained on.
    """
    pts = scene.xyz[idx]
    shift = torch.cat([pts[:, :2].mean(0), pts[:, 2].min()[None]])
    local = pts - shift[None, :]
    feat = torch.cat([local, torch.zeros_like(local), scene.normal[idx]], 1)
    data = dict(
        coord=local,
        grid_coord=((local - local.min(0).values[None, :]) / GRID).floor().long(),
        feat=feat,
        offset=torch.tensor([len(idx)], device=scene.device, dtype=torch.long),
    )
    return upcast(enc(data), local)


class StemHead(nn.Module):
    """Per-point tree logit + horizontal offset to the stem base.

    The offset layer starts at zero: before it has learned anything every point
    votes for itself, which is a weak detector rather than a random one.
    """

    def __init__(self, feat_dim: int, d: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU())
        self.logit = nn.Linear(d, 1)
        self.off = nn.Linear(d, 2)
        nn.init.zeros_(self.off.weight)
        nn.init.zeros_(self.off.bias)

    def forward(self, feat):
        h = self.trunk(feat)
        return self.logit(h)[:, 0], self.off(h)


# ---------------------------------------------------------------------------
# the stem band: the only points this model ever sees
# ---------------------------------------------------------------------------
class Buckets:
    """2-D bucket index over a point set, for rectangle queries."""

    def __init__(self, xy: np.ndarray, cell: float):
        self.cell = cell
        self.x0, self.y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
        self.nx = int((xy[:, 0].max() - self.x0) / cell) + 1
        self.ny = int((xy[:, 1].max() - self.y0) / cell) + 1
        ix = ((xy[:, 0] - self.x0) / cell).astype(np.int64).clip(0, self.nx - 1)
        iy = ((xy[:, 1] - self.y0) / cell).astype(np.int64).clip(0, self.ny - 1)
        cid = iy * self.nx + ix
        self.order = np.argsort(cid, kind="stable")
        self.start = np.searchsorted(cid[self.order], np.arange(self.nx * self.ny))
        self.start = np.r_[self.start, len(cid)]

    def rect(self, x0, x1, y0, y1):
        """Positions (into the indexed arrays) whose bucket meets the rectangle.

        Buckets, so the result is a superset of the rectangle — callers mask
        exactly. Returning a superset is the point: the caller usually wants a
        padded crop anyway.
        """
        ix0 = max(0, int((x0 - self.x0) / self.cell))
        ix1 = min(self.nx - 1, int((x1 - self.x0) / self.cell))
        iy0 = max(0, int((y0 - self.y0) / self.cell))
        iy1 = min(self.ny - 1, int((y1 - self.y0) / self.cell))
        if ix1 < ix0 or iy1 < iy0:
            return np.zeros(0, np.int64)
        out = [self.order[self.start[iy * self.nx + ix0]:
                          self.start[iy * self.nx + ix1 + 1]]
               for iy in range(iy0, iy1 + 1)]
        return np.concatenate(out) if out else np.zeros(0, np.int64)


def voxel_unique(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """One representative point per `voxel` cube — a lossless reduction as far
    as the encoder is concerned, since it serialises on a GRID-metre lattice
    anyway, and it is what bounds the cost of a window."""
    k = np.floor(xyz / voxel).astype(np.int64)
    k -= k.min(0)
    n = k.max(0) + 1
    flat = k[:, 0] + n[0] * (k[:, 1] + n[1] * k[:, 2])
    return np.unique(flat, return_index=True)[1]


class Band:
    """The breast-height slice of the scene, deduplicated and indexed.

    Everything outside it is dropped before the model ever runs: the ground
    carries no stem, the crown cannot say where a trunk meets the soil, and
    together they are most of the cloud. This is why one pass over the scene is
    affordable where 199,278 sphere encodes were not.
    """

    def __init__(self, scene: SceneStore, dem: Dem, lo=BAND_LO, hi=BAND_HI,
                 voxel=VOXEL, cell=None, chunk=1 << 20):
        xyz = scene.xyz
        keep = []
        for s in range(0, len(xyz), chunk):
            blk = xyz[s:s + chunk]
            h = blk[:, 2] - dem.height(blk[:, :2])
            keep.append(torch.nonzero((h >= lo) & (h <= hi), as_tuple=True)[0] + s)
        keep = torch.cat(keep) if keep else torch.zeros(0, dtype=torch.long,
                                                        device=xyz.device)
        pts = xyz[keep].double().cpu().numpy()
        u = voxel_unique(pts, voxel)
        self.idx = keep[torch.as_tensor(u, device=xyz.device)]
        self.xyz = pts[u]
        self.n_raw = int(len(keep))
        self.buckets = Buckets(self.xyz[:, :2], cell or max(WIN / 2.0, 1.0))
        print(f"[gw-sn] band {lo}-{hi} m: {len(xyz):,} scene pts -> "
              f"{self.n_raw:,} in band -> {len(self.xyz):,} after {voxel} m "
              f"dedup ({len(self.xyz)/max(len(xyz),1)*100:.1f}% of the scene)")

    def window(self, cx, cy, win=WIN, pad=PAD):
        """(crop positions, inner mask) for the square centred at (cx, cy).

        The crop carries a `pad` ring of context that is encoded but not
        predicted on; `inner` selects the square that is actually this
        window's responsibility, so stepping by `win` tiles the scene exactly.
        """
        h = win / 2.0 + pad
        pos = self.buckets.rect(cx - h, cx + h, cy - h, cy + h)
        if not len(pos):
            return pos, np.zeros(0, bool)
        xy = self.xyz[pos, :2]
        crop = ((np.abs(xy[:, 0] - cx) <= h) & (np.abs(xy[:, 1] - cy) <= h))
        pos = pos[crop]
        xy = xy[crop]
        inner = ((np.abs(xy[:, 0] - cx) <= win / 2.0)
                 & (np.abs(xy[:, 1] - cy) <= win / 2.0))
        return pos, inner


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def nearest_base(xy: torch.Tensor, bases_xy: torch.Tensor, chunk: int = 1 << 14):
    """Horizontal distance to the nearest labelled base, and which one.

    Chunked so a 250k-point window against 758 bases never allocates the full
    cross product at once.
    """
    d_out = torch.empty(len(xy), device=xy.device)
    j_out = torch.empty(len(xy), dtype=torch.long, device=xy.device)
    for s in range(0, len(xy), chunk):
        d = torch.cdist(xy[s:s + chunk], bases_xy)
        m, j = d.min(1)
        d_out[s:s + chunk] = m
        j_out[s:s + chunk] = j
    return d_out, j_out


def targets(xy: torch.Tensor, bases_xy: torch.Tensor, near_r=NEAR_R):
    """(is_stem, offset_to_base) for a window's points."""
    d, j = nearest_base(xy, bases_xy)
    return d <= near_r, bases_xy[j] - xy


# ---------------------------------------------------------------------------
# votes -> detections
# ---------------------------------------------------------------------------
def vote(xy: np.ndarray, p: np.ndarray, off: np.ndarray, p_thresh=P_THRESH,
         cell=VOTE_CELL, sep=VOTE_SEP, pool_r=VOTE_R, min_mass=VOTE_MIN):
    """Peaks in the p-weighted histogram of shifted points.

    Returns (xy, mass, n_votes). `mass` is the summed confidence of the points
    that agreed on a location and is the detection's score; `n_votes` is how
    many points those were. This is the number the walker could never produce:
    its median cluster had one vote, here every stem point in the band is an
    independent opinion about the same base.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from baseline import peaks

    k = p >= p_thresh
    if k.sum() < 1:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)
    v = xy[k] + off[k]
    w = p[k].astype(np.float64)

    x0, y0 = v[:, 0].min() - 1.0, v[:, 1].min() - 1.0
    nx = int((v[:, 0].max() + 1.0 - x0) / cell) + 1
    ny = int((v[:, 1].max() + 1.0 - y0) / cell) + 1
    acc = np.zeros((ny, nx), np.float64)
    ix = ((v[:, 0] - x0) / cell).astype(np.int64).clip(0, nx - 1)
    iy = ((v[:, 1] - y0) / cell).astype(np.int64).clip(0, ny - 1)
    np.add.at(acc, (iy, ix), w)
    # a stem's votes land in a puddle a few cells wide, not one cell
    sm = ndimage.uniform_filter(acc, size=3) * 9.0
    cand, _ = peaks(sm, cell, sep, min_mass, x0, y0)
    if not len(cand):
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)

    # refine each peak to the weighted centroid of the voters it pooled: the
    # raster cell centre is quantised to `cell`, the centroid is not
    kd = cKDTree(v)
    out = []
    for c in cand:
        hit = kd.query_ball_point(c, pool_r)
        if not hit:
            continue
        ww = w[hit]
        out.append((float((v[hit, 0] * ww).sum() / ww.sum()),
                    float((v[hit, 1] * ww).sum() / ww.sum()),
                    float(ww.sum()), len(hit)))
    if not out:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)
    out = np.array(out)
    out = out[out[:, 2] >= min_mass]
    if not len(out):
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)

    # two peaks can refine onto the same stem; keep the heavier
    order = np.argsort(-out[:, 2])
    kept = []
    for i in order:
        if all(((out[i, :2] - out[j, :2]) ** 2).sum() > sep * sep for j in kept):
            kept.append(i)
    out = out[kept]
    return out[:, :2], out[:, 2], out[:, 3].astype(np.int64)


def split_x(labels=None):
    """The train/val boundary in local x, as prep_scene recorded it."""
    lab = np.load(BW / "labels.npz") if labels is None else labels
    return float(lab["split_x"])


def tile_centres(x0, x1, y0, y1, win=WIN):
    """Inner-square centres that tile the rectangle exactly."""
    nx = max(1, int(np.ceil((x1 - x0) / win)))
    ny = max(1, int(np.ceil((y1 - y0) / win)))
    cx = x0 + (np.arange(nx) + 0.5) * win
    cy = y0 + (np.arange(ny) + 0.5) * win
    gx, gy = np.meshgrid(cx, cy)
    return np.column_stack([gx.ravel(), gy.ravel()])
