"""StemNet v2 network: frozen Sonata read densely, and the trained head.

Sonata input per point: crop-local xyz (x, y to the crop's bounding-box
midpoint, z to its minimum — Sonata's own CenterShift), colour as zeros (the scan's
RGB is unreliable on many points; feeding it is a later experiment), and the
normal from the feature tiles. Its forward pass returns the coarsest level;
walking the pooling chain back up and concatenating each level onto its
parent gives one 1,232-d vector per input point (48+96+192+384+512).
As in Sonata's own GridSample, one point per 2 cm voxel goes in (augmentation
can put two in one voxel) and every point gets its voxel's vector back.

The head sees those 1,232 values plus the 12 covariance features
(standardised), and outputs per point:
  gate    a logit for "close to a tree base"
  offset  the 2-D vector to that base, zero-initialised so an untrained head
          votes for the point itself rather than somewhere random
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

GRID = float(os.environ.get("V2_GRID", "0.02"))


def load_sonata(device):
    import sonata
    m = sonata.load("sonata", repo_id="facebook/sonata",
                    custom_config=dict(enc_patch_size=[1024] * 5, enable_flash=False)).to(device)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class FakeEncoder:
    """CPU stand-in with Sonata's interface, for tests: fixed random features of
    the input, the same width as the real read-out."""

    def __init__(self, dim=1232, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(9, dim, generator=g) * 0.5
        self.dim = dim

    def dense(self, local, normal):
        x = torch.cat([local, torch.zeros_like(local), normal], 1)
        return torch.tanh(x @ self.W.to(x.device))


@torch.no_grad()
def dense_features(enc, local, normal):
    """(n, 3) crop-local xyz and normals -> (n, 1232) features, one per point."""
    if isinstance(enc, FakeEncoder):
        return enc.dense(local, normal)
    grid = torch.floor((local - local.min(0).values) / GRID).long()
    grid, inv = torch.unique(grid, dim=0, return_inverse=True)
    one = torch.empty(len(grid), dtype=torch.long, device=local.device)
    one.scatter_(0, inv, torch.arange(len(inv), device=local.device))   # a point per voxel
    local, normal = local[one], normal[one]
    feat = torch.cat([local, torch.zeros_like(local), normal], 1)
    point = enc(dict(coord=local, grid_coord=grid, feat=feat,
                     offset=torch.tensor([len(local)], device=local.device, dtype=torch.long)))
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    if len(point.feat) != len(local):
        raise RuntimeError(f"Sonata returned {len(point.feat)} points for {len(local)} — "
                           f"the dense read-out did not reach the input level")
    return point.feat.float()[inv]


def to_local(xyz):
    """Sonata's CenterShift(apply_z=True): x, y to the bounding-box midpoint,
    z to the minimum."""
    lo, hi = xyz.min(0), xyz.max(0)
    shift = np.r_[(lo[:2] + hi[:2]) / 2, lo[2]]
    return (xyz - shift).astype(np.float32)


class Head(nn.Module):
    def __init__(self, d_in, d=256):
        super().__init__()
        self.d_in = d_in
        self.trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(),
                                   nn.Linear(d, d), nn.GELU())
        self.gate = nn.Linear(d, 1)
        self.off = nn.Linear(d, 2)
        nn.init.zeros_(self.off.weight)
        nn.init.zeros_(self.off.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.gate(h)[:, 0], self.off(h)
