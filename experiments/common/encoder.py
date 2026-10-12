"""The frozen Sonata encoder, and reading it densely.

Both tracks here use the same pretrained weights with the same input
convention, so the loader lives in one place: a crop that was centre-shifted
differently, or normalised differently, is a crop the pretrained model has
never seen, and the failure is silent — features that look fine and mean
nothing.

`load_encoder` returns it frozen and in eval mode. `upcast` is the dense
read-out: Sonata's forward returns the COARSEST level, and the documented way
back to per-point features is to concatenate each level onto its parent through
the inverse pooling map. A model that only needs one token per crop (a walker
asking "which way from here") ignores this; a model that needs an opinion from
every point cannot.
"""
from __future__ import annotations

import torch

READOUT_NOTE: list[str] = []


def load_encoder(device="cuda"):
    import sonata
    model = sonata.load("sonata", repo_id="facebook/sonata",
                        custom_config=dict(enc_patch_size=[1024] * 5,
                                           enable_flash=False)).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def upcast(point, want_xyz, note=None):
    """Coarse Sonata tokens -> one feature vector per input point.

    If a build does not expose the pooling chain, or hands back a different
    number of points than it was given, we fall back to the nearest token:
    coarser features, but the stage runs instead of dying on a library detail.
    Which path ran is printed once into `note`.
    """
    note = READOUT_NOTE if note is None else note
    depth = 0
    while hasattr(point, "keys") and "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
        depth += 1

    feat, coord = point.feat.float(), point.coord.float()
    if len(feat) == len(want_xyz) and torch.allclose(coord, want_xyz, atol=1e-4):
        if not note:
            note.append("chain")
            print(f"[enc] dense read-out: pooling chain, depth {depth}, "
                  f"{feat.shape[1]}-d per point")
        return feat

    from scipy.spatial import cKDTree
    if not note:
        note.append("nn")
        print(f"[enc] dense read-out: NEAREST-TOKEN fallback (chain depth "
              f"{depth}, {len(feat):,} tokens for {len(want_xyz):,} points, "
              f"{feat.shape[1]}-d) — the pooling chain was unusable, features "
              f"are token-resolution")
    j = cKDTree(coord.detach().cpu().numpy()).query(
        want_xyz.detach().cpu().numpy())[1]
    return feat[torch.as_tensor(j, device=feat.device)]
