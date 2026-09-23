#!/usr/bin/env python3
"""CPU regression tests for the GeoWalker stage — the two things that are easy
to get silently wrong:

  1. the through-cloud distance field really goes THROUGH the cloud (it detours
     around gaps, it is never shorter than the straight line, and it refuses to
     cross empty air at all);
  2. the two training terms are differentiable in the walker's position and
     their gradients point the right way (downhill on the field, back toward
     structure when the walker steps into air).

No GPU, no data, no Sonata — everything runs on synthetic geometry:

    python tests/test_geowalker.py      # or: pytest tests/
"""
import os
import sys
from pathlib import Path

# geodesic.py reads its graph parameters from the environment at import time.
os.environ.setdefault("GW_VOXEL", "0.1")
os.environ.setdefault("GW_MAX_EDGE", "0.2")
os.environ.setdefault("GW_KNN", "16")

import numpy as np
import torch

# Works both in a repo checkout (experiments/{basewalker,geowalker}) and inside
# the container image, where the same modules live at /app and /app/geowalker.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (ROOT, ROOT / "basewalker", ROOT / "geowalker",
                                 Path("/app"), Path("/app/geowalker"))
                if p.is_dir()]

import geodesic as G          # noqa: E402
import model as M             # noqa: E402

V = float(os.environ["GW_VOXEL"])


# ---------------------------------------------------------------------------
# a scene you have to walk around: a 10 x 4 m floor with a 1 m trench cut
# through it, bridged only along the far edge (y > 3).
# ---------------------------------------------------------------------------
def _scene():
    gx, gy = np.meshgrid(np.arange(0, 10, V / 2), np.arange(0, 4, V / 2))
    p = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    trench = (p[:, 0] > 4.5) & (p[:, 0] < 5.5) & (p[:, 1] < 3.0)
    floor = p[~trench]
    island = np.column_stack([                       # unreachable: 5 m of air
        np.full(400, 20.0) + np.random.rand(400) * 0.5,
        np.random.rand(400) * 0.5, np.zeros(400)])
    return np.vstack([floor, island])


def _field(pts, bases):
    from scipy.spatial import cKDTree
    nodes = (G._unique_voxels(np.floor(pts / V)).astype(np.float64) + 0.5) * V
    kd = cKDTree(nodes)
    graph = G.build_graph(nodes, kd)
    g = G.geodesic(graph, G.seed_nodes(kd, bases, "test"), "test")
    return nodes, g


def test_field_detours_around_the_gap():
    """A point straight across the trench must be reached the long way round."""
    pts = _scene()
    base = np.array([[0.5, 0.5, 0.0]])
    nodes, g = _field(pts, base)
    from scipy.spatial import cKDTree
    kd = cKDTree(nodes)
    far = np.array([9.5, 0.5, 0.0])
    gi = g[kd.query(far)[1]]
    euclid = np.linalg.norm(far - base[0])
    assert np.isfinite(gi), "the far side of the bridge must still be reachable"
    assert gi > euclid * 1.1, (gi, euclid)     # forced detour via the bridge
    # and the detour is the RIGHT length: the shortest legal path runs
    # (0.5,0.5) -> (4.5,3) -> (5.5,3) -> (9.5,0.5), i.e. 2*hypot(4,2.5) + 1
    detour = 2 * float(np.hypot(4.0, 2.5)) + 1.0
    assert abs(gi - detour) < 0.8, (gi, detour)


def test_field_never_shorter_than_the_straight_line():
    pts = _scene()
    base = np.array([[0.5, 0.5, 0.0]])
    nodes, g = _field(pts, base)
    fin = np.isfinite(g)
    euclid = np.linalg.norm(nodes[fin] - base[0][None, :], axis=1)
    # the seed ball has radius BASE_R, so allow exactly that much slack
    assert (g[fin] >= euclid - G.BASE_R - 1e-6).all(), \
        float((euclid - g[fin]).max())


def test_field_does_not_cross_empty_air():
    """The island 10 m away shares no path with the floor: it stays +inf."""
    pts = _scene()
    nodes, g = _field(pts, np.array([[0.5, 0.5, 0.0]]))
    island = nodes[:, 0] > 15.0
    assert island.sum() > 0
    assert not np.isfinite(g[island]).any()


def test_unlabelled_scene_reports_zero_reachable():
    """A base outside the cloud still seeds (nearest-node fallback) — it must
    not silently drop out and leave the whole field at +inf."""
    pts = _scene()
    nodes, g = _field(pts, np.array([[0.5, 0.5, 5.0]]))   # 5 m above the floor
    assert np.isfinite(g).mean() > 0.5


# ---------------------------------------------------------------------------
# the loss terms
# ---------------------------------------------------------------------------
def _store(device="cpu"):
    """A 20 m strip of cloud whose field value is simply x: walking -x is
    always the right move, and everything above z = 0 is empty air."""
    gx, gy = np.meshgrid(np.arange(0, 20, 0.1), np.arange(-1, 1, 0.1))
    xyz = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    return M.FieldStore(xyz.astype(np.float32), xyz[:, 0].astype(np.float32),
                        device=device)


def test_geodesic_term_is_the_local_field_value():
    f = _store()
    c = torch.tensor([[10.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    gbar, _, has, fin, _ = M.field_terms(f, c)
    assert has.all() and fin.all()
    assert abs(float(gbar[0]) - 10.0) < 0.1, float(gbar[0])
    assert abs(float(gbar[1]) - 4.0) < 0.1, float(gbar[1])


def test_geodesic_gradient_points_downhill():
    f = _store()
    c = torch.tensor([[10.0, 0.0, 0.0]], requires_grad=True)
    gbar, _, _, _, _ = M.field_terms(f, c)
    gbar.sum().backward()
    gx = float(c.grad[0, 0])
    assert gx > 0.05, gx            # d(gbar)/dx > 0  => descending means -x
    assert abs(float(c.grad[0, 1])) < gx * 0.5      # no sideways bias


def test_density_term_is_free_inside_structure_and_costs_in_air():
    f = _store()
    c = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 1.2]])
    _, dens, has, _, npts = M.field_terms(f, c)
    assert float(dens[0]) == 0.0, float(dens[0])    # sitting on the cloud
    assert float(dens[1]) > 0.2, float(dens[1])     # 1.2 m up in the air
    assert int(npts[0]) > int(npts[1])


def test_density_gradient_pulls_back_to_the_cloud():
    f = _store()
    c = torch.tensor([[10.0, 0.0, 1.2]], requires_grad=True)
    _, dens, _, _, _ = M.field_terms(f, c)
    dens.sum().backward()
    assert float(c.grad[0, 2]) > 0.05, float(c.grad[0, 2])   # cost rises with z


def test_empty_ball_is_the_worst_density_and_not_nan():
    f = _store()
    c = torch.tensor([[10.0, 0.0, 50.0]])           # nothing within FIELD_R
    gbar, dens, has, fin, _ = M.field_terms(f, c)
    assert not bool(has[0]) and not bool(fin[0])
    assert float(dens[0]) == 1.0
    assert torch.isfinite(gbar).all()


def test_unreachable_points_are_excluded_from_the_mean():
    """+inf field values must not poison the weighted mean with NaN."""
    f = _store()
    g = f.g.clone()
    g[::2] = float("inf")                            # half the cloud unreachable
    f.g = g
    c = torch.tensor([[10.0, 0.0, 0.0]])
    gbar, _, _, fin, _ = M.field_terms(f, c)
    assert bool(fin[0]) and torch.isfinite(gbar).all()
    assert abs(float(gbar[0]) - 10.0) < 0.2


# ---------------------------------------------------------------------------
# decoder mechanics
# ---------------------------------------------------------------------------
def test_step_clamp_caps_length_and_keeps_direction():
    d = torch.tensor([[30.0, 40.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = M.clamp_step(d, 1.0)
    assert abs(float(out[0].norm()) - 1.0) < 1e-5
    assert torch.allclose(out[0], d[0] / 50.0, atol=1e-6)   # direction kept
    assert torch.allclose(out[1], d[1])                     # below the cap: id
    assert float(out[2].norm()) == 0.0                      # no NaN at zero


def test_decoder_runs_and_starts_stationary():
    torch.manual_seed(0)
    dec = M.GeoWalkerDecoder(blocks=2)
    B, T = 3, 7
    tf = torch.randn(B, T, 512)
    rel = torch.randn(B, T, 3)
    gate = torch.rand(B, T)
    age = torch.randint(0, M.MEM + 1, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[2, 3:] = False
    delta, g_pred = dec(tf, rel, gate, age, mask, step=2)
    assert delta.shape == (B, 3) and g_pred.shape == (B,)
    assert float(delta.detach().abs().max()) == 0.0   # zero-init offset head
    assert torch.isfinite(g_pred).all()


def test_context_carries_gradient_to_the_current_center():
    """Memory tokens are re-expressed relative to the CURRENT center, so the
    whole context — not just this step's sphere — must move its gradient."""
    B, mem = 2, M.MEM
    hist = [([torch.randn(4, 512) for _ in range(B)],
             [torch.randn(4, 3) for _ in range(B)]) for _ in range(mem + 1)]
    c = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], requires_grad=True)
    tf, rel, gate, age, mask = M.pad_context(hist, c)
    assert tf.shape[1] == 4 * (mem + 1) + 1        # +1 dummy key
    assert mask[:, 0].all() and float(gate[0, 0].detach()) == 1.0
    assert set(age[0, 1:].tolist()) == set(range(mem + 1))
    (rel.sum() + gate.sum()).backward()
    assert c.grad is not None and float(c.grad.abs().sum()) > 0


def test_rollout_backpropagates_through_the_whole_walk():
    """The full loop — encode, remember, step, score — with the frozen encoder
    stubbed out. Catches the shape/grad wiring that only shows up end to end."""
    torch.manual_seed(0)
    f = _store()
    xyz = f.xyz.numpy()
    scene = M.SceneStore(xyz, np.zeros_like(xyz), device="cpu")
    dec = M.GeoWalkerDecoder(blocks=1)
    torch.nn.init.normal_(dec.offset_head.weight, std=0.05)   # make it move

    real = M.encode_spheres
    M.encode_spheres = lambda enc, sc, c, r=M.SPHERE_R: (
        [torch.randn(5, 512) for _ in range(len(c))],
        [c[b].detach()[None, :] + torch.randn(5, 3) * 0.5 for b in range(len(c))])
    try:
        seeds = torch.tensor([[10.0, 0.0, 0.0], [6.0, 0.0, 0.0], [17.0, 0.0, 0.0]])
        loss, end, score, info = M.rollout(dec, None, scene, f, None, seeds,
                                           n_steps=3)
    finally:
        M.encode_spheres = real

    assert torch.isfinite(loss), float(loss.detach())
    loss.backward()
    grads = [p.grad for p in dec.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0, "no gradient reached the decoder"
    assert end.shape == (3, 3) and score.shape == (3,)
    sc = score.detach()
    assert 0.0 <= float(sc.min()) and float(sc.max()) <= 1.0
    # seeds at field 10 / 6 / 17 m, REACH_M = 12 m -> two are learnable
    assert info["n_reach"] == 2, info
    assert info["n_empty"] == 0 and info["step"] > 0


def test_score_is_monotone_in_predicted_distance():
    g = torch.tensor([0.0, 0.5, 2.0, 10.0])
    s = torch.exp(-torch.expm1(g).clamp_min(0.0) / M.SCORE_TAU)
    assert float(s[0]) == 1.0
    assert bool((s[1:] < s[:-1]).all())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} GeoWalker tests passed.")
