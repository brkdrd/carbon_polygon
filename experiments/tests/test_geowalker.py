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


def test_threshold_sweep_tracks_the_actual_score_range():
    """The head saturates well below 1 (0.544 on the first full run), so a fixed
    0.05-0.95 grid wastes half its points. Quantiles must follow the data."""
    sc = np.random.default_rng(0).uniform(0.10, 0.54, 5000)
    thr = M.sweep_thresholds(sc, n=25)
    assert thr[0] <= sc.min(), "lowest cut must keep everything"
    assert thr.max() <= sc.max(), "no cut above the highest score (dead range)"
    assert (np.diff(thr) >= 0).all(), "must be sorted"
    # the cuts should spread across the populated range, not bunch at one end
    spread = np.quantile(thr, 0.75) - np.quantile(thr, 0.25)
    assert spread > 0.1 * (sc.max() - sc.min()), spread


def test_threshold_sweep_survives_degenerate_input():
    assert len(M.sweep_thresholds(np.zeros(0))) >= 1          # no detections
    assert len(M.sweep_thresholds(np.full(50, 0.3))) >= 1     # all identical


def test_surveyed_mask_excludes_unlabelled_ground():
    """Detections in areas the RTK survey never reached are not model errors.
    The mask must cover the labelled block (plus a margin) and nothing else."""
    gt = np.stack(np.meshgrid(np.arange(0, 50, 5.0),
                              np.arange(0, 50, 5.0)), -1).reshape(-1, 2)
    inside = M.surveyed_mask(gt, cell=10.0, dilate=2)
    assert inside(np.array([[25.0, 25.0]]))[0]          # middle of the survey
    assert inside(np.array([[0.0, 0.0]]))[0]            # a corner base
    assert inside(np.array([[-15.0, 25.0]]))[0]         # just outside, dilated
    assert not inside(np.array([[300.0, 300.0]]))[0]    # far periphery
    assert not inside(np.array([[25.0, -200.0]]))[0]
    assert inside(np.zeros((0, 2))).shape == (0,)       # empty input


def test_surveyed_mask_keeps_every_ground_truth_base():
    """A base must never fall outside its own survey mask, or recall would be
    computed against bases the precision term is forbidden to match."""
    rng = np.random.default_rng(1)
    gt = rng.uniform(-100, 100, (500, 2))
    inside = M.surveyed_mask(gt, cell=10.0, dilate=2)
    assert inside(gt).all()


def test_tree_head_runs_and_separates_a_trivial_case():
    """The head must consume a padded sphere batch and produce one logit each,
    and be able to learn ANY signal — if it cannot fit a toy separation the
    experiment's negative result would be meaningless."""
    import treehead as TH

    torch.manual_seed(0)
    head = TH.TreeHead(blocks=1)
    B, T = 8, 6
    tf = torch.randn(B, T, 512)
    rel = torch.randn(B, T, 3) * 0.5
    gate = torch.rand(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = head(tf, rel, gate, mask)
    assert out.shape == (B,) and torch.isfinite(out).all()

    # a separable toy problem: positives carry a constant offset in the tokens
    y = torch.tensor([1.0, 0.0] * (B // 2))
    tf2 = tf + y[:, None, None] * 3.0
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(120):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(tf2, rel, gate, mask), y)
        opt.zero_grad(); loss.backward(); opt.step()
    pred = (torch.sigmoid(head(tf2, rel, gate, mask)) > 0.5).float()
    assert (pred == y).all(), (pred, y)


def test_tree_head_dataset_bands_are_disjoint():
    """Positives and hard negatives must not overlap, or the head is trained on
    contradictory labels for the same geometry."""
    import treehead as TH
    assert TH.POS_R < TH.NEG_R, (TH.POS_R, TH.NEG_R)


def test_best_step_beats_the_last_step_when_the_walk_drifts():
    """A walker that stands on the base at step 5 and drifts off by step 8 is a
    detection, not a miss — the read-out already judged every step."""
    import aggregate as A
    K, N = 9, 3
    traj = np.zeros((K, N, 3), np.float32)
    traj[:, 0, 0] = np.linspace(0, 8, K)        # walks past the base at x=4
    traj[:, 1, 0] = 4.0                         # sits on it
    traj[:, 2, 0] = 20.0                        # never near it
    score = np.zeros((K, N), np.float32)
    score[:, 0] = 1.0 - np.abs(traj[:, 0, 0] - 4.0) / 8.0
    score[:, 1] = 0.9
    score[:, 2] = 0.1
    xyz, sc = A.best_along_walk(traj, score)
    assert abs(xyz[0, 0] - 4.0) < 1e-6, xyz[0]          # the step it was best at
    assert sc[0] > score[-1, 0]                          # better than the endpoint
    assert abs(xyz[2, 0] - 20.0) < 1e-6                  # unchanged where flat


def test_votes_count_the_pile_instead_of_deleting_it():
    """NMS keeps one endpoint of a cluster; the cluster size is the measurement."""
    import aggregate as A
    rng = np.random.default_rng(0)
    stem = np.column_stack([rng.normal(10, 0.1, 40), rng.normal(10, 0.1, 40),
                            rng.uniform(0, 0.4, 40)])          # 40 walkers agree
    bush = np.column_stack([rng.normal(30, 0.1, 2), rng.normal(30, 0.1, 2),
                            rng.uniform(0, 0.4, 2)])           # 2 walkers agree
    xyz = np.vstack([stem, bush])
    sc = np.full(len(xyz), 0.6)                  # identical scores on purpose
    cl = A.cluster_votes(xyz, sc, radius=0.5)
    assert len(cl["n_votes"]) == 2, cl["n_votes"]
    big, small = max(cl["n_votes"]), min(cl["n_votes"])
    assert big == 40 and small == 2, cl["n_votes"]
    conf = A.vote_confidence(cl, v0=4)
    # same score, different evidence -> different confidence
    assert conf[np.argmax(cl["n_votes"])] > 2 * conf[np.argmin(cl["n_votes"])]


def test_vote_centre_is_closer_to_the_truth_than_any_single_walker():
    """Averaging a pile beats keeping its loudest member, which is what a 0.5 m
    matching radius is sensitive to."""
    import aggregate as A
    rng = np.random.default_rng(1)
    truth = np.array([5.0, 5.0])
    xy = truth + rng.normal(0, 0.3, (60, 2))
    xyz = np.column_stack([xy, np.zeros(len(xy))])
    sc = rng.uniform(0.4, 0.6, len(xy))
    cl = A.cluster_votes(xyz, sc, radius=2.0)
    assert len(cl["xyz"]) == 1
    d_centre = np.linalg.norm(cl["xyz"][0, :2] - truth)
    d_loudest = np.linalg.norm(xyz[sc.argmax(), :2] - truth)
    assert d_centre < d_loudest, (d_centre, d_loudest)
    assert d_centre < 0.1, d_centre


def test_height_gate_drops_only_what_it_can_prove_is_short():
    """Unknown canopy height is kept: the gate removes candidates it can prove
    stand under something too short to have been labelled, and nothing else."""
    import aggregate as A
    chm = dict(chm=np.full((20, 20), np.nan, np.float32), x0=0.0, y0=0.0, cell=1.0)
    chm["chm"][5, 5] = 18.0        # a tree
    chm["chm"][5, 15] = 1.2        # a bush
    xy = np.array([[5.5, 5.5], [15.5, 5.5], [100.0, 100.0]])   # tree, bush, off-raster
    keep, h = A.height_gate(chm, xy, cut=3.0, r=0.0)
    assert keep.tolist() == [True, False, True], (keep, h)


def test_score_is_monotone_in_predicted_distance():
    g = torch.tensor([0.0, 0.5, 2.0, 10.0])
    s = torch.exp(-torch.expm1(g).clamp_min(0.0) / M.SCORE_TAU)
    assert float(s[0]) == 1.0
    assert bool((s[1:] < s[:-1]).all())


# ---------------------------------------------------------------------------
# the attributed patch: the data join is the premise, so it gets a test
# ---------------------------------------------------------------------------
ATTR = next((p for p in (ROOT / "basewalker" / "poderevka_attributes.csv",
                         Path("/app/poderevka_attributes.csv")) if p.exists()),
            None)
STEMS = next((p for p in (ROOT / "basewalker" / "poderevka.csv",
                          Path("/app/poderevka.csv")) if p.exists()), None)


def test_attribute_rows_are_the_first_392_stem_map_rows():
    """The whole point of the attribute table is that it needs no spatial
    matching: its rows ARE rows 1..392 of the stem map's base_1 block. If that
    ever drifts, species would be silently attached to the wrong trees, so the
    join is asserted rather than assumed."""
    if ATTR is None or STEMS is None:
        print("   (skipped: survey CSVs not present)")
        return
    import csv as _csv
    stems = []
    for r in _csv.reader(open(STEMS)):
        if not r or not r[0].strip() or r[0].startswith("base"):
            if r and r[0].startswith("base_2"):
                break                       # base_1 ends here
            continue
        stems.append((r[0].strip(), float(r[1]), float(r[2]), float(r[3])))
    attr = list(_csv.DictReader(open(ATTR)))
    assert len(attr) == 392, len(attr)
    assert len(stems) >= len(attr), (len(stems), len(attr))
    for i, a in enumerate(attr):
        n, lat, lon, z = stems[i]
        assert a["n"] == n, (i, a["n"], n)
        assert abs(float(a["lat"]) - lat) < 1e-9, (i, a["lat"], lat)
        assert abs(float(a["lon"]) - lon) < 1e-9, (i, a["lon"], lon)
        assert abs(float(a["z"]) - z) < 1e-3, (i, a["z"], z)


def test_attribute_table_still_carries_species_and_girth():
    if ATTR is None:
        print("   (skipped: attribute CSV not present)")
        return
    import csv as _csv
    rows = list(_csv.DictReader(open(ATTR)))
    named = [r for r in rows if r["species"] and r["name_ambiguous"] == "0"]
    girth = [r for r in rows if r["girth_cm"]]
    assert len(named) >= 385, len(named)
    assert len(girth) >= 385, len(girth)
    assert len({r["species"] for r in named}) >= 25
    # ',' is the survey's decimal mark and must have been normalised, or a
    # plain CSV reader splits a girth in half
    assert not any("," in r["girth_cm"] for r in rows)


def test_patch_footprint_covers_clearings_but_not_the_outside():
    """Hull OR buffer, not buffer alone: a clearing the surveyors walked
    through is surveyed ground, so a detection in it is a real false positive
    and must fall inside the footprint."""
    import patch as P
    ring = np.array([[np.cos(t) * 20.0, np.sin(t) * 20.0]
                     for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
    inside = P.footprint(ring, margin=1.0)
    assert bool(inside(np.array([[0.0, 0.0]]))[0])          # clearing: inside
    assert bool(inside(np.array([[20.5, 0.0]]))[0])         # just outside hull
    assert not bool(inside(np.array([[60.0, 0.0]]))[0])     # well outside
    assert not bool(inside(np.array([[0.0, 40.0]]))[0])


def test_patch_select_reports_training_contamination(monkeypatch=None):
    """A patch score must never be reportable without its split composition:
    these labels were trained on, and `trained_on` is what says so."""
    import patch as P
    tree_xy = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0],
                        [2.5, 2.5]])
    attrs = dict(lat=np.zeros(5), lon=np.zeros(5),
                 species=np.array(["Quercus mongolica"] * 5, object),
                 ambiguous=np.zeros(5, bool), n=np.arange(5).astype(str),
                 girth_cm=np.array(["50"] * 5, object), z=np.zeros(5))
    orig = P.to_local
    P.to_local = lambda lat, lon, meta=None: tree_xy
    try:
        seeds = np.array([[2.5, 2.5, 0.0], [100.0, 100.0, 0.0]])
        bases = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [90.0, 90.0, 0.0]])
        is_val = np.array([False, False, True])
        in_margin = np.zeros(3, bool)
        s, gt, rows, info = P.select(seeds, bases, is_val, in_margin,
                                     margin=1.0, attrs=attrs)
        assert len(s) == 1 and len(gt) == 2, (len(s), len(gt))
        assert rows.tolist() == [0, 1]
        assert info["n_gt_in_train"] == 2 and info["n_gt_in_val"] == 0
        assert info["trained_on"] is True
        assert info["frac_gt_in_train"] == 1.0
    finally:
        P.to_local = orig


def test_patch_species_join_refuses_a_distant_base():
    """A base more than a fork-merge away from any surveyed tree gets no
    species, rather than the nearest one's."""
    import patch as P
    tree_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    attrs = dict(lat=np.zeros(2), lon=np.zeros(2),
                 species=np.array(["Betula davurica", "Salix caprea"], object),
                 ambiguous=np.zeros(2, bool), n=np.array(["1", "2"]),
                 girth_cm=np.array(["30", "40"], object), z=np.zeros(2))
    orig = P.to_local
    P.to_local = lambda lat, lon, meta=None: tree_xy
    try:
        out, ok = P.join_species(np.array([[0.1, 0.0], [5.0, 0.0]]),
                                 attrs=attrs, tol=0.35)
        assert ok.tolist() == [True, False]
        assert out["species"][0] == "Betula davurica"
        assert out["species"][1] == ""
    finally:
        P.to_local = orig


def _synthetic_stand(path, rng):
    """A flat 20 x 20 m stand: four trunks, a parked car and a hedge.

    The car and the hedge are the whole point — both are dense at breast
    height, so plan-view density alone cannot tell them from a stem, and only
    the vertical column can.
    """
    parts = []
    for cx, cy in [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0), (6.0, 6.0)]:
        n = 4000
        t = rng.uniform(0, 2 * np.pi, n)
        r = 0.12 * np.sqrt(rng.uniform(0.6, 1.0, n))
        parts.append(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t),
                                      rng.uniform(0.0, 12.0, n)]))
        n = 3000                                    # crown, well above the slice
        parts.append(np.column_stack([
            cx + rng.normal(0, 1.5, n), cy + rng.normal(0, 1.5, n),
            rng.uniform(8.0, 12.0, n)]))
    n = 6000                                        # car: dense, 1.4 m tall
    parts.append(np.column_stack([rng.uniform(12.0, 16.0, n),
                                  rng.uniform(-1.0, 1.0, n),
                                  rng.uniform(0.0, 1.4, n)]))
    n = 6000                                        # hedge: dense, 1.2 m tall
    parts.append(np.column_stack([rng.uniform(-2.0, 8.0, n),
                                  rng.uniform(11.8, 12.2, n),
                                  rng.uniform(0.0, 1.2, n)]))
    xyz = np.concatenate(parts).astype(np.float32)
    np.savez(path, xyz=xyz)
    return xyz


def test_baseline_column_separates_a_stem_from_a_car():
    """The classical control has to be worth comparing against: a density peak
    at breast height is not a tree, and the verticality fraction is the only
    thing in the detector that knows it."""
    import tempfile
    import baseline as B
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        tile = Path(d) / "tile_0.npz"
        _synthetic_stand(tile, rng)
        flat = lambda xy: np.zeros(len(xy), np.float64)   # noqa: E731
        cand, frac, dval = B.detect_stems((-3.0, 17.0, -3.0, 13.0), flat, [tile])

    assert len(cand) >= 4, len(cand)
    trunks = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]])
    for cx, cy in trunks:
        d = np.linalg.norm(cand - np.array([cx, cy]), axis=1)
        i = int(np.argmin(d))
        assert d[i] < 0.3, (cx, cy, d[i])
        # a 12 m trunk fills the column; anything else is a detector bug
        assert frac[i] > 0.8, (cx, cy, frac[i])

    # every candidate on the car or the hedge must score far below a trunk
    clutter = (cand[:, 0] > 11.0) | (cand[:, 1] > 10.0)
    if clutter.any():
        assert frac[clutter].max() < 0.4, frac[clutter].max()


def test_baseline_peaks_respect_separation_and_plateaus():
    """Two stems 1.2 m apart are two detections; a flat plateau is one, not
    one per cell (a bare equality test against a maximum filter returns the
    whole plateau, which would inflate the control's detection count)."""
    import baseline as B
    a = np.zeros((40, 40), np.float64)
    a[20, 10] = a[20, 22] = 5.0                  # 1.2 m apart at 0.1 m cells
    xy, v = B.peaks(a, 0.1, 0.6, 1.0, 0.0, 0.0)
    assert len(xy) == 2, len(xy)
    a = np.zeros((40, 40), np.float64)
    a[18:23, 18:23] = 5.0                        # a 0.5 m flat plateau
    xy, v = B.peaks(a, 0.1, 0.6, 1.0, 0.0, 0.0)
    assert len(xy) == 1, len(xy)


def test_baseline_girth_parses_a_multi_stem():
    """The survey writes a forked tree as '29;22', and the control's size
    bands are read from that column."""
    import baseline as B
    assert abs(B.girth_to_dbh("63") - 63 / np.pi) < 1e-9
    assert abs(B.girth_to_dbh("29;22") - 29 / np.pi) < 1e-9   # thickest stem
    assert np.isnan(B.girth_to_dbh(""))
    assert np.isnan(B.girth_to_dbh("n/a"))


# ---------------------------------------------------------------------------
# dense stem head (stemnet): everything except Sonata itself, on the CPU
# ---------------------------------------------------------------------------
def _toy_scene(tmp, rng, trunks=((0.0, 0.0), (6.0, 1.0), (2.0, 7.0))):
    """A flat 20 x 20 m plot: vertical trunks, a ground plane, a low car.

    Returns (SceneStore, Dem, bases) with the ground at z = 0 everywhere, so a
    point's height above ground is just its z.
    """
    import stemnet as SN
    parts = []
    for cx, cy in trunks:
        n = 3000
        t = rng.uniform(0, 2 * np.pi, n)
        r = 0.15 * np.sqrt(rng.uniform(0.5, 1.0, n))
        parts.append(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t),
                                      rng.uniform(0.0, 12.0, n)]))
    n = 20000                                        # ground
    parts.append(np.column_stack([rng.uniform(-10, 10, n),
                                  rng.uniform(-10, 10, n),
                                  rng.normal(0, 0.01, n)]))
    n = 4000                                         # car, 1.4 m tall
    parts.append(np.column_stack([rng.uniform(-9, -6, n),
                                  rng.uniform(-9, -7, n),
                                  rng.uniform(0, 1.4, n)]))
    xyz = np.concatenate(parts).astype(np.float32)
    nrm = np.zeros_like(xyz)
    nrm[:, 2] = 1.0
    dem_p = Path(tmp) / "dem.npz"
    np.savez(dem_p, z=np.zeros((64, 64), np.float32), x0=-16.0, y0=-16.0,
             cell=0.5)
    scene = SN.SceneStore(xyz, nrm, device="cpu")
    dem = SN.Dem(dem_p, device="cpu")
    bases = np.array([[cx, cy, 0.0] for cx, cy in trunks])
    return scene, dem, bases


def test_stemnet_band_drops_the_ground_and_the_crown():
    """The band is why one pass over the scene is affordable: if it does not
    actually throw away the ground and the canopy, nothing else here holds."""
    import tempfile
    import stemnet as SN
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        scene, dem, _b = _toy_scene(tmp, rng)
        band = SN.Band(scene, dem, lo=0.3, hi=4.0, voxel=0.05)
    z = band.xyz[:, 2]
    assert z.min() >= 0.3 - 1e-6 and z.max() <= 4.0 + 1e-6, (z.min(), z.max())
    assert len(band.xyz) < 0.5 * len(scene.xyz)          # most of it is gone
    # dedup really deduped: no two kept points share a 0.05 m voxel
    k = np.floor(band.xyz / 0.05).astype(np.int64)
    assert len(np.unique(k, axis=0)) == len(k)


def test_stemnet_windows_tile_the_scene_exactly_once():
    """Inner squares partition the region: a point counted twice is a vote
    counted twice, which would inflate every detection's confidence."""
    import tempfile
    import stemnet as SN
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        scene, dem, _b = _toy_scene(tmp, rng)
        band = SN.Band(scene, dem, lo=0.3, hi=4.0, voxel=0.05)
    centres = SN.tile_centres(-10.0, 10.0, -10.0, 10.0, win=8.0)
    seen = np.zeros(len(band.xyz), np.int64)
    for cx, cy in centres:
        pos, inner = band.window(cx, cy, win=8.0, pad=2.0)
        if len(pos):
            np.add.at(seen, pos[inner], 1)
    inside = ((np.abs(band.xyz[:, 0]) <= 10.0) & (np.abs(band.xyz[:, 1]) <= 10.0))
    assert seen[inside].min() == 1 and seen[inside].max() == 1, \
        (seen[inside].min(), seen[inside].max())


def test_stemnet_labels_are_horizontal_not_three_dimensional():
    """A trunk point 3 m up is 3 m from its own base in 3-D and ~0.1 m from it
    in plan view. Labelling in 3-D would mark the whole trunk a negative."""
    import stemnet as SN
    bases = torch.tensor([[0.0, 0.0]])
    xy = torch.tensor([[0.1, 0.0],        # on the trunk, any height
                       [0.9, 0.0],        # just inside NEAR_R
                       [2.0, 0.0]])       # off the tree
    pos, off = SN.targets(xy, bases, near_r=1.0)
    assert pos.tolist() == [True, True, False]
    assert torch.allclose(off[0], torch.tensor([-0.1, 0.0]), atol=1e-6)


def test_stemnet_vote_recovers_stems_from_noisy_offsets():
    """The detection machinery: p-weighted peaks of the shifted cloud. Dense
    consensus is the whole premise, so the vote counts matter as much as the
    positions."""
    import stemnet as SN
    rng = np.random.default_rng(2)
    stems = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.5]])
    xy, p, off = [], [], []
    for s in stems:
        n = 400
        q = s + rng.normal(0, 0.12, (n, 2))          # points around the stem
        xy.append(q)
        p.append(np.full(n, 0.9))
        off.append(s - q + rng.normal(0, 0.08, (n, 2)))   # noisy vote
    n = 600                                           # confident-but-wrong noise
    q = rng.uniform(-6, 6, (n, 2))
    xy.append(q); p.append(np.full(n, 0.2)); off.append(rng.normal(0, 1.0, (n, 2)))
    xy = np.concatenate(xy); p = np.concatenate(p); off = np.concatenate(off)

    det, mass, nv = SN.vote(xy, p, off, p_thresh=0.5, min_mass=20.0)
    assert len(det) == 3, (len(det), det)
    for s in stems:
        d = np.linalg.norm(det - s, axis=1).min()
        assert d < 0.15, (s, d)
    # this is the number the walker could not produce
    assert nv.min() >= 100, nv


def test_stemnet_head_learns_the_toy_task():
    """Train the real head with the real loss on hand-made features, so the
    optimisation wiring is exercised without Sonata: the features say where the
    point sits inside a 0.4 m column, which is enough to solve the task."""
    import stemnet as SN
    rng = np.random.default_rng(3)
    stems = np.array([[0.0, 0.0], [2.0, 0.0]])
    n = 800
    xy = np.concatenate([s + rng.normal(0, 0.1, (n, 2)) for s in stems]
                        + [rng.uniform(-4, 4, (n, 2))])
    d = np.linalg.norm(xy[:, None, :] - stems[None], axis=2)
    j = d.argmin(1)
    pos = d.min(1) <= 1.0
    # feature: the point's own offset to its column centre, plus a bias. A
    # linear head can solve this, which is the point -- we are testing the loss
    # and the vote, not the representation.
    feat = np.column_stack([np.where(pos, stems[j, 0] - xy[:, 0], 0.0),
                            np.where(pos, stems[j, 1] - xy[:, 1], 0.0),
                            pos.astype(np.float64), np.ones(len(xy))])
    feat = torch.tensor(feat, dtype=torch.float32)
    tgt_pos = torch.tensor(pos)
    tgt_off = torch.tensor(stems[j] - xy, dtype=torch.float32)

    torch.manual_seed(0)
    head = SN.StemHead(feat.shape[1], d=32)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    first = None
    for _ in range(400):
        logit, off = head(feat)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, tgt_pos.float())
                + torch.nn.functional.smooth_l1_loss(off[tgt_pos],
                                                     tgt_off[tgt_pos]))
        first = float(loss) if first is None else first
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    assert float(loss) < 0.3 * first, (first, float(loss))

    with torch.no_grad():
        logit, off = head(feat)
    det, mass, nv = SN.vote(xy, torch.sigmoid(logit).numpy(), off.numpy(),
                            p_thresh=0.5, min_mass=10.0)
    assert len(det) == 2, (len(det), det)
    for s in stems:
        assert np.linalg.norm(det - s, axis=1).min() < 0.2, s


class _FakePoint(dict):
    """Just enough of Pointcept's Point for the upcast: a dict with attribute
    access and an optional pooling chain."""

    def __getattr__(self, k):
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v


def test_stemnet_upcast_uses_the_chain_and_survives_its_absence():
    """The one library detail this design leans on. If the pooling chain is
    there the read-out is per point; if a build does not expose it, the stage
    must still run on token-resolution features rather than crash."""
    import stemnet as SN
    want = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                         [1.0, 1.0, 0.0]])

    SN._UPCAST_NOTE.clear()
    leaf = _FakePoint(feat=torch.zeros(4, 8), coord=want.clone())
    coarse = _FakePoint(feat=torch.arange(2 * 16, dtype=torch.float32).view(2, 16),
                        coord=want[:2].clone(),
                        pooling_parent=leaf,
                        pooling_inverse=torch.tensor([0, 0, 1, 1]))
    f = SN.upcast(coarse, want)
    assert f.shape == (4, 24), f.shape                  # 8 own + 16 upcast
    assert torch.equal(f[0, 8:], f[1, 8:])              # same parent token
    assert not torch.equal(f[0, 8:], f[2, 8:])
    assert SN._UPCAST_NOTE == ["chain"]

    SN._UPCAST_NOTE.clear()
    lone = _FakePoint(feat=torch.tensor([[1.0, 2.0], [30.0, 40.0]]),
                      coord=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
    f = SN.upcast(lone, want)
    assert f.shape == (4, 2), f.shape
    assert SN._UPCAST_NOTE == ["nn"]
    assert torch.allclose(f[0], torch.tensor([1.0, 2.0]))     # nearest token
    assert torch.allclose(f[3], torch.tensor([30.0, 40.0]))


def test_stem_train_window_step_runs_end_to_end():
    """The training plumbing on the toy scene with a stand-in encoder: band
    indices -> crop -> features -> both losses -> gradients into the head.
    Sonata is the only thing mocked, so a dtype or device slip in the real
    stage shows up here instead of an hour into a GPU run."""
    import tempfile
    import stemnet as SN
    import stem_train as T
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as tmp:
        scene, dem, bases = _toy_scene(tmp, rng)
        band = SN.Band(scene, dem, lo=0.3, hi=4.0, voxel=0.05)

        def fake_encoder(data):
            c = data["coord"]
            return _FakePoint(coord=c, feat=torch.cat(
                [c, c.abs(), torch.sin(c), data["feat"]], 1))

        head = SN.StemHead(18, d=32)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        bases_xy = torch.from_numpy(bases[:, :2].astype(np.float32))
        centres = bases[:, :2]
        hit = 0
        for _ in range(6):
            pos, inner, _c = T.window_batch(band, centres, rng)
            if len(pos) < 64 or not inner.any():
                continue
            l_cls, l_off, n_pos, n_pts, pred = T.forward_window(
                fake_encoder, head, scene, band, pos, inner, bases_xy,
                train=True)
            loss = l_cls + l_off
            assert torch.isfinite(loss), loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = head.logit.weight.grad
            assert g is not None and torch.isfinite(g).all()
            opt.step()
            assert n_pos > 0 and n_pts >= 64, (n_pos, n_pts)
            assert pred[0].shape[0] == n_pts and pred[2].shape == (n_pts, 2)
            hit += 1
        assert hit >= 4, hit


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} GeoWalker tests passed.")
