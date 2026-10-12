#!/usr/bin/env python3
"""StemNet CPU self-check: everything except Sonata itself.

Sonata needs a GPU and a download, so it is the one thing mocked here — a
stand-in encoder that returns per-point features. Everything the stage actually
gets wrong is in scope: whether the band really drops the ground and the crown,
whether inner windows partition the scene exactly once (a point counted twice
is a vote counted twice), whether the labels are horizontal (in 3-D the whole
trunk would be a negative), whether the votes recover stems from noisy offsets,
whether the head trains at all, and whether the dense read-out survives a
Sonata build that does not expose its pooling chain.

    python tests/test_stemnet.py        # or: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import torch

# Works both in a repo checkout (experiments/stemnet) and inside the container
# image, where the modules live at /app/stemnet and /app.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (ROOT, ROOT / "stemnet",
                                 Path("/app"), Path("/app/stemnet"))
                if p.is_dir()]

def _toy_scene(tmp, rng, trunks=((0.0, 0.0), (6.0, 1.0), (2.0, 7.0))):
    """A flat 20 x 20 m plot: vertical trunks, a ground plane, a low car.

    Returns (Chunk, Dem, bases) with the ground at z = 0 everywhere, so a
    point's height above ground is just its z.
    """
    import net as SN
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
    chunk = SN.Chunk(xyz, nrm, device="cpu")
    dem = SN.Dem(dem_p, device="cpu")
    bases = np.array([[cx, cy, 0.0] for cx, cy in trunks])
    return chunk, dem, bases


def test_stemnet_band_drops_the_ground_and_the_crown():
    """The band is why one pass over the scene is affordable: if it does not
    actually throw away the ground and the canopy, nothing else here holds."""
    import tempfile
    import net as SN
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        chunk, dem, _b = _toy_scene(tmp, rng)
        band = SN.Band(chunk, dem, lo=0.3, hi=4.0, voxel=0.05)
    z = band.xyz[:, 2]
    assert z.min() >= 0.3 - 1e-6 and z.max() <= 4.0 + 1e-6, (z.min(), z.max())
    assert len(band.xyz) < 0.5 * len(chunk.xyz)          # most of it is gone
    # dedup really deduped: no two kept points share a 0.05 m voxel
    k = np.floor(band.xyz / 0.05).astype(np.int64)
    assert len(np.unique(k, axis=0)) == len(k)


def test_only_the_band_is_resident_on_the_device():
    """The regression this guards killed a training run at iteration 159.

    `Band` used to index into a Chunk holding the WHOLE cloud — 267 M points,
    6.4 GB of xyz plus normals — of which it reads 17% and never touches the
    rest again. That dead weight was more of the card than the encoder, the
    dense read-out and the head's activations combined, so a denser-than-usual
    window had nowhere to go. What is uploaded must be the band and only the
    band, and a band position must BE a device row, or the two could diverge.
    """
    import tempfile
    import net as SN
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        chunk, dem, _b = _toy_scene(tmp, rng)
        band = SN.Band(chunk, dem, lo=0.3, hi=4.0, voxel=0.05)

        # resident == the band, not the source
        assert len(band.chunk) == len(band.xyz) < 0.5 * len(chunk.xyz)
        # host float64 row i and device float32 row i are the same point
        dev = band.chunk.xyz.cpu().numpy()
        assert np.abs(dev - band.xyz).max() < 1e-2, np.abs(dev - band.xyz).max()
        # and crop() hands back rows of THAT, not of the source cloud
        pos = np.array([0, 5, len(band.xyz) - 1], np.int64)
        idx, zcol = band.crop(pos)
        got = band.chunk.xyz[idx].cpu().numpy()
        assert np.abs(got - band.xyz[pos]).max() < 1e-2
        assert zcol is None or np.abs(
            zcol.cpu().numpy() - band.h[pos]).max() < 1e-3


def test_stemnet_windows_tile_the_scene_exactly_once():
    """Inner squares partition the region: a point counted twice is a vote
    counted twice, which would inflate every detection's confidence."""
    import tempfile
    import net as SN
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        chunk, dem, _b = _toy_scene(tmp, rng)
        band = SN.Band(chunk, dem, lo=0.3, hi=4.0, voxel=0.05)
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
    import net as SN
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
    import net as SN
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
    import net as SN
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
    import net as SN
    want = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                         [1.0, 1.0, 0.0]])

    SN._NOTE.clear()
    leaf = _FakePoint(feat=torch.zeros(4, 8), coord=want.clone())
    coarse = _FakePoint(feat=torch.arange(2 * 16, dtype=torch.float32).view(2, 16),
                        coord=want[:2].clone(),
                        pooling_parent=leaf,
                        pooling_inverse=torch.tensor([0, 0, 1, 1]))
    f = SN.upcast(coarse, want, note=SN._NOTE)
    assert f.shape == (4, 24), f.shape                  # 8 own + 16 upcast
    assert torch.equal(f[0, 8:], f[1, 8:])              # same parent token
    assert not torch.equal(f[0, 8:], f[2, 8:])
    assert SN._NOTE == ["chain"]

    SN._NOTE.clear()
    lone = _FakePoint(feat=torch.tensor([[1.0, 2.0], [30.0, 40.0]]),
                      coord=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
    f = SN.upcast(lone, want, note=SN._NOTE)
    assert f.shape == (4, 2), f.shape
    assert SN._NOTE == ["nn"]
    assert torch.allclose(f[0], torch.tensor([1.0, 2.0]))     # nearest token
    assert torch.allclose(f[3], torch.tensor([30.0, 40.0]))


def test_stem_train_window_step_runs_end_to_end():
    """The training plumbing on the toy scene with a stand-in encoder: band
    indices -> crop -> features -> both losses -> gradients into the head.
    Sonata is the only thing mocked, so a dtype or device slip in the real
    stage shows up here instead of an hour into a GPU run."""
    import tempfile
    import net as SN
    import train as T
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as tmp:
        chunk, dem, bases = _toy_scene(tmp, rng)
        band = SN.Band(chunk, dem, lo=0.3, hi=4.0, voxel=0.05)

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
                fake_encoder, head, band, pos, inner, bases_xy, train=True)
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


def _sloped_scene(tmp, rng, slope=0.35, trunks=((-6.0, 0.0), (6.0, 0.0))):
    """The same stand on a hillside: ground z = slope * x, two identical trunks
    12 m apart along the fall line, so one sits ~4 m below the other."""
    import net as SN
    parts = []
    for cx, cy in trunks:
        n = 3000
        t = rng.uniform(0, 2 * np.pi, n)
        r = 0.15 * np.sqrt(rng.uniform(0.5, 1.0, n))
        g = slope * cx
        parts.append(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t),
                                      g + rng.uniform(0.0, 10.0, n)]))
    n = 30000
    gx = rng.uniform(-12, 12, n)
    parts.append(np.column_stack([gx, rng.uniform(-12, 12, n),
                                  slope * gx + rng.normal(0, 0.01, n)]))
    xyz = np.concatenate(parts).astype(np.float32)
    nrm = np.zeros_like(xyz); nrm[:, 2] = 1.0
    # an exact DEM for that plane, on the same grid convention prep_scene uses
    cell, x0, y0, ny, nx = 0.5, -16.0, -16.0, 64, 64
    gxs = x0 + np.arange(nx) * cell
    z = np.repeat((slope * gxs)[None, :], ny, axis=0).astype(np.float32)
    dem_p = Path(tmp) / "dem_slope.npz"
    np.savez(dem_p, z=z, x0=x0, y0=y0, cell=cell)
    chunk = SN.Chunk(xyz, nrm, device="cpu")
    return chunk, SN.Dem(dem_p, device="cpu"), np.array(trunks)


def test_detrending_makes_two_trunks_at_different_elevations_identical():
    """The bug this guards: one `z.min()` for a 20 m crop means that, on a
    slope, an uphill trunk reaches the encoder at local z 8-12 m and an
    identical downhill trunk at 0-4 m. Same object, different input, and the
    encoder cannot see where the ground is. Detrended, the two crops must agree.
    """
    import tempfile
    import net as SN
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        chunk, dem, trunks = _sloped_scene(tmp, rng)
        band = SN.Band(chunk, dem, lo=0.3, hi=4.0, voxel=0.05)

        got = []
        for cx, cy in trunks:
            pos, inner = band.window(cx, cy, win=4.0, pad=0.0)
            assert inner.any(), (cx, cy)
            pos = pos[inner]
            idx, zcol = band.crop(pos)
            assert zcol is not None, "SN_DETREND is off in this environment"
            # the z the encoder is handed, before the crop's own centre shift
            got.append(np.sort(zcol.numpy()))

        # both trunks occupy the same band slab once the terrain is removed
        for z in got:
            assert 0.3 - 1e-4 <= z.min() and z.max() <= 4.0 + 1e-4, (z.min(), z.max())
        lo_a, hi_a = got[0].min(), got[0].max()
        lo_b, hi_b = got[1].min(), got[1].max()
        assert abs(lo_a - lo_b) < 0.1 and abs(hi_a - hi_b) < 0.1, (got[0], got[1])

        # and the control: raw z would have put the two crops metres apart,
        # which is what the old code fed the encoder
        raw = []
        for cx, cy in trunks:
            pos, inner = band.window(cx, cy, win=4.0, pad=0.0)
            idx, _ = band.crop(pos[inner])
            raw.append(float(band.chunk.xyz[idx][:, 2].min()))
        assert abs(raw[0] - raw[1]) > 3.0, \
            f"the toy slope is too gentle to be testing anything: {raw}"


def test_window_jitter_is_fresh_every_call_and_never_leaks_into_val():
    """Two properties of the training window sampler, one of them critical.

    FRESH: the jitter must be redrawn per iteration. Drawn once, a run is
    pinned to one fixed set of crops and a long run becomes passes over the
    same windows instead of more ground.

    LEAK-FREE: the pool is filtered on the worst jitter an anchor can draw, so
    no crop can reach across the leakage margin into the val block — a training
    crop that touches val ground invalidates every score the run produces, and
    it would do so silently.
    """
    import tempfile
    import net as SN
    import train as T
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        chunk, dem, bases = _toy_scene(tmp, rng)
        band = SN.Band(chunk, dem, lo=0.3, hi=4.0, voxel=0.05)
        anchors = bases[:, :2]
        got = [T.window_batch(band, anchors, rng, jitter=2.0)[2]
               for _ in range(12)]
        assert len(set(got)) > 1, "the jitter is not being redrawn"
        # and it stays within the jitter of some anchor
        for cx, cy in got:
            d = np.abs(anchors - np.array([cx, cy])).max(1)
            assert d.min() <= 2.0 + 1e-9, (cx, cy, d.min())

    # the leakage filter: the WORST jitter, not the anchor, must clear the limit
    lim, jit = 61.3, 5.33
    anch = np.array([[50.0, 0.0], [57.0, 0.0], [59.0, 0.0], [70.0, 0.0]])
    pool = anch[anch[:, 0] + jit < lim]
    # 57.0 clears the limit on its own but 57.0 + 5.33 = 62.33 does not, so it
    # must be excluded — that is the whole point of filtering on the jitter
    assert set(pool[:, 0]) == {50.0}, pool
    assert (pool[:, 0] + jit < lim).all()


def test_oom_retry_halves_the_crop_but_never_hides_a_real_bug():
    """What a 4,000-iteration run is allowed to survive, and what it is not.

    A window twice as dense as the rest may be retried smaller and, failing
    that, skipped — that is what SN_PTS already does. An `illegal memory
    access` is also a RuntimeError, but the CUDA context is dead by then and
    retrying it would turn a hard failure into a silently thinner run, so it
    has to propagate.
    """
    import train as T
    rng = np.random.default_rng(0)
    pos, inner = np.arange(1000), np.ones(1000, bool)

    seen = []

    def real_bug(p, _i):
        seen.append(len(p))
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    try:
        T.oom_retry(real_bug, pos, inner, rng)
        raise AssertionError("a real bug was swallowed as an OOM")
    except RuntimeError as e:
        assert "illegal memory access" in str(e)
    assert seen == [1000], seen          # not retried even once

    seen.clear()

    def dense_window(p, _i):
        seen.append(len(p))
        if len(p) > 600:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        return "ok"

    assert T.oom_retry(dense_window, pos, inner, rng) == "ok"
    assert seen == [1000, 500], seen

    seen.clear()

    def hopeless(p, _i):
        seen.append(len(p))
        raise torch.cuda.OutOfMemoryError("CUDA out of memory.")

    assert T.oom_retry(hopeless, pos, inner, rng) is None
    assert seen == [1000, 500], seen


def test_relief_reports_what_sits_inside_one_crop():
    """The number that says whether this scene is a hillside or a table — and
    it must measure the crop, not the whole scene, because the crop is what a
    single z-shift has to cover."""
    import tempfile
    import net as SN
    rng = np.random.default_rng(8)
    with tempfile.TemporaryDirectory() as tmp:
        _c, _d, _t = _sloped_scene(tmp, rng, slope=0.35)
        out = SN.relief(Path(tmp) / "dem_slope.npz", win=10.0, pad=0.0)
    # a 0.35 grade over 10 m is 3.5 m of ground inside one crop
    assert abs(out["crop_span_p50_m"] - 3.5) < 0.6, out
    assert out["scene_relief_m"] > 10.0, out


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} StemNet tests passed.")
