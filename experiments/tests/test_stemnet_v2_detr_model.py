import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stemnet_v2'))
import detr as R
import train_detr as train


def test_voxel_cap_is_deterministic_across_torch_rng(monkeypatch):
    monkeypatch.setattr(R, 'MAX_TOKENS', 3)
    xyz = np.column_stack([np.arange(10, dtype=np.float64), np.zeros((10, 2))])
    feat = torch.arange(10, dtype=torch.float32)[:, None]
    torch.manual_seed(1)
    a = R.voxelise(feat, xyz, np.zeros(2))
    torch.manual_seed(7)
    b = R.voxelise(feat, xyz, np.zeros(2))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_predict_preserves_eval_and_rejects_nonfinite():
    head = R.DetrHead(2, d=8, n_query=4, layers=1, heads=2, ff=16)
    runner = train.DetrRunner(None, head, None, None, 'cpu')
    c = {'xyz': np.array([[0., 0., 0.], [1., 1., 1.]])}
    runner.feat.inputs = lambda c: torch.ones(2, 2)
    head.eval()
    xy, score = runner.predict(c, np.zeros(2))
    assert xy.shape == (4, 2) and score.shape == (4,) and not head.training
    runner.feat.inputs = lambda c: torch.full((2, 2), float('nan'))
    with pytest.raises(ValueError, match='finite'):
        runner.predict(c, np.zeros(2))
    assert not head.training


def test_save_rejects_nonfinite_optimizer_state(tmp_path, monkeypatch):
    monkeypatch.setattr(train.T, 'MODELS', tmp_path)
    monkeypatch.setattr(train, 'log', lambda msg: None)
    head = R.DetrHead(2, d=8, n_query=4, layers=1, heads=2, ff=16)
    opt = torch.optim.AdamW(head.parameters())
    p = next(head.parameters())
    opt.state[p]['exp_avg'] = torch.full_like(p, float('nan'))
    path = tmp_path / 'bad.pth'
    train.save(path, head, opt, 1, 0., np.zeros(12), np.ones(12), {})
    assert not path.exists()


def test_save_records_validation_filter_and_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(train.T, 'MODELS', tmp_path)
    head = R.DetrHead(2, d=8, n_query=4, layers=1, heads=2, ff=16)
    opt = torch.optim.AdamW(head.parameters())
    audit = {'policy': 'height-v1'}
    path = tmp_path / 'ck.pth'
    train.save(path, head, opt, 1, 0., np.zeros(12), np.ones(12), {'validation_filter': audit})
    ck = torch.load(path, weights_only=False)
    assert ck['validation_filter'] == audit
    assert ck['cfg']['crop'] == R.CROP and ck['cfg']['max_tokens'] == R.MAX_TOKENS
    assert ck['cfg']['nms_r'] == R.NMS_R and ck['cfg']['min_p'] == R.MIN_P
    assert ck['cfg']['grid'] == train.M.GRID and ck['cfg']['seed'] == train.T.SEED


def test_resume_rejects_legacy_policy_geometry_and_nan_optimizer():
    audit = {'policy': 'height-v1'}
    ck = {'validation_filter': audit, 'hcfg': {'d': 8}, 'd_in': 2,
          'cfg': {'crop': R.CROP, 'own': R.OWN, 'voxel': R.VOXEL,
                  'max_tokens': R.MAX_TOKENS, 'cap': train.CAP, 'encoder': train.T.ENCODER,
                  'nms_r': R.NMS_R, 'min_p': R.MIN_P, 'grid': train.M.GRID, 'seed': train.T.SEED},
          'rng': {'numpy': np.random.default_rng().bit_generator.state, 'torch': torch.get_rng_state(), 'cuda': None},
          'opt': {'state': {}, 'param_groups': []}}
    train.validate_resume(ck, audit)
    for change in ({'validation_filter': None}, {'cfg': dict(ck['cfg'], voxel=123)},
                   {'opt': {'state': {'v': torch.tensor(float('nan'))}}}):
        with pytest.raises(ValueError, match='V2_RESUME=0'):
            train.validate_resume(dict(ck, **change), audit)


def test_crop_retry_exhaustion_is_explicit():
    class EmptyStore:
        def crop(self, *args):
            return None
    with pytest.raises(RuntimeError, match='20'):
        train.training_crop(EmptyStore(), np.array([[5., 5.]]), np.random.default_rng(4))


def test_runner_rejects_nonfinite_raw_features_before_encoder():
    head = R.DetrHead(2, d=8, n_query=4, layers=1, heads=2, ff=16)
    runner = train.DetrRunner(None, head, None, None, 'cpu')
    runner.feat.inputs = lambda c: torch.ones(2, 2)
    c = {'xyz': np.array([[0., 0., 0.], [1., 1., 1.]]), 'feat': np.array([[float('nan')], [1.]])}
    with pytest.raises(ValueError, match='finite'):
        runner.predict(c, np.zeros(2))


def test_real_head_cpu_backward_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(train.T, 'MODELS', tmp_path)
    head = R.DetrHead(12, d=16, n_query=4, layers=2, heads=4, ff=32)
    opt = torch.optim.AdamW(head.parameters(), lr=0.001)
    xyz = np.random.default_rng(0).uniform(0, 3, (80, 3))
    vf, vp = R.voxelise(torch.randn(80, 12), xyz, np.array([1.5, 1.5]))
    before = head.cls.weight.detach().clone()
    loss, stats = R.set_loss(head(vf, vp), torch.tensor([[0.2, -0.3]]))
    loss.backward(); opt.step()
    assert torch.isfinite(loss) and stats['gt'] == 1 and not torch.equal(before, head.cls.weight)
    path = tmp_path / 'roundtrip.pth'
    train.save(path, head, opt, 1, 0., np.zeros(12), np.ones(12), {'validation_filter': {'policy': 'test'}})
    ck = torch.load(path, weights_only=False)
    restored = R.DetrHead(ck['d_in'], **ck['hcfg'])
    restored.load_state_dict(ck['head'])
    assert torch.equal(restored.cls.weight, head.cls.weight)


def test_main_cpu_train_and_resume_uses_truth_and_checkpoint(tmp_path, monkeypatch, accum=1, before_resume=None):
    import json

    out = tmp_path / 'data'; out.mkdir()
    res = tmp_path / 'results'; res.mkdir()
    (out / 'frame.json').write_text(json.dumps({'labels': {}}))
    (out / 'label_bases.csv').write_text('x,y,tier\n4,4,A\n14,4,A\n')
    split = np.zeros((20, 20), dtype=np.uint8); split[:, 12:] = 1
    np.savez(out / 'regions.npz', split=split, occupancy=np.ones_like(split))
    class Store:
        def crop(self, cx, cy, size):
            r = np.random.default_rng(0)
            xyz = r.uniform(-3, 3, (240, 3)) + [cx, cy, 0]
            feat = np.ones((240, 12), np.float32)
            return {'xyz': xyz, 'feat': feat, 'normal': feat[:, :3].copy()}
    monkeypatch.setattr(train, 'OUT', out)
    monkeypatch.setattr(train.T, 'OUT', out)
    monkeypatch.setattr(train, 'RESULTS', res)
    monkeypatch.setattr(train, 'LOG', res / 'log.txt')
    monkeypatch.setattr(train.T, 'MODELS', out / 'models')
    monkeypatch.setattr(train, 'BEST', out / 'models' / 'best.pth')
    monkeypatch.setattr(train, 'LAST', out / 'models' / 'last.pth')
    monkeypatch.setattr(train.T, 'ENCODER', 'fake')
    monkeypatch.setattr(train.T, 'DEVICE', 'cpu')
    monkeypatch.setattr(train.T, 'RESUME', True)
    monkeypatch.setattr(train.T, 'figure', lambda *a, **k: None)
    monkeypatch.setattr(train.D, 'open_store', lambda log: Store())
    monkeypatch.setattr(train.T, 'feature_stats', lambda *a: (np.zeros(12, np.float32), np.ones(12, np.float32)))
    monkeypatch.setattr(train, 'STEPS', 1)
    monkeypatch.setattr(train, 'EVAL_EVERY', 1)
    monkeypatch.setattr(train, 'ACCUM', accum)
    monkeypatch.setattr(train, 'CAP', 200)
    monkeypatch.setattr(train, 'D_MODEL', 8)
    monkeypatch.setattr(train, 'QUERIES', 4)
    monkeypatch.setattr(train, 'LAYERS', 1)
    audit = {'version': 1, 'split': 'val', 'filter': 'all_survey_bases'}
    train.main()
    ck = torch.load(train.LAST, weights_only=False)
    assert ck['step'] == 1 and ck['eval']['validation_filter'] == audit
    if before_resume is not None:
        before_resume(ck)
    monkeypatch.setattr(train, 'STEPS', 2)
    monkeypatch.setattr(train, 'D_MODEL', 16)  # resume must use saved architecture
    train.main()
    ck = torch.load(train.LAST, weights_only=False)
    assert ck['step'] == 2 and ck['hcfg']['d'] == 8


def test_predict_rejects_nonfinite_logits_and_restores_training():
    head = R.DetrHead(2, d=8, n_query=4, layers=1, heads=2, ff=16)
    runner = train.DetrRunner(None, head, None, None, 'cpu')
    runner.feat.inputs = lambda c: torch.ones(2, 2)
    c = {'xyz': np.array([[0., 0., 0.], [1., 1., 1.]])}
    with torch.no_grad():
        head.cls.bias.fill_(float('nan'))
    head.train()
    with pytest.raises(ValueError, match='finite'):
        runner.predict(c, np.zeros(2))
    assert head.training


def test_matching_rejects_more_targets_than_queries():
    outs = [(torch.zeros(2), torch.zeros(2, 2))]
    with pytest.raises(ValueError, match='queries'):
        R.set_loss(outs, torch.zeros(3, 2))


@pytest.mark.parametrize('layer,field', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_nonfinite_head_output_is_skipped_before_matching(layer, field):
    head = R.DetrHead(2, d=8, n_query=4, layers=2, heads=2, ff=16)
    runner = train.DetrRunner(None, head, None, None, 'cpu')
    runner.feat.inputs = lambda c: torch.ones(2, 2)
    c = {'xyz': np.array([[0., 0., 0.], [1., 1., 1.]])}
    original = head.forward
    def poisoned(*args):
        outputs = original(*args)
        pair = list(outputs[layer])
        pair[field] = pair[field] * float('nan')
        outputs[layer] = tuple(pair)
        return outputs
    head.forward = poisoned
    assert runner.outs(c, np.zeros(2)) is None
    with pytest.raises(ValueError, match='finite'):
        runner.predict(c, np.zeros(2))


def test_main_skips_poisoned_crop_among_valid_crops(tmp_path, monkeypatch):
    original = R.DetrHead.forward
    calls = []
    def poisoned_once(self, *args):
        outs = original(self, *args)
        if self.training:
            calls.append(1)
            if len(calls) == 1:
                outs[0] = (outs[0][0], outs[0][1] * float('nan'))
        return outs
    monkeypatch.setattr(R.DetrHead, 'forward', poisoned_once)
    monkeypatch.setattr(R, 'crop_bases', lambda kd, xy, cx, cy: np.array([[cx, cy]]))
    test_main_cpu_train_and_resume_uses_truth_and_checkpoint(tmp_path, monkeypatch, accum=2)
    assert len(calls) >= 2
    assert torch.load(train.LAST, weights_only=False)['opt']['state']
    assert 'skipped crop' in (tmp_path / 'results' / 'log.txt').read_text()


def test_checkpoint_restores_numpy_sampler_and_torch_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(train.T, 'DEVICE', 'cpu')
    monkeypatch.setattr(train.T, 'MODELS', tmp_path)
    rng = np.random.default_rng(37)
    torch.manual_seed(47)
    head = R.DetrHead(2, d=8, n_query=4, layers=1, heads=2, ff=16)
    opt = torch.optim.AdamW(head.parameters())
    class Store:
        def crop(self, cx, cy, size):
            return {'xyz': np.zeros((200, 3))}
    centres = np.array([[1., 2.], [10., 20.], [100., 200.]])
    train.training_crop(Store(), centres, rng)
    path = tmp_path / 'rng.pth'
    audit = {'policy': 'test'}
    train.save(path, head, opt, 1, 0., np.zeros(12), np.ones(12), {'validation_filter': audit}, rng=rng)
    expected = [train.training_crop(Store(), centres, rng)[1] for _ in range(4)]
    expected_torch = torch.rand(5)
    ck = torch.load(path, weights_only=False)
    train.validate_resume(ck, audit)
    resumed = np.random.default_rng(train.T.SEED)
    R.DetrHead(2, d=8, n_query=4, layers=1, heads=2, ff=16)  # initialization consumes torch RNG
    train.restore_rng(ck, resumed)
    actual = [train.training_crop(Store(), centres, resumed)[1] for _ in range(4)]
    assert np.array_equal(actual, expected)
    assert torch.equal(torch.rand(5), expected_torch)


def test_resume_rejects_missing_rng_and_changed_eval_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(train.T, 'MODELS', tmp_path)
    head = R.DetrHead(2, d=8, n_query=4, layers=1, heads=2, ff=16)
    opt = torch.optim.AdamW(head.parameters())
    audit = {'policy': 'test'}
    path = tmp_path / 'rng.pth'
    train.save(path, head, opt, 1, 0., np.zeros(12), np.ones(12), {'validation_filter': audit}, rng=np.random.default_rng(3))
    ck = torch.load(path, weights_only=False)
    legacy = dict(ck); legacy.pop('rng')
    with pytest.raises(ValueError, match='V2_RESUME=0'):
        train.validate_resume(legacy, audit)
    for key in ('nms_r', 'min_p', 'grid', 'seed'):
        changed = dict(ck, cfg=dict(ck['cfg'], **{key: ck['cfg'][key] + 1}))
        with pytest.raises(ValueError, match='V2_RESUME=0'):
            train.validate_resume(changed, audit)


def test_main_resume_continues_sampler_and_torch_state(tmp_path, monkeypatch):
    original = train.training_crop
    observed = []
    def recording(store, centres, rng):
        result = original(store, centres, rng)
        observed.append((result[1].copy(), torch.rand(4), store, centres))
        return result
    monkeypatch.setattr(train, 'training_crop', recording)
    expected = []
    def before_resume(ck):
        clone = np.random.default_rng()
        clone.bit_generator.state = ck['rng']['numpy']
        _, _, store, centres = observed[-1]
        expected.append(original(store, centres, clone)[1])
        state = torch.get_rng_state()
        torch.set_rng_state(ck['rng']['torch'].cpu())
        expected.append(torch.rand(4))
        torch.set_rng_state(state)
    test_main_cpu_train_and_resume_uses_truth_and_checkpoint(tmp_path, monkeypatch, before_resume=before_resume)
    assert np.array_equal(observed[1][0], expected[0])
    assert torch.equal(observed[1][1], expected[1])
