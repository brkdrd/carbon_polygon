"""Validation targets use measured size, never elevation or guessed radii."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

V2 = Path(__file__).resolve().parents[1] / 'stemnet_v2'
sys.path.insert(0, str(V2))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_filter_boundaries_keep_corrected_coordinates_and_member_identity(tmp_path):
    import detr_truth as Q
    labels = tmp_path / 'label_bases.csv'
    attrs = tmp_path / 'attrs.csv'
    write_csv(labels, [dict(base=i, x=1.123456789 + i, y=2.234567891, tier='C',
                            members=f'1-{i}', radius='0.05') for i in range(5)])
    write_csv(attrs, [dict(id=f'1-{i}', diameter_cm=d, height_m=h)
                      for i, (d, h) in enumerate([(10, 5.01), (9.99, 12), (20, 5), (30, ''), (20, 8)])])
    split = np.ones((10, 10), dtype=np.uint8)
    split[2, 5] = 0  # qualifying tree outside validation
    truth = Q.load_truth({'split': split}, labels_path=labels, attrs_path=attrs)
    assert truth['id'].tolist() == [0, 2, 3]  # user deferred height filtering
    assert truth['xy'].dtype == np.float64
    np.testing.assert_array_equal(truth['xy'][0], [1.123456789, 2.234567891])
    assert truth['audit']['eligible'] == 3
    assert truth['audit']['unknown'] == 0
    assert truth['audit']['below_threshold'] == 1
    assert truth['audit']['total'] == 4
    assert truth['audit']['height_filter'] == 'pending_not_applied'


def test_repository_attributes_join_to_survey_members_not_row_order(tmp_path):
    import detr_truth as Q
    from core import parse_stem_map
    source = V2.parent / 'basewalker'
    sm = parse_stem_map(source / 'poderevka.csv')
    labels = tmp_path / 'label_bases.csv'
    # All actual survey rows at synthetic local positions, reversed deliberately.
    write_csv(labels, [dict(base=i, x=2, y=2, tier='C', members=f'{block}-{n}', radius='0.05')
                      for i, (block, n) in reversed(list(enumerate(zip(sm.block, sm.n))))])
    truth = Q.load_truth({'split': np.full((5, 5), 2)}, 2, labels_path=labels,
                         attrs_path=source / 'poderevka_attributes.csv', stem_map_path=source / 'poderevka.csv')
    assert truth['audit']['total'] == 4050
    assert truth['audit']['eligible'] == 179  # single girths + explicitly semicolon-separated stems
    assert truth['audit']['unknown'] == 3702  # 3658 without attributes + 40 ambiguous + 4 blank
    assert truth['audit']['below_threshold'] == 169
    assert 0 in truth['id']  # survey 1-1: 70 cm girth
    assert 1 not in truth['id']  # survey 1-2: 15 cm girth


def test_unknown_measurements_never_use_default_radius_and_forks_count_once(tmp_path):
    import detr_truth as Q
    labels, attrs = tmp_path / 'labels.csv', tmp_path / 'attrs.csv'
    write_csv(labels, [dict(base=i, x=1+i, y=1, tier='C', members=m, radius='0.05')
                      for i, m in enumerate(['1-1;1-2', '1-3', '1-4', '1-5', '1-6'])])
    write_csv(attrs, [dict(id=f'1-{i+1}', girth_cm=g)
                      for i, g in enumerate(['20', '40;20', '36.27.25', '', 'nan', '-30'])])
    truth = Q.load_truth({'split': np.ones((8, 8))}, labels_path=labels, attrs_path=attrs)
    assert truth['id'].tolist() == [0]
    assert truth['audit']['unknown'] == 4


@pytest.mark.parametrize('rows,error', [
    ([dict(id='1-1', diameter_cm=20), dict(id='1-1', diameter_cm=30)], 'duplicate'),
    ([dict(id='1-1x', diameter_cm=20)], 'Invalid'),
])
def test_invalid_attribute_identifiers_fail_closed(tmp_path, rows, error):
    import detr_truth as Q
    labels, attrs = tmp_path / 'labels.csv', tmp_path / 'attrs.csv'
    write_csv(labels, [dict(base=0, x=1, y=1, tier='C', members='1-1', radius='0.05')])
    write_csv(attrs, rows)
    with pytest.raises(ValueError, match=error):
        Q.load_truth({'split': np.ones((4, 4))}, labels_path=labels, attrs_path=attrs)


def test_girth_threshold_uses_pi_not_ten_cm_circumference(tmp_path):
    import detr_truth as Q
    labels, attrs = tmp_path / 'labels.csv', tmp_path / 'attrs.csv'
    write_csv(labels, [dict(base=i, x=1+i, y=1, tier='C', members=f'1-{i}', radius='0.05')
                      for i in range(3)])
    write_csv(attrs, [dict(id=f'1-{i}', girth_cm=g)
                      for i, g in enumerate([10, 10*np.pi, 10*np.pi-0.001])])
    truth = Q.load_truth({'split': np.ones((6, 6))}, labels_path=labels, attrs_path=attrs)
    assert truth['id'].tolist() == [1]
