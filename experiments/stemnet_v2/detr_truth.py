"""Measured validation eligibility, joined to already corrected label bases.

The filter selects survey BASES, not LiDAR points. The input cloud is unchanged.
A diameter of 10 cm qualifies. Height filtering is pending by explicit user
decision: the attributed CSV has no tree heights. Unknown diameters are
excluded and counted, never inferred from survey elevation or
prep_labels' default radius. Coordinates come exclusively from label_bases.csv.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path

import numpy as np

from core import ATTRS, OUT, STEM_MAP, parse_stem_map
from curves import split_at


def _read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def _number(value):
    try:
        v = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _diameter(value, column):
    # Semicolons unambiguously separate stems. Repeated dots are NOT decimals.
    text = str(value or '').strip()
    if not re.fullmatch(r'\d+(?:\.\d+)?(?:\s*;\s*\d+(?:\.\d+)?)*', text):
        return None
    values = [_number(v) for v in text.split(';')]
    if any(v is None for v in values):
        return None
    return max(values) / np.pi if column == 'girth_cm' else max(values)


def _attributes_by_id(attrs, fields, stem_map_path):
    coord_ids = None
    if 'id' not in fields and not {'block', 'n'} <= set(fields):
        if not {'lat', 'lon'} <= set(fields):
            raise ValueError('Attributes require id (block-n), block+n, or survey lat+lon')
        sm = parse_stem_map(Path(stem_map_path or STEM_MAP))
        coord_ids = {}
        for b, n, lat, lon in zip(sm.block, sm.n, sm.lat, sm.lon):
            key = (round(float(lat), 8), round(float(lon), 8))
            if key in coord_ids:
                raise ValueError(f'Ambiguous survey coordinate key {key}')
            coord_ids[key] = f'{b}-{n}'
    out = {}
    for r in attrs:
        if 'id' in fields:
            key = r['id'].strip()
        elif coord_ids is None:
            key = f"{r['block'].strip()}-{r['n'].strip()}"
        else:
            key = coord_ids.get((round(float(r['lat']), 8), round(float(r['lon']), 8)))
            if key is None:
                raise ValueError('Attribute coordinates do not match the source survey')
            if r.get('n') and r['n'].strip() != key.split('-', 1)[1]:
                raise ValueError(f'Attribute n disagrees with coordinates for {key}')
        if not re.fullmatch(r'\d+-\d+', key) or key in out:
            raise ValueError(f'Invalid or duplicate survey id {key!r}')
        out[key] = r
    return out


def load_truth(regions, sid=1, *, labels_path=None, attrs_path=None, stem_map_path=None):
    """Return float64 xy, ids, tier, bbox, per-base rows and eligibility audit."""
    labels_path = Path(labels_path or OUT / 'label_bases.csv')
    attrs_path = Path(attrs_path or os.environ.get('V2_DETR_ATTRS') or ATTRS)
    labels, _ = _read(labels_path)
    attrs, fields = _read(attrs_path)

    diameter = next((k for k in ('diameter_cm', 'dbh_cm', 'girth_cm') if k in fields), None)
    if diameter is None:
        raise ValueError(f'{attrs_path}: require diameter_cm, dbh_cm or girth_cm')
    by_id = _attributes_by_id(attrs, fields, stem_map_path)
    split = regions['split']
    jj, ii = np.nonzero(split == sid)
    if not len(ii):
        raise ValueError(f'No cells for split {sid}')
    xy = np.array([[float(r['x']), float(r['y'])] for r in labels], dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(xy).all():
        raise ValueError('Non-finite corrected base coordinates')
    selected = np.flatnonzero(split_at(split, xy) == sid)
    kept, rows = [], []
    for i in selected:
        base = labels[i]
        status = []
        for member in base['members'].split(';'):
            attr = by_id.get(member, {})
            d = _diameter(attr.get(diameter), diameter)
            status.append('unknown' if d is None else
                          'eligible' if d >= 10.0 else 'below_threshold')
        reason = 'eligible' if 'eligible' in status else 'unknown' if 'unknown' in status else 'below_threshold'
        if reason == 'eligible':
            kept.append(i)
        rows.append(dict(base=int(base['base']), members=base['members'], x=float(xy[i, 0]),
                         y=float(xy[i, 1]), tier=base.get('tier', ''), eligible=reason == 'eligible', reason=reason))
    audit = dict(version=1, min_diameter_cm=10.0, height_filter='pending_not_applied', split_id=int(sid),
                 attributes_sha256=hashlib.sha256(attrs_path.read_bytes()).hexdigest(),
                 labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                 diameter_column=diameter,
                 total=len(rows), eligible=len(kept),
                 unknown=sum(r['reason'] == 'unknown' for r in rows),
                 below_threshold=sum(r['reason'] == 'below_threshold' for r in rows))
    return dict(xy=xy[kept], id=np.array([int(labels[i]['base']) for i in kept], dtype=np.int64),
                tier=np.array([labels[i].get('tier', '') for i in kept]),
                bbox=(float(ii.min()), float(ii.max() + 1), float(jj.min()), float(jj.max() + 1)),
                audit=audit, rows=rows)
