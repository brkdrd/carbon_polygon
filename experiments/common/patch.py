#!/usr/bin/env python3
"""The densely attributed survey patch — the one place the ground truth is whole.

The RTK stem map (poderevka.csv) gives 4,050 tree positions and nothing else.
One block of it was surveyed properly: 392 trees in a 136 x 146 m patch, with a
girth and a Latin species name on 388 of them. That file is
basewalker/poderevka_attributes.csv, and its rows are rows 1..392 of the stem
map's base_1 block — the ids, latitudes, longitudes and elevations agree to
1e-9, so species joins onto the labels by position, with no spatial matching.

This module turns that patch into a region any model here can be run over
— it describes the SURVEY, not any one architecture, which is why it lives
in common:

  * the FOOTPRINT, in the scene's local frame. The patch is defined in WGS84 in
    the CSV, so it is projected with the same origin and EPSG prep_scene wrote
    into scene_meta.json — never a hard-coded offset, which would silently
    drift the moment the chunk is re-prepared.
  * the SEEDS inside it (for a model that uses seeds at all), taken from the
    scene's own seed grid rather than a new one, so a patch run is a subset of
    the full run and not a different experiment.
  * the GT inside it, taken from labels.npz, so the fork-merge and coverage
    filtering that produced every other number here apply unchanged.

WHAT THIS PATCH IS NOT: held out. The train/val split is a quantile on local x
(rightmost 20% is val) and this patch sits in the LEFT tail — the model trained
on these labels. `select` measures that and returns it, and every stage that reads it
says so out loud. A number from here is an upper bound on
what the model would do on unseen ground, not an estimate of it.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from . import config as C

# PREP, not BaseWalker: tiles / dem / labels / seeds are the shared scene prep
# that every track reads; the directory is named after the stage that writes it.
PREP = C.DATA / "basewalker"
BW = PREP                       # kept: older call sites import patch.BW
SCENE_META = PREP / "scene_meta.json"
# Baked into the image beside the stem map. The repo path is a fallback so the
# module also works in a plain checkout (tests, a local smoke run) instead of
# failing with a container path that cannot exist there.
ATTR_CSV = next(
    (q for q in (Path("/app/poderevka_attributes.csv"),
                 Path(__file__).resolve().parents[1]
                 / "basewalker" / "poderevka_attributes.csv")
     if q.exists()),
    Path("/app/poderevka_attributes.csv"))

# how far outside the surveyed trees still counts as "in the patch". The survey
# is exhaustive inside its polygon and says nothing beyond it, so a detection
# just outside is neither a hit nor a miss — it is unjudgeable, exactly like the
# unsurveyed terrain in the full run.
MARGIN = float(os.environ.get("PATCH_MARGIN",
                              os.environ.get("GW_PATCH_MARGIN", "3.0")))


def load_attributes(path=ATTR_CSV):
    """The 392-tree attribute table as a dict of columns."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[patch] {path} not found — it ships in the image "
                         f"(basewalker/Dockerfile copies it); rebuild, or pass "
                         f"an explicit path")
    n, lat, lon, z, girth, species, amb = [], [], [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            n.append(r["n"]); lat.append(float(r["lat"])); lon.append(float(r["lon"]))
            z.append(float(r["z"])); girth.append(r["girth_cm"])
            species.append(r["species"]); amb.append(r["name_ambiguous"] == "1")
    return dict(n=np.array(n), lat=np.array(lat), lon=np.array(lon),
                z=np.array(z), girth_cm=np.array(girth, object),
                species=np.array(species, object), ambiguous=np.array(amb))


def to_local(lat, lon, meta=SCENE_META):
    """WGS84 -> the scene's local frame, via the origin prep_scene recorded."""
    meta = Path(meta)
    if not meta.exists():
        raise SystemExit(f"[patch] {meta} not found — run bw_prep first")
    m = json.loads(meta.read_text())
    origin = np.asarray(m["origin_utm"], np.float64)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{m['utm_epsg']}",
                              always_xy=True)
    ex, ny = tr.transform(np.asarray(lon), np.asarray(lat))
    return np.column_stack([ex, ny]) - origin[None, :2]


def _hull_area(xy):
    from scipy.spatial import ConvexHull
    return ConvexHull(np.asarray(xy, np.float64)).volume      # 2-D: area


def footprint(patch_xy, margin=MARGIN):
    """A point-in-patch test: inside the surveyed hull, or within `margin` of a
    surveyed tree.

    The union of the two is deliberate. The hull alone would clip trees on the
    boundary; a buffer alone would leave holes in any clearing inside the
    patch, and a clearing the surveyors walked through IS surveyed — a
    detection in it is a real false positive and must be counted as one.
    """
    from scipy.spatial import Delaunay, cKDTree
    hull = Delaunay(patch_xy)
    kd = cKDTree(patch_xy)

    def inside(xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return (hull.find_simplex(xy) >= 0) | (kd.query(xy)[0] <= margin)
    return inside


def select(seeds, bases, is_val, in_margin, margin=MARGIN, attrs=None,
           meta=SCENE_META):
    """Restrict a run to the attributed patch.

    Returns (seeds_in, gt_in, gt_rows, info). `gt_rows` indexes `bases`, so a
    caller can carry the GT's own identity through to its output. `info`
    carries the split composition, which is the thing a reader of a patch score
    most needs to know.
    """
    attrs = load_attributes() if attrs is None else attrs
    patch_xy = to_local(attrs["lat"], attrs["lon"], meta=meta)
    inside = footprint(patch_xy, margin=margin)

    s_in = inside(seeds[:, :2])
    g_in = inside(bases[:, :2]) & ~in_margin
    rows = np.flatnonzero(g_in)

    n_val = int(is_val[rows].sum())
    info = dict(
        n_surveyed_trees=int(len(patch_xy)),
        n_species=int(len({s for s, a in zip(attrs["species"], attrs["ambiguous"])
                           if s and not a})),
        margin_m=float(margin),
        bbox=[float(patch_xy[:, 0].min()), float(patch_xy[:, 0].max()),
              float(patch_xy[:, 1].min()), float(patch_xy[:, 1].max())],
        # the hull, not the bounding box: the patch is not rectangular and the
        # bbox overstates it by ~70%, which would misprice every run estimated
        # from it.
        area_ha=float(_hull_area(patch_xy) / 1e4),
        bbox_area_ha=float(np.ptp(patch_xy[:, 0]) * np.ptp(patch_xy[:, 1]) / 1e4),
        n_seeds=int(s_in.sum()), n_gt=int(len(rows)),
        # the surveyed trees are only part of the GT here: their hull also
        # contains stem-map trees from the same RTK session that nobody
        # attributed (a median 12 m from the nearest attributed one, so they
        # are different trees, not re-measurements). They are real trees, so
        # missing one is a real miss and they stay in the GT; only the
        # per-species table is restricted to the attributed subset.
        n_gt_attributed=int(len(patch_xy)),
        n_gt_in_val=n_val, n_gt_in_train=int(len(rows) - n_val),
        frac_gt_in_train=float(1.0 - n_val / max(len(rows), 1)),
    )
    info["trained_on"] = info["frac_gt_in_train"] > 0.0
    return seeds[s_in], bases[rows], rows, info


def join_species(gt_xy, attrs=None, tol=0.35, meta=SCENE_META):
    """Attach each GT base's surveyed species and girth, by position.

    `tol` is the fork-merge radius plus a little: a merged base is the centroid
    of its stems, so it does not sit exactly on any one surveyed row.
    """
    attrs = load_attributes() if attrs is None else attrs
    from scipy.spatial import cKDTree
    patch_xy = to_local(attrs["lat"], attrs["lon"], meta=meta)
    d, j = cKDTree(patch_xy).query(np.asarray(gt_xy, np.float64))
    ok = d <= tol
    out = dict(
        n=np.where(ok, attrs["n"][j], ""),
        species=np.where(ok, attrs["species"][j], ""),
        girth_cm=np.where(ok, attrs["girth_cm"][j].astype(str), ""),
        dist_m=np.where(ok, d, np.nan),
    )
    return out, ok


def girth_to_dbh(v):
    """Girth in cm -> the thickest stem's DBH in cm.

    The survey writes a forked tree as '29;22', so a bare float() loses the
    whole row. Recall by stem size is read off this column, which is the table
    that separates a model's failure from the target's definition: about half
    these stems are under 10 cm DBH and together they carry ~4% of the basal
    area.
    """
    if not v:
        return float("nan")
    try:
        g = max(float(x) for x in str(v).split(";") if x.strip())
    except ValueError:
        return float("nan")
    return g / np.pi
