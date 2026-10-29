# carbon_polygon

## Never cast georeferenced coordinates to float32 before centring

The campus is in UTM: easting ~736,000 m, northing ~4,767,000 m. At those
magnitudes float32 can only represent steps of 0.0625 m (easting) and 0.5 m
(northing), so `utm.astype(np.float32)` silently snaps the cloud to a coarse
grid and destroys the geometry with no error. This already happened once: a
tree model scoring 0.95 IoU on local coordinates collapsed on the same scene in
UTM, because Sonata was fed 0.5 m-quantised points. Full write-up:
`FLOAT32_COORDINATE_BUG.md`.

Rule: subtract a local origin first, in float64 or in the LAS raw integers,
and cast to float32 LAST.
- Old tracks: `experiments/common/geo.py` `center_float64()` (tests:
  `experiments/tests/test_float32_bug.py`).
- StemNet v2: `stemnet_v2/core.py` `Frame.local()` subtracts the origin from the
  raw LAS integers; tiles store small frame coordinates (`frame.json` `origin_m`);
  `stemnet_v2/model.py` `to_local()` centres each crop before float32.
- Anything new that reads a .las/.laz, a survey, or writes UTM back: same rule.
  Read `las.x` (float64) or the raw integers, never cast them first. When writing,
  add the origin in float64 (as `infer.py` / `export_val.py` do).

## Survey (poderevka.csv) vs scan: the fixed shift

The RTK tree survey and the campus scan are offset as a whole. In the scan's
frame every survey point must be moved by

    (+0.68, +1.02) m  (x east, y north; ~1.23 m at 56 deg, north-east)

before it is used. One shift for the whole survey: every survey block and every
100 m area prefers the same one (audited in `experiments/stemnet_v2/prep_bases.py`,
results/stemnet_v2_bases.txt).

Measured (2026-10-15): distance from a survey point to the nearest trunk surface
in the scan, median 23.5 cm raw (no better than chance) -> 1.9 cm shifted;
95% within 20 cm after the shift.

- Never use raw survey x/y as a training target or as scoring truth. Use the
  derived bases in `data/stemnet_v2/label_bases.csv` (StemNet v2 prep), which
  already include the shift.
- Survey z is unusable (0.2-0.46 m below the scan ground).
- The surveyors touched the stick to the SIDE of each trunk: a shifted survey
  point lies on the bark, about one stem radius from the trunk centre. The v2
  bases correct that where they can (tier A: trunk circle centre; tier B: moved
  inward by the measured girth radius); tier C (77% of rows) stays on the bark,
  median 11 cm from the centre.
- BaseWalker, GeoWalker and StemNet v1 (`basewalker/prep_scene.py` labels) were
  trained and scored WITHOUT the shift; their numbers are not comparable with
  StemNet v2's.
- The survey is not complete: some trees (especially small / young ones) were
  never recorded, and they are labelled background in training. Precision
  measured against the survey is a lower bound.
