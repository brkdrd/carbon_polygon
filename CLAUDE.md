# carbon_polygon

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
