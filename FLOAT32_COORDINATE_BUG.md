# The float32 coordinate-quantization trap

## One-line version

Never cast georeferenced point-cloud coordinates (UTM, national grids) to
`float32` before subtracting a local origin. Center in `float64`, cast last.
Casting first silently quantizes your geometry to decimetre-or-worse steps and
can destroy a model's predictions without raising a single error.

## What happened

A tree-vs-background segmentation model scored **0.95 IoU** on the training
dataset (local coordinates) and then collapsed to garbage on the target dataset
(UTM coordinates) — canopy scored *lowest* tree-probability, the exact opposite
of correct. It looked like a domain-gap problem. It was not. The target
coordinates had been quantized to 0.5 m before the model ever saw them.

## Why it happens

`float32` has a 24-bit mantissa: about 7 significant decimal digits. The gap
between representable values (the ULP) grows with magnitude. At the large
absolute values typical of projected CRS coordinates, that gap becomes larger
than the detail you care about.

Measured ULP at realistic UTM magnitudes:

| coordinate       | example value | float32 spacing (ULP) |
|------------------|---------------|-----------------------|
| UTM easting      | ~736,000      | 0.0625 m              |
| UTM northing     | ~4,767,000    | 0.5 m                 |
| local / centered | < ~1,000      | < 0.0001 m            |

Northing is the killer: ~4.77 million lands in a float32 band where the step is
a full **0.5 m**. Easting (~7.36e5) quantizes to ~0.06 m. The result is
**anisotropic** — different step per axis — so it does not even look like clean
isotropic noise; it shreds the cloud into sheets.

## The bug, concretely

```python
# WRONG — cast, then center
xyz = np.asarray(xyz, dtype=np.float32)   # quantization already happened here
xyz = xyz - xyz.mean(0, keepdims=True)    # subtracting AFTER cannot recover it
```

Simulation on a 1 m canopy patch of 20,000 points at UTM magnitudes:

```
cast-then-center (the bug)
  distinct values per axis: x=17  y=3  z=19,242
  y quantization step: 0.5000 m
  max geometric error: 166.75 m
```

20,000 points collapsed to **3 distinct northing values**. The geometry is gone
before centering runs, and centering a already-quantized array does nothing.

## The fix

```python
# RIGHT — center in float64, cast last
xyz = np.asarray(xyz, dtype=np.float64)   # keep full precision
xyz = xyz - xyz.mean(0, keepdims=True)    # now values are small (< tile size)
xyz = xyz.astype(np.float32)              # cast the SMALL, centered values
```

Same simulation, fixed order:

```
center-then-cast (the fix)
  distinct values per axis: x=19,999  y=19,999  z=19,996
  y quantization step: ~0 m
  max geometric error: 0.0000 m
```

The reason it works: after subtracting the origin, coordinates are on the order
of the tile size (metres to hundreds of metres), where float32 ULP is
micrometres. The cast is then lossless for any purpose.

## Why it was hard to spot

- **No error, no warning.** `astype(np.float32)` succeeds silently.
- **Asymmetric.** Only the georeferenced dataset was affected. The training data
  used local coordinates (range roughly -300..300), where float32 is fine, so
  the *same code path* mangled one dataset and left the other pristine — which
  looks exactly like a domain gap.
- **Plausible wrong story.** "Indoor-pretrained model fails on outdoor data" is
  a real phenomenon with real literature, so the symptom matched a believable
  explanation and sent debugging in the wrong direction for several iterations.
- **Downstream corruption.** Estimated normals, grid subsampling, and any
  neighborhood feature were all computed on the shredded cloud, so every
  derived quantity was wrong too.

## Red flags that should trigger this check

- Coordinate values in the hundreds of thousands / millions (UTM, State Plane,
  national grids, ECEF).
- A model that works on one point-cloud source and fails inexplicably on
  another that "should" be similar.
- Axis labels in plots showing large offsets (e.g. matplotlib's `+4.767e6`).
- Predictions that are confidently *wrong* (saturated to the wrong class) rather
  than merely uncertain — a sign the input geometry is corrupted, not just
  out-of-distribution.

## Quick diagnostic

```python
import numpy as np
# Print float32 spacing at your actual coordinate magnitudes:
for name, v in [('x', xyz[:, 0].mean()),
                ('y', xyz[:, 1].mean()),
                ('z', xyz[:, 2].mean())]:
    print(f'{name}: value ~{v:,.0f}  float32 ULP = {np.spacing(np.float32(v)):.4f} m')
```

If any ULP is a meaningful fraction of your feature size (voxel size, minimum
branch/trunk width, etc.), you must center before casting.

## General rule

Keep georeferenced coordinates in `float64` for every step that involves their
absolute magnitude — reading, centering, tiling by absolute position. Only cast
to `float32` after you have reduced them to small, origin-relative values.
Applies to any framework (NumPy, PyTorch, Open3D) and any projected CRS, not
just this pipeline.
