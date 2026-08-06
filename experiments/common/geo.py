"""Geometry + point-cloud I/O shared by every experiment stage.

This module is the single place that owns the rule from FLOAT32_COORDINATE_BUG.md:

    Never cast georeferenced coordinates (UTM, national grids) to float32 before
    subtracting a local origin. Center in float64, cast last.

Every stage that touches coordinates goes through the helpers here, so the trap
is fixed once and cannot be reintroduced per-stage. `center_float64` is the
canonical implementation of the fix; `float32_ulp_report` is the diagnostic from
the bug write-up, run automatically whenever we ingest a georeferenced tile.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# The fix, and the diagnostic that proves we need it.
# ---------------------------------------------------------------------------


def float32_ulp_report(xyz: np.ndarray) -> dict:
    """Return the float32 spacing (ULP) at the coordinate magnitudes in `xyz`.

    Straight out of FLOAT32_COORDINATE_BUG.md: if any axis ULP is a meaningful
    fraction of the feature size (voxel, branch width) you MUST center before
    casting. UTM northing (~4.77e6) lands in a band where the float32 step is a
    full 0.5 m.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    report = {}
    for name, col in zip("xyz", range(3)):
        v = float(xyz[:, col].mean())
        ulp = float(np.spacing(np.float32(v)))
        report[name] = {"magnitude": v, "float32_ulp_m": ulp}
    return report


def warn_if_quantizing(xyz: np.ndarray, feature_size_m: float = 0.05) -> None:
    """Log a loud warning if casting these coords to float32 would quantize them
    below `feature_size_m`. Purely diagnostic — it never mutates anything."""
    rep = float32_ulp_report(xyz)
    bad = {k: v for k, v in rep.items() if v["float32_ulp_m"] >= feature_size_m}
    if bad:
        print("  [float32-guard] WARNING: casting raw coords to float32 would "
              "quantize below feature size", feature_size_m, "m:")
        for k, v in bad.items():
            print(f"      axis {k}: magnitude ~{v['magnitude']:,.0f} m  "
                  f"-> float32 ULP {v['float32_ulp_m']:.4f} m")
        print("  [float32-guard] center_float64() below removes the magnitude "
              "before any float32 cast, so this is handled — reporting for audit.")


def center_float64(xyz: np.ndarray, origin: np.ndarray | None = None):
    """THE FIX. Center coordinates in float64, return centered float64 + origin.

    Caller casts to float32 *after* this, on the small centered values, where the
    float32 ULP is micrometres. Never cast before calling this.

    Parameters
    ----------
    xyz : (N,3) array-like of georeferenced coordinates.
    origin : optional fixed (3,) origin to subtract. If None, uses the mean so
        the result is zero-centered. Pass a fixed origin to keep multiple tiles
        on a common local frame (we do this so every experiment shares a frame).

    Returns
    -------
    centered : (N,3) float64, small magnitude.
    origin   : (3,) float64, the value that was subtracted (add it back to
               recover georeferenced coordinates).
    """
    xyz = np.asarray(xyz, dtype=np.float64)  # full precision, no quantization
    if origin is None:
        origin = xyz.mean(axis=0)
    origin = np.asarray(origin, dtype=np.float64)
    centered = xyz - origin  # subtraction happens while still float64
    return centered, origin


def to_float32_local(xyz_centered: np.ndarray) -> np.ndarray:
    """Cast already-centered coordinates to float32. Safe ONLY on centered input;
    named explicitly so a reviewer can see the cast is the *last* step."""
    return np.asarray(xyz_centered, dtype=np.float32)


# ---------------------------------------------------------------------------
# LAS/LAZ I/O. laspy stores coords as int32 * scale + offset, so georeference is
# exact on disk; we keep it in float64 in memory.
# ---------------------------------------------------------------------------


def read_las(path, want_intensity=True):
    """Read a LAS/LAZ file. Returns (xyz float64, intensity float32 or None)."""
    import laspy

    with laspy.open(str(path)) as fh:
        las = fh.read()
    xyz = np.column_stack([np.asarray(las.x, dtype=np.float64),
                           np.asarray(las.y, dtype=np.float64),
                           np.asarray(las.z, dtype=np.float64)])
    inten = None
    if want_intensity and "intensity" in [d.name for d in las.point_format.dimensions]:
        inten = np.asarray(las.intensity, dtype=np.float32)
    return xyz, inten


def write_las(path, xyz, intensity=None, classification=None,
              instance=None, extra_scalars=None):
    """Write points to LAS 1.2 pf3. `xyz` may be georeferenced or local; the
    header offset absorbs the magnitude and scale=1mm keeps it exact.

    extra_scalars: dict[name -> (float32 array)] added as extra dims.
    instance: optional int per-point instance id, stored as extra dim `instance`.
    """
    import laspy

    xyz = np.asarray(xyz, dtype=np.float64)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if intensity is not None:
        las.intensity = np.clip(intensity, 0, 65535).astype(np.uint16)
    if classification is not None:
        las.classification = np.asarray(classification, dtype=np.uint8)
    extras = dict(extra_scalars or {})
    if instance is not None:
        extras["instance"] = np.asarray(instance, dtype=np.float32)
    for name, arr in extras.items():
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
        setattr(las, name, np.asarray(arr, dtype=np.float32))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))
    return path


# ---------------------------------------------------------------------------
# Subsampling / density (ported from the notebook, float64-safe).
# ---------------------------------------------------------------------------


def grid_subsample(xyz, feats, size):
    """Voxel-grid subsample: one point per `size`-metre cell. Flattens 3D cell
    indices to an int64 key before np.unique (unique(axis=0) is far slower)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    k = np.floor(xyz / size).astype(np.int64)
    k -= k.min(0)
    dim = k.max(0) + 1
    flat = k[:, 0] + k[:, 1] * dim[0] + k[:, 2] * dim[0] * dim[1]
    _, idx = np.unique(flat, return_index=True)
    idx.sort()
    return xyz[idx], [np.asarray(f)[idx] for f in feats]


def occupied_density(xyz, cell=1.0):
    """Median points per occupied `cell` x `cell` column, in pts/m^2."""
    xyz = np.asarray(xyz, dtype=np.float64)
    k = np.floor(xyz[:, :2] / cell).astype(np.int64)
    _, cnt = np.unique(k, axis=0, return_counts=True)
    return float(np.median(cnt)) / (cell ** 2)


# ---------------------------------------------------------------------------
# Chunk metadata sidecar. Keeps the local<->UTM mapping so every experiment
# shares one frame and results can be georeferenced back.
# ---------------------------------------------------------------------------


def save_meta(path, **fields):
    def _san(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: _san(v) for k, v in fields.items()}, indent=2))
    return path


def load_meta(path):
    return json.loads(Path(path).read_text())
