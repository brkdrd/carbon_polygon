"""Raster helpers shared by the detectors.

`peaks` is the local-maximum finder three different things here need: the
classical CHM filter, the classical breast-height density detector, and the
dense head's vote histogram. One implementation, so "two stems 0.6 m apart are
two detections" means the same thing in all three.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def peaks(raster, cell, sep, min_val, x0, y0):
    """Local maxima of a raster, at least `sep` metres apart.

    Ties inside a flat plateau would both survive a bare equality test, so the
    maximum filter is compared against a copy whose plateaus are broken by a
    tiny deterministic ramp.
    """
    r = max(1, int(round(sep / cell)))
    a = np.where(np.isfinite(raster), raster, -np.inf).astype(np.float64)
    a = a + np.linspace(0, 1e-6, a.size).reshape(a.shape)
    mx = ndimage.maximum_filter(a, size=2 * r + 1, mode="nearest")
    jj, ii = np.nonzero((a == mx) & (raster >= min_val) & np.isfinite(raster))
    xy = np.column_stack([x0 + (ii + 0.5) * cell, y0 + (jj + 0.5) * cell])
    return xy, raster[jj, ii].astype(np.float64)
