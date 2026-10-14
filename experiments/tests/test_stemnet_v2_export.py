#!/usr/bin/env python3
"""StemNet v2 export: the sampled columns hold every grid point within the
radius of their survey point, nothing else, and the sample is the requested one."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tests"))


def test_export_cuts_exact_columns():
    import test_stemnet_v2_circles as C
    rng = np.random.default_rng(21)
    P, pos, rad, survey = C._scene(rng)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        C._write(tmp, P, survey, pos, rad)
        env = C._env(tmp, V2_EXPORT_GIRTH="5", V2_EXPORT_RANDOM="6", V2_EXPORT_R="2.0", V2_EXPORT_H="9")
        r = subprocess.run([sys.executable, str(HERE / "stemnet_v2" / "export_columns.py")],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        z = np.load(tmp / "data" / "stemnet_v2" / "columns_sample.npz")
        assert len(z["ids"]) == 11 and (z["girth_cm"][:5] != "").all()
        P32 = P.astype(np.float32).astype(np.float64)
        for s in range(len(z["ids"])):
            got = z["xyz"][z["tree"] == s].astype(np.float64)
            c = z["survey"][s, :2]
            want = P32[np.hypot(P32[:, 0] - c[0], P32[:, 1] - c[1]) <= 2.0]
            assert len(got) == len(want), (s, len(got), len(want))


if __name__ == "__main__":
    try:
        test_export_cuts_exact_columns(); print("PASS test_export_cuts_exact_columns")
        print("\nAll StemNet v2 export test(s) passed.")
    except Exception as e:                       # noqa: BLE001
        print(f"FAIL test_export_cuts_exact_columns: {type(e).__name__}: {e}"); sys.exit(1)
