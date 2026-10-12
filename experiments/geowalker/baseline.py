#!/usr/bin/env python3
"""The control experiment: find the trees with no model at all.

Nothing here is learned. Two detectors from the forest-inventory literature run
over the same tiles, inside the same patch footprint, and are scored by the same
matcher as GeoWalker — so the three numbers are directly comparable, and the
question "is the network worth its 1.6 GPU-hours?" finally has an answer.

  stem  Slice the cloud at breast height (GW_STEM_LO..HI above the DEM) and look
        for dense little blobs in plan view. A trunk is a near-vertical column,
        so each candidate is then asked to prove it reaches upward: of the
        height bins between GW_STEM_COL_LO and HI within GW_STEM_COL_R of the
        candidate, what fraction hold a point? A trunk fills nearly all of them,
        a parked car or a bush fills only the bottom few, and that fraction is
        the detector's whole confidence score.

  chm   Local maxima of the canopy height model (gw_height's chm.npz), the
        classical Popescu-style variable-window filter. This is what most
        remote-sensing tree counts still do, and it has a structural blind spot
        worth seeing plainly: an understory sapling under an oak produces NO
        local maximum, so a CHM detector cannot count it at any threshold.

Why this matters for reading the GeoWalker score: half of the surveyed stems in
the patch are under 10 cm DBH and carry 4% of its basal area, and GeoWalker's
own canopy gate drops anything standing under 6 m of canopy. A large part of the
"missing" trees is therefore a target definition, not a model failure — and the
only way to tell which part is to run a detector that has no such gate.

Scored at 0.5 m and 1.0 m, and the per-tree table joins all three methods
against species and girth, because recall by stem size is the decision.

Env:
  GW_INFER_REGION   patch (default) | val | all
  GW_BASE_METHOD    both (default) | stem | chm
  GW_STEM_LO/HI     breast-height slice, default 1.0 / 3.0 m above ground
  GW_STEM_CELL      plan-view raster, default 0.10 m
  GW_STEM_MIN_PTS   points in the slice blob, default 6
  GW_STEM_SEP       minimum separation of two stems, default 0.6 m
  GW_STEM_COL_R/LO/HI  verticality column: 0.35 m radius, 0.5..8.0 m
  GW_CHM_MIN_H      minimum canopy height, default = gw_height's derived cut
  GW_CHM_SEP        local-maximum window radius, default 1.5 m

Outputs (TAG is "_patch" for the patch region):
  /results/geowalker_baselineTAG_<method>_detections.csv   x,y,z,score
  /results/geowalker_baselineTAG_metrics.json              every method, one file
  /results/geowalker_baselineTAG_trees.csv       per surveyed tree, all methods
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402
from walker import match_metrics, match_pairs  # noqa: E402

BW = C.DATA / "basewalker"
GW = C.DATA / "geowalker"

REGION = os.environ.get("GW_INFER_REGION", "patch")
METHOD = os.environ.get("GW_BASE_METHOD", "both")

STEM_LO = float(os.environ.get("GW_STEM_LO", "1.0"))
STEM_HI = float(os.environ.get("GW_STEM_HI", "3.0"))
STEM_CELL = float(os.environ.get("GW_STEM_CELL", "0.10"))
STEM_MIN_PTS = int(os.environ.get("GW_STEM_MIN_PTS", "6"))
STEM_SEP = float(os.environ.get("GW_STEM_SEP", "0.6"))
COL_R = float(os.environ.get("GW_STEM_COL_R", "0.35"))
COL_LO = float(os.environ.get("GW_STEM_COL_LO", "0.5"))
COL_HI = float(os.environ.get("GW_STEM_COL_HI", "8.0"))
COL_BIN = float(os.environ.get("GW_STEM_COL_BIN", "0.5"))
CHM_SEP = float(os.environ.get("GW_CHM_SEP", "1.5"))


# ---------------------------------------------------------------------------
# ground
# ---------------------------------------------------------------------------
class Ground:
    """Nearest-cell DEM lookup. Numpy only — this stage never touches a GPU."""

    def __init__(self, path=BW / "dem.npz"):
        d = np.load(path, allow_pickle=True)
        self.z = d["z"].astype(np.float32)
        self.x0, self.y0 = float(d["x0"]), float(d["y0"])
        self.cell = float(d["cell"])
        self.ny, self.nx = self.z.shape

    def __call__(self, xy):
        i = ((xy[:, 0] - self.x0) / self.cell).astype(np.int64).clip(0, self.nx - 1)
        j = ((xy[:, 1] - self.y0) / self.cell).astype(np.int64).clip(0, self.ny - 1)
        return self.z[j, i]


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


# ---------------------------------------------------------------------------
# stem detector
# ---------------------------------------------------------------------------
def detect_stems(bbox, ground, tiles):
    """Breast-height density peaks, confirmed by a vertical column."""
    x0, x1, y0, y1 = bbox
    nx = int((x1 - x0) / STEM_CELL) + 1
    ny = int((y1 - y0) / STEM_CELL) + 1
    cnt = np.zeros((ny, nx), np.int32)
    n_in = 0
    for p in tiles:
        xyz = np.load(p)["xyz"].astype(np.float64)
        m = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1)
             & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
        if not m.any():
            continue
        xyz = xyz[m]
        h = xyz[:, 2] - ground(xyz[:, :2])
        s = xyz[(h >= STEM_LO) & (h <= STEM_HI)]
        if not len(s):
            continue
        n_in += len(s)
        ix = ((s[:, 0] - x0) / STEM_CELL).astype(np.int64).clip(0, nx - 1)
        iy = ((s[:, 1] - y0) / STEM_CELL).astype(np.int64).clip(0, ny - 1)
        np.add.at(cnt, (iy, ix), 1)

    # a trunk is wider than one 10 cm cell, so count over a ~30 cm box first
    dens = ndimage.uniform_filter(cnt.astype(np.float32), size=3) * 9.0
    cand, dval = peaks(dens, STEM_CELL, STEM_SEP, float(STEM_MIN_PTS), x0, y0)
    print(f"[gw-base] stem slice {STEM_LO}-{STEM_HI} m: {n_in:,} points | "
          f"{len(cand)} density peaks >= {STEM_MIN_PTS} pts")
    if not len(cand):
        return cand.reshape(0, 2), np.zeros(0), np.zeros(0)

    # verticality: fraction of COL_BIN height bins occupied in the column
    from scipy.spatial import cKDTree
    kd = cKDTree(cand)
    nb = int(round((COL_HI - COL_LO) / COL_BIN))
    occ = np.zeros((len(cand), nb), bool)
    for p in tiles:
        xyz = np.load(p)["xyz"].astype(np.float64)
        m = ((xyz[:, 0] >= x0 - COL_R) & (xyz[:, 0] <= x1 + COL_R)
             & (xyz[:, 1] >= y0 - COL_R) & (xyz[:, 1] <= y1 + COL_R))
        if not m.any():
            continue
        xyz = xyz[m]
        h = xyz[:, 2] - ground(xyz[:, :2])
        m = (h >= COL_LO) & (h < COL_HI)
        if not m.any():
            continue
        xyz, h = xyz[m], h[m]
        d, i = kd.query(xyz[:, :2])
        m = d <= COL_R
        if not m.any():
            continue
        b = ((h[m] - COL_LO) / COL_BIN).astype(np.int64).clip(0, nb - 1)
        occ[i[m], b] = True
    frac = occ.mean(1)
    print(f"[gw-base] verticality: p50 {np.percentile(frac,50):.2f} "
          f"p90 {np.percentile(frac,90):.2f} of {nb} bins filled")
    return cand, frac, dval


# ---------------------------------------------------------------------------
def main():
    import patch as P

    lab = np.load(BW / "labels.npz")
    bases, is_val, in_margin = lab["bases"], lab["is_val"], lab["in_margin"]
    tiles = sorted((BW / "tiles").glob("tile_*.npz"))
    if not tiles:
        raise SystemExit(f"[gw-base] no tiles in {BW/'tiles'} — run bw_prep first")

    attrs = info = None
    if REGION == "patch":
        attrs = P.load_attributes()
        sp = BW / "seeds.npz"
        seeds = np.load(sp)["seeds"] if sp.exists() else np.zeros((0, 3))
        _s, gt, rows, info = P.select(seeds, bases, is_val, in_margin,
                                      attrs=attrs)
    elif REGION == "val":
        gt = bases[is_val & ~in_margin]
    else:
        gt = bases[~in_margin]
    if not len(gt):
        raise SystemExit(f"[gw-base] region={REGION} selected no GT")
    tag = "_patch" if REGION == "patch" else ""

    pad = 2.0
    bbox = (float(gt[:, 0].min() - pad), float(gt[:, 0].max() + pad),
            float(gt[:, 1].min() - pad), float(gt[:, 1].max() + pad))
    print(f"[gw-base] region={REGION}: {len(gt)} GT bases | "
          f"bbox {bbox[1]-bbox[0]:.0f} x {bbox[3]-bbox[2]:.0f} m"
          + (f" | {info['area_ha']:.2f} ha hull" if info else ""))

    ground = Ground()
    out, dets = {}, {}

    if METHOD in ("stem", "both"):
        xy, score, dval = detect_stems(bbox, ground, tiles)
        if len(xy):
            xyz = np.column_stack([xy, ground(xy)])
            dets["stem"] = (xyz, score)

    if METHOD in ("chm", "both"):
        cp = GW / "chm.npz"
        if not cp.exists():
            print(f"[gw-base] {cp} missing — run gw_height first, skipping chm")
        else:
            c = np.load(cp)
            chm = c["chm"]
            cx0, cy0 = float(c["x0"]), float(c["y0"])
            ccell = float(c["cell"])
            hj = json.loads((C.RESULTS / "geowalker_height.json").read_text()) \
                if (C.RESULTS / "geowalker_height.json").exists() else {}
            min_h = float(os.environ.get("GW_CHM_MIN_H",
                                         hj.get("derived_cut_m", 2.0)))
            i0 = max(0, int((bbox[0] - cx0) / ccell))
            i1 = int((bbox[1] - cx0) / ccell) + 2
            j0 = max(0, int((bbox[2] - cy0) / ccell))
            j1 = int((bbox[3] - cy0) / ccell) + 2
            sub = chm[j0:j1, i0:i1]
            sm = ndimage.gaussian_filter(np.where(np.isfinite(sub), sub, 0.0),
                                         sigma=max(1.0, 0.3 / ccell))
            sm = np.where(np.isfinite(sub), sm, np.nan)
            xy, h = peaks(sm, ccell, CHM_SEP, min_h,
                          cx0 + i0 * ccell, cy0 + j0 * ccell)
            print(f"[gw-base] chm >= {min_h:.1f} m, sep {CHM_SEP} m: "
                  f"{len(xy)} maxima")
            if len(xy):
                xyz = np.column_stack([xy, ground(xy)])
                dets["chm"] = (xyz, h / max(h.max(), 1e-6))

    # ---- score every method the same way -----------------------------------
    for name, (xyz, score) in dets.items():
        np.savetxt(C.RESULTS / f"geowalker_baseline{tag}_{name}_detections.csv",
                   np.column_stack([xyz, score]), delimiter=",",
                   header="x,y,z,score", comments="")
        rec = dict(n=int(len(xyz)))
        for r in (0.5, 1.0):
            rec[f"@{r}m"] = match_metrics(xyz[:, :2], score, gt[:, :2], r)
        # best-F1 threshold, since a classical score is weak and the honest
        # headline is the unthresholded count
        best, bq = None, -1.0
        for q in np.linspace(0, 95, 20):
            t = float(np.percentile(score, q))
            k = score >= t
            if k.sum() < 5:
                continue
            m = match_metrics(xyz[k, :2], score[k], gt[:, :2], 0.5)
            if m["f1"] > bq:
                bq, best = m["f1"], dict(thresh=t, n=int(k.sum()), **m)
        rec["best_f1@0.5m"] = best
        out[name] = rec
        a = rec["@0.5m"]
        print(f"\n[gw-base] ===== {name}: {len(xyz)} detections =====")
        print(f"   @0.5 m  tp {a['tp']:4d} fp {a['fp']:4d} fn {a['fn']:4d}  "
              f"P {a['precision']:.3f} R {a['recall']:.3f} F1 {a['f1']:.3f}")
        b = rec["@1.0m"]
        print(f"   @1.0 m  tp {b['tp']:4d} fp {b['fp']:4d} fn {b['fn']:4d}  "
              f"P {b['precision']:.3f} R {b['recall']:.3f} F1 {b['f1']:.3f}")
        if best:
            print(f"   best F1 @0.5 m {best['f1']:.3f} keeping {best['n']} "
                  f"(score >= {best['thresh']:.3f}): "
                  f"P {best['precision']:.3f} R {best['recall']:.3f}")

    # ---- the per-tree answer sheet, all methods side by side ---------------
    if REGION == "patch":
        per_tree(gt, dets, attrs, tag)

    res = dict(region=REGION, method=METHOD, methods=out,
               params=dict(stem_lo=STEM_LO, stem_hi=STEM_HI, stem_cell=STEM_CELL,
                           stem_min_pts=STEM_MIN_PTS, stem_sep=STEM_SEP,
                           col_r=COL_R, col_lo=COL_LO, col_hi=COL_HI,
                           chm_sep=CHM_SEP),
               n_gt=int(len(gt)), patch=info)
    (C.RESULTS / f"geowalker_baseline{tag}_metrics.json").write_text(
        json.dumps(res, indent=2))
    print(f"\n[gw-base] -> results/geowalker_baseline{tag}_metrics.json")


def girth_to_dbh(v):
    """Girth in cm (';' separated for a multi-stem) -> the thickest stem's DBH."""
    if not v:
        return float("nan")
    try:
        g = max(float(x) for x in str(v).split(";") if x.strip())
    except ValueError:
        return float("nan")
    return g / np.pi


def per_tree(gt, dets, attrs, tag, radius=0.5):
    """One row per GT base, one found/err column per method, plus GeoWalker's
    own result if a patch run left it behind — recall by stem size is the
    decision this table exists to make."""
    import patch as P
    n = len(gt)
    cols = {}
    for name, (xyz, score) in dets.items():
        f = np.zeros(n, np.int8)
        e = np.full(n, np.nan)
        pairs, _fp, _fn = match_pairs(xyz[:, :2], score, gt[:, :2], radius)
        for i, j in pairs:
            f[j] = 1
            e[j] = float(np.linalg.norm(xyz[i, :2] - gt[j, :2]))
        cols[name] = (f, e)

    # GeoWalker's own per-tree table, if gw_infer_patch has run
    gwp = C.RESULTS / "geowalker_patch_trees.csv"
    gw_found = None
    if gwp.exists():
        import csv
        rows = list(csv.DictReader(gwp.open(newline="")))
        if len(rows) == n:
            gw_found = np.array([int(r["found"]) for r in rows], np.int8)
        else:
            print(f"[gw-base] {gwp.name} has {len(rows)} rows, GT has {n} — "
                  f"not joining (rerun gw_infer_patch)")

    jn, _ok = P.join_species(gt[:, :2], attrs=attrs)
    sp, gi = jn["species"], jn["girth_cm"]
    dbh = np.array([girth_to_dbh(v) for v in gi])

    hdr = ["x", "y", "species", "girth_cm", "dbh_cm"]
    for name in dets:
        hdr += [f"{name}_found", f"{name}_err_m"]
    if gw_found is not None:
        hdr += ["geowalker_found"]
    lines = [",".join(hdr)]
    for j in range(n):
        row = [f"{gt[j,0]:.3f}", f"{gt[j,1]:.3f}",
               (sp[j] or "").replace(",", " "), str(gi[j] or ""),
               "" if not np.isfinite(dbh[j]) else f"{dbh[j]:.1f}"]
        for name in dets:
            f, e = cols[name]
            row += [str(int(f[j])), "" if not np.isfinite(e[j]) else f"{e[j]:.3f}"]
        if gw_found is not None:
            row += [str(int(gw_found[j]))]
        lines.append(",".join(row))
    p = C.RESULTS / f"geowalker_baseline{tag}_trees.csv"
    p.write_text("\n".join(lines) + "\n")
    print(f"[gw-base] -> results/{p.name}")

    # the decisive table: recall by stem size, every method
    names = list(dets) + (["geowalker"] if gw_found is not None else [])
    def found_of(name):
        return gw_found if name == "geowalker" else cols[name][0]
    print(f"\n[gw-base] ===== recall by stem size (match {radius} m) =====")
    print("   DBH band        n   " + "  ".join(f"{k:>10}" for k in names))
    known = np.isfinite(dbh)
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        sel = known & (dbh >= lo) & (dbh < hi)
        if sel.sum() < 3:
            continue
        cells = "  ".join(f"{found_of(k)[sel].mean():>9.1%}" for k in names)
        band = f"{lo:>3.0f}-{'inf' if hi > 1e8 else f'{hi:.0f}':>3} cm"
        print(f"   {band}  {int(sel.sum()):5d}   {cells}")
    cells = "  ".join(f"{found_of(k)[~known].mean():>9.1%}" for k in names)
    print(f"   unmeasured   {int((~known).sum()):5d}   {cells}")
    cells = "  ".join(f"{found_of(k).mean():>9.1%}" for k in names)
    print(f"   ALL          {n:5d}   {cells}")

    # basal-area weighting: what fraction of the biomass did each method see?
    ba = np.where(known, (dbh / 200.0) ** 2 * np.pi, 0.0)
    print(f"\n[gw-base] basal-area-weighted recall (total {ba.sum():.2f} m2 "
          f"over {int(known.sum())} measured stems)")
    for k in names:
        print(f"   {k:>10}: {(ba*found_of(k)).sum()/max(ba.sum(),1e-9):>7.1%}")


if __name__ == "__main__":
    main()
