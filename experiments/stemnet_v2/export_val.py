#!/usr/bin/env python3
"""StemNet v2: the val square as LAZ, with both models' predictions on every
point. CPU; reads the val predictions v2_compare (or v2_validate) saved, so
nothing is predicted again and the numbers are the ones the spreadsheet scored.

Each model's detection settings are chosen on val exactly as v2_compare does:
the gate threshold with the best AP@0.5 against the survey, then the
vote-mass cut with the best F1 on that curve.

Outputs:
  results/stemnet_v2_val.laz   ONE file: every grid point of the val square
      (UTM x / y / z, scan RGB) followed by a vertical post of points for every
      base, 3 m tall from the lowest scan point within ~0.3 m.
      marker (also the LAS classification): 0 scan point, 1 base-only detection
      post (blue), 2 base + tree detection post (red), 3 survey base post (yellow)
      scan points:
        gate_base_only, gate_base_tree   "close to a tree base" probability, per model
        vote_base_only, vote_base_tree   id of the detected base this point's vote
                                         landed on (within 0.15 m of it), -1 none:
                                         the points that make each detection
        tree_prob, pred_tree             base + tree model's tree output (>= 0.5)
        label_tree                       campus_pred_B24 label (1 tree, 0 non-tree,
                                         2 removed ground, -1 unknown)
        tree_vs_b24                      0 tree agree, 1 non-tree agree, 2 said tree /
                                         B24 non-tree, 3 said non-tree / B24 tree,
                                         4 no B24 label
      posts: base_id (the vote_* id of a detection, or the survey base), vote_mass,
        matched (detection: 1 if matched to a survey base at 0.5 m; survey base:
        1 found by base-only, 2 by base + tree, 3 by both, 0 by neither)
  results/stemnet_v2_val_bases.csv    the same bases as a table
  results/stemnet_v2_val_laz.txt      settings and counts
Models: V2_EXPORT_CKPTS (default head_best,head_tree_best).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS, TILE  # noqa: E402
import curves as C  # noqa: E402
import vote as V  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
GRID_DIR = OUT / f"grid{GRID:g}"
CKPTS = [c.strip().replace(".pth", "") for c in os.environ.get("V2_EXPORT_CKPTS", "head_best,head_tree_best").split(",")
         if c.strip()]
SEP = float(os.environ.get("V2_CURVE_SEP", "0.3"))
POST_H, POST_STEP = 3.0, 0.01
POST_RGB = np.array([[0, 0, 0], [0, 90, 255], [255, 0, 0], [255, 220, 0]], np.uint8)   # by marker
LOG = RESULTS / "stemnet_v2_val_laz.txt"


def log(msg):
    line = f"[v2-val-laz] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def pack(i, lo, span):
    v = i - lo
    return (v[:, 0] * span[1] + v[:, 1]) * span[2] + v[:, 2]


def lookup(ref_keys, q_keys):
    """Row of each query key in ref_keys, -1 where absent."""
    o = np.argsort(ref_keys, kind="stable")
    pos = np.clip(np.searchsorted(ref_keys[o], q_keys), 0, max(len(ref_keys) - 1, 0))
    ok = (ref_keys[o][pos] == q_keys) if len(ref_keys) else np.zeros(len(q_keys), bool)
    return np.where(ok, o[pos], -1)


def load_model(name):
    d = OUT / "predictions" / name / "val"
    if not (d / "stamp.json").exists():
        raise SystemExit(f"[v2-val-laz] no saved val predictions in {d} — run v2_compare (or v2_validate) first")
    R = {k: np.load(d / f"{k}.npy") for k in ("xy", "z", "p", "off", "tree", "sem")}
    R["has_tree"] = bool(np.isfinite(R["tree"][: min(len(R["tree"]), 100000)].astype(np.float32)).any())
    R["label"] = "base_tree" if R["has_tree"] else "base_only"
    R["stamp"] = json.loads((d / "stamp.json").read_text())
    return R


def choose(R, gt, split):
    """v2_compare's val choice: gate by AP@0.5, then the vote-mass cut with the best F1."""
    xy, p, off = R["xy"].astype(np.float64), R["p"], R["off"].astype(np.float64)
    best = None
    for pt in C.P_GRID:
        det, sc, nv = V.detect(xy, p, off, p_thresh=pt, peak_sep=SEP, peak_min=C.MIN_MASS)
        ign = C.split_at(split, det) != 1 if len(det) else None
        c = C.curve(det, sc, gt, 0.5, ign)
        a = C.ap(c)
        if best is None or a > best[1]:
            best = (pt, a, det, sc, nv, c)
    pt, a, det, sc, nv, c = best
    if len(c["tp"]):
        rank = np.arange(1, len(c["tp"]) + 1)
        k = int(np.argmax(2 * c["tp"] / (rank + c["n_gt"])))
        cut = float(c["score"][k])
    else:
        cut = np.inf
    keep = sc >= cut
    return dict(gate=pt, ap=a, cut=cut, det=det[keep], score=sc[keep], votes=nv[keep])


def vote_ids(R, ch):
    """Id of the kept detection each gated point votes for (its vote within PEAK_R), -1 none."""
    from scipy.spatial import cKDTree
    out = np.full(len(R["p"]), -1, np.int32)
    if not len(ch["det"]):
        return out
    g = np.flatnonzero(R["p"] >= ch["gate"])
    v = R["xy"][g].astype(np.float64) + R["off"][g].astype(np.float64)
    d, i = cKDTree(ch["det"]).query(v, distance_upper_bound=V.PEAK_R)
    ok = np.isfinite(d)
    out[g[ok]] = i[ok]
    return out


def grid_rgb(xy, z):
    """Scan RGB for each point, from the grid tiles, joined on exact coordinates. -> rgb, joined mask."""
    lo, hi = xy.min(0), xy.max(0)
    G, RGB = [], []
    for a in range(int(np.floor(lo[0] / TILE)), int(np.floor(hi[0] / TILE)) + 1):
        for b in range(int(np.floor(lo[1] / TILE)), int(np.floor(hi[1] / TILE)) + 1):
            q = GRID_DIR / f"tile_{a}_{b}.npz"
            if q.exists():
                g = np.load(q)
                G.append(g["xyz"]); RGB.append(g["rgb"])
    rgb = np.zeros((len(xy), 3), np.uint8)
    if not G:
        return rgb, np.zeros(len(xy), bool)
    g, grgb = np.concatenate(G), np.concatenate(RGB)
    a = np.round(np.column_stack([xy[:, 0], xy[:, 1], z]).astype(np.float64) * 1000).astype(np.int64)
    b = np.round(g.astype(np.float64) * 1000).astype(np.int64)
    lo_ = np.minimum(a.min(0), b.min(0)); span = np.maximum(a.max(0), b.max(0)) - lo_ + 1
    row = lookup(pack(b, lo_, span), pack(a, lo_, span))
    ok = row >= 0
    rgb[ok] = grgb[row[ok]]
    return rgb, ok


def header(origin, extra):
    import laspy
    hdr = laspy.LasHeader(point_format=7, version="1.4")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.floor(np.asarray(origin, np.float64))
    hdr.add_extra_dims([laspy.ExtraBytesParams(name=n, type=t) for n, t in extra])
    return hdr


def main():
    import laspy
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    frame = json.loads((OUT / "frame.json").read_text())
    origin = np.asarray(frame["origin_m"], np.float64)
    regions = np.load(OUT / "regions.npz")
    split = regions["split"]
    rows = list(csv.DictReader(open(OUT / "label_bases.csv")))
    gxy_all = np.array([[float(r["x"]), float(r["y"])] for r in rows]).reshape(-1, 2)
    vm = C.split_at(split, gxy_all) == 1
    gxy, gid = gxy_all[vm], np.array([int(r["base"]) for r in rows])[vm]

    models = []
    for name in CKPTS:
        R = load_model(name)
        ch = choose(R, gxy, split)
        m = V.match(ch["det"], ch["score"], gxy, 0.5, ignore=C.split_at(split, ch["det"]) != 1 if len(ch["det"]) else None)
        ch["pair"] = m["pair"]
        log(f"{name} ({R['label']}, step {R['stamp'].get('step')}): {len(R['p']):,} points | gate >= {ch['gate']:g} "
            f"(val AP@0.5 {ch['ap']:.3f}), vote mass >= {ch['cut']:.0f} -> {len(ch['det'])} detections: "
            f"{m['tp']} matched / {m['fp']} not / {m['fn']} survey bases missed (P {m['precision']:.3f} R {m['recall']:.3f})")
        models.append((name, R, ch))

    # every model's points in the first model's order (exact-coordinate join)
    ref = models[0][1]
    a = np.round(np.column_stack([ref["xy"][:, 0], ref["xy"][:, 1], ref["z"]]).astype(np.float64) * 1000).astype(np.int64)
    lo_, span = a.min(0), a.max(0) - a.min(0) + 1
    ka = pack(a, lo_, span)
    fields = {}
    for name, R, ch in models:
        if R is ref:
            idx = np.arange(len(ref["p"]))
        else:
            b = np.round(np.column_stack([R["xy"][:, 0], R["xy"][:, 1], R["z"]]).astype(np.float64) * 1000).astype(np.int64)
            inside = np.all((b >= lo_) & (b < lo_ + span), 1)
            kb = np.full(len(b), -1, np.int64); kb[inside] = pack(b[inside], lo_, span)
            j = lookup(kb, ka)
            if (j < 0).any():
                log(f"!! {(j < 0).sum():,} points of {models[0][0]} not in {name}'s predictions — left at -1 / 0")
            idx = j
        ok = idx >= 0
        gate = np.zeros(len(ka), np.float32); gate[ok] = R["p"][idx[ok]]
        vid = np.full(len(ka), -1, np.int32); vid[ok] = vote_ids(R, ch)[idx[ok]]
        fields[f"gate_{R['label']}"] = gate
        fields[f"vote_{R['label']}"] = vid
        if R["has_tree"]:
            t = np.zeros(len(ka), np.float32); t[ok] = R["tree"][idx[ok]].astype(np.float32)
            fields["tree_prob"] = t
            fields["pred_tree"] = (t >= 0.5).astype(np.uint8)
    sem = ref["sem"].astype(np.int8)
    fields["label_tree"] = sem
    if "tree_prob" in fields:
        pred, y, lab = fields["tree_prob"] >= 0.5, sem == 1, sem >= 0
        c = np.full(len(sem), 4, np.uint8)
        c[lab & pred & y] = 0; c[lab & ~pred & ~y] = 1; c[lab & pred & ~y] = 2; c[lab & ~pred & y] = 3
        fields["tree_vs_b24"] = c
    rgb, joined = grid_rgb(ref["xy"], ref["z"])
    log(f"scan RGB joined for {joined.sum():,} of {len(joined):,} points ({joined.mean() * 100:.2f}%)")

    # bases as posts
    cell = 0.25                                        # lowest scan point per 25 cm cell -> post foot
    c0 = ref["xy"].min(0).astype(np.float64)
    ci = np.floor((ref["xy"] - c0) / cell).astype(np.int64)
    nx, ny = ci.max(0) + 1
    zmin = np.full(nx * ny, np.inf, np.float32)
    np.minimum.at(zmin, ci[:, 1] * nx + ci[:, 0], ref["z"])
    zmin = zmin.reshape(ny, nx)
    posts, table = [], []

    def post(x, y, kind, bid, mass, matched):
        i, j = int(np.floor((x - c0[0]) / cell)), int(np.floor((y - c0[1]) / cell))
        win = zmin[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2]
        z0 = float(win.min()) if win.size and np.isfinite(win).any() else float(np.median(ref["z"]))
        zz = np.arange(0.0, POST_H + 1e-9, POST_STEP) + z0
        posts.append(np.column_stack([np.full(len(zz), x), np.full(len(zz), y), zz, np.full(len(zz), kind),
                                      np.full(len(zz), bid), np.full(len(zz), mass), np.full(len(zz), matched)]))

    found = np.zeros(len(gxy), np.int64)
    for k, (name, R, ch) in enumerate(models):
        kind = 2 if R["has_tree"] else 1
        for i, ((x, y), s_, nv) in enumerate(zip(ch["det"], ch["score"], ch["votes"])):
            pr = int(ch["pair"][i])
            post(x, y, kind, i, s_, int(pr >= 0))
            if pr >= 0:
                found[pr] |= kind
            table.append([R["label"], i, f"{x:.3f}", f"{y:.3f}", f"{x + origin[0]:.3f}", f"{y + origin[1]:.3f}", f"{s_:.2f}",
                          int(nv), int(gid[pr]) if pr >= 0 else pr])
    for j, (x, y) in enumerate(gxy):
        post(x, y, 3, int(gid[j]), 0.0, int(found[j]))
        table.append(["survey", int(gid[j]), f"{x:.3f}", f"{y:.3f}", f"{x + origin[0]:.3f}", f"{y + origin[1]:.3f}", "", "",
                      int(found[j])])
    P = np.concatenate(posts) if posts else np.zeros((0, 7))

    # one file: the scan points, then the posts; a post carries no prediction (gate 0, vote -1, ...)
    n, m = len(ka), len(P)
    out = {}
    for k, v in fields.items():
        fill = -1 if k.startswith("vote_") or k == "label_tree" else (4 if k == "tree_vs_b24" else 0)
        out[k] = np.concatenate([v, np.full(m, fill, v.dtype)])
    out["marker"] = np.concatenate([np.zeros(n, np.uint8), P[:, 3].astype(np.uint8)])
    out["base_id"] = np.concatenate([np.full(n, -1, np.int32), P[:, 4].astype(np.int32)])
    out["vote_mass"] = np.concatenate([np.zeros(n, np.float32), P[:, 5].astype(np.float32)])
    out["matched"] = np.concatenate([np.zeros(n, np.uint8), P[:, 6].astype(np.uint8)])
    hdr = header(origin, [(k, v.dtype) for k, v in out.items()])
    las = laspy.LasData(hdr)
    las.x = np.r_[ref["xy"][:, 0].astype(np.float64), P[:, 0]] + origin[0]
    las.y = np.r_[ref["xy"][:, 1].astype(np.float64), P[:, 1]] + origin[1]
    las.z = np.r_[ref["z"].astype(np.float64), P[:, 2]] + origin[2]
    cc = np.r_[rgb, POST_RGB[P[:, 3].astype(np.int64)]].astype(np.uint16) * 257
    las.red, las.green, las.blue = cc[:, 0], cc[:, 1], cc[:, 2]
    las.classification = out["marker"]
    for k, v in out.items():
        las[k] = v
    las.write(str(RESULTS / "stemnet_v2_val.laz"))
    with open(RESULTS / "stemnet_v2_val_bases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "id", "x", "y", "utm_x", "utm_y", "vote_mass", "votes",
                    "matched survey base (detections; -1 false, -2 outside val) / found by bit 1 base-only, bit 2 base+tree (survey)"])
        w.writerows(table)
    log(f"-> results/stemnet_v2_val.laz: {n:,} scan points + {m:,} post points "
        f"({sum(len(ch['det']) for _, _, ch in models)} detected + {len(gxy)} survey bases) | fields {', '.join(out)}")
    log(f"   survey bases found by both models {int((found == 3).sum())}, base-only only {int((found == 1).sum())}, "
        f"base+tree only {int((found == 2).sum())}, neither {int((found == 0).sum())}")

if __name__ == "__main__":
    main()
