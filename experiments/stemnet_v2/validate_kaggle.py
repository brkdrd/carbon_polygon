#!/usr/bin/env python3
"""Evaluate the trained base-only StemNet v2 head on an independent Kaggle set.

The Kaggle dataset must contain one LAS/LAZ point cloud and either:

* a CSV whose name contains base, stem, tree, truth or gt and which has x/y,
  easting/northing, or lon/lat columns; or
* an integer LAS extra dimension named tree_id, treeID, instance_id, or
  instance. In that case a base is derived from the median of the lowest
  points of each instance.

CSV coordinates may be in the source cloud's CRS or in its local frame. The
script detects which from the cloud bounds, keeps source coordinates float64,
and subtracts the LAS origin before converting model inputs to float32.

This stage never tunes on the external data. It uses detection parameters
previously selected for head_best.pth, when available, otherwise the repository
defaults. A plot-edge band is excluded from both truth and detections.

Configuration:
  V2_EXTVAL_KAGGLE       required owner/dataset slug
  V2_EXTVAL_CLOUD        optional filename/path suffix when several clouds exist
  V2_EXTVAL_GT           optional CSV filename/path suffix
  V2_EXTVAL_EDGE         ignored edge band in metres (default 4)
  V2_EXTVAL_BACKGROUND   comma-separated instance ids to ignore (default -1,0)
  V2_EXTVAL_FORCE        1 downloads and prepares again

Outputs:
  results/stemnet_v2_kaggle_validate.txt
  results/stemnet_v2_kaggle_detections.csv
  results/60_stemnet_v2_kaggle_validate.png
  data/kaggle_validation/<owner>__<dataset>/ download and prepared cache
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import DATA, RESULTS, TILE  # noqa: E402
import data as D  # noqa: E402
import infer as I  # noqa: E402
import train as T  # noqa: E402
import vote as V  # noqa: E402

SLUG = os.environ.get("V2_EXTVAL_KAGGLE", "").strip()
CLOUD_HINT = os.environ.get("V2_EXTVAL_CLOUD", "").strip()
GT_HINT = os.environ.get("V2_EXTVAL_GT", "").strip()
EDGE = float(os.environ.get("V2_EXTVAL_EDGE", "4"))
FORCE = os.environ.get("V2_EXTVAL_FORCE", "0") != "0"
BACKGROUND = {int(x) for x in os.environ.get("V2_EXTVAL_BACKGROUND", "-1,0").split(",") if x.strip()}
LOG = RESULTS / "stemnet_v2_kaggle_validate.txt"
DET_CSV = RESULTS / "stemnet_v2_kaggle_detections.csv"
FIG = RESULTS / "60_stemnet_v2_kaggle_validate.png"
INSTANCE_NAMES = ("tree_id", "treeid", "instance_id", "instance")


def log(msg):
    line = f"[v2-kaggle-val] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def safe_slug(slug):
    parts = slug.split("/")
    if len(parts) != 2 or any(not p or not all(c.isalnum() or c in "._-" for c in p) for p in parts):
        raise SystemExit("[v2-kaggle-val] V2_EXTVAL_KAGGLE must be owner/dataset")
    return slug.replace("/", "__")


def choose(paths, hint, what, preferred=()):
    paths = sorted(paths)
    if hint:
        hit = [p for p in paths if str(p).endswith(hint)]
        if len(hit) != 1:
            raise SystemExit(f"[v2-kaggle-val] V2_EXTVAL_{what.upper()}={hint!r} matched {len(hit)} files")
        return hit[0]
    for word in preferred:
        hit = [p for p in paths if word in p.stem.lower()]
        if len(hit) == 1:
            return hit[0]
    if len(paths) != 1:
        shown = ", ".join(str(p.name) for p in paths[:10])
        raise SystemExit(f"[v2-kaggle-val] found {len(paths)} possible {what} files ({shown}); "
                         f"set V2_EXTVAL_{what.upper()} to the intended filename")
    return paths[0]


def run_kaggle(args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if r.returncode:
        tail = "\n".join((r.stdout + "\n" + r.stderr).splitlines()[-12:])
        raise SystemExit(f"[v2-kaggle-val] Kaggle command failed (exit {r.returncode}):\n{tail}")
    return r


def download(root):
    done = root / "download.complete.json"
    dst = root / "download"
    if done.exists() and dst.exists() and not FORCE:
        meta = json.loads(done.read_text())
        if meta.get("dataset") == SLUG:
            log(f"download reused: {dst}")
            return dst
    tmp = root / ".download"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    log(f"downloading Kaggle dataset {SLUG}")
    run_kaggle(["datasets", "download", SLUG, "-p", str(tmp), "--force"])
    for z in sorted(tmp.rglob("*.zip")):
        with zipfile.ZipFile(z) as f:
            root = z.parent.resolve()
            if any((root / name).resolve() != root and root not in (root / name).resolve().parents
                   for name in f.namelist()):
                raise SystemExit(f"[v2-kaggle-val] unsafe path in {z.name}")
            f.extractall(z.parent)
        z.unlink()
    clouds = list(tmp.rglob("*.las")) + list(tmp.rglob("*.laz"))
    if not clouds:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"[v2-kaggle-val] {SLUG} contains no LAS/LAZ point cloud")
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    done.write_text(json.dumps(dict(dataset=SLUG, downloaded_at=int(time.time())), indent=2))
    log(f"downloaded {len(clouds)} point cloud file(s) -> {dst}")
    return dst


def select_files(downloaded):
    clouds = list(downloaded.rglob("*.las")) + list(downloaded.rglob("*.laz"))
    cloud = choose(clouds, CLOUD_HINT, "cloud")
    csvs = list(downloaded.rglob("*.csv"))
    gt = choose(csvs, GT_HINT, "gt", ("base", "stem", "truth", "tree", "gt")) if csvs else None
    return cloud, gt


def prepared_cloud(root, cloud):
    """Run the repository's exact grid and feature preparation in an isolated DATA_DIR."""
    work = root / "work"
    raw = work / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    source = raw / cloud.name
    if (not source.exists() or source.stat().st_size != cloud.stat().st_size or
            source.stat().st_mtime_ns != cloud.stat().st_mtime_ns or FORCE):
        source.unlink(missing_ok=True)
        try:
            os.link(cloud, source)
        except OSError:
            shutil.copy2(cloud, source)
    env = dict(os.environ, DATA_DIR=str(work), RESULTS_DIR=str(root / "prep_results"),
               SOURCE_LAZ_NAME=source.name, V2_FORCE="1" if FORCE else "0")
    Path(env["RESULTS_DIR"]).mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    for script in ("prep_grid.py", "prep_features.py"):
        log(f"preparing external data with {script}")
        r = subprocess.run([sys.executable, str(here / script)], env=env, text=True)
        if r.returncode:
            raise SystemExit(f"[v2-kaggle-val] {script} failed (exit {r.returncode})")
    return work / "stemnet_v2"


def _column(names, aliases):
    low = {n.strip().lower(): n for n in names}
    return next((low[name] for name in aliases if name in low), None)


def cloud_info(cloud):
    import laspy
    with laspy.open(cloud) as f:
        h = f.header
        epsg = None
        try:
            crs = h.parse_crs()
            epsg = crs.to_epsg() if crs is not None else None
        except Exception:
            pass
        return np.asarray(h.mins, np.float64), np.asarray(h.maxs, np.float64), epsg


def csv_truth(path, cloud, frame):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"[v2-kaggle-val] {path} has no CSV header")
        fieldnames = reader.fieldnames
        rows = list(reader)
    xcol = _column(fieldnames, ("x", "utm_x", "easting", "east"))
    ycol = _column(fieldnames, ("y", "utm_y", "northing", "north"))
    lon = _column(fieldnames, ("lon", "longitude"))
    lat = _column(fieldnames, ("lat", "latitude"))
    if xcol and ycol:
        xy = np.array([[float(r[xcol]), float(r[ycol])] for r in rows], np.float64)
    elif lon and lat:
        _, _, epsg = cloud_info(cloud)
        if epsg is None:
            raise SystemExit("[v2-kaggle-val] lon/lat truth needs an EPSG CRS in the LAS header")
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = tr.transform([float(r[lon]) for r in rows], [float(r[lat]) for r in rows])
        xy = np.column_stack([x, y])
    else:
        raise SystemExit(f"[v2-kaggle-val] {path.name} needs x/y, easting/northing, or lon/lat columns")
    if not len(xy) or not np.isfinite(xy).all():
        raise SystemExit(f"[v2-kaggle-val] {path.name} has no finite tree bases")
    mins, maxs, _ = cloud_info(cloud)
    origin = np.asarray(frame["origin_m"][:2], np.float64)
    global_fit = np.mean(np.all((xy >= mins[:2] - 1) & (xy <= maxs[:2] + 1), axis=1))
    local_hi = maxs[:2] - origin
    local_fit = np.mean(np.all((xy >= -1) & (xy <= local_hi + 1), axis=1))
    if global_fit >= 0.95:
        xy -= origin
        mode = "source CRS"
    elif local_fit >= 0.95:
        mode = "prepared local frame"
    else:
        raise SystemExit(f"[v2-kaggle-val] only {max(global_fit, local_fit):.0%} of CSV bases fall inside "
                         "the point cloud in either source or local coordinates")
    idcol = _column(fieldnames, ("tree_id", "tree", "id"))
    ids = [r.get(idcol) or str(i) if idcol else str(i) for i, r in enumerate(rows)]
    return xy, ids, f"{path.name}, {mode}"


def instance_truth(cloud, frame):
    """Derive robust base XY from the lowest 128 points of each labelled instance."""
    import laspy
    with laspy.open(cloud) as f:
        dims = {str(n).lower(): str(n) for n in f.header.point_format.dimension_names}
        name = next((dims[n] for n in INSTANCE_NAMES if n in dims), None)
        if name is None:
            raise SystemExit("[v2-kaggle-val] no truth CSV and no tree_id/instance_id LAS dimension")
        low = {}
        for pts in f.chunk_iterator(2_000_000):
            ids = np.asarray(pts[name]).astype(np.int64)
            xyz = np.column_stack([np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)])
            for tree in np.unique(ids):
                if int(tree) in BACKGROUND:
                    continue
                a = xyz[ids == tree]
                keep = min(128, len(a))
                a = a[np.argpartition(a[:, 2], keep - 1)[:keep]]
                if int(tree) in low:
                    a = np.vstack([low[int(tree)], a])
                    keep = min(128, len(a))
                    a = a[np.argpartition(a[:, 2], keep - 1)[:keep]]
                low[int(tree)] = a
    if not low:
        raise SystemExit(f"[v2-kaggle-val] {name} contains no non-background instances")
    ids = sorted(low)
    origin = np.asarray(frame["origin_m"][:2], np.float64)
    xy = np.array([np.median(low[i][:, :2], axis=0) - origin for i in ids])
    return xy, [str(i) for i in ids], f"LAS dimension {name}; median XY of 128 lowest points"


def bbox_and_occupancy(store):
    xs = [i for i, _ in store.have]
    ys = [j for _, j in store.have]
    bbox = (float(min(xs) * TILE), float((max(xs) + 1) * TILE),
            float(min(ys) * TILE), float((max(ys) + 1) * TILE))
    shape = (int(np.ceil(bbox[3])), int(np.ceil(bbox[1])))
    occ = np.zeros(shape, np.uint8)
    for i, j in store.have:
        occ[max(0, int(j * TILE)):int((j + 1) * TILE),
            max(0, int(i * TILE)):int((i + 1) * TILE)] = 1
    return bbox, occ


def inset_mask(xy, bbox, edge):
    return ((xy[:, 0] >= bbox[0] + edge) & (xy[:, 0] < bbox[1] - edge) &
            (xy[:, 1] >= bbox[2] + edge) & (xy[:, 1] < bbox[3] - edge))


def figure(gt, det, pair, bbox, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    found = np.zeros(len(gt), bool)
    found[pair[pair >= 0]] = True
    fig, ax = plt.subplots(figsize=(10, 10))
    if len(gt):
        ax.scatter(gt[~found, 0], gt[~found, 1], s=18, c="tab:red", label="missed tree")
        ax.scatter(gt[found, 0], gt[found, 1], s=16, c="tab:green", label="found tree")
    if len(det):
        ax.scatter(det[pair == -1, 0], det[pair == -1, 1], s=12, marker="x", c="tab:orange",
                   label="false detection")
        ax.scatter(det[pair >= 0, 0], det[pair >= 0, 1], s=8, c="tab:blue", label="matched detection")
    ax.set(xlim=(bbox[0], bbox[1]), ylim=(bbox[2], bbox[3]), aspect="equal", title=title,
           xlabel="local x (m)", ylabel="local y (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG, dpi=180)
    plt.close(fig)


def main():
    if not SLUG:
        raise SystemExit("[v2-kaggle-val] set V2_EXTVAL_KAGGLE=owner/dataset")
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    root = DATA / "kaggle_validation" / safe_slug(SLUG)
    root.mkdir(parents=True, exist_ok=True)
    downloaded = download(root)
    cloud, gt_csv = select_files(downloaded)
    log(f"cloud: {cloud.relative_to(downloaded)} ({cloud.stat().st_size / 2**30:.2f} GiB)")
    prepared = prepared_cloud(root, cloud)
    frame = json.loads((prepared / "frame.json").read_text())
    truth, truth_id, truth_source = (csv_truth(gt_csv, cloud, frame) if gt_csv is not None
                                     else instance_truth(cloud, frame))
    log(f"truth: {len(truth):,} trees from {truth_source}")

    runner, ck, cpath = I.load_runner(log, name="head_best.pth")
    if ck.get("task", "base") != "base" or runner.head.has_tree:
        raise SystemExit(f"[v2-kaggle-val] {cpath.name} is not the base-only model")
    params = I.detect_params(cpath, log)
    store = D.TileStore(keep=I.KEEP, labels=False, root=prepared)
    bbox, occ = bbox_and_occupancy(store)
    cores = I.blocks(bbox, occ)
    rng = np.random.default_rng(T.SEED)
    detections, scores, votes = [], [], []
    t0 = time.time()
    for n, (_, _, _, det, score, nv) in enumerate(
            I.iter_blocks(runner.predict, store, cores, params, rng), 1):
        detections.append(det)
        scores.append(score)
        votes.append(nv)
        log(f"inference block {n}/{len(cores)}: {len(det)} base candidates | {(time.time() - t0) / 60:.1f} min")
    det = np.concatenate(detections) if detections else np.zeros((0, 2))
    score = np.concatenate(scores) if scores else np.zeros(0)
    nv = np.concatenate(votes) if votes else np.zeros(0, int)

    keep_gt = inset_mask(truth, bbox, EDGE)
    keep_det = inset_mask(det, bbox, EDGE)
    truth_eval, ids_eval = truth[keep_gt], np.asarray(truth_id)[keep_gt]
    det, score, nv = det[keep_det], score[keep_det], nv[keep_det]
    a = V.match(det, score, truth_eval, 0.5)
    b = V.match(det, score, truth_eval, 0.3)
    log(f"evaluation extent {bbox}, edge band {EDGE:g} m: {len(truth_eval):,}/{len(truth):,} trees, "
        f"{len(det):,} detections")
    log(f"@0.5 m F1 {a['f1']:.3f} (P {a['precision']:.3f}, R {a['recall']:.3f}; "
        f"{a['tp']} TP / {a['fp']} FP / {a['fn']} FN; RMSE {a['rmse'] * 100:.1f} cm)")
    log(f"@0.3 m F1 {b['f1']:.3f} (P {b['precision']:.3f}, R {b['recall']:.3f}; "
        f"{b['tp']} TP / {b['fp']} FP / {b['fn']} FN)")
    origin = np.asarray(frame["origin_m"], np.float64)
    with open(DET_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "source_x", "source_y", "score", "votes", "matched_tree_id"])
        for (x, y), s, n, p in zip(det, score, nv, a["pair"]):
            w.writerow([f"{x:.3f}", f"{y:.3f}", f"{x + origin[0]:.3f}", f"{y + origin[1]:.3f}",
                        f"{s:.4f}", int(n), ids_eval[p] if p >= 0 else ""])
    figure(truth_eval, det, a["pair"], bbox,
           f"Base-only StemNet v2 on {SLUG} — F1 {a['f1']:.3f}, P {a['precision']:.3f}, R {a['recall']:.3f}")
    summary = dict(dataset=SLUG, cloud=str(cloud.relative_to(downloaded)), truth=truth_source,
                   checkpoint=cpath.name, checkpoint_step=int(ck["step"]), detection=params,
                   bbox=bbox, edge_m=EDGE, trees=len(truth_eval), detections=len(det),
                   at_05={k: v for k, v in a.items() if k != "pair"},
                   at_03={k: v for k, v in b.items() if k != "pair"})
    (root / "last_result.json").write_text(json.dumps(summary, indent=2))
    log(f"-> {LOG.name}, {DET_CSV.name}, {FIG.name}")


if __name__ == "__main__":
    main()
