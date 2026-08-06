#!/usr/bin/env python3
"""Sonata stage, part A — train the linear vegetation head on combined sources.

Frozen Sonata features + a small linear head, trained to separate
**vegetation (trees + shrubs/bushes) from everything else** — two classes —
from the union of two labelled sources:

  * WHU-STree (MLS street trees): positive = tree instances. Matches the campus
    sensor domain but has no shrub label.
  * Semantic3D (TLS): positive = high vegetation + low vegetation (labels 3 & 4).
    Terrestrial, so its geometry is close to the MLS target, and its low-veg
    class supplies the shrub/bush supervision WHU-STree lacks.

Both are mapped to the same binary target and **density-normalized into one
common regime** (the WHU density) before feature extraction, so the frozen
encoder sees a consistent sampling across sources. That target density is saved
in the checkpoint and reused verbatim by run_mask.py to match the campus cloud —
train and inference share one density regime by construction.

Set TRAIN_SOURCES=whu (or semantic3d) to train on a single source.
Caches the trained head to MODELS_DIR; set FORCE_TRAIN=1 to retrain.
"""
from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/app")
from common import config as C
from common import geo
import sonata_lib as SL

TRAIN_TILE_SIZE = 30.0


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, dim, n=2):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.fc = nn.Linear(dim, n)

    def forward(self, x):
        return self.fc(self.norm(x))


def iou_pos(p, y):
    inter = ((p == 1) & (y == 1)).sum()
    union = ((p == 1) | (y == 1)).sum()
    return float(inter) / max(1, int(union))


def tile_scene(xyz, inten, lab, tile, min_pts=8000, max_tiles=999):
    o = xyz[:, :2].min(0)
    ij = np.floor((xyz[:, :2] - o) / tile).astype(np.int64)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    out = []
    for k in np.random.permutation(len(keys)):
        m = inv == k
        if m.sum() < min_pts:
            continue
        out.append((xyz[m], inten[m], lab[m]))
        if len(out) >= max_tiles:
            break
    return out


# ---------------------------------------------------------------------------
# source 1: WHU-STree
# ---------------------------------------------------------------------------
def fetch_whu():
    import gdown, json
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    whu_dir = C.RAW_DIR / "whu"
    whu_dir.mkdir(parents=True, exist_ok=True)

    manifest = C.CACHE_DIR / "drive_manifest.json"
    if manifest.exists():
        listing = json.loads(manifest.read_text())
    else:
        files = gdown.download_folder(url=C.WHU_DRIVE_URL, skip_download=True, quiet=True)
        listing = [{"id": f.id, "path": f.path} for f in files]
        manifest.write_text(json.dumps(listing))
    plys = [f for f in listing
            if f["path"].lower().endswith(".ply") and C.WHU_CITY.lower() in f["path"].lower()]
    plys.sort(key=lambda f: f["path"])
    print(f"[whu] {len(plys)} ply in {C.WHU_CITY}")

    got = []
    for f in plys[:C.WHU_MAX_PLY]:
        parts = Path(f["path"]).parts
        tile = next((p for p in parts if p.isdigit()), "x")
        dst = whu_dir / f"{tile}_{Path(f['path']).name}"
        if not dst.exists():
            gdown.download(id=f["id"], output=str(dst), quiet=False)
        got.append(dst)
    return got


def _read_ply_scene(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    inten = np.asarray(v["intensity"], dtype=np.float32)
    inst = np.asarray(v["tree"], dtype=np.int64)
    return xyz, inten, inst


def whu_tiles():
    """WHU tiles at the base grid. positive = tree instances (inst>0)."""
    tiles = []
    got = fetch_whu()
    per_file = C.SEM3D_MAX_TILES  # reuse as a generous per-file cap
    for p in got:
        xyz, inten, inst = _read_ply_scene(p)
        veg = (inst > 0).astype(np.int64)
        xyz, (inten, veg) = geo.grid_subsample(xyz, [inten, veg], C.GRID_SIZE)
        ts = tile_scene(xyz, inten, veg, TRAIN_TILE_SIZE, max_tiles=per_file)
        tiles += ts
        print(f"[whu] {p.name}: veg frac {veg.mean():.3f} -> {len(ts)} tiles")
    return tiles


def reference_density():
    """Common target density = median occupied density of the WHU tiles at the
    base grid. This anchors both training-source normalization and the campus
    density match, so everything the frozen encoder sees is one regime."""
    dens = [geo.occupied_density(t[0]) for t in whu_tiles()]
    return float(np.median(dens)) if dens else 400.0


# ---------------------------------------------------------------------------
# source 2: Semantic3D
# ---------------------------------------------------------------------------
def _sem3d_point_stem(station):
    # sg27/sg28 files are named *_intensity_rgb; the rest *_xyz_intensity_rgb
    suffix = "_intensity_rgb" if station.startswith(("sg27", "sg28")) else "_xyz_intensity_rgb"
    return station + suffix


def _extract_7z(archive: Path, dest: Path):
    import py7zr
    dest.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(str(archive), "r") as z:
        z.extractall(path=str(dest))


def _download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[semantic3d] downloading {url}")
    # semantic3d.net serves a mismatched TLS cert (phd-apache.ethz.ch); allow
    # unverified so http->https redirects don't hard-fail. Point files are large.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, timeout=60, context=ctx) as r, open(dst, "wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)


def _find(dirp: Path, *names):
    for n in names:
        hits = sorted(dirp.rglob(n))
        if hits:
            return hits[0]
    return None


def _ensure_sem3d(station):
    """Return (points_txt, labels) paths for a station, or None if unavailable.
    Prefers already-extracted files; extracts local .7z; downloads only if
    SEM3D_DOWNLOAD=1."""
    d = C.SEM3D_DIR
    d.mkdir(parents=True, exist_ok=True)
    stem = _sem3d_point_stem(station)

    pts = _find(d, f"{stem}.txt", f"{station}.txt", f"{station}*.txt")
    lbl = _find(d, f"{stem}.labels", f"{station}.labels", f"{station}*.labels")

    # extract a local point .7z if the .txt is missing
    if pts is None:
        arc = _find(d, f"{stem}.7z", f"{station}*.7z")
        if arc is not None:
            _extract_7z(arc, d)
            pts = _find(d, f"{stem}.txt", f"{station}*.txt")
    # extract the shared labels archive if the .labels is missing
    if lbl is None:
        larc = _find(d, "sem8_labels_training.7z")
        if larc is not None:
            _extract_7z(larc, d)
            lbl = _find(d, f"{stem}.labels", f"{station}*.labels")

    # last resort: download
    if (pts is None or lbl is None) and C.SEM3D_DOWNLOAD:
        try:
            if pts is None:
                arc = d / f"{stem}.7z"
                if not arc.exists():
                    _download(f"{C.SEM3D_BASE_URL}/point-clouds/training1/{stem}.7z", arc)
                _extract_7z(arc, d)
                pts = _find(d, f"{stem}.txt", f"{station}*.txt")
            if lbl is None:
                larc = d / "sem8_labels_training.7z"
                if not larc.exists():
                    _download(f"{C.SEM3D_BASE_URL}/sem8_labels_training.7z", larc)
                _extract_7z(larc, d)
                lbl = _find(d, f"{stem}.labels", f"{station}*.labels")
        except Exception as e:
            print(f"[semantic3d] download/extract failed for {station}: {e}")

    if pts is None or lbl is None:
        print(f"[semantic3d] {station}: files not found in {d} "
              f"(txt={pts is not None}, labels={lbl is not None}) — skipping")
        return None
    return pts, lbl


def _read_sem3d(pts_path: Path, lbl_path: Path):
    """Stream x/y/z/intensity + labels, drop unlabelled (0), subsample to the
    cap. Returns (xyz float64, intensity float32, veg int64) where veg = label
    in {3,4}. RGB is intentionally dropped — the pipeline neutralizes colour."""
    import pandas as pd
    rng = np.random.default_rng(0)
    xs, ints, labs, total = [], [], [], 0
    pit = pd.read_csv(pts_path, sep=r"\s+", header=None, usecols=[0, 1, 2, 3],
                      names=list("xyzi"), dtype=np.float64,
                      chunksize=2_000_000, engine="c")
    lit = pd.read_csv(lbl_path, header=None, names=["l"], dtype=np.int64,
                      chunksize=2_000_000, engine="c")
    for pc, lc in zip(pit, lit):
        p = pc.to_numpy()
        l = lc["l"].to_numpy()
        n = min(len(p), len(l))
        p, l = p[:n], l[:n]
        m = l != 0                       # drop unlabelled
        if not m.any():
            continue
        p, l = p[m], l[m]
        if C.SEM3D_KEEP_FRAC < 1.0:      # per-chunk subsample to bound memory
            keep = rng.random(len(p)) < C.SEM3D_KEEP_FRAC
            p, l = p[keep], l[keep]
        xs.append(p[:, :3]); ints.append(p[:, 3].astype(np.float32)); labs.append(l)
        total += len(p)
    if not xs:
        return None
    xyz = np.concatenate(xs)
    inten = np.concatenate(ints)
    lab = np.concatenate(labs)
    if len(xyz) > C.SEM3D_MAX_PTS:
        sel = rng.choice(len(xyz), C.SEM3D_MAX_PTS, replace=False)
        xyz, inten, lab = xyz[sel], inten[sel], lab[sel]
    veg = np.isin(lab, C.SEM3D_VEG_LABELS).astype(np.int64)
    return xyz, inten, veg


def semantic3d_tiles(target_density):
    """Load configured stations, normalize each to `target_density`, tile."""
    tiles = []
    for st in C.SEM3D_STATIONS:
        paths = _ensure_sem3d(st)
        if paths is None:
            continue
        got = _read_sem3d(*paths)
        if got is None:
            continue
        xyz, inten, veg = got
        d0 = geo.occupied_density(xyz)
        grid = C.GRID_SIZE * np.sqrt(max(d0, 1) / max(target_density, 1))
        xyz, (inten, veg) = geo.grid_subsample(xyz, [inten, veg], grid)
        ts = tile_scene(xyz, inten, veg, TRAIN_TILE_SIZE, max_tiles=C.SEM3D_MAX_TILES)
        tiles += ts
        print(f"[semantic3d] {st}: {len(xyz):,} pts @ {geo.occupied_density(xyz):.0f} "
              f"pts/m^2 | veg frac {veg.mean():.3f} -> {len(ts)} tiles")
    return tiles


# ---------------------------------------------------------------------------
# combined tile set
# ---------------------------------------------------------------------------
def build_tiles():
    """Combine all TRAIN_SOURCES into one binary-vegetation tile set at a common
    density. Returns (tiles_tr, tiles_va, target_density)."""
    sources = C.TRAIN_SOURCES
    print(f"[train] sources: {sources}")

    target = reference_density() if "whu" in sources else None

    tiles = []
    if "whu" in sources:
        tiles += whu_tiles()
    if "semantic3d" in sources:
        if target is None:  # no WHU anchor; derive target from Semantic3D itself
            probe = semantic3d_tiles(400.0)
            target = float(np.median([geo.occupied_density(t[0]) for t in probe])) if probe else 400.0
            tiles += probe
        else:
            tiles += semantic3d_tiles(target)
    if target is None:
        target = 400.0

    assert tiles, "no training tiles built from any source"
    rng = np.random.default_rng(0)
    tiles = [tiles[i] for i in rng.permutation(len(tiles))]
    split = int(len(tiles) * 0.8)
    tr, va = tiles[:split], tiles[split:]
    print(f"[train] {len(tiles)} tiles (target density {target:.0f} pts/m^2) "
          f"| train {len(tr)} | val {len(va)}")
    print(f"[train] mean veg frac — train {np.mean([t[2].mean() for t in tr]):.3f} "
          f"| val {np.mean([t[2].mean() for t in va]):.3f}")
    return tr, va, target


# ---------------------------------------------------------------------------
# features + training
# ---------------------------------------------------------------------------
def build_set(model, transform, tiles, sub=40000):
    from tqdm.auto import tqdm
    X, Y = [], []
    for xyz, inten, lab in tqdm(tiles):
        for f, y in SL.feat_recursive(model, transform, xyz, inten, lab):
            n = min(sub, len(y))
            sel = np.random.choice(len(y), n, replace=False)
            X.append(f[sel]); Y.append(y[sel])
        SL.gpu_free()
    return np.concatenate(X), np.concatenate(Y)


def main():
    if C.SONATA_HEAD.exists() and not os.environ.get("FORCE_TRAIN"):
        print(f"[train] head already trained at {C.SONATA_HEAD} — reuse. "
              "Set FORCE_TRAIN=1 to retrain.")
        return

    torch.manual_seed(0); np.random.seed(0)
    model, transform = SL.load_model_and_transform(C.GRID_SIZE)
    tiles_tr, tiles_va, target_density = build_tiles()

    Xtr, Ytr = build_set(model, transform, tiles_tr)
    Xva, Yva = build_set(model, transform, tiles_va)
    feat_dim = Xtr.shape[1]
    print(f"[train] feat dim {feat_dim} | train {Xtr.shape} veg {Ytr.mean():.3f} "
          f"| val {Xva.shape} veg {Yva.mean():.3f}")

    head = LinearHead(feat_dim).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(Ytr.mean())
    w = torch.tensor([pos, 1 - pos]).float().cuda()
    w = w / w.sum() * 2
    crit = nn.CrossEntropyLoss(weight=w)

    Xtr_t, Ytr_t = torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).long()
    Xva_t = torch.from_numpy(Xva).float().cuda()

    best, best_state = 0.0, None
    for ep in range(int(os.environ.get("EPOCHS", "30"))):
        head.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 8192):
            idx = perm[i:i + 8192]
            xb, yb = Xtr_t[idx].cuda(), Ytr_t[idx].cuda()
            opt.zero_grad()
            crit(head(xb), yb).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xva_t).argmax(1).cpu().numpy()
        iou = iou_pos(pv, Yva)
        if iou > best:
            best, best_state = iou, {k: v.clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0:
            print(f"[train] ep {ep:2d} | val veg IoU {iou:.4f}")

    C.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feat_dim": feat_dim, "iou": best,
                "target_density": target_density, "sources": C.TRAIN_SOURCES,
                "grid_size": C.GRID_SIZE},
               str(C.SONATA_HEAD))
    print(f"[train] best val vegetation IoU {best:.4f} "
          f"(sources={C.TRAIN_SOURCES}, target density {target_density:.0f}) "
          f"-> saved {C.SONATA_HEAD}")


if __name__ == "__main__":
    main()
