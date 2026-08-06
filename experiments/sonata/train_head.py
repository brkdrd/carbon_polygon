#!/usr/bin/env python3
"""Sonata stage, part A — train the linear vegetation head.

Frozen Sonata features + a small linear head, trained on WHU-STree to separate
**vegetation (trees + bushes) from everything else** — two classes.

Note on the target: WHU-STree provides per-tree instance ids (street trees) but
no separate shrub/bush class, so "bush" cannot be *supervised* from this source.
The positive class is therefore woody vegetation (tree instances); the head
learns vegetation morphology from Sonata's self-supervised features, which
transfers to bushes at inference (they are geometrically vegetation-like). This
is the honest limitation — documented in the README.

Caches the trained head to MODELS_DIR so the four experiments reuse it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/app")
from common import config as C
from common import geo
import sonata_lib as SL


# ---------------------------------------------------------------------------
# WHU-STree fetch + tiling (ported from notebook)
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
    print(f"[sonata] {len(plys)} WHU ply in {C.WHU_CITY}")

    got = []
    for f in plys[:C.WHU_MAX_PLY]:
        parts = Path(f["path"]).parts
        tile = next((p for p in parts if p.isdigit()), "x")
        dst = whu_dir / f"{tile}_{Path(f['path']).name}"
        if not dst.exists():
            gdown.download(id=f["id"], output=str(dst), quiet=False)
        got.append(dst)
    return got


def read_ply_scene(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    inten = np.asarray(v["intensity"], dtype=np.float32)
    inst = np.asarray(v["tree"], dtype=np.int64)
    return xyz, inten, inst


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


def build_tiles():
    TRAIN_TILE_SIZE, MAX_TILES = 30.0, int(os.environ.get("MAX_TRAIN_TILES", "40"))
    got = fetch_whu()
    tiles = []
    per_file = MAX_TILES // max(1, len(got)) + 2
    for p in got:
        xyz, inten, inst = read_ply_scene(p)
        lab = (inst > 0).astype(np.int64)   # vegetation = tree instances (see docstring)
        print(f"[sonata] {p.name}: {len(xyz):,} pts | veg frac {lab.mean():.3f}")
        xyz, (inten, lab) = geo.grid_subsample(xyz, [inten, lab], C.GRID_SIZE)
        tiles += tile_scene(xyz, inten, lab, TRAIN_TILE_SIZE, max_tiles=per_file)
    tiles = tiles[:MAX_TILES]
    assert tiles, "no training tiles built"
    rng = np.random.default_rng(0)
    tiles = [tiles[i] for i in rng.permutation(len(tiles))]
    split = int(len(tiles) * 0.8)
    return tiles[:split], tiles[split:]


# ---------------------------------------------------------------------------
# feature set + head
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
        print(f"[sonata] head already trained at {C.SONATA_HEAD} — reuse. "
              "Set FORCE_TRAIN=1 to retrain.")
        return

    torch.manual_seed(0); np.random.seed(0)
    model, transform = SL.load_model_and_transform(C.GRID_SIZE)
    tiles_tr, tiles_va = build_tiles()
    print(f"[sonata] train {len(tiles_tr)} | val {len(tiles_va)} tiles")

    Xtr, Ytr = build_set(model, transform, tiles_tr)
    Xva, Yva = build_set(model, transform, tiles_va)
    feat_dim = Xtr.shape[1]
    print(f"[sonata] feat dim {feat_dim} | train {Xtr.shape} veg {Ytr.mean():.3f} "
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
            print(f"[sonata] ep {ep:2d} | val veg IoU {iou:.4f}")

    C.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feat_dim": feat_dim, "iou": best},
               str(C.SONATA_HEAD))
    print(f"[sonata] best val vegetation IoU {best:.4f} -> saved {C.SONATA_HEAD}")


if __name__ == "__main__":
    main()
