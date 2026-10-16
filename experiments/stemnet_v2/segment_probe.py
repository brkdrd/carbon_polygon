#!/usr/bin/env python3
"""StemNet v2 probe: how separable are the points by what frozen Sonata sees,
with and without the paper's 12 covariance features? No training; the
clustering never sees a label.

1. A square region (V2_SEG_SIZE m, default 24) lying wholly in the val split,
   placed where it holds the most val bases (or centred on V2_SEG_XY="x,y").
2. Tiled with 8 m crops at a 6 m stride: every point takes its vector from the
   crop whose inner 6 m square holds it, so it is >= 1 m from that crop's
   edge, as in training and eval. A crop over V2_CAP points is split into
   random disjoint parts, so every point is encoded (each part is a thinned
   copy of the crop, as a capped training crop is).
3. Per point: Sonata's dense read-out (1,232 values; frozen, colour zeros, the
   same input as training) and the 12 covariance features.
4. Three versions of the vector. Every column is standardised (z-score over
   the fit sample, clipped at +-5, so a few saturated sum_eig values cannot
   dominate the distances):
     sonata           the 1,232 Sonata columns
     sonata+feat      + the 12 features as plain columns: 12 of 1,244, so
                      they carry ~1% of the squared distance
     sonata+feat_bal  + the 12 features, each block scaled to the same total
                      variance, so the 12 carry half of it
5. k-means (k-means++ seeding, Lloyd) for each k in V2_SEG_K (default
   4,8,16), fitted on V2_SEG_FIT random points, applied to every point. Also
   the top 3 principal components, shown as colour.
6. Mapped back onto the points: plots and one PLY per version.

Separability in numbers. The gate label (within base radius + 0.25 m of a
surveyed base, -0.1..2.5 m above the ground there) is the only per-point truth.
Cluster ids are ranked by their share of near-base points (cluster 0 = the most
trunk-like), so colours mean the same in every panel. Per version and k: the
best union of clusters for the near-base points (clusters added most-pure
first, kept while F1 rises), with recall, purity and F1. Purity is a lower bound:
unsurveyed trunks are labelled background, trunk above 2.5 m is background,
and the near-base ring includes ground. Read the plots, not only the F1.

Outputs:
  results/stemnet_v2_segment.txt                per-cluster tables, F1 per version and k
  results/47_stemnet_v2_segment_top.png         top view; rows = versions,
                                                columns = labels | PCA | k ...
  results/48_stemnet_v2_segment_side.png        side view of a 3 m slab through the most bases
  results/stemnet_v2_segment_<version>.ply      x y z (frame metres), PCA colour,
                                                k<k> cluster ids, near_base (for CloudCompare)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS  # noqa: E402
import data as D  # noqa: E402
import model as M  # noqa: E402
from train import val_truth  # noqa: E402

SIZE = float(os.environ.get("V2_SEG_SIZE", "24"))
KS = [int(k) for k in os.environ.get("V2_SEG_K", "4,8,16").split(",")]
FIT = int(float(os.environ.get("V2_SEG_FIT", "200000")))
XY = os.environ.get("V2_SEG_XY", "")
CAP = int(float(os.environ.get("V2_CAP", "300000")))
DEVICE = os.environ.get("V2_DEVICE", "cuda")
ENCODER = os.environ.get("V2_ENCODER", "sonata")
SEED = int(os.environ.get("V2_SEED", "0"))
CROP, INNER, CLIP, SLAB = 8.0, 6.0, 5.0, 3.0
VERSIONS = ("sonata", "sonata+feat", "sonata+feat_bal")
LOG = RESULTS / "stemnet_v2_segment.txt"


def log(msg):
    line = f"[v2-segment] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def pick_region(regions, gt):
    """-> (x0, y0): the lower-left corner of the SIZE square."""
    if XY:
        cx, cy = (float(v) for v in XY.split(","))
        return cx - SIZE / 2, cy - SIZE / 2
    centres = D.eligible_centres(regions, SIZE, 0.45, 1)
    if not len(centres):
        raise SystemExit(f"[v2-segment] no {SIZE:g} m square fits wholly in the val split — lower V2_SEG_SIZE")
    h = SIZE / 2
    n = np.zeros(len(centres), np.int64)
    for g in gt:
        n += (np.abs(centres[:, 0] - g[0]) < h) & (np.abs(centres[:, 1] - g[1]) < h)
    c = centres[int(np.argmax(n))]
    return c[0] - h, c[1] - h


def encode_region(enc, store, x0, y0, dev, rng):
    """Every point of the region with its Sonata vector, features and gate."""
    x1, y1 = x0 + SIZE, y0 + SIZE
    parts, n_crops, n_split = [], 0, 0
    for cx in np.arange(x0 + INNER / 2, x1 + INNER / 2, INNER):
        for cy in np.arange(y0 + INNER / 2, y1 + INNER / 2, INNER):
            c = store.crop(cx, cy, CROP)
            if c is None:
                continue
            x, y = c["xyz"][:, 0], c["xyz"][:, 1]
            inner = ((x >= cx - INNER / 2) & (x < cx + INNER / 2) & (y >= cy - INNER / 2) & (y < cy + INNER / 2)
                     & (x >= x0) & (x < x1) & (y >= y0) & (y < y1))
            if not inner.any():
                continue
            n_crops += 1
            pos = np.full(len(x), -1, np.int64)
            pos[inner] = np.arange(inner.sum())
            S = np.zeros((int(inner.sum()), 1232), np.float16)
            done = np.zeros(int(inner.sum()), bool)
            k = int(np.ceil(len(x) / CAP))
            n_split += k > 1
            for sub in np.array_split(rng.permutation(len(x)), k):
                local = torch.from_numpy(M.to_local(c["xyz"][sub])).to(dev)
                normal = torch.from_numpy(np.ascontiguousarray(c["normal"][sub], np.float32)).to(dev)
                f = M.dense_features(enc, local, normal)
                keep = inner[sub]
                S[pos[sub[keep]]] = f[torch.from_numpy(keep).to(dev)].half().cpu().numpy()
                done[pos[sub[keep]]] = True
            if not done.all():
                raise RuntimeError(f"crop ({cx:.1f}, {cy:.1f}): {(~done).sum()} points got no vector")
            parts.append(dict(xyz=c["xyz"][inner], feat=c["feat"][inner].astype(np.float32),
                              gate=c["gate"][inner].astype(bool), S=S))
    if not parts:
        raise SystemExit(f"[v2-segment] no points in x {x0:.1f}-{x1:.1f} y {y0:.1f}-{y1:.1f}")
    out = {}
    for key in ("xyz", "feat", "gate", "S"):
        out[key] = np.concatenate([p.pop(key) for p in parts])
    return out, n_crops, n_split


class Vectors:
    """The three versions, built chunk by chunk from the stored columns."""

    def __init__(self, S, feat, sample, dev):
        self.S, self.feat, self.dev = S, feat, dev
        A = np.concatenate([S[sample].astype(np.float64), feat[sample].astype(np.float64)], 1)
        self.mu = A.mean(0)
        self.sd = np.maximum(A.std(0), 1e-6)
        z = (A - self.mu) / self.sd
        self.clipped = float((np.abs(z) > CLIP).mean())
        self.clipped_feat = float((np.abs(z[:, 1232:]) > CLIP).mean())

    def get(self, version, idx):
        a = np.concatenate([self.S[idx].astype(np.float32), self.feat[idx]], 1)
        z = torch.from_numpy(a).to(self.dev)
        z = ((z - torch.from_numpy(self.mu).float().to(self.dev)) / torch.from_numpy(self.sd).float().to(self.dev))
        z = z.clamp(-CLIP, CLIP)
        if version == "sonata":
            return z[:, :1232]
        if version == "sonata+feat_bal":
            z = torch.cat([z[:, :1232] / np.sqrt(1232.0), z[:, 1232:] / np.sqrt(12.0)], 1)
        return z


def kmeans(X, k, rng, iters=100):
    """k-means++ seeding, then Lloyd until fewer than 1 in 10,000 points move.
    -> centres (k, d), inertia, iterations."""
    n = len(X)
    C = X[[int(rng.integers(n))]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = (d2 / d2.sum()).double().cpu().numpy()
        i = int(rng.choice(n, p=p / p.sum()))
        C = torch.cat([C, X[[i]]])
        d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
    lab = None
    for it in range(1, iters + 1):
        d = (X * X).sum(1, keepdim=True) - 2 * X @ C.T + (C * C).sum(1)[None]
        new = d.argmin(1)
        moved = n if lab is None else int((new != lab).sum())
        lab = new
        cnt = torch.bincount(lab, minlength=k)
        C = torch.zeros_like(C).index_add_(0, lab, X) / cnt.clamp(min=1)[:, None].to(X.dtype)
        for e in torch.nonzero(cnt == 0).flatten().tolist():   # empty: re-seed at the worst-fit point
            C[e] = X[int(d.min(1).values.argmax())]
        if moved < max(1, n // 10000):
            break
    inertia = float(((X - C[lab]) ** 2).sum(1).mean())
    return C, inertia, it


def assign(vec, version, C, n, chunk=100000):
    out = np.empty(n, np.int64)
    for a in range(0, n, chunk):
        idx = np.arange(a, min(n, a + chunk))
        X = vec.get(version, idx)
        out[idx] = ((X * X).sum(1, keepdim=True) - 2 * X @ C.T + (C * C).sum(1)[None]).argmin(1).cpu().numpy()
    return out


def rank_by_purity(lab, gate, k):
    """Relabel clusters so 0 has the highest share of near-base points."""
    n = np.bincount(lab, minlength=k)
    pos = np.bincount(lab, weights=gate, minlength=k)
    order = np.argsort(-(pos / np.maximum(n, 1)), kind="stable")
    remap = np.empty(k, np.int64); remap[order] = np.arange(k)
    return remap[lab]


def best_union(lab, gate, k):
    """Clusters added in rank order (most pure first) while F1 rises."""
    n = np.bincount(lab, minlength=k).astype(np.float64)
    pos = np.bincount(lab, weights=gate, minlength=k)
    P = max(gate.sum(), 1)
    tp, cnt, best = 0.0, 0.0, dict(f1=0.0, recall=0.0, precision=0.0, clusters=0)
    for c in range(k):
        tp += pos[c]; cnt += n[c]
        r, p = tp / P, tp / max(cnt, 1)
        f1 = 2 * r * p / max(r + p, 1e-12)
        if f1 <= best["f1"]:
            break
        best = dict(f1=f1, recall=r, precision=p, clusters=c + 1)
    return best


def pca_colour(vec, version, sample, n, chunk=100000):
    Xs = vec.get(version, sample)
    mean = Xs.mean(0)
    U, Sv, V = torch.pca_lowrank(Xs - mean, q=3, center=False, niter=6)
    total = float(((Xs - mean) ** 2).sum())
    share = (Sv ** 2 / max(total, 1e-12)).cpu().numpy()
    ps = ((Xs - mean) @ V).cpu().numpy()
    lo, hi = np.percentile(ps, 1, 0), np.percentile(ps, 99, 0)
    rgb = np.empty((n, 3), np.uint8)
    for a in range(0, n, chunk):
        idx = np.arange(a, min(n, a + chunk))
        p = ((vec.get(version, idx) - mean) @ V).cpu().numpy()
        rgb[idx] = (np.clip((p - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    return rgb, share


def palette(k):
    import matplotlib
    cmap = matplotlib.colormaps["tab10" if k <= 10 else "tab20"]
    return np.array([cmap(i % cmap.N)[:3] for i in range(k)])


def write_ply(path, xyz, rgb, labs, gate):
    cols = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    cols += [(f"k{k}", "u1") for k in labs] + [("near_base", "u1")]
    a = np.empty(len(xyz), dtype=cols)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for k, lab in labs.items():
        a[f"k{k}"] = lab
    a["near_base"] = gate
    names = {"<f4": "float", "u1": "uchar"}
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(a)}"]
    head += [f"property {names[t]} {nm}" for nm, t in cols] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode())
        f.write(a.tobytes())


def figures(R, res, gt_in, x0, y0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz, gate = R["xyz"], R["gate"]
    ncol = 2 + len(KS)

    def colours(v, col):
        if col == 0:
            c = np.full((len(gate), 3), 0.78); c[gate] = (0.85, 0.1, 0.1); return c
        if col == 1:
            return res[v]["rgb"] / 255.0
        k = KS[col - 2]
        return palette(k)[res[v]["lab"][k]]

    def title(v, col):
        if col == 0:
            return f"{v}\nlabel: near a surveyed base (red)"
        if col == 1:
            return f"PCA top 3 as RGB ({res[v]['pca_share'].sum() * 100:.0f}% of variance)"
        k = KS[col - 2]; b = res[v]["union"][k]
        return f"k={k}: best F1 {b['f1']:.2f} (R {b['recall']:.2f} P {b['precision']:.2f})"

    # top view, highest points drawn last
    o = np.argsort(xyz[:, 2])
    fig, axs = plt.subplots(len(VERSIONS), ncol, figsize=(4.6 * ncol, 4.8 * len(VERSIONS)), squeeze=False)
    for r, v in enumerate(VERSIONS):
        for col in range(ncol):
            ax = axs[r, col]
            ax.scatter(xyz[o, 0], xyz[o, 1], c=colours(v, col)[o], s=0.15, marker=".", lw=0, rasterized=True)
            ax.scatter(gt_in[:, 0], gt_in[:, 1], s=30, facecolors="none", edgecolors="k", lw=0.7)
            ax.set_xlim(x0, x0 + SIZE); ax.set_ylim(y0, y0 + SIZE); ax.set_aspect("equal")
            ax.set_title(title(v, col), fontsize=8); ax.tick_params(labelsize=6)
    fig.suptitle("Frozen Sonata, clustered without labels: top view (canopy on top); circles = surveyed bases",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / "47_stemnet_v2_segment_top.png", dpi=110); plt.close(fig)

    # side view of the 3 m slab (along y) through the most bases, nearest points drawn last
    ys = np.arange(y0 + SLAB / 2, y0 + SIZE - SLAB / 2 + 1e-9, 0.5)
    yc = ys[int(np.argmax([(np.abs(gt_in[:, 1] - q) <= SLAB / 2).sum() for q in ys]))] if len(gt_in) else y0 + SIZE / 2
    m = np.flatnonzero(np.abs(xyz[:, 1] - yc) <= SLAB / 2)
    m = m[np.argsort(-xyz[m, 1])]
    gs = gt_in[np.abs(gt_in[:, 1] - yc) <= SLAB / 2] if len(gt_in) else gt_in
    zlo, zhi = (np.percentile(xyz[m, 2], [0, 100]) if len(m) else (0, 1))
    fig, axs = plt.subplots(len(VERSIONS), ncol, figsize=(4.6 * ncol, 4.8 * len(VERSIONS)), squeeze=False)
    for r, v in enumerate(VERSIONS):
        for col in range(ncol):
            ax = axs[r, col]
            ax.scatter(xyz[m, 0], xyz[m, 2], c=colours(v, col)[m], s=0.3, marker=".", lw=0, rasterized=True)
            ax.scatter(gs[:, 0], np.full(len(gs), zlo - 0.3), marker="^", s=25, c="k")
            ax.set_xlim(x0, x0 + SIZE); ax.set_ylim(zlo - 0.6, zhi + 0.2); ax.set_aspect("equal")
            ax.set_title(title(v, col), fontsize=8); ax.tick_params(labelsize=6)
    fig.suptitle(f"Side view (x vs z) of the {SLAB:g} m slab y {yc - SLAB / 2:.1f}-{yc + SLAB / 2:.1f}; "
                 f"triangles = surveyed bases in it", fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / "48_stemnet_v2_segment_side.png", dpi=110); plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[v2-segment] V2_DEVICE=cuda but no GPU is visible in the container")
    t0 = time.time()
    regions = np.load(OUT / "regions.npz")
    gt, tier, _ = val_truth(regions)
    x0, y0 = pick_region(regions, gt)
    inb = (gt[:, 0] >= x0) & (gt[:, 0] < x0 + SIZE) & (gt[:, 1] >= y0) & (gt[:, 1] < y0 + SIZE) if len(gt) else []
    gt_in = gt[inb] if len(gt) else np.zeros((0, 2))
    log(f"region x {x0:.1f}-{x0 + SIZE:.1f} y {y0:.1f}-{y0 + SIZE:.1f} (frame m), {len(gt_in)} val bases in it "
        f"(tiers {', '.join(f'{t} {int((tier[inb] == t).sum())}' for t in 'ABC') if len(gt) else '-'})")
    store = D.open_store(log)
    enc = M.load_sonata(DEVICE) if ENCODER == "sonata" else M.FakeEncoder()
    R, n_crops, n_split = encode_region(enc, store, x0, y0, DEVICE, rng)
    n = len(R["xyz"])
    log(f"encoded {n:,} points from {n_crops} crops ({n_split} over the {CAP:,} cap, split into parts) | "
        f"{R['gate'].sum():,} near a base ({R['gate'].mean() * 100:.2f}%) | {time.time() - t0:.0f} s")
    sample = np.sort(rng.choice(n, min(FIT, n), replace=False))
    vec = Vectors(R["S"], R["feat"], sample, DEVICE)
    log(f"standardised on {len(sample):,} points; clipped at +-{CLIP:g} sd: {vec.clipped * 100:.3f}% of all "
        f"values, {vec.clipped_feat * 100:.3f}% of the 12 features")
    res = {}
    for v in VERSIONS:
        tv = time.time()
        Xs = vec.get(v, sample)
        r = res[v] = dict(lab={}, union={})
        r["rgb"], r["pca_share"] = pca_colour(vec, v, sample, n)
        log(f"\n== {v} ({Xs.shape[1]} values per point) | PCA top 3 = "
            f"{', '.join(f'{s * 100:.1f}' for s in r['pca_share'])}% of variance")
        for k in KS:
            C, inertia, it = kmeans(Xs, k, rng)
            lab = rank_by_purity(assign(vec, v, C, n), R["gate"], k)
            r["lab"][k] = lab
            b = r["union"][k] = best_union(lab, R["gate"], k)
            log(f"k={k}: {it} iterations, inertia {inertia:.3g} | near-base points: best union of "
                f"{b['clusters']} cluster(s) -> recall {b['recall']:.3f}, purity {b['precision']:.3f}, "
                f"F1 {b['f1']:.3f}")
            log("   cluster  points%  near-base%  of-all-near-base%  verticality  linearity  (median)")
            cnt = np.bincount(lab, minlength=k); pos = np.bincount(lab, weights=R["gate"], minlength=k)
            for c in range(k):
                mm = lab == c
                vt, ln = (np.median(R["feat"][mm, 11]), np.median(R["feat"][mm, 0])) if mm.any() else (np.nan, np.nan)
                log(f"   {c:>7} {cnt[c] / n * 100:8.2f} {pos[c] / max(cnt[c], 1) * 100:11.2f} "
                    f"{pos[c] / max(R['gate'].sum(), 1) * 100:18.1f} {vt:12.2f} {ln:10.2f}")
        write_ply(RESULTS / f"stemnet_v2_segment_{v}.ply", R["xyz"], r["rgb"],
                  {k: r["lab"][k].astype(np.uint8) for k in KS}, R["gate"].astype(np.uint8))
        log(f"   ({time.time() - tv:.0f} s)")
    log("\nsummary: best F1 for the near-base points (k: F1 / recall / purity)")
    for v in VERSIONS:
        log(f"  {v:<16} " + " | ".join(f"k={k}: {res[v]['union'][k]['f1']:.3f} / {res[v]['union'][k]['recall']:.2f}"
                                       f" / {res[v]['union'][k]['precision']:.2f}" for k in KS))
    figures(R, res, gt_in, x0, y0)
    log(f"-> results/stemnet_v2_segment.txt, 47_/48_ png, stemnet_v2_segment_<version>.ply | "
        f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
