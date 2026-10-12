#!/usr/bin/env python3
"""Dense stem head on frozen Sonata: per point, is this a tree, and where is
its base?

The walker's problem was never perception. Its misses on the attributed patch
reached within a metre of 94% of the trees they missed and stopped short, and
the median detection cluster had ONE vote out of the five walkers that passed
near each tree. Eight sequential sphere encodes per walker buy a single opinion
about one trajectory; the information to localise a stem is already in the
points around it.

So: one forward pass, every point votes.

  * the encoder is the same frozen Sonata, but read densely — the coarse tokens
    are upcast back to the input points through the pooling chain, which is the
    standard Sonata linear-probe read-out.
  * two heads per point: a logit for "I belong to a tree stem" and a 2-D
    horizontal offset to that stem's base.
  * detections are peaks in the p-weighted histogram of SHIFTED points. A
    cluster's vote count is the number of points that agreed, so consensus is
    dense by construction instead of something to hope for.

Why these targets and not something more ambitious: the only per-tree truth in
this project is a set of RTK stem bases. There is no per-point instance label
and no semantic label on this cloud, so "is a tree" is operationalised as "lies
within SN_NEAR_R metres HORIZONTALLY of a surveyed base" — which, restricted
to the breast-height band, is a trunk and its immediate surroundings. That is a
narrower claim than tree/not-tree, and it is the one the
data can actually support — and it is exactly what we need, since a stem base
is what gets reported.

Horizontal, not 3-D: a trunk point 3 m up is 3 m from its own base in 3-D but
~0.15 m from it in plan view. The base's z comes from the DEM, so the head
never has to predict it, and the target is a short 2-D vector with far less
variance than the walker's 8 m trajectory.

Window geometry: the scene is processed in WIN-metre squares with a PAD-metre
context ring that is encoded but never predicted on, so a trunk at a window
edge is still seen inside somebody's neighbourhood. Inner squares tile exactly,
so no point is counted twice.

Not a walker. There is no sphere, no step, no memory of previous positions, no
through-cloud distance field and no seed grid — the scene is just a chunk of
points, processed a window at a time, and every point is read once. That is why
this is its own track rather than another GeoWalker stage: the only things the
two share are the prepared scene (tiles, DEM, labels), the frozen encoder and
the matcher that scores them both, and those all live in `common`.

Shared by train.py and infer.py; nothing here touches the filesystem at import.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path[:0] = [str(Path(__file__).resolve().parent), "/app"]
from common import config as C  # noqa: E402
from common.dem import Dem  # noqa: E402,F401
from common.detect import match_metrics, match_pairs  # noqa: E402,F401
from common.encoder import load_encoder, upcast  # noqa: E402,F401
from common.raster import peaks  # noqa: E402

# PREP, not BaseWalker: tiles / dem / labels / seeds are the shared scene prep
# that every track in this repo reads. The directory is named after the stage
# that happens to write them.
PREP = C.DATA / "basewalker"
# SN_TAG names a run, so two settings can be compared instead of the second
# silently overwriting the first. Empty is the default, i.e. stemnet.pth.
TAG = os.environ.get("SN_TAG", "")
CKPT = C.MODELS / f"stemnet{TAG}.pth"
LAST_CKPT = C.MODELS / f"stemnet{TAG}_last.pth"

GRID = float(os.environ.get("GRID_SIZE", "0.05"))
VOXEL = float(os.environ.get("SN_VOXEL", str(GRID)))   # band dedup resolution
BAND_LO = float(os.environ.get("SN_BAND_LO", "0.3"))   # m above the DEM
BAND_HI = float(os.environ.get("SN_BAND_HI", "4.0"))
NEAR_R = float(os.environ.get("SN_NEAR_R", "1.0"))     # positive-label radius
WIN = float(os.environ.get("SN_WIN", "16.0"))          # predicted square, m
PAD = float(os.environ.get("SN_PAD", "2.0"))           # encoded context ring
# The crop point cap — a MEMORY knob, not a modelling one. Stage-1 attention
# materialises an (N/K, heads, K, K) tensor, i.e. N x heads x patch_size
# floats, and it exists twice while its softmax runs: ~25 kB of device memory
# per crop point, on top of the band, the frozen encoder and the head's
# retained activations. A typical crop on this scene is ~150 k points, so this
# sits just above it: ordinary windows pass through untouched and only a
# thicket gets subsampled.
PTS_CAP = int(float(os.environ.get("SN_PTS", "160000")))
P_THRESH = float(os.environ.get("SN_P", "0.5"))        # a point votes above this
VOTE_CELL = float(os.environ.get("SN_VOTE_CELL", "0.1"))
VOTE_SEP = float(os.environ.get("SN_VOTE_SEP", "0.6"))  # two stems this far apart
VOTE_R = float(os.environ.get("SN_VOTE_R", "0.3"))     # voters pooled per peak
VOTE_MIN = float(os.environ.get("SN_VOTE_MIN", "6.0"))  # minimum vote mass
DETREND = os.environ.get("SN_DETREND", "1") != "0"   # z -> height above ground
VAL_MARGIN = float(os.environ.get("BW_VAL_MARGIN", "10.0"))
DEVICE = os.environ.get("SN_DEVICE", "cuda")


# ---------------------------------------------------------------------------
# one window of points -> per-point features -> two predictions
# ---------------------------------------------------------------------------
_NOTE: list[str] = []


@torch.no_grad()
def encode_points(enc, chunk: "Chunk", idx: torch.Tensor, zcol=None):
    """Per-point features for the chunk points `idx`, as one Sonata crop.

    DETRENDED (`zcol` = height above ground, which `Band.crop` supplies unless
    SN_DETREND=0). This scene has serious relief, and a crop is 20 m wide: with
    raw z and one `z.min()` for the whole crop, an uphill trunk arrives at local
    z 8-12 m and an identical downhill trunk at 0-4 m. Same object, two
    completely different inputs, and the encoder has no way to know where the
    ground is — the single worst thing that could happen to a stem detector.

    Subtracting the DEM fixes it and costs almost nothing geometrically: the
    shift depends only on (x, y), and a trunk occupies essentially one (x, y),
    so it translates RIGIDLY — its shape is preserved exactly. What flattens is
    the ground, which is what we wanted gone. (The normals are left on the real
    surface: a trunk's normal is horizontal and unaffected, and the band starts
    at 0.3 m so there is little ground left to disagree about.)

    Then centre-shifted exactly as the pretrained CenterShift does (xy to the
    mean, z to the minimum), because a crop shifted some other way is a crop
    the pretrained weights have never seen and the failure is silent.
    """
    pts = chunk.xyz[idx]
    if zcol is not None:
        pts = torch.cat([pts[:, :2], zcol[:, None]], 1)
    shift = torch.cat([pts[:, :2].mean(0), pts[:, 2].min()[None]])
    local = pts - shift[None, :]
    feat = torch.cat([local, torch.zeros_like(local), chunk.normal[idx]], 1)
    data = dict(
        coord=local,
        grid_coord=((local - local.min(0).values[None, :]) / GRID).floor().long(),
        feat=feat,
        offset=torch.tensor([len(idx)], device=chunk.device, dtype=torch.long),
    )
    return upcast(enc(data), local, note=_NOTE)


class StemHead(nn.Module):
    """Per-point tree logit + horizontal offset to the stem base.

    The offset layer starts at zero: before it has learned anything every point
    votes for itself, which is a weak detector rather than a random one.
    """

    def __init__(self, feat_dim: int, d: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU())
        self.logit = nn.Linear(d, 1)
        self.off = nn.Linear(d, 2)
        nn.init.zeros_(self.off.weight)
        nn.init.zeros_(self.off.bias)

    def forward(self, feat):
        h = self.trunk(feat)
        return self.logit(h)[:, 0], self.off(h)


# ---------------------------------------------------------------------------
# the stem band: the only points this model ever sees
# ---------------------------------------------------------------------------
class Buckets:
    """2-D bucket index over a point set, for rectangle queries."""

    def __init__(self, xy: np.ndarray, cell: float):
        self.cell = cell
        self.x0, self.y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
        self.nx = int((xy[:, 0].max() - self.x0) / cell) + 1
        self.ny = int((xy[:, 1].max() - self.y0) / cell) + 1
        ix = ((xy[:, 0] - self.x0) / cell).astype(np.int64).clip(0, self.nx - 1)
        iy = ((xy[:, 1] - self.y0) / cell).astype(np.int64).clip(0, self.ny - 1)
        cid = iy * self.nx + ix
        self.order = np.argsort(cid, kind="stable")
        self.start = np.searchsorted(cid[self.order], np.arange(self.nx * self.ny))
        self.start = np.r_[self.start, len(cid)]

    def rect(self, x0, x1, y0, y1):
        """Positions (into the indexed arrays) whose bucket meets the rectangle.

        Buckets, so the result is a superset of the rectangle — callers mask
        exactly. Returning a superset is the point: the caller usually wants a
        padded crop anyway.
        """
        ix0 = max(0, int((x0 - self.x0) / self.cell))
        ix1 = min(self.nx - 1, int((x1 - self.x0) / self.cell))
        iy0 = max(0, int((y0 - self.y0) / self.cell))
        iy1 = min(self.ny - 1, int((y1 - self.y0) / self.cell))
        if ix1 < ix0 or iy1 < iy0:
            return np.zeros(0, np.int64)
        out = [self.order[self.start[iy * self.nx + ix0]:
                          self.start[iy * self.nx + ix1 + 1]]
               for iy in range(iy0, iy1 + 1)]
        return np.concatenate(out) if out else np.zeros(0, np.int64)


def voxel_unique(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """One representative point per `voxel` cube — a lossless reduction as far
    as the encoder is concerned, since it serialises on a GRID-metre lattice
    anyway, and it is what bounds the cost of a window."""
    k = np.floor(xyz / voxel).astype(np.int64)
    k -= k.min(0)
    n = k.max(0) + 1
    flat = k[:, 0] + n[0] * (k[:, 1] + n[1] * k[:, 2])
    return np.unique(flat, return_index=True)[1]


class Chunk:
    """A flat set of points on the device: xyz, normals, nothing else.

    Deliberately NOT a spatial store with a ball index: this model never asks
    "what is within r of this moving point". It asks for a rectangle once per
    window, which `Band`'s bucket index answers on the CPU, and after that the
    points are just rows.

    What lives here is the BAND, not the scene. `Band` builds this, and builds
    it from band points only. The full cloud is 267 M points — 6.4 GB of xyz
    plus normals — and 83% of it is below 0.3 m or above 4.0 m and is never
    read again once the band is selected. Keeping it resident cost more of the
    card than the encoder, the dense read-out and the head's activations put
    together, which is what made training die partway through a run.
    """

    def __init__(self, xyz: np.ndarray, normal: np.ndarray, device="cuda"):
        self.device = device
        self.xyz = torch.from_numpy(
            np.ascontiguousarray(xyz, np.float32)).to(device)
        self.normal = torch.from_numpy(
            np.ascontiguousarray(normal, np.float32)).to(device)

    def __len__(self):
        return len(self.xyz)


def _point_source(src, pattern="tile_*.npz"):
    """Yield (xyz, normal) host arrays from whatever the caller has.

    A directory is yielded ONE TILE AT A TIME and never concatenated, so the
    whole cloud is never materialised — not on the card and not in host RAM.
    `Band` is about to discard five sixths of it, and this is the only reason
    it can afford to look at all of it.
    """
    if isinstance(src, Chunk):           # a cloud already resident (tests)
        yield src.xyz.cpu().numpy(), src.normal.cpu().numpy()
        return
    if isinstance(src, (str, Path)):
        qs = sorted(Path(src).glob(pattern))
        if not qs:
            raise SystemExit(f"[sn] no tiles in {src} — run bw_prep first")
        for q in qs:
            d = np.load(q)
            yield d["xyz"].astype(np.float32), d["normal"].astype(np.float32)
        return
    xyz, normal = src
    yield np.asarray(xyz, np.float32), np.asarray(normal, np.float32)


def gpu_report(tag=""):
    """What the card is holding. Printed, never inferred.

    An exhausted card does not always announce itself as a clean
    torch.OutOfMemoryError here: this image runs spconv-cu126 on sm_120 through
    its NVRTC fallback, and a workspace allocation that fails inside those
    kernels surfaces as `CUDA error: an illegal memory access was encountered`
    at the next device sync, several frames away from the real cause. So the
    high-water mark gets printed, and a run that dies at iteration 159 says why.
    """
    dev = str(DEVICE)
    if not (dev.startswith("cuda") and torch.cuda.is_available()):
        return {}
    free, total = torch.cuda.mem_get_info()
    out = dict(total_gb=total / 2 ** 30, free_gb=free / 2 ** 30,
               held_gb=torch.cuda.memory_allocated() / 2 ** 30,
               peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    print(f"[sn] gpu{tag}: {out['held_gb']:.2f} GB held, peak "
          f"{out['peak_gb']:.2f} GB, {out['free_gb']:.2f} of "
          f"{out['total_gb']:.2f} GB free", flush=True)
    return out


class Band:
    """The breast-height slice of the scene — deduplicated, indexed, and the
    only points that are ever uploaded.

    Everything outside it is dropped before the model ever runs: the ground
    carries no stem, the crown cannot say where a trunk meets the soil, and
    together they are most of the cloud. This is why one pass over the scene is
    affordable where 199,278 sphere encodes were not.

    `src` is normally the tiles directory, which is streamed a tile at a time —
    the selection happens on the way in, so the 83% of the scene that the band
    rejects never reaches the device at all. A `Chunk` or a plain (xyz, normal)
    pair is also accepted, for a cloud small enough to hold whole.

    After construction, a band position IS a device row: `self.xyz[i]` (host,
    float64, for the rectangle maths) and `self.chunk.xyz[i]` (device, float32,
    for the encoder) are the same point. There is no index translation left to
    get wrong.
    """

    def __init__(self, src, dem: Dem, lo=BAND_LO, hi=BAND_HI, voxel=VOXEL,
                 cell=None, device=None, block=1 << 20):
        if device is None:
            device = src.device if isinstance(src, Chunk) else DEVICE
        xs, ns, hs = [], [], []
        n_src = n_raw = n_part = 0
        for xyz, normal in _point_source(src):
            n_src += len(xyz)
            n_part += 1
            for s in range(0, len(xyz), block):
                bx = xyz[s:s + block]
                t = torch.from_numpy(np.ascontiguousarray(bx)).to(device)
                h = t[:, 2] - dem.height(t[:, :2])
                k = torch.nonzero((h >= lo) & (h <= hi), as_tuple=True)[0]
                if not len(k):
                    continue
                kc = k.cpu().numpy()
                xs.append(bx[kc])
                ns.append(normal[s:s + block][kc])
                hs.append(h[k].cpu().numpy())
                n_raw += len(kc)
                del t, h, k
        if not xs:
            raise SystemExit(f"[sn] nothing between {lo} and {hi} m above the "
                             f"DEM — the band is empty, so the DEM is wrong")
        xyz_b, nrm_b = np.concatenate(xs), np.concatenate(ns)
        h_b = np.concatenate(hs)
        del xs, ns, hs
        # the dedup key in float64, on the HOST: at this scene's scale a float32
        # coordinate quantises to ~2e-5 m, enough to flip which side of a voxel
        # wall a point falls on, and doing it here means no float64 copy of the
        # band (1.5 GB of it) is ever made on the card.
        u = voxel_unique(xyz_b.astype(np.float64), voxel)
        self.xyz = xyz_b[u].astype(np.float64)
        self.h = h_b[u].astype(np.float64)
        self.chunk = Chunk(xyz_b[u], nrm_b[u], device=device)
        # height above ground, kept rather than recomputed: it selects the band
        # AND it is the z the encoder is given, so the two can never disagree
        self.h_t = torch.from_numpy(self.h.astype(np.float32)).to(device)
        self.device = device
        self.n_src = int(n_src)
        self.n_raw = int(n_raw)
        self.buckets = Buckets(self.xyz[:, :2], cell or max(WIN / 2.0, 1.0))
        mb = (self.chunk.xyz.numel() + self.chunk.normal.numel()
              + self.h_t.numel()) * 4 / 2 ** 20
        print(f"[sn] band {lo}-{hi} m: {n_src:,} scene pts from {n_part} "
              f"tiles -> {self.n_raw:,} in band -> {len(self.xyz):,} after "
              f"{voxel} m dedup "
              f"({len(self.xyz)/max(n_src,1)*100:.1f}% of the scene)")
        print(f"[sn] on device: the band alone, {mb:,.1f} MB — the other "
              f"{n_src - len(self.xyz):,} points were never uploaded")
        print(f"[sn] encoder z: "
              + ("HEIGHT ABOVE GROUND (detrended)" if DETREND else
                 "RAW z — SN_DETREND=0, so relief is inside every crop"))

    def __len__(self):
        return len(self.xyz)

    def crop(self, pos):
        """(device rows into self.chunk, ground-relative z) for band positions."""
        t = torch.as_tensor(np.ascontiguousarray(pos, np.int64),
                            device=self.device)
        return t, (self.h_t[t] if DETREND else None)

    def window(self, cx, cy, win=WIN, pad=PAD):
        """(crop positions, inner mask) for the square centred at (cx, cy).

        The crop carries a `pad` ring of context that is encoded but not
        predicted on; `inner` selects the square that is actually this
        window's responsibility, so stepping by `win` tiles the scene exactly.
        """
        h = win / 2.0 + pad
        pos = self.buckets.rect(cx - h, cx + h, cy - h, cy + h)
        if not len(pos):
            return pos, np.zeros(0, bool)
        xy = self.xyz[pos, :2]
        crop = ((np.abs(xy[:, 0] - cx) <= h) & (np.abs(xy[:, 1] - cy) <= h))
        pos = pos[crop]
        xy = xy[crop]
        inner = ((np.abs(xy[:, 0] - cx) <= win / 2.0)
                 & (np.abs(xy[:, 1] - cy) <= win / 2.0))
        return pos, inner


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def nearest_base(xy: torch.Tensor, bases_xy: torch.Tensor, block: int = 1 << 14):
    """Horizontal distance to the nearest labelled base, and which one.

    Chunked so a 250k-point window against 758 bases never allocates the full
    cross product at once.
    """
    d_out = torch.empty(len(xy), device=xy.device)
    j_out = torch.empty(len(xy), dtype=torch.long, device=xy.device)
    for s in range(0, len(xy), block):
        d = torch.cdist(xy[s:s + block], bases_xy)
        m, j = d.min(1)
        d_out[s:s + block] = m
        j_out[s:s + block] = j
    return d_out, j_out


def targets(xy: torch.Tensor, bases_xy: torch.Tensor, near_r=NEAR_R):
    """(is_stem, offset_to_base) for a window's points."""
    d, j = nearest_base(xy, bases_xy)
    return d <= near_r, bases_xy[j] - xy


# ---------------------------------------------------------------------------
# votes -> detections
# ---------------------------------------------------------------------------
def vote(xy: np.ndarray, p: np.ndarray, off: np.ndarray, p_thresh=P_THRESH,
         cell=VOTE_CELL, sep=VOTE_SEP, pool_r=VOTE_R, min_mass=VOTE_MIN):
    """Peaks in the p-weighted histogram of shifted points.

    Returns (xy, mass, n_votes). `mass` is the summed confidence of the points
    that agreed on a location and is the detection's score; `n_votes` is how
    many points those were. This is the number the walker could never produce:
    its median cluster had one vote, here every stem point in the band is an
    independent opinion about the same base.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    k = p >= p_thresh
    if k.sum() < 1:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)
    v = xy[k] + off[k]
    w = p[k].astype(np.float64)

    x0, y0 = v[:, 0].min() - 1.0, v[:, 1].min() - 1.0
    nx = int((v[:, 0].max() + 1.0 - x0) / cell) + 1
    ny = int((v[:, 1].max() + 1.0 - y0) / cell) + 1
    acc = np.zeros((ny, nx), np.float64)
    ix = ((v[:, 0] - x0) / cell).astype(np.int64).clip(0, nx - 1)
    iy = ((v[:, 1] - y0) / cell).astype(np.int64).clip(0, ny - 1)
    np.add.at(acc, (iy, ix), w)
    # a stem's votes land in a puddle a few cells wide, not one cell
    sm = ndimage.uniform_filter(acc, size=3) * 9.0
    cand, _ = peaks(sm, cell, sep, min_mass, x0, y0)
    if not len(cand):
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)

    # refine each peak to the weighted centroid of the voters it pooled: the
    # raster cell centre is quantised to `cell`, the centroid is not
    kd = cKDTree(v)
    out = []
    for c in cand:
        hit = kd.query_ball_point(c, pool_r)
        if not hit:
            continue
        ww = w[hit]
        out.append((float((v[hit, 0] * ww).sum() / ww.sum()),
                    float((v[hit, 1] * ww).sum() / ww.sum()),
                    float(ww.sum()), len(hit)))
    if not out:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)
    out = np.array(out)
    out = out[out[:, 2] >= min_mass]
    if not len(out):
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64)

    # two peaks can refine onto the same stem; keep the heavier
    order = np.argsort(-out[:, 2])
    kept = []
    for i in order:
        if all(((out[i, :2] - out[j, :2]) ** 2).sum() > sep * sep for j in kept):
            kept.append(i)
    out = out[kept]
    return out[:, :2], out[:, 2], out[:, 3].astype(np.int64)


def relief(dem_npz=None, win=WIN, pad=PAD):
    """How much ground elevation varies, over the scene and inside one crop.

    The second number is the one that matters: it is exactly the error the
    encoder would be fed if the crop were not detrended, because that much
    terrain would sit inside a single `z.min()` shift. Printed rather than
    asserted — the detrending removes it either way, but a reader of a score
    should know whether this scene is a hillside or a table.
    """
    d = np.load(dem_npz or PREP / "dem.npz", allow_pickle=True)
    z, cell = d["z"].astype(np.float64), float(d["cell"])
    # which ground filter actually ran: prep falls back to a per-cell 5th
    # percentile if CSF is missing (its install is `|| true`), and that
    # fallback does not classify ground at all — under closed canopy it sits
    # in the understory, which lifts the DEM and slides the band off the stems
    meth = str(d["method"]) if "method" in d.files else "unrecorded"
    note = {"csf": "  (cloth simulation filter, Zhang et al.)",
            "percentile": "  <-- NOT the cloth filter: CSF was unavailable at "
                          "prep time and this is the per-cell 5th-percentile "
                          "fallback, which does not classify ground at all",
            "unrecorded": "  (no method recorded in dem.npz)"}.get(meth, "")
    print(f"[sn] ground filter: {meth}{note}")
    if "valid" in d.files:
        v = d["valid"]
        print(f"[sn] DEM cells with OBSERVED ground: {v.mean()*100:.1f}% — the "
              f"rest are nan-filled by dilation, and Dem.height cannot tell "
              f"you which is which")
    ok = np.isfinite(z)
    if not ok.any():
        print("[sn] DEM is empty — cannot report relief")
        return {}
    k = max(1, int(round((win + 2 * pad) / cell)))
    from scipy import ndimage
    f = np.where(ok, z, np.nan)
    hi = ndimage.maximum_filter(np.where(ok, z, -np.inf), size=k, mode="nearest")
    lo = ndimage.minimum_filter(np.where(ok, z, np.inf), size=k, mode="nearest")
    span = np.where(ok, hi - lo, np.nan)
    span = span[np.isfinite(span)]
    out = dict(ground_filter=meth,
               frac_observed_ground=(float(d["valid"].mean())
                                     if "valid" in d.files else None),
               scene_relief_m=float(np.nanmax(f) - np.nanmin(f)),
               crop_span_p50_m=float(np.percentile(span, 50)) if len(span) else float("nan"),
               crop_span_p95_m=float(np.percentile(span, 95)) if len(span) else float("nan"),
               crop_span_max_m=float(span.max()) if len(span) else float("nan"),
               crop_m=float(win + 2 * pad))
    print(f"[sn] relief: the scene spans {out['scene_relief_m']:.1f} m of ground "
          f"elevation | within one {out['crop_m']:.0f} m crop, "
          f"p50 {out['crop_span_p50_m']:.1f} m p95 {out['crop_span_p95_m']:.1f} m "
          f"max {out['crop_span_max_m']:.1f} m")
    if out["crop_span_p95_m"] > 2.0 and not DETREND:
        print("[sn] !! SN_DETREND=0 with that much relief inside a crop: the "
              "encoder sees terrain, not trees. This will not work.")
    return out


def dem_check(dem, device=None):
    """Is the DEM actually the ground, on this terrain?

    The band, the labels' z and the detrending are all measured FROM the DEM, so
    a DEM that is wrong on a slope moves the slab off the stems and nothing
    downstream can recover. The survey's own levelled elevations are the only
    independent check available: 392 attributed trees with a measured z.

    A constant offset is a vertical datum difference, not an error, so the
    median is removed and the SPREAD is what is reported.
    """
    try:
        from common import patch as P
        attrs = P.load_attributes()
        xy = P.to_local(attrs["lat"], attrs["lon"])
        zs = np.asarray(attrs["z"], np.float64)
    except SystemExit as e:
        print(f"[sn] DEM check skipped: {e}")
        return {}
    except Exception as e:                       # noqa: BLE001 - diagnostic only
        print(f"[sn] DEM check skipped: {type(e).__name__}: {e}")
        return {}
    ok = np.isfinite(zs)
    if ok.sum() < 20:
        print("[sn] DEM check skipped: too few surveyed elevations")
        return {}
    xy, zs = xy[ok], zs[ok]
    t = torch.from_numpy(xy.astype(np.float32))
    if device is not None:
        t = t.to(device)
    g = dem.height(t).detach().cpu().numpy().astype(np.float64)
    r = zs - g
    off = float(np.median(r))
    r = r - off
    q = np.percentile(r, [5, 25, 50, 75, 95])
    out = dict(n=int(len(r)), datum_offset_m=off,
               spread_p5_m=float(q[0]), spread_p25_m=float(q[1]),
               spread_p75_m=float(q[3]), spread_p95_m=float(q[4]),
               iqr_m=float(q[3] - q[1]),
               frac_beyond_half_m=float((np.abs(r) > 0.5).mean()))
    print(f"[sn] DEM vs {out['n']} surveyed elevations (datum offset "
          f"{off:+.2f} m removed): IQR {out['iqr_m']:.2f} m | "
          f"p5 {q[0]:+.2f} p95 {q[4]:+.2f} m | "
          f"{out['frac_beyond_half_m']*100:.0f}% beyond 0.5 m")
    if out["iqr_m"] > 0.5:
        print(f"[sn] !! the DEM disagrees with the survey by more than the "
              f"band is thick in places. The band ({BAND_LO}-{BAND_HI} m) may "
              f"be sitting off the stems; widen SN_BAND_LO/HI or rebuild the "
              f"DEM (BW_DEM_METHOD=csf, FORCE_BW_PREP=1).")
    return out


def split_x(labels=None):
    """The train/val boundary in local x, as prep_scene recorded it."""
    lab = np.load(PREP / "labels.npz") if labels is None else labels
    return float(lab["split_x"])


def tile_centres(x0, x1, y0, y1, win=WIN):
    """Inner-square centres that tile the rectangle exactly."""
    nx = max(1, int(np.ceil((x1 - x0) / win)))
    ny = max(1, int(np.ceil((y1 - y0) / win)))
    cx = x0 + (np.arange(nx) + 0.5) * win
    cy = y0 + (np.arange(ny) + 0.5) * win
    gx, gy = np.meshgrid(cx, cy)
    return np.column_stack([gx.ravel(), gy.ravel()])
