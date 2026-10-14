#!/usr/bin/env python3
"""StemNet v2 prep, part 1: the raw scan on Sonata's grid, in the shared frame.

What happens, and only this:

  1. stream the LAZ once, in chunks
  2. assign every point to a V2_GRID voxel (default 0.02 m, Sonata's own
     GridSample size), in the file's integer units
  3. keep ONE REAL point per voxel — the one nearest the voxel centre. Not an
     average: an averaged point is a position the scanner never measured, and
     its colour would be a blend of several surfaces
  4. write it in V2_TILE-metre storage tiles, coordinates local to the shared
     origin, with its colour and intensity

Nothing is classified, filtered, denoised or removed except by the voxel grid
itself. Ground stays. Exact duplicates (12% of this scan) disappear because they
share a voxel, not because anything looks for them.

Two passes, because a voxel's points arrive in different chunks — the scanner
walked past the same tree more than once:
  pass A  each chunk is reduced to its own nearest-to-centre point per voxel
          (exact: the global nearest is the nearest among the chunk winners)
          and appended to a per-tile scratch file
  pass B  each tile's scratch file is reduced again across chunks, written
          out, and deleted

Outputs (data/stemnet_v2/):
  frame.json                        shared origin, EPSG, grid, scale, counts
  grid0.02/tile_<i>_<j>.npz         xyz float32 (local m), rgb uint8 (N,3),
                                    intensity uint8
Idempotent: skips when frame.json says this file and grid are done
(V2_FORCE=1 rebuilds).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, RESULTS, TILE, Frame, find_laz  # noqa: E402

GRID = float(os.environ.get("V2_GRID", "0.02"))
CHUNK = int(float(os.environ.get("V2_CHUNK", "5000000")))
FORCE = os.environ.get("V2_FORCE", "0") != "0"
GRID_DIR = OUT / f"grid{GRID:g}"
SCRATCH = OUT / f"scratch_grid{GRID:g}"
FRAME_JSON = OUT / "frame.json"
ROW = np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"),
                ("rgb", "u1", 3), ("i", "u1")])          # 16 bytes per point


def log(msg):
    print(f"[v2-grid] {msg}", flush=True)


def to_u8(a, vmax):
    """Colour / intensity to 8 bits. This scan stores 0-255 in 16-bit fields;
    a scan that uses the full 16 bits is scaled down instead of clipped."""
    a = np.asarray(a)
    return (a >> 8).astype(np.uint8) if vmax > 255 else a.astype(np.uint8)


class Voxels:
    """Voxel index and squared distance to the voxel centre, in integers."""

    def __init__(self, fr: Frame, grid: float):
        if not fr.exact(grid):
            raise SystemExit(f"[v2-grid] grid {grid} m is not a whole number of "
                             f"LAS units (scale {fr.s.tolist()}) — exact voxels impossible")
        self.fr = fr
        self.q = np.array([fr.units(grid, a)[0] for a in range(3)], np.int64)

    def __call__(self, X, Y, Z):
        D = np.column_stack([X, Y, Z]).astype(np.int64) - self.fr.oi
        V = D // self.q
        # 2*(d - v*q) - q is twice the offset from the centre: integer, exact
        E = 2 * (D - V * self.q) - self.q
        return V, (E * E).sum(1)


def nearest_per_voxel(V, d2):
    """Index of the nearest-to-centre point in every occupied voxel.
    Ties go to the earliest point, so the result does not depend on chunking."""
    V0 = V.min(0)
    W = V - V0
    span = W.max(0) + 1
    key = W[:, 0] + span[0] * (W[:, 1] + span[1] * W[:, 2])
    order = np.lexsort((np.arange(len(key)), d2, key))
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    return order[first]


def main():
    import laspy
    OUT.mkdir(parents=True, exist_ok=True)
    laz = find_laz()
    st = laz.stat()
    stamp = dict(file=str(laz), bytes=st.st_size, mtime=int(st.st_mtime),
                 grid_m=GRID, tile_m=TILE)
    if FRAME_JSON.exists() and not FORCE:
        prev = json.loads(FRAME_JSON.read_text())
        if prev.get("grids", {}).get(f"{GRID:g}", {}).get("stamp") == stamp:
            log(f"grid {GRID:g} m already built for this file — skipping "
                f"(V2_FORCE=1 rebuilds)")
            return

    with laspy.open(str(laz)) as f:
        hdr = f.header
        fr = Frame(hdr.scales, hdr.offsets, hdr.mins)
        vox = Voxels(fr, GRID)
        dims = set(hdr.point_format.dimension_names)
        has_rgb = {"red", "green", "blue"} <= dims
        n_all = int(hdr.point_count)
        epsg = None
        try:
            c = hdr.parse_crs()
            epsg = c.to_epsg() if c is not None else None
        except Exception:                          # noqa: BLE001 - recorded as None
            pass

        need = n_all * ROW.itemsize
        free = shutil.disk_usage(OUT).free
        log(f"source {laz.name}: {n_all:,} pts | grid {GRID} m | scratch worst case "
            f"{need / 2**30:.1f} GB, {free / 2**30:.1f} GB free")
        if need > 0.9 * free:
            raise SystemExit("[v2-grid] not enough disk for the scratch tiles")
        for d in (SCRATCH, GRID_DIR):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        # 16-bit colour that only ever holds 0-255 is the common case; decide
        # once, from the first chunk, and record it
        rgb_max = i_max = None
        n_in = n_scratch = 0
        t0 = time.time()
        for pts in f.chunk_iterator(CHUNK):
            X = np.asarray(pts.X, np.int64); Y = np.asarray(pts.Y, np.int64)
            Z = np.asarray(pts.Z, np.int64)
            n_in += len(X)
            if has_rgb:
                R = np.asarray(pts["red"]); G = np.asarray(pts["green"]); B = np.asarray(pts["blue"])
                if rgb_max is None:
                    rgb_max = int(max(R.max(), G.max(), B.max()))
            I = np.asarray(pts["intensity"]) if "intensity" in dims else np.zeros(len(X), np.uint16)
            if i_max is None:
                i_max = int(I.max())
            V, d2 = vox(X, Y, Z)
            k = nearest_per_voxel(V, d2)
            rows = np.empty(len(k), ROW)
            rows["X"], rows["Y"], rows["Z"] = X[k], Y[k], Z[k]
            rows["rgb"] = (np.column_stack([to_u8(R[k], rgb_max), to_u8(G[k], rgb_max),
                                            to_u8(B[k], rgb_max)]) if has_rgb else 0)
            rows["i"] = to_u8(I[k], i_max)
            tx = V[k, 0] * vox.q[0] // fr.units(TILE, 0)[0]
            ty = V[k, 1] * vox.q[1] // fr.units(TILE, 1)[0]
            tkey = tx * 100000 + ty
            o = np.argsort(tkey, kind="stable")
            tk, rows = tkey[o], rows[o]
            cuts = np.flatnonzero(np.r_[True, tk[1:] != tk[:-1], True])
            for a, b in zip(cuts[:-1], cuts[1:]):
                t = int(tk[a])
                with open(SCRATCH / f"tile_{t // 100000}_{t % 100000}.bin", "ab") as g:
                    g.write(rows[a:b].tobytes())
            n_scratch += len(k)
            el = time.time() - t0
            log(f"pass A: {n_in:,}/{n_all:,} -> {n_scratch:,} chunk-reduced | "
                f"{el / 60:.1f} min, eta {el / n_in * (n_all - n_in) / 60:.1f} min")

    tiles = sorted(SCRATCH.glob("tile_*.bin"))
    n_out, per_tile = 0, {}
    t0 = time.time()
    for j, q in enumerate(tiles):
        rows = np.fromfile(q, dtype=ROW)
        V, d2 = vox(rows["X"], rows["Y"], rows["Z"])
        rows = rows[np.sort(nearest_per_voxel(V, d2))]
        P = np.column_stack([rows["X"], rows["Y"], rows["Z"]])
        np.savez(GRID_DIR / f"{q.stem}.npz",
                 xyz=fr.local(P).astype(np.float32),
                 rgb=np.ascontiguousarray(rows["rgb"]),
                 intensity=np.ascontiguousarray(rows["i"]))
        per_tile[q.stem] = int(len(rows))
        n_out += len(rows)
        q.unlink()
        if (j + 1) % 25 == 0 or j + 1 == len(tiles):
            el = time.time() - t0
            log(f"pass B: tile {j + 1}/{len(tiles)} | {n_out:,} pts kept | "
                f"{el / 60:.1f} min, eta {el / (j + 1) * (len(tiles) - j - 1) / 60:.1f} min")
    shutil.rmtree(SCRATCH, ignore_errors=True)

    frame = json.loads(FRAME_JSON.read_text()) if FRAME_JSON.exists() else {}
    if frame and frame.get("origin_m") != fr.origin_m.tolist():
        log("!! the shared origin moved since the last build — every stage "
            "built on the old one must be rebuilt")
    tc = np.array(list(per_tile.values()))
    frame.update(
        source=str(laz), epsg=epsg, scale_m=fr.s.tolist(), offset_m=fr.o.tolist(),
        origin_m=fr.origin_m.tolist(), tile_m=TILE,
        note="every stage subtracts origin_m from cloud AND survey, all three axes")
    frame.setdefault("grids", {})[f"{GRID:g}"] = dict(
        stamp=stamp, dir=GRID_DIR.name, points_in=n_in, points_kept=n_out,
        fraction_kept=n_out / max(n_in, 1), tiles=len(per_tile),
        tile_points_p50=int(np.median(tc)) if len(tc) else 0,
        tile_points_max=int(tc.max()) if len(tc) else 0,
        rgb_source_max=rgb_max, intensity_source_max=i_max,
        rule="one real point per voxel, the one nearest the voxel centre")
    FRAME_JSON.write_text(json.dumps(frame, indent=2))

    log(f"done: {n_in:,} -> {n_out:,} pts ({n_out / max(n_in, 1) * 100:.1f}%) in "
        f"{len(per_tile)} tiles -> {GRID_DIR}")
    m = RESULTS / "stemnet_v2_measure.json"
    if m.exists():
        want = json.loads(m.read_text()).get("pass2", {}).get("voxels", {}).get(f"{GRID:g}", {})
        if want.get("points") is not None:
            ok = want["points"] == n_out
            log(f"cross-check vs v2_measure: {want['points']:,} expected -> "
                + ("MATCH" if ok else f"MISMATCH ({n_out - want['points']:+,})"))


if __name__ == "__main__":
    main()
