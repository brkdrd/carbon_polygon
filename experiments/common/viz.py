"""Shared plotting for every experiment. Two products:

* semantic mask (binary: vegetation vs background)  -> `plot_semantic_mask`
* instance segmentation (one colour per tree)        -> `plot_instances`

Both save a multi-panel PNG (top-down + side view + summary) so the four
experiments are visually comparable at a glance. Coordinates coming in are the
local/centered frame; that is fine for plotting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless container
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


VEG_CMAP = ListedColormap(["#B4B2A9", "#1D9E75"])  # background, vegetation


def _subsample(n, cap):
    if n <= cap:
        return np.arange(n)
    return np.random.default_rng(0).choice(n, cap, replace=False)


def plot_semantic_mask(xyz, mask, out_path, title, cap=400_000):
    """Binary vegetation mask. `mask` is bool/0-1 per point."""
    xyz = np.asarray(xyz, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    s = _subsample(len(xyz), cap)

    fig = plt.figure(figsize=(16, 7))

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(xyz[s, 0], xyz[s, 1], c=mask[s].astype(int), cmap=VEG_CMAP,
               s=0.15, marker=".")
    ax.set_title(f"Top-down | {mask.sum():,} veg pts ({mask.mean()*100:.1f}%)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")

    ax = fig.add_subplot(1, 2, 2)
    ymid = xyz[:, 1].mean()
    b = np.where(np.abs(xyz[:, 1] - ymid) < 8)[0]
    if len(b) > 200_000:
        b = np.random.default_rng(1).choice(b, 200_000, replace=False)
    ax.scatter(xyz[b, 0], xyz[b, 2], c=mask[b].astype(int), cmap=VEG_CMAP,
               s=0.25, marker=".")
    ax.set_title("Side view (16 m slice)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_aspect("equal")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _instance_palette(n):
    """A large, high-contrast, deterministic categorical palette. Instance 0 /
    negative (unassigned / non-tree) is rendered grey by the caller."""
    rng = np.random.default_rng(42)
    # golden-ratio hue walk for well-separated colours, random value/sat jitter
    import colorsys
    cols = []
    h = 0.0
    for _ in range(max(1, n)):
        h = (h + 0.61803398875) % 1.0
        s = 0.55 + 0.35 * rng.random()
        v = 0.75 + 0.2 * rng.random()
        cols.append(colorsys.hsv_to_rgb(h, s, v))
    return np.array(cols)


def plot_instances(xyz, labels, out_path, title, unassigned=(-1, 0), cap=400_000):
    """Instance segmentation. `labels` = per-point integer instance id.

    Points whose label is in `unassigned` are drawn grey (non-tree / floor /
    unlabelled); every other id gets a distinct colour.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels)
    unassigned = set(unassigned)

    uniq = np.array(sorted(set(labels.tolist()) - unassigned))
    n_inst = len(uniq)
    remap = {lab: i for i, lab in enumerate(uniq)}
    palette = _instance_palette(max(1, n_inst))

    rgb = np.full((len(labels), 3), 0.70)  # grey for unassigned
    is_inst = np.isin(labels, list(uniq)) if n_inst else np.zeros(len(labels), bool)
    if n_inst:
        idx = np.array([remap[int(l)] for l in labels[is_inst]])
        rgb[is_inst] = palette[idx]

    s = _subsample(len(xyz), cap)

    fig = plt.figure(figsize=(16, 7))

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(xyz[s, 0], xyz[s, 1], c=rgb[s], s=0.18, marker=".")
    ax.set_title(f"Top-down | {n_inst} instances | "
                 f"{is_inst.sum():,} tree pts ({is_inst.mean()*100:.1f}%)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")

    ax = fig.add_subplot(1, 2, 2)
    ymid = xyz[:, 1].mean()
    b = np.where(np.abs(xyz[:, 1] - ymid) < 8)[0]
    if len(b) > 200_000:
        b = np.random.default_rng(2).choice(b, 200_000, replace=False)
    ax.scatter(xyz[b, 0], xyz[b, 2], c=rgb[b], s=0.3, marker=".")
    ax.set_title("Side view (16 m slice)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_aspect("equal")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
