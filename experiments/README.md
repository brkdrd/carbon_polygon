# Four-experiment tree-segmentation study (Docker)

Four tree-segmentation experiments, all on **one shared chunk** of the campus
LiDAR cloud, each saved as an image:

| # | Experiment | Model | Input |
|---|------------|-------|-------|
| 1 | `exp1_treelearn_raw.png` | **TreeLearn** | the raw chunk |
| 2 | `exp2_segmentanytree_raw.png` | **SegmentAnyTree** | the raw chunk |
| 3 | `exp3_treelearn_on_sonata_mask.png` | **TreeLearn** | chunk pre-filtered by Sonata |
| 4 | `exp4_segmentanytree_on_sonata_mask.png` | **SegmentAnyTree** | chunk pre-filtered by Sonata |

The **Sonata** stage sits between them: a frozen PTv3 encoder + a linear head
retrained to separate **vegetation (trees + bushes) from everything else** — two
classes. Its mask keeps only vegetation points, and that masked cloud is the
input to experiments 3 and 4. So the study asks: *does pre-masking vegetation
with Sonata change what the instance-segmentation models (TreeLearn,
SegmentAnyTree) produce?*

Two extra context images are also written: `00_chunk_raw.png` (the shared chunk)
and `03_sonata_vegetation_mask.png` (the Sonata mask itself).

## The one command

```bash
cd experiments
cp .env.example .env          # then fill in KAGGLE_USERNAME / KAGGLE_KEY
./run_experiments.sh
```

That builds/pulls every image and runs all four experiments in order. Results
land in `experiments/results/*.png`.

Run a single stage if you want: `./run_experiments.sh prep` (just the chunk),
`./run_experiments.sh sonata`, `./run_experiments.sh exp1` … `exp4`.

## Requirements

- **Docker** + **Compose v2**.
- An **NVIDIA GPU** with the [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed (`docker run --gpus all …` must work). All three model families need
  CUDA — there is no CPU path for real inference.
- **Kaggle credentials** to pull the private source cloud
  (`samsuperman12/my-private-lidar-dataset`). Put `KAGGLE_USERNAME` / `KAGGLE_KEY`
  in `.env`, or drop the `.laz` into `experiments/data/raw/` yourself to skip
  Kaggle entirely.
- Disk/RAM: the images are large (the Sonata image is the Kaggle GPU image; SAT
  is ~10 GB). TreeLearn tile generation is RAM-heavy — a 40 m chunk is modest,
  but budget several GB. TreeLearn needs ~10 GB VRAM.

## The float32 coordinate trap — and how it's handled

The source cloud is in UTM (easting ~7.36e5, northing ~4.77e6). At those
magnitudes, casting to `float32` **before** subtracting a local origin quantizes
northing to a **0.5 m** grid and destroys the geometry — silently, with no error.
See [`../FLOAT32_COORDINATE_BUG.md`](../FLOAT32_COORDINATE_BUG.md).

This pipeline neutralizes it in two ways:

1. **One centering primitive.** `common/geo.center_float64()` is the single place
   coordinates get centered — always in float64, casting to float32 **last**. Its
   behaviour is verified against the bug doc's own numbers in
   [`tests/test_float32_bug.py`](tests/test_float32_bug.py).
2. **Local coordinates on disk.** The prep stage subtracts a fixed origin (in
   float64) and writes the shared chunk in **local coordinates**. Every
   downstream tool — Sonata, TreeLearn, SegmentAnyTree — therefore only ever sees
   small-magnitude coordinates and *cannot* trip the trap, even if its own
   internals cast to float32. The origin is recorded in `data/chunk/chunk_meta.json`
   to georeference results back to UTM.

The prep stage also runs the bug doc's ULP diagnostic on ingest and logs a loud
(handled) warning, so the trap is visible in the logs as an audit trail.

## How it flows

```
prep ──────────────► data/chunk/chunk_local.laz        (shared chunk, exp1 & exp2)
  │
sonata ────────────► data/chunk/chunk_masked_local.laz (vegetation only, exp3 & exp4)
  │                  results/03_sonata_vegetation_mask.png
  ├─ treelearn(raw)      ─► results/exp1_treelearn_raw.png
  ├─ treelearn(masked)   ─► results/exp3_treelearn_on_sonata_mask.png
  ├─ segmentanytree(raw) ─► results/exp2_segmentanytree_raw.png
  └─ segmentanytree(masked)─► results/exp4_segmentanytree_on_sonata_mask.png
```

Every stage bind-mounts the same `./data` and `./results`, and the chunk is
built exactly once — so the four experiments provably share one input.

## Stages / images

| Service | Image | GPU | Role |
|---------|-------|-----|------|
| `prep` | built (`prep/Dockerfile`, CPU) | — | Kaggle download, cut + center the shared chunk, also renders SAT output |
| `sonata` | built (`sonata/Dockerfile`) | ✅ | train vegetation head, write masked chunk + mask image |
| `treelearn_raw` / `_masked` | built (`treelearn/Dockerfile`) | ✅ | TreeLearn instance segmentation → exp1 / exp3 |
| `segmentanytree_raw` / `_masked` | official `maciekwielgosz/segment-any-tree` | ✅ | SegmentAnyTree instance segmentation |
| `sat_render_raw` / `_masked` | reuses `prep` image (CPU) | — | render SAT output → exp2 / exp4 |

- **TreeLearn** ([ecker-lab/TreeLearn](https://github.com/ecker-lab/TreeLearn))
  ships no image, so `treelearn/Dockerfile` reproduces its `setup.sh`
  (CUDA 11.8 + PyTorch 2.0 + `spconv-cu118`). Pretrained weights
  (`model_weights_20241213`) download to `data/models/treelearn/` on first run.
  Output carries a per-point `treeID` (0 = non-tree, 1..N = trees).
- **SegmentAnyTree** ([SmartForest-no/SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree))
  ships a complete image with weights baked in, so we drive it directly. It reads
  a folder of clouds and writes `final_results/*.laz` with a `PredInstance`
  dimension. The default image is compiled for GPU arch sm_60…sm_86; on
  **RTX 40xx / H100** set `SAT_IMAGE=maciekwielgosz/segment-any-tree-cuda11.8.0`
  in `.env`.

## Notes & caveats

- **"Trees + bushes" — combined training sources.** The vegetation head is
  trained on the union of two labelled sources (`TRAIN_SOURCES=whu,semantic3d`):
  - **WHU-STree** (MLS street trees) — positive = tree instances. Matches the
    campus sensor domain but has no shrub class.
  - **Semantic3D** (TLS) — positive = **high vegetation ∪ low vegetation**
    (labels 3 & 4). Its terrestrial geometry is close to the MLS target, and its
    low-vegetation class supplies the shrub/bush supervision WHU-STree lacks.

  Both are mapped to the same binary target and **density-normalized into one
  regime** (the WHU density) before feature extraction; that target density is
  saved in the head checkpoint and reused verbatim to match the campus cloud, so
  training and inference share one density regime. RGB is dropped from Semantic3D
  to keep the colour channel neutral on both sources (matching the campus MLS,
  which has none). Use `TRAIN_SOURCES=whu` to fall back to trees-only.

  **Getting Semantic3D:** it is not auto-bundled. Either mount pre-downloaded
  files into `data/raw/semantic3d/` (the `*.7z` point archives + the shared
  `sem8_labels_training.7z`, or already-extracted `.txt` + `.labels`), or set
  `SEM3D_DOWNLOAD=1` to fetch the configured stations at runtime. The
  semantic3d.net TLS cert is mismatched, so downloads use http with unverified
  SSL — mounting the files yourself is more reliable. If Semantic3D is
  unavailable and download is off, the head trains on WHU alone with a warning.
- **Instance vs semantic.** Sonata produces a *semantic* mask (vegetation vs
  background); TreeLearn and SegmentAnyTree produce *instance* segmentation (one
  colour per tree). The result images reflect that: mask images are two-tone,
  instance images are multi-colour with grey = unassigned/non-tree.
- **Chunk size.** Defaults to a 40 m tile at the cloud centroid (`CHUNK_SIZE` in
  `.env`). TreeLearn tiles internally at ~13.5 m and needs decent trunk density
  (~1 pt per 0.1 m³); very sparse trunks are its main failure mode.
- **Reproducibility.** Downloaded weights, the trained Sonata head, and the chunk
  are all cached under `data/`, so re-runs reuse them. Set `FORCE_PREP=1` /
  `FORCE_TRAIN=1` to rebuild.

## Layout

```
experiments/
  run_experiments.sh        # the one command
  docker-compose.yml
  .env.example
  common/                   # geo.py (the float32 fix), viz.py, config.py
  prep/                     # chunk builder (+ SAT renderer), Dockerfile
  sonata/                   # vegetation head train + mask, Dockerfile
  treelearn/                # TreeLearn wrapper, Dockerfile
  segmentanytree/           # SAT output renderer (SAT itself is the official image)
  tests/                    # float32-fix regression test (CPU, no GPU needed)
  data/                     # runtime volume (git-ignored artifacts)
  results/                  # output PNGs
```
