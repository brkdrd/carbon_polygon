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

### Windows

Two options, both use the same images and compose file:

- **WSL2 (simplest):** open your WSL2 distro and run `./run_experiments.sh`
  unchanged.
- **Native PowerShell:** use the equivalent launcher

  ```powershell
  cd experiments
  Copy-Item .env.example .env      # then fill in KAGGLE_USERNAME / KAGGLE_KEY
  .\run_experiments.ps1            # or: build | prep | sonata | exp1..exp4
  ```

  If execution policy blocks it: `powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1`.

Either way you need Docker with **GPU passthrough into WSL2** — see
[Giving WSL2 access to the GPU](#giving-wsl2-access-to-the-gpu) below.

## Requirements

- **Docker** + **Compose v2**.
- An **NVIDIA GPU** with the [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed (`docker run --gpus all …` must work). All three model families need
  CUDA — there is no CPU path for real inference.
- **Kaggle credentials** to pull the private source cloud
  (`samsuperman12/my-private-lidar-dataset`). Put `KAGGLE_USERNAME` / `KAGGLE_KEY`
  in `.env`, or drop the `.laz` into `experiments/data/raw/` yourself to skip
  Kaggle entirely.
- Disk/RAM: the images are large (Sonata and TreeLearn build on PyTorch CUDA
  bases, ~7 GB each; SegmentAnyTree is ~10 GB). TreeLearn tile generation is
  RAM-heavy — a 40 m chunk is modest, but budget several GB. TreeLearn needs
  ~10 GB VRAM.

## Giving WSL2 access to the GPU

To run the CUDA containers on Windows, the GPU has to reach the Linux side. What
you need depends on how Docker is installed:

**Prerequisites (both options):**
- Windows 10 21H2+ or Windows 11, WSL2 (not WSL1); `wsl --update`.
- The **NVIDIA driver installed on Windows** (any recent GeForce/RTX/Quadro
  driver includes WSL2 CUDA support). **Do not install a Linux GPU driver inside
  WSL** — the Windows driver projects the GPU in and provides `libcuda`;
  installing a driver in WSL breaks it.
- Verify inside WSL2: `nvidia-smi` should list your GPU (it lives in
  `/usr/lib/wsl/lib/`). If it works, the GPU is visible to Linux.

**Option A — Docker Desktop (least setup):** enable the WSL2 backend
(Settings → General → *Use the WSL 2 based engine*) and WSL integration
(Settings → Resources → *WSL Integration*). GPU support is built in — no
container toolkit to install. `docker run --gpus all …` just works.

**Option B — Docker Engine inside WSL2** (what a bare `apt install docker.io`
gives you; `docker info` shows only the `runc` runtime): install the **NVIDIA
Container Toolkit** in the distro:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker   # registers the nvidia runtime
sudo service docker restart                            # or: systemctl restart docker
```

**Verify GPU passthrough (either option):**

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that prints your GPU, `./run_experiments.sh` / `run_experiments.ps1` will too.
(Keep project files under the WSL/Linux filesystem, not `/mnt/c/...`, for
tolerable I/O speed.)

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
| `stemnet_train` / `_infer` | built (`stemnet/Dockerfile`, on `basewalker`) | ✅ | dense chunk model: per point, tree + offset to its base |

- **TreeLearn** ([ecker-lab/TreeLearn](https://github.com/ecker-lab/TreeLearn))
  ships no image, so `treelearn/Dockerfile` reproduces its `setup.sh` on a
  CUDA 12.8 + PyTorch 2.7 base (`spconv-cu126`) — the repo's own CUDA 11.8
  recipe cannot run on Blackwell (RTX 50xx). Pretrained weights
  (`model_weights_20241213`) download to `data/models/treelearn/` on first run.
  Output carries a per-point `treeID` (0 = non-tree, 1..N = trees).
- **SegmentAnyTree** ([SmartForest-no/SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree))
  ships a complete image with weights baked in, so we drive it directly. It reads
  a folder of clouds and writes `final_results/*.laz` with a `PredInstance`
  dimension. The default image is compiled for GPU arch sm_60…sm_86; on
  **RTX 40xx / H100** set `SAT_IMAGE=maciekwielgosz/segment-any-tree-cuda11.8.0`
  in `.env`. **RTX 50xx (Blackwell, sm_120) is not supported by either official
  image** — CUDA 11.8 predates the architecture entirely — so the SAT stages
  (exp 2 & 4) need a CPU fallback or a self-rebuilt image on that hardware.

## BaseWalker (tree-base detection — separate study track)

Direct detection of tree base points instead of instance segmentation: walkers
seeded on a CSF ground surface iteratively step toward the nearest tree base
(frozen Sonata encoder on a small sphere crop → gated cross-attention decoder →
offset + confidence), trained on the campus RTK stem map (`basewalker/
poderevka.csv`, ~4k bases, fork-merged, spatial train/val split). Detections =
walker endpoints after NMS; metrics are P/R/F1 vs the held-out val block.

```
./run_experiments.sh bw_prep    # scene tiles + normals + DEM + labels + seeds
./run_experiments.sh bw_train   # trains the walker decoder (GPU, ~hours)
./run_experiments.sh bw_infer   # detections + metrics + results image
./run_experiments.sh bw_render  # 4 chunk images: detections vs ground truth
```

### The attributed survey patch

`basewalker/poderevka.csv` gives 4,050 tree positions and nothing else. One
block of it was surveyed properly, and that file is now in the repo:

| file | what it is |
| --- | --- |
| `basewalker/poderevka_attributes.csv` | 392 trees in ~1.14 ha with **girth** (388) and a **Latin species name** (388, 25 distinct). Derived from the xlsx below; `n,lat,lon,z` are rows 1..392 of the stem map's `base_1` block, matching to 1e-9, so species joins **by row index** with no spatial matching |
| `basewalker/poderevka_attributes_source.xlsx` | the survey workbook as received, kept for provenance |

```
./run_experiments.sh gw_infer_patch     # or: run_experiments.cmd gw_infer_patch
```

Outputs, named apart so a patch run never overwrites a full one:

```
results/geowalker_patch_trees.csv              one row per GT base: species,
                                               girth, found 0/1, error, score
results/geowalker_patch_detections_status.csv  one row per detection: tp/fp,
                                               which tree, error
results/geowalker_patch_metrics.json           P/R/F1 + the patch composition
results/21_geowalker_patch_detections.png
```

Two things the stage prints that any reader of its score needs:

* **The patch is training ground.** The train/val split is a quantile on local
  `x` (rightmost 20% is val) and this patch sits in the left tail, so the model
  fitted these labels. `patch.trained_on` in the json is `true` and the console
  says so in capitals. A number from here is an upper bound on unseen
  performance, not an estimate of it. Holding this patch out instead would mean
  changing the split in `prep_scene.py` and retraining.
* **The attributed trees are not the whole GT.** Their hull also contains 147
  stem-map trees from the same RTK session that nobody attributed — a median
  12 m from the nearest attributed tree, so different trees, not
  re-measurements. They are real trees, so they stay in the GT and a miss
  counts; only the per-species table is restricted to the attributed subset.

Knobs: `PATCH_MARGIN` (3.0 m; `GW_PATCH_MARGIN` still honoured) is how far
outside the surveyed hull still
counts as judgeable. The footprint is the hull **or** that buffer, because a
clearing inside the patch is surveyed ground and a detection in it is a real
false positive.

### The control: is the network beating a density peak?

Nothing in this repo had ever measured that, so every GeoWalker number was
being read without a scale. `gw_baseline` supplies one. It runs two detectors
from the forest-inventory literature over the same tiles, inside the same patch
footprint, and scores them with the same `match_pairs` matcher:

* **stem** — slice the cloud at breast height (`GW_STEM_LO`..`GW_STEM_HI`, default
  1–3 m above the DEM), find density peaks in plan view, then make each
  candidate prove it goes *up*: of the 0.5 m height bins between
  `GW_STEM_COL_LO` and `GW_STEM_COL_HI` within `GW_STEM_COL_R` of it, what
  fraction hold a point? A trunk fills ~all of them, a parked car fills the
  bottom two. That fraction is the detector's entire confidence score.
* **chm** — local maxima of `gw_height`'s canopy height model, the classical
  variable-window filter. Its blind spot is worth seeing plainly: an understory
  sapling under an oak produces no local maximum, so a CHM detector cannot
  count it at any threshold.

```bash
./run_experiments.sh gw_baseline        # or: run_experiments.cmd gw_baseline
```

| file | what it holds |
| --- | --- |
| `results/geowalker_baseline_patch_metrics.json` | every method, P/R/F1 at 0.5 m and 1.0 m |
| `results/geowalker_baseline_patch_<method>_detections.csv` | `x,y,z,score` |
| `results/geowalker_baseline_patch_trees.csv` | one row per GT base, a found/err column per method, plus GeoWalker's own column when `gw_infer_patch` has run |

The console table that matters is **recall by stem size**, because that is where
the target definition and the model part company: about half the surveyed stems
in the patch are under 10 cm DBH and together carry ~4% of its basal area. A
headline recall counts them one-for-one with a 40 cm oak; the basal-area
weighted recall printed underneath does not. Report both.

CPU only, no GPU, no training — so this is the cheap number to get first, before
spending another 1.6 GPU-hours on a walk.


`bw_train` **resumes**: if `data/models/basewalker_decoder.pth` exists it is
loaded — weights, AdamW moments and the best val F1 — and `BW_ITERS` more
iterations are trained on top, on a fresh cosine cycle from `BW_RESUME_LR`
(default = `BW_LR`) and an RNG offset by the iterations already done, so the
extra run does not replay the first run's seed batches. `BW_ITERS` counts
*this* run and defaults to **9000** (3x the 3000 of the original from-scratch
run); `BW_RESUME=0` starts cold. `basewalker_decoder.pth` still only ever holds
the best-val-F1 decoder, so a fine-tune that fails to improve cannot degrade
what `bw_infer` / `bw_render` read; the final-iteration decoder is always
written alongside it as `basewalker_decoder_last.pth`.

`bw_render` cuts four 40 m chunks out of the full map and draws, per chunk, the
cloud (with the 0.3–3 m trunk slab highlighted), the RTK bases and the model's
detections coloured TP / FP / missed, plus per-chunk P/R/F1. Chunks are the
densest non-overlapping windows, **two from the held-out val block and two from
the train block**, so the same picture shows both regimes — each panel says
which block it came from. Outputs `results/11_basewalker_chunks.png` (2×2
overview), `results/11_basewalker_chunk{1..4}.png`,
`basewalker_chunk_metrics.json` and `basewalker_chunk_detections.csv`.

Knobs (env): `BW_SPHERE_R` (2 m), `BW_STEPS` (8), `BW_GAMMA` (0.85),
`BW_CONF_R` (0.5 m), `BW_NMS_R` (0.5 m), `BW_ITERS` (9000, per run), `BW_BATCH`,
`BW_UTM_EPSG` (32652), `BW_DEM_METHOD` (csf|percentile).
Train-only: `BW_RESUME` (1), `BW_RESUME_LR` (= `BW_LR`), `BW_SEED` (0).
Render-only: `BW_CHUNK_M` (40 m), `BW_N_CHUNKS` (4), `BW_CHUNK_SPLIT`
(`mix`|`val`|`train`), `BW_CHUNK_MIN_GT` (5), `BW_CONF_THRESH` (the
checkpoint's best-F1 threshold), and `BW_CHUNKS="cx,cy;cx,cy;..."` to place the
chunks by hand in local coordinates.

## GeoWalker (walking a through-cloud distance field — separate study track)

A second, simpler take on tree-base detection. Same frozen Sonata encoder, same
scene, but the supervision is a **distance field computed through the point
cloud** instead of the straight line to the nearest label, and the model is a
plain transformer decoder that answers one question per step: *given this small
sphere of points (and the last few I walked through), which single point should
I move to?*

**Windows** — the usual case for the GPU box:

```bat
git pull
cd experiments
run_experiments.cmd geowalker     :: build + self-check + field + train + infer
```

Use the `.cmd` wrapper, not the `.ps1` directly. A stock Windows client ships
with `ExecutionPolicy=Restricted` and refuses to run the PowerShell script at
all (*"running scripts is disabled on this system"* / *"выполнение сценариев
отключено в этой системе"*). Batch files are exempt from that policy, so the
wrapper launches the same script with a **per-process** bypass — it changes
nothing about the machine. Calling the `.ps1` yourself works too if your policy
already allows it, or via
`powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 geowalker`.

**Linux / WSL2:**

```bash
git pull
cd experiments
./run_experiments.sh geowalker    # identical sequence
```

One stage at a time (same target names in both launchers):

| target | what it does | needs |
|--------|--------------|-------|
| `gw_test` | CPU self-check on synthetic geometry | nothing, ~10 s |
| `gpu_check` | proves the GPU reaches a container **and that kernels run** | GPU, ~5 s |
| `gw_field` | the through-cloud distance field | CPU, RAM-bound |
| `gw_train` | train the decoder by simulated walking | GPU, hours |
| `gw_infer` | detections + metrics + figures | GPU, ~minutes |
| `gw_infer_head` | the same, with endpoints ranked by the `gw_head` classifier instead of the walker's read-out — both rankings reported from one walk | GPU, same cost as `gw_infer` |
| `gw_infer_agg` | the same walk read properly: best step per walker instead of the last, converging walkers counted as votes instead of suppressed, canopy-height gate from `gw_height`. Every pipeline reported in one table | GPU, same cost as `gw_infer` |
| `gw_diag` | why recall is what it is (walk / NMS / threshold) | GPU, ~minutes |
| `gw_infer_patch` | inference restricted to the densely attributed survey patch — 392 trees, 25 species, girth on 388, ~1.14 ha. Writes a per-tree answer sheet and a per-species recall. **These are training labels**, and the stage says so | GPU, ~1.6 h (2.9k seeds) |
| `gw_baseline` | the control: classical training-free detection on the same patch, scored by the same matcher — breast-height density peaks confirmed by a vertical column, and CHM local maxima. Says whether the network is beating a 1990s algorithm | CPU, ~minutes |
| `gw_height` | canopy height vs the annotators' criterion | CPU, ~minutes |
| `gw_head` | can the tree/not-tree decision be trained on its own? | GPU, walk-bound; resumable |

```bat
run_experiments.cmd gw_test       :: ...or gw_field | gw_train | gw_infer
```

If you would rather not use a wrapper at all, the stages are plain compose
services — this is exactly what the launchers run:

```bat
:: three separate builds, in this order: the images chain via FROM
:: (geowalker <- basewalker <- sonata) and compose only understands
:: depends_on, so one combined build call can order them wrong.
docker compose build sonata
docker compose build basewalker_prep
docker compose build geowalker_field

docker compose run --rm --no-deps geowalker_test
docker compose run --rm --no-deps basewalker_prep
docker compose run --rm --no-deps geowalker_field
docker compose run --rm --no-deps geowalker_train
docker compose run --rm --no-deps geowalker_infer
```

(Going that route, copy `.env.example` to `.env` first — the launchers do that
for you, and compose errors out without it.)

**The CPU stages print an NVIDIA warning — that is expected.** `bw_prep`,
`gw_field` and `gw_test` deliberately reserve no GPU (the field is scipy
Dijkstra, not CUDA), so the CUDA base image's entrypoint prints *"The NVIDIA
Driver was not detected"* on startup. It is cosmetic and those stages run fine.
Only `gw_train` and `gw_infer` request the card; run `gpu_check` to confirm they
will get it before committing to the long stages.

On an RTX 50xx (Blackwell, sm_120), `torch.cuda.is_available()` alone is not
proof: a torch built only to sm_90 enumerates the device and then dies on the
first kernel. `gpu_check` launches a real matmul, which is why it is worth the
five seconds.

Nothing else is needed on a fresh machine: `.env` is created from `.env.example`
if absent (both launchers do this; Compose v2 handles the CRLF a Windows
checkout gives it), the images build in dependency order (`sonata` → `basewalker` →
`geowalker`), and every stage is idempotent — `gw_field` skips an existing
`field.npz`, `bw_prep` skips existing tiles, and `gw_train` resumes from the
checkpoint. A re-run after an interruption picks up where it stopped.

`gw_test` is worth running first on any new machine: it exercises the field
maths and the loss gradients on synthetic geometry in about ten seconds, so a
broken environment surfaces before you commit hours of GPU time. It is also the
first thing `geowalker` runs.

`gw_field` runs the BaseWalker scene prep first (it is idempotent), because both
tracks share exactly the same scene: `data/basewalker/tiles/`, `dem.npz`,
`labels.npz`, `seeds.npz`, one local frame, one spatial train/val split.

### Stage A — the field (`geowalker/geodesic.py`)

For every point, the distance to the nearest RTK tree base **measured along the
cloud, never through air**: start with the bases as the reached set, repeatedly
absorb the closest not-yet-reached point and carry its edge length along. That
is Dijkstra with the frontier in a heap, so it is one `scipy.sparse.csgraph`
call over a kNN graph whose edges are capped at `GW_MAX_EDGE` metres.

* The result is always ≥ the straight-line distance (a path through the cloud
  cannot be shorter than the chord) — checked and reported on every build, and
  regression-tested in `tests/test_geowalker.py`.
* Points with no path at all (an island across 5 m of air) stay at `+inf` and
  are simply excluded from the loss, not clamped to a lie.
* Resolution: the graph is built on a `GW_VOXEL` (0.25 m) occupancy grid, not on
  the raw 0.05 m cloud — the campus is ~1.5·10⁹ raw points and a kNN graph on
  even its 0.05 m subsample does not fit in RAM. Nodes are voxel centres; edge
  weights are true 3D distances. If the node count would exceed `GW_MAX_NODES`
  the voxel is coarsened automatically rather than dying half way through.
* Two fields are written: `g_train` (seeded on train bases only — the one the
  loss reads) and `g_all` (for figures and diagnostics). Training supervision
  therefore never sees a val base.

Output: `data/geowalker/field.npz`, `field_meta.json` and
`results/20_geowalker_field.png` — the field from above, its distribution, and a
vertical slice showing it run down the trunks to the ground.

### Stage B — the walker (`geowalker/model.py`, `train.py`)

Per step: the sphere of radius `GW_SPHERE_R` around the current center is encoded
once by the **frozen** Sonata encoder into coarse tokens. The decoder cross-
attends from a single learned query over those tokens **plus the tokens of the
previous `GW_MEM` (3) spheres** — the dynamics context, so it knows where it came
from — and emits the next point, as a displacement capped at `GW_MAX_STEP`.
Every token, memory included, is re-expressed relative to the *current* center
and gated by `relu(1 - |rel| / R_ctx)`, which is what carries gradient back to
the center even though the encoder itself never sees one.

Each simulated step is scored by two terms, both differentiable in the predicted
point (the point *set* is a detached lookup; the distances to it are not):

| term | what it is | why |
|------|------------|-----|
| `l_geo` | Gaussian-weighted (σ = `GW_GEO_SIGMA`) mean **field value** of the cloud around the new center | its gradient is a mean-shift toward the better-connected side; its minimum is the base itself. Linearised past `GW_HUBER_M` so a walker 40 m out cannot drown one that is 1 m out |
| `l_dens` | mean distance to the `GW_DENS_K` nearest lidar points, hinged at `GW_DENS_D0` | "is there actually structure here" — free along a trunk, expensive in open air. This is the term that keeps the walker attached to the cloud |

A third, **detached** read-out term trains the decoder to also report `log1p` of
the field value at its current position. It cannot steer the walk (the target is
detached) and it needs no extra labels — it exists so an endpoint can be scored
at inference, where no field exists: `score = exp(-ĝ / GW_SCORE_TAU)`, so 1 means
"I am standing on a tree base". Detections are endpoints after NMS + threshold,
matched greedily against the held-out val bases, exactly as BaseWalker scores.

Walkers whose seed is further than `GW_REACH_M` through the cloud are excluded
from `l_geo` only: a 2 m sphere carries no hint of direction at 40 m, so that
term would be unlearnable noise setting the loss magnitude. They still train the
density term and the read-out (as true negatives). Same lesson as BaseWalker;
so are the step-norm cap and the `GW_TBPTT` truncation, which is what keeps the
8-step Jacobian product from exploding.

`gw_train` **resumes** on the same contract as `bw_train`: an existing
`data/models/geowalker_decoder.pth` is loaded with its AdamW state and best val
F1, `GW_ITERS` (9000) more iterations run on a fresh cosine cycle from
`GW_RESUME_LR`, and the checkpoint only ever holds the best-val-F1 decoder
(`geowalker_decoder_last.pth` always holds the final one). `GW_RESUME=0` starts
cold.

### Stage C — inference (`geowalker/infer.py`)

No field is used: the model only ever sees the sphere around itself and its last
3 spheres. Writes `results/geowalker_detections.csv`,
`geowalker_metrics.json`, `results/21_geowalker_detections.png` and
`results/22_geowalker_walks.png` — the actual walks drawn on top of the field
they were trained to descend, which is the fastest way to see whether the thing
learned to walk or learned to stand still.

### Knobs (env)

Field: `GW_VOXEL` (0.25 m), `GW_KNN` (16), `GW_MAX_EDGE` (2×voxel),
`GW_BASE_R` (0.5 m), `GW_MAX_NODES` (12e6), `FORCE_GW_FIELD`.
Walker: `GW_SPHERE_R` (2 m), `GW_STEPS` (8), `GW_MEM` (3), `GW_MAX_STEP` (1 m),
`GW_GAMMA` (0.85), `GW_FIELD_R` (2 m), `GW_GEO_SIGMA` (0.4 m), `GW_HUBER_M` (2 m),
`GW_DENS_K` (12), `GW_DENS_D0` (0.5 m), `GW_W_DENS` (2.0), `GW_W_AUX` (0.5),
`GW_REACH_M` (12 m), `GW_TBPTT` (2), `GW_DEM_PROJECT` (0 — set 1 to make the
walker ride the ground instead of moving freely in 3D), `GW_SCORE_TAU` (0.5),
`GW_NMS_R` (0.5 m).
Train: `GW_ITERS` (9000, per run), `GW_BATCH` (48), `GW_LR`, `GW_EVAL_EVERY`,
`GW_NEAR_FRAC` (0.7), `GW_NEAR_R`, `GW_RESUME` (1), `GW_SEED`.
Infer: `GW_INFER_REGION` (`val`|`all`), `GW_CONF_THRESH`, `GW_WALK_WIN` (40 m).
Throughput: `GW_DETECT_BATCH` (64 — seeds walked per encoder forward; this was a
hard-coded 256, which is ~1M points per Sonata pass and, on top of the ~8.5 GB
the scene store pins on the card, does not leave much of a 16 GB GPU. Windows
spills to host RAM instead of raising, at a fraction of the bandwidth: a
59k-seed walk at 256 ran **over 24 h without finishing**, while training, at 48,
holds its rate. Raise it only if the progress line's GPU figure shows headroom), `GW_TF32` (1),
`GW_ENC_AMP` (0 — bf16 encoder, roughly another 2x, at the cost of the last
digits of every reported number), `GW_PROGRESS_S` (30 s between progress lines,
0 = silent), `GW_PROFILE` (1 = add a `ball/enc/ctx/dec` time breakdown to each
progress line; it synchronises, so it is not free), `GW_DEVICE` (`cuda`).
Tree head: `GW_HEAD_SEEDS` (0 = walk every seed; a cap thins the hard negatives
only), `GW_HEAD_CHUNK` (8192 — how often the walk is checkpointed to disk),
`GW_HEAD_CACHE_GB` (6 — RAM for the one-time token cache, which is what makes
the epochs cost no encoder time), `GW_HEAD_FORCE` (1 = ignore both caches and
mine again), plus `GW_HEAD_EPOCHS`/`BATCH`/`LR`/`EASY`/`JITTER`/`TARGET_F1`.

The walk (`gw_infer`, `gw_diag`, `gw_head`, and every `gw_train` eval) prints
position, rate, ETA and peak GPU memory as it goes, and `gw_head` checkpoints
its endpoints every `GW_HEAD_CHUNK` seeds — a Ctrl-C costs one chunk, not the
whole pass. Both of its caches live in `data/geowalker/`
(`treehead_endpoints.npz`, `treehead_dataset.npz`).

### How it differs from BaseWalker

Both walk a frozen-Sonata sphere toward a tree base; the difference is what
"closer" means and what the model remembers.

| | BaseWalker | GeoWalker |
|---|---|---|
| target | straight-line distance to the nearest base | distance **through the cloud** to the nearest base |
| loss | Huber on that distance + BCE "base within 0.5 m" | field mean around the new point + a point-density penalty |
| context | the current sphere only | the current sphere **+ the previous 3** |
| step | unbounded offset, projected onto the DEM | norm-capped, free in 3D by default |
| confidence | a supervised binary head | a read-out of the predicted field value |

They share the scene, the split and the metric, so their numbers are directly
comparable — `10_/11_` images are BaseWalker's, `20_/21_/22_` are GeoWalker's.


## StemNet — a dense chunk model (separate study track)

**Not a walker.** No sphere, no step, no memory of previous positions, no
through-cloud distance field, no seed grid. The scene is just a chunk of points,
processed one window at a time, and every point is read exactly once. It has its
own image and imports nothing from `geowalker/` or `basewalker/` — the only
things the tracks share are the prepared scene (tiles, DEM, labels), the frozen
encoder and the matcher that scores them, and all three live in `common/`:

| in `common/` | why it is shared |
| --- | --- |
| `detect.py` | `match_metrics` / `match_pairs` — four tracks are scored by this, and a second greedy matcher that drifted by one tie-break would make their results incomparable |
| `dem.py` | one bilinear ground lookup; a half-cell disagreement would shift every detection it touched |
| `encoder.py` | the frozen Sonata loader and the dense read-out; a crop centre-shifted differently is one the pretrained weights have never seen, and the failure is silent |
| `patch.py` | the attributed survey patch — it describes the *survey*, not any one model |
| `raster.py` | `peaks`, so "two stems 0.6 m apart are two detections" means the same thing in the CHM filter, the density detector and the vote histogram |


The walker's own measurement is what motivated this. On the attributed patch it
missed 305 of 522 trees at 0.5 m, but **94% of those misses had a detection
cluster within 2 m** and 143 of them within a single metre — and the median
cluster carried **one vote** out of the ~5 walkers that passed near each tree.
Eight sequential sphere encodes buy one opinion about one trajectory. The
information needed to place a stem is already in the points standing around it.

So the same frozen Sonata encoder is read **densely** instead:

* the coarse tokens are upcast back to the input points through the pooling
  chain (the standard Sonata linear-probe read-out; `common.encoder.upcast`
  falls back to nearest-token features and says so if a build does not expose
  the chain);
* two heads per point — a logit for *this point belongs to a tree stem*, and a
  **2-D horizontal** offset to that stem's base;
* detections are peaks in the confidence-weighted histogram of the **shifted**
  points. A detection's score is the vote mass that agreed on it, so consensus
  is dense by construction.

**Why those two targets.** The only per-tree truth in this project is a set of
RTK stem bases — no per-point instance or semantic labels exist for this cloud.
So "is a tree" is defined as *within `SN_NEAR_R` metres **horizontally** of a
surveyed base*, which inside the breast-height band is a trunk and its
immediate surroundings. Narrower than tree/not-tree, and it is what the data can
actually supervise. Horizontal because a trunk point 3 m up is 3 m from its base
in 3-D but ~0.15 m from it in plan view; the base's z comes from the DEM, so the
head never predicts it and the target is a short, low-variance 2-D vector
instead of the walker's 8 m trajectory.

**The terrain is handled explicitly, because this scene is a hillside.** The
z handed to the encoder is **height above ground**, not raw z (`SN_DETREND=1`,
the default). A crop is 20 m wide and the pretrained CenterShift subtracts a
single `z.min()` from it, so with raw z an uphill trunk arrives at local
z 8–12 m and an identical downhill trunk at 0–4 m — same object, two completely
different inputs, and nothing tells the encoder where the ground is. Subtracting
the DEM costs almost nothing geometrically: the shift depends only on (x, y) and
a trunk occupies essentially one (x, y), so it translates **rigidly** and its
shape is preserved exactly. What flattens is the ground, which is what we wanted
gone. (Normals are left describing the real surface — a trunk's is horizontal
and unaffected, and the band starts at 0.3 m so there is little ground left to
disagree about.) A head trained one way refuses to run the other way rather than
be fed a z convention it never learned.

Both stages print the terrain before they do anything (shape of the output —
the actual numbers for this scene have not been measured yet):

```
[sn] relief: the scene spans NN.N m of ground elevation | within one 20 m crop,
     p50 N.N m p95 N.N m max NN.N m
[sn] DEM vs 392 surveyed elevations (datum offset -N.NN m removed):
     IQR N.NN m | p5 -N.NN p95 +N.NN m | N% beyond 0.5 m
```

The second line is the one that can invalidate everything downstream: the band,
the labels' z and the detrending are all measured **from** the DEM, so a DEM
that is wrong on a slope slides the slab off the stems and no amount of training
recovers it. The survey's own levelled elevations on the 392 attributed trees are
the only independent check available, and a constant offset is a vertical datum
difference rather than an error, so the median is removed and the **spread** is
what gets reported. If the IQR approaches the band thickness the stage says so
and tells you to widen `SN_BAND_LO/HI` or rebuild the DEM with
`BW_DEM_METHOD=csf FORCE_BW_PREP=1`.

**Why it is cheap.** Only the `SN_BAND_LO`..`SN_BAND_HI` slice above the
DEM is ever encoded — the ground carries no stem and the crown cannot say where
a trunk meets the soil — and that slice is voxel-deduplicated at `SN_VOXEL`,
which is lossless as far as an encoder that serialises on a 0.05 m lattice is
concerned. One pass per window over ~5-10% of the points replaces 9 encodes per
2 m seed. The stage prints its own ha/h next to the walker's 1.35 h/ha.

**What fits on the card.** This is the constraint that decides `SN_WIN` and
`SN_PTS`, so both stages print their memory rather than leaving it to be
inferred. Only the **band** is ever uploaded: the scene is streamed tile by
tile, points outside `SN_BAND_LO`..`SN_BAND_HI` are dropped on the way in, and
the 83% of the cloud that the band rejects never reaches the device. Keeping
the full 267 M-point cloud resident cost 6.4 GB — more than the encoder, the
dense read-out and the head's activations put together.

The per-crop cost is dominated by stage-1 attention, which materialises
`N x heads x patch_size` floats and holds two copies while its softmax runs:
about **25 kB of device memory per crop point**, plus ~5 kB for the 1232-d
read-out and ~10 kB for the head's retained activations. A 20 m crop on this
scene is ~150 k points, so:

| crop points | peak above the band | note |
| --- | --- | --- |
| 80 k | ~3.1 GB | sparse window |
| 160 k | ~6.2 GB | `SN_PTS` default, just above typical |
| 250 k | ~9.6 GB | the old default; needed 16.5 GB all-in and did not fit |

A window denser than `SN_PTS` is **subsampled, never truncated** (`pos` is in
bucket order, so a slice would drop one spatial corner and every tree in it);
raising `SN_VOXEL` thins dense windows evenly instead and is the better lever
if many of them are being subsampled. Training survives an exhausted card: the
window is retried at half the crop and, failing that, skipped and counted, so
one thicket cannot kill a 4,000-iteration run.

One trap worth knowing, because it does not announce itself: this image runs
`spconv-cu126` on sm_120 through its NVRTC fallback, and an allocation that
fails inside those kernels surfaces as `CUDA error: an illegal memory access
was encountered` at the next device sync — several frames away from the real
cause, and reported from Sonata's attention rather than from the allocator. If
you see that, read the `[sn] gpu ...` high-water line, not the traceback.

```bash
./run_experiments.sh stem_test      # CPU self-check, seconds, no GPU
./run_experiments.sh stemnet        # test, train, val, patch
./run_experiments.sh stem_near      # settle SN_NEAR_R first, ~55 min
./run_experiments.sh stem_long      # 40k iters, ~2 h
./run_experiments.sh stem_full      # the overnight run: 150k iters, ~8 h
./run_experiments.sh stem_infer     # the held-out score, on its own
```

`SN_TAG` names a run, so `stemnet<TAG>.pth` and `results/stemnet<TAG>_*` do not
collide — which is what makes a comparison like `stem_near` possible at all.

**How long is long enough.** `SN_ITERS` defaults to 4,000, which is ~1.3
passes over the 3,028 train window centres and about 13 minutes on a 5090 at
190 ms per window — a smoke test, not a trained model. A run that ends with the
cosine schedule at 4e-6 while the classification loss is still falling has
converged its *schedule*, not its head. `stem_long` is 40,000 iterations
(~2 h) with the eval wide enough to choose a checkpoint on, and starts cold,
because a 13-minute head is not worth warm-starting from. `stem_full` is
150,000 (~8 h), the overnight run.

**Window sampling.** A training window is an anchor base plus a jitter of up to
`SN_JITTER` (default `SN_WIN/3`), and **the jitter is redrawn every
iteration**. It used to be drawn once before the loop, which pinned a run to
one fixed set of 3,028 crops, so a longer run bought more passes over the same
windows rather than more ground — at 13 passes tolerable, at 50 it is
memorisation. The anchor pool is filtered on the **worst** jitter an anchor can
draw rather than on the anchor itself, so no training crop can reach across the
leakage margin into the val block however the dice fall; a crop that did would
invalidate every score the run produced, silently.

An interrupted run resumes from `stemnet_last.pth`, the training state, not
from `stemnet.pth`, the best-scoring one (`SN_RESUME_FROM=best` to override).
Resuming from `best` rewinds the optimiser to whichever earlier iteration
happened to score well and re-covers ground already walked, which on an
overnight run is hours thrown away — and the score it rewinds to may itself be
an eval fluke.

Beware the training-time eval: it is what **best-F1 checkpointing selects on**,
so it has to be wide. At the old `SN_EVAL_WIN=12` it covered 85 val bases in
0.31 ha, which puts one standard error on recall at ±0.054 — wider than the
entire spread of a run's eval scores (0.407-0.470 over 11 evals, sd 0.020). The
"best" head was then the luckiest of 16 noisy draws rather than the best one.
The default is now 48 windows (~340 bases, ±0.027) and `stem_long` uses 64
(~450 bases, ±0.023). The honest number is still `stem_infer` over the whole
val block, never the training eval.

| file | what it holds |
| --- | --- |
| `data/models/stemnet.pth` | best-val-F1 head (+ `_last.pth`) |
| `results/stemnet_val_metrics.json` | the honest score: P/R/F1/RMSE at 0.5 m and 1.0 m, threshold sweep, ha/h |
| `results/stemnet_val_detections.csv` | `x,y,z,score,n_votes` |
| `results/stemnet_patch_trees.csv` | same columns as `geowalker_patch_trees.csv`, so the walker and the head diff line for line |
| `results/30_stemnet_val.png` | detections, and the vote cloud beside them |

Training windows are centred on train bases and never allowed to reach across
the leakage margin into the val block; validation during training is the **full
detection metric** (vote, cluster, match) on a fixed sample of val windows, and
only the best-F1 head is checkpointed. Resumes by default (`SN_RESUME=0` starts
cold); `SN_ITERS` counts *this* run.

Knobs: `SN_WIN` (16 m predicted square), `SN_PAD` (2 m context ring,
encoded but not predicted on, so inner squares tile exactly and no point votes
twice), `SN_BAND_LO/HI` (0.3/4.0 m), `SN_NEAR_R` (1.0 m), `SN_VOXEL`
(0.05 m), `SN_PTS` (160k per crop, a memory bound — see below), `SN_P` (0.5, a point votes above
this), `SN_VOTE_CELL/SEP/R/MIN` (0.1 / 0.6 / 0.3 / 6.0), `SN_DETREND` (1),
`SN_JITTER` (`SN_WIN/3`, redrawn per iteration), `SN_TAG` (names a run),
`SN_RESUME_FROM` (`last`), `SN_ITERS` (4000, a smoke test — see above),
`SN_LR`, `SN_W_OFF`, `SN_EVAL` (250), `SN_EVAL_WIN` (48 windows, which is what
best-F1 checkpointing is selected on).


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
  common/                   # shared by every track, and the reason the
                            #   numbers compare: geo.py (the float32 fix),
                            #   dem.py, detect.py (the one matcher),
                            #   encoder.py (frozen Sonata + dense read-out),
                            #   patch.py (the attributed survey), raster.py
  prep/                     # chunk builder (+ SAT renderer), Dockerfile
  sonata/                   # vegetation head train + mask, Dockerfile
  treelearn/                # TreeLearn wrapper, Dockerfile
  segmentanytree/           # SAT output renderer (SAT itself is the official image)
  geowalker/                # through-cloud distance field + the walker on it
  stemnet/                  # dense chunk model: per point, tree + offset to
                            #   its base. No walk; its own image
  tests/                    # float32-fix, GeoWalker and StemNet CPU tests
  data/                     # runtime volume (git-ignored artifacts)
  results/                  # output PNGs
```
