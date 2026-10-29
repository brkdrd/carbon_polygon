#Requires -Version 5.1
<#
  run_experiments.ps1 - Windows PowerShell launcher for the four-experiment
  tree-segmentation study. Mirrors run_experiments.sh exactly.

  Usage (from the experiments/ folder):
    .\run_experiments.ps1            # build + run the TreeLearn experiments (exp1, exp3)
    .\run_experiments.ps1 build      # build images only
    .\run_experiments.ps1 prep       # just build the shared chunk
    .\run_experiments.ps1 sonata     # Sonata vegetation mask
    .\run_experiments.ps1 exp1       # a single experiment (exp1..exp4;
                                     # exp2/exp4 need a non-Blackwell GPU)
    .\run_experiments.ps1 bw_render  # BaseWalker: 4 chunk images, detections
                                     # vs ground truth (needs a trained model)
    .\run_experiments.ps1 cluster    # Sonata embeddings + k-means on one random
                                     # square, coloured .laz in results\sonata_cluster
                                     # ($env:SC_SEED / SC_SQUARE / SC_K to steer it)
    .\run_experiments.ps1 cluster_tile # bare Sonata + k-means on the Sonata+SEGR
                                     # notebooks' 100 m tile (bounding-box centre)
    .\run_experiments.ps1 cluster_geom # same square again, Sonata + the paper's
                                     # 12 covariance features concatenated
    .\run_experiments.ps1 geowalker  # GeoWalker: through-cloud distance field,
                                     # then train + infer the walker on it
    .\run_experiments.ps1 gw_infer_patch # inference on the attributed ~2 ha
    .\run_experiments.ps1 gw_baseline    # classical control, CPU, minutes
    .\run_experiments.ps1 stemnet        # dense chunk model: train, val, patch
                                         # patch only: per-tree and
                                         # per-species results
    .\run_experiments.ps1 gw_infer_head  # inference, but endpoints ranked by
                                     # the gw_head classifier (both reported)

  If PowerShell blocks the script, launch it as:
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1

  Requires: Docker Desktop (WSL2 backend + GPU support) OR Docker Engine in WSL2
  with the NVIDIA Container Toolkit, and Kaggle credentials in .env
  (KAGGLE_USERNAME / KAGGLE_KEY).
#>
param([string]$Target = "all")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# compose declares `env_file: [.env]` on most services and errors out if it is
# missing - which is exactly the state of a fresh `git clone`. Seed it from the
# example so `git pull` then a run just works; the Kaggle credentials only
# matter for the very first prep on a machine with no cloud in data\raw\ yet.
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "!! no .env found - created one from .env.example." -ForegroundColor Yellow
    Write-Host "   Fill in KAGGLE_USERNAME / KAGGLE_KEY if data\raw\ has no .laz yet."
}

# docker-compose does the heavy lifting; this file only sequences the stages.
$Compose = @("compose", "-f", "docker-compose.yml")

function Log($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# Run docker and fail loudly on a non-zero exit (native commands don't throw
# on their own, even with $ErrorActionPreference = 'Stop').
function Invoke-Docker([string[]]$DockerArgs) {
    & docker @DockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($DockerArgs -join ' ') failed (exit $LASTEXITCODE)"
    }
}

# --no-deps: this script sequences every stage itself, so a `run` must not
# re-trigger the compose depends_on chain.
#
# $Vars is for the handful of stages that exist in more than one variant (the
# same image and command, one environment variable apart). compose has no
# "run with this variable" shorthand, so the -e flags are assembled here — the
# alternative is telling people to reach past the launcher for raw compose,
# which is how a documented one-command stage turns into a pasted incantation.
function Invoke-Stage($svc, [string[]]$Vars = @()) {
    $e = @()
    foreach ($kv in $Vars) { $e += @("-e", $kv) }
    Invoke-Docker ($Compose + @("run", "--rm", "--no-deps") + $e + @($svc))
}

function Build {
    Log "Building images (prep, sonata, treelearn)"
    Invoke-Docker ($Compose + @("build", "prep", "sonata", "treelearn_raw"))
}

function Prep   { Log "Stage 0 - shared chunk";           Invoke-Stage "prep" }
function Sonata { Log "Stage 1 - Sonata vegetation mask"; Invoke-Stage "sonata" }

function Exp1 { Log "Experiment 1 - TreeLearn on raw chunk";   Invoke-Stage "treelearn_raw" }
function Exp3 { Log "Experiment 3 - TreeLearn on Sonata mask"; Invoke-Stage "treelearn_masked" }

# SegmentAnyTree segments every file in its input folder, so stage exactly one.
function Invoke-Sat($kind, $src) {
    $inDir  = "data\sat\${kind}_in"
    $outDir = "data\sat\${kind}_out"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $inDir, $outDir
    New-Item -ItemType Directory -Force -Path $inDir, $outDir | Out-Null
    $chunk = "data\chunk\$src"
    if (-not (Test-Path $chunk)) {
        throw "$chunk missing - run prep/sonata first"
    }
    Copy-Item -Path $chunk -Destination $inDir -Force
    Log "SegmentAnyTree ($kind) - segmenting $src"
    Invoke-Stage "segmentanytree_$kind"
}

function Exp2 { Invoke-Sat "raw"    "chunk_local.laz";        Log "Rendering exp2"; Invoke-Stage "sat_render_raw" }
function Exp4 { Invoke-Sat "masked" "chunk_masked_local.laz"; Log "Rendering exp4"; Invoke-Stage "sat_render_masked" }

# BaseWalker (tree-base detection; separate study track)
function BwBuild { Log "Building basewalker image"
    Invoke-Docker ($Compose + @("build", "sonata"))
    Invoke-Docker ($Compose + @("build", "basewalker_prep")) }
function BwPrep  { Log "BaseWalker - scene prep"; Invoke-Stage "basewalker_prep" }
function BwTrain { Log "BaseWalker - training";   Invoke-Stage "basewalker_train" }
function BwInfer { Log "BaseWalker - inference";  Invoke-Stage "basewalker_infer" }
function BwRender { Log "BaseWalker - chunk renders vs ground truth"; Invoke-Stage "basewalker_render" }
function Basewalker { BwBuild; BwPrep; BwTrain; BwInfer; BwRender }

# GeoWalker (walks a through-cloud distance field; separate study track).
# Reuses BaseWalker's scene artifacts, so GwField runs BwPrep first (idempotent).
function GwBuild { Log "Building geowalker image"
    Invoke-Docker ($Compose + @("build", "sonata"))
    Invoke-Docker ($Compose + @("build", "basewalker_prep"))
    Invoke-Docker ($Compose + @("build", "geowalker_field")) }
function GwField { BwPrep; Log "GeoWalker - geodesic distance field"; Invoke-Stage "geowalker_field" }
function GwTest  { Log "GeoWalker - CPU self-check (no GPU, no data)"; Invoke-Stage "geowalker_test" }
function GpuCheck { Log "GPU preflight - does the card reach a container?"; Invoke-Stage "gpu_check" }
function GwTrain { Log "GeoWalker - training";  Invoke-Stage "geowalker_train" }
function GwInfer { Log "GeoWalker - inference"; Invoke-Stage "geowalker_infer" }
# Same walk, endpoints ranked by the gw_head classifier instead of the walker's
# own read-out; the run prints both rankings side by side. Needs gw_head first.
function GwInferHead { Log "GeoWalker - inference, ranked by the tree head"
    Invoke-Stage "geowalker_infer" @("GW_INFER_HEAD=1") }
# The post-walk evidence stage: best step of each walk, converging walkers
# counted as votes, canopy-height gate. Needs the CHM, so gw_height runs first
# (it reuses an existing chm.npz, so this is cheap on a second run).
function GwInferAgg { Log "GeoWalker - inference + post-walk evidence stage"
    Invoke-Stage "geowalker_infer" @("GW_AGGREGATE=1") }
# The attributed survey patch: ~2 ha, 392 trees, species and girth on 388 -
# the only region that yields a per-tree and per-species answer sheet, and
# small enough to finish in a couple of hours. It is TRAINING ground, which
# the stage prints itself rather than leaving to the reader.
function GwInferPatch { Log "GeoWalker - inference on the attributed patch"
    Invoke-Stage "geowalker_infer" @("GW_INFER_REGION=patch", "GW_AGGREGATE=1") }
# The control nobody had run: classical detection on the same ground, scored by
# the same matcher. No GPU, no training, minutes - and it is what says whether
# the network is earning its hours. Needs the CHM, so GwHeight runs first.
function GwBaseline { Log "GeoWalker - classical baseline on the attributed patch"
    Invoke-Stage "geowalker_baseline" }
# ---- StemNet: a dense chunk model, no walk --------------------------------
# Per point: tree-or-not, and the horizontal offset to its base; then the
# shifted points are clustered. One encoder pass per window. Its own image and
# its own track - it imports nothing from the walkers.
function StemBuild { Log "Building stemnet image"
    Invoke-Docker ($Compose + @("build", "sonata"))
    Invoke-Docker ($Compose + @("build", "basewalker_prep"))
    Invoke-Docker ($Compose + @("build", "stemnet_build")) }
function StemTest  { Log "StemNet - CPU self-check (no GPU, no data)"
    Invoke-Stage "stemnet_test" }
function StemTrain { BwPrep; Log "StemNet - train the dense stem head"
    Invoke-Stage "stemnet_train" }
# 4,000 iterations is ~1.3 passes over the 3,028 train window centres and takes
# ~13 min on a 5090: a smoke test, not a trained model. This is the real run -
# ~13 epochs, cold (the 12-min head is not worth warm-starting from), with the
# eval wide enough that best-F1 selection means something.
function StemLong  { BwPrep; Log "StemNet - the long run (~2 h)"
    Invoke-Stage "stemnet_train" @("SN_ITERS=40000", "SN_EVAL=1000",
                                   "SN_EVAL_WIN=64", "SN_RESUME=0") }
# The overnight run. 150k iterations at ~190 ms/window is ~8 h; with the
# jitter redrawn per iteration these are 150k distinct crops, not 50 passes
# over a fixed 3,028.
function StemFull  { BwPrep; Log "StemNet - the full run (~8 h)"
    Invoke-Stage "stemnet_train" @("SN_ITERS=150000", "SN_EVAL=2500",
                                   "SN_EVAL_WIN=64", "SN_RESUME=0") }
# Settle SN_NEAR_R before spending 8 h on it. At 1.0 m a third of all band
# points are labelled stem, so the head learns "near a tree" rather than
# "trunk"; 0.5 m is far more specific but leaves fewer voters. Two tagged
# short runs, identical seed, one variable. ~55 min for both.
function StemNear  { BwPrep
    Log "StemNet - SN_NEAR_R=1.0 (tag _r10)"
    Invoke-Stage "stemnet_train" @("SN_ITERS=8000", "SN_EVAL=1000",
                                   "SN_NEAR_R=1.0", "SN_TAG=_r10",
                                   "SN_RESUME=0")
    Log "StemNet - SN_NEAR_R=0.5 (tag _r05)"
    Invoke-Stage "stemnet_train" @("SN_ITERS=8000", "SN_EVAL=1000",
                                   "SN_NEAR_R=0.5", "SN_TAG=_r05",
                                   "SN_RESUME=0") }
# Both scores in one go: the honest val number, then the per-tree answer sheet
# with recall by DBH band. The second is a ceiling (training labels) but the
# SIZE BREAKDOWN is the diagnostic - it says whether a recall plateau is the
# model or the target definition.
function StemEval  { StemInfer; StemPatch }
function StemInfer { Log "StemNet - inference on the held-out val block"
    Invoke-Stage "stemnet_infer" }
function StemPatch { Log "StemNet - inference on the attributed patch"
    Invoke-Stage "stemnet_infer" @("SN_REGION=patch") }
# ---- StemNet v2: rebuilt from scratch ------------------------------------
function V2Build   { Log "Building stemnet_v2 image"
    Invoke-Docker ($Compose + @("build", "v2_measure")) }
function V2Test    { Log "StemNet v2 - synthetic scan, known answers (CPU)"
    Invoke-Stage "v2_test" }
function V2Measure { Log "StemNet v2 - measure the raw scan and the stem map (CPU)"
    Invoke-Stage "v2_measure" }
function V2Grid    { Log "StemNet v2 - prep part 1: Sonata grid, colour, shared frame (CPU)"
    Invoke-Stage "v2_grid" }
function V2Features { Log "StemNet v2 - prep part 2: 12 covariance features, 20 NN (CPU)"
    Invoke-Stage "v2_features" }
function V2Bases   { Log "StemNet v2 - prep part 3: bases = survey + shift, audited (CPU)"
    Invoke-Stage "v2_bases" }
function V2Labels  { Log "StemNet v2 - prep part 5: per-point labels (CPU)"
    Invoke-Stage "v2_labels" }
function V2Export  { Log "StemNet v2 - export scan columns around 80 survey points (CPU)"
    Invoke-Stage "v2_export" }
function V2Splits  { Log "StemNet v2 - prep part 4: train/val/test regions (CPU)"
    Invoke-Stage "v2_splits" }
function V2Semantic { Log "StemNet v2 - prep part 5b: tree / non-tree labels from campus_pred_B24 (CPU; downloads it from Kaggle if data/raw has none; skips if current)"
    Invoke-Stage "v2_semantic" }
function V2Cache   { Log "StemNet v2 - prep part 6: sorted training cache (CPU, ~10 min, skips if current)"
    Invoke-Stage "v2_cache" }
function V2TrainBuild { Log "Building stemnet_v2 GPU image (on the sonata image)"
    Invoke-Docker ($Compose + @("build", "sonata"))
    Invoke-Docker ($Compose + @("build", "v2_train_build")) }
function V2TrainTest { Log "StemNet v2 - training self-check, stand-in encoder (CPU, ~30 s)"
    Invoke-Stage "v2_train_test" }
function V2Train   { Log "StemNet v2 - train to 50,000 steps, val every 5,000 (GPU)"
    Invoke-Stage "v2_train" }
function V2View    { Log "StemNet v2 - look at the points in squares across the campus (CPU)"
    Invoke-Stage "v2_view" }
function V2TrainTree { Log "StemNet v2 - train gate/offset + tree/non-tree, val every 5,000 (GPU)"
    Invoke-Stage "v2_train_tree" }
function V2Validate { Log "StemNet v2 - validate: tune detection on val, score the test patch once (GPU)"
    Invoke-Stage "v2_validate" }
function V2Infer { Log "StemNet v2 - inference over the campus: LAZ with tree / gate probabilities + bases (GPU)"
    Invoke-Stage "v2_infer" }
function V2Inspect { Log "StemNet v2 - inspect the inference: errors by height, surveyed trunks, close-ups (CPU)"
    Invoke-Stage "v2_inspect" }
function V2Curves { Log "StemNet v2 - detection curves: precision-recall / AP and FROC on val and test (CPU)"
    Invoke-Stage "v2_curves" }
function V2Compare { Log "StemNet v2 - base-only vs base+tree against the survey, every threshold -> spreadsheet (GPU)"
    Invoke-Stage "v2_compare" }
function V2ValLaz { Log "StemNet v2 - val square as LAZ with both models' predictions + base posts (CPU)"
    Invoke-Stage "v2_val_laz" }
function V2Kaggle { Log "StemNet v2 - push result files (default the val LAZ) to a private Kaggle dataset (CPU)"
    Invoke-Stage "v2_kaggle" }
function V2TrainDetr { Log "StemNet v2 - train the DETR-style head (set prediction, Hungarian matching) on frozen Sonata + features (GPU)"
    Invoke-Stage "v2_train_detr" }
function V2Segment { Log "StemNet v2 - probe: frozen Sonata +/- features, k-means, no labels (GPU)"
    Invoke-Stage "v2_segment" }
function ScBuild { Log "Building sonata_cluster image (on the sonata image)"
    Invoke-Docker ($Compose + @("build", "sonata"))
    Invoke-Docker ($Compose + @("build", "sc_build")) }
function ScTest  { Log "Sonata cluster - self-check, stand-in encoder (CPU, ~5 s)"
    Invoke-Stage "sc_test" }
function ScRun   { Log "Sonata cluster - random square, embeddings, k-means (GPU)"
    Invoke-Stage "sc_run" }
function ScTile  { Log "Sonata cluster - the Sonata+SEGR 100 m tile, embeddings, k-means (GPU)"
    Invoke-Stage "sc_tile" }
function ScGeom  { Log "Sonata cluster - same square, Sonata + 12 geometric features (GPU)"
    Invoke-Stage "sc_geom" }
function GwDiag  { Log "GeoWalker - recall diagnostics"; Invoke-Stage "geowalker_diag" }
function GwHeight { Log "GeoWalker - canopy height vs the label criterion"; Invoke-Stage "geowalker_height" }
function GwHead  { Log "GeoWalker - can the tree/not-tree head be trained?"; Invoke-Stage "geowalker_head" }
function Geowalker { GwBuild; GwTest; GwField; GwTrain; GwInfer }

# exp2/exp4 (SegmentAnyTree) are OFF the default path: its official images are
# compiled for sm_60-86 and cannot run on Blackwell (RTX 50xx). The exp2/exp4
# targets still work when invoked explicitly (on compatible hardware).
function Invoke-All {
    Build
    Prep            # shared chunk (exp1 input)
    Sonata          # vegetation mask (exp3 input)
    Exp1            # TreeLearn / raw
    Exp3            # TreeLearn / masked
    Log "Done. Result images in .\results:"
    Get-ChildItem -Path "results\*.png" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Name
}

switch ($Target) {
    "all"    { Invoke-All }
    "build"  { Build }
    "prep"   { Prep }
    "sonata" { Sonata }
    "exp1"   { Exp1 }
    "exp2"   { Exp2 }
    "exp3"   { Exp3 }
    "exp4"   { Exp4 }
    "basewalker" { Basewalker }
    "bw_build" { BwBuild }
    "bw_prep"  { BwBuild; BwPrep }
    "bw_train" { BwBuild; BwTrain }
    "bw_infer" { BwBuild; BwInfer }
    "bw_render" { BwBuild; BwRender }
    "geowalker" { Geowalker }
    "gw_build" { GwBuild }
    "gw_test" { GwBuild; GwTest }
    "gpu_check" { GwBuild; GpuCheck }
    "gw_field" { GwBuild; GwField }
    "gw_train" { GwBuild; GwTrain }
    "gw_infer" { GwBuild; GwInfer }
    "gw_infer_head" { GwBuild; GwInferHead }
    "gw_infer_agg" { GwBuild; GwHeight; GwInferAgg }
    "gw_infer_patch" { GwBuild; GwHeight; GwInferPatch }
    "gw_baseline" { GwBuild; GwHeight; GwBaseline }
    "stem_build" { StemBuild }
    "stem_test" { StemBuild; StemTest }
    "stem_train" { StemBuild; StemTrain }
    "stem_long" { StemBuild; StemLong }
    "stem_full" { StemBuild; StemFull }
    "stem_near" { StemBuild; StemNear }
    "stem_infer" { StemBuild; StemInfer }
    "stem_patch" { StemBuild; StemPatch }
    "stem_eval" { StemBuild; StemEval }
    "v2_test" { V2Build; V2Test }
    "v2_measure" { V2Build; V2Test; V2Measure }
    "v2_grid" { V2Build; V2Test; V2Grid }
    "v2_features" { V2Build; V2Test; V2Features }
    "v2_bases" { V2Build; V2Test; V2Bases }
    "v2_splits" { V2Build; V2Test; V2Splits }
    "v2_labels" { V2Build; V2Test; V2Labels }
    "v2_export" { V2Build; V2Export }
    "v2_prep" { V2Build; V2Test; V2Grid; V2Features; V2Bases; V2Splits; V2Labels; V2Cache }
    "v2_train_test" { V2TrainBuild; V2TrainTest }
    "v2_cache" { V2Build; V2Cache }
    "v2_train" { V2Build; V2Cache; V2TrainBuild; V2TrainTest; V2Train }
    "v2_segment" { V2TrainBuild; V2TrainTest; V2Segment }
    "v2_validate" { V2TrainBuild; V2TrainTest; V2Validate }
    "v2_infer" { V2TrainBuild; V2TrainTest; V2Infer }
    "v2_eval" { V2TrainBuild; V2TrainTest; V2Validate; V2Infer }
    "v2_inspect" { V2Build; V2Inspect }
    "v2_curves" { V2TrainBuild; V2TrainTest; V2Validate; V2Build; V2Curves }
    "v2_compare" { V2TrainBuild; V2TrainTest; V2Compare }
    "v2_val_laz" { V2Build; V2ValLaz }
    "v2_kaggle" { V2Build; V2Kaggle }
    "v2_train_detr" { V2Build; V2Cache; V2TrainBuild; V2TrainTest; V2TrainDetr }
    "v2_val_kaggle" { V2Build; V2ValLaz; V2Kaggle }
    "sc_test" { ScBuild; ScTest }
    "cluster" { ScBuild; ScTest; ScRun }
    "cluster_geom" { ScBuild; ScTest; ScGeom }
    "cluster_tile" { ScBuild; ScTest; ScTile }
    "v2_view" { V2Build; V2View }
    "v2_semantic" { V2Build; V2Semantic }
    "v2_train_tree" { V2Build; V2Test; V2Semantic; V2Cache; V2TrainBuild; V2TrainTest; V2TrainTree }
    "stemnet" { StemBuild; StemTest; StemTrain; StemInfer; StemPatch }
    "gw_diag" { GwBuild; GwDiag }
    "gw_height" { GwBuild; GwHeight }
    "gw_head" { GwBuild; GwHead }
    default  {
        Write-Host "usage: .\run_experiments.ps1 [all|build|prep|sonata|exp1|exp2|exp3|exp4|basewalker|bw_prep|bw_train|bw_infer|bw_render|geowalker|gw_test|gpu_check|gw_field|gw_train|gw_infer|gw_infer_head|gw_infer_agg|gw_infer_patch|gw_baseline|gw_diag|gw_height|gw_head|stemnet|stem_build|stem_test|stem_train|stem_long|stem_full|stem_near|stem_infer|stem_patch|stem_eval|v2_test|v2_measure|v2_grid|v2_features|v2_bases|v2_splits|v2_labels|v2_prep|v2_export|v2_cache|v2_train_test|v2_train|v2_segment|v2_view|v2_semantic|v2_train_tree|v2_validate|v2_infer|v2_eval|v2_inspect|v2_curves|v2_compare|v2_val_laz|v2_kaggle|v2_val_kaggle|v2_train_detr|cluster|cluster_tile|cluster_geom|sc_test]"
        exit 2
    }
}
