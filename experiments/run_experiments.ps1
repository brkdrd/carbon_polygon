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
    .\run_experiments.ps1 geowalker  # GeoWalker: through-cloud distance field,
                                     # then train + infer the walker on it

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
function Invoke-Stage($svc) { Invoke-Docker ($Compose + @("run", "--rm", "--no-deps", $svc)) }

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
    default  {
        Write-Host "usage: .\run_experiments.ps1 [all|build|prep|sonata|exp1|exp2|exp3|exp4|basewalker|bw_prep|bw_train|bw_infer|bw_render|geowalker|gw_test|gpu_check|gw_field|gw_train|gw_infer]"
        exit 2
    }
}
