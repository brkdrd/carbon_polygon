#Requires -Version 5.1
<#
  run_experiments.ps1 — Windows PowerShell launcher for the four-experiment
  tree-segmentation study. Mirrors run_experiments.sh exactly.

  Usage (from the experiments/ folder):
    .\run_experiments.ps1            # build everything + run all four experiments
    .\run_experiments.ps1 build      # build/pull images only
    .\run_experiments.ps1 prep       # just build the shared chunk
    .\run_experiments.ps1 sonata     # Sonata vegetation mask
    .\run_experiments.ps1 exp1       # a single experiment (exp1..exp4)

  If PowerShell blocks the script, launch it as:
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1

  Requires: Docker Desktop (WSL2 backend + GPU support) OR Docker Engine in WSL2
  with the NVIDIA Container Toolkit, and Kaggle credentials in .env
  (KAGGLE_USERNAME / KAGGLE_KEY).
#>
param([string]$Target = "all")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

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
    Log "Building images (prep, sonata, treelearn) and pulling SegmentAnyTree"
    Invoke-Docker ($Compose + @("build", "prep", "sonata", "treelearn_raw"))
    # official image; pull is best-effort (mirrors `|| true`)
    try { & docker @($Compose + @("pull", "segmentanytree_raw")) } catch { }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  (SegmentAnyTree pull skipped/failed — it will pull on run)" -ForegroundColor Yellow
    }
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

function Invoke-All {
    Build
    Prep            # shared chunk (exp1/exp2 input)
    Sonata          # vegetation mask (exp3/exp4 input)
    Exp1            # TreeLearn / raw
    Exp3            # TreeLearn / masked
    Exp2            # SegmentAnyTree / raw
    Exp4            # SegmentAnyTree / masked
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
    default  {
        Write-Host "usage: .\run_experiments.ps1 [all|build|prep|sonata|exp1|exp2|exp3|exp4]"
        exit 2
    }
}
