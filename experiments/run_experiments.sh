#!/usr/bin/env bash
# One command to run the whole four-experiment tree-segmentation study.
#
#   ./run_experiments.sh            # build everything + run all four experiments
#   ./run_experiments.sh build      # build/pull images only
#   ./run_experiments.sh prep       # just build the shared chunk
#   ./run_experiments.sh exp1|exp2|exp3|exp4   # a single experiment (deps must exist)
#
# Requires: Docker + Compose, an NVIDIA GPU with the Container Toolkit, and
# Kaggle credentials in .env (KAGGLE_USERNAME / KAGGLE_KEY).
set -euo pipefail
cd "$(dirname "$0")"

DC="docker compose -f docker-compose.yml"
# --no-deps: this script sequences every stage itself, so a `run` must not
# re-trigger the compose depends_on chain.
RUN="$DC run --rm --no-deps"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

build() {
  log "Building images (prep, sonata, treelearn) and pulling SegmentAnyTree"
  $DC build prep sonata treelearn_raw
  $DC pull segmentanytree_raw || true   # official image; pull is best-effort
}

prep()   { log "Stage 0 — shared chunk";              $RUN prep; }
sonata() { log "Stage 1 — Sonata vegetation mask";    $RUN sonata; }

exp1() { log "Experiment 1 — TreeLearn on raw chunk";              $RUN treelearn_raw; }
exp3() { log "Experiment 3 — TreeLearn on Sonata mask";            $RUN treelearn_masked; }

# SegmentAnyTree segments every file in its input folder, so stage exactly one.
sat_run() { # $1 = raw|masked   $2 = source laz basename
  local kind="$1" src="$2"
  local in="data/sat/${kind}_in" out="data/sat/${kind}_out"
  rm -rf "$in" "$out"; mkdir -p "$in" "$out"
  if [[ ! -f "data/chunk/${src}" ]]; then
    echo "!! data/chunk/${src} missing — run prep/sonata first"; exit 1
  fi
  cp "data/chunk/${src}" "$in/"
  log "SegmentAnyTree (${kind}) — segmenting ${src}"
  $RUN "segmentanytree_${kind}"
}

exp2() { sat_run raw    chunk_local.laz;        log "Rendering exp2"; $RUN sat_render_raw; }
exp4() { sat_run masked chunk_masked_local.laz; log "Rendering exp4"; $RUN sat_render_masked; }

all() {
  build
  prep            # shared chunk (exp1/exp2 input)
  sonata          # vegetation mask (exp3/exp4 input)
  exp1            # TreeLearn / raw
  exp3            # TreeLearn / masked
  exp2            # SegmentAnyTree / raw
  exp4            # SegmentAnyTree / masked
  log "Done. Result images in ./results:"
  ls -1 results/*.png 2>/dev/null || true
}

case "${1:-all}" in
  all) all ;;
  build) build ;;
  prep) prep ;;
  sonata) sonata ;;
  exp1) exp1 ;; exp2) exp2 ;; exp3) exp3 ;; exp4) exp4 ;;
  *) echo "usage: $0 [all|build|prep|sonata|exp1|exp2|exp3|exp4]"; exit 2 ;;
esac
