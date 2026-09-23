#!/usr/bin/env bash
# One command to run the whole four-experiment tree-segmentation study.
#
#   ./run_experiments.sh            # build + run the TreeLearn experiments (exp1, exp3)
#   ./run_experiments.sh build      # build images only
#   ./run_experiments.sh prep       # just build the shared chunk
#   ./run_experiments.sh exp1|exp2|exp3|exp4   # a single experiment (deps must
#                                   # exist; exp2/exp4 need a non-Blackwell GPU)
#   ./run_experiments.sh bw_render  # BaseWalker: 4 chunk images, detections
#                                   # vs ground truth (needs a trained model)
#   ./run_experiments.sh geowalker  # GeoWalker: through-cloud distance field,
#                                   # then train + infer the walker on it
#                                   # (gw_test | gw_field | gw_train | gw_infer
#                                   # singly; gw_test is a CPU-only self-check)
#
# Requires: Docker + Compose, an NVIDIA GPU with the Container Toolkit, and
# Kaggle credentials in .env (KAGGLE_USERNAME / KAGGLE_KEY).
set -euo pipefail
cd "$(dirname "$0")"

# compose declares `env_file: [.env]` on most services and errors out if it is
# missing — which is exactly the state of a fresh `git clone`. Seed it from the
# example so `git pull && ./run_experiments.sh <target>` just works; the Kaggle
# credentials only matter for the very first prep on a machine with no cloud in
# data/raw/ yet.
if [[ ! -f .env ]]; then
  cp .env.example .env
  printf '\033[1;33m!! no .env found — created one from .env.example.\033[0m\n'
  printf '   Fill in KAGGLE_USERNAME / KAGGLE_KEY if data/raw/ has no .laz yet.\n'
fi

DC="docker compose -f docker-compose.yml"
# --no-deps: this script sequences every stage itself, so a `run` must not
# re-trigger the compose depends_on chain.
RUN="$DC run --rm --no-deps"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

build() {
  log "Building images (prep, sonata, treelearn)"
  $DC build prep sonata treelearn_raw
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

# BaseWalker (tree-base detection; separate study track)
bw_build() { log "Building basewalker image"; $DC build sonata; $DC build basewalker_prep; }
bw_prep()  { log "BaseWalker — scene prep";   $RUN basewalker_prep; }
bw_train() { log "BaseWalker — training";     $RUN basewalker_train; }
bw_infer() { log "BaseWalker — inference";    $RUN basewalker_infer; }
bw_render() { log "BaseWalker — chunk renders vs ground truth"; $RUN basewalker_render; }
basewalker() { bw_build; bw_prep; bw_train; bw_infer; bw_render; }

# GeoWalker (walks a through-cloud distance field; separate study track).
# Reuses BaseWalker's scene artifacts, so gw_field runs bw_prep first (idempotent).
gw_build() { log "Building geowalker image"; $DC build sonata; $DC build basewalker_prep; $DC build geowalker_field; }
gw_field() { bw_prep; log "GeoWalker — geodesic distance field"; $RUN geowalker_field; }
gw_test()  { log "GeoWalker — CPU self-check (no GPU, no data)"; $RUN geowalker_test; }
gw_train() { log "GeoWalker — training";  $RUN geowalker_train; }
gw_infer() { log "GeoWalker — inference"; $RUN geowalker_infer; }
geowalker() { gw_build; gw_test; gw_field; gw_train; gw_infer; }

# exp2/exp4 (SegmentAnyTree) are OFF the default path: its official images are
# compiled for sm_60-86 and cannot run on Blackwell (RTX 50xx). The exp2/exp4
# targets still work when invoked explicitly (on compatible hardware).
all() {
  build
  prep            # shared chunk (exp1 input)
  sonata          # vegetation mask (exp3 input)
  exp1            # TreeLearn / raw
  exp3            # TreeLearn / masked
  log "Done. Result images in ./results:"
  ls -1 results/*.png 2>/dev/null || true
}

case "${1:-all}" in
  all) all ;;
  build) build ;;
  prep) prep ;;
  sonata) sonata ;;
  exp1) exp1 ;; exp2) exp2 ;; exp3) exp3 ;; exp4) exp4 ;;
  basewalker) basewalker ;;
  bw_build) bw_build ;;
  bw_prep) bw_build; bw_prep ;;
  bw_train) bw_build; bw_train ;;
  bw_infer) bw_build; bw_infer ;;
  bw_render) bw_build; bw_render ;;
  geowalker) geowalker ;;
  gw_build) gw_build ;;
  gw_test) gw_build; gw_test ;;
  gw_field) gw_build; gw_field ;;
  gw_train) gw_build; gw_train ;;
  gw_infer) gw_build; gw_infer ;;
  *) echo "usage: $0 [all|build|prep|sonata|exp1|exp2|exp3|exp4|\
basewalker|bw_prep|bw_train|bw_infer|bw_render|\
geowalker|gw_test|gw_field|gw_train|gw_infer]"; exit 2 ;;
esac
