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
#   ./run_experiments.sh cluster    # Sonata embeddings + k-means on one random
#                                   # square, coloured .laz in results/sonata_cluster
#                                   # (SC_SEED=… SC_SQUARE=… SC_K=… to steer it)
#   ./run_experiments.sh cluster_tile # bare Sonata + k-means on the Sonata+SEGR
#                                   # notebooks' 100 m tile (bounding-box centre)
#   ./run_experiments.sh cluster_geom # same square again, Sonata + the paper's
#                                   # 12 covariance features concatenated
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
gpu_check() { log "GPU preflight — does the card reach a container?"; $RUN gpu_check; }
gw_train() { log "GeoWalker — training";  $RUN geowalker_train; }
gw_infer() { log "GeoWalker — inference"; $RUN geowalker_infer; }
# Same walk, endpoints ranked by the gw_head classifier instead of the walker's
# own read-out; the run prints both rankings side by side. Needs gw_head first.
gw_infer_head() { log "GeoWalker — inference, ranked by the tree head"
                  $RUN -e GW_INFER_HEAD=1 geowalker_infer; }
# The post-walk evidence stage: best step of each walk, converging walkers
# counted as votes, canopy-height gate. Needs the CHM, so gw_height runs first.
gw_infer_agg() { log "GeoWalker — inference + post-walk evidence stage"
                 $RUN -e GW_AGGREGATE=1 geowalker_infer; }
# The attributed survey patch: ~2 ha, 392 trees, species and girth on 388 —
# the only region that yields a per-tree answer sheet. TRAINING ground; the
# stage prints that itself rather than leaving it to the reader.
gw_infer_patch() { log "GeoWalker — inference on the attributed patch"
                   $RUN -e GW_INFER_REGION=patch -e GW_AGGREGATE=1 geowalker_infer; }
# The control nobody had run: classical detection on the same ground, scored by
# the same matcher. No GPU, no training, minutes — and it is what says whether
# the network is earning its hours. Needs the CHM, so gw_height runs first.
gw_baseline() { log "GeoWalker — classical baseline on the attributed patch"
                $RUN geowalker_baseline; }
# ---- StemNet: a dense chunk model, no walk --------------------------------
# Per point: tree-or-not, and the horizontal offset to its base; then the
# shifted points are clustered. One encoder pass per window. Its own image and
# its own track -- it imports nothing from the walkers.
stem_build() { log "Building stemnet image"; $DC build sonata
               $DC build basewalker_prep; $DC build stemnet_build; }
stem_test()  { log "StemNet — CPU self-check (no GPU, no data)"
               $RUN stemnet_test; }
stem_train() { bw_prep; log "StemNet — train the dense stem head"
               $RUN stemnet_train; }
# 4,000 iterations is ~1.3 passes over the 3,028 train window centres and takes
# ~13 min on a 5090: a smoke test, not a trained model. This is the real run —
# ~13 epochs, cold, with the eval wide enough that best-F1 selection means
# something.
stem_long()  { bw_prep; log "StemNet — the long run (~2 h)"
               $RUN -e SN_ITERS=40000 -e SN_EVAL=1000 -e SN_EVAL_WIN=64 \
                    -e SN_RESUME=0 stemnet_train; }
# The overnight run. 150k iterations at ~190 ms/window is ~8 h; with the jitter
# redrawn per iteration these are 150k distinct crops, not 50 passes over a
# fixed 3,028.
stem_full()  { bw_prep; log "StemNet — the full run (~8 h)"
               $RUN -e SN_ITERS=150000 -e SN_EVAL=2500 -e SN_EVAL_WIN=64 \
                    -e SN_RESUME=0 stemnet_train; }
# Settle SN_NEAR_R before spending 8 h on it: at 1.0 m a third of all band
# points are labelled stem, so the head learns "near a tree" rather than
# "trunk". Two tagged short runs, identical seed, one variable. ~55 min.
stem_near()  { bw_prep
               log "StemNet — SN_NEAR_R=1.0 (tag _r10)"
               $RUN -e SN_ITERS=8000 -e SN_EVAL=1000 -e SN_NEAR_R=1.0 \
                    -e SN_TAG=_r10 -e SN_RESUME=0 stemnet_train
               log "StemNet — SN_NEAR_R=0.5 (tag _r05)"
               $RUN -e SN_ITERS=8000 -e SN_EVAL=1000 -e SN_NEAR_R=0.5 \
                    -e SN_TAG=_r05 -e SN_RESUME=0 stemnet_train; }
# Both scores in one go: the honest val number, then the per-tree answer sheet
# with recall by DBH band. The second is a ceiling (training labels) but the
# SIZE BREAKDOWN is the diagnostic — it says whether a recall plateau is the
# model or the target definition.
stem_eval()  { stem_infer; stem_patch; }
stem_infer() { log "StemNet — inference on the held-out val block"
               $RUN stemnet_infer; }
stem_patch() { log "StemNet — inference on the attributed patch"
               $RUN -e SN_REGION=patch stemnet_infer; }
# ---- StemNet v2: rebuilt from scratch ------------------------------------
v2_build()   { log "Building stemnet_v2 image"; $DC build v2_measure; }
v2_test()    { log "StemNet v2 — synthetic scan, known answers (CPU)"; $RUN v2_test; }
v2_measure() { log "StemNet v2 — measure the raw scan and the stem map (CPU)"; $RUN v2_measure; }
v2_grid()    { log "StemNet v2 — prep part 1: Sonata grid, colour, shared frame (CPU)"; $RUN v2_grid; }
v2_features() { log "StemNet v2 — prep part 2: 12 covariance features, 20 NN (CPU)"; $RUN v2_features; }
v2_bases()   { log "StemNet v2 — prep part 3: bases = survey + shift, audited (CPU)"; $RUN v2_bases; }
v2_labels()  { log "StemNet v2 — prep part 5: per-point labels (CPU)"; $RUN v2_labels; }
v2_export()  { log "StemNet v2 — export scan columns around 80 survey points (CPU)"; $RUN v2_export; }
v2_splits()  { log "StemNet v2 — prep part 4: train/val/test regions (CPU)"; $RUN v2_splits; }
v2_semantic() { log "StemNet v2 — prep part 5b: tree / non-tree labels from campus_pred_B24 (CPU; downloads it from Kaggle if data/raw has none; skips if current)"; $RUN v2_semantic; }
v2_cache()   { log "StemNet v2 — prep part 6: sorted training cache (CPU, ~10 min, skips if current)"; $RUN v2_cache; }
v2_train_build() { log "Building stemnet_v2 GPU image (on the sonata image)"
                   $DC build sonata; $DC build v2_train_build; }
v2_train_test() { log "StemNet v2 — training self-check, stand-in encoder (CPU, ~30 s)"; $RUN v2_train_test; }
v2_train()   { log "StemNet v2 — train to 50,000 steps, val every 5,000 (GPU)"; $RUN v2_train; }
v2_view()    { log "StemNet v2 — look at the points in squares across the campus (CPU)"; $RUN v2_view; }
v2_train_tree() { log "StemNet v2 — train gate/offset + tree/non-tree, val every 5,000 (GPU)"; $RUN v2_train_tree; }
v2_validate() { log "StemNet v2 — validate: tune detection on val, score the test patch once (GPU)"; $RUN v2_validate; }
v2_infer()   { log "StemNet v2 — inference over the campus: LAZ with tree / gate probabilities + bases (GPU)"; $RUN v2_infer; }
v2_inspect() { log "StemNet v2 — inspect the inference: errors by height, surveyed trunks, close-ups (CPU)"; $RUN v2_inspect; }
v2_curves() { log "StemNet v2 — detection curves: precision-recall / AP and FROC on val and test (CPU)"; $RUN v2_curves; }
v2_segment() { log "StemNet v2 — probe: frozen Sonata +/- features, k-means, no labels (GPU)"; $RUN v2_segment; }
sc_build() { log "Building sonata_cluster image (on the sonata image)"
             $DC build sonata; $DC build sc_build; }
sc_test()  { log "Sonata cluster — self-check, stand-in encoder (CPU, ~5 s)"; $RUN sc_test; }
sc_run()   { log "Sonata cluster — random square, embeddings, k-means (GPU)"; $RUN sc_run; }
sc_tile()  { log "Sonata cluster — the Sonata+SEGR 100 m tile, embeddings, k-means (GPU)"; $RUN sc_tile; }
sc_geom()  { log "Sonata cluster — same square, Sonata + 12 geometric features (GPU)"; $RUN sc_geom; }
gw_diag()  { log "GeoWalker — recall diagnostics"; $RUN geowalker_diag; }
gw_height() { log "GeoWalker — canopy height vs the label criterion"; $RUN geowalker_height; }
gw_head()  { log "GeoWalker — can the tree/not-tree head be trained?"; $RUN geowalker_head; }
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
  gpu_check) gw_build; gpu_check ;;
  gw_field) gw_build; gw_field ;;
  gw_train) gw_build; gw_train ;;
  gw_infer) gw_build; gw_infer ;;
  gw_infer_head) gw_build; gw_infer_head ;;
  gw_infer_agg) gw_build; gw_height; gw_infer_agg ;;
  gw_infer_patch) gw_build; gw_height; gw_infer_patch ;;
  gw_baseline) gw_build; gw_height; gw_baseline ;;
  gw_diag) gw_build; gw_diag ;;
  gw_height) gw_build; gw_height ;;
  gw_head) gw_build; gw_head ;;
  stem_build) stem_build ;;
  stem_test) stem_build; stem_test ;;
  stem_train) stem_build; stem_train ;;
  stem_long) stem_build; stem_long ;;
  stem_full) stem_build; stem_full ;;
  stem_near) stem_build; stem_near ;;
  stem_infer) stem_build; stem_infer ;;
  stem_patch) stem_build; stem_patch ;;
  stem_eval) stem_build; stem_eval ;;
  v2_test) v2_build; v2_test ;;
  v2_measure) v2_build; v2_test; v2_measure ;;
  v2_grid) v2_build; v2_test; v2_grid ;;
  v2_features) v2_build; v2_test; v2_features ;;
  v2_bases) v2_build; v2_test; v2_bases ;;
  v2_splits) v2_build; v2_test; v2_splits ;;
  v2_labels) v2_build; v2_test; v2_labels ;;
  v2_export) v2_build; v2_export ;;
  v2_prep) v2_build; v2_test; v2_grid; v2_features; v2_bases; v2_splits; v2_labels; v2_cache ;;
  v2_train_test) v2_train_build; v2_train_test ;;
  v2_cache) v2_build; v2_cache ;;
  v2_train) v2_build; v2_cache; v2_train_build; v2_train_test; v2_train ;;
  v2_view) v2_build; v2_view ;;
  v2_semantic) v2_build; v2_semantic ;;
  v2_train_tree) v2_build; v2_test; v2_semantic; v2_cache; v2_train_build; v2_train_test; v2_train_tree ;;
  v2_segment) v2_train_build; v2_train_test; v2_segment ;;
  v2_validate) v2_train_build; v2_train_test; v2_validate ;;
  v2_infer) v2_train_build; v2_train_test; v2_infer ;;
  v2_eval) v2_train_build; v2_train_test; v2_validate; v2_infer ;;
  v2_inspect) v2_build; v2_inspect ;;
  v2_curves) v2_train_build; v2_train_test; v2_validate; v2_build; v2_curves ;;
  sc_test) sc_build; sc_test ;;
  cluster) sc_build; sc_test; sc_run ;;
  cluster_geom) sc_build; sc_test; sc_geom ;;
  cluster_tile) sc_build; sc_test; sc_tile ;;
  stemnet) stem_build; stem_test; stem_train; stem_infer; stem_patch ;;
  *) echo "usage: $0 [all|build|prep|sonata|exp1|exp2|exp3|exp4|\
basewalker|bw_prep|bw_train|bw_infer|bw_render|\
geowalker|gw_test|gpu_check|gw_field|gw_train|gw_infer|gw_infer_head|gw_infer_agg|gw_infer_patch|gw_baseline|\
gw_diag|\
gw_height|gw_head|\
stemnet|stem_build|stem_test|stem_train|stem_long|stem_full|stem_near|stem_infer|stem_patch|stem_eval|v2_test|v2_measure|v2_grid|v2_features|v2_bases|v2_splits|v2_labels|v2_prep|v2_export|v2_cache|v2_train_test|v2_train|v2_segment|v2_view|v2_semantic|v2_train_tree|v2_validate|v2_infer|v2_eval|v2_inspect|v2_curves|cluster|cluster_tile|cluster_geom|sc_test]"; exit 2 ;;
esac
