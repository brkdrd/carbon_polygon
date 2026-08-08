# TreeLearn stage. GPU required. The repo ships no Dockerfile; its setup.sh
# recipe is CUDA 11.8 + torch 2.0 + spconv-cu118, but that stack cannot run on
# Blackwell (RTX 50xx, sm_120) — torch cu118 kernels stop at sm_90 and die with
# "no kernel image is available". So this uses torch 2.7 + CUDA 12.8 (the
# oldest Blackwell-capable stack) and swaps the CUDA-keyed pins to match: the
# repo's spconv-cu118 → spconv-cu126 (newest spconv build; JIT-compiles for
# sm_120 via NVRTC), and open3d 0.17.0 → 0.18.0 (this base is Python 3.11;
# 0.18.0 is the first open3d with a cp311 wheel). We only run TreeLearn
# inference from a pretrained checkpoint, so the known Blackwell
# spconv-backward issue doesn't apply. torch 2.1→2.7 is API-compatible for
# TreeLearn's usage.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    TL_REPO=/opt/TreeLearn \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libgomp1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Clone at a pinned point for reproducibility (override TL_REF at build time).
ARG TL_REF=main
RUN git clone https://github.com/ecker-lab/TreeLearn.git /opt/TreeLearn \
    && cd /opt/TreeLearn && git checkout ${TL_REF}

WORKDIR /opt/TreeLearn
# Resume stalled wheel downloads instead of failing (placed after the clone
# layer to keep its cache).
ENV PIP_RESUME_RETRIES=10

# requirements.txt pins spconv-cu118 and open3d==0.17.0, both wrong for this
# base (CUDA 12.8 / Python 3.11) — rewrite those two pins, keep the rest.
RUN sed -e 's/^spconv-cu118.*/spconv-cu126/' \
        -e 's/^open3d==0.17.0/open3d==0.18.0/' \
        setup/requirements.txt > /tmp/requirements.cu128.txt \
    && pip install --no-cache-dir -r /tmp/requirements.cu128.txt \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir matplotlib pyyaml

# tree_learn/util/logger.py imports tensorboardX, but upstream installs it (and
# tensorboard) via conda in setup.sh, not requirements.txt — so mirroring only
# requirements.txt missed them. timm here OVERRIDES the requirements.txt pin
# (0.6.12): that release's dataclasses break on this base's Python 3.11, and
# TreeLearn only uses timm.scheduler.CosineLRScheduler, unchanged in 0.9.x.
# Own layer (not a sed above) so the big install layer stays cached.
RUN pip install --no-cache-dir tensorboardX tensorboard "timm==0.9.16"

# torch >= 2.6 defaults torch.load to weights_only=True, which can reject the
# pretrained checkpoint (written by torch 2.0). The checkpoint is trusted, so
# restore the old behavior at TreeLearn's single torch.load call site.
RUN grep -q "torch.load(checkpoint, map_location=" ${TL_REPO}/tree_learn/util/train.py \
    && sed -i "s/torch.load(checkpoint, map_location=/torch.load(checkpoint, weights_only=False, map_location=/" \
        ${TL_REPO}/tree_learn/util/train.py

# The pipeline crashes when NO point passes the grouping filters (tree
# confidence AND verticality AND |z-offset| — i.e. trunk-like points), which
# legitimately happens on masked/partial clouds: HDBSCAN refuses empty input,
# and the remaining-points KNN then has zero reference points to fit on.
# Guard both so such a run finishes and reports zero instances instead.
RUN grep -q "^def group_hdbscan(cluster_coords" ${TL_REPO}/tree_learn/util/pipeline.py \
    && grep -q "reference_idx = np.argwhere(predictions != remaining_points_idx)" ${TL_REPO}/tree_learn/util/pipeline.py \
    && grep -q "^def save_data(data" ${TL_REPO}/tree_learn/util/pipeline.py \
    && sed -i \
        -e '/^def save_data(data/a\    if len(data) == 0: return  # empty laz breaks the offset/color math below' \
        -e '/^def group_hdbscan(cluster_coords/a\    if len(cluster_coords) == 0: return np.zeros(0, dtype=np.int64)' \
        -e '/^def group_dbscan(cluster_coords/a\    if len(cluster_coords) == 0: return np.zeros(0, dtype=np.int64)' \
        -e '/reference_idx = np.argwhere(predictions != remaining_points_idx)/a\    if len(query_idx) == 0 or len(reference_idx) == 0: return predictions.astype(np.int64)' \
        ${TL_REPO}/tree_learn/util/pipeline.py

COPY common /app/common
COPY treelearn/run_treelearn.py /app/run_treelearn.py

WORKDIR /app
CMD ["python", "/app/run_treelearn.py"]
