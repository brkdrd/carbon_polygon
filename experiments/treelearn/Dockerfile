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

COPY common /app/common
COPY treelearn/run_treelearn.py /app/run_treelearn.py

WORKDIR /app
CMD ["python", "/app/run_treelearn.py"]
