# TreeLearn stage. GPU required. The repo ships no Dockerfile, so this mirrors
# setup/setup.sh: CUDA 11.8 + PyTorch + spconv-cu118 (the load-bearing native
# dep). Do NOT bump to CUDA 12.x without switching the spconv variant — a
# CUDA/spconv mismatch silently breaks inference (the repo's Colab is broken for
# exactly this reason).
#
# NOTE on the tag: setup.sh installs torch 2.0.0 with pytorch-cuda=11.8 via
# CONDA, but Docker Hub has no such image — pytorch/pytorch:2.0.0 only ships
# cuda11.7. 2.1.0-cuda11.8-cudnn8-devel is the lowest published torch that keeps
# CUDA 11.8 (so spconv-cu118 stays valid); 2.1 is API-compatible with TreeLearn.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    TL_REPO=/opt/TreeLearn

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libgomp1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Clone at a pinned point for reproducibility (override TL_REF at build time).
ARG TL_REF=main
RUN git clone https://github.com/ecker-lab/TreeLearn.git /opt/TreeLearn \
    && cd /opt/TreeLearn && git checkout ${TL_REF}

WORKDIR /opt/TreeLearn
# requirements.txt pins spconv-cu118, numpy==1.24.4, open3d, laspy[lazrs], etc.
RUN pip install --no-cache-dir -r setup/requirements.txt \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir matplotlib pyyaml

COPY common /app/common
COPY treelearn/run_treelearn.py /app/run_treelearn.py

WORKDIR /app
CMD ["python", "/app/run_treelearn.py"]
