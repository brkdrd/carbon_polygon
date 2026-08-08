# Sonata stage (frozen PTv3 + linear vegetation head). GPU required.
#
# torch 2.7 + CUDA 12.8: the oldest stack that runs on Blackwell (RTX 50xx,
# sm_120) — torch cu124/cu126 builds ship kernels only up to sm_90 and die with
# "no kernel image is available". The two compiled deps Sonata needs are
# torch-scatter (sonata/model.py grid pooling; PyG publishes pt27cu128 wheels)
# and spconv (every block's CPE). No spconv-cu128 wheel exists; spconv-cu126
# covers sm_120 via its NVRTC runtime-compile fallback for archs missing from
# the prebuilt set — which is why JIT must NOT be disabled here. Our backbone
# runs inference-only (@torch.no_grad in sonata_lib), so the known Blackwell
# spconv-backward issue doesn't apply.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libgomp1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# One layer per package group so a slow/flaky link resumes from the last
# successful layer and any failure names the exact package.
#
# torch-scatter: PyG keys wheels by torch MAJOR.MINOR.0, so torch 2.7.1 uses the
# torch-2.7.0+cu128 index (matches this base's CUDA 12.8). Verified to carry
# cp310/311/312 wheels. spconv-cu126 is the newest spconv build (no cu128
# exists); it runs on CUDA 12.8 and JIT-compiles for sm_120. flash-attn is
# optional and we run with enable_flash=False (as the notebook did), so it is
# NOT needed.
RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir spconv-cu126
# Resume stalled wheel downloads instead of failing (data.pyg.org in particular
# stalls mid-transfer). ENV sits here, not in the top block, so the already-
# built layers above keep their cache.
ENV PIP_RESUME_RETRIES=10
RUN pip install --no-cache-dir torch-scatter \
      -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
RUN pip install --no-cache-dir -U "gdown>=6.1"
RUN pip install --no-cache-dir huggingface_hub timm addict "laspy[lazrs]" plyfile
RUN pip install --no-cache-dir open3d
RUN pip install --no-cache-dir matplotlib tqdm pandas py7zr
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/sonata.git"
# scipy: cKDTree for the mask dilation in run_mask.py (own layer to keep the
# layers above cached)
RUN pip install --no-cache-dir scipy

COPY common /app/common
COPY sonata/sonata_lib.py /app/sonata_lib.py
COPY sonata/train_head.py /app/train_head.py
COPY sonata/run_mask.py /app/run_mask.py

# default: train head (if needed) then produce the mask
CMD ["bash", "-lc", "python /app/train_head.py && python /app/run_mask.py"]
