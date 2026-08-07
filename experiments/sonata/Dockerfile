# Sonata stage (frozen PTv3 + linear vegetation head). GPU required.
#
# Built from Sonata's OWN environment.yml recipe: torch 2.5.x + CUDA 12.4, for
# which the two compiled deps Sonata needs — torch-scatter (used by
# sonata/model.py's grid pooling) and spconv — have guaranteed prebuilt wheels.
# (Earlier this used the Kaggle image for notebook fidelity, but that image is a
# ~35 GB moving target that rolled forward to Python 3.12 + a torch too new for
# any torch-scatter wheel. A pinned base is smaller AND more reproducible.)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    CUMM_DISABLE_JIT=1 \
    SPCONV_DISABLE_JIT=1 \
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
# torch-scatter: PyG keys wheels by torch MAJOR.MINOR.0, so torch 2.5.1 uses the
# torch-2.5.0+cu124 index (matches this base's CUDA 12.4). Verified to carry
# cp310/311/312 wheels. spconv-cu124 matches CUDA 12.4. flash-attn is optional
# and we run with enable_flash=False (as the notebook did), so it is NOT needed.
RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir spconv-cu124
RUN pip install --no-cache-dir torch-scatter \
      -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
RUN pip install --no-cache-dir -U "gdown>=6.1"
RUN pip install --no-cache-dir huggingface_hub timm addict "laspy[lazrs]" plyfile
RUN pip install --no-cache-dir open3d
RUN pip install --no-cache-dir matplotlib tqdm pandas py7zr
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/sonata.git"

COPY common /app/common
COPY sonata/sonata_lib.py /app/sonata_lib.py
COPY sonata/train_head.py /app/train_head.py
COPY sonata/run_mask.py /app/run_mask.py

# default: train head (if needed) then produce the mask
CMD ["bash", "-lc", "python /app/train_head.py && python /app/run_mask.py"]
