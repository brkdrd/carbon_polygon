# Sonata stage (frozen PTv3 + linear vegetation head). GPU required.
#
# Based on the exact Kaggle image the reference notebook was validated on
# (sha from sonata-segr-gt/kernel-metadata.json), then replays the notebook's
# proven install sequence. This is the highest-fidelity way to reproduce a
# working Sonata/spconv environment without a GPU to test against locally.
# (digest copied verbatim from sonata-segr-gt/kernel-metadata.json "docker_image")
FROM gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9

ENV PYTHONUNBUFFERED=1 \
    CUMM_DISABLE_JIT=1 \
    SPCONV_DISABLE_JIT=1 \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

# Replays notebook install cell 4. torch-scatter is pulled from the PyG wheel
# index matching the base image's torch+cuda; falls back to source build.
RUN python -c "import torch,sys; \
    v=torch.__version__.split('+')[0]; \
    cu=(torch.version.cuda or '12.4').replace('.',''); \
    open('/tmp/tv','w').write(v+' '+cu); \
    print('torch',v,'cuda',cu,'py',sys.version.split()[0])"

RUN set -eux; read TORCH_V CU < /tmp/tv; \
    pip install -q spconv-cu126; \
    IDX="https://data.pyg.org/whl/torch-${TORCH_V}+cu${CU}.html"; \
    (pip install -q torch-scatter -f "$IDX" \
      || pip install -q --no-build-isolation torch-scatter); \
    pip install -q -U "gdown>=6.1"; \
    pip install -q huggingface_hub timm addict "laspy[lazrs]" plyfile; \
    pip install -q open3d "numpy<2"; \
    pip install -q matplotlib tqdm pandas py7zr; \
    pip install -q "git+https://github.com/facebookresearch/sonata.git"

# Some Kaggle images ship a stray setup.py/.gitignore in dist-packages that makes
# pccm treat spconv as editable and JIT-compile (notebook cell 2 guards this).
RUN python - <<'PY'
import os, sys
sp = f"/usr/local/lib/python3.{sys.version_info.minor}/dist-packages"
for f in ("setup.py", ".gitignore"):
    p = os.path.join(sp, f)
    if os.path.exists(p):
        os.rename(p, p + ".disabled")
        print("renamed stray", f)
PY

COPY common /app/common
COPY sonata/sonata_lib.py /app/sonata_lib.py
COPY sonata/train_head.py /app/train_head.py
COPY sonata/run_mask.py /app/run_mask.py

# default: train head (if needed) then produce the mask
CMD ["bash", "-lc", "python /app/train_head.py && python /app/run_mask.py"]
