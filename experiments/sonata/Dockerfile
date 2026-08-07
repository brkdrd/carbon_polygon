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

# Split into one layer per package group (was a single -q RUN). On a slow/flaky
# link this lets a rebuild resume from the last successful layer instead of
# re-downloading everything, and a failure names the exact package. Verbose on
# purpose so build logs are debuggable.
RUN pip install spconv-cu126
# torch-scatter: PyG publishes wheels keyed by torch MAJOR.MINOR.0 (NOT the exact
# patch), so a URL with e.g. 2.4.1 has no wheel and pip silently falls to a
# failing source build. Build the .0 key; --no-index makes each wheel attempt
# fail fast so we only source-build as a last resort.
RUN read TORCH_V CU < /tmp/tv; \
    MM="$(echo "$TORCH_V" | cut -d. -f1-2).0"; \
    echo "torch-scatter for torch ${TORCH_V} / cu${CU} (wheel key ${MM})"; \
    pip install --no-index torch-scatter -f "https://data.pyg.org/whl/torch-${MM}+cu${CU}.html" \
      || pip install --no-index torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_V}+cu${CU}.html" \
      || TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1 pip install --no-build-isolation torch-scatter
RUN pip install -U "gdown>=6.1"
RUN pip install huggingface_hub timm addict "laspy[lazrs]" plyfile
RUN pip install open3d "numpy<2"
RUN pip install matplotlib tqdm pandas py7zr
RUN pip install "git+https://github.com/facebookresearch/sonata.git"

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
