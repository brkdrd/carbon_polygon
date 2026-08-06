# Stage 0 — data prep. CPU-only: pulls the Kaggle LAZ, cuts the shared chunk,
# centers it in float64, writes the canonical chunk files. No GPU needed.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    KAGGLE_CONFIG_DIR=/root/.kaggle \
    DATA_DIR=/data \
    RESULTS_DIR=/results

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY prep/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# common/ and the prep script are bind-mounted by compose, but copy them too so
# the image runs standalone.
COPY common /app/common
COPY prep/prepare_chunk.py /app/prepare_chunk.py
# same CPU image also renders SegmentAnyTree's output (laspy + matplotlib here)
COPY segmentanytree/render_sat_output.py /app/render_sat_output.py

CMD ["python", "/app/prepare_chunk.py"]
