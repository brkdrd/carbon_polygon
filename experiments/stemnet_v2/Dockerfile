# StemNet v2 — its own image, built from nothing in the older tracks.
# The measure stage is CPU-only; later v2 stages add what they need here.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN pip install --no-cache-dir numpy scipy matplotlib "laspy[lazrs]" pyproj
# kaggle: v2_semantic downloads campus_pred_B24.laz when data/raw has none
RUN pip install --no-cache-dir "kaggle>=1.6"

# the survey is data, not code: the stem map and the girth-attributed patch
COPY basewalker/poderevka.csv /app/poderevka.csv
COPY basewalker/poderevka_attributes.csv /app/poderevka_attributes.csv
COPY stemnet_v2 /app/stemnet_v2
COPY tests/test_stemnet_v2_measure.py tests/test_stemnet_v2_grid.py tests/test_stemnet_v2_features.py tests/test_stemnet_v2_bases.py tests/test_stemnet_v2_splits.py tests/test_stemnet_v2_export.py tests/test_stemnet_v2_labels.py tests/test_stemnet_v2_semantic.py /app/tests/

WORKDIR /app
CMD ["python", "/app/stemnet_v2/measure.py"]
