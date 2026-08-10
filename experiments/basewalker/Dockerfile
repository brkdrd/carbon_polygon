# BaseWalker stage: iterative tree-base detection (frozen Sonata encoder +
# trained walker decoder). Builds ON TOP of the sonata image so torch/spconv/
# torch-scatter/sonata are inherited — build `sonata` first (the run scripts do).
FROM carbonpoly/sonata

# pyproj: WGS84 -> UTM for the RTK tree-base labels.
# cloth-simulation-filter: CSF ground extraction (DEM); prep_scene falls back to
# a percentile-grid DEM if this import fails at runtime.
# kaggle: source-cloud download when prep hasn't run on this machine yet.
RUN pip install --no-cache-dir pyproj kaggle
RUN pip install --no-cache-dir cloth-simulation-filter || true

COPY common /app/common
COPY basewalker/poderevka.csv /app/poderevka.csv
COPY sonata/sonata_lib.py /app/sonata_lib.py
COPY basewalker/prep_scene.py /app/prep_scene.py
COPY basewalker/walker.py /app/walker.py
COPY basewalker/train.py /app/train.py
COPY basewalker/infer.py /app/infer.py

WORKDIR /app
CMD ["bash", "-lc", "python /app/prep_scene.py && python /app/train.py && python /app/infer.py"]
