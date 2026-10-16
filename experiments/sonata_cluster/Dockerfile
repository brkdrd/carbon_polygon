# Sonata embeddings + k-means on one random square of the campus scan.
#
# Builds ON carbonpoly/sonata, which already carries torch 2.7 + CUDA 12.8
# (Blackwell), spconv, torch-scatter, the sonata package, laspy[lazrs], scipy
# and matplotlib — build `sonata` first (the run scripts' sc targets do).
FROM carbonpoly/sonata

COPY sonata_cluster /app/sonata_cluster
# CPU self-check: synthetic scan, stand-in encoder. ~5 s, no GPU, no data.
COPY tests/test_sonata_cluster.py /app/tests/

WORKDIR /app
CMD ["python", "/app/sonata_cluster/cluster_square.py"]
