# GeoWalker stage: a walker trained on a THROUGH-CLOUD distance field.
#
# Builds ON TOP of the basewalker image, which already carries torch/spconv/
# torch-scatter/sonata (from carbonpoly/sonata) plus scipy, open3d, pyproj, CSF
# and the shared scene prep — build `sonata` then `basewalker_prep` first (the
# run scripts' gw targets do). Nothing new to install: the field is scipy's
# csgraph Dijkstra and the decoder is plain torch.
FROM carbonpoly/basewalker

# Kept in their own directory, NOT in /app: the base image already has a
# train.py / infer.py (BaseWalker's) and they must not be shadowed. Python puts
# a script's own directory on sys.path, so `import model` still resolves here
# while /app keeps serving `walker`, `common` and `sonata_lib`.
COPY geowalker/geodesic.py /app/geowalker/geodesic.py
COPY geowalker/model.py /app/geowalker/model.py
COPY geowalker/train.py /app/geowalker/train.py
COPY geowalker/infer.py /app/geowalker/infer.py
COPY geowalker/gpu_check.py /app/geowalker/gpu_check.py
# CPU regression tests, so a freshly pulled machine can verify the field maths
# and the loss gradients (seconds, no GPU) before committing hours to training.
COPY tests/test_geowalker.py /app/tests/test_geowalker.py

WORKDIR /app
CMD ["bash", "-lc", "python /app/prep_scene.py && python /app/geowalker/geodesic.py && python /app/geowalker/train.py && python /app/geowalker/infer.py"]
