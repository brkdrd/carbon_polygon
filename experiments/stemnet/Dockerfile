# StemNet: a dense chunk model. Per point, is this a tree stem, and where is
# its base — one encoder pass per window, no walk.
#
# Builds ON TOP of the basewalker image, which already carries torch/spconv/
# torch-scatter/sonata (from carbonpoly/sonata) plus scipy and the shared scene
# prep — build `sonata` then `basewalker_prep` first (the run scripts' stem
# targets do). Nothing new to install: the head is four Linear layers and the
# clustering is scipy.
#
# It does NOT build on the geowalker image and imports nothing from it. The two
# share the prepared scene, the frozen encoder and the matcher, and all three of
# those live in /app/common.
FROM carbonpoly/basewalker

# Own directory, NOT /app: the base image already has train.py and infer.py
# (BaseWalker's) and these must not shadow them. Python puts a script's own
# directory on sys.path, so `import net` resolves here while /app keeps serving
# `common` and `sonata_lib`.
COPY stemnet/net.py /app/stemnet/net.py
COPY stemnet/train.py /app/stemnet/train.py
COPY stemnet/infer.py /app/stemnet/infer.py
# CPU regression tests: the band, the window tiling, the targets, the vote and
# the dense read-out, with Sonata mocked. Seconds, no GPU.
COPY tests/test_stemnet.py /app/tests/test_stemnet.py

WORKDIR /app
CMD ["bash", "-lc", "python /app/stemnet/train.py && python /app/stemnet/infer.py"]
