#!/usr/bin/env python3
"""Push result files to a Kaggle dataset, so they can be downloaded anywhere
instead of copied off the box by hand.

  files     V2_KAGGLE_FILES, comma-separated names or globs in results/
            (default the val LAZ and its table / log: stemnet_v2_val.laz,
            stemnet_v2_val_bases.csv, stemnet_v2_val_laz.txt). Every entry must
            match something, or nothing is pushed.
  dataset   V2_KAGGLE_DATASET, <owner>/<slug> (default
            <KAGGLE_USERNAME>/carbon-polygon-results), PRIVATE unless
            V2_KAGGLE_PUBLIC=1. Created on the first push; every later push is a
            new version holding exactly the files of that push (Kaggle keeps the
            old versions; V2_KAGGLE_DELETE_OLD=1 deletes them).
  message   V2_KAGGLE_MESSAGE (default: the date and the file names)

Credentials: KAGGLE_USERNAME / KAGGLE_KEY in .env (the same ones the
downloads use; KAGGLE_API_TOKEN works too, with V2_KAGGLE_DATASET naming the
owner); the dataset goes to that account. Refused credentials stop the push
before anything is uploaded.

Download afterwards (any machine with the same credentials):
  kaggle datasets download <owner>/<slug> -f stemnet_v2_val.laz -p <dir>
Output: results/stemnet_v2_kaggle.txt
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import RESULTS  # noqa: E402

FILES = [f.strip() for f in os.environ.get(
    "V2_KAGGLE_FILES", "stemnet_v2_val.laz,stemnet_v2_val_bases.csv,stemnet_v2_val_laz.txt").split(",") if f.strip()]
PUBLIC = os.environ.get("V2_KAGGLE_PUBLIC", "0") == "1"
DELETE_OLD = os.environ.get("V2_KAGGLE_DELETE_OLD", "0") == "1"
LOG = RESULTS / "stemnet_v2_kaggle.txt"


def log(msg):
    line = f"[v2-kaggle] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def kaggle(*args, check=True):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"[v2-kaggle] `kaggle {' '.join(args[:2])}` failed (exit {r.returncode}) — see above")
    return r


def dataset_slug():
    slug = os.environ.get("V2_KAGGLE_DATASET", "").strip()
    if slug:
        return slug
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not user or user.startswith("your_"):
        raise SystemExit("[v2-kaggle] no Kaggle account: set KAGGLE_USERNAME / KAGGLE_KEY in .env "
                         "(or V2_KAGGLE_DATASET=<owner>/<slug>)")
    return f"{user}/carbon-polygon-results"


def pick_files():
    picked, missing = [], []
    for pat in FILES:
        hits = sorted(p for p in RESULTS.glob(pat) if p.is_file() and not p.name.startswith("."))
        (picked.extend(hits) if hits else missing.append(pat))
    if missing:
        raise SystemExit(f"[v2-kaggle] nothing in {RESULTS} matches: {', '.join(missing)} — nothing pushed")
    return list(dict.fromkeys(picked))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    slug = dataset_slug()
    files = pick_files()
    size = sum(p.stat().st_size for p in files)
    log(f"dataset {slug} ({'public' if PUBLIC else 'private'}) | {len(files)} files, {size / 2**20:,.1f} MB: "
        + ", ".join(p.name for p in files))

    stage = RESULTS / ".kaggle_upload"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    try:
        for p in files:
            try:
                os.link(p, stage / p.name)                  # same disk: no copy of a 1 GB LAZ
            except OSError:
                shutil.copy2(p, stage / p.name)
        (stage / "dataset-metadata.json").write_text(json.dumps(dict(
            id=slug, title=slug.split("/", 1)[1].replace("-", " ")[:50].ljust(6, "_"),
            licenses=[{"name": "CC0-1.0"}]), indent=1))

        st = kaggle("datasets", "status", slug, check=False)
        said = (st.stdout + st.stderr).lower()
        if st.returncode != 0 and any(w in said for w in ("authentication required", "401", "403", "unauthorized")):
            print(st.stdout[-1500:]); print(st.stderr[-1500:])
            raise SystemExit("[v2-kaggle] Kaggle refused the credentials — check KAGGLE_USERNAME / KAGGLE_KEY "
                             "(or KAGGLE_API_TOKEN) in .env; nothing pushed")
        exists = st.returncode == 0
        msg = os.environ.get("V2_KAGGLE_MESSAGE") or f"{time.strftime('%Y-%m-%d %H:%M')}: " + ", ".join(p.name for p in files)
        t0 = time.time()
        if exists:
            log(f"{slug} exists — uploading a new version")
            kaggle("datasets", "version", "-p", str(stage), "-m", msg, "-t", "-r", "skip", *(["-d"] if DELETE_OLD else []))
        else:
            log(f"{slug} does not exist yet — creating it")
            kaggle("datasets", "create", "-p", str(stage), "-t", "-r", "skip", *(["-u"] if PUBLIC else []))
        log(f"uploaded in {(time.time() - t0) / 60:.1f} min (Kaggle may take a few more minutes to process it)")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    log(f"-> https://www.kaggle.com/datasets/{slug}")
    for p in files:
        log(f"   kaggle datasets download {slug} -f {p.name} -p <folder>")


if __name__ == "__main__":
    main()
