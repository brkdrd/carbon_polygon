#!/usr/bin/env python3
"""Push result files to a Kaggle dataset, so they can be downloaded anywhere
instead of copied off the box by hand.

  files     V2_KAGGLE_FILES, comma-separated names or globs in results/
            (default only stemnet_v2_val.laz: the val square's scan points and
            the base posts in one file). Every entry must match something, or
            nothing is pushed.
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

Completeness, reported in the log:
  before    each file's size; a LAZ is opened and its point count read (a
            truncated or unreadable LAZ stops the push)
  during    the Kaggle CLI's own upload progress, live
  after     waits (up to V2_KAGGLE_WAIT_MIN, 30 min) until Kaggle says the dataset
            is ready, then lists the files Kaggle holds and compares every
            name and byte size with the local file: COMPLETE, or INCOMPLETE
            with what is missing / different (exit code 1)

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

FILES = [f.strip() for f in os.environ.get("V2_KAGGLE_FILES", "stemnet_v2_val.laz").split(",") if f.strip()]
WAIT_MIN = float(os.environ.get("V2_KAGGLE_WAIT_MIN", "30"))
POLL_S = float(os.environ.get("V2_KAGGLE_POLL_S", "20"))
PUBLIC = os.environ.get("V2_KAGGLE_PUBLIC", "0") == "1"
DELETE_OLD = os.environ.get("V2_KAGGLE_DELETE_OLD", "0") == "1"
LOG = RESULTS / "stemnet_v2_kaggle.txt"


def log(msg):
    line = f"[v2-kaggle] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def kaggle_live(*args):
    """Run the CLI with its output (upload progress) shown as it comes."""
    pr = subprocess.Popen(["kaggle", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in pr.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line])[-40:]
    if pr.wait() != 0:
        raise SystemExit(f"[v2-kaggle] `kaggle {' '.join(args[:2])}` failed (exit {pr.returncode}):\n" + "".join(tail[-10:]))


def check_local(files):
    """Size of every file; a LAZ must open and hold points."""
    for p in files:
        line = f"local {p.name}: {p.stat().st_size:,} bytes"
        if p.suffix.lower() in (".laz", ".las"):
            import laspy
            try:
                with laspy.open(str(p)) as f:
                    n = f.header.point_count
            except Exception as e:                       # noqa: BLE001
                raise SystemExit(f"[v2-kaggle] {p.name} cannot be read as LAS/LAZ ({e}) — nothing pushed")
            if not n:
                raise SystemExit(f"[v2-kaggle] {p.name} holds no points — nothing pushed")
            line += f", {n:,} points"
        log(line)


def wait_ready(slug):
    t0 = time.time()
    while True:
        st = kaggle("datasets", "status", slug, check=False)
        said = (st.stdout + st.stderr).strip().lower()
        if st.returncode == 0 and "ready" in said:
            log(f"Kaggle: dataset ready ({(time.time() - t0) / 60:.1f} min after the upload)")
            return True
        if "error" in said and st.returncode == 0:
            log(f"Kaggle reports an error processing the dataset: {said[-300:]}")
            return False
        if time.time() - t0 > WAIT_MIN * 60:
            log(f"Kaggle still not ready after {WAIT_MIN:g} min (status: {said[-200:]!r})")
            return False
        log(f"Kaggle: processing ({said[-60:] or 'no status yet'}), checking again in {POLL_S:g} s")
        time.sleep(POLL_S)


def remote_files(slug):
    """name -> size in bytes, as Kaggle lists them."""
    import csv
    import io
    r = kaggle("datasets", "files", slug, "-v", "--page-size", "200", check=False)
    out = {}
    if r.returncode != 0:
        return out
    for row in csv.DictReader(io.StringIO(r.stdout)):
        try:
            out[row["name"]] = int(float(row["size"]))
        except (KeyError, ValueError):
            pass
    return out


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
    check_local(files)
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
            kaggle_live("datasets", "version", "-p", str(stage), "-m", msg, "-t", "-r", "skip", *(["-d"] if DELETE_OLD else []))
        else:
            log(f"{slug} does not exist yet — creating it")
            kaggle_live("datasets", "create", "-p", str(stage), "-t", "-r", "skip", *(["-u"] if PUBLIC else []))
        log(f"uploaded in {(time.time() - t0) / 60:.1f} min")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    ready = wait_ready(slug)
    remote = remote_files(slug)
    bad = []
    for p in files:
        got = remote.get(p.name)
        if got is None:
            bad.append(f"{p.name} missing on Kaggle")
        elif got != p.stat().st_size:
            bad.append(f"{p.name} {got:,} bytes on Kaggle vs {p.stat().st_size:,} here")
        else:
            log(f"on Kaggle: {p.name} {got:,} bytes = local")
    log(f"-> https://www.kaggle.com/datasets/{slug}")
    for p in files:
        log(f"   kaggle datasets download {slug} -f {p.name} -p <folder>")
    if bad or not ready:
        log("INCOMPLETE: " + ("; ".join(bad) if bad else "Kaggle did not finish processing") +
            " — run v2_kaggle again (it uploads a fresh version)")
        sys.exit(1)
    log(f"COMPLETE: all {len(files)} file(s) on Kaggle, byte sizes match")


if __name__ == "__main__":
    main()
