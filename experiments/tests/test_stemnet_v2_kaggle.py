#!/usr/bin/env python3
"""StemNet v2 Kaggle push, against a stand-in `kaggle` on PATH: creates the
dataset the first time, versions it after, pushes exactly the chosen files,
waits for Kaggle to finish and compares what it holds with the local files,
and pushes nothing when a file is missing / unreadable or the credentials are
refused."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "stemnet_v2" / "kaggle_push.py"

FAKE = """#!{py}
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
rec = Path(os.environ["FAKE_KAGGLE_LOG"])
if os.environ.get("FAKE_KAGGLE_NOAUTH"):
    print("Authentication required to call the Kaggle API."); sys.exit(1)
state = Path(os.environ["FAKE_KAGGLE_STATE"])
if a[:2] == ["datasets", "list"]:
    print("ref  title  size"); sys.exit(0)
if a[:2] == ["datasets", "status"]:
    if not state.exists():                                         # what the real API answers for a new slug
        print("403 Client Error: Forbidden for url: https://api.kaggle.com/v1/datasets.DatasetApiService/GetDatasetStatus")
        sys.exit(1)
    st = json.loads(state.read_text())
    st["polls"] += 1; state.write_text(json.dumps(st))
    print("pending" if st["polls"] == 2 else "ready"); sys.exit(0)        # one "pending" right after an upload
if a[:2] == ["datasets", "files"]:
    st = json.loads(state.read_text())
    print("name,size,creationDate")
    for n, z in st["sizes"].items():
        print(f"{{n}},{{z - (1 if os.environ.get('FAKE_KAGGLE_SHORT') else 0)}},2026-10-29 10:00:00")
    sys.exit(0)
if a[:2] in (["datasets", "create"], ["datasets", "version"]):
    d = Path(a[a.index("-p") + 1])
    files = sorted(p.name for p in d.iterdir())
    meta = json.loads((d / "dataset-metadata.json").read_text())
    sizes = {{p.name: p.stat().st_size for p in d.iterdir()}}
    with open(rec, "a") as f:
        f.write(json.dumps(dict(cmd=a[1], args=a, files=files, meta=meta, sizes=sizes)) + "\\n")
    import time
    for q in sorted(d.iterdir()):                                 # stream every file, slowly, like an upload
        if q.suffix == ".laz":
            with open(q, "rb", buffering=0) as fh:
                n = q.stat().st_size
                while fh.read(max(n // 6, 1)):
                    time.sleep(0.25)
    print("Uploading stemnet_v2_val.laz: 100%|#####| 15k/15k\\rUpload successful: stemnet_v2_val.laz")
    state.write_text(json.dumps(dict(polls=1, sizes={{k: v for k, v in sizes.items() if k != "dataset-metadata.json"}})))
    sys.exit(0)
sys.exit(2)
"""


def _env(tmp, **extra):
    b = tmp / "bin"; b.mkdir(exist_ok=True)
    k = b / "kaggle"; k.write_text(FAKE.format(py=sys.executable)); k.chmod(0o755)
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"), KAGGLE_USERNAME="someone",
               KAGGLE_KEY="k", PATH=f"{b}{os.pathsep}{os.environ['PATH']}", FAKE_KAGGLE_LOG=str(tmp / "calls.jsonl"),
               FAKE_KAGGLE_STATE=str(tmp / "state.json"), V2_KAGGLE_POLL_S="0", V2_KAGGLE_PROGRESS_S="0.05")
    env.pop("V2_KAGGLE_DATASET", None)
    env.update(extra)
    return env


def _run(env):
    return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)


def _laz(path, n=40000):
    import laspy
    import numpy as np
    las = laspy.LasData(laspy.LasHeader(point_format=7, version="1.4"))
    las.x, las.y, las.z = (np.random.default_rng(0).uniform(0, 10, n) for _ in range(3))
    las.write(str(path))


def _calls(tmp):
    q = tmp / "calls.jsonl"
    return [json.loads(l) for l in q.read_text().splitlines()] if q.exists() else []


def test_create_then_version_with_exactly_the_chosen_files():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); res = tmp / "res"; res.mkdir()
        _laz(res / "stemnet_v2_val.laz")
        size = (res / "stemnet_v2_val.laz").stat().st_size
        (res / "stemnet_v2_val_bases.csv").write_text("a,b\n")
        (res / "other.png").write_bytes(b"p")
        r = _run(_env(tmp))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        c = _calls(tmp)
        assert len(c) == 1 and c[0]["cmd"] == "create" and "-u" not in c[0]["args"], c   # private by default
        assert c[0]["files"] == ["dataset-metadata.json", "stemnet_v2_val.laz"], c[0]["files"]      # only the LAZ
        assert c[0]["meta"]["id"] == "someone/carbon-polygon-results" and c[0]["sizes"]["stemnet_v2_val.laz"] == size
        log = (res / "stemnet_v2_kaggle.txt").read_text()
        assert "40,000 points" in log and "kaggle: Upload successful" in r.stdout, log + r.stdout[-1000:]
        pcts = [float(l.split("uploading:")[1].split("%")[0]) for l in log.splitlines() if "uploading:" in l]
        assert any(0 < x < 100 for x in pcts) and pcts[-1] == 100.0, pcts               # a percentage while it runs
        assert "MB/s" in log and "min left" in log, log
        assert "processing (pending)" in log and "dataset ready" in log, log
        assert f"on Kaggle: stemnet_v2_val.laz {size:,} bytes = local" in log and "COMPLETE: all 1 file(s)" in log, log
        assert not (res / ".kaggle_upload").exists() and (res / "stemnet_v2_val.laz").exists()
        r = _run(_env(tmp, V2_KAGGLE_FILES="*.png,stemnet_v2_val.laz", V2_KAGGLE_MESSAGE="second"))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        c = _calls(tmp)
        assert c[1]["cmd"] == "version" and "second" in c[1]["args"], c[1]
        assert c[1]["files"] == ["dataset-metadata.json", "other.png", "stemnet_v2_val.laz"], c[1]["files"]
        log = (res / "stemnet_v2_kaggle.txt").read_text()
        assert "kaggle datasets download someone/carbon-polygon-results -f stemnet_v2_val.laz" in log, log
        r = _run(_env(tmp, FAKE_KAGGLE_SHORT="1"))                 # Kaggle holds a byte less than we sent
        assert r.returncode == 1 and "INCOMPLETE: stemnet_v2_val.laz" in r.stdout, r.stdout[-2000:]


def test_nothing_pushed_when_a_file_is_missing_or_credentials_are_refused():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); res = tmp / "res"; res.mkdir()
        r = _run(_env(tmp))                                       # no LAZ at all
        assert r.returncode != 0 and "nothing pushed" in r.stdout + r.stderr, r.stdout + r.stderr
        (res / "stemnet_v2_val.laz").write_bytes(b"L" * 100)     # a broken / truncated LAZ
        r = _run(_env(tmp))
        assert r.returncode != 0 and "cannot be read as LAS/LAZ" in r.stdout + r.stderr, r.stdout + r.stderr
        _laz(res / "stemnet_v2_val.laz")
        r = _run(_env(tmp, V2_KAGGLE_FILES="stemnet_v2_val.laz", FAKE_KAGGLE_NOAUTH="1"))
        assert r.returncode != 0 and "refused the credentials" in r.stdout + r.stderr, r.stdout + r.stderr
        r = _run(_env(tmp, V2_KAGGLE_FILES="stemnet_v2_val.laz", KAGGLE_USERNAME="your_kaggle_username"))
        assert r.returncode != 0 and "no Kaggle account" in r.stdout + r.stderr, r.stdout + r.stderr
        e = _env(tmp, V2_KAGGLE_FILES="stemnet_v2_val.laz"); e.pop("KAGGLE_KEY"); e.pop("KAGGLE_API_TOKEN", None)
        r = _run(e)                                               # username but no key: say so, by name
        assert r.returncode != 0 and "KAGGLE_KEY MISSING" in r.stdout + r.stderr and "KAGGLE_USERNAME set" in r.stdout + r.stderr, \
            r.stdout + r.stderr
        e["KAGGLE_API_TOKEN"] = "tok"                             # a token alone is enough
        r = _run(e)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert len(_calls(tmp)) == 1 and not (res / ".kaggle_upload").exists()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:                    # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'All' if not fails else fails} StemNet v2 kaggle test(s) {'passed' if not fails else 'FAILED'}.")
    sys.exit(1 if fails else 0)
