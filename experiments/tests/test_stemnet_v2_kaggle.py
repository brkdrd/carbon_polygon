#!/usr/bin/env python3
"""StemNet v2 Kaggle push, against a stand-in `kaggle` on PATH: creates the
dataset the first time, versions it after, pushes exactly the chosen files,
and pushes nothing when a file is missing or the credentials are refused."""
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
if a[:2] == ["datasets", "status"]:
    if Path(os.environ["FAKE_KAGGLE_STATE"]).exists():
        print("ready"); sys.exit(0)
    print("404 Client Error: Not Found"); sys.exit(1)
if a[:2] in (["datasets", "create"], ["datasets", "version"]):
    d = Path(a[a.index("-p") + 1])
    files = sorted(p.name for p in d.iterdir())
    meta = json.loads((d / "dataset-metadata.json").read_text())
    sizes = {{p.name: p.stat().st_size for p in d.iterdir()}}
    with open(rec, "a") as f:
        f.write(json.dumps(dict(cmd=a[1], args=a, files=files, meta=meta, sizes=sizes)) + "\\n")
    Path(os.environ["FAKE_KAGGLE_STATE"]).write_text("x")
    sys.exit(0)
sys.exit(2)
"""


def _env(tmp, **extra):
    b = tmp / "bin"; b.mkdir(exist_ok=True)
    k = b / "kaggle"; k.write_text(FAKE.format(py=sys.executable)); k.chmod(0o755)
    env = dict(os.environ, DATA_DIR=str(tmp / "data"), RESULTS_DIR=str(tmp / "res"), KAGGLE_USERNAME="someone",
               KAGGLE_KEY="k", PATH=f"{b}{os.pathsep}{os.environ['PATH']}", FAKE_KAGGLE_LOG=str(tmp / "calls.jsonl"),
               FAKE_KAGGLE_STATE=str(tmp / "exists"))
    env.pop("V2_KAGGLE_DATASET", None)
    env.update(extra)
    return env


def _run(env):
    return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)


def _calls(tmp):
    q = tmp / "calls.jsonl"
    return [json.loads(l) for l in q.read_text().splitlines()] if q.exists() else []


def test_create_then_version_with_exactly_the_chosen_files():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); res = tmp / "res"; res.mkdir()
        (res / "stemnet_v2_val.laz").write_bytes(b"L" * 5000)
        (res / "stemnet_v2_val_bases.csv").write_text("a,b\n")
        (res / "stemnet_v2_val_laz.txt").write_text("log\n")
        (res / "other.png").write_bytes(b"p")
        r = _run(_env(tmp))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        c = _calls(tmp)
        assert len(c) == 1 and c[0]["cmd"] == "create" and "-u" not in c[0]["args"], c   # private by default
        assert c[0]["files"] == ["dataset-metadata.json", "stemnet_v2_val.laz", "stemnet_v2_val_bases.csv",
                                 "stemnet_v2_val_laz.txt"], c[0]["files"]
        assert c[0]["meta"]["id"] == "someone/carbon-polygon-results" and c[0]["sizes"]["stemnet_v2_val.laz"] == 5000
        assert not (res / ".kaggle_upload").exists() and (res / "stemnet_v2_val.laz").exists()
        r = _run(_env(tmp, V2_KAGGLE_FILES="*.png,stemnet_v2_val.laz", V2_KAGGLE_MESSAGE="second"))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        c = _calls(tmp)
        assert c[1]["cmd"] == "version" and "second" in c[1]["args"], c[1]
        assert c[1]["files"] == ["dataset-metadata.json", "other.png", "stemnet_v2_val.laz"], c[1]["files"]
        log = (res / "stemnet_v2_kaggle.txt").read_text()
        assert "kaggle datasets download someone/carbon-polygon-results -f stemnet_v2_val.laz" in log, log


def test_nothing_pushed_when_a_file_is_missing_or_credentials_are_refused():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); res = tmp / "res"; res.mkdir()
        (res / "stemnet_v2_val.laz").write_bytes(b"L")
        r = _run(_env(tmp))                                       # the csv and txt are missing
        assert r.returncode != 0 and "nothing pushed" in r.stdout + r.stderr, r.stdout + r.stderr
        r = _run(_env(tmp, V2_KAGGLE_FILES="stemnet_v2_val.laz", FAKE_KAGGLE_NOAUTH="1"))
        assert r.returncode != 0 and "refused the credentials" in r.stdout + r.stderr, r.stdout + r.stderr
        r = _run(_env(tmp, V2_KAGGLE_FILES="stemnet_v2_val.laz", KAGGLE_USERNAME="your_kaggle_username"))
        assert r.returncode != 0 and "no Kaggle account" in r.stdout + r.stderr, r.stdout + r.stderr
        assert _calls(tmp) == [] and not (res / ".kaggle_upload").exists()


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
