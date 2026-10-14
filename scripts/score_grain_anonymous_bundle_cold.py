#!/usr/bin/env python3
"""Cold CLI: authenticated anonymous bundle + JSONL records (cached encoder vectors) -> JSONL."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import anonymous_bundle as AB

ap = argparse.ArgumentParser()
for name in ("bundle", "input", "output", "sha256"):
    ap.add_argument(f"--{name}", required=True)
a = ap.parse_args()
b = AB.load_bundle(a.bundle, a.sha256)
recs = [json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
with Path(a.output).open("x") as fh:
    fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in AB.cold_score(b, recs)))
