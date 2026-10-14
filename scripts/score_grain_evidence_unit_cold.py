#!/usr/bin/env python3
"""Cold CLI: authenticated bundle + JSONL author records -> JSONL scores (no source files)."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import evidence_unit as EU

ap = argparse.ArgumentParser()
for name in ("bundle", "input", "output", "sha256"):
    ap.add_argument(f"--{name}", required=True)
args = ap.parse_args()
bundle = EU.load_bundle(args.bundle, args.sha256)
records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
with open(args.output, "x") as fh:
    for row in EU.cold_score(bundle, records):
        fh.write(json.dumps(row, sort_keys=True) + "\n")
