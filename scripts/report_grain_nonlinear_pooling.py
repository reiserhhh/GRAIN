#!/usr/bin/env python3
"""Publish only audited NEXT69 aggregate tables, with code and lineage hashes."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_nonlinear_pooling import OUT, old, prior


def publish():
    """Copy audited aggregates without source text, author rows or model weights."""
    m = prior.manifest(OUT)
    audit = old.read(OUT / "audit.json")
    summary = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Primary audit required")
    old.verify_hashes(OUT / "analysis", summary["files"])
    dst = ROOT / "reports/grain_vector_mechanisms/nonlinear_pooling_v1"
    dst.mkdir(exist_ok=True)
    hashes = {}
    for name, digest in summary["files"].items():
        if name.startswith("private_"):
            continue
        src = OUT / "analysis" / name
        if set(pd.read_csv(src, nrows=0).columns) & {"author_key", "text", "user_id"}:
            raise ValueError("Private field in aggregate")
        shutil.copyfile(src, dst / name)
        hashes[name] = digest
    if len(hashes) != 7:
        raise ValueError("Seven declared aggregate tables required")
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": audit,
            "files": hashes,
            "scope": "source4 matched views; exposed disjoint new8; native query outputs, not human criteria",
            "publisher_sha256": old.sha(Path(__file__)),
            "new_fits_or_neural_calls": 0,
        },
    )
    print({"status": "PUBLISHED", "tables": len(hashes)}, flush=True)


if __name__ == "__main__":
    publish()
