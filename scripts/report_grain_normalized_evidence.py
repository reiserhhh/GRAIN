#!/usr/bin/env python3
"""Publish the six audited NEXT70 aggregates, never author rows or model weights."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_normalized_evidence import OUT, old, prior


def publish():
    """Verify frozen outputs and copy only the declared aggregate schema."""
    m = prior.manifest(OUT)
    audit = old.read(OUT / "audit.json")
    summary = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Primary audit required")
    old.verify_hashes(OUT / "analysis", summary["files"])
    dst = ROOT / "reports/grain_vector_mechanisms/normalized_evidence_v1"
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
    if len(hashes) != 6:
        raise ValueError("Six declared aggregate tables required")
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
