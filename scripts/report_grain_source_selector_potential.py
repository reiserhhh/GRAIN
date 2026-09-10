#!/usr/bin/env python3
"""Publish the audited NEXT74 aggregates without token or author records."""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_selector_potential import OUT, TABLES, freeze, old


def publish():
    """Use exact canonical filenames, excluding filesystem metadata sidecars."""
    m = old.verify(OUT)
    audit = old.read(OUT / "audit.json")
    summary = freeze(OUT, "analysis")
    freeze(OUT, "models")
    if audit["status"] != "PASS" or set(summary["files"]) != set(TABLES):
        raise ValueError("Audited exact table inventory required")
    dst = ROOT / "reports/grain_vector_mechanisms/source_selector_potential_v1"
    dst.mkdir(exist_ok=True)
    for name in TABLES:
        src = OUT / "analysis" / name
        if set(pd.read_csv(src, nrows=0).columns) & {
            "author_key",
            "user_id",
            "text",
            "token",
            "head",
        }:
            raise ValueError("Private fields in public aggregate")
        shutil.copyfile(src, dst / name)
    training = pd.read_csv(OUT / "models/source_losses.csv")
    training.drop(columns="reference").groupby(
        ["source_authors", "query", "objective"], as_index=False
    ).mean().to_csv(dst / "source_training.csv", index=False)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": audit,
            "files": {n: old.sha(dst / n) for n in (*TABLES, "source_training.csv")},
            "publisher_sha256": old.sha(Path(__file__)),
            "scope": "Scalar selector from H16; native V/g retained; not questionnaire prediction",
        },
    )
    print({"status": "PUBLISHED", "tables": 5}, flush=True)


if __name__ == "__main__":
    publish()
