#!/usr/bin/env python3
"""Describe exact within/between-fold OOF contributions without changing predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.external_heads import TRAITS
from grain_research.vector_mechanisms.external_oof_geometry import (
    auc_partition,
    pearson_partition,
)
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof


def main():
    """Write diagnostic components; never return a post-hoc corrected winning score."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    rows, commitments = [], {}
    for folder in a.runs:
        result = load_oof(folder)
        m = result["manifest"]
        commitments[str(folder)] = hashlib.sha256(
            (folder / "private_oof_predictions.csv").read_bytes()
        ).hexdigest()
        common = {
            "run": folder.name,
            "task": m["task"],
            "feature": m["feature"],
            "arm": m["arm"],
            "mode": m.get("thread_control", "retain"),
            "seed": m.get("drop_seed", 731),
            "exclude_reference": m.get("exclude_reference", False),
            "authors": len(result["ids"]),
            "source_identity": m["identity"],
        }
        if m["task"] == "big5":
            for j, trait in enumerate(TRAITS):
                rows.append(
                    common
                    | {"trait": trait}
                    | pearson_partition(
                        result["y"][:, j], result["p"][:, j], result["folds"]
                    )
                )
        else:
            rows.append(
                common
                | {"trait": m["task"]}
                | auc_partition(result["y"], result["p"], result["folds"])
            )
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new diagnostic output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    pd.DataFrame(rows).to_csv(out / "oof_score_partition.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "runs": len(a.runs),
        "trait_rows": len(rows),
        "maximum_reconstruction_error": max(r["reconstruction_error"] for r in rows),
        "prediction_files": commitments,
        "model_refits": 0,
        "test_recalibration": False,
        "bridge_labels_read": False,
        "new_label_values_read": False,
        "analysis_status": "post-hoc descriptive decomposition, not a revised performance criterion",
    }
    write_json(out / "summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "prediction_files"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
