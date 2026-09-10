#!/usr/bin/env python3
"""Publish NEXT60 aggregate tables, retaining raw artifacts and zero-response flags."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_token_potential_extraction import OUT, read, sha, write


def publish():
    """Export aggregate rows only; undefined numerical-zero correlations stay blank."""
    audit = read(OUT / "replay_audit.json")
    summary = read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Audited frozen prediction/replay required")
    dst = ROOT / "reports/grain_vector_mechanisms/token_potential_extraction_v1"
    dst.mkdir(exist_ok=True)
    names = (
        "scores",
        "score_contrasts",
        "actions",
        "factor_actions",
        "action_contrasts",
        "factor_contrasts",
        "fields",
        "source_diagnostics",
    )
    for name in names:
        source = OUT / "analysis" / (name + ".csv")
        if sha(source) != summary["files"][source.name]:
            raise ValueError("Primary aggregate hash differs")
        frame = pd.read_csv(source)
        if "author_key" in frame:
            raise ValueError("Do not publish individual rows")
        if "allocation_predicted_action_rms" in frame:
            zero = frame.candidate.eq("mean_then_kernel")
            if not (frame.loc[zero, "allocation_predicted_action_rms"] < 1e-12).all():
                raise ValueError("Mean-first allocation is not numerical zero")
            frame["allocation_status"] = np.where(
                zero, "zero_by_construction", "measured"
            )
            for col in ("allocation_action_r", "allocation_same_sign"):
                frame.loc[zero, col] = np.nan
        frame.to_csv(dst / source.name, index=False)
    write(
        dst / "publication.json",
        {
            "source_primary_commit": read(OUT / "manifest.json")["commit"],
            "audit": audit,
            "promotion": summary["promotion"],
            "criteria": summary["criteria"],
            "transformation": "Mean-first zero-allocation r/sign undefined, blank in public tables; raw audited tables retained unchanged.",
            "files": {name + ".csv": sha(dst / (name + ".csv")) for name in names},
        },
    )
    print({"published_aggregate_tables": len(names), "directory": str(dst)})


if __name__ == "__main__":
    publish()
