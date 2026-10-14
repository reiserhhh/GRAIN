#!/usr/bin/env python3
"""Inventory existing D01 evidence without reusing criteria for model selection."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_dialogue_claim_ledger import BASE, sha, write

VM = ROOT / "results/grain_vector_mechanisms"
INPUTS = [
    (
        "trait_reference",
        "all_factor_inventory_m27_v1/external_query_criterion_matrix.csv",
        "aggregate association, not new dialogue-ledger validation",
    ),
    (
        "relational_criteria",
        "all_factor_inventory_m27_v1/private_meps_scores.csv",
        "validation-only pre/post criteria; no individual values exported",
    ),
    (
        "relational_scoring",
        "all_factor_inventory_m27_v1/meps_score_audit.csv",
        "existing item/key audit",
    ),
    (
        "mood_criteria",
        "counseling_mood_m30c_v1/private_mood_scores.csv",
        "validation-only adapted composites at recorded phases",
    ),
    (
        "mood_scoring",
        "counseling_mood_m30c_v1/mood_scoring_audit.csv",
        "existing adapted-item audit, not canonical scale norms",
    ),
    (
        "mood_association",
        "counseling_mood_m30c_v1/intended_associations.csv",
        "native model versus recorded questionnaire occasion",
    ),
    (
        "compiled_association",
        "all_plan_closure_v1/narrow_history_r2/transport/native_function_external_metrics.csv",
        "fixed narrow functions; older linked targets only",
    ),
    (
        "compiled_replay",
        "all_plan_closure_v1/narrow_history_r2/replay_comparisons.csv",
        "existing byte-replay comparisons, not another independent sample",
    ),
    (
        "compiled_equations",
        "all_plan_closure_v1/narrow_history_r2/query_evidence_ledger.json",
        "target/source/formula provenance",
    ),
]


def inventory(out):
    """Save only paths, hashes, shapes and declared permissible uses."""
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = []
    for key, relative, use in INPUTS:
        path = VM / relative
        row = {
            "id": key,
            "path": str(path.relative_to(ROOT)),
            "exists": path.is_file(),
            "use": use,
        }
        if path.is_file():
            row.update(sha256=sha(path), bytes=path.stat().st_size)
            if path.suffix == ".csv":
                with path.open(newline="") as stream:
                    reader = csv.reader(stream)
                    row["columns"] = next(reader)
                    row["rows"] = sum(1 for _ in reader)
        rows.append(row)
    write(out / "source_inventory.json", rows)
    summary = {
        "allowlisted_sources": len(rows),
        "present": sum(r["exists"] for r in rows),
        "private_rows_exported": 0,
        "new_neural_calls": 0,
        "new_fitting": False,
        "coefficient_recalculation": False,
        "missing_files": [r["path"] for r in rows if not r["exists"]],
        "closure": "D01 evidence inventory only; old criterion links do not validate new dialogue operations",
        "minimum_missing_evidence": [
            "independent role/event/reference coding for an eligible natural-dialogue sample",
            "criterion matched to the declared new observation target, actor and occasion",
            "independent construction remains D02, not this inventory",
        ],
    }
    write(out / "summary.json", summary)
    write(
        out / "manifest.json",
        {
            "script": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha(Path(__file__)),
            },
            "outputs": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        },
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "interpretation_inventory_r1"
    )
    inventory(parser.parse_args().output.resolve())
