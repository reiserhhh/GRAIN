#!/usr/bin/env python3
"""Independent arithmetic replay of frozen response-accounting results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from grain_research.response_accounting import PARTS, direct_difference


def run(path):
    """Verify frozen sources, actual cell means, and held aggregate metrics."""
    summary = json.loads((path / "summary.json").read_text())
    contract = json.loads((path / "contract.json").read_text())
    for name, checksum in summary["files"].items():
        if digest(path / name) != checksum:
            raise ValueError("Changed result file: " + name)
    for name, key in (
        ("scripts/run_grain_response_accounting.py", "runner_sha256"),
        ("grain_research/response_accounting.py", "module_sha256"),
        ("docs/GRAIN_SEMANTIC_RESPONSE_ACCOUNTING_PLAN_V1.md", "plan_sha256"),
    ):
        if digest(ROOT / name) != contract[key]:
            raise ValueError("Changed frozen source: " + name)
    with np.load(path / "private_panels.npz", allow_pickle=False) as z:
        panels, conditions, assignment, folds = (
            z["values"],
            z["conditions"],
            z["assignments"],
            z["folds"],
        )
    with np.load(path / "private_accounting.npz", allow_pickle=False) as z:
        result, valid, common, repeated, strict = (
            z["results"],
            z["valid"],
            z["common"],
            z["repeated"],
            z["strict"],
        )
    numeric_error = 0.0
    replayed = 0
    # Recompute all partitions directly from cell means, without stored operator coefficients.
    for r in range(len(assignment)):
        for a in np.flatnonzero(valid[r]):
            for m in range(len(contract["readers"])):
                expected = direct_difference(
                    conditions[a], assignment[r, a], panels[a, m]
                )
                numeric_error = max(
                    numeric_error, float(np.max(abs(expected - result[r, a, m])))
                )
                replayed += 1
    metric_error = 0.0
    metrics = pd.read_csv(path / "reader_agreement_by_facet.csv")
    for row in metrics.itertuples():
        part = PARTS.index(row.part)
        reader = contract["readers"].index(row.reader)
        q = contract["target_names"].index(row.function + ":" + row.facet)
        mask = (folds == 0) & strict & valid[0]
        if part in (1, 2):
            mask &= common[0]
        if row.support == "repeated_cell_author":
            mask &= repeated[0]
        t, p = result[0, mask, 0, part, q], result[0, mask, reader, part, q]
        if len(t) != row.authors:
            raise ValueError("Reader metric population differs")
        if len(t):
            metric_error = max(
                metric_error, abs(float(np.mean((t - p) ** 2)) - row.mse)
            )
        if len(t) >= 3 and np.std(t) > 1e-12 and np.std(p) > 1e-12:
            metric_error = max(
                metric_error, abs(float(np.corrcoef(t, p)[0, 1]) - row.reader_teacher_r)
            )
    energy_error = 0.0
    energy = pd.read_csv(path / "energy_accounting.csv")
    for row in energy.itertuples():
        total = row.within_squared + row.composition_squared + row.turnover_squared
        total += (
            row.within_x_composition
            + row.within_x_turnover
            + row.composition_x_turnover
        )
        energy_error = max(energy_error, abs(total - row.total_energy))
        share = (
            row.within_signed_projection
            + row.composition_signed_projection
            + row.turnover_signed_projection
        )
        energy_error = max(energy_error, abs(share - 1))
    if max(numeric_error, metric_error, energy_error) > 1e-10:
        raise ValueError("Independent arithmetic check failed")
    audit = {
        "status": "PASS",
        "author_partition_reader_units_replayed": replayed,
        "cell_replay_max_error": numeric_error,
        "metric_max_error": metric_error,
        "energy_identity_max_error": energy_error,
        "scope": "independent arithmetic implementation, not a second investigator or psychological validation",
        "summary_sha256": digest(path / "summary.json"),
        "audit_script_sha256": digest(Path(__file__)),
    }
    (path / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/all_plan_closure_v1/response_accounting_r1",
    )
    run(parser.parse_args().output)
