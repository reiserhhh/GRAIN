#!/usr/bin/env python3
"""Replay complete conduit records and independently reconstruct factorial coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def audit(parent: Path, previous: Path) -> tuple[dict, pd.DataFrame]:
    """Use an explicit polynomial design, not the collector's inclusion-exclusion code."""
    manifest = json.loads((parent / "manifest.json").read_text())
    status = json.loads((parent / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] > 1800:
        raise ValueError("Complete bounded P1C required")
    with zipfile.ZipFile(parent / "source_snapshot.zip") as z:
        for name, sha in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError("Source snapshot mismatch")
    for name, sha in json.loads((parent / "output_hashes.json").read_text()).items():
        if hashlib.sha256((parent / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Output mismatch")
    base = pd.read_csv(parent / "baselines.csv").set_index("case_id")
    effects = pd.read_csv(parent / "interventions.csv", dtype={"mask": str})
    interactions = pd.read_csv(parent / "factorial_interactions.csv")
    old = pd.read_csv(previous / "baselines.csv").set_index("case_id")
    if len(base) != 60 or len(effects) != 1260 or len(interactions) != 1260:
        raise ValueError("Incomplete grid")
    if effects.duplicated(["case_id", "site", "mask"]).any():
        raise ValueError("Duplicate patch")
    score_error = float(
        np.max(abs(effects.high_logit - effects.low_logit - effects.margin))
    )
    native_error = float(np.max(abs(base.margin - old.loc[base.index].margin)))
    if max(score_error, native_error) > 1e-5:
        raise ValueError("Native or candidate arithmetic mismatch")
    masks = list(product((0, 1), repeat=3))
    terms = ("E", "Q", "D", "EQ", "ED", "QD", "EQD")
    design = np.array(
        [[1, e, q, d, e * q, e * d, q * d, e * q * d] for e, q, d in masks], dtype=float
    )
    coefficient_error, recovery_error, decomposition_error = 0.0, 0.0, 0.0
    allocation = []
    for (case, site), sub in effects.groupby(["case_id", "site"]):
        values = {r.mask: r.margin for r in sub.itertuples()}
        values["000"] = float(base.loc[case, "margin"])
        if set(values) != {"".join(map(str, m)) for m in masks}:
            raise ValueError("Incomplete factorial case")
        y = np.array([values["".join(map(str, m))] for m in masks])
        coefficients = np.linalg.solve(design, y)[1:]
        recorded = (
            interactions[(interactions.case_id == case) & (interactions.site == site)]
            .set_index("term")
            .value
        )
        coefficient_error = max(
            coefficient_error,
            float(np.max(abs(coefficients - recorded.loc[list(terms)].to_numpy()))),
        )
        donor_case = sub.donor_case.iloc[0]
        gap = float(base.loc[donor_case, "margin"] - base.loc[case, "margin"])
        recovery_error = max(
            recovery_error, abs(values["111"] - base.loc[donor_case, "margin"])
        )
        # Shapley allocation splits each interaction equally across its participants.
        # This is one accounting convention, not identification of unique pathways.
        phi = {
            part: sum(
                v / len(term) for term, v in zip(terms, coefficients) if part in term
            )
            for part in "EQD"
        }
        decomposition_error = max(
            decomposition_error,
            abs(sum(phi.values()) - (values["111"] - values["000"])),
        )
        allocation.append(
            {
                "case_id": case,
                "site": site,
                "variant": sub.variant.iloc[0],
                "group_id": sub.group_id.iloc[0],
                "donor_gap": gap,
                **phi,
            }
        )
    if max(coefficient_error, recovery_error, decomposition_error) > 1e-5:
        raise ValueError("Factorial or full restoration failed")
    return {
        "status": "VERIFIED",
        "parent_identity": manifest["identity"],
        "calls": status["calls"],
        "score_arithmetic_max": score_error,
        "previous_native_replay_max": native_error,
        "factorial_coefficient_max": coefficient_error,
        "full_donor_restore_max": recovery_error,
        "allocation_identity_max": decomposition_error,
        "additional_neural_calls": 0,
    }, pd.DataFrame(allocation)


def main():
    """Save a new audit, retaining the production output unchanged."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--previous-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result, allocation = audit(a.parent_dir, a.previous_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    allocation.to_csv(a.output_dir / "interaction_allocation.csv", index=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
