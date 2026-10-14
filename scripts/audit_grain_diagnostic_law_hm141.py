#!/usr/bin/env python3
"""Independently enumerate HM141 author accounts and stipulated-law identities."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_diagnostic_law_hm141 as P


def audit(out):
    """Recompute complete finite estimands; do not run or fit a neural model."""
    cube, checks, status, manifest = P.observe(out)
    P.H.verify_hashes(out, P.H.load(out / "02/output_seal.json"))
    coefficients, accounts, actions, _summary, _effects, _parts = P.outputs(
        cube, manifest
    )
    normative = np.empty((2, 2))
    inverse_error = prior_error = 0.0
    for c in range(2):
        counts = np.array(manifest["spec"]["counts"]["positive" if c else "negative"])
        for b in range(2):
            option = 0 if b else 1
            odds = (
                (counts[0].sum() / counts[1].sum())
                * (counts[0, option] / counts[0].sum())
                / (counts[1, option] / counts[1].sum())
            )
            normative[b, c] = odds / (1 + odds)
            expected = np.log(1.5) + (2 * b - 1) * (2 * c - 1) * np.log(4)
            inverse_error = max(inverse_error, abs(float(np.log(odds) - expected)))
        action_mass = counts.sum(axis=0)[::-1] / counts.sum()
        prior_error = max(prior_error, abs(float(action_mass @ normative[:, c]) - 0.6))
    fields = {"score": 0, "logodds": 1, "digit_mass": 2, "entropy": 3}
    reconstruction_error = account_error = 0.0
    targets = manifest["spec"]["targets"]
    for row in coefficients:
        i, mode, unit, q = (
            targets.index(row["target"]),
            P.MODES.index(row["mode"]),
            fields[row["unit"]],
            P.QUERIES.index(row["query"]),
        )
        for b, c in itertools.product(range(2), repeat=2):
            signs = [1, 2 * b - 1, 2 * c - 1, (2 * b - 1) * (2 * c - 1)]
            prediction = sum(
                s * row[k]
                for s, k in zip(
                    signs,
                    ("constant", "behavior", "reference", "behavior_reference"),
                    strict=True,
                )
            )
            reconstruction_error = max(
                reconstruction_error,
                abs(float(prediction - cube[row["bank"], i, b, c, mode, q, unit])),
            )
    for row in accounts:
        i, c, mode, q, unit = (
            targets.index(row["target"]),
            (row["reference_sign"] + 1) // 2,
            P.MODES.index(row["mode"]),
            P.QUERIES.index(row["query"]),
            fields[row["unit"]],
        )
        values = cube[row["bank"], i, :, c, mode, q, unit]
        direct = np.array([0.25, 0.75]) @ values - np.array([0.75, 0.25]) @ values
        account_error = max(
            account_error, abs(float(direct - row["author_difference"]))
        )
    banks = []
    for bank in range(2):
        x = cube[bank, :, :, :, 0, :, 0]
        slopes = x[:, 1] - x[:, 0]
        interaction = slopes[:, 1] - slopes[:, 0]
        reference_effect = x[:, :, 1] - x[:, :, 0]
        banks.append(
            {
                "bank": bank,
                "per_target": [
                    {
                        "target": target,
                        "named_negative_positive_slopes": slopes[i, :, 0].tolist(),
                        "category_negative_positive_slopes": slopes[i, :, 1].tolist(),
                        "B_negative_positive_slopes": slopes[i, :, 2].tolist(),
                    }
                    for i, target in enumerate(targets)
                ],
                "mean_named_character_constant_behavior_reference_interaction": P.M.characters(
                    x[..., 0].transpose(1, 2, 0)
                )
                .mean(axis=1)
                .tolist(),
                "named_category_interaction_RMS": np.sqrt(
                    np.mean(interaction[:, :2] ** 2, axis=0)
                ).tolist(),
                "B_interaction_RMS": float(np.sqrt(np.mean(interaction[:, 2] ** 2))),
                "A_fact_reference_effect_max": float(
                    np.max(abs(reference_effect[..., 3]))
                ),
                "B_fact_reference_effect_max": float(
                    np.max(abs(reference_effect[..., 4]))
                ),
                "native_digit_mass_min": float(cube[bank, :, :, :, 0, :, 2].min()),
                "score_law_interventions": [
                    r for r in actions if r["bank"] == bank and r["unit"] == "score"
                ],
            }
        )
    error = max(inverse_error, prior_error, reconstruction_error, account_error)
    if error > 1e-12:
        raise ValueError("Independent finite-law/account arithmetic differs")
    receipt = P.H.load(out / "status_io_repair.json")
    P.H.verify_hashes(ROOT, receipt["repair_code"])
    result = {
        "status": "FINITE_HM141_NUMERICAL_AUDIT_NOT_BLIND_REPLICATION",
        "source": str(out.relative_to(ROOT)),
        "checks": checks,
        "author_values_replayed": len(accounts),
        "character_rows_reconstructed": len(coefficients),
        "inverse_logodds_error": inverse_error,
        "population_prior_recovery_error": prior_error,
        "author_account_error": account_error,
        "character_reconstruction_error": reconstruction_error,
        "normative": normative.tolist(),
        "banks": banks,
        "budget_charged_seconds": status["model_seconds"],
        "repair_conservative_seconds": receipt["conservative_failed_load_wall_seconds"],
        "successful_model_seconds": status["model_seconds"]
        - receipt["conservative_failed_load_wall_seconds"],
        "new_neural_calls": 0,
        "original_scope_complete": False,
        "audit_code": {str(Path(__file__).relative_to(ROOT)): P.H.sha(Path(__file__))},
    }
    P.H.write(out / "numerical_audit.json", result)
    print(result, flush=True)


def main():
    """Run one explicit read-only reconstruction with a new audit receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / P.H.load(P.SPEC)["out"])
    audit(parser.parse_args().out.resolve())


if __name__ == "__main__":
    main()
