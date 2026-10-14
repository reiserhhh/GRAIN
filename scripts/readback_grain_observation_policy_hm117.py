#!/usr/bin/env python3
"""Separate native observation-policy dependence from fixed-action transport."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observer_focus_hm117 as R


def readback(out):
    """Sum actual corners independently; report all families without new forwards."""
    m = R.check(out)
    verification = R.R.H.load(out / "verification.json")
    if verification["status"] != "VERIFIED_FINITE_HM117_NOT_GLOBAL":
        raise ValueError("Complete verified observations required")
    R.R.H.verify_hashes(out, R.R.H.load(out / "02/output_seal.json"))
    with np.load(out / "02/accounts.npz", allow_pickle=False) as stream:
        native, beta, effects = (stream[k] for k in ("raw", "native", "effects"))
    folder = out / "policy_readback"
    if folder.exists():
        raise FileExistsError("Preserve prior readback")
    rows, summaries = [], []
    max_error = 0.0
    cases = [("native_baseline", native[0], beta[0])]
    cases.extend(
        (arm, native[0] - native[a], effects[a])
        for a, arm in enumerate(R.R.M.ARMS[1:], start=1)
    )
    for name, cells, coefficients in cases:
        # Cells are focus/family/condition/polarity/owner/person/query/unit.
        direct = np.zeros((2, 6, 2, 2, 6, 4))
        for b, o in itertools.product(range(2), repeat=2):
            direct += (2 * b - 1) * (2 * o - 1) * cells[:, :, :, b, o] / 4
        common = (direct[:, :, :, 1] + direct[:, :, :, 0]) / 2
        person = (direct[:, :, :, 1] - direct[:, :, :, 0]) / 2
        for f, ci, ri, q, u in itertools.product(
            range(6), range(2), range(2), range(6), range(4)
        ):
            c, r = 2 * ci - 1, 2 * ri - 1
            expected = R.M.qualified_effect(coefficients[f], c, r)
            for field, value in (
                ("actor_common", common[ri, f, ci, q, u]),
                ("actor_difference", person[ri, f, ci, q, u]),
            ):
                max_error = max(max_error, abs(float(value - expected[field][q, u])))
            rows.append(
                {
                    "object": name,
                    "family": m["spec"]["targets"][f],
                    "condition": c,
                    "focus": "A" if r == 1 else "B",
                    "query": m["spec"]["targets"][q],
                    "unit": R.R.UNITS[u],
                    "person_contrast": float(person[ri, f, ci, q, u]),
                    "common_response": float(common[ri, f, ci, q, u]),
                }
            )
        for u, unit in enumerate(R.R.UNITS):
            for field, x in (("person", person[..., u]), ("common", common[..., u])):
                averaged = x.mean(axis=2)
                even, odd = (
                    (averaged[1] + averaged[0]) / 2,
                    (averaged[1] - averaged[0]) / 2,
                )
                diagonal = np.stack([np.diag(averaged[r]) for r in range(2)])
                summaries.append(
                    {
                        "object": name,
                        "unit": unit,
                        "field": field,
                        "focus_A_RMS": float(np.sqrt(np.mean(averaged[1] ** 2))),
                        "focus_B_RMS": float(np.sqrt(np.mean(averaged[0] ** 2))),
                        "focus_even_RMS": float(np.sqrt(np.mean(even**2))),
                        "focus_odd_RMS": float(np.sqrt(np.mean(odd**2))),
                        "A_B_cosine": float(
                            np.vdot(averaged[0], averaged[1])
                            / np.linalg.norm(averaged[0])
                            / np.linalg.norm(averaged[1])
                        ),
                        "A_diagonal_mean": float(diagonal[1].mean()),
                        "B_diagonal_mean": float(diagonal[0].mean()),
                        "A_positive_matched": int((diagonal[1] > 0).sum()),
                        "B_positive_matched": int((diagonal[0] > 0).sum()),
                        "matched_same_sign": int((diagonal[0] * diagonal[1] > 0).sum()),
                    }
                )
    if max_error > m["spec"]["arithmetic_tolerance"]:
        raise ValueError(f"Direct corner account mismatch: {max_error}")
    folder.mkdir()
    R.R.H.table(folder / "conditional_accounts.csv", rows)
    R.R.H.table(folder / "policy_summaries.csv", summaries)
    R.R.H.write(
        folder / "summary.json",
        {
            "status": "VERIFIED_NATIVE_AND_ACTION_POLICY_ACCOUNTS_NOT_GLOBAL",
            "rows": [len(rows), len(summaries)],
            "direct_character_error": max_error,
            "new_forwards": 0,
            "neural_runtime_loaded": any(
                n in sys.modules for n in ("torch", "transformers")
            ),
            "scientific_scope": "Actual natural readouts and fixed physical action effects are separate;cosines and sign counts descriptive,not human validity or causal information fractions",
            "source_action": "Unchanged A-source operation;no B-specific fit or alignment",
            "code": {str(Path(__file__).relative_to(ROOT)): R.R.H.sha(__file__)},
            "source": {
                p: R.R.H.sha(out / p)
                for p in ("02/accounts.npz", "02/output_seal.json", "verification.json")
            },
            "original_scope_complete": False,
        },
    )
    R.R.H.write(folder / "output_seal.json", R.R.H.seals(out, folder))
    print("POLICY READBACK", max_error, len(rows), len(summaries), flush=True)


def main():
    """Read the declared complete result once; never collect or resume models."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / R.R.H.load(R.SPEC)["output"]
    )
    readback(parser.parse_args().output.resolve())


if __name__ == "__main__":
    main()
