#!/usr/bin/env python3
"""Verify HM94 matched-null scores by a candidate-indexed score table."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_grain_opportunity_current_hm94 as R


def verify():
    """Read frozen numerical products only; never fit or encode new observations."""
    out = R.OUT
    cfg = R.sealed(out)
    for name in ("01", "02", "03", "04_matched_null"):
        R.previous(out, name)
    target = out / "04_matched_null"
    manifest = R.read(target / "manifest.json")
    for key, path in (
        ("plan_hash", ROOT / "docs/GRAIN_OPPORTUNITY_MATCHED_NULL_HM94.md"),
        ("code_hash", ROOT / "scripts/analyze_grain_opportunity_null_hm94.py"),
        ("input_hash", out / "02/private_author_accounts.npz"),
    ):
        if R.sha(path) != manifest[key]:
            raise AssertionError("Changed companion seal: " + key)
    destination = out / "matched_null_verification.json"
    if destination.exists():
        raise ValueError("Preserve earlier verification")
    meta = R.read(out / "02/private_metadata.json")
    with np.load(out / "02/private_author_accounts.npz", allow_pickle=False) as z:
        cycle = np.stack([z[r["key"] + "_parts"] for r in meta])[..., 2, :, :]
        q = np.stack([z[r["key"] + "_q"] for r in meta]).sum(3)
    with np.load(target / "null_distributions.npz", allow_pickle=False) as z:
        draws = {key: z[key] for key in z.files}
    battery = np.array([len(r["queries"]) for r in meta])
    n, permutations = len(meta), draws.pop("permutations")
    expected = np.tile(np.arange(n), (cfg["permutation_draws"], 1))
    rng = np.random.default_rng(cfg["seed"])
    for row in expected:
        for value in (4, 5):
            indices = np.flatnonzero(battery == value)
            row[indices] = rng.permutation(indices)
    if not np.array_equal(expected, permutations):
        raise AssertionError("Permutation seed/order mismatch")
    if not np.all(np.sort(permutations, axis=1) == np.arange(n)):
        raise AssertionError("Not whole-author bijections")
    if not np.all(battery[permutations] == battery):
        raise AssertionError("Battery crossing")
    table = pd.read_csv(target / "matched_null_and_unselected.csv").set_index(
        ["arm", "channel"]
    )
    tolerance = manifest["tie_tolerance"]
    errors, checks = [], 0
    for ai, arm in enumerate(cfg["arms"]):
        for ci, channel in enumerate(("raw", "opportunity", "response")):
            matrices = cycle[:, :, ai, ci]
            degenerate = np.max(np.ptp(matrices.reshape(-1, 16), axis=0)) < tolerance
            if bool(table.loc[(arm, channel), "degenerate"]) != bool(degenerate):
                raise AssertionError("Constant account flag mismatch")
            observed, null, unselected = [], [], []
            for a, b in ((0, 1), (0, 2), (1, 2)):
                # For skew matrices, half the Frobenius norm equals the upper-edge norm.
                distance = 0.5 * np.sum(
                    (matrices[:, a, None] - matrices[None, :, b]) ** 2, (-2, -1)
                )
                context = np.sum((q[:, a, None] - q[None, :, b]) ** 2, -1)
                scores = np.empty((n, n))
                ordinary = np.empty(n)
                for author in range(n):
                    order = sorted(range(n), key=lambda j: (context[author, j], j))
                    for candidate in range(n):
                        rivals = [j for j in order if j != candidate][
                            : cfg["matched_strangers"]
                        ]
                        margin = distance[author, rivals] - distance[author, candidate]
                        scores[author, candidate] = (
                            0.5
                            if degenerate
                            else np.mean(
                                (margin > tolerance).astype(float)
                                + 0.5 * (np.abs(margin) <= tolerance)
                            )
                        )
                    rivals = np.flatnonzero(
                        (battery == battery[author]) & (np.arange(n) != author)
                    )
                    margin = distance[author, rivals] - distance[author, author]
                    ordinary[author] = (
                        0.5
                        if degenerate
                        else np.mean(
                            (margin > tolerance).astype(float)
                            + 0.5 * (np.abs(margin) <= tolerance)
                        )
                    )
                observed.append(np.diag(scores))
                null.append(scores[np.arange(n)[None, :], permutations].mean(1))
                unselected.append(ordinary)
                checks += scores.size + len(permutations) + n
            observed = np.mean(observed, axis=0)
            null = np.mean(null, axis=0)
            unselected = np.mean(unselected, axis=0)
            if np.max(np.abs(null - draws[f"{arm}_{channel}"])) > 1e-12:
                raise AssertionError("Saved null distribution differs")
            bootstrap_rng = np.random.default_rng(cfg["seed"])
            bootstrap = unselected[
                bootstrap_rng.integers(0, n, (cfg["bootstrap_draws"], n))
            ].mean(1)
            values = {
                "matched_observed": observed.mean(),
                "matched_null_mean": null.mean(),
                "matched_null_lower": np.quantile(null, 0.025),
                "matched_null_upper": np.quantile(null, 0.975),
                "delta_over_matched_null": observed.mean() - null.mean(),
                "permutation_p": (1 + np.count_nonzero(null >= observed.mean()))
                / (len(null) + 1),
                "unselected_within_battery_auc": unselected.mean(),
                "unselected_ci_lower": np.quantile(bootstrap, 0.025),
                "unselected_ci_upper": np.quantile(bootstrap, 0.975),
            }
            errors.extend(
                abs(value - table.loc[(arm, channel), key])
                for key, value in values.items()
            )
    maximum = float(max(errors))
    if maximum > 1e-12:
        raise AssertionError(f"Companion arithmetic mismatch: {maximum}")
    result = {
        "status": "VERIFIED_FINITE_HM94_MATCHED_NULL",
        "maximum_alternate_error": maximum,
        "candidate_score_checks": checks,
        "accounts": len(table),
        "permutations": len(permutations),
        "new_fits": 0,
        "new_calls": 0,
        "same_agent_alternate_arithmetic": True,
        "original_scope_complete": False,
        "verifier_hash": R.sha(Path(__file__)),
    }
    R.write(destination, result)
    print(result)


if __name__ == "__main__":
    verify()
