#!/usr/bin/env python3
"""Recompute Gaussian condition predictions with precision-matrix leave-one-out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.conditional_response import (
    condition_kernel,
    fit_response_reference,
    observed_cells,
)
from scripts.prepare_grain_predecessor_reference import digest, write
from scripts.run_grain_conditional_response import (
    PRIOR,
    author_metrics,
    load_material,
    paired,
)
from scripts.score_grain_conditional_response import restore_reference, score_packet


def run(out):
    """Replay all folds, selection, hidden cells and completed functions from cache."""
    contract = json.loads((out / "contract.json").read_text())
    for name, value in contract["inputs"].items():
        if digest(Path(name)) != value:
            raise ValueError("Changed source input")
    for name, field in (
        ("docs/GRAIN_CONDITIONAL_RESPONSE_PLAN_V1.md", "plan_sha256"),
        ("scripts/run_grain_conditional_response.py", "runner_sha256"),
        ("grain_research/conditional_response.py", "module_sha256"),
    ):
        if digest(ROOT / name) != contract[field]:
            raise ValueError("Changed pre-execution plan or code")
    c, x, g, a, folds, d, contrasts, _facets, _ = load_material()
    selected = json.loads((out / "selected.json").read_text())
    table = pd.read_csv(out / "source_selection.csv")
    for family in ("intercept", "grain_gp"):
        row = (
            table.loc[table.family == family]
            .sort_values(["nmse", "alpha", "beta"])
            .iloc[0]
        )
        for key in ("alpha", "beta", "nmse"):
            np.testing.assert_allclose(
                selected[family][key], row[key], rtol=0, atol=1e-14
            )
    results = {
        "status": "PASS",
        "hidden_cell_predictions": 0,
        "completion_objects": 0,
        "cold_author_scores": 0,
        "max_abs_error": 0.0,
        "independent_executor": False,
    }

    def check(left, right, tol=2e-12):
        """Accumulate the maximum numerical replay discrepancy."""
        left, right = np.asarray(left, float), np.asarray(right, float)
        np.testing.assert_allclose(left, right, rtol=0, atol=tol)
        if left.size:
            results["max_abs_error"] = max(
                results["max_abs_error"], float(abs(left - right).max())
            )

    metrics = pd.read_csv(out / "held_metrics.csv")
    am_by_name = {}
    for phase, source, panel in (
        ("source_validation", np.flatnonzero(folds >= 2), np.flatnonzero(folds == 1)),
        ("held", np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)),
    ):
        root = out / phase
        ref = restore_reference(
            json.loads((root / "private_reference.json").read_text())
        )
        cold = fit_response_reference(c, x, g, a, source)
        for key in (
            "mean",
            "weights",
            "common5_weights",
            "descriptor",
            "noise",
            "descriptor_mean",
            "descriptor_scale",
        ):
            check(cold[key], ref[key], 0)
        if phase == "source_validation":
            arms = {row.arm: (row.alpha, row.beta) for row in table.itertuples()}
        else:
            arms = {
                "source_prior": (0, 0),
                **{k: (v["alpha"], v["beta"]) for k, v in selected.items()},
                "shuffled_geometry": (
                    selected["grain_gp"]["alpha"],
                    selected["grain_gp"]["beta"],
                ),
            }
        for name, (alpha, beta) in arms.items():
            permutation = (
                np.load(out / "private_descriptor_permutation.npy")
                if name == "shuffled_geometry"
                else None
            )
            kernel = condition_kernel(ref, alpha, beta, permutation)
            rows = pd.read_csv(root / f"private_{name}_cells.csv")
            with np.load(
                root / f"private_{name}_predictions.npz", allow_pickle=False
            ) as saved:
                for uid in panel:
                    cell = observed_cells(c[a == uid], x[a == uid], ref)
                    s, means, n = cell["indices"], cell["means"], cell["counts"]
                    ix = np.flatnonzero(rows.author_index == uid)
                    if not len(s):
                        assert not len(ix)
                        continue
                    np.testing.assert_array_equal(rows.condition_index.iloc[ix], s)
                    np.testing.assert_array_equal(rows.target_units.iloc[ix], n)
                    assert (rows.other_known_cells.iloc[ix] == len(s) - 1).all()
                    precision = np.linalg.inv(kernel[np.ix_(s, s)] + np.diag(1 / n))
                    diag = np.diag(precision)
                    # Independent LOCO identity: self-target cancels algebraically.
                    prediction = (
                        means - (precision @ (means - ref["mean"][s])) / diag[:, None]
                    )
                    variance = np.maximum(1 / diag - 1 / n, 0)[:, None] * ref["noise"]
                    check(saved["actual"][ix], means)
                    check(saved["prior"][ix], ref["mean"][s])
                    check(saved["prediction"][ix], prediction)
                    check(saved["personal"][ix], prediction - ref["mean"][s])
                    check(saved["model_sd"][ix], np.sqrt(variance))
                    results["hidden_cell_predictions"] += len(ix)
                am = author_metrics(
                    rows.to_dict("records"),
                    saved["actual"],
                    saved["prediction"],
                    ref["noise"],
                )
                recorded = pd.read_csv(root / f"private_{name}_authors.csv")
                check(am[["mse", "mae", "nmse"]], recorded[["mse", "mae", "nmse"]])
                if phase == "held":
                    am_by_name[name] = am
                    target_metrics = metrics.loc[metrics.arm == name].iloc[0]
                else:
                    target_metrics = table.loc[table.arm == name].iloc[0]
                check(
                    [am[k].mean() for k in ("mse", "mae", "nmse")],
                    [target_metrics[k] for k in ("mse", "mae", "nmse")],
                )
            if phase != "held":
                continue
            for target, key in (("full", "weights"), ("common5", "common5_weights")):
                nu = ref[key]
                with np.load(
                    root / f"private_{name}_{target}_completion.npz", allow_pickle=False
                ) as z:
                    support = pd.read_csv(root / f"private_{name}_{target}_support.csv")
                    for row, uid in enumerate(panel):
                        cells = observed_cells(c[a == uid], x[a == uid], ref)
                        s, means, n = cells["indices"], cells["means"], cells["counts"]
                        w = nu.copy()
                        w[s] = 0
                        adjustment = np.zeros_like(ref["mean"])
                        if len(s):
                            system = kernel[np.ix_(s, s)] + np.diag(1 / n)
                            solution = np.linalg.solve(system, means - ref["mean"][s])
                            adjustment = kernel[:, s] @ solution
                        check(z["observed"][row], nu[s] @ means)
                        check(z["missing_source_prior"][row], w @ ref["mean"])
                        check(z["missing_personal_adjustment"][row], w @ adjustment)
                        check(
                            z["model_completed"][row],
                            nu[s] @ means + w @ (ref["mean"] + adjustment),
                        )
                        check(support.coverage.iloc[row], nu[s].sum())
                        assert bool(support.empirical_complete.iloc[row]) == (
                            w.sum() <= 1e-14
                        )
                        results["completion_objects"] += 1
                        if name != "shuffled_geometry":
                            # Original vectors are not needed: response-interface scores
                            # are already audited upstream. Cold encoder packet below.
                            check(
                                z["observed"][row]
                                + z["missing_source_prior"][row]
                                + z["missing_personal_adjustment"][row],
                                z["model_completed"][row],
                            )
        if phase == "held":
            packet = np.load(
                PRIOR / "pandora/private_single_input.npz", allow_pickle=False
            )
            for name in ("source_prior", "intercept", "grain_gp"):
                alpha, beta = arms[name]
                for target in ("full", "common5"):
                    output = score_packet(
                        packet["conditions"],
                        packet["vectors"],
                        ref,
                        d,
                        contrasts,
                        alpha,
                        beta,
                        target,
                    )
                    with np.load(
                        root / f"private_{name}_{target}_completion.npz",
                        allow_pickle=False,
                    ) as z:
                        for key in (
                            "observed",
                            "missing_source_prior",
                            "missing_personal_adjustment",
                            "model_completed",
                            "model_latent_sd",
                        ):
                            check(output[key], z[key][0])
                    assert output["empirical_point"] is None
                    results["cold_author_scores"] += 1
    intervals = pd.read_csv(out / "paired_author_intervals.csv")
    for row in intervals.itertuples():
        left, right = row.contrast.split("_minus_")
        expected = paired(am_by_name[left], am_by_name[right], row.contrast)
        check(
            [row.delta_nmse, row.ci_low, row.ci_high],
            [expected[k] for k in ("delta_nmse", "ci_low", "ci_high")],
        )
    ref = restore_reference(
        json.loads((out / "held/private_reference.json").read_text())
    )
    risk_rows = []
    for left, right in (
        ("intercept", "source_prior"),
        ("grain_gp", "source_prior"),
        ("grain_gp", "intercept"),
        ("shuffled_geometry", "source_prior"),
    ):
        with (
            np.load(
                out / f"held/private_{left}_predictions.npz", allow_pickle=False
            ) as l,
            np.load(
                out / f"held/private_{right}_predictions.npz", allow_pickle=False
            ) as r,
        ):
            address = pd.read_csv(out / f"held/private_{left}_cells.csv")
            correction = l["prediction"] - r["prediction"]
            residual = r["actual"] - r["prediction"]
            by_author = (
                pd.DataFrame(
                    {
                        "author": address.author_index,
                        "correction_energy": (correction**2 / ref["noise"]).mean(1),
                        "twice_useful_alignment": (
                            2 * correction * residual / ref["noise"]
                        ).mean(1),
                        "delta_nmse": (
                            (
                                (l["prediction"] - l["actual"]) ** 2
                                - (r["prediction"] - r["actual"]) ** 2
                            )
                            / ref["noise"]
                        ).mean(1),
                    }
                )
                .groupby("author")
                .mean()
            )
            check(
                by_author.correction_energy - by_author.twice_useful_alignment,
                by_author.delta_nmse,
            )
            risk_rows.append(
                {"contrast": f"{left}_minus_{right}", **by_author.mean().to_dict()}
            )
    pd.DataFrame(risk_rows).to_csv(out / "risk_decomposition.csv", index=False)
    results["risk_identities"] = 4 * len(am_by_name["source_prior"])
    write(out / "audit.json", results)
    return results


def main():
    """Audit one immutable output folder without refitting or tuning held outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        result = run(args.output.resolve())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
