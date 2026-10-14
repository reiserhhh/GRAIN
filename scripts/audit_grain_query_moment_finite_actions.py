#!/usr/bin/env python3
"""Independently recompute finite scalar formulas and every recorded comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.query_moment_actions import array_hash
from scripts.run_grain_query_moment_finite_actions import (
    DEFAULT,
    PRIOR,
    check_predictions,
    pure,
    verify,
)
from scripts.run_grain_query_moment_operator import (
    ARMS,
    NATIVE,
    QUERIES,
    arrays,
    directions,
    load_models,
    read,
    sha,
    write,
)


def reference(g, h, model):
    """Compute scores without the application polynomial/residual/scalar helpers."""
    z = (g - model["g_mean"]) / model["g_scale"]
    x1 = np.column_stack((np.ones(len(g)), z))
    x2 = np.column_stack(
        (x1, *[z[:, i] * z[:, j] for i in range(4) for j in range(i, 4)])
    )
    first = np.sum((x1.T @ h / len(h)) * model["beta1"])
    second = np.sum((x2.T @ h / len(h)) * model["beta2"])
    _, ix = np.unique(g, axis=0, return_inverse=True)
    residual = h.astype(float).copy()
    for group in np.unique(ix):
        subset = ix == group
        residual[subset] -= h[subset].astype(float).mean(0)
    projected = residual @ model["query_basis"]
    within = sum(row @ model["within_matrix"] @ row for row in projected) / len(h)
    return np.array([first, second, second + within]) + model["offsets"]


def audit(out):
    """Check immutable inputs, independent finite math and recorded native bookkeeping."""
    contract, models = verify(out), load_models(PRIOR)
    freeze = check_predictions(out)
    status = read(out / "native/status.json")
    if status["status"] != "COMPLETE" or status["calls"] != contract["max_calls"]:
        raise ValueError("Only completed finite packet can pass full acceptance")
    if (
        not contract["prepared_time_ns"]
        < freeze["frozen_time_ns"]
        < status["started_time_ns"]
    ):
        raise ValueError("Prediction/native chronology invalid")
    for folder in ("native", "evaluation"):
        meta = status if folder == "native" else read(out / folder / "freeze.json")
        for name, expected in meta["files"].items():
            if sha(out / folder / name) != expected:
                raise ValueError("Changed output: " + name)
    pred = pd.read_csv(out / "application/private_predictions.csv").set_index(
        ["query_key", "action", "arm"]
    )
    obs = pd.read_csv(out / "application/private_observations.csv").set_index(
        ["query_key", "action"]
    )
    actions = pd.read_csv(out / "evaluation/private_actions.csv")
    stresses = pd.read_csv(out / "evaluation/private_stresses.csv")
    compare = pd.concat([actions, stresses]).set_index(
        ["author_key", "query", "action", "arm"]
    )
    components = pd.read_csv(out / "evaluation/private_components.csv").set_index(
        ["author_key", "query", "direction", "epsilon", "arm"]
    )
    max_error, total, component_checks = 0.0, 0, 0
    for job in contract["jobs"]:
        for query, key in zip(QUERIES, job["query_keys"], strict=True):
            data = pure(key)
            h, g = data["state"][data["positions"]], data["grain"]
            model = models[query]
            ds = directions({"g": g, "h": h.astype(float)}, models)
            y0 = reference(g, h, model)
            packet = read(out / "native" / (key + ".json"))
            if [r["action"] for r in packet["rows"]] != [
                s["action"] for s in contract["actions"]
            ]:
                raise ValueError("Missing, reordered or duplicate action")
            values = {r["action"]: r for r in packet["rows"]}
            baseline = values["baseline"]["value"]
            archived = read(NATIVE / "private_packets" / (key + ".json"))["contrast"]
            np.testing.assert_allclose(
                baseline, archived, atol=contract["replay_score_atol"], rtol=0
            )
            j = arrays(NATIVE / "private_packets" / (key + ".npz"))[
                "body_gradient"
            ].astype(float)
            explicit, tangent = {}, {}
            _, cells = np.unique(g, axis=0, return_inverse=True)
            for spec in contract["actions"]:
                changed = h.astype(float).copy()
                if spec["kind"] == "finite":
                    changed += spec["sign"] * spec["epsilon"] * ds[spec["direction"]]
                elif spec["kind"] == "stress":
                    for cell in np.unique(cells):
                        ix = np.flatnonzero(cells == cell)
                        changed[ix] = (
                            h[ix[::-1]]
                            if spec["action"] == "cell_reverse"
                            else 2 * h[ix].astype(float).mean(0) - h[ix]
                        )
                changed = changed.astype(h.dtype)
                full = data["state"].copy()
                full[data["positions"]] = changed
                unmodified = np.ones(len(full), bool)
                unmodified[data["positions"]] = False
                np.testing.assert_array_equal(
                    full[unmodified], data["state"][unmodified]
                )
                name = spec["action"]
                hashes = {
                    "body_sha256": array_hash(changed),
                    "full_sha256": array_hash(full),
                }
                for k, expected in hashes.items():
                    if (
                        obs.loc[(key, name), k] != expected
                        or values[name][k] != expected
                    ):
                        raise ValueError(
                            "Independent reconstruction differs from executed bytes"
                        )
                scalar = reference(g, changed, model)
                explicit[name] = scalar - y0
                tangent[name] = float(np.sum(j * (changed.astype(float) - h)))
                for ai, arm in enumerate(ARMS):
                    row = pred.loc[key, name, arm]
                    error = abs(row.value - scalar[ai])
                    max_error = max(max_error, error)
                    np.testing.assert_allclose(
                        row.value, scalar[ai], atol=1e-8, rtol=1e-10
                    )
                    np.testing.assert_allclose(
                        row.delta, explicit[name][ai], atol=1e-8, rtol=1e-10
                    )
                    if spec["kind"] != "baseline":
                        rr = compare.loc[job["key"], query, name, arm]
                        np.testing.assert_allclose(
                            [rr.actual, rr.predicted, rr.native_tangent],
                            [
                                values[name]["value"] - baseline,
                                explicit[name][ai],
                                tangent[name],
                            ],
                            atol=1e-8,
                            rtol=1e-10,
                        )
                total += 1
            for direction in ds:
                for e in contract["epsilons"]:
                    plus, minus = [f"{direction}_{e:.6g}_{s:+d}" for s in (1, -1)]
                    actual = [
                        (values[plus]["value"] - values[minus]["value"]) / 2,
                        (values[plus]["value"] + values[minus]["value"]) / 2 - baseline,
                    ]
                    for ai, arm in enumerate((*ARMS, "native_tangent")):
                        pv, mv = (
                            (tangent[plus], tangent[minus])
                            if arm == "native_tangent"
                            else (explicit[plus][ai], explicit[minus][ai])
                        )
                        expected = [*actual, (pv - mv) / 2, (pv + mv) / 2]
                        row = components.loc[job["key"], query, direction, e, arm]
                        np.testing.assert_allclose(
                            row[
                                [
                                    "actual_odd",
                                    "actual_even",
                                    "predicted_odd",
                                    "predicted_even",
                                ]
                            ].to_numpy(float),
                            expected,
                            atol=1e-8,
                            rtol=1e-10,
                        )
                        component_checks += 1
    if total != 496 or len(pred) != total * 3 or len(compare) != (total - 16) * 3:
        raise ValueError("Unexpected number of application/evaluation records")
    # A separate author-first aggregation checks all RMSE and interval rows.
    cc = components.reset_index()
    table = pd.read_csv(out / "evaluation/metrics.csv")
    groups = {
        "all7": list(ds),
        "F4": list(ds)[:4],
        "within2": list(ds)[4:6],
        "random": ["within_random"],
    }
    for row in table.itertuples():
        p = cc[
            (cc["query"] == row.query)
            & (cc.epsilon == row.epsilon)
            & cc.direction.isin(groups[row.group])
        ]
        order = sorted(p.author_key.unique())
        errors = []
        for arm in ("quadratic", row.arm):
            errors.append(
                np.array(
                    [
                        np.mean(
                            (
                                part["actual_" + row.component]
                                - part["predicted_" + row.component]
                            )
                            ** 2
                        )
                        for author in order
                        for part in [p[(p.author_key == author) & (p.arm == arm)]]
                    ]
                )
            )
        baseline, candidate = errors
        ix = np.random.default_rng(contract["bootstrap_seed"]).integers(
            len(order), size=(contract["bootstrap_samples"], len(order))
        )
        draw = np.sqrt(np.mean(baseline[ix], axis=1)) - np.sqrt(
            np.mean(candidate[ix], axis=1)
        )
        expected = [
            np.sqrt(candidate.mean()),
            np.sqrt(baseline.mean()) - np.sqrt(candidate.mean()),
            *np.quantile(draw, [0.025, 0.975]),
        ]
        np.testing.assert_allclose(
            [
                row.rmse,
                row.rmse_reduction_vs_quadratic,
                row.delta_ci_low,
                row.delta_ci_high,
            ],
            expected,
            atol=1e-10,
            rtol=1e-8,
        )
        if row.positive_author_count != int(
            np.sum(candidate < baseline)
        ) or row.authors != len(order):
            raise ValueError("Wrong author count/positive count")
    stress_table = pd.read_csv(out / "evaluation/stress_summary.csv")
    for row in stress_table.itertuples():
        p = stresses[(stresses["query"] == row.query) & (stresses.action == row.action)]
        actual = p[p.arm == "quadratic"].actual.to_numpy()
        np.testing.assert_allclose(
            [
                row.native_delta_rms,
                row.native_abs_min,
                row.native_abs_max,
                row.explicit_abs_max,
            ],
            [
                np.sqrt(np.mean(actual**2)),
                min(abs(actual)),
                max(abs(actual)),
                p.predicted.abs().max(),
            ],
            atol=1e-10,
        )
    result = {
        "status": "PASS",
        "intervention_byte_reconstructions": total,
        "independent_scalar_checks": len(pred),
        "independent_component_checks": component_checks,
        "metric_rows": len(table),
        "stress_summary_rows": len(stress_table),
        "max_scalar_error": max_error,
        "native_calls_repeated_by_audit": 0,
        "scope": "Independent formulas and recorded native output/address replay; not an independently repeated model run",
    }
    write(out / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        audit(args.output.resolve())
