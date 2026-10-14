#!/usr/bin/env python3
"""Coordinator-side T3 evaluation: planned outcomes and alternate Gaussian audit.

No construction or parameter optimization is performed here. Ground-truth
parameters are allowed only in the separate post-score outcome evaluator.
"""

import numpy as np


def planned_outcomes(rows, true_slope, plan):
    """Reconstruct the explicit intervention under the additive core simulator.

    Reuse exactly the archived latent states and measurement innovations. This
    identity is inapplicable when the condition also changes state transitions.
    """
    if len(rows) != len(plan) or not rows or not np.isfinite(true_slope):
        raise ValueError("finite slope and equally sized nonempty future required")
    if any(row["world"] != "core" for row in rows):
        raise ValueError("planned-outcome identity is restricted to core")
    y = np.asarray([row["y"] for row in rows], dtype=float)
    z = np.asarray([row["z"] for row in rows], dtype=float)
    plan = np.asarray(plan, dtype=float)
    if not all(np.isfinite(value).all() for value in (y, z, plan)):
        raise ValueError("nonfinite future input")
    return y + true_slope * (plan - z)


def dense_future(parameters, rows, log_sigma, plan=(1, -1, 1, -1)):
    """Condition the joint stationary past/future law without a Kalman recursion."""
    a, b, eta = np.asarray(parameters, dtype=float)
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.asarray(log_sigma, dtype=float))
    times = np.asarray([row["t"] for row in rows], dtype=int)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("ordered nonempty prefix required")
    future_times = times[-1] + np.arange(1, len(plan) + 1)
    stationary = q / (1 - rho * rho)
    past = stationary * rho ** np.abs(times[:, None] - times[None, :])
    past += r * np.eye(len(times))
    cross = stationary * rho ** (future_times[:, None] - times[None, :])
    future = stationary * rho ** np.abs(
        future_times[:, None] - future_times[None, :]
    ) + r * np.eye(len(plan))
    residual = np.asarray([row["y"] - a - b * row["z"] for row in rows])
    solutions = np.linalg.solve(past, np.column_stack([residual, cross.T]))
    mean = a + b * np.asarray(plan) + cross @ solutions[:, 0]
    covariance = future - cross @ solutions[:, 1:]
    return float(mean.mean()), float(covariance.sum() / len(plan) ** 2)


def moment_discrepancy(score, bundle):
    """Recompute every stored component with raw moments rather than scorer sums."""
    if not score["supported"]:
        raise ValueError("unsupported score has no Gaussian mixture to audit")
    parts = score["per_source"]
    if len(parts) != bundle["source_bootstraps"] or not all(
        row["supported"] for row in parts
    ):
        raise ValueError("incomplete source mixture")
    n = len(parts)
    means = [row["mean"] for row in parts]
    center = sum(means) / n
    source = sum(value * value for value in means) / n - center * center
    expected = {
        key: sum(row[key] for row in parts) / n
        for key in (
            "event_state",
            "personal_parameter",
            "terminal_state",
            "future_innovation",
            "future_observation",
        )
    }
    expected["source"] = source
    expected["total"] = sum(
        expected[key] for key in ("event_state", "personal_parameter", "source")
    )
    errors = [
        abs(score["raw_score"] - center),
        *(abs(expected[key] - score["variance_components"][key]) for key in expected),
        abs(
            expected["event_state"]
            - sum(
                expected[key]
                for key in ("terminal_state", "future_innovation", "future_observation")
            )
        ),
        *(
            abs(row["within_total"] - row["event_state"] - row["personal_parameter"])
            for row in parts
        ),
    ]
    if score["source_standardized_score"] is not None:
        errors.extend(
            [
                abs(
                    score["source_standardized_score"]
                    - (center - bundle["reference_mu"]) / bundle["reference_sigma"]
                ),
                abs(
                    score["source_standardized_variance"]
                    - expected["total"] / bundle["reference_sigma"] ** 2
                ),
            ]
        )
    return max(errors)
