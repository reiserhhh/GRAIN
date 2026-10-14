"""Observation-only source fitting and sequential scoring for all F02 worlds."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from grain_research.measurement_discovery.estimators import (
    evaluate_author,
    fit_b1,
    fit_b2,
    fit_m,
    ordered_rows,
)


def fit_observed_source(records: list[dict], world: str) -> dict:
    """Fit only source records; reject evaluation roles and explicit truth fields."""
    if not records or any(row["role"] != "source" for row in records):
        raise ValueError("source fitting requires nonempty source-only observations")
    if any({"truth", "kind", "pi", "q", "latent"}.intersection(row) for row in records):
        raise ValueError("truth-bearing fitting input")
    if world.startswith("c2_"):
        groups = ordered_rows(records)
        return {
            "world": world,
            "b1": fit_b1(groups),
            "b2": fit_b2(groups),
            "m": fit_m(groups),
        }
    if world == "c1":
        sums = defaultdict(float)
        counts = defaultdict(int)
        for row in records:
            z = str(row["condition"]["z"])
            sums[z] += row["observation"]
            counts[z] += 1
        return {
            "world": world,
            "source_probability": {z: (sums[z] + 1) / (counts[z] + 2) for z in counts},
        }
    if world == "c3":
        return {
            "world": world,
            "estimator": "observed opportunity and selection counts; no fitted population truth",
        }
    raise ValueError(world)


def sequential_predictions(
    records: list[dict], fitted: dict, prefix: int
) -> list[dict]:
    """Predict before appending each observation; no hidden type or truth access."""
    if any(row["role"] == "source" for row in records):
        raise ValueError("evaluation accepts validation/application observations only")
    world = fitted["world"]
    output = []
    for author, rows in ordered_rows(records).items():
        if world.startswith("c2_"):
            predictions = evaluate_author(rows, fitted, prefix_len=prefix, alpha=1.0)
        elif world == "c1":
            history = list(rows[:prefix])
            predictions = []
            for row in rows[prefix:]:
                z = row["condition"]["z"]
                cell = [r["observation"] for r in history if r["condition"]["z"] == z]
                probabilities = {
                    "source_condition": fitted["source_probability"][str(z)],
                    "personal_condition": (sum(cell) + 1) / (len(cell) + 2),
                    "personal_mean": (sum(r["observation"] for r in history) + 1)
                    / (len(history) + 2),
                }
                for candidate, probability in probabilities.items():
                    predictions.append(
                        {
                            "author_id": author,
                            "event_id": row["event_id"],
                            "t": row["t"],
                            "candidate": candidate,
                            "y": row["observation"],
                            "p": probability,
                        }
                    )
                history.append(row)
        else:
            raise ValueError(
                "C3 has a selection target, not this response-scoring interface"
            )
        for prediction in predictions:
            prediction["role"] = rows[0]["role"]
        output.extend(predictions)
    return output


def selection_views(records: list[dict], prefixes: tuple[int, ...]) -> list[dict]:
    """Estimate observable C3 quantities and bounds without accessing latent types.

    The masked view is constructed from selected indicators only. Published-only
    content has no denominator and therefore yields no point publication rate.
    """
    output = []
    for author, rows in ordered_rows(records).items():
        for prefix in prefixes:
            history = rows[:prefix]
            n = len(history)
            selected = np.array([row["selected"] for row in history], dtype=float)
            opportunity = np.array([row["opportunity"] for row in history], dtype=float)
            rate = float(selected.mean())
            for view in ("opportunity_visible", "opportunity_masked", "published_only"):
                output.append(
                    {
                        "author_id": author,
                        "role": rows[0]["role"],
                        "prefix": n,
                        "view": view,
                        "pi_hat": float(opportunity.mean())
                        if view == "opportunity_visible"
                        else None,
                        "q_hat": float(selected.sum() / opportunity.sum())
                        if view == "opportunity_visible" and opportunity.sum()
                        else None,
                        "publication_hat": rate if view != "published_only" else None,
                        "q_lower": rate if view == "opportunity_masked" else None,
                        "q_upper": 1.0 if view == "opportunity_masked" else None,
                        "bound_type": "sample_plugin_not_confidence_interval"
                        if view == "opportunity_masked"
                        else None,
                        "missing_reason": "unknown total opportunities"
                        if view == "published_only"
                        else None,
                    }
                )
    return output


def paired_bootstrap(
    values_by_seed: dict[int, np.ndarray], repetitions: int = 2000
) -> dict:
    """Paired author-loss differences, resampled within seeds with equal seed weight."""
    rng = np.random.default_rng(20260915)
    groups = [
        np.asarray(value, dtype=float) for _, value in sorted(values_by_seed.items())
    ]
    if not groups or any(
        len(group) == 0 or not np.isfinite(group).all() for group in groups
    ):
        raise ValueError("finite nonempty author differences required for each seed")
    samples = np.mean(
        [
            group[rng.integers(0, len(group), size=(repetitions, len(group)))].mean(
                axis=1
            )
            for group in groups
        ],
        axis=0,
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "delta": float(np.mean([group.mean() for group in groups])),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap": repetitions,
        "paired_authors": sum(map(len, groups)),
        "seeds": len(groups),
        "decision": "SUPPORTED_LOWER_LOSS"
        if high < 0
        else "HIGHER_LOSS"
        if low > 0
        else "INCONCLUSIVE",
    }
