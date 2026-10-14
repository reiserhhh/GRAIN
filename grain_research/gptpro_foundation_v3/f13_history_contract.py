#!/usr/bin/env python3
"""Cold uncertainty contract backed by the existing versioned evidence ledger."""

import numpy as np

from grain_research.evidence_history import active_records, kernels

from .f13_uncertainty import validate_covariance


def conditional_moments(mean, prior, x, noise, values):
    """Condition on centered residuals, including valid deterministic observations."""
    mean, x, values = np.asarray(mean), np.asarray(x), np.asarray(values)
    prior, noise = validate_covariance(prior), validate_covariance(noise)
    if not np.isfinite(values).all():
        raise ValueError("finite observations required")
    k = validate_covariance(x @ prior @ x.T + noise)
    inverse = np.linalg.pinv(k, hermitian=True, rcond=1e-12)
    residual = values - x @ mean
    if not np.allclose(residual, k @ inverse @ residual, atol=1e-10, rtol=1e-8):
        raise ValueError("observations outside specified Gaussian support")
    weights = prior @ x.T @ inverse
    covariance = validate_covariance(prior - weights @ x @ prior)
    return {
        "mean": mean + weights @ residual,
        "covariance": covariance,
        "weights": weights,
    }


def score_history(history, model, target):
    """Replay explicit revisions and score a linear target under working moments.

    Empty evidence with a specified model is borrowed, not empirical support.
    Without a model no numerical score or invented interval is returned.
    """
    target = np.asarray(target, float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("finite level/condition-response target required")
    authors = {r.get("author_id") for r in history}
    if history and (None in authors or len(authors) != 1):
        raise ValueError("one explicitly identified author per ledger required")
    ledger = active_records(history)
    records = ledger["records"]
    result = {
        "score": None,
        "variance": None,
        "lower": None,
        "upper": None,
        "status": "unobserved_without_model",
        "active_records": len(records),
        "active_events": ledger["active_events"],
        "stale_derived": ledger["stale_derived"],
        "variance_components": None,
        "empirical_target_identified": False,
    }
    if model is None:
        result["reason"] = (
            "no specified score model; no finite bounds without an explicit parameter domain"
        )
        return result
    components = np.asarray(model["components"], float)
    mean = np.asarray(model["mean"], float)
    if (
        components.shape != (5,)
        or not np.isfinite(components).all()
        or np.any(components < 0)
        or mean.shape != (2,)
        or not np.isfinite(mean).all()
    ):
        raise ValueError("finite nonnegative working components and mean required")
    prior = np.diag(components[:2])
    if records:
        x, bases = kernels(records)
        values = np.array([r["value"] for r in records], float)
        noise = np.einsum("k,kij->ij", components[2:], bases[2:])
        fitted = conditional_moments(mean, prior, x, noise, values)
        covariance = validate_covariance(fitted["covariance"])
        score = float(fitted["mean"] @ target)
        weights = target @ fitted["weights"]
        remainder = target - weights @ x
        # This is target estimation-error variance, not raw-score variance.
        terms = [remainder[i] ** 2 * components[i] for i in range(2)]
        terms.extend(
            float(components[i] * (weights @ bases[i] @ weights)) for i in range(2, 5)
        )
        projection = target @ np.linalg.pinv(x) @ x
        identified = bool(np.allclose(projection, target, atol=1e-10, rtol=1e-8))
        result["empirical_target_identified"] = identified
        result["status"] = (
            "observed_model_conditioned"
            if identified
            else "model_borrowed_partial_support"
        )
    else:
        covariance = prior
        score = float(mean @ target)
        terms = [target[i] ** 2 * components[i] for i in range(2)] + [0.0, 0.0, 0.0]
        result["status"] = "model_borrowed_no_active_evidence"
    variance = float(target @ covariance @ target)
    if variance < -1e-10 or not np.isclose(sum(terms), variance, atol=1e-10, rtol=1e-8):
        raise ValueError("conditional covariance and error-component identity disagree")
    variance = max(0.0, variance)
    half = 1.96 * np.sqrt(variance)
    result.update(
        score=score,
        variance=variance,
        lower=score - half,
        upper=score + half,
        variance_components=dict(
            zip(
                (
                    "person_prior_remainder",
                    "person_condition_prior_remainder",
                    "block",
                    "event",
                    "report",
                ),
                terms,
                strict=True,
            )
        ),
        interval_kind="fixed_source_gaussian_working_model",
        source_uncertainty_included=False,
    )
    return result
