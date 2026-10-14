#!/usr/bin/env python3
"""Finite paired active-covector/input decomposition; no fitted prediction law."""

from __future__ import annotations

import itertools

import numpy as np

PAIR_COMPONENTS = ("carrier", "shared_transport", "coupling")
WORD_COMPONENTS = (*PAIR_COMPONENTS, "coefficient_contrast", "input_contrast")


def physical_pair(coefficients, inputs, norms):
    """Keep physical action while expressing coefficients per actual H norm."""
    a, z, n = (np.asarray(v, float) for v in (coefficients, inputs, norms))
    if a.ndim != 2 or a.shape[0] != 2 or z.ndim != 1 or n.shape != (2,):
        raise ValueError("two-arm coefficients, one input, two physical norms required")
    if (
        a.shape[1] != len(z)
        or not all(np.isfinite(v).all() for v in (a, z, n))
        or not (n > 0).all()
    ):
        raise ValueError("finite inputs and positive physical units required")
    return a / n[:, None], n[:, None] * z[None, :]


def decompose(theta, eta, full, source_reference):
    """Separate pair mean, covariance and each wording's five error components."""
    theta, eta, full, ref = (
        np.asarray(v, float) for v in (theta, eta, full, source_reference)
    )
    if (
        theta.ndim != 2
        or theta.shape[0] != 2
        or eta.shape != theta.shape
        or full.shape != (2,)
        or ref.shape != (theta.shape[1],)
    ):
        raise ValueError(
            "two ordered wordings with one common source reference required"
        )
    if not all(np.isfinite(v).all() for v in (theta, eta, full, ref)):
        raise ValueError("finite paired values required")
    tm, em = theta.mean(axis=0), eta.mean(axis=0)
    dt, de = (theta[0] - theta[1]) / 2, (eta[0] - eta[1]) / 2
    supported = np.sum(theta * eta, axis=1)
    predicted = eta @ ref
    common, coupling = float(tm @ em), float(dt @ de)
    transport = float((tm - ref) @ em)
    coefficient_contrast, input_contrast = float(dt @ em), float((tm - ref) @ de)
    covariance_norm = float(np.linalg.norm(dt) * np.linalg.norm(de))
    pair = {
        "actual": float(full.mean()),
        "supported": float(supported.mean()),
        "predicted": float(predicted.mean()),
        "common": common,
        "coupling": coupling,
        "carrier": float((full - supported).mean()),
        "shared_transport": transport,
        "error": float((full - predicted).mean()),
        "supported_contrast": float(dt @ em + tm @ de),
        "reading_contrast": coefficient_contrast + input_contrast,
        "coefficient_contrast": coefficient_contrast,
        "input_contrast": input_contrast,
        "covariance_frobenius": covariance_norm,
        "coupling_direction_cosine": coupling / covariance_norm
        if covariance_norm
        else None,
        "pairing_identity_error": abs(float(supported.mean()) - common - coupling),
    }
    pair["error_identity_error"] = abs(
        pair["error"] - sum(pair[k] for k in PAIR_COMPONENTS)
    )
    words = []
    for i, sign in enumerate((1, -1)):
        row = {
            "actual": float(full[i]),
            "supported": float(supported[i]),
            "predicted": float(predicted[i]),
            "carrier": float(full[i] - supported[i]),
            "shared_transport": transport,
            "coupling": coupling,
            "coefficient_contrast": sign * coefficient_contrast,
            "input_contrast": sign * input_contrast,
            "error": float(full[i] - predicted[i]),
        }
        row["error_identity_error"] = abs(
            row["error"] - sum(row[k] for k in WORD_COMPONENTS)
        )
        row["supported_identity_error"] = abs(
            row["supported"] - common - coupling - sign * pair["supported_contrast"]
        )
        words.append(row)
    return pair, words


def component_metrics(rows, components):
    """Retain signed second moments rather than treating RMS terms as additive."""
    if not rows:
        raise ValueError("nonempty observation measure required")
    error = np.array([r["error"] for r in rows], float)
    actual = np.array([r["actual"] for r in rows], float)
    terms = np.array([[r[k] for k in components] for r in rows], float)
    if (
        not np.isfinite(terms).all()
        or not np.isfinite(actual).all()
        or not np.isfinite(error).all()
    ):
        raise ValueError("finite signed observations required")
    actual_ms, error_ms = float(np.mean(actual * actual)), float(np.mean(error * error))
    result = {
        "count": len(rows),
        "actual_rms": float(np.sqrt(actual_ms)),
        "error_rms": float(np.sqrt(error_ms)),
        "relative_error": float(np.sqrt(error_ms / actual_ms)) if actual_ms else None,
    }
    rebuilt = 0.0
    for i, k in enumerate(components):
        ms = float(np.mean(terms[:, i] ** 2))
        result[k + "_rms"], result[k + "_mean"] = (
            float(np.sqrt(ms)),
            float(np.mean(terms[:, i])),
        )
        rebuilt += ms
    for i, j in itertools.combinations(range(len(components)), 2):
        value = float(2 * np.mean(terms[:, i] * terms[:, j]))
        result[components[i] + "__" + components[j] + "_cross"] = value
        rebuilt += value
    result["second_moment_identity_error"] = abs(error_ms - rebuilt)
    return result
