#!/usr/bin/env python3
"""Finite algebra checks for evidence states; no model, text or outcome data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def anchored_mean(counts, *, ratio, baseline, evidence):
    """Evaluate a fixed-content competition rule with positive relative mass."""
    k = np.asarray(counts, dtype=float)
    if not np.all(np.isfinite(k)) or np.any(k < 0):
        raise ValueError("Counts must be finite and nonnegative")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Mass ratio must be finite and positive")
    if not np.all(np.isfinite([baseline, evidence])):
        raise ValueError("Endpoints must be finite")
    return baseline + (evidence - baseline) * (ratio * k / (1 + ratio * k))


def cross_ratio(values):
    """Return the projective cross-ratio of four distinct finite scalar values.

    This exact-world check is not an estimator for noisy, nearly flat curves.
    """
    x = np.asarray(values, dtype=float)
    if x.shape != (4,) or not np.all(np.isfinite(x)):
        raise ValueError("Four finite values required")
    if len(np.unique(x)) != 4:
        raise ValueError("Four distinct values required")
    a, b, c, d = x
    return float((a - c) * (b - d) / ((a - d) * (b - c)))


def append_evidence(mass, moment, new_mass, value):
    """Append a fixed representation using its declared, nonnegative weight."""
    moment = np.asarray(moment, dtype=float)
    value = np.asarray(value, dtype=float)
    if moment.shape != value.shape:
        raise ValueError("Moment and value shapes must match")
    if not np.all(np.isfinite([mass, new_mass])) or mass <= 0 or new_mass < 0:
        raise ValueError("Positive existing mass and nonnegative new mass required")
    if not np.all(np.isfinite(moment)) or not np.all(np.isfinite(value)):
        raise ValueError("Vectors must be finite")
    return float(mass + new_mass), moment + new_mass * value


def query_moments(query_features, key_features, values, weights):
    """Compare direct positive-kernel attention with finite moment readout.

    Inputs are already fixed feature maps, not learned LLM keys or embeddings.
    Positive kernel weights, rather than PSD alone, are required here.
    """
    q, keys, v, w = (
        np.asarray(x, dtype=float)
        for x in (query_features, key_features, values, weights)
    )
    if q.ndim != 2 or keys.ndim != 2 or v.ndim != 2 or w.ndim != 1:
        raise ValueError("Queries, keys, values must be matrices; weights a vector")
    if keys.shape[1] != q.shape[1] or len(v) != len(keys) or len(w) != len(keys):
        raise ValueError("Incompatible feature shapes")
    if any(not np.all(np.isfinite(a)) for a in (q, keys, v, w)) or np.any(w < 0):
        raise ValueError("Finite inputs and nonnegative weights required")
    kernel = q @ keys.T
    mass = kernel @ w
    if np.any(kernel < 0) or np.any(mass <= 0):
        raise ValueError("Nonnegative kernel and positive query masses required")
    z = keys.T @ w
    matrix = keys.T @ (w[:, None] * v)
    direct = (kernel @ (w[:, None] * v)) / mass[:, None]
    compressed = (q @ matrix) / (q @ z)[:, None]
    return direct, compressed, z, matrix


def covariance_terms(target, content, gate, baseline):
    """Decompose gate/content covariance on a common fixed empirical population."""
    y, b, g = (np.asarray(a, dtype=float) for a in (target, content, gate))
    if y.ndim != 1 or y.shape != b.shape or y.shape != g.shape or len(y) < 2:
        raise ValueError("Matching vectors of length at least two required")
    if any(not np.all(np.isfinite(a)) for a in (y, b, g)):
        raise ValueError("Finite vectors required")
    if not np.isfinite(baseline) or np.any((g < 0) | (g > 1)):
        raise ValueError("Finite common baseline and gates in [0,1] required")
    dy, db, dg = y - y.mean(), b - b.mean(), g - g.mean()
    score = baseline + g * (b - baseline)
    terms = {
        "content": float(g.mean() * np.mean(dy * db)),
        "coverage": float((b.mean() - baseline) * np.mean(dy * dg)),
        "interaction": float(np.mean(dy * dg * db)),
    }
    actual = float(np.mean(dy * (score - score.mean())))
    return {
        **terms,
        "total": actual,
        "closure_error": abs(actual - sum(terms.values())),
    }


def witnesses():
    """Construct exact worlds demonstrating what the assumptions do and do not imply."""
    counts = np.array([1.0, 2.0, 4.0, 8.0])
    baseline, evidence, ratio = -0.2, 0.9, 0.7
    curve = anchored_mean(counts, ratio=ratio, baseline=baseline, evidence=evidence)
    recovered = (curve - baseline) / (counts * (evidence - curve))
    l2_curve = counts / np.sqrt(1 + counts**2)
    a0, a = 3.0, 2.1
    homogeneous = (a0 * baseline + counts * a * evidence) / (a0 + counts * a)
    scaled = (7 * a0 * baseline + counts * 7 * a * evidence) / (7 * a0 + counts * 7 * a)

    # Fixed query, fixed keys and values: duplicate a whole evidence bank.
    default_logits, evidence_logits = np.array([0.2, -0.5]), np.array([0.7, -0.2])
    default_values = np.array([-0.4, 0.1])
    evidence_values = np.array([0.5, 1.0])
    w0, we = np.exp(default_logits), np.exp(evidence_logits)
    exact = []
    for k in counts.astype(int):
        logits = np.r_[default_logits, np.tile(evidence_logits, k)]
        values = np.r_[default_values, np.tile(evidence_values, k)]
        w = np.exp(logits - logits.max())
        exact.append(float(w @ values / w.sum()))
    head_curve = anchored_mean(
        counts,
        ratio=we.sum() / w0.sum(),
        baseline=w0 @ default_values / w0.sum(),
        evidence=we @ evidence_values / we.sum(),
    )

    updates = []
    for mass in (1.0, 9.0):
        updated_mass, updated_moment = append_evidence(
            mass, np.array([0.5 * mass]), 1.0, [1.0]
        )
        updates.append(float(updated_moment[0] / updated_mass))
    total, moment = 2.0, np.array([0.4, -0.2])
    bank = [(0.5, [1.0, 0.2]), (2.0, [-0.3, 0.8]), (1.0, [0.7, -0.1])]
    batch_total = total + sum(w for w, _ in bank)
    batch_moment = moment + sum(w * np.array(v) for w, v in bank)
    for weight, value in bank:
        total, moment = append_evidence(total, moment, weight, value)

    # Two source histories have the same (mass,moment), but different provenance.
    source_histories = ({"a", "b"}, {"c", "d"})
    after_same_id = []
    for sources in source_histories:
        weight = 0.0 if "a" in sources else 1.0
        z, t = append_evidence(2.0, np.array([1.0]), weight, [1.0])
        after_same_id.append({"mass": z, "mean": float(t[0] / z)})

    q = np.array([[1, 0.2], [0.3, 1], [1, 1]])
    keys = np.array([[0.1, 1], [1, 0.2], [0.4, 0.6], [0.8, 0.3]])
    values = np.array([[-0.2, 0.1], [0.9, -0.4], [0.4, 0.8], [0.1, 0.7]])
    weights = np.array([1, 0.5, 1.2, 0.8])
    direct, compressed, _, _ = query_moments(q, keys, values, weights)
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    u = np.array([-1.0, 1.0, -1.0, 1.0])
    b = 0.6 * y + 0.3 * u
    constant_gate = 0.4
    score = 0.1 + constant_gate * (b - 0.1)
    variable_gate = 0.5 + 0.1 * y + 0.1 * u
    terms = covariance_terms(y, 1.0 + y * u, variable_gate, 0.0)
    coverage_only = 0.5 + 0.2 * y
    return {
        "scope": "Finite algebra only; no LLM, author or psychological mechanism identified",
        "anchored_curve": {
            "counts": counts.tolist(),
            "values": curve.tolist(),
            "ratio": ratio,
            "recovered_ratios": recovered.tolist(),
            "cross_ratio_input": cross_ratio(counts),
            "cross_ratio_output": cross_ratio(curve),
            "l2_cross_ratio": cross_ratio(l2_curve),
            "common_mass_scale_error": float(np.max(np.abs(scaled - curve))),
            "additive_moment_ratio_error": float(np.max(np.abs(homogeneous - curve))),
        },
        "frozen_attention": {
            "maximum_identity_error": float(np.max(np.abs(head_curve - exact)))
        },
        "update_state": {
            "same_initial_mean": 0.5,
            "initial_masses": [1.0, 9.0],
            "next_means": updates,
            "batch_stream_error": float(
                np.max(np.abs(batch_moment / batch_total - moment / total))
            ),
        },
        "provenance": {
            "same_initial_moments": [2.0, 1.0],
            "after_observation_a": after_same_id,
            "interpretation": "Moments alone cannot implement source-aware duplicate exclusion",
        },
        "query_moments": {
            "direct": direct.tolist(),
            "compressed": compressed.tolist(),
            "maximum_identity_error": float(np.max(np.abs(direct - compressed))),
        },
        "external_association": {
            "constant_gate_correlation_before": float(np.corrcoef(y, b)[0, 1]),
            "constant_gate_correlation_after": float(np.corrcoef(y, score)[0, 1]),
            "mae_before": float(np.mean(np.abs(y - b))),
            "mae_after": float(np.mean(np.abs(y - score))),
            "varying_gate_covariance": terms,
            "constant_content_coverage_only_correlation": float(
                np.corrcoef(y, coverage_only)[0, 1]
            ),
            "constant_content_variance": 0.0,
        },
        "private_data_read": False,
        "neural_forwards": 0,
    }


def main():
    """Write a new, hashed finite-world packet and preserve its code snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = witnesses()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    source = Path(__file__).read_bytes()
    (args.output_dir / "witnesses.json").write_bytes(payload)
    (args.output_dir / "source.py").write_bytes(source)
    summary = {
        "code_sha256": hashlib.sha256(source).hexdigest(),
        "witnesses_sha256": hashlib.sha256(payload).hexdigest(),
        "scope": result["scope"],
        "private_data_read": False,
        "neural_forwards": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
