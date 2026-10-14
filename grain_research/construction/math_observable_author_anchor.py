#!/usr/bin/env python3
"""Observable author functions with a declared behavioral endpoint and regime.

No latent population, hidden trait law, class labels or teacher answers are
constructor inputs. Dictionary decoding is exact finite-world decoding only.
"""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np

TARGETS = ("future_expression_q0", "future_expression_q1")
REGIMES = ("randomized_condition", "self_selected_condition")


def decode_vectors(vectors, alphabet):
    """Decode exact registered vectors; reject arbitrary new text embeddings."""
    a, v = np.asarray(alphabet, float), np.asarray(vectors, float)
    if (
        a.ndim != 2
        or a.shape[0] != 4
        or v.ndim != 2
        or v.shape[1] != a.shape[1]
        or not np.isfinite(a).all()
        or not np.isfinite(v).all()
        or len(np.unique(a, axis=0)) != 4
    ):
        raise ValueError("Four distinct finite prototypes and aligned vectors required")
    distances = np.sum((v[:, None] - a[None]) ** 2, axis=-1)
    index = np.argmin(distances, axis=1)
    if np.any(distances[np.arange(len(v)), index] > 1e-20):
        raise ValueError("Out-of-alphabet observation; no natural-text calibration")
    return index


def observed_table(author_ids, histories, endpoints, alphabet):
    """Count one history and one distinct endpoint record per source author."""
    ids = np.asarray(author_ids)
    h, endpoint = (
        decode_vectors(histories, alphabet),
        decode_vectors(endpoints, alphabet),
    )
    if ids.ndim != 1 or len(ids) != len(h) or len(h) != len(endpoint):
        raise ValueError("Source pairing must be row-aligned")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate source-author record; do not multiply carriers")
    q = endpoint // 2
    y = q ^ (endpoint % 2)
    counts = np.zeros((4, 2, 2), dtype=np.int64)
    np.add.at(counts, (h, q, y), 1)
    return counts


def compile_bundle(table, regime):
    """Compile conditional endpoint functions from observed joint frequencies.

    A randomized regime is supplied by the collection design, not deduced from
    this table. Missing support is preserved without pseudocounts or rescue fits.
    """
    t = np.asarray(table, float)
    if regime not in REGIMES:
        raise ValueError("Unknown collection regime")
    if (
        t.shape != (4, 2, 2)
        or not np.isfinite(t).all()
        or np.any(t < 0)
        or t.sum() <= 0
    ):
        raise ValueError("Positive finite history/condition/outcome table required")
    mass = t.sum(axis=(1, 2)) / t.sum()
    support = t.sum(axis=2)
    probability = np.divide(
        t[:, :, 1], support, out=np.full((4, 2), np.nan), where=support > 0
    )
    reference = []
    for q in range(2):
        reference.append(
            float(mass[mass > 0] @ probability[mass > 0, q])
            if np.all(support[mass > 0, q] > 0)
            else None
        )
    return {
        "version": "HM160_observable_anchor_v1",
        "regime": regime,
        "target_ids": list(TARGETS),
        "history_mass": mass.tolist(),
        "probabilities": [
            [None if np.isnan(x) else float(x) for x in row] for row in probability
        ],
        "reference": reference,
        "support": support.tolist(),
    }


def cold_profile(bundle, history_vector, alphabet):
    """Score a new history without application outcomes or hidden coordinates."""
    fields = {
        "version",
        "regime",
        "target_ids",
        "history_mass",
        "probabilities",
        "reference",
        "support",
    }
    if set(bundle) != fields or bundle["version"] != "HM160_observable_anchor_v1":
        raise ValueError("Cold bundle schema excludes latent/teacher fields")
    if bundle["regime"] not in REGIMES or bundle["target_ids"] != list(TARGETS):
        raise ValueError("Registered targets and regime required")
    mass = np.asarray(bundle["history_mass"], float)
    probability = np.array(bundle["probabilities"], float)
    reference = np.array(bundle["reference"], float)
    support = np.asarray(bundle["support"], float)
    if (
        mass.shape != (4,)
        or probability.shape != (4, 2)
        or reference.shape != (2,)
        or support.shape != (4, 2)
        or not np.isfinite(mass).all()
        or not np.isfinite(support).all()
        or np.any(support < 0)
        or np.any(mass < 0)
        or not np.isclose(mass.sum(), 1)
        or np.any((probability < 0) | (probability > 1))
        or np.any((reference < 0) | (reference > 1))
        or np.any(np.isinf(probability))
        or np.any(np.isinf(reference))
    ):
        raise ValueError("Invalid observable probabilities/support")
    observed = support > 0
    if np.any(np.isnan(probability) == observed):
        raise ValueError("Probability/support mismatch")
    if support.sum() <= 0 or not np.allclose(mass, support.sum(axis=1) / support.sum()):
        raise ValueError("History mass must come from registered source support")
    for q in range(2):
        covered = np.all(support[mass > 0, q] > 0)
        if covered != np.isfinite(reference[q]):
            raise ValueError("Reference support mismatch")
        if covered and not np.isclose(
            reference[q], mass[mass > 0] @ probability[mass > 0, q]
        ):
            raise ValueError("Reference must use the same registered source population")
    h = int(decode_vectors(np.asarray(history_vector, float)[None], alphabet)[0])
    status = (
        "RANDOMIZED_CONDITION_PROFILE"
        if bundle["regime"] == "randomized_condition"
        else "ASSOCIATIONAL_CONDITION_PROFILE"
    )
    outputs = []
    for q in range(2):
        supported = support[h, q] > 0
        outputs.append(
            {
                "target": TARGETS[q],
                "status": "OBSERVED_SOURCE_SUPPORT"
                if supported
                else "UNOBSERVED_HISTORY_CONDITION",
                "probability": float(probability[h, q]) if supported else None,
                "reference": float(reference[q]) if np.isfinite(reference[q]) else None,
                "relative": float(probability[h, q] - reference[q])
                if supported and np.isfinite(reference[q])
                else None,
            }
        )
    return {
        "status": status,
        "history_address": h,
        "history_mass": float(mass[h]),
        "targets": outputs,
    }


def history_law(p):
    """Exact declared (C,Y) address probabilities for the simulator only."""
    return [(1 - p) ** 2, (1 - p) * p, p**2, p * (1 - p)]


def exact_table(regime="randomized_condition", null=False):
    """Enumerate the declared source law; not called by the constructor."""
    if regime not in REGIMES:
        raise ValueError("Unknown collection regime")
    support, mass = [F(1, 20), F(9, 20), F(19, 20)], [F(1, 2), F(3, 10), F(1, 5)]
    table = [[[F(0) for _ in range(2)] for _ in range(2)] for _ in range(4)]
    for p, w in zip(support, mass, strict=True):
        assignment = (
            [F(1, 2), F(1, 2)] if regime == "randomized_condition" else [1 - p, p]
        )
        rates = [F(3, 10), F(3, 5)] if null else [p, p * p]
        for h, ph in enumerate(history_law(p)):
            for q in range(2):
                table[h][q][0] += w * ph * assignment[q] * (1 - rates[q])
                table[h][q][1] += w * ph * assignment[q] * rates[q]
    return table


def score_vectors(bundle, histories, alphabet):
    """Apply four frozen supported profiles without any new-author outcome."""
    addresses = decode_vectors(histories, alphabet)
    profiles = [cold_profile(bundle, row, alphabet) for row in np.asarray(alphabet)]
    probabilities = np.array(
        [[x["probability"] for x in profile["targets"]] for profile in profiles], float
    )
    return probabilities[addresses], profiles


def gauge_witness():
    """Exact reparameterization witness, separate from any scoring input."""
    p, w = [F(1, 20), F(9, 20), F(19, 20)], [F(1, 2), F(3, 10), F(1, 5)]
    likelihood = [x * (1 - x) for x in p]
    den = sum(a * b for a, b in zip(w, likelihood, strict=True))
    posterior = [a * b / den for a, b in zip(w, likelihood, strict=True)]
    result = {}
    for name, values in (("p", p), ("p_squared", [x * x for x in p])):
        prior = sum(a * x for a, x in zip(w, values, strict=True))
        post = sum(a * x for a, x in zip(posterior, values, strict=True))
        result[name] = {
            "reference": prior,
            "conditional_mean": post,
            "relative": post - prior,
        }
    midrank = []
    previous = F(0)
    for weight in w:
        midrank.append(previous + weight / 2)
        previous += weight
    result["midrank_invariant_not_scored"] = {
        "reference": sum(a * x for a, x in zip(w, midrank, strict=True)),
        "conditional_mean": sum(a * x for a, x in zip(posterior, midrank, strict=True)),
    }
    return result


def selection_alias_table(world, regime):
    """Exact observational alias with different randomized endpoint responses."""
    if world not in (0, 1) or regime not in REGIMES:
        raise ValueError("Declared alias world/regime required")
    rates = (
        [[F(1, 5), F(3, 10)], [F(7, 10), F(4, 5)]]
        if world == 0
        else [[F(1, 5), F(9, 10)], [F(1, 10), F(4, 5)]]
    )
    table = [[[F(0) for _ in range(2)] for _ in range(2)] for _ in range(4)]
    for author_type, p in enumerate((F(1, 4), F(3, 4))):
        assignment = (
            [F(1, 2)] * 2
            if regime == "randomized_condition"
            else [F(int(author_type == q)) for q in range(2)]
        )
        for h, ph in enumerate(history_law(p)):
            for q in range(2):
                table[h][q][0] += (
                    F(1, 2) * ph * assignment[q] * (1 - rates[author_type][q])
                )
                table[h][q][1] += F(1, 2) * ph * assignment[q] * rates[author_type][q]
    return table
