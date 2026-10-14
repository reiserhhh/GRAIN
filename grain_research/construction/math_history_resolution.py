#!/usr/bin/env python3
"""Design-supported author functionals, without a fixed latent class count.

The declared world has C|p ~ Bernoulli(p), R|p ~ Bernoulli(p), independently,
and Y = C xor R. A free event records both its selected condition and response.
This is a stipulated observation law, not an inference about natural authors.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np
from scipy.optimize import linprog


def matched_populations(order=4):
    """Return exact positive laws indistinguishable through `order` trials."""
    if not isinstance(order, int) or order < 1:
        raise ValueError("Positive integer order required")
    degree = order + 1
    support = [Fraction(i, degree) for i in range(degree + 1)]
    normalizer = 2**order
    even = [
        Fraction(comb(degree, i), normalizer) if i % 2 == 0 else Fraction(0)
        for i in range(degree + 1)
    ]
    odd = [
        Fraction(comb(degree, i), normalizer) if i % 2 else Fraction(0)
        for i in range(degree + 1)
    ]
    return support, even, odd


def exact_fingerprint(support, mass, trials, erase_choice=False):
    """Integrate every count probability using rational arithmetic."""
    probabilities = [2 * p * (1 - p) if erase_choice else p for p in support]
    return [
        sum(
            w * comb(trials, k) * p**k * (1 - p) ** (trials - k)
            for p, w in zip(probabilities, mass, strict=True)
        )
        for k in range(trials + 1)
    ]


def beta_fingerprint(a, b, trials):
    """Return an exact Beta-binomial source law for integer Beta parameters."""

    def beta(x, y):
        from math import factorial

        return Fraction(factorial(x - 1) * factorial(y - 1), factorial(x + y - 1))

    if min(a, b) < 1:
        raise ValueError("Positive Beta parameters required")
    return [
        comb(trials, k) * beta(a + k, b + trials - k) / beta(a, b)
        for k in range(trials + 1)
    ]


def binomial_matrix(support, trials, erase_choice=False):
    """Observation matrix on an explicitly finite witness support."""
    p = np.asarray(support, dtype=float)
    if erase_choice:
        p = 2 * p * (1 - p)
    return np.stack(
        [comb(trials, k) * p**k * (1 - p) ** (trials - k) for k in range(trials + 1)]
    )


def compile_history_moment(source_trials, history_trials, successes, power):
    """Compile p**power * Binomial-history likelihood in a Bernstein basis.

    These weights act on source count frequencies, not on latent source labels.
    Polynomial identity holds on the entire interval, not just a witness grid.
    """
    if not 0 <= successes <= history_trials or min(power, history_trials) < 0:
        raise ValueError("Invalid history/moment")
    excess = source_trials - history_trials - power
    if excess < 0:
        raise ValueError("Requested functional exceeds source observation degree")
    weights = np.zeros(source_trials + 1)
    for extra in range(excess + 1):
        j = successes + power + extra
        weights[j] = (
            comb(history_trials, successes)
            * comb(excess, extra)
            / comb(source_trials, j)
        )
    return weights


def cold_profile(bundle, history):
    """Score only source frequencies and a qualified new author's history.

    A point here is a population-supported conditional propensity estimate.
    Its conditional variance is not a population-estimation confidence interval.
    Empirical frequencies can violate mixture feasibility; retain that failure.
    """
    if set(bundle) != {"source_trials", "frequencies"}:
        raise ValueError("Cold source bundle may not contain latent/teacher targets")
    if set(history) != {"history_trials", "successes"}:
        raise ValueError("Cold history may not contain latent/teacher targets")
    n = bundle["source_trials"]
    f = np.asarray(bundle["frequencies"], dtype=float)
    if f.shape != (n + 1,) or np.any(f < 0) or not np.isclose(f.sum(), 1):
        raise ValueError("Invalid source count frequencies")
    m, k = history["history_trials"], history["successes"]
    if n < m + 1:
        return {
            "status": "SOURCE_DEGREE_INSUFFICIENT",
            "mean": None,
            "variance": None,
            "history_mass": None,
        }
    denominator = float(compile_history_moment(n, m, k, 0) @ f)
    if denominator <= 0:
        return {
            "status": "UNOBSERVED_HISTORY",
            "mean": None,
            "variance": None,
            "history_mass": 0.0,
        }
    mean = float(compile_history_moment(n, m, k, 1) @ f / denominator)
    variance = None
    status = "MEAN_IDENTIFIED_VARIANCE_UNIDENTIFIED"
    if n >= m + 2:
        second = float(compile_history_moment(n, m, k, 2) @ f / denominator)
        variance = second - mean * mean
        status = "MEAN_AND_VARIANCE_IDENTIFIED"
        if variance < -1e-12:
            status = "INADMISSIBLE_EMPIRICAL_MOMENTS"
    return {
        "status": status,
        "mean": mean,
        "variance": variance,
        "history_mass": denominator,
    }


def finite_ratio_bounds(matrix, fingerprint, numerator, denominator):
    """Sharp finite-support bounds by a Charnes-Cooper linear program.

    Grid bounds are not asserted sharp over all distributions on [0,1].
    Matrix columns are probability laws; normalization is enforced explicitly.
    """
    a, b = np.asarray(matrix, float), np.asarray(fingerprint, float)
    num, den = np.asarray(numerator, float), np.asarray(denominator, float)
    if a.shape != (len(b), len(num)) or den.shape != num.shape:
        raise ValueError("Incompatible finite observation/target dimensions")
    a = np.vstack((a, np.ones(len(num))))
    b = np.r_[b, 1.0]
    equality = np.vstack((np.c_[a, -b], np.r_[den, 0]))
    rhs = np.r_[np.zeros(len(b)), 1.0]
    witnesses, values = [], []
    for sign in (1, -1):
        result = linprog(
            sign * np.r_[num, 0],
            A_eq=equality,
            b_eq=rhs,
            bounds=(0, None),
            method="highs",
        )
        if not result.success or result.x[-1] <= 0:
            raise ValueError(
                "No positive-denominator finite witness: " + result.message
            )
        witness = result.x[:-1] / result.x[-1]
        witnesses.append(witness)
        values.append(float(num @ witness / (den @ witness)))
    return values, witnesses


def decode_vectors(vectors, alphabet):
    """Decode the declared exact emission alphabet; not arbitrary natural text."""
    vectors, alphabet = np.asarray(vectors), np.asarray(alphabet)
    if alphabet.ndim != 2 or vectors.ndim != 2 or vectors.shape[1] != alphabet.shape[1]:
        raise ValueError("Invalid vector alphabet")
    distance = np.sum((vectors[:, None] - alphabet[None]) ** 2, axis=2)
    nearest = distance.argmin(axis=1)
    if np.max(distance[np.arange(len(vectors)), nearest], initial=0) > 1e-20:
        raise ValueError("Input is outside this finite exact-emission experiment")
    return nearest


def count_distinct_events(records):
    """Deduplicate event identity before allocating observation degree."""
    seen = {}
    for row in records:
        if set(row) != {"event_id", "condition", "response", "qualification"}:
            raise ValueError("Unrecognized event fields")
        if row["qualification"] != "self_actual" or not row["event_id"]:
            raise ValueError("Unqualified evidence")
        pair = row["condition"], row["response"]
        if any(x not in (0, 1) for x in pair):
            raise ValueError("Nonbinary declared observation")
        if row["event_id"] in seen and seen[row["event_id"]] != pair:
            raise ValueError("An event version changed without a correction rule")
        seen[row["event_id"]] = pair
    return len(seen)
