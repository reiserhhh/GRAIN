#!/usr/bin/env python3
"""Nested finite path references: count, excursion configuration and order.

Conditional Markov tests and Euler counting are prior art. This bounded
instrument preserves all components; it does not label a component noise.
"""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_excursion_order as E
from grain_research.construction.math_path_completion import cached_count, count_matrix

READINGS = (
    "raw",
    "excursion_reference",
    "count_reference",
    "configuration_contrast",
    "order_contrast",
)
MAX_EVENTS = 18


def bounded_path(path, states=8):
    """Accept the declared three-to-eighteen-event instrument, without truncation."""
    x = E.path_array(path, states)
    if len(x) > MAX_EVENTS:
        raise ValueError("Path exceeds declared eighteen-event instrument")
    return x


def family_counts(path, states=8):
    """Sum triple counts over ALL distinct words with fixed counts/boundary edges.

    Contract a marked triple abc to a marked edge ac. On the interior graph N,
    the summed occurrences equal C(N-Eab-Ebc+Eac;s,t) times the number of ac
    edges in the contracted graph. Prefix and suffix triples use exact counts
    after removing the corresponding first or last interior edge. Parallel
    edges are not labeled as separate observations.
    """
    x = bounded_path(path, states)
    inner = x[1:-1]
    n = count_matrix(inner, states)
    start, end = int(inner[0]), int(inner[-1])
    denominator = cached_count(n, start, end)
    if denominator < 1:
        raise ArithmeticError("Observed path absent from its own count family")
    if denominator * (len(x) - 2) > np.iinfo(np.int64).max:
        raise OverflowError("Exact path account exceeds registered integer range")
    numerator = np.zeros((states, states, states), dtype=np.int64)
    if len(x) == 3:
        numerator[tuple(x)] = 1
    else:
        for a, b in zip(*np.nonzero(n), strict=True):
            remaining = n.copy()
            remaining[a, b] -= 1
            for c in np.flatnonzero(remaining[b]):
                contracted = remaining.copy()
                contracted[b, c] -= 1
                contracted[a, c] += 1
                numerator[a, b, c] = cached_count(contracted, start, end) * int(
                    contracted[a, c]
                )
        for j in np.flatnonzero(n[start]):
            remaining = n.copy()
            remaining[start, j] -= 1
            numerator[int(x[0]), start, j] += cached_count(remaining, int(j), end)
        for i in np.flatnonzero(n[:, end]):
            remaining = n.copy()
            remaining[i, end] -= 1
            numerator[i, end, int(x[-1])] += cached_count(remaining, start, int(i))
    if numerator.sum() != (len(x) - 2) * denominator:
        raise ArithmeticError("Full count-family triple mass is incomplete")
    return {"numerator": numerator, "denominator": denominator}


def family_words(path, maximum=720, states=8):
    """Enumerate a small COMPLETE verification family; never sample or truncate."""
    x = bounded_path(path, states)
    n = count_matrix(x[1:-1], states)
    start, end = int(x[1]), int(x[-2])
    size = cached_count(n, start, end)
    if size > maximum:
        raise ValueError("Count family exceeds fixed verification enumeration bound")

    def walk(word, left):
        if not left:
            return [(int(x[0]), *word, int(x[-1]))] if word[-1] == end else []
        values = []
        current = word[-1]
        for following in np.flatnonzero(n[current]):
            n[current, following] -= 1
            values.extend(walk(word + (int(following),), left - 1))
            n[current, following] += 1
        return values

    words = walk((start,), int(n.sum()))
    if len(words) != size:
        raise ArithmeticError("Complete word enumeration and exact count disagree")
    return words


def evaluate(bundle, path):
    """Score S,A,C,Q,D with missing support preserved and no target refitting.

    S=A+D=C+Q+D. The old S/A/D remain available on their original support;
    C/Q require the larger full-family support. Cross-component comparisons
    must use the same joint complete set, not different availability sets.
    """
    x = bounded_path(path)
    fine = E.orbit_counts(x)
    coarse = family_counts(x)
    raw = fine["observed_counts"] / (len(x) - 2)
    a = fine["numerator"] / (fine["denominator"] * (len(x) - 2))
    c = coarse["numerator"] / (coarse["denominator"] * (len(x) - 2))
    if np.any((a > 0) & ~(c > 0)):
        raise ArithmeticError("Excursion orbit not contained in full count family")
    masses = np.array([raw, a, c, a - c, raw - a])
    fields = np.asarray(bundle["fields"], float)
    supported = np.asarray(bundle["supported_triples"], bool)
    readings = np.full((3, 2, 5, 32), np.nan)
    for ref in range(3):
        if np.all(supported[ref][a > 0]):
            for kind in (0, 1, 4):
                readings[ref, :, kind] = np.einsum(
                    "abc,pgabc->pg", masses[kind], fields[ref]
                )
        if np.all(supported[ref][c > 0]):
            for kind in (2, 3):
                readings[ref, :, kind] = np.einsum(
                    "abc,pgabc->pg", masses[kind], fields[ref]
                )
    return {
        "family_numerator": coarse["numerator"],
        "family_size": coarse["denominator"],
        "orbit_size": fine["orbit_size"],
        "orbit_numerator": fine["numerator"],
        "orbit_denominator": fine["denominator"],
        "count_coverage": np.sum(c[None] * supported, axis=(1, 2, 3)),
        "orbit_coverage": np.sum(a[None] * supported, axis=(1, 2, 3)),
        "profile_readings": readings,
        "scores": readings @ E.PROFILE_BITS,
        "not_whole_person_posterior": True,
    }


def score_record(bundle, record):
    """Event identity and chronology guard the path; author ID is never a feature."""
    return {
        "author_key": record["author_key"],
        **evaluate(bundle, E.parse_events(record)),
    }


def family_margin_check(path, account):
    """Check total mass and both cropped pair margins with exact integer sums."""
    x = bounded_path(path)
    n = count_matrix(x)
    first, last = n.copy(), n.copy()
    first[x[-2], x[-1]] -= 1
    last[x[0], x[1]] -= 1
    num, den = account["numerator"], account["denominator"]
    return (
        int(num.sum()) == (len(x) - 2) * den
        and np.array_equal(num.sum(2), first * den)
        and np.array_equal(num.sum(0), last * den)
    )
