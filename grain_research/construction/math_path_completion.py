#!/usr/bin/env python3
"""Exact finite path-completion laws, separated from author target anchors.

BEST/matrix-tree counting and completion conditioning are standard mathematics.
This module does not assert that text follows any of these declared laws.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from math import factorial, prod

import numpy as np

LAWS = ("local_depletion", "per_catalog_completion", "global_completion")
REFERENCES = ("pooled", "source_bank_0", "source_bank_1")
TARGETS = ("E", "N", "A", "C", "O")


def count_matrix(path, states=8):
    """Count directed transitions without replacing the ordered input by a bag."""
    a = np.asarray(path)
    if (
        a.ndim != 1
        or not len(a)
        or not np.issubdtype(a.dtype, np.integer)
        or np.any(a < 0)
        or np.any(a >= states)
    ):
        raise ValueError("Nonempty integer state sequence required")
    n = np.zeros((states, states), dtype=int)
    np.add.at(n, (a[:-1], a[1:]), 1)
    return n


def determinant_bareiss(matrix):
    """Compute an integer determinant exactly, including singular/empty minors."""
    a = [[int(x) for x in row] for row in matrix]
    size = len(a)
    if any(len(row) != size for row in a):
        raise ValueError("Square minor required")
    if not size:
        return 1
    sign, previous = 1, 1
    for k in range(size - 1):
        pivot = next((i for i in range(k, size) if a[i][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            sign *= -1
        value = a[k][k]
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                numerator = a[i][j] * value - a[i][k] * a[k][j]
                if numerator % previous:
                    raise ArithmeticError("Nonexact Bareiss division")
                a[i][j] = numerator // previous
            a[i][k] = 0
        previous = value
    return sign * a[-1][-1]


def _graph(n, start, end):
    """Validate graph counts and expose the incident support, not isolated states."""
    a = np.asarray(n)
    if (
        a.ndim != 2
        or a.shape[0] != a.shape[1]
        or not np.issubdtype(a.dtype, np.integer)
        or np.any(a < 0)
        or not 0 <= start < len(a)
        or not 0 <= end < len(a)
    ):
        raise ValueError("Invalid nonnegative integer graph or endpoints")
    out, incoming = a.sum(1), a.sum(0)
    balance = np.zeros(len(a), int)
    balance[start] += 1
    balance[end] -= 1
    support = sorted(set(np.flatnonzero(out + incoming)) | {start, end})
    return a, out, support, bool(np.array_equal(out - incoming, balance))


def tree_count(n, start, end, determinant=determinant_bareiss):
    """Count in-arborescences to the terminal state using the out-Laplacian."""
    a, out, support, balanced = _graph(n, start, end)
    if not balanced:
        return 0
    if not a.sum():
        return int(start == end)
    if any(out[i] == 0 for i in support if i != end):
        return 0
    vertices = [i for i in support if i != end]
    laplacian = np.diag(out) - a
    return determinant(laplacian[np.ix_(vertices, vertices)].tolist())


def word_count(n, start, end, determinant=determinant_bareiss):
    """Count distinct state words; parallel edges are not distinct observations."""
    a, out, support, balanced = _graph(n, start, end)
    if not balanced:
        return 0
    if not a.sum():
        return int(start == end)
    tau = tree_count(a, start, end, determinant)
    if not tau:
        return 0
    numerator = tau * prod(factorial(int(out[i]) - 1 + int(i == end)) for i in support)
    denominator = prod(factorial(int(v)) for v in a.ravel())
    if numerator % denominator:
        raise ArithmeticError("Euler word count is not integral")
    return numerator // denominator


@lru_cache(maxsize=100000)
def _cached_count(flat, states, start, end):
    """Memoize a graph identity, never a target score or an author ID."""
    return word_count(np.array(flat, int).reshape(states, states), start, end)


def cached_count(n, start, end):
    """Return a verified graph count with a bounded computational cache."""
    return _cached_count(tuple(int(v) for v in n.ravel()), len(n), start, end)


def success_probability(n, start, end, count=None):
    """Probability that local edge depletion uses every edge and ends correctly."""
    a, out, _, _ = _graph(n, start, end)
    c = cached_count(a, start, end) if count is None else count
    return Fraction(
        c * prod(factorial(int(v)) for v in a.ravel()),
        prod(factorial(int(v)) for v in out),
    )


def next_laws(n, current, end):
    """Return local and completion-conditioned next-state probabilities exactly."""
    a, out, _, _ = _graph(n, current, end)
    local = [Fraction(0) for _ in a]
    complete = [Fraction(0) for _ in a]
    c = cached_count(a, current, end)
    if not out[current]:
        return local, complete
    for j in range(len(a)):
        if a[current, j]:
            local[j] = Fraction(int(a[current, j]), int(out[current]))
            remainder = a.copy()
            remainder[current, j] -= 1
            if c:
                complete[j] = Fraction(cached_count(remainder, j, end), c)
    return local, complete


def likelihoods(path, start, end, n):
    """All prefix likelihoods for one catalogue; unsupported events stay zero.

    The empty observation is included. Global completion uses unnormalized
    local-prefix times remaining-success weights, not per-catalogue division.
    """
    a = np.asarray(path, int)
    c0 = cached_count(n, start, end)
    if not c0:
        raise ValueError("Source catalogue has no complete word")
    h0 = success_probability(n, start, end, c0)
    values = np.zeros((len(a) + 1, 3), float)
    values[0] = (1, 1, float(h0))
    exact_checks = 0
    if not len(a) or a[0] != start:
        return values, exact_checks
    values[1] = values[0]
    remaining, local = n.copy(), Fraction(1)
    current = start
    for k, state in enumerate(a[1:], 2):
        denominator = int(remaining[current].sum())
        edge = int(remaining[current, state])
        if not edge or not denominator:
            break
        local *= Fraction(edge, denominator)
        remaining[current, state] -= 1
        current = int(state)
        c = cached_count(remaining, current, end)
        h = success_probability(remaining, current, end, c)
        complete = Fraction(c, c0)
        if complete != local * h / h0:
            raise ArithmeticError("Completion likelihood identity failed")
        exact_checks += 1
        values[k] = float(local), float(complete), float(local * h)
    return values, exact_checks


def compile_bundle(paths, targets, banks):
    """Freeze SOURCE inventories and full anchor profiles, not application outcomes."""
    p, y, bank = np.asarray(paths), np.asarray(targets), np.asarray(banks)
    if p.ndim != 3 or y.shape != (2, len(p), 5) or bank.shape != (len(p),):
        raise ValueError("SOURCE shape mismatch")
    if not np.isin(y, (0, 1)).all() or not np.isin(bank, (0, 1)).all():
        raise ValueError("SOURCE profiles/banks must be categorical")
    catalogues = []
    for u, windows in enumerate(p):
        for w, path in enumerate(windows):
            n = count_matrix(path)
            start, end = int(path[0]), int(path[-1])
            c = cached_count(n, start, end)
            h = success_probability(n, start, end, c)
            catalogues.append(
                {
                    "source_author_index": u,
                    "source_window": w,
                    "bank": int(bank[u]),
                    "start": start,
                    "end": end,
                    "counts": n.tolist(),
                    "words": c,
                    "success": [h.numerator, h.denominator],
                    "profiles": y[:, u].tolist(),
                }
            )
    return {
        "version": "HM264-v1",
        "states": 8,
        "targets": list(TARGETS),
        "laws": list(LAWS),
        "references": list(REFERENCES),
        "catalogues": catalogues,
    }


def parse_events(record):
    """Accept only attested ordered numeric events; IDs are guards, not features."""
    if set(record) != {"author_key", "events"} or not isinstance(
        record["author_key"], str
    ):
        raise ValueError("Outcome-free record schema required")
    events = record["events"]
    if not events or any(set(e) != {"event_id", "timestamp", "cell"} for e in events):
        raise ValueError("Event identity, chronology and cell required")
    ids = [e["event_id"] for e in events]
    times = np.array([e["timestamp"] for e in events], float)
    cells = [e["cell"] for e in events]
    if (
        any(not isinstance(i, str) for i in ids)
        or len(set(ids)) != len(ids)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or any(type(i) is not int or not 0 <= i < 8 for i in cells)
    ):
        raise ValueError("Repeated, unordered, nonfinite or invalid event")
    return np.array(cells, int)


def score_record(bundle, record):
    """Score a whole path against three declared laws and three SOURCE references.

    Returns full 32-profile posteriors as well as five marginal scores. No
    unsupported posterior is substituted with 0, 0.5, a smoother or a teacher.
    """
    path = parse_events(record)
    cats = bundle["catalogues"]
    weights, exact_checks = [], 0
    for c in cats:
        v, checks = likelihoods(path, c["start"], c["end"], np.array(c["counts"], int))
        weights.append(v)
        exact_checks += checks
    weights = np.array(weights)
    bank = np.array([c["bank"] for c in cats])
    profiles = np.array([c["profiles"] for c in cats], int)
    codes = profiles @ (1 << np.arange(5))
    post = np.full((3, 2, len(path) + 1, 3, 32), np.nan)
    evidence = np.zeros((3, len(path) + 1, 3))
    support = np.zeros_like(evidence, int)
    authors = np.zeros_like(evidence, int)
    doomed = np.full((3, len(path) + 1), np.nan)
    for ref in range(3):
        subset = np.ones(len(cats), bool) if ref == 0 else bank == ref - 1
        w = weights[subset]
        denominator = w[:, 0].sum(0)
        total = w.sum(0)
        evidence[ref] = total / denominator
        support[ref] = (w > 0).sum(0)
        author_ids = np.array([c["source_author_index"] for c in cats])[subset]
        for length in range(len(path) + 1):
            for law in range(3):
                authors[ref, length, law] = len(set(author_ids[w[:, length, law] > 0]))
            if total[length, 0]:
                doomed[ref, length] = (
                    w[:, length, 0][w[:, length, 1] == 0].sum() / total[length, 0]
                )
        for assignment in range(2):
            code = codes[subset, assignment]
            for length in range(len(path) + 1):
                for law in range(3):
                    if total[length, law]:
                        post[ref, assignment, length, law] = (
                            np.bincount(code, weights=w[:, length, law], minlength=32)
                            / total[length, law]
                        )
    alphabet = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
    scores = post @ alphabet
    prior = scores[:, :, :1]
    return {
        "author_key": record["author_key"],
        "scores": scores,
        "departures": scores - prior,
        "profiles": post,
        "evidence_probability": evidence,
        "support_catalogues": support,
        "support_authors": authors,
        "doomed_local_weight_fraction": doomed,
        "exact_prefix_checks": exact_checks,
    }


def serializable(value):
    """Serialize missing numerical observations as null while retaining shapes."""
    if isinstance(value, np.ndarray):
        return np.where(np.isfinite(value), value, None).tolist()
    return value


def toy_results():
    """Expose a controlled closure mechanism with identical local first-step laws."""
    paths = np.array([[[0, 1, 0, 2, 3]], [[0, 2, 0, 1, 3]]], int)
    targets = np.array([[[1] * 5, [0] * 5]] * 2, int)
    bundle = compile_bundle(paths, targets, np.array([0, 1]))
    record = {
        "author_key": "designed-toy",
        "events": [
            {"event_id": str(i), "timestamp": i, "cell": s}
            for i, s in enumerate((0, 1))
        ],
    }
    result = score_record(bundle, record)
    row = result["scores"][0, 0, 2, :, 0]
    return {
        "designed_not_human": True,
        "equal_start_end_bag_outdegrees_and_local_first_step": True,
        "first_edge_posterior_local_PC_GC": row.tolist(),
        "initial_completion_success": [c["success"] for c in bundle["catalogues"]],
        "law_order": list(LAWS),
    }


def enumerable_words(n, start, end):
    """Independent recursive enumeration for small graphs only, used by tests."""
    a = np.array(n, int)
    if not a.sum():
        return [(start,)] if start == end else []
    words = []
    for j in range(len(a)):
        if a[start, j]:
            b = a.copy()
            b[start, j] -= 1
            words.extend((start,) + tail for tail in enumerable_words(b, j, end))
    return words


def enumeration_groups(words, states):
    """Group explicit words by graph/endpoints, independent of the BEST formula."""
    return Counter((tuple(count_matrix(w, states).ravel()), w[0], w[-1]) for w in words)
