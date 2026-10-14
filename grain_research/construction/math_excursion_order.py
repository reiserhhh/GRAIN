#!/usr/bin/env python3
"""Complete-path order functionals conditional on a return-excursion orbit.

Unknown homogeneous Markov probabilities cancel inside the declared orbit.
Natural text need not satisfy that null; no component is called pure noise.
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from math import factorial, prod

import numpy as np

from grain_research.construction.math_path_completion import count_matrix, parse_events

TARGETS = ("E", "N", "A", "C", "O")
REFERENCES = ("pooled", "source_bank_0", "source_bank_1")
ASSIGNMENTS = ("actual", "old_permutation")
PROFILE_BITS = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(int)


def path_array(path, states=8):
    """Reject short, noninteger or out-of-alphabet ordered observations."""
    x = np.asarray(path)
    if (
        x.ndim != 1
        or len(x) < 3
        or not np.issubdtype(x.dtype, np.integer)
        or np.any((x < 0) | (x >= states))
    ):
        raise ValueError("At least three in-range integer states required")
    return x


def triple_counts(path, states=8):
    """Count width3 observations without declaring overlapping units independent."""
    x = path_array(path, states)
    n = np.zeros((states, states, states), dtype=np.int64)
    np.add.at(n, (x[:-2], x[1:-1], x[2:]), 1)
    return n


def split_excursions(path, states=8):
    """Use an orbit-invariant anchor and uniquely parsed nonempty return blocks."""
    x = path_array(path, states)
    anchor = int(np.bincount(x, minlength=states).argmax())
    visits = np.flatnonzero(x == anchor)
    prefix = tuple(int(v) for v in x[: visits[0] + 1])
    suffix = tuple(int(v) for v in x[visits[-1] + 1 :])
    blocks = tuple(
        tuple(int(v) for v in x[a + 1 : b + 1])
        for a, b in pairwise(visits)
    )
    # Boundary edges are fixed so even the cropped triple pair margins match.
    if len(blocks) >= 2:
        prefix += blocks[0]
        suffix = blocks[-1] + suffix
        blocks = blocks[1:-1]
    elif blocks:
        prefix += blocks[0]
        blocks = ()
    multiplicity = Counter(blocks)
    size = factorial(len(blocks)) // prod(factorial(v) for v in multiplicity.values())
    return anchor, prefix, blocks, suffix, size


def orbit_counts(path, states=8):
    """Compute exact expected triple counts over ALL distinct block permutations.

    Internal triples are fixed. Ordered adjacent labeled-block pairs occur with
    probability1/k; ordered triples occur with probability1/(k(k-1)). Fixed
    prefix/suffix boundary probabilities are treated separately, including loops.
    """
    x = path_array(path, states)
    anchor, prefix, blocks, suffix, size = split_excursions(x, states)
    k = len(blocks)
    denominator = k * (k - 1) if k >= 2 else 1
    numerator = np.zeros((states, states, states), dtype=np.int64)

    def add(triple, weight):
        numerator[triple] += weight

    def pair(left, right, weight):
        if not left or not right:
            return
        if len(left) >= 2:
            add((left[-2], left[-1], right[0]), weight)
        if len(right) >= 2:
            add((left[-1], right[0], right[1]), weight)

    def three(left, middle, right, weight):
        if left and len(middle) == 1 and right:
            add((left[-1], middle[0], right[0]), weight)

    for segment in (prefix, *blocks, suffix):
        for i in range(len(segment) - 2):
            add(tuple(segment[i : i + 3]), denominator)
    if k == 0:
        pair(prefix, suffix, denominator)
    else:
        single = denominator // k
        for a in blocks:
            pair(prefix, a, single)
            pair(a, suffix, single)
        for i, a in enumerate(blocks):
            for j, b in enumerate(blocks):
                if i != j:
                    pair(a, b, single)
        if k == 1:
            three(prefix, blocks[0], suffix, 1)
        else:
            for i, a in enumerate(blocks):
                for j, b in enumerate(blocks):
                    if i == j:
                        continue
                    three(prefix, a, b, 1)
                    three(a, b, suffix, 1)
                    if len(b) == 1:
                        for h, c in enumerate(blocks):
                            if h != i and h != j:
                                three(a, b, c, 1)
    if numerator.sum() != (len(x) - 2) * denominator:
        raise ArithmeticError("Orbit triple mass is incomplete")
    return {
        "anchor": anchor,
        "blocks": k,
        "orbit_size": size,
        "numerator": numerator,
        "denominator": denominator,
        "observed_counts": triple_counts(x, states),
    }


def orbit_words(path, maximum=720, states=8):
    """Enumerate only a fixed small verification orbit, never truncate a score."""
    _, prefix, blocks, suffix, size = split_excursions(path, states)
    if size > maximum:
        raise ValueError("Verification orbit exceeds declared enumeration bound")
    remaining = Counter(blocks)

    def walk(chosen):
        if len(chosen) == len(blocks):
            return [prefix + sum(chosen, ()) + suffix]
        values = []
        for block in sorted(remaining):
            if remaining[block]:
                remaining[block] -= 1
                values.extend(walk(chosen + (block,)))
                remaining[block] += 1
        return values

    return walk(())


def compile_bundle(paths, targets, banks):
    """Compile equal-author/window SOURCE profile fields; no application labels."""
    p, y, bank = np.asarray(paths), np.asarray(targets), np.asarray(banks)
    if (
        p.ndim != 3
        or y.shape != (2, len(p), 5)
        or bank.shape != (len(p),)
        or not np.isin(y, (0, 1)).all()
        or not np.isin(bank, (0, 1)).all()
    ):
        raise ValueError("SOURCE paths/profiles/banks are misaligned")
    masses = np.array(
        [[triple_counts(x) / (len(x) - 2) for x in row] for row in p]
    ).mean(1)
    fields, priors, coverage = [], [], []
    for ref in range(3):
        keep = np.ones(len(p), bool) if ref == 0 else bank == ref - 1
        if not keep.any():
            raise ValueError("Empty source reference")
        tables = np.zeros((2, 32, 8, 8, 8))
        prior = np.zeros((2, 32))
        for assignment in range(2):
            codes = y[assignment, keep] @ (1 << np.arange(5))
            codes = codes.astype(int)
            np.add.at(tables[assignment], codes, masses[keep] / keep.sum())
            prior[assignment] = np.bincount(codes, minlength=32) / keep.sum()
        den = tables.sum(1)
        field = np.divide(
            tables,
            den[:, None],
            out=np.zeros_like(tables),
            where=den[:, None] > 0,
        )
        fields.append(field.tolist())
        priors.append(prior.tolist())
        coverage.append((den[0] > 0).tolist())
    return {
        "version": "HM265-v1",
        "targets": list(TARGETS),
        "profile_bits": PROFILE_BITS.tolist(),
        "references": list(REFERENCES),
        "fields": fields,
        "priors": priors,
        "supported_triples": coverage,
        "not_whole_person_posterior": True,
        "states": 8,
    }


def evaluate(bundle, path):
    """Keep raw, exact orbit reference and contrast; unsupported scores stay NaN."""
    x = path_array(path)
    geometry = orbit_counts(x)
    observed = geometry["observed_counts"] / (len(x) - 2)
    expected = geometry["numerator"] / (geometry["denominator"] * (len(x) - 2))
    active = expected > 0
    if np.any((observed > 0) & ~active):
        raise ArithmeticError("Observed member absent from its own orbit")
    fields = np.asarray(bundle["fields"], float)
    supported = np.asarray(bundle["supported_triples"], bool)
    readings = np.full((3, 2, 3, 32), np.nan)
    coverage = np.sum(expected[None] * supported, axis=(1, 2, 3))
    for ref in range(3):
        if np.all(supported[ref][active]):
            raw = np.einsum("abc,pgabc->pg", observed, fields[ref])
            baseline = np.einsum("abc,pgabc->pg", expected, fields[ref])
            readings[ref] = np.stack((raw, baseline, raw - baseline), axis=1)
    return {
        **geometry,
        "coverage": coverage,
        "profile_readings": readings,
        "scores": readings @ PROFILE_BITS,
        "not_whole_person_posterior": True,
    }


def score_record(bundle, record):
    """Use event IDs/time only as guards; measure the attested ordered state path."""
    return {
        "author_key": record["author_key"],
        **evaluate(bundle, parse_events(record)),
    }


def json_safe(value):
    """Serialize unsupported quantities as null, never a neutral target value."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def adjacent_margin_check(path, account):
    """Check both triple pair margins exactly, using fixed first and last edges."""
    x = path_array(path)
    n = count_matrix(x)
    first, last = n.copy(), n.copy()
    first[x[-2], x[-1]] -= 1
    last[x[0], x[1]] -= 1
    num, den = account["numerator"], account["denominator"]
    return bool(np.array_equal(num.sum(2), first * den)) and bool(
        np.array_equal(num.sum(0), last * den)
    )
