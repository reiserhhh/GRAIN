#!/usr/bin/env python3
"""Fixed cross-target sensitivity diagnostics, without prediction optimization."""

from __future__ import annotations

import itertools

import numpy as np

TARGETS = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
    "anthropomorphism",
)
REGIMES = ("balanced", "alias_focus", "distinct_focus", "other_focus")
UNITS = ("score", "log_odds")
ZERO_NORM = 1e-12
UPDATES = (
    ("alias_minus_balanced", "alias_focus", "balanced"),
    ("distinct_minus_alias", "distinct_focus", "alias_focus"),
    ("other_minus_balanced", "other_focus", "balanced"),
)


def decode(logits):
    """Decode ten finite candidate logits in both fixed, non-interchangeable units."""
    x = np.asarray(logits, dtype=float)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError("Ten finite logits required")
    y = x - x.max()
    p = np.exp(y)
    p /= p.sum()
    high, low = x[5:], x[:5]
    high_lse = high.max() + np.log(np.exp(high - high.max()).sum())
    low_lse = low.max() + np.log(np.exp(low - low.max()).sum())
    return {
        "score": float(p @ (np.arange(10) / 9)),
        "log_odds": float(high_lse - low_lse),
    }


def observation_key(row):
    """Use explicit role, condition, family, polarity and query identities."""
    return tuple(
        row[k]
        for k in ("role", "regime", "family", "polarity", "order", "content", "query")
    )


def validate_observations(rows):
    """Reject incomplete, duplicate, mismatched or nonfinite factorial observations."""
    expected = {
        ("atom", "single_event", f, p, -1, c, q)
        for f, p, c, q in itertools.product((0, 1), (0, 1), TARGETS, TARGETS)
    } | {
        ("account", r, f, p, o, c, q)
        for r, f, p, o, c, q in itertools.product(
            REGIMES, (0, 1), (0, 1), (0, 1), TARGETS, TARGETS
        )
    }
    seen, units, parents = set(), set(), {}
    for row in rows:
        key = observation_key(row)
        if key in seen or row["unit"] in units or key not in expected:
            raise ValueError("Duplicate or unexpected observation")
        decoded = decode(row["candidate_logits"])
        if any(
            not np.isfinite(row[k]) or abs(row[k] - decoded[k]) > 1e-10 for k in UNITS
        ):
            raise ValueError("Readout/logit disagreement")
        if not 0 <= row["candidate_mass"] <= 1:
            raise ValueError("Invalid candidate vocabulary coverage")
        doc = key[:-1]
        if doc in parents and parents[doc] != row["parent_unit"]:
            raise ValueError("One body has inconsistent parent identities")
        parents[doc] = row["parent_unit"]
        seen.add(key)
        units.add(row["unit"])
    if seen != expected or units != set(range(1296)) or len(parents) != 216:
        raise ValueError("Complete 216-body/1296-readout Cartesian design required")
    if len(set(parents.values())) != 216:
        raise ValueError("Different bodies share a parent identity")
    return {observation_key(r): r for r in rows}


def matrices(rows, metric):
    """Construct the full predeclared 50-operator roster without selecting conditions."""
    if metric not in UNITS:
        raise ValueError("Unknown measurement unit")
    lookup = validate_observations(rows)
    result, atoms, accounts = {}, {}, {}

    def contrast(role, regime, f, o):
        return np.asarray(
            [
                [
                    lookup[role, regime, f, 1, o, c, q][metric]
                    - lookup[role, regime, f, 0, o, c, q][metric]
                    for c in TARGETS
                ]
                for q in TARGETS
            ],
            dtype=float,
        )

    for f in (0, 1):
        atoms[f] = contrast("atom", "single_event", f, -1)
        result[f"atom/f{f}"] = atoms[f]
    result["atom/pooled"] = (atoms[0] + atoms[1]) / 2
    for r in REGIMES:
        for f, o in itertools.product((0, 1), (0, 1)):
            accounts[r, f, o] = contrast("account", r, f, o)
            result[f"account/{r}/f{f}/o{o}"] = accounts[r, f, o]
        result[f"account/{r}/pooled"] = np.mean(
            [accounts[r, f, o] for f, o in itertools.product((0, 1), (0, 1))], axis=0
        )
    for name, high, low in UPDATES:
        cells = []
        for f, o in itertools.product((0, 1), (0, 1)):
            d = accounts[high, f, o] - accounts[low, f, o]
            result[f"update/{name}/f{f}/o{o}"] = d
            cells.append(d)
        result[f"update/{name}/pooled"] = np.mean(cells, axis=0)
    for r in REGIMES:
        cells = []
        for f in (0, 1):
            d = accounts[r, f, 1] - accounts[r, f, 0]
            result[f"order/{r}/f{f}"] = d
            cells.append(d)
        result[f"order/{r}/pooled"] = np.mean(cells, axis=0)
    if len(result) != 50:
        raise ValueError("Fixed roster changed")
    return result


def finite_matrix(value):
    """Require a complete, finite square six-target operator."""
    d = np.asarray(value, dtype=float)
    if d.shape != (6, 6) or not np.isfinite(d).all():
        raise ValueError("Finite 6x6 matrix required")
    return d


def incidence():
    """Return oriented K6,6 incidence in query-major edge order."""
    b = np.zeros((36, 12))
    for q, c in itertools.product(range(6), repeat=2):
        b[6 * q + c, q] = 1
        b[6 * q + c, 6 + c] = -1
    return b


def graph_split(value):
    """Separate additive node effects and divergence-free interaction edges."""
    d = finite_matrix(value)
    g = d.mean(axis=1)[:, None] + d.mean(axis=0)[None, :] - d.mean()
    c = d - g
    return g, c


def cosine(a, b):
    """Return cosine or an explicit undefined value for zero response energy."""
    a, b = finite_matrix(a), finite_matrix(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return (
        None
        if min(np.linalg.norm(a), np.linalg.norm(b)) <= ZERO_NORM
        else float(np.sum(a * b) / denom)
    )


def spectrum(value):
    """Report the full fixed spectrum; numerical rank is not psychological factor count."""
    d = finite_matrix(value)
    s = np.linalg.svd(d, compute_uv=False)
    energy = float(np.sum(s**2))
    tol = max(ZERO_NORM, 6 * np.finfo(float).eps * s[0])
    return s, {
        "energy": energy,
        "rank1_energy_share": None
        if np.sqrt(energy) <= ZERO_NORM
        else float(s[0] ** 2 / energy),
        "rank1_tail_share": None
        if np.sqrt(energy) <= ZERO_NORM
        else float(np.sum(s[1:] ** 2) / energy),
        "numerical_rank": int(np.sum(s > tol)),
        "machine_rank_tolerance": float(tol),
    }


def assignment(value, tolerance=1e-12):
    """Enumerate all named-label matchings; return diagnostic rank, never a sampling p."""
    d = finite_matrix(value)
    perms = list(itertools.permutations(range(6)))
    traces = np.asarray([sum(d[q, c] for q, c in enumerate(p)) for p in perms])
    observed = float(np.trace(d))
    return (
        perms,
        traces,
        {
            "named_trace": observed,
            "strictly_better_assignments": int(np.sum(traces > observed + tolerance)),
            "tied_assignments": int(np.sum(np.abs(traces - observed) <= tolerance)),
            "named_rank": 1 + int(np.sum(traces > observed + tolerance)),
            "total_assignments": len(perms),
            "assignment_maximum": float(traces.max()),
            "assignment_minimum": float(traces.min()),
        },
    )


def rectangles(value):
    """Compute all 225 rectangular graph cycles, not triangle curls."""
    d = finite_matrix(value)
    return [
        {
            "q0": q0,
            "q1": q1,
            "c0": c0,
            "c1": c1,
            "contrast": float(d[q0, c0] - d[q0, c1] - d[q1, c0] + d[q1, c1]),
        }
        for (q0, q1), (c0, c1) in itertools.product(
            itertools.combinations(range(6), 2), repeat=2
        )
    ]


def projection(value, reference):
    """Orthogonally describe scale versus shape; do not fit or select a predictor."""
    d, a = finite_matrix(value), finite_matrix(reference)
    e, total = float(np.sum(a * a)), float(np.sum(d * d))
    if np.sqrt(e) <= ZERO_NORM:
        return {
            "atomic_scale": None,
            "shape_remainder_rms": None,
            "shape_remainder_share": None,
        }
    alpha = float(np.sum(d * a) / e)
    remainder = d - alpha * a
    return {
        "atomic_scale": alpha,
        "shape_remainder_rms": float(np.sqrt(np.mean(remainder**2))),
        "shape_remainder_share": None
        if np.sqrt(total) <= ZERO_NORM
        else float(np.sum(remainder**2) / total),
    }


def describe(value, reference):
    """Summarize mechanism shape, specificity and exact decomposition identities."""
    d, a = finite_matrix(value), finite_matrix(reference)
    g, c = graph_split(d)
    _, spec = spectrum(d)
    _, _, named = assignment(d)
    e = float(np.sum(d**2))
    off = d[~np.eye(6, dtype=bool)]
    return (
        spec
        | named
        | projection(d, a)
        | {
            "cycle_energy_share": None
            if np.sqrt(e) <= ZERO_NORM
            else float(np.sum(c**2) / e),
            "gradient_energy_share": None
            if np.sqrt(e) <= ZERO_NORM
            else float(np.sum(g**2) / e),
            "diagonal_mean": float(np.trace(d) / 6),
            "off_diagonal_mean": float(off.mean()),
            "named_diagonal_excess": float(np.trace(d) / 6 - off.mean()),
            "positive_diagonals": int(np.sum(np.diag(d) > 0)),
            "diagonal_row_winners": int(np.sum(np.diag(d) >= d.max(axis=1) - 1e-12)),
            "gradient_cycle_inner_product": float(np.sum(g * c)),
            "energy_identity_error": float(abs(e - np.sum(g**2) - np.sum(c**2))),
            "cycle_divergence_max": float(np.max(np.abs(incidence().T @ c.ravel()))),
        }
    )


def family_comparisons(operators):
    """Compare whole operators and interactions across two fixed behavior families."""
    pairs = [("atom", operators["atom/f0"], operators["atom/f1"])]
    for r in REGIMES:
        pairs.append(
            (
                "account/" + r,
                (operators[f"account/{r}/f0/o0"] + operators[f"account/{r}/f0/o1"]) / 2,
                (operators[f"account/{r}/f1/o0"] + operators[f"account/{r}/f1/o1"]) / 2,
            )
        )
    for name, _, _ in UPDATES:
        pairs.append(
            (
                "update/" + name,
                (operators[f"update/{name}/f0/o0"] + operators[f"update/{name}/f0/o1"])
                / 2,
                (operators[f"update/{name}/f1/o0"] + operators[f"update/{name}/f1/o1"])
                / 2,
            )
        )
    return [
        {
            "comparison": name,
            "operator_cosine": cosine(a, b),
            "cycle_cosine": cosine(graph_split(a)[1], graph_split(b)[1]),
            "difference_rms": float(np.sqrt(np.mean((a - b) ** 2))),
            "diagonal_sign_matches": int(np.sum(np.diag(a) * np.diag(b) > 0)),
        }
        for name, a, b in pairs
    ]
