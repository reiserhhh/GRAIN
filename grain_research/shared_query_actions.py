#!/usr/bin/env python3
"""Finite component exchanges distinguish common evidence from query readout."""

from itertools import combinations, product

import numpy as np

REGIONS = ("C", "P", "Q")


def worlds(spec):
    """Build the full balanced two-evidence world without outcome-based selection."""
    result = []
    for frame, (owner, prefix), a, d in product(
        spec["frames"], spec["owners"].items(), range(2), range(2)
    ):
        body = frame["arousal"][a] + " " + frame["dejection"][d]
        result.append(
            {
                "id": f"{frame['id']}_{owner}_{a}{d}",
                "frame": frame["id"],
                "owner": owner,
                "arousal": a,
                "dejection": d,
                "body": prefix + '"' + body + '" ' + spec["unrelated_anchor"],
            }
        )
    return result


def region_masks(question):
    """Partition every boundary row into common context, question, or final state."""
    p = np.asarray(question, bool)
    if p.ndim != 1 or len(p) < 3 or p[-1] or not p.any() or p[:-1].all():
        raise ValueError("Require nonempty C/P and a separate final Q")
    q = np.zeros(len(p), bool)
    q[-1] = True
    return {"C": ~(p | q), "P": p, "Q": q}


def exchange(base, donor, question, mask):
    """Exchange complete declared groups without mutating either source array."""
    x, y = np.asarray(base), np.asarray(donor)
    regions = region_masks(question)
    if x.shape != y.shape or x.ndim != 2 or len(x) != len(question):
        raise ValueError("Matched token positions and hidden dimensions required")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or mask not in range(8):
        raise ValueError("Finite states and an integer three-bit mask required")
    out = x.copy()
    for i, region in enumerate(REGIONS):
        if mask & (1 << i):
            out[regions[region]] = y[regions[region]]
    return out


def cube_terms(values):
    """Return all finite interactions and exact equal-order carrier allocations."""
    f = np.asarray(values, float)
    if f.shape != (8,) or not np.isfinite(f).all():
        raise ValueError("Eight finite component cells required")
    interaction = {}
    for mask in range(1, 8):
        bits = [i for i in range(3) if mask & (1 << i)]
        total = 0.0
        for size in range(len(bits) + 1):
            for subset in combinations(bits, size):
                index = sum(1 << b for b in subset)
                total += (-1) ** (len(bits) - size) * f[index]
        interaction[mask] = total
    result = {"total": float(f[7] - f[0])}
    for mask, value in interaction.items():
        name = "".join(r for i, r in enumerate(REGIONS) if mask & (1 << i))
        result["I_" + name] = float(value)
    for i, region in enumerate(REGIONS):
        result["phi_" + region] = float(
            sum(
                value / mask.bit_count()
                for mask, value in interaction.items()
                if mask & (1 << i)
            )
        )
    result["sum_error"] = float(
        abs(result["total"] - sum(result["phi_" + r] for r in REGIONS))
    )
    return result


def make_edges(design):
    """Pair low/high in one factor, holding query, frame, owner and other factor fixed."""
    lookup = {
        (d["frame"], d["owner"], d["query"], d["arousal"], d["dejection"]): d
        for d in design
    }
    if len(lookup) != len(design):
        raise ValueError("Duplicate input identity")
    edges = []
    for base in design:
        for factor in ("arousal", "dejection"):
            if base[factor]:
                continue
            a, d = base["arousal"], base["dejection"]
            donor = lookup[
                (
                    base["frame"],
                    base["owner"],
                    base["query"],
                    1 if factor == "arousal" else a,
                    1 if factor == "dejection" else d,
                )
            ]
            if (
                len(base["ids"]) != len(donor["ids"])
                or base["question_mask"] != donor["question_mask"]
            ):
                raise ValueError("Natural donor changed position geometry")
            p = np.asarray(base["question_mask"], bool)
            if not np.array_equal(
                np.asarray(base["ids"])[p], np.asarray(donor["ids"])[p]
            ):
                raise ValueError("Query token identities changed")
            edges.append(
                {
                    "index": len(edges),
                    "base": base["index"],
                    "donor": donor["index"],
                    "factor": factor,
                    "other_level": d if factor == "arousal" else a,
                    **{k: base[k] for k in ("frame", "owner", "query")},
                }
            )
    return edges
