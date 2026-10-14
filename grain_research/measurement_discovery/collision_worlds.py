"""Exact finite worlds exposing information lost by common measurement views."""

from __future__ import annotations

from itertools import product

import numpy as np


def c1_law():
    """Return C1 marginal and conditional probabilities for type and condition."""
    rows = []
    for typ, z in product((-1, 1), (-1, 1)):
        p = 0.5 + 0.25 * typ * z
        rows.append({"type": typ, "z": z, "p_y1": p})
    return rows


def c2_transition():
    """Return the exact second-order XOR transition matrix."""
    rows = []
    for a, b in product((0, 1), repeat=2):
        target = a ^ b
        rows.append({"a": a, "b": b, "p_next": {target: 0.75, 1 - target: 0.25}})
    return rows


def c2_triple_law(world="xor"):
    """Enumerate the probability of every ordered binary triple."""
    out = {}
    for a, b, c in product((0, 1), repeat=3):
        if world == "iid":
            probability = 1 / 8
        elif world == "xor":
            probability = 0.25 * (0.75 if c == (a ^ b) else 0.25)
        else:
            raise ValueError("world must be iid or xor")
        out[f"{a}{b}{c}"] = probability
    return out


def c2_lag3_equal_probability(world="xor"):
    """Compute P(Y_t=Y_(t-3)) for a four-step trajectory."""
    total = 0.0
    for values in product((0, 1), repeat=4):
        a, b, c, d = values
        if world == "iid":
            probability = 1 / 16
        elif world == "xor":
            probability = 0.25 * (0.75 if c == (a ^ b) else 0.25) * (
                0.75 if d == (b ^ c) else 0.25
            )
        else:
            raise ValueError("world must be iid or xor")
        total += probability * (a == d)
    return total


def c3_exposure_bounds():
    """Return two exposure/selection pairs with identical observed S rate."""
    pairs = ((0.8, 0.5), (0.4, 1.0))
    return [
        {"pi": pi, "q": q, "p_selected": pi * q, "q_bounds_given_f": (pi * q, 1.0)}
        for pi, q in pairs
    ]


def distance_chain(epsilon=1.0):
    """Return a three-point threshold chain demonstrating non-transitivity."""
    points = np.array([0.0, 0.75 * epsilon, 1.5 * epsilon])
    return {
        "points": points.tolist(),
        "ab": float(abs(points[0] - points[1])),
        "bc": float(abs(points[1] - points[2])),
        "ac": float(abs(points[0] - points[2])),
        "epsilon": epsilon,
        "transitive_at_epsilon": bool(
            abs(points[0] - points[1]) <= epsilon
            and abs(points[1] - points[2]) <= epsilon
            and abs(points[0] - points[2]) <= epsilon
        ),
    }


def exact_law(world, view="full"):
    """Expose a named exact law used by the GP01 certificate."""
    if world == "c1":
        law = c1_law()
        if view == "marginal":
            return {"p_y1": 0.5}
        return law
    if world == "c2":
        if view == "triples":
            return c2_triple_law("xor")
        if view == "iid_triples":
            return c2_triple_law("iid")
        if view == "lag3":
            return c2_lag3_equal_probability("xor")
        return c2_transition()
    if world == "c3":
        return c3_exposure_bounds()
    raise ValueError("unknown world")


def generate_records(world, design, seed, *, n_authors=None, events=48):
    """Generate synthetic observations while keeping truth separate from records."""
    rng = np.random.default_rng(seed)
    if n_authors is None:
        n_authors = {"source": 300, "validation": 100, "application": 200}
    records, truth = [], []
    index = 0
    splits = ("source", "validation", "application")
    for split in splits:
        for author_number in range(n_authors[split]):
            author_id = f"{world}_{split}_{author_number:04d}"
            typ = -1 if author_number % 2 == 0 else 1
            history = [int(rng.integers(2)) for _ in range(2)]
            latent = {"author_id": author_id, "world": world, "type": typ}
            truth.append(latent)
            for t in range(events):
                z = -1 if t % 2 == 0 else 1
                if world == "c1":
                    probability = 0.5 + 0.25 * typ * z
                    observation = int(rng.random() < probability)
                    opportunity = 1
                elif world == "c2":
                    expected = history[-2] ^ history[-1]
                    observation = int(expected ^ (rng.random() >= 0.75))
                    history.append(observation)
                    opportunity = 1
                elif world == "c3":
                    opportunity = int(rng.random() < 0.8)
                    behavior = int(rng.random() < 0.5)
                    observation = opportunity * behavior
                else:
                    raise ValueError("unknown world")
                records.append(
                    {
                        "author_id": author_id,
                        "event_id": f"{world}_e{index:07d}",
                        "episode_id": f"{author_id}_episode",
                        "t": t,
                        "condition": z,
                        "opportunity": opportunity,
                        "opportunity_status": "observed" if world == "c3" else "not_applicable",
                        "selected": observation,
                        "observation": observation,
                        "text_ref": f"synthetic://{world}/{index}",
                        "reader_id": "synthetic_identity_reader",
                        "role": split,
                        "condition_status": "assigned",
                    }
                )
                index += 1
    return records, truth
