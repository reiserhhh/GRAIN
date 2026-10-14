"""Fixed-condition rival response families for GP04."""

from __future__ import annotations

from itertools import product

import numpy as np

FAMILIES = ("M1_axis_1", "M2_axis_2", "M3_axis_3")
BETAS = (0.5, 1.0, 1.5)
CORNERS = tuple(product((-1, 1), repeat=3))


def sigmoid(value):
    """Numerically stable logistic function."""
    value = np.asarray(value, dtype=float)
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def response_probability(family, beta, condition):
    """Return a candidate binary response probability for one fixed condition."""
    z = np.asarray(condition, dtype=float)
    if z.shape != (3,) or not np.isfinite(z).all():
        raise ValueError("condition must contain three finite axes")
    if family in FAMILIES:
        axis = FAMILIES.index(family)
        value = float(beta) * z[axis]
    elif family == "M4_mixed":
        value = float(beta) * np.sum(z) / np.sqrt(3)
    elif family == "baseline":
        return 0.5
    else:
        raise ValueError("unknown family")
    return float(sigmoid(value))


def posterior_predictive(history, candidates, condition):
    """Compute posterior predictive probability under a uniform fixed rival set."""
    log_weights = []
    for family, beta in candidates:
        value = 0.0
        for row in history:
            p = response_probability(family, beta, row["condition"])
            value += row["observation"] * np.log(p) + (1 - row["observation"]) * np.log1p(-p)
        log_weights.append(value)
    log_weights = np.asarray(log_weights)
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    probability = sum(
        weight * response_probability(family, beta, condition)
        for weight, (family, beta) in zip(weights, candidates, strict=True)
    )
    return float(probability), weights


def generate_records(seed, n_authors, events=24, include_m4=False):
    """Generate author-level fixed-condition records for the rival audit."""
    rng = np.random.default_rng(seed)
    records, truth = [], []
    index = 0
    for split, count in n_authors.items():
        for author_number in range(count):
            if include_m4 and split == "application" and author_number % 4 == 3:
                family = "M4_mixed"
            else:
                family = FAMILIES[author_number % len(FAMILIES)]
            beta = BETAS[author_number % len(BETAS)]
            author = f"rival_s{seed}_{split}_{author_number:04d}"
            truth.append({"author_id": author, "family": family, "beta": beta, "role": split})
            for t in range(events):
                condition = list(CORNERS[t % len(CORNERS)])
                p = response_probability(family, beta, condition)
                observation = int(rng.random() < p)
                records.append({
                    "author_id": author,
                    "event_id": f"rival_s{seed}_e{index:07d}",
                    "episode_id": f"{author}_episode",
                    "t": t,
                    "condition": condition,
                    "opportunity": 1,
                    "opportunity_status": "observed",
                    "selected": observation,
                    "observation": observation,
                    "text_ref": f"synthetic://rival/{index}",
                    "reader_id": "synthetic_condition_reader",
                    "role": split,
                    "condition_status": "assigned",
                })
                index += 1
    return records, truth
