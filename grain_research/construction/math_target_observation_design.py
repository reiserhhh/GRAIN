#!/usr/bin/env python3
"""Target-qualified archive observation value, distinct from identity value."""

from __future__ import annotations

import numpy as np

VERSION = "HM206_target_observation_design_v1"
TARGETS = ("E", "N", "A", "C", "O")
POLICIES = (
    "target_risk",
    "target_information",
    "identity_information",
    "condition0",
    "condition1",
    "uniform",
)


def event_probabilities(cells, units):
    """Preserve distinct events and convert fixed windows to archive frequencies."""
    x, ids = np.asarray(cells), np.asarray(units)
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] < 1 or ids.shape != x.shape:
        raise ValueError("Two nonempty addressed event windows required")
    if not np.isin(x, np.arange(8)).all():
        raise ValueError("Changed eight-cell observation alphabet")
    for row in ids:
        if len(set(row.ravel().tolist())) != row.size:
            raise ValueError("Repeated canonical event is not a new observation")
    return np.eye(8)[x.astype(int)].mean(2)


def validate(p, labels):
    """Require normalized archive frequencies and all five binary source links."""
    p, y = np.asarray(p, float), np.asarray(labels, float)
    if (
        p.ndim != 3
        or p.shape[1:] != (2, 8)
        or not len(p)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(2), 1, atol=1e-12, rtol=0)
        or y.shape != (len(p), 5)
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError("Finite archive law and binary source targets required")
    return p, y


def archive_menu(p):
    """Expose late condition availability, not future fine content or counts."""
    late = np.asarray(p, float)[:, 1].reshape(-1, 2, 4)
    mass = late.sum(2)
    r = np.divide(
        late, mass[:, :, None], out=np.zeros_like(late), where=mass[:, :, None] > 0
    )
    mask = (mass > 0) @ np.array([1, 2])
    return mask, r


def entropy(probabilities):
    """Use zero-mass entropy convention without inventing support."""
    p = np.asarray(probabilities, float)
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Entropy requires finite nonnegative probabilities")
    return -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=-1)


def compile_design(probabilities, labels):
    """Construct source-only scores, observation values and all six policies."""
    p, y = validate(probabilities, labels)
    masks, r = archive_menu(p)
    codes = (y @ (2 ** np.arange(5))).astype(int)
    profiles = np.eye(32)[codes]
    shape = (4, 8)
    mass = np.zeros(shape)
    before = np.full((*shape, 5), np.nan)
    before_profile = np.full((*shape, 32), np.nan)
    after = np.full((*shape, 2, 4, 5), np.nan)
    after_profile = np.full((*shape, 2, 4, 32), np.nan)
    outcome = np.full((*shape, 2, 4), np.nan)
    value = np.full((*shape, 2, 5), np.nan)
    target_mi = np.full_like(value, np.nan)
    identity_mi = np.full((*shape, 2), np.nan)
    tower_error = 0.0
    variance_error = 0.0
    for m in (1, 2, 3):
        for a in range(8):
            raw = (masks == m) * p[:, 0, a]
            mass[m, a] = raw.sum() / len(p)
            if not raw.sum():
                continue
            w = raw / raw.sum()
            mu = w @ y
            before[m, a], before_profile[m, a] = mu, w @ profiles
            for c in range(2):
                if not m & (1 << c):
                    continue
                joint = w[:, None] * r[:, c]
                pb = joint.sum(0)
                supported = pb > 0
                post = np.divide(
                    joint.T @ y,
                    pb[:, None],
                    out=np.full((4, 5), np.nan),
                    where=pb[:, None] > 0,
                )
                post_profile = np.divide(
                    joint.T @ profiles,
                    pb[:, None],
                    out=np.full((4, 32), np.nan),
                    where=pb[:, None] > 0,
                )
                outcome[m, a, c], after[m, a, c], after_profile[m, a, c] = (
                    pb,
                    post,
                    post_profile,
                )
                v = np.sum(pb[supported, None] * (post[supported] - mu) ** 2, axis=0)
                value[m, a, c] = v
                target_mi[m, a, c] = entropy(np.stack((mu, 1 - mu), axis=-1)) - np.sum(
                    pb[supported, None]
                    * entropy(
                        np.stack((post[supported], 1 - post[supported]), axis=-1)
                    ),
                    axis=0,
                )
                ratio = np.divide(
                    r[:, c], pb[None], out=np.ones_like(joint), where=pb[None] > 0
                )
                identity_mi[m, a, c] = np.sum(
                    joint * np.log(np.where(ratio > 0, ratio, 1))
                )
                tower_error = max(
                    tower_error,
                    float(np.max(np.abs(pb[supported] @ post[supported] - mu))),
                )
                bayes_gain = mu * (1 - mu) - np.sum(
                    pb[supported, None] * post[supported] * (1 - post[supported]),
                    axis=0,
                )
                variance_error = max(
                    variance_error, float(np.max(np.abs(v - bayes_gain)))
                )
    policies = np.zeros((len(POLICIES), *shape, 5, 2))
    for m, a in np.argwhere(mass > 0):
        available = np.flatnonzero([(int(m) & (1 << c)) > 0 for c in (0, 1)])
        for q in range(5):
            picks = (
                available[np.argmax(value[m, a, available, q])],
                available[np.argmax(target_mi[m, a, available, q])],
                available[np.argmax(identity_mi[m, a, available])],
                available[0],
                available[-1],
            )
            for k, c in enumerate(picks):
                policies[k, m, a, q, c] = 1
            policies[5, m, a, q, available] = 1 / len(available)
    return {
        "mass": mass,
        "before": before,
        "before_profile": before_profile,
        "after": after,
        "after_profile": after_profile,
        "outcome": outcome,
        "value": value,
        "target_mi": target_mi,
        "identity_mi": identity_mi,
        "policies": policies,
        "tower_error": tower_error,
        "variance_error": variance_error,
    }


def evaluate(design, probabilities, labels):
    """Evaluate frozen policies on disjoint authors, preserving shared support."""
    p, y = validate(probabilities, labels)
    masks, r = archive_menu(p)
    n = len(p)
    before_risk = np.zeros((n, 5))
    after_risk = np.zeros((n, len(POLICIES), 5))
    forecast = np.zeros_like(after_risk)
    support = np.zeros(n)
    missing_pre = np.zeros(n)
    missing_outcome = np.zeros(n)
    for u, m in enumerate(masks):
        for a in range(8):
            pa = p[u, 0, a]
            if pa == 0:
                continue
            if design["mass"][m, a] == 0:
                missing_pre[u] += pa
                continue
            available = [c for c in (0, 1) if m & (1 << c)]
            if any(
                np.any((r[u, c] > 0) & (design["outcome"][m, a, c] == 0))
                for c in available
            ):
                missing_outcome[u] += pa
                continue
            support[u] += pa
            mu = design["before"][m, a]
            before_risk[u] += pa * (y[u] - mu) ** 2
            for c in available:
                b = r[u, c] > 0
                weights = design["policies"][:, m, a, :, c]
                mse = np.sum(
                    r[u, c, b, None] * (y[u] - design["after"][m, a, c, b]) ** 2, axis=0
                )
                after_risk[u] += pa * weights * mse
                forecast[u] += pa * weights * design["value"][m, a, c]
    return {
        "support": support,
        "missing_pre": missing_pre,
        "missing_outcome": missing_outcome,
        "before_risk": before_risk,
        "after_risk": after_risk,
        "gain": before_risk[:, None] - after_risk,
        "forecast": forecast,
    }


def recommend(bundle, record):
    """Cold one-step choice uses only an addressed current cell, menu and target."""
    if bundle.get("version") != VERSION or set(record) != {
        "event_id",
        "current_cell",
        "availability",
        "target",
    }:
        raise ValueError("Changed selector identity or extra input fields")
    if not isinstance(record["event_id"], str) or not record["event_id"]:
        raise ValueError("Current evidence requires a canonical address")
    a, q, menu = record["current_cell"], record["target"], record["availability"]
    if type(a) is not int or not 0 <= a < 8 or type(q) is not int or not 0 <= q < 5:
        raise ValueError("Invalid observation or target index")
    if (
        not isinstance(menu, list)
        or not menu
        or any(type(c) is not int or c not in (0, 1) for c in menu)
        or len(set(menu)) != len(menu)
    ):
        raise ValueError("Distinct nonempty available conditions required")
    m = sum(1 << c for c in menu)
    entry = bundle["states"].get(f"{m}/{a}")
    if entry is None:
        return {
            "supported": False,
            "reason": "source_state_unobserved",
            "event_id": record["event_id"],
        }
    return {
        "supported": True,
        "event_id": record["event_id"],
        "target": TARGETS[q],
        "condition_probabilities": {
            name: entry["policies"][i][q] for i, name in enumerate(POLICIES)
        },
        "source_before": entry["before"][q],
        "source_expected_gain": [
            entry["value"][c][q] if c in menu else None for c in (0, 1)
        ],
    }


def bundle(design):
    """Export an aggregate selector without author addresses or target catalogues."""
    states = {}
    for m, a in np.argwhere(design["mass"] > 0):
        value = design["value"][m, a]
        states[f"{m}/{a}"] = {
            "before": design["before"][m, a].tolist(),
            "value": [
                [float(v) if np.isfinite(v) else None for v in row] for row in value
            ],
            "policies": design["policies"][:, m, a].tolist(),
        }
    return {
        "version": VERSION,
        "targets": list(TARGETS),
        "policies": list(POLICIES),
        "states": states,
        "scope": "one_step_retrospective_archive_selection_not_prompt_design_or_human_causal_effect",
    }
