"""Finite rival-process and observational-equivalence worlds for F09."""

from __future__ import annotations

import hashlib

import numpy as np


def logistic(value: float) -> float:
    """Return a stable logistic probability."""
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def family_law(z: tuple[float, float, float], family: str, beta: float) -> float:
    """Return the declared main-family probability."""
    if family == "M1":
        score = z[0]
    elif family == "M2":
        score = z[1]
    elif family == "M3":
        score = z[2]
    elif family == "M4":
        score = (z[0] + z[1] + z[2]) / np.sqrt(3.0)
    elif family == "belief":
        score = z[0]
    elif family == "audience":
        score = z[1]
    else:
        raise ValueError(f"unknown family {family}")
    return logistic(beta * score)


def opaque_author(seed: int, role: str, index: int) -> str:
    """Create opaque author identifiers."""
    return "u_" + hashlib.sha256(f"{seed}|{role}|{index}".encode()).hexdigest()[:16]


def generate_records(seed: int, *, counts: dict[str, int] | None = None, events: int = 24) -> tuple[list[dict], list[dict]]:
    """Generate nine main families and two process-world truth labels."""
    counts = counts or {"source": 60, "validation": 20, "application": 40}
    rng = np.random.default_rng(seed); rows, truth = [], []
    for role in ("source", "validation", "application"):
        for number in range(counts[role]):
            author = opaque_author(seed, role, number); family = ("M1", "M2", "M3")[number % 3]; beta = float(rng.choice([.5, 1.0, 1.5])); gamma = 1.2
            truth.append({"author_id": author, "role": role, "family": family, "beta": beta, "gamma": gamma})
            for t in range(events):
                z = tuple(float(v) for v in ((-1 if t % 2 == 0 else 1), (-1 if t % 3 == 0 else 1), (-1 if t % 4 < 2 else 1)))
                p = family_law(z, family, beta); y = int(rng.random() < p)
                rows.append({"schema_version": "grain-gptpro-record-v3", "author_id": author, "role": role, "event_id": f"f09_{seed}_{role}_{number}_{t:04d}", "t": t, "z": z, "y": y, "split": "even" if t % 2 == 0 else "odd"})
    return rows, truth
