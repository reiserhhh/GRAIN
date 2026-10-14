"""Finite openness/reader-channel curves for F07."""

from __future__ import annotations

import numpy as np


def logistic(value: float) -> float:
    """Return a stable logistic probability."""
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def channel_prob(theta: float, level: float, world: str) -> float:
    """Return a predeclared binary channel probability."""
    if world == "U":
        score = theta + level
    elif world == "D":
        score = theta - level
    elif world == "P":
        score = theta * level
    elif world == "flat":
        return 0.5
    else:
        raise ValueError(f"unknown channel world {world}")
    return logistic(score)


def channel_derivative(theta: float, level: float, world: str) -> float:
    """Return the analytic derivative with respect to theta."""
    p = channel_prob(theta, level, world)
    multiplier = level if world == "P" else 1.0
    return float(multiplier * p * (1.0 - p)) if world != "flat" else 0.0


def fisher_information(theta: float, levels: list[float], world: str) -> float:
    """Return Bernoulli Fisher information in theta."""
    return float(sum((channel_derivative(theta, level, world) ** 2) / max(channel_prob(theta, level, world) * (1.0 - channel_prob(theta, level, world)), 1e-12) for level in levels))


def simulate_channel(theta: float, levels: list[float], world: str, seed: int) -> list[dict]:
    """Generate response-level observations without latent label leakage."""
    rng = np.random.default_rng(seed)
    return [{"level": level, "probability": channel_prob(theta, level, world), "response": int(rng.random() < channel_prob(theta, level, world))} for level in levels]
