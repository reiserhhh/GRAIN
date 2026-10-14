"""Finite measured-reader emission channels for the F07/F08 interface."""

import numpy as np
from scipy.special import expit


def fit_emission(rows, panel="mixed"):
    """Estimate P(reader symbol|personal Y) using source-only declared materials.

    Unqualified material polarity is ancillary and averaged over both values,
    rather than incorrectly identifying another speaker's label with personal Y.
    Symbols are missing, positive, negative. Status/variant frequencies are the
    frozen material design, not frequencies estimated from a human population.
    """
    if not rows or any(row["role"] != "source" for row in rows):
        raise ValueError("source-only material rows required")
    if panel not in ("complete", "mixed"):
        raise ValueError("declared observation panel required")
    bank = {(int(r["pole"]), r["subject_status"], int(r["variant"])): r for r in rows}
    if len(bank) != 32 or len(rows) != 32:
        raise ValueError("complete unique 32-row source grammar required")
    statuses = (
        ("actual", "negated")
        if panel == "complete"
        else ("actual", "quoted", "hypothetical", "negated")
    )
    emission = np.zeros((2, 3))
    for y in (0, 1):
        for status in statuses:
            eligible = status in ("actual", "negated")
            poles = (y,) if eligible else (0, 1)
            for pole in poles:
                for variant in range(4):
                    row = bank[(pole, status, variant)]
                    q = float(row["qualification_score"])
                    p = float(row["pole_score"])
                    if not 0 <= q <= 1 or not 0 <= p <= 1:
                        raise ValueError("finite scored probabilities required")
                    symbol = 0 if panel == "mixed" and q < 0.5 else 1 if p >= 0.5 else 2
                    emission[y, symbol] += 1 / (len(statuses) * len(poles) * 4)
    return emission


def reader_law(theta, z, emission):
    """Compose a latent Bernoulli response with arbitrary measured emissions."""
    emission = np.asarray(emission, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if (
        emission.shape != (2, 3)
        or not np.isfinite(emission).all()
        or (emission < 0).any()
        or not np.allclose(emission.sum(axis=1), 1)
    ):
        raise ValueError("two stochastic emission rows required")
    if theta.shape != (2,) or not np.isfinite(theta).all() or not np.isfinite(z):
        raise ValueError("finite theta and condition required")
    p = expit(theta[0] + theta[1] * z)
    derivative = np.outer(emission[1] - emission[0], p * (1 - p) * np.array([1, z]))
    return (1 - p) * emission[0] + p * emission[1], derivative
