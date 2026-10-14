#!/usr/bin/env python3
"""Joint evidence moments retain role binding that separate marginals discard.

This is a finite polynomial identity, not a new psychological theorem. Event
identity and semantic observation must already be defined before aggregation.
The continuous form reconstructs soft contribution units, not the hard-qualified
author profile or a calibrated probability of actual behavior.
"""

from __future__ import annotations

import numpy as np


def joint_moments(z):
    """Return all 16 mean square-free monomials of aligned k,p,o,a coordinates."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 4 or not len(z) or not np.isfinite(z).all():
        raise ValueError("finite nonempty event-by-four coordinates required")
    return np.array(
        [
            np.prod(z[:, [j for j in range(4) if mask & (1 << j)]], axis=1).mean()
            for mask in range(16)
        ]
    )


def contribution_from_moments(m):
    """Contract the eight polarity-containing terms of p(1+k)(1+o)(1+a)/8."""
    m = np.asarray(m, dtype=float)
    if m.shape != (16,) or not np.isfinite(m).all():
        raise ValueError("16 finite joint moments required")
    return float(sum(m[mask] for mask in range(16) if mask & 2) / 8)


def gate_mass_from_moments(m):
    """Contract the eight qualification terms; zero mass is absent evidence."""
    m = np.asarray(m, dtype=float)
    if m.shape != (16,) or not np.isfinite(m).all():
        raise ValueError("16 finite joint moments required")
    return float(sum(m[mask] for mask in range(16) if not mask & 2) / 8)


def independent_marginal_contribution(z):
    """Evaluate the independence approximation for an explicit negative control."""
    mean = np.asarray(z, dtype=float).mean(axis=0)
    return float(mean[1] * (1 + mean[0]) * (1 + mean[2]) * (1 + mean[3]) / 8)
