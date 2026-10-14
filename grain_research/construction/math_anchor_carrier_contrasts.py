#!/usr/bin/env python3
"""Paired source-anchor contacts under fixed cached mechanism transformations."""

import numpy as np

from grain_research.construction import math_same_event_multitarget as N

MODES = (
    "native",
    "zero_increment",
    "common_increment",
    "foreign_increment",
    "common_value",
    "foreign_value",
)
CONTRASTS = (
    "recipient_baseline",
    "own_increment",
    "common_baseline",
    "foreign_increment_effect",
    "matched_increment",
    "common_increment_effect",
)
CHANNELS = MODES + CONTRASTS
PAIRS = ((3, 5), (0, 1), (2, 4), (3, 1), (0, 3), (2, 1))


def contrast_matrix():
    """Return fixed signed mode differences; no outcome-dependent weights."""
    c = np.eye(6)
    differences = np.zeros((len(PAIRS), 6))
    for row, (positive, negative) in enumerate(PAIRS):
        differences[row, positive], differences[row, negative] = 1, -1
    return np.concatenate((c, differences))


def accounts(mode_actions):
    """Keep all six modes plus signed contrasts and common/specific readouts."""
    a = N.finite(mode_actions, 5)
    if a.shape[1:] != (2, 6, 5, 4) or len(a) < 2:
        raise ValueError(
            "Matched authors x2 views x6 modes x5 targets x4 units required"
        )
    channels = np.einsum("cm,uvmqt->uvcqt", contrast_matrix(), a)
    common, specific = N.target_account(channels)
    return np.stack((channels, common, specific), axis=3)


def projections(source_means, vector_actions, response):
    """Project paired source vectors into fixed author action covariances."""
    h, dx, r = (
        N.finite(source_means, 3),
        N.finite(vector_actions, 3),
        N.finite(response, 6),
    )
    if (
        h.shape[1] != 2
        or dx.shape[1:] != (2, h.shape[2])
        or r.shape != (len(dx), 2, len(CHANNELS), 3, 5, 4)
    ):
        raise ValueError("SOURCE/panel features, views and readout addresses mismatch")
    cov = np.empty((len(CHANNELS), 3, 4, 3, 5, h.shape[2]))
    for channel, account, unit, view in np.ndindex(len(CHANNELS), 3, 4, 3):
        x = dx[:, view] if view < 2 else dx.mean(1)
        y = (
            r[:, view, channel, account, :, unit]
            if view < 2
            else r[:, :, channel, account, :, unit].mean(1)
        )
        cov[channel, account, unit, view] = N.covariances(x, y).T
    return np.einsum("ibd,mcuvqd->ibmcuvq", h, cov)


def primary_columns(projected, target):
    """Read the two registered contrasts, never independent SOURCE replicates."""
    t = N.finite(projected, 7)
    if t.shape[1:] != (2, len(CHANNELS), 3, 4, 3, 5) or target not in range(5):
        raise ValueError("Complete projected accounts and a target0..4 required")
    return t[:, :, 6:8, 2, 2, 2, target].mean(1)


def telescoping_error(response):
    """Verify bookkeeping identities without assuming additive neural mediation."""
    a = N.finite(response, 6)
    if a.shape[2:] != (len(CHANNELS), 3, 5, 4):
        raise ValueError("Complete mode/contrast response accounts required")
    return max(
        float(np.max(abs(a[:, :, 0] - a[:, :, 5] - a[:, :, 6] - a[:, :, 10]))),
        float(np.max(abs(a[:, :, 0] - a[:, :, 1] - a[:, :, 7]))),
        float(np.max(abs(a[:, :, 7] - a[:, :, 10] - a[:, :, 9]))),
        float(np.max(abs(a[:, :, 7] - (a[:, :, 0] - a[:, :, 2]) - a[:, :, 11]))),
    )
