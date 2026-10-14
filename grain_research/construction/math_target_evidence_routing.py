#!/usr/bin/env python3
"""Conditional target evidence and other-profile routing in a fixed joint law."""

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_relational_diagnostic_update as P

ACCOUNTS = ("conditional_change", "profile_rerouting", "interaction")


def edges(q):
    """Return the16 low/high profile pairs for a registered target bit."""
    if type(q) is not int or q not in range(5):
        raise ValueError("A registered target bit is required")
    low = np.arange(32)[P.BITS[:, q] == 0]
    return low, low | (1 << q)


def analyze(log_profile, log_factor):
    """Disintegrate J/I posteriors without observing a current author's labels."""
    lp, d = np.asarray(log_profile, float), np.asarray(log_factor, float)
    if lp.ndim != 3 or lp.shape[1:] != (2, 32) or d.shape != (len(lp), 32):
        raise ValueError("Both complete32-profile laws and their factor required")
    if not np.isfinite(lp).all() or not np.isfinite(d).all():
        raise ValueError("Finite positive log laws required")
    if np.max(np.abs(logsumexp(lp, -1))) > 1e-12:
        raise ValueError("Registered profile laws must already be normalized")
    identity = lp[:, 0] - lp[:, 1] - d
    if np.max(np.abs(identity - identity[:, :1])) > 1e-12:
        raise ValueError("Relation factor and posterior laws are inconsistent")
    weights, conditional, odds, corners = [], [], [], []
    for q in range(5):
        low, high = edges(q)
        lc = logsumexp(np.stack((lp[:, :, low], lp[:, :, high]), -1), -1)
        w, r = np.exp(lc), np.exp(lp[:, :, high] - lc)
        # Returned corner order is II,IJ,JI,JJ while parent law order is J,I.
        c = np.stack(
            [(w[:, a] * r[:, b]).sum(-1) for a, b in ((1, 1), (1, 0), (0, 1), (0, 0))],
            -1,
        )
        weights.append(w)
        conditional.append(r)
        odds.append(d[:, high] - d[:, low])
        corners.append(c)
    c = np.stack(corners, 1)
    return {
        "profile_weights": np.stack(weights, 2),
        "conditional_high": np.stack(conditional, 2),
        "conditional_log_odds_update": np.stack(odds, 1),
        "corners": c,
        "probability_accounts": differences(c),
    }


def differences(corners):
    """Keep all three signed differences; neither cross-term is discarded."""
    c = np.asarray(corners, float)
    if c.shape[-1] != 4 or not np.isfinite(c).all():
        raise ValueError("Four finite registered corners required")
    return np.stack(
        (
            c[..., 1] - c[..., 0],
            c[..., 2] - c[..., 0],
            c[..., 3] - c[..., 1] - c[..., 2] + c[..., 0],
        ),
        -1,
    )


def scored_accounts(corners, codes):
    """Evaluate fixed corners on held SOURCE targets only,not select a reader."""
    c, g = np.asarray(corners, float), np.asarray(codes)
    if c.shape != (len(g), 5, 4) or not np.isin(g, np.arange(32)).all():
        raise ValueError("Registered held source profiles and corners required")
    if not np.isfinite(c).all() or np.any(c <= 0) or np.any(c >= 1):
        raise ValueError("All four probabilities must remain strictly inside(0,1)")
    bit = P.BITS[g.astype(int)]
    scores = np.where(bit[..., None] == 1, np.log(c), np.log1p(-c))
    return differences(scores)


def literal(log_profile, q):
    """Recount all32 entries by scalar bit matching,not the vectorized edges."""
    p = np.exp(np.asarray(log_profile, float))
    others = [j for j in range(5) if j != q]
    rows = []
    for code in range(16):
        group = [
            g
            for g in range(32)
            if sum(((g >> j) & 1) << k for k, j in enumerate(others)) == code
        ]
        w = [sum(float(p[law, g]) for g in group) for law in (0, 1)]
        r = [
            sum(float(p[law, g]) for g in group if (g >> q) & 1) / w[law]
            for law in (0, 1)
        ]
        rows.append((w, r))
    return np.array(
        [sum(w[a] * r[b] for w, r in rows) for a, b in ((1, 1), (1, 0), (0, 1), (0, 0))]
    )


def cold(bank, record):
    """Expose all conditional hypothetical profiles,not inferred known traits."""
    r = P.read(bank, P.cold_point(record, bank["prototypes"].shape[-1])[None])
    a = analyze(r["log_profile"], r["log_factor"])
    rows = []
    for q, name in enumerate(P.TARGETS):
        low, high = edges(q)
        for c in range(16):
            rows.append(
                {
                    "target": name,
                    "other_bits_low_profile_code": int(low[c]),
                    "high_profile_code": int(high[c]),
                    "hypothetical_not_observed_author_labels": True,
                    "conditional_high_J_I": a["conditional_high"][0, :, q, c].tolist(),
                    "profile_weight_J_I": a["profile_weights"][0, :, q, c].tolist(),
                    "conditional_log_odds_update": float(
                        a["conditional_log_odds_update"][0, q, c]
                    ),
                }
            )
    return {
        "frame_key": record["frame_key"],
        "targets": list(P.TARGETS),
        "probability_accounts": a["probability_accounts"][0].tolist(),
        "original_update": (r["targets"][0, 0] - r["targets"][0, 1]).tolist(),
        "conditional_profiles": rows,
    }
