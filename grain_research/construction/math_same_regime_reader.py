#!/usr/bin/env python3
"""Matched evidence regimes, singleton mixtures and display-order reading."""

import numpy as np
from scipy.special import logsumexp

from grain_research.construction.math_behavior_predictive import key_accounts

CONTEXTS = ("forward_pair", "swapped_pair", "left_only", "right_only")


def reader_parts(values):
    """Keep exchange-even, exchange-odd and singleton-mixture departures."""
    x = np.asarray(values, float)
    if x.ndim != 3 or x.shape[1:] != (4, 5) or not np.isfinite(x).all():
        raise ValueError("All four contexts and five finite target readings required")
    even, odd = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    single = (x[:, 2] + x[:, 3]) / 2
    return {"even": even, "odd": odd, "single": single, "join": even - single}


def source_parts(values):
    """Exchange the two displayed evidence blocks, not chronological human events."""
    x = np.asarray(values, float)
    if x.ndim != 3 or x.shape[1:] != (2, 5) or not np.isfinite(x).all():
        raise ValueError("Both display orders and every source target required")
    return {"even": (x[:, 0] + x[:, 1]) / 2, "odd": (x[:, 0] - x[:, 1]) / 2}


def contacts(source, reader):
    """Retain joint/order/singleton and raw cross-target products in fixed units."""
    s, r = np.asarray(source, float), np.asarray(reader, float)
    if len(s) == 0 or len(s) != len(r):
        raise ValueError("The same nonempty addressed observation set is required")
    a, b = source_parts(s), reader_parts(r)
    n = len(s)
    return {
        "join": a["even"].T @ b["join"] / n,
        "order": a["odd"].T @ b["odd"] / n,
        "single": a["even"].T @ b["single"] / n,
        "raw": (s[:, 0].T @ r[:, 0] + s[:, 1].T @ r[:, 1]) / (2 * n),
    }


def cold(packet):
    """Read all frozen token accounts without neural imports or target fitting."""
    if packet["version"] != "HM274-v1":
        raise ValueError("Unknown same-regime observation contract")
    a, b = [np.asarray(packet[k], float) for k in ("chosen", "partition")]
    lengths = np.asarray(packet["lengths"], int)
    if (
        a.shape != b.shape
        or a.shape != (48, 4, 50)
        or lengths.shape != (50,)
        or np.min(lengths) < 2
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("All48 pairs/four contexts/50 complete candidates required")
    ell = a - b
    mass = logsumexp(ell, -1)
    if np.max(mass) > 1e-3:
        raise ValueError("Nonexhaustive complete candidate probabilities exceed one")
    result = {"candidate_log_mass": mass.tolist(), "accounts": {}}
    for scale, values in (("sum", ell), ("per_token", ell / lengths)):
        result["accounts"][scale] = {}
        for key, v in key_accounts(values, packet["bank"]).items():
            if key == "imbalance":
                continue
            result["accounts"][scale][key] = {
                "values": v.tolist(),
                "parts": {k: x.tolist() for k, x in reader_parts(v).items()},
            }
    return result
