"""Cold-construction primitives for two finite, non-psychological targets."""

from __future__ import annotations

import numpy as np


def affine_standardize(value, mean, scale):
    """Apply a frozen source reference; return null for unsupported values."""
    if value is None or not np.isfinite(value) or not np.isfinite([mean, scale]).all() or scale <= 0:
        return None
    return float((value - mean) / scale)


def transport_standardization(mean, scale, multiplier, offset):
    """Transport a reference under x'=multiplier*x+offset exactly when positive."""
    if multiplier <= 0 or not np.isfinite([mean, scale, multiplier, offset]).all():
        raise ValueError("positive finite affine multiplier required")
    return float(multiplier * mean + offset), float(multiplier * scale)


def theta_response(positive_mean, negative_mean):
    """Compute a two-condition response contrast, not a personality trait."""
    if not np.isfinite([positive_mean, negative_mean]).all():
        raise ValueError("finite condition means required")
    return float((positive_mean - negative_mean) / 2)


def next_observation_probability(previous_a, previous_b, source_table):
    """Read a frozen four-cell conditional table from an observed prefix."""
    if int(previous_a) not in (0, 1) or int(previous_b) not in (0, 1):
        raise ValueError("binary observed prefix required")
    table = np.asarray(source_table, dtype=float)
    if table.shape != (4,) or not np.isfinite(table).all() or np.any((table <= 0) | (table >= 1)):
        raise ValueError("bounded four-cell source table required")
    return float(table[2 * int(previous_a) + int(previous_b)])


def make_bundle(target_id, formula, unit, reference, reader_id, support):
    """Build a serializable target bundle with explicit interpretation limits."""
    if not target_id or not formula or not unit or not reader_id:
        raise ValueError("target bundle fields cannot be empty")
    return {
        "target_id": target_id,
        "formula": formula,
        "unit": unit,
        "reference": reference,
        "reader_id": reader_id,
        "support": support,
        "evidence_level": "SIM/TEXT_SYNTHETIC/READER",
        "interpretation": "finite conditional response or next-observation target; not a latent psychological label",
    }
