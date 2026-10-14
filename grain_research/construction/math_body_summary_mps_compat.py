#!/usr/bin/env python3
"""Float32 installation of the sealed HM203 body-access donor formula."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_body_summary_innovation as I
from grain_research.construction import math_relay_value_summary as V


def compatible_donor(native, blind, mode, mapping=None):
    """Compute the unchanged formula on CPU and install in the native bank dtype.

    The sealed formula intentionally computes in float64. MPS cannot construct
    such a tensor. Rounding its result into the observed float32 bank is a device
    compatibility operation, not a change of formula or a precision search.
    """
    n, b = np.asarray(native), np.asarray(blind)
    if n.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError("This companion requires the frozen float32 banks")
    donor = I.increment_donor(n, b, mode, mapping).astype(n.dtype)
    if not np.isfinite(donor).all():
        raise ValueError("Installed float32 donor must remain finite")
    return donor


def reader_batch(observer, unit, native_bank, blind_values, mode, mapping=None):
    """Replay one unchanged donor action with native p and prefix guards intact."""
    donor = compatible_donor(native_bank["values"], blind_values, mode, mapping)
    exact = I.increment_donor(native_bank["values"], blind_values, mode, mapping)
    receipt = {
        "installed_dtype": str(donor.dtype),
        "formula_dtype": str(exact.dtype),
        "cast_max_abs": float(np.max(abs(donor.astype(float) - exact))),
        "scientific_formula_changed": False,
    }
    original = V.summary_values

    def supplied(values, selected, supplied_mapping=None):
        """Supply only the new donor; never substitute the native prefix bank."""
        if selected != "self" or supplied_mapping is not None:
            raise ValueError("Explicit serial donor replay required")
        if np.asarray(values).shape != donor.shape:
            raise ValueError("Frozen donor support changed")
        return donor.copy()

    try:
        V.summary_values = supplied
        result = V.native_batch(observer, unit, native_bank, "self")
    finally:
        V.summary_values = original
    return (*result, receipt)
