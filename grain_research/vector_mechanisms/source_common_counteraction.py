#!/usr/bin/env python3
"""Source-only query-common local readouts and exact error accounting."""

import numpy as np

from .compiled_gated_readout import mean_readout
from .compiled_query_transfer import QUERIES, validate_sources
from .gated_update import apply_coefficients, coefficients, rms_direction


def select_contexts(bank, cohort, index):
    """Select both queries for each independent source; never average in held author."""
    slots = list(zip(bank["index"].tolist(), bank["query"].tolist(), strict=True))
    if len(slots) != 8 or set(slots) != {(i, q) for i in range(4) for q in QUERIES}:
        raise ValueError("Exact paired4-source bank required")
    sources = [i for i in range(4) if cohort != "source4" or i != index]
    validate_sources(cohort, index, sources)
    x = np.asarray(bank["x"], float)
    if x.ndim != 2 or len(x) != 8 or not np.isfinite(x).all():
        raise ValueError("Finite aligned source contexts required")
    keep = [k for k, (i, _) in enumerate(slots) if i in sources]
    return x[keep], sources


def compile_readers(source_x, shared, weights):
    """Compile both shared-reader modes through the paired mean local Jacobian."""
    x, w = np.asarray(source_x, float), np.asarray(shared, float)
    if x.ndim != 2 or not len(x) or w.shape != (2, x.shape[-1]):
        raise ValueError("Nonempty source bank and two shared readers required")
    return np.stack([mean_readout(x, t, weights) for t in w])


def apply_readers(v, d):
    """Deployment-shaped step: only two compiled vectors and current incoming d."""
    v, d = np.asarray(v, float), np.asarray(d, float)
    if v.ndim != 2 or v.shape[0] != 2 or d.shape != (2, 4, v.shape[-1]):
        raise ValueError("Two modes, two queries, four aligned directions required")
    if not np.isfinite(v).all() or not np.isfinite(d).all():
        raise ValueError("Finite inputs required")
    actions = np.einsum("mh,qfh->qmf", v, d)
    return actions, actions[0] - actions[1]


def direct_actions(source_x, shared, d, weights):
    """Independent forward-Jacobian computation for the compilation identity."""
    updates = np.mean(
        [
            apply_coefficients(
                coefficients(x, weights), rms_direction(x, d, weights), weights
            ).sum(0)
            for x in source_x
        ],
        axis=0,
    )
    return np.einsum("mh,qfh->qmf", shared, updates)


def errors(predicted, carried, operator, native):
    """Split approximation error; native floating-point parity is a separate third term."""
    p, c, o, n = [np.asarray(x, float) for x in (predicted, carried, operator, native)]
    if (
        not (p.shape == c.shape == o.shape == n.shape)
        or not np.isfinite([p, c, o, n]).all()
    ):
        raise ValueError("Matched finite action arrays required")
    return {
        "source_error": p - c,
        "missing_operator": -o,
        "parity_remainder": c + o - n,
        "total_error": p - n,
    }
