#!/usr/bin/env python3
"""Post-outcome signed error account for composed evidence and paired readers."""

import numpy as np


def decompose(current, native, reader, native_reader, common):
    """Split query-contrast error into evidence, common and differential terms.

    Evidence arrays have shape [author, query, observable, hidden]. Native
    quantities are evaluation-only. These are algebraic errors, not causal
    personality variance components or an oracle application path.
    """
    current, native, reader, native_reader, common = [
        np.asarray(a, float) for a in (current, native, reader, native_reader, common)
    ]
    if (
        current.ndim != 4
        or current.shape[1] != 2
        or native.shape != current.shape
        or reader.shape != (len(current), 2, current.shape[-1])
        or native_reader.shape != reader.shape
        or common.shape != (current.shape[-1],)
        or not all(
            np.isfinite(a).all()
            for a in (current, native, reader, native_reader, common)
        )
    ):
        raise ValueError("Aligned paired evidence and reader arrays required")
    np.testing.assert_allclose(
        reader.mean(axis=1),
        np.broadcast_to(common, (len(reader), len(common))),
        atol=1e-10,
    )
    native_pair = np.einsum("nqkh,nqh->nqk", native, native_reader)
    predicted_pair = np.einsum("nqkh,nqh->nqk", current, reader)
    target, predicted = (
        native_pair[:, 0] - native_pair[:, 1],
        predicted_pair[:, 0] - predicted_pair[:, 1],
    )
    moved_pair = np.einsum("nqkh,nqh->nqk", current - native, native_reader)
    transport = moved_pair[:, 0] - moved_pair[:, 1]
    common_error = np.einsum(
        "nkh,nh->nk", current[:, 0] - current[:, 1], common - native_reader.mean(axis=1)
    )
    difference = (
        reader[:, 0] - reader[:, 1] - (native_reader[:, 0] - native_reader[:, 1])
    )
    differential = np.einsum("nkh,nh->nk", current.mean(axis=1), difference)
    pieces = np.stack((transport, common_error, differential), axis=1)
    gap = float(abs(pieces.sum(axis=1) - (predicted - target)).max())
    np.testing.assert_allclose(pieces.sum(axis=1), predicted - target, atol=1e-10)
    return {
        "target": target,
        "prediction": predicted,
        "pieces": pieces,
        "closure_max": gap,
    }


def energies(account):
    """Retain all signed cross terms; sum them to the observed total error."""
    pieces, target, predicted = [account[k] for k in ("pieces", "target", "prediction")]
    t, c, d = [pieces[:, i] for i in range(3)]
    fields = {
        "transport_error2": np.sum(t * t),
        "common_error2": np.sum(c * c),
        "differential_error2": np.sum(d * d),
        "transport_common_cross": 2 * np.sum(t * c),
        "transport_differential_cross": 2 * np.sum(t * d),
        "common_differential_cross": 2 * np.sum(c * d),
    }
    total = float(np.sum((predicted - target) ** 2))
    np.testing.assert_allclose(sum(fields.values()), total, atol=1e-10)
    other = t + c
    return {k: float(v) for k, v in fields.items()} | {
        "total_error2": total,
        "actual_energy": float(np.sum(target**2)),
        "other_error2": float(np.sum(other**2)),
        "other_differential_cross": float(2 * np.sum(other * d)),
        "oracle_native_difference_error2": float(np.sum(other**2)),
        "closure_max": account["closure_max"],
    }
