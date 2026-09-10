#!/usr/bin/env python3
"""Source-only read-in replacement and finite BODY/context interaction accounting."""

from __future__ import annotations

import numpy as np


def other_source_last(bank, index, query, queries):
    """Average only the other three source authors under the same fixed query."""
    if set(bank) != {(i, q) for i in range(4) for q in queries}:
        raise ValueError("Exactly four paired-query sources required")
    if index not in range(4) or query not in queries:
        raise ValueError("Known source and query required")
    members = [i for i in range(4) if i != index]
    rows = [np.asarray(bank[i, query], float) for i in members]
    if any(
        x.ndim != 1 or x.shape != rows[0].shape or not np.isfinite(x).all()
        for x in rows
    ):
        raise ValueError("Aligned finite last states required")
    return np.mean(rows, axis=0), members


def replace_last(state, replacement, positions):
    """Replace exactly the final H16 row, retaining original dtype and BODY bytes."""
    x, last, pos = np.asarray(state), np.asarray(replacement), np.asarray(positions)
    if (
        x.ndim != 2
        or last.shape != x.shape[1:]
        or len(x) < 2
        or pos.ndim != 1
        or not len(pos)
        or pos.dtype.kind not in "iu"
        or np.any(np.diff(pos) <= 0)
        or pos[0] < 0
        or pos[-1] >= len(x) - 1
        or not np.isfinite(x).all()
        or not np.isfinite(last).all()
    ):
        raise ValueError("Finite full state, last and nonfinal ordered BODY required")
    changed = x.copy()
    changed[-1] = last
    np.testing.assert_array_equal(changed[:-1], x[:-1])
    np.testing.assert_array_equal(changed[pos], x[pos])
    return changed


def account(original, changed, directions, basis):
    """Decompose finite gradient change and contract the ORIGINAL fixed directions."""
    g, gp, ds, u = (
        np.asarray(x, float) for x in (original, changed, directions, basis)
    )
    if (
        g.ndim != 2
        or gp.shape != g.shape
        or ds.shape != (4, *g.shape)
        or u.ndim != 2
        or u.shape[0] != g.shape[1]
        or any(not np.isfinite(x).all() for x in (g, gp, ds, u))
    ):
        raise ValueError("Aligned finite field/direction/source-basis required")
    np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-10)
    delta = gp - g
    inside = (delta @ u) @ u.T
    outside = delta - inside
    energies = [float(np.sum(x * x)) for x in (g, delta, inside, outside)]
    if energies[0] <= 0:
        raise ValueError("Nonzero original BODY gradient required")
    tau, newtau, dtau, itau, otau = [
        np.einsum("th,fth->f", x, ds) for x in (g, gp, delta, inside, outside)
    ]
    record = {
        "field_change_ratio": float(np.sqrt(energies[1] / energies[0])),
        "delta_field_energy": energies[1],
        "inside_field_energy": energies[2],
        "outside_field_energy": energies[3],
        "inside_field_fraction": energies[2] / energies[1] if energies[1] else 0.0,
        "energy_relative_error": abs(energies[1] - energies[2] - energies[3])
        / max(energies[1], 1e-30),
        "action_closure_error": float(np.max(abs(dtau - itau - otau))),
        "finite_action_error": float(np.max(abs(newtau - tau - dtau))),
    }
    actions = {
        "tau": tau,
        "changed_tau": newtau,
        "delta_tau": dtau,
        "inside_delta_tau": itau,
        "outside_delta_tau": otau,
    }
    return record, actions


def native_body_gradient(model, state, kwargs, ids, contrast, positions):
    """One actual suffix17..end forward/backward; fixed logit contrast, not text generation."""
    import torch

    from .selection_endpoint import endpoint

    with torch.enable_grad():
        x = state.detach().requires_grad_(True)
        value = endpoint(model, x, kwargs, ids, contrast, block=17)
        (gradient,) = torch.autograd.grad(value, (x,))
    return (
        float(value.detach().cpu()),
        gradient[0, positions].detach().float().cpu().numpy(),
    )
