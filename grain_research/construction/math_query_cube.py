#!/usr/bin/env python3
"""HM32 Fourier characters on the finite H/R/Q intervention design."""

from __future__ import annotations

import itertools

import numpy as np

BITS = tuple(itertools.product(range(2), repeat=3))
LABELS = ("mean", "Q", "R", "RQ", "H", "HQ", "HR", "HRQ")


def cube(values):
    """Require the entire cube, not impute missing off-diagonal observations."""
    x = np.asarray(values, float)
    if x.shape[:3] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("finite complete H/R/Q cube required")
    return x


def characters():
    """Return exact +/-1 characters of the experimental index group Z2^3."""
    return np.array(
        [
            [
                (-1) ** sum(a * b for a, b in zip(mask, point, strict=True))
                for point in BITS
            ]
            for mask in BITS
        ],
        float,
    )


def split(values):
    """Parseval decomposition and optimistic conditional action bounds."""
    x = cube(values)
    y = x.reshape((8,) + x.shape[3:])
    chars = characters()
    coefficients = np.tensordot(chars / 8, y, axes=(1, 0))
    reconstructed = np.tensordot(chars.T, coefficients, axes=(1, 0)).reshape(x.shape)
    energies = {
        label: float(np.sum(c * c))
        for label, c in zip(LABELS, coefficients, strict=True)
    }
    total = sum(v for k, v in energies.items() if k != "mean")
    all_energy = float(np.sum(x * x) / 8)
    h_only = sum(energies[k] for k in ("Q", "R", "RQ", "HQ", "HR", "HRQ"))
    hq_only = sum(energies[k] for k in ("R", "RQ", "HR", "HRQ"))
    hr_only = sum(energies[k] for k in ("Q", "RQ", "HQ", "HRQ"))
    stats = {
        **{k + "_energy": v for k, v in energies.items()},
        **{
            k + "_share": v / total if total else None
            for k, v in energies.items()
            if k != "mean"
        },
        "total_centered_energy": total,
        "mean_square_energy": all_energy,
        "energy_identity_error": abs(all_energy - sum(energies.values())),
        "reconstruction_error": float(np.max(abs(x - reconstructed))),
        "H_only_mse_bound": h_only,
        "HQ_only_mse_bound": hq_only,
        "HR_only_mse_bound": hr_only,
    }
    for k, v in (("H", h_only), ("HQ", hq_only), ("HR", hr_only)):
        stats[k + "_only_relative_rmse_bound"] = (
            float(np.sqrt(v / all_energy)) if all_energy else None
        )
    return coefficients, stats


def contrasts(values):
    """Compare one factor while holding the other two exactly fixed."""
    x = cube(values)
    rows = []
    for axis, name in enumerate(("H", "R", "Q")):
        other = [a for a in range(3) if a != axis]
        for sides in itertools.product(range(2), repeat=2):
            indices = [0, 0, 0]
            for a, s in zip(other, sides, strict=True):
                indices[a] = s
            left = x[tuple(indices)]
            indices[axis] = 1
            right = x[tuple(indices)]
            delta = left - right
            norm = float(np.sqrt(np.sum(delta * delta)))
            ref = float(np.sqrt((np.sum(left * left) + np.sum(right * right)) / 2))
            rows.append(
                {
                    "vary": name,
                    "fixed0_axis": ("H", "R", "Q")[other[0]],
                    "fixed0_side": sides[0],
                    "fixed1_axis": ("H", "R", "Q")[other[1]],
                    "fixed1_side": sides[1],
                    "difference_norm": norm,
                    "reference_rms_norm": ref,
                    "relative_difference": norm / ref if ref else None,
                    "signed_difference": float(delta) if np.ndim(delta) == 0 else None,
                }
            )
    return rows


def design(parent_rows, natural_rows):
    """Independently vary query donor while preserving original paired materials."""
    old = {
        (r["pair"], r["wording"], r["query"], r["donor_side"], r["receiver_side"]): r
        for r in parent_rows
    }
    rows = []
    for r in parent_rows:
        for side in range(2):
            qrow = old[r["pair"], r["wording"], r["query"], side, side]
            query_unit = qrow["receiver_unit"]
            qnatural = natural_rows[query_unit]
            if (
                r["input_ids"][r["question_start"] :]
                != qnatural["input_ids"][qnatural["question_start"] :]
            ):
                raise ValueError("identical query suffix bytes/tokens required")
            rows.append(
                {
                    **r,
                    "unit": len(rows),
                    "fiber_unit": r["unit"],
                    "query_side": side,
                    "query_donor_unit": query_unit,
                    "query_endpoint_token_id": r["input_ids"][-1],
                    "inherited": side == r["receiver_side"],
                }
            )
    return rows


def replace_leaf(hidden, positions, receiver_h, receiver_q, donor_h, donor_q, gradient):
    """Change only H and endpoint Q, then differentiate on their postpatch state."""
    import torch

    if (
        len(positions) != 4
        or len(set(positions)) != 4
        or min(positions) < 0
        or max(positions) >= hidden.shape[1] - 1
    ):
        raise ValueError("four distinct property positions before endpoint required")
    live_h = hidden[0, positions].detach().cpu().numpy()
    live_q = hidden[0, -1].detach().cpu().numpy()
    if (
        live_h.shape != np.shape(receiver_h)
        or live_h.shape != np.shape(donor_h)
        or live_q.shape != np.shape(receiver_q)
        or live_q.shape != np.shape(donor_q)
    ):
        raise ValueError("exact H/Q shapes required")
    audit = {
        "reference_h_max_error": float(np.max(abs(live_h - receiver_h))),
        "reference_q_max_error": float(np.max(abs(live_q - receiver_q))),
    }
    if (
        any(audit.values())
        or not np.isfinite(donor_h).all()
        or not np.isfinite(donor_q).all()
    ):
        raise ValueError("natural receiver changed or nonfinite donor")
    changed = hidden.detach().clone()
    changed[0, positions] = torch.as_tensor(
        donor_h, dtype=hidden.dtype, device=hidden.device
    )
    changed[0, -1] = torch.as_tensor(donor_q, dtype=hidden.dtype, device=hidden.device)
    outside = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
    outside[positions] = False
    outside[-1] = False
    if not outside.any():
        raise ValueError("nonempty receiver complement required")
    audit.update(
        {
            "R_max_error": float(
                torch.max(abs(changed[:, outside] - hidden[:, outside])).detach().cpu()
            ),
            "replacement_h_max_error": float(
                np.max(abs(changed[0, positions].cpu().numpy() - donor_h))
            ),
            "replacement_q_max_error": float(
                np.max(abs(changed[0, -1].cpu().numpy() - donor_q))
            ),
        }
    )
    if any(audit.values()):
        raise ValueError("H/Q replacement or R preservation failed")
    if gradient:
        changed.requires_grad_(True)
    return changed, audit
