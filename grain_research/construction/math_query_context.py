#!/usr/bin/env python3
"""HM33 addressed early-query versus body-context carrier localization."""

from __future__ import annotations

import itertools

import numpy as np

AXES = ("H", "B", "P", "Q")
BITS = tuple(itertools.product(range(2), repeat=4))
LABELS = tuple(
    "".join(name for name, bit in zip(AXES, (8, 4, 2, 1), strict=True) if mask & bit)
    or "mean"
    for mask in range(16)
)


def characters():
    """Characters on the experimental Z2^4 indices, not natural vector actions."""
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
    """Retain every interaction and conditional-variance compilation bound."""
    x = np.asarray(values, float)
    if x.shape[:4] != (2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("complete finite H/B/P/Q design required")
    chars = characters()
    coef = np.tensordot(chars / 16, x.reshape((16,) + x.shape[4:]), axes=(1, 0))
    reconstructed = np.tensordot(chars.T, coef, axes=(1, 0)).reshape(x.shape)
    energies = [float(np.sum(c * c)) for c in coef]
    total, all_energy = sum(energies[1:]), float(np.sum(x * x) / 16)
    stats = {
        **{
            name + "_energy": value
            for name, value in zip(LABELS, energies, strict=True)
        },
        **{
            name + "_share": value / total if total else None
            for name, value in zip(LABELS[1:], energies[1:], strict=True)
        },
        "total_centered_energy": total,
        "mean_square_energy": all_energy,
        "energy_identity_error": abs(all_energy - sum(energies)),
        "reconstruction_error": float(np.max(abs(x - reconstructed))),
    }
    for name, omitted in (("HQ", 6), ("HPQ", 4), ("HBQ", 2)):
        bound = sum(energy for mask, energy in enumerate(energies) if mask & omitted)
        stats[name + "_only_mse_bound"] = bound
        stats[name + "_only_relative_rmse_bound"] = (
            float(np.sqrt(bound / all_energy)) if all_energy else None
        )
    return coef, stats


def contrasts(values):
    """Change one addressed carrier with the other three held exactly fixed."""
    x = np.asarray(values, float)
    if x.shape[:4] != (2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("complete finite H/B/P/Q design required")
    rows = []
    for axis, name in enumerate(AXES):
        other = [a for a in range(4) if a != axis]
        for sides in itertools.product(range(2), repeat=3):
            indices = [0] * 4
            fixed = {}
            for n, (a, side) in enumerate(zip(other, sides, strict=True)):
                indices[a] = side
                fixed.update({f"fixed{n}_axis": AXES[a], f"fixed{n}_side": side})
            left = x[tuple(indices)]
            indices[axis] = 1
            right = x[tuple(indices)]
            delta = left - right
            norm = float(np.sqrt(np.sum(delta * delta)))
            ref = float(np.sqrt((np.sum(left * left) + np.sum(right * right)) / 2))
            rows.append(
                {
                    "vary": name,
                    **fixed,
                    "difference_norm": norm,
                    "reference_rms_norm": ref,
                    "relative_difference": norm / ref if ref else None,
                    "signed_difference": float(delta) if delta.ndim == 0 else None,
                }
            )
    return rows


def positions(row):
    """Map the exact query suffix, excluding endpoint Q and property addresses."""
    start, n = row["question_start"], len(row["input_ids"])
    h = row["spans"]["A_property"]["positions"]
    if (
        not 0 < start < n - 1
        or len(h) != 4
        or len(set(h)) != 4
        or min(h) < 0
        or max(h) >= start
    ):
        raise ValueError(
            "nonoverlapping property/early-query/endpoint addresses required"
        )
    return list(range(start, n - 1))


def design(old_rows, natural_rows):
    """Independently select early-query donors without changing materials or roles."""
    lookup = {
        (
            r["pair"],
            r["wording"],
            r["query"],
            r["donor_side"],
            r["receiver_side"],
            r["query_side"],
        ): r
        for r in old_rows
    }
    rows = []
    for r in old_rows:
        pp = positions(r)
        for side in range(2):
            donor = lookup[r["pair"], r["wording"], r["query"], side, side, side]
            natural = natural_rows[donor["receiver_unit"]]
            if (
                r["role"] != natural["role"]
                or [r["input_ids"][p] for p in pp]
                != [natural["input_ids"][p] for p in positions(natural)]
                or r["input_ids"][-1] != natural["input_ids"][-1]
            ):
                raise ValueError("identical query token sequence and roles required")
            rows.append(
                {
                    **r,
                    "unit": len(rows),
                    "cube_unit": r["unit"],
                    "cube_inherited": r["inherited"],
                    "early_query_side": side,
                    "early_query_donor_unit": natural["unit"],
                    "early_query_positions": pp,
                    "early_query_token_ids": [r["input_ids"][p] for p in pp],
                    "inherited": side == r["receiver_side"],
                }
            )
    return rows


def replace_leaf(hidden, hp, pp, receiver, donors, gradient):
    """Patch mapped H/P/Q only; keep every other live body state bitwise unchanged."""
    import torch

    if (
        len(hp) != 4
        or len(set(hp)) != 4
        or not pp
        or len(set(pp)) != len(pp)
        or set(hp) & set(pp)
        or min(hp + pp) < 0
        or max(hp + pp) >= hidden.shape[1] - 1
    ):
        raise ValueError("distinct mapped H/P before endpoint Q required")
    changed = hidden.detach().clone()
    audit = {}
    for name, address in (("h", hp), ("p", pp), ("q", -1)):
        live = hidden[0, address].detach().cpu().numpy()
        if (
            live.shape != np.shape(receiver[name])
            or live.shape != np.shape(donors[name])
            or not np.isfinite(donors[name]).all()
        ):
            raise ValueError("exact finite receiver/donor shapes required")
        audit["reference_" + name + "_max_error"] = float(
            np.max(abs(live - receiver[name]))
        )
        if audit["reference_" + name + "_max_error"]:
            raise ValueError("natural receiver state changed")
        changed[0, address] = torch.as_tensor(
            donors[name], dtype=hidden.dtype, device=hidden.device
        )
        audit["replacement_" + name + "_max_error"] = float(
            np.max(abs(changed[0, address].cpu().numpy() - donors[name]))
        )
    outside = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
    outside[hp + pp + [hidden.shape[1] - 1]] = False
    if not outside.any():
        raise ValueError("nonempty text/system complement required")
    audit["B_max_error"] = float(
        torch.max(abs(changed[:, outside] - hidden[:, outside])).detach().cpu()
    )
    if any(audit.values()):
        raise ValueError("H/P/Q replacement or B preservation failed")
    if gradient:
        changed.requires_grad_(True)
    return changed, audit


def collapse_coefficients(coefficients):
    """Restrict P=B; masks differing by0110 become the same old cube character."""
    c = np.asarray(coefficients, float)
    if c.shape[0] != 16 or not np.isfinite(c).all():
        raise ValueError("sixteen finite coefficients required")
    return np.array(
        [
            c[((m & 4) << 1) | ((m & 2) << 1) | (m & 1)]
            + c[(((m & 4) << 1) | ((m & 2) << 1) | (m & 1)) ^ 6]
            for m in range(8)
        ]
    )
