#!/usr/bin/env python3
"""HM31 identical-property actions on different receiver contexts."""

from __future__ import annotations

import numpy as np

from . import math_crossed_field as C


def corners(parent_rows, pairs):
    """Cross donor and receiver within predeclared same-wording/query pairs."""
    lookup = {(r["context"], r["wording"], r["query"]): r for r in parent_rows}
    wordings = sorted({r["wording"] for r in parent_rows})
    queries = list(dict.fromkeys(r["query"] for r in parent_rows))
    rows = []
    for pair, contexts in enumerate(pairs):
        for w in wordings:
            for q in queries:
                selected = [lookup[c, w, q] for c in contexts]
                if len(selected) != 2 or selected[0]["role"] != selected[1]["role"]:
                    raise ValueError("paired contexts must have the same source role")
                ids = []
                for r in selected:
                    pos = r["spans"]["A_property"]["positions"]
                    ids.append([r["input_ids"][p] for p in pos])
                if len(ids[0]) != 4 or ids[0] != ids[1]:
                    raise ValueError("identical four-token property IDs required")
                for i in range(2):
                    for j in range(2):
                        donor, receiver = selected[i], selected[j]
                        rows.append(
                            {
                                **receiver,
                                "unit": len(rows),
                                "pair": pair,
                                "donor_side": i,
                                "receiver_side": j,
                                "donor_unit": donor["unit"],
                                "receiver_unit": receiver["unit"],
                                "diagonal": i == j,
                                "donor_context": donor["context"],
                                "property_token_ids": ids[i],
                            }
                        )
    return rows


def split(values):
    """Name product projections donor-H/receiver-c, not context/wording."""
    x = np.asarray(values, float)
    if x.shape[:2] != (2, 2):
        raise ValueError("complete two-by-two recombination required")
    parts, stats = C.decompose(x)
    rename = {"context": "donor", "wording": "receiver"}
    return (
        {rename.get(k, k): v for k, v in parts.items()},
        {
            next((k.replace(a, b) for a, b in rename.items() if k.startswith(a)), k): v
            for k, v in stats.items()
        },
    )


def contrasts(values):
    """Compare fixed H across c, and fixed c across H, in explicit units."""
    x = np.asarray(values, float)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("finite complete corners required")
    result = []
    for axis in ("receiver", "donor"):
        for fixed in range(2):
            a, b = (
                (x[fixed, 0], x[fixed, 1])
                if axis == "receiver"
                else (x[0, fixed], x[1, fixed])
            )
            delta = np.asarray(a - b)
            norm = float(np.linalg.norm(delta.reshape(-1)))
            reference = float(np.sqrt((np.sum(a * a) + np.sum(b * b)) / 2))
            result.append(
                {
                    "vary": axis,
                    "fixed_side": fixed,
                    "difference_norm": norm,
                    "reference_rms_norm": reference,
                    "relative_difference": norm / reference if reference else None,
                    "signed_difference": float(delta) if delta.ndim == 0 else None,
                }
            )
    return result


def replace_leaf(hidden, positions, receiver, replacement, gradient):
    """Audit only-property replacement; gradients address the patched post-layer leaf."""
    import torch

    if len(positions) != 4 or len(set(positions)) != 4:
        raise ValueError("four distinct property positions required")
    if min(positions) < 0 or max(positions) >= hidden.shape[1] - 1:
        raise ValueError("property must precede the query endpoint")
    live = hidden[0, positions].detach().cpu().numpy()
    receiver, replacement = np.asarray(receiver), np.asarray(replacement)
    if live.shape != receiver.shape or live.shape != replacement.shape:
        raise ValueError("exact property shapes required")
    reference_error = float(np.max(abs(live - receiver)))
    if reference_error != 0 or not np.isfinite(replacement).all():
        raise ValueError("receiver reference changed or nonfinite donor")
    changed = hidden.detach().clone()
    changed[0, positions] = torch.as_tensor(
        replacement, device=hidden.device, dtype=hidden.dtype
    )
    outside = torch.ones(hidden.shape[1], device=hidden.device, dtype=torch.bool)
    outside[positions] = False
    complement_error = float(
        torch.max(abs(changed[:, outside] - hidden[:, outside])).detach().cpu()
    )
    donor_error = float(np.max(abs(changed[0, positions].cpu().numpy() - replacement)))
    if complement_error or donor_error:
        raise ValueError("replacement changed complement or donor")
    if gradient:
        changed.requires_grad_(True)
    return changed, {
        "reference_max_error": reference_error,
        "other_states_max_error": complement_error,
        "replacement_max_error": donor_error,
    }
