#!/usr/bin/env python3
"""Fixed-operation context contrasts, with no output-selected intervention."""

from __future__ import annotations

import itertools

import numpy as np

from .math_construct_routing import token_positions
from .math_source_relational_transport import source_delta

CLASSES = ("own", "other_report", "unrelated", "roster")
ARMS = ("exact_suffix", "source_full", "source_bound", "wrong_person", "random", "noop")


def documents(spec):
    """Construct the complete new family; control layouts have no forced true score."""
    rows = []
    for target, kind, direction in itertools.product(spec["targets"], CLASSES, (0, 1)):
        body = "This account reports actual actions by two anonymous people.\n"
        for actor in ("A", "B") if direction == 0 or kind != "roster" else ("B", "A"):
            polarity = direction if actor == "A" else 1 - direction
            if kind == "own":
                action = spec["new_behaviours"][target][polarity]
            elif kind == "other_report":
                other = "C" if actor == "A" else "D"
                action = (
                    f"reported that Person {other} "
                    + spec["new_behaviours"][target][polarity]
                )
            elif kind == "unrelated":
                action = spec["unrelated_actions"][polarity]
            else:
                action = "was included in the roster"
            body += f"Person {actor} {action}.\n"
        body += "End of account."
        rows.append(
            {
                "doc_id": len(rows),
                "content": target,
                "class": kind,
                "direction": direction,
                "body": body,
            }
        )
    return rows


def partition(row, encoded):
    """Locate a complete suffix using actual offsets, without semantic token labels."""
    if (
        list(encoded["input_ids"]) != row["input_ids"]
        or row["rendered"].count(row["body"]) != 1
    ):
        raise ValueError("Actual full input and unique body required")
    start = row["rendered"].index(row["body"])
    body = token_positions(encoded["offset_mapping"], start, start + len(row["body"]))
    if not body or body != list(range(body[0], body[-1] + 1)):
        raise ValueError("Contiguous body required")
    suffix = list(range(body[-1] + 1, len(row["input_ids"])))
    if not suffix:
        raise ValueError("Nonempty suffix required")
    return {
        "prefix": list(range(body[0])),
        "body": body,
        "suffix": suffix,
        "length": len(row["input_ids"]),
    }


def prove_pair(receiver, donor):
    """Allow natural body changes while requiring identical complete query token IDs."""
    if any(receiver[k] != donor[k] for k in ("content", "query", "person", "class")):
        raise ValueError("Wrong context, actor or query donor")
    if receiver["direction"] == donor["direction"]:
        raise ValueError("Opposite predetermined directions required")
    p, q = receiver["partition"]["suffix"], donor["partition"]["suffix"]
    if [receiver["input_ids"][i] for i in p] != [donor["input_ids"][i] for i in q]:
        raise ValueError("Actual complete suffix token IDs differ")
    return {"receiver_positions": p, "donor_positions": q}


def patch_values(entry, current, donor, sectors, seed):
    """Apply a fixed source change or a matched donor without shape interpolation."""
    arm = entry["arm"]
    p, q = entry["partition"]["suffix"], entry["mapping"]["donor_positions"]
    if arm not in ARMS:
        raise ValueError("Unexpected arm")
    if arm == "exact_suffix":
        value = donor[q].copy()
    elif arm == "noop":
        value = current[p].copy()
    else:
        delta = source_delta(
            sectors, entry["person"], entry["direction"], arm, entry["pair"], seed
        )
        if delta.shape != current[p].shape:
            raise ValueError("Source suffix shape mismatch")
        value = (current[p].astype(float) + delta).astype(current.dtype)
    if value.shape != current[p].shape or not np.isfinite(value).all():
        raise ValueError("Finite exact suffix assignment required")
    return p, value


def high_low_log_odds(logits):
    """Measure fixed candidate5-9 versus0-4 logits, not a psychological latent unit."""
    x = np.asarray(logits, float)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError("Ten finite predeclared digit logits required")

    def lse(v):
        m = np.max(v)
        return float(m + np.log(np.exp(v - m).sum()))

    return lse(x[5:]) - lse(x[:5])


def orientation(person, direction):
    """Fix high/low action orientation from source design, not the receiver answer."""
    if person not in ("A", "B") or direction not in (0, 1):
        raise ValueError("Explicit source actor and binary direction required")
    return (1 - 2 * direction) * (-1 if person == "A" else 1)


def summary(rows):
    """Keep output action and native contrast; do not divide by tiny contrast silently."""
    if not rows:
        raise ValueError("Actual observations required")
    full = np.asarray([r["full_change"] for r in rows])
    delta = np.asarray([r["patch_change"] for r in rows])
    log_delta = np.asarray([r["patch_log_odds"] - r["native_log_odds"] for r in rows])
    eta = np.asarray([orientation(r["person"], r["direction"]) for r in rows])
    square = float(full @ full)
    return {
        "rows": len(rows),
        "full_rms": float(np.sqrt(np.mean(full**2))),
        "patch_rms": float(np.sqrt(np.mean(delta**2))),
        "signed_projection": float(full @ delta / square) if square else None,
        "native_contrast_tiny": bool(square < 1e-12),
        "remaining_rmse": float(np.sqrt(np.mean((full - delta) ** 2))),
        "direction_count": int(np.sum(full * delta > 0)),
        "oriented_score_mean": float(np.mean(eta * delta)),
        "log_odds_action_rms": float(np.sqrt(np.mean(log_delta**2))),
        "oriented_log_odds_mean": float(np.mean(eta * log_delta)),
    }
