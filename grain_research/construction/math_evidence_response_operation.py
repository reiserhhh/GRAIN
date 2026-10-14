#!/usr/bin/env python3
"""Fixed hidden interventions and evidence contrasts, not receiver-oriented steering."""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from .math_relational_operation_specificity import high_low_log_odds

CLASSES = ("own_1", "own_2", "other_report", "roster")
ARMS = ("source_plus", "source_minus", "exact_suffix", "wrong_person", "random", "noop")


def documents(spec):
    """Construct paired fresh accounts without assigning a psychological gold score."""
    rows = []
    for target, kind, direction in itertools.product(spec["targets"], CLASSES, (0, 1)):
        body = (
            "This account reports actions and attendance records of anonymous people.\n"
        )
        actors = ("B", "A") if kind == "roster" and direction else ("A", "B")
        for actor in actors:
            polarity = direction if actor == "A" else 1 - direction
            family = "family_2" if kind == "own_2" else "family_1"
            action = spec["fresh_behaviours"][target][family][polarity]
            if kind == "other_report":
                other = "C" if actor == "A" else "D"
                action = f"reported that Person {other} " + action
            elif kind == "roster":
                action = "was registered in the workshop attendance list"
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
    index = {(r["content"], r["class"], r["direction"]): r for r in rows}
    for target, kind in itertools.product(spec["targets"], CLASSES):
        a, b = (index[target, kind, d]["body"] for d in (0, 1))
        if a == b or Counter(a.split()) != Counter(b.split()):
            raise ValueError(
                "Paired ownership/order accounts must preserve word counts"
            )
    return rows


def patch_values(entry, receiver, current, donor, source, seed):
    """Apply the same signed source delta to both receiver ownership directions."""
    arm, p = entry["arm"], receiver["partition"]["suffix"]
    if arm not in ARMS:
        raise ValueError("Unknown fixed intervention")
    if arm == "exact_suffix":
        value = donor[entry["donor_positions"]].copy()
    elif arm == "noop":
        value = current[p].copy()
    else:
        person = (
            ("B" if receiver["person"] == "A" else "A")
            if arm == "wrong_person"
            else receiver["person"]
        )
        delta = source[f"{receiver['content']}__{person}"]
        if delta.shape != current[p].shape:
            raise ValueError("Complete source/receiver slot shape mismatch")
        if arm == "random":
            noise = np.random.default_rng(seed + receiver["pair_id"]).normal(
                size=delta.shape
            )
            delta = (
                noise
                * (np.linalg.norm(delta, axis=1) / np.linalg.norm(noise, axis=1))[
                    :, None
                ]
            )
        sign = -1 if arm == "source_minus" else 1
        value = (current[p].astype(float) + sign * delta).astype(current.dtype)
    if value.shape != current[p].shape or not np.isfinite(value).all():
        raise ValueError("Finite complete suffix required")
    return p, value


def readout(native):
    """Keep candidate score, raw log-odds and vocabulary coverage separately."""
    return {
        "score": native["conditional_score"],
        "log_odds": high_low_log_odds(native["candidate_logits"]),
        "candidate_mass": native["candidate_mass"],
    }


def output_offset(logits, offset):
    """Add an output-only high-group logit offset, preserving within-group ratios."""
    z = np.asarray(logits, float)
    if z.shape != (10,) or not np.isfinite(z).all() or not np.isfinite(offset):
        raise ValueError("Ten finite logits and a finite fixed offset required")
    shifted = z.copy()
    shifted[5:] += offset
    q = np.exp(shifted - shifted.max())
    q /= q.sum()
    return {
        "score": float(q @ (np.arange(10) / 9)),
        "log_odds": high_low_log_odds(shifted),
        "probabilities": q,
    }


def paired(native0, native1, patched0, patched1):
    """Measure action x evidence interaction without selecting its sign or size."""
    result = {}
    for metric in ("score", "log_odds"):
        n0, n1, p0, p1 = (v[metric] for v in (native0, native1, patched0, patched1))
        result.update(
            {
                f"native_{metric}_contrast": n1 - n0,
                f"patched_{metric}_contrast": p1 - p0,
                f"action_{metric}_0": p0 - n0,
                f"action_{metric}_1": p1 - n1,
                f"mixed_{metric}": (p1 - p0) - (n1 - n0),
            }
        )
    return result


def paired_summary(rows):
    """Describe signed response signatures, not significance over template replicates."""
    if not rows:
        raise ValueError("Observed paired actions required")
    result = {"pairs": len(rows)}
    for key in (
        "native_score_contrast",
        "patched_score_contrast",
        "mixed_score",
        "native_log_odds_contrast",
        "patched_log_odds_contrast",
        "mixed_log_odds",
    ):
        x = np.asarray([r[key] for r in rows], float)
        result[key + "_mean"] = float(x.mean())
        result[key + "_rms"] = float(np.sqrt(np.mean(x**2)))
    result["native_patched_log_direction_matches"] = sum(
        r["native_log_odds_contrast"] * r["patched_log_odds_contrast"] > 0 for r in rows
    )
    return result
