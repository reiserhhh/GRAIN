#!/usr/bin/env python3
"""Separate diagnostic, prospective-choice and instruction-policy endpoints."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import expit

AXES = ("extraversion", "neuroticism")
FAMILIES = ("diagnostic", "prospective", "forced_yes", "forced_no")
MEASURES = ("log_odds", "restricted_yes", "full_yes", "answer_mass")


def questions():
    """Register concrete new opportunities without using future model outputs."""
    definitions, texts = [], []
    for axis, family, person in itertools.product(AXES, FAMILIES, (-1, 1)):
        name = "A" if person == 1 else "B"
        if family == "diagnostic":
            text = (
                f"Does this record provide evidence that Person {name} has high "
                f"{axis}? Answer Yes or No."
            )
        else:
            scene = (
                "At a voluntary reunion, an unfamiliar guest is standing nearby. "
                "Do you choose to approach the guest and start a conversation?"
                if axis == "extraversion"
                else "Before a new presentation, you have one free minute. "
                "Do you choose to spend it checking again for possible mistakes?"
            )
            text = f"For the next scene, role-play as Person {name}. " + scene
            if family.startswith("forced_"):
                answer = "Yes" if family == "forced_yes" else "No"
                text += (
                    f" For this instruction control, output {answer} regardless "
                    "of the record or your choice."
                )
            text += " Answer Yes or No."
        definitions.append({"axis": axis, "family": family, "person": person})
        texts.append(text)
    return definitions, texts


def endpoint_scores(selected, normalizer):
    """Keep restricted binary policy and full-vocabulary response mass distinct."""
    logits, norm = np.asarray(selected, float), np.asarray(normalizer, float)
    if logits.ndim != 2 or logits.shape[1] != 2 or norm.shape != logits.shape[:1]:
        raise ValueError("Aligned Yes/No logits and vocabulary normalizers required")
    if not np.isfinite(logits).all() or not np.isfinite(norm).all():
        raise ValueError("Nonfinite endpoint observations")
    probabilities = np.exp(logits - norm[:, None])
    if np.any(probabilities.sum(axis=1) > 1 + 1e-5):
        raise ValueError("Selected answer mass exceeds vocabulary probability")
    difference = logits[:, 0] - logits[:, 1]
    return {
        "log_odds": difference,
        "restricted_yes": expit(difference),
        "full_yes": probabilities[:, 0],
        "answer_mass": probabilities.sum(axis=1),
    }


def natural_partner(row, axis, indexed):
    """Exchange one actual actor assignment, retaining condition and clause order."""
    if axis not in ("e", "n"):
        raise ValueError("Only registered e/n assignment edits are available")
    return indexed[
        row["condition"],
        row["order"],
        -row["e"] if axis == "e" else row["e"],
        -row["n"] if axis == "n" else row["n"],
    ]


def policy_contrast(free, instructed_yes, instructed_no):
    """Describe endpoint dependence on output instructions, not individual validity."""
    f, y, n = (np.asarray(a, float) for a in (free, instructed_yes, instructed_no))
    if f.shape != y.shape or y.shape != n.shape or not np.isfinite([f, y, n]).all():
        raise ValueError("Aligned finite endpoint actions required")
    return {
        "free_rms": float(np.sqrt(np.mean(f**2))),
        "forced_yes_rms": float(np.sqrt(np.mean(y**2))),
        "forced_no_rms": float(np.sqrt(np.mean(n**2))),
        "free_minus_forced_mean_rms": float(np.sqrt(np.mean((f - (y + n) / 2) ** 2))),
    }
