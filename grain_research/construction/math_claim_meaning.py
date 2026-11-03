#!/usr/bin/env python3
"""Matched claim involution fields,not an optimized or human-calibrated reader."""

import numpy as np

from . import math_person_bound_tree as T

PRIMARY = ("semantic_direction", "affirmation_direction")
CHANNELS = (*PRIMARY, "availability_even", "availability_odd")


def question(item, queried, polarity):
    """Deny the whole author-fit claim,not the quoted behavior or response digit."""
    if queried not in (-1, 1) or polarity not in (-1, 1):
        raise ValueError("Signed queried actor and claim polarity required")
    if not isinstance(item, str) or not item or not item.endswith("."):
        raise ValueError("Complete public behavioral statement required")
    actor, truth = (
        ("R" if queried == 1 else "S"),
        ("true" if polarity == 1 else "false"),
    )
    return (
        f'First-person behavioral statement: "I {item[0].lower() + item[1:]}"\n'
        f"Claim: It is {truth} that the behavioral statement describes writer {actor}'s "
        "usual behavior across occasions.\n"
        "Using only the identified comments, evaluate THE CLAIM. Return 1 if the "
        "comments support THE CLAIM, 0 if they contradict THE CLAIM, or 2 if they "
        "are insufficient to judge THE CLAIM.\nAnswer:"
    )


def exported_bias(size, values=None):
    """Represent an absent bias by the full registered alphabet,without fitting."""
    if type(size) is not int or size < 1:
        raise ValueError("Positive alphabet size required")
    bias = np.zeros(size) if values is None else np.asarray(values, float)
    if bias.shape != (size,) or not np.isfinite(bias).all():
        raise ValueError("Output bias must match every registered category")
    return bias


def readings(logits, bank):
    """Keep all involution sectors and exact native category laws on both sides."""
    z = np.asarray(logits, float)
    if z.shape != (2, 48, 2, 2, 2, 50, 3) or not np.isfinite(z).all():
        raise ValueError("Both complete48-case/eight-corner/50-item/three-digit laws")
    plus, tp = T.readings(z[0], bank)
    minus, tm = T.readings(z[1], bank)
    result = {}
    for name, original, sign in (
        ("semantic_direction", "direction", -1),
        ("affirmation_direction", "direction", 1),
        ("availability_even", "assessability", 1),
        ("availability_odd", "assessability", -1),
    ):
        result[name] = {
            key: (plus[original][key] + sign * minus[original][key]) / 2
            for key in plus[original]
        }
    return result, (plus, minus), (tp, tm)


def cold(record):
    """Replay actual addressed endpoints;validate both real polarity records."""
    keys = {"version", "cases", "source_components", "item_logits", "item_bank"}
    if set(record) != keys or record["version"] != "HM289-v1":
        raise ValueError("Only the registered numeric claim record is allowed")
    z = np.asarray(record["item_logits"], float)
    fields, _, trees = readings(z, record["item_bank"])
    for side in range(2):
        T.cold(dict(record, version="HM288-v1", item_logits=z[side].tolist()))
    source = np.asarray(record["source_components"], float)
    return {
        "channels": {
            channel: {
                key: T.contacts(source, fields[channel][key]).tolist()
                for key in ("balanced_key", "common_support", "global_common")
            }
            for channel in CHANNELS
        },
        "positive_negative_category_means": [
            np.exp(tree["log_probabilities"]).mean(axis=(0, 1, 2, 3, 4)).tolist()
            for tree in trees
        ],
        "no_new_neural_inference_or_application_labels": True,
        "not_a_raw_text_personality_predictor": True,
    }
