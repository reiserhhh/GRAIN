#!/usr/bin/env python3
"""Finite answer-code contrasts; no hidden-state equivariance is assumed."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .math_construct_routing import query, token_positions
from .math_relational_operation_specificity import high_low_log_odds, partition

FORWARD_CUE = "Use 0 for very low, 9 for very high"
REVERSE_CUE = "Use 9 for very low, 0 for very high"
ARMS = (
    "exact_suffix",
    "source_forward",
    "source_reverse",
    "wrong_person",
    "random",
    "noop",
)


def decode(raw, codec):
    """Convert the declared raw answer coordinate into the semantic coordinate."""
    if codec not in (1, -1):
        raise ValueError("Explicit standard or reversed code required")
    return (1 - codec) / 2 + codec * raw


def render(body, target, person, codec, system, tokenizer):
    """Change exactly the two scale cues in the unchanged person-neutral template."""
    if person not in ("A", "B") or codec not in (1, -1):
        raise ValueError("Known person and answer code required")
    question = query(target).replace("the writer's", f"Person {person}'s")
    if question.count(FORWARD_CUE) != 1 or "writer" in question:
        raise ValueError("One unchanged explicit digit rule required")
    if codec == -1:
        question = question.replace(FORWARD_CUE, REVERSE_CUE)
    user = "The following account describes two anonymous people.\n\n" + body
    user += "\n\nEvidence ends here.\n\nQuestion: " + question + "\nAnswer:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    actual = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(actual, Mapping):
        actual = actual["input_ids"]
    if list(actual) != list(encoded["input_ids"]) or text.count(body) != 1:
        raise ValueError("Actual chat-template IDs and unique body required")
    if len(actual) + 16 > 3072:
        raise ValueError("Full untruncated input required")
    row = {"body": body, "rendered": text, "input_ids": list(actual)}
    part = partition(row, encoded)
    cue = FORWARD_CUE if codec == 1 else REVERSE_CUE
    start = text.index(cue, text.index("Question: "))
    slots = []
    for offset in (4, cue.index(", ") + 2):
        positions = token_positions(
            encoded["offset_mapping"], start + offset, start + offset + 1
        )
        if len(positions) != 1:
            raise ValueError("Each scale cue must be a single token")
        slots.append(part["suffix"].index(positions[0]))
    actor_start = text.index(f"Person {person}'s", text.index("Question: ")) + 7
    actor = token_positions(encoded["offset_mapping"], actor_start, actor_start + 1)
    if len(actor) != 1:
        raise ValueError("One actual query actor token required")
    return text, list(actual), part, slots, part["suffix"].index(actor[0])


def prove_mapping(source, receiver, allow_actor=False):
    """Require identical ordered suffix slots except declared code/name changes."""
    if any(source[k] != receiver[k] for k in ("content", "query")):
        raise ValueError("Wrong target mapping")
    p, q = source["partition"]["suffix"], receiver["partition"]["suffix"]
    a = [source["input_ids"][i] for i in p]
    b = [receiver["input_ids"][i] for i in q]
    if len(a) != len(b) or source["cue_slots"] != receiver["cue_slots"]:
        raise ValueError("Incompatible complete suffix slots")
    expected = set()
    if source["codec"] != receiver["codec"]:
        expected.update(source["cue_slots"])
    if source["person"] != receiver["person"]:
        if not allow_actor or source["actor_slot"] != receiver["actor_slot"]:
            raise ValueError("Undeclared actor change")
        expected.add(source["actor_slot"])
    differences = {i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y}
    if differences != expected:
        raise ValueError("Changes outside the exact declared code/name slots")
    return {
        "source_positions": p,
        "receiver_positions": q,
        "changed_slots": sorted(expected),
    }


def code_pair(standard, inverse):
    """Prove that the full recoding changes only two digit-cue token identities."""
    if standard["codec"] != 1 or inverse["codec"] != -1:
        raise ValueError("Ordered standard/reverse pair required")
    if any(
        standard[k] != inverse[k]
        for k in ("role", "class", "person", "direction", "body")
    ):
        raise ValueError("Same material and person required")
    proof = prove_mapping(standard, inverse)
    ids, other = standard["input_ids"], inverse["input_ids"]
    expected = {standard["partition"]["suffix"][s] for s in standard["cue_slots"]}
    if (
        len(ids) != len(other)
        or {i for i, (a, b) in enumerate(zip(ids, other, strict=True)) if a != b}
        != expected
    ):
        raise ValueError("Only the two actual full-input cue tokens may change")
    if standard["rendered"].replace(FORWARD_CUE, REVERSE_CUE) != inverse["rendered"]:
        raise ValueError("Undeclared textual recoding")
    return proof


def source_key(target, person, codec):
    """Name a source-only operator without receiver answers or fitted weights."""
    return f"{target}__{person}__{codec}"


def patch_values(entry, receiver, current, donor, source, seed):
    """Apply one fixed source/receiver-code arm, preserving every other position."""
    arm, p = entry["arm"], receiver["partition"]["suffix"]
    if arm not in ARMS:
        raise ValueError("Undeclared intervention")
    if arm == "exact_suffix":
        value = donor[entry["donor_positions"]].copy()
    elif arm == "noop":
        value = current[p].copy()
    else:
        codec = (
            1
            if arm == "source_forward"
            else -1
            if arm == "source_reverse"
            else receiver["codec"]
        )
        person = (
            ("B" if receiver["person"] == "A" else "A")
            if arm == "wrong_person"
            else receiver["person"]
        )
        delta = source[source_key(receiver["content"], person, codec)]
        if delta.shape != current[p].shape:
            raise ValueError("Exact source/receiver suffix shape required")
        if arm == "random":
            noise = np.random.default_rng(
                seed + receiver["standard_reference_uid"] // 2
            ).normal(size=delta.shape)
            noise *= (np.linalg.norm(delta, axis=1) / np.linalg.norm(noise, axis=1))[
                :, None
            ]
            delta = noise
        value = (
            current[p].astype(float) + (1 - 2 * receiver["direction"]) * delta
        ).astype(current.dtype)
    if value.shape != current[p].shape or not np.isfinite(value).all():
        raise ValueError("Finite complete suffix assignment required")
    return p, value


def readout(native, codec):
    """Keep raw and decoded coordinates; no compliance-based selection or rescue."""
    raw = native["conditional_score"]
    odds = high_low_log_odds(native["candidate_logits"])
    return {
        "raw_score": raw,
        "semantic_score": decode(raw, codec),
        "raw_log_odds": odds,
        "semantic_log_odds": codec * odds,
        "candidate_mass": native["candidate_mass"],
    }


def orbit(positive, negative):
    """Decompose fixed-source raw action into finite C2 even/odd contrasts."""
    return {"raw_even": (positive + negative) / 2, "raw_odd": (positive - negative) / 2}


def action_summary(rows):
    """Report each predeclared action group without predicting psychological truth."""
    return {
        "rows": len(rows),
        **{
            k + "_mean": float(np.mean([r[k] for r in rows]))
            for k in (
                "raw_action",
                "semantic_action",
                "raw_score_action",
                "semantic_score_action",
            )
        },
        "semantic_direction_count": sum(r["semantic_action"] > 0 for r in rows),
    }
