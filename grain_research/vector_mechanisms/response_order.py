"""Exact-token-multiset context perturbations; no text decoding or learned choice."""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.response_context import (
    native_parts,
    paired_interval,
)

MODES = ("block8", "token")
SEEDS = (731, 1731)


def assistant_span(tokenizer, response, context):
    """Isolate complete assistant-content tokens while freezing every role boundary."""
    prefix, body = native_parts(tokenizer, response, context)
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "assistant", "content": context},
            {"role": "user", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    response_start = rendered.rfind(response)
    start = rendered.find(context, 0, response_start)
    if (
        not context
        or start < 0
        or rendered.find(context, start + 1, response_start) >= 0
    ):
        raise ValueError("Assistant content span is absent or ambiguous")
    stop = start + len(context)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    indices = [i for i, (a, b) in enumerate(offsets) if b > a >= start and b <= stop]
    if (
        len(indices) < 8
        or indices != list(range(indices[0], indices[-1] + 1))
        or indices[-1] >= len(prefix)
    ):
        raise ValueError("Insufficient or noncontiguous isolated assistant tokens")
    if list(encoded["input_ids"][: len(prefix)]) != prefix:
        raise ValueError("Native prefix changed during assistant localization")
    special = set(tokenizer.all_special_ids)
    if any(prefix[i] in special for i in indices):
        raise ValueError("Unexpected special token inside assistant body")
    crossing = sum(
        b > start and a < stop and (a < start or b > stop) for a, b in offsets if b > a
    )
    return (
        prefix,
        body,
        indices,
        {
            "isolated_context_tokens": len(indices),
            "fixed_boundary_crossing_tokens": int(crossing),
        },
    )


def permute_prefix(prefix, indices, code, mode, seed):
    """Shuffle only the isolated span, with deterministic donor-consistent ordering."""
    if mode not in MODES or seed not in SEEDS:
        raise ValueError("Unregistered order condition")
    if len(indices) < 8 or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Unsupported isolated token span")
    if indices[0] < 0 or indices[-1] >= len(prefix):
        raise ValueError("Context span exceeds native prefix")
    x = np.asarray(prefix, dtype=np.int64)
    context = x[indices]
    width = 8 if mode == "block8" else 1
    groups = [
        np.arange(i, min(i + width, len(indices)))
        for i in range(0, len(indices), width)
    ]
    salt = hashlib.sha256(f"RC3|{code}|{mode}|{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(salt[:8], "big"))
    group_order = rng.permutation(len(groups))
    if len(groups) > 1 and np.array_equal(group_order, np.arange(len(groups))):
        group_order = np.roll(group_order, 1)
    order = np.concatenate([groups[i] for i in group_order])
    changed = x.copy()
    changed[indices] = context[order]
    restored = changed.copy()
    restored[indices] = changed[indices][np.argsort(order)]
    outside = np.ones(len(prefix), dtype=bool)
    outside[indices] = False
    bag_before = sorted(Counter(context.tolist()).items())
    bag_after = sorted(Counter(changed[indices].tolist()).items())
    if (
        bag_before != bag_after
        or not np.array_equal(restored, x)
        or not np.array_equal(changed[outside], x[outside])
    ):
        raise ValueError("Token multiset, inverse or role boundary invariant failed")
    return changed.tolist(), {
        "isolated_context_tokens": len(indices),
        "permutation_groups": len(groups),
        "changed_tokens": int(np.count_nonzero(changed[indices] != context)),
        "bag_sha256": material_hash(bag_before),
        "native_context_order_sha256": material_hash(context.tolist()),
        "altered_context_order_sha256": material_hash(changed[indices].tolist()),
        "permutation_sha256": material_hash(order.tolist()),
        "outside_span_changed": False,
        "inverse_max_token_error": 0,
    }


def summarize_order(frame):
    """Paired order interactions, never treating seeds as additional participants."""
    effects = {}
    for score in ("mean_logp", "tail_logp"):
        pivot = frame.pivot(index="participant_code", columns="condition", values=score)
        native = pivot.actual_native - pivot.matched_native
        effects[f"{score}/native"] = paired_interval(native)
        for mode in MODES:
            differences, penalties = [], []
            for seed in SEEDS:
                actual = pivot[f"actual_{mode}_{seed}"]
                matched = pivot[f"matched_{mode}_{seed}"]
                difference = actual - matched
                penalty = (
                    (pivot.actual_native - actual) + (pivot.matched_native - matched)
                ) / 2
                key = f"{score}/{mode}/{seed}"
                effects[key] = {
                    "actual_minus_matched": paired_interval(difference),
                    "order_interaction": paired_interval(native - difference),
                    "common_degradation": paired_interval(penalty),
                }
                differences.append(difference)
                penalties.append(penalty)
            average = sum(differences) / len(SEEDS)
            effects[f"{score}/{mode}/seed_mean"] = {
                "actual_minus_matched": paired_interval(average),
                "order_interaction": paired_interval(native - average),
                "common_degradation": paired_interval(sum(penalties) / len(SEEDS)),
            }
    return effects
