"""Outcome-blind equal-evidence material operators and support checks."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

import numpy as np

from .data_views import stable_key

CUE_PATTERN = re.compile(
    r"\b(?:[EI][NS][FT][JP]|big\s*(?:five|5)|ocean|openness|conscientiousness|"
    r"extraversion|agreeableness|neuroticism|personality\s+test|percentile|"
    r"my\s+(?:test\s+)?results|scored\s+\d+|TYPE_MENTION)\b",
    re.IGNORECASE,
)
ARMS = ("raw", "cue_mask", "placebo_mask")
MASK = "[MASKED]"
MASK_SEED = "grain_vm05_equal_word_mask_20260906"


def pack_features(features):
    """Losslessly store identical arrays once; aliases preserve every logical field."""
    unique, aliases, seen = {}, {}, {}
    for name, value in features.items():
        x = np.ascontiguousarray(value)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite feature cannot enter cache")
        key = (x.shape, x.dtype.str, hashlib.sha256(x.tobytes()).hexdigest())
        if key in seen:
            target = seen[key]
            if not np.array_equal(x, unique[target]):
                raise ValueError("Feature hash collision")
            aliases[name] = target
        else:
            unique[name] = x
            seen[key] = name
    return unique, aliases


def unpack_features(stored, aliases):
    """Restore exactly named logical arrays, rejecting malformed cache references."""
    result = dict(stored)
    for name, target in aliases.items():
        if name in result or target not in stored:
            raise ValueError("Invalid feature alias")
        result[name] = stored[target]
    return result


def shard_material(source, count=1, index=0):
    """Partition a complete label-blind cohort once, independent of source order."""
    if not 1 <= count <= min(256, len(source)) or not 0 <= index < count:
        raise ValueError("Invalid deterministic shard")
    if len({b["author_id"] for b in source}) != len(source):
        raise ValueError("Repeated source author")
    ordered = sorted(
        source, key=lambda b: stable_key(b["author_id"], "grain_vm05_shards_731")
    )
    return ordered[index::count]


def masked_comment(text, identity):
    """Mask cue-overlapping or equal-count noncue words, preserving whitespace.

    This lexical ablation cannot remove all semantic self-disclosure or scores.
    """
    words = list(re.finditer(r"\S+", text))
    spans = [(m.start(), m.end()) for m in CUE_PATTERN.finditer(text)]
    selected = {
        i
        for i, w in enumerate(words)
        if any(w.start() < b and w.end() > a for a, b in spans)
    }
    other = [i for i in range(len(words)) if i not in selected]
    if len(other) < len(selected):
        raise ValueError("Insufficient noncue words for an equal-count placebo")
    placebo = set(
        sorted(other, key=lambda i: stable_key(f"{identity}:{i}", MASK_SEED))[
            : len(selected)
        ]
    )

    def replace(indices):
        chunks, start = [], 0
        for i, word in enumerate(words):
            chunks.extend(
                (text[start : word.start()], MASK if i in indices else word.group())
            )
            start = word.end()
        chunks.append(text[start:])
        return "".join(chunks)

    return {
        "raw": text,
        "cue_mask": replace(selected),
        "placebo_mask": replace(placebo),
    }, {
        "cue_words": len(selected),
        "placebo_words": len(placebo),
        "total_words": len(words),
        "cue_placebo_overlap": len(selected & placebo),
    }


def chat_ids(tokenizer, system, user):
    """Flatten actual template output without loading weights or truncating."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if not ids or not all(isinstance(i, int) for i in ids):
        raise ValueError("Need flat chat token IDs")
    return list(ids)


def raw_author_features(units, names):
    """Construct named mean/std columns without assuming feature order."""
    h = np.asarray(units, dtype=np.float64)
    if h.ndim != 3 or not np.isfinite(h).all():
        raise ValueError("Need finite author-by-unit-by-feature array")
    mean, std = h.mean(1), h.std(1, ddof=0)
    cols = []
    for name in names:
        prefix, index = name.split("::svd_")
        if prefix not in ("mean", "std"):
            raise ValueError("Unsupported frozen GRAIN feature")
        cols.append((mean if prefix == "mean" else std)[:, int(index)])
    return np.stack(cols, axis=1)


def purge_counts(train, test):
    """Measure train material loss from test-thread purging without labels."""
    if {b["author_id"] for b in train} & {b["author_id"] for b in test}:
        raise ValueError("Train/test author overlap")
    test_threads = {c["thread_id"] for b in test for v in b["views"] for c in v}
    counts = [
        sum(c["thread_id"] not in test_threads for v in b["views"] for c in v)
        for b in train
    ]
    total = [sum(map(len, b["views"])) for b in train]
    return {
        "train_authors": len(train),
        "test_authors": len(test),
        "removed_comments": sum(total) - sum(counts),
        "train_with_36_remaining": sum(c >= 36 for c in counts),
        "train_with_24_remaining": sum(c >= 24 for c in counts),
        "train_with_12_remaining": sum(c >= 12 for c in counts),
        "remaining_min_median_max": np.quantile(counts, [0, 0.5, 1]).tolist()
        if counts
        else [],
    }
