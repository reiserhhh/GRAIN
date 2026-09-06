#!/usr/bin/env python3
"""Source-safe history selection and token-exact, fixed-position counterfactuals."""

from __future__ import annotations

import math

import numpy as np

from .context import QUESTIONS, SYSTEM
from .controls import material_hash
from .data_views import normalized_text_key
from .external_material import chat_ids


def split_view(view):
    """Keep all12 comments and reject unsupported temporal/discussion provenance."""
    if len(view) != 12 or len({c["author_id"] for c in view}) != 1:
        raise ValueError("Expected12 comments by one author")
    for field in ("comment_id", "thread_id"):
        if any(not c[field] for c in view) or len({c[field] for c in view}) != 12:
            raise ValueError("Repeated or missing comment/discussion provenance")
    if len({normalized_text_key(c["body"]) for c in view}) != 12:
        raise ValueError("Repeated normalized comment text")
    if any(not math.isfinite(c["timestamp"]) for c in view):
        raise ValueError("Invalid timestamp")
    ordered = sorted(view, key=lambda c: (c["timestamp"], c["comment_id"]))
    history, current = ordered[:-1], ordered[-1]
    if history[-1]["timestamp"] >= current["timestamp"]:
        raise ValueError("History must strictly predate current")
    return history, current


def history_text(history):
    """Preserve original bodies and neutral excerpt boundaries, no identity metadata."""
    return "\n\n".join(
        f"Earlier excerpt {i + 1}:\n{c['body']}" for i, c in enumerate(history)
    )


def profile(history, token_count):
    """Compute outcome-blind matching inputs once per history."""
    return {
        "tokens": int(token_count(history_text(history))),
        "end": max(c["timestamp"] for c in history),
        "threads": {c["thread_id"] for c in history},
        "comments": {c["comment_id"] for c in history},
        "bodies": {normalized_text_key(c["body"]) for c in history},
        "subreddits": {c["subreddit"] for c in history},
    }


def select_donor(own, current, target_comments, donors):
    """Select solely from an explicitly held-out reference dictionary, with no labels."""
    uid, time = current["author_id"], current["timestamp"]
    forbidden = {
        key: {c[key] for c in target_comments} for key in ("thread_id", "comment_id")
    }
    bodies = {normalized_text_key(c["body"]) for c in target_comments}
    candidates = []
    for d in donors:
        p = d["profile"]
        if (
            d["author_id"] == uid
            or p["end"] >= time
            or p["threads"] & forbidden["thread_id"]
            or p["comments"] & forbidden["comment_id"]
            or p["bodies"] & bodies
        ):
            continue
        length_error = abs(math.log(p["tokens"] / own["tokens"]))
        time_error = abs(
            math.log1p((time - p["end"]) / 86400)
            - math.log1p((time - own["end"]) / 86400)
        )
        union = p["subreddits"] | own["subreddits"]
        jaccard = len(p["subreddits"] & own["subreddits"]) / max(1, len(union))
        candidates.append(
            {
                "donor": d,
                "length_error": length_error,
                "time_error": time_error,
                "subreddit_jaccard": jaccard,
                "cost": length_error + 0.15 * time_error + 0.5 * (1 - jaccard),
            }
        )
    if not candidates:
        raise ValueError("No eligible reference history")
    winner = min(
        candidates,
        key=lambda c: (c["cost"], c["donor"]["author_id"], c["donor"]["view"]),
    )
    return winner | {"eligible_donors": len(candidates)}


def render(history, current, task):
    """Return full history/current intervals in a neutral anonymous-writer prompt."""
    pre = "The earlier and current excerpts are presented as writing by one anonymous writer.\n\n"
    h = history_text(history)
    middle = "\n\nCurrent excerpt:\n"
    start = len(pre) + len(h) + len(middle)
    user = (
        pre
        + h
        + middle
        + current
        + "\nEvidence ends here.\n\nQuestion: "
        + QUESTIONS[task]
        + "\nAnswer:"
    )
    return user, (len(pre), len(pre) + len(h)), (start, start + len(current))


def tokenize_parts(tokenizer, history, current, task, *, enclose_separator=False):
    """Split native chat tokens; absorb boundary-straddling tokens into history."""
    user, history_span, current_span = render(history, current, task)
    if enclose_separator:
        if user[history_span[1] : history_span[1] + 2] != "\n\n":
            raise ValueError("Expected exact history/current separator")
        history_span = (history_span[0], history_span[1] + 2)
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered.count(user) != 1:
        raise ValueError("Ambiguous rendered user message")
    start = rendered.index(user)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    if ids != chat_ids(tokenizer, SYSTEM, user):
        raise ValueError("Offsets disagree with actual chat template")
    h0, h1 = (start + x for x in history_span)
    c0, c1 = (start + x for x in current_span)
    hist = [
        i
        for i, (a, b) in enumerate(encoded["offset_mapping"])
        if b > a and a < h1 and b > h0
    ]
    cur = [
        i
        for i, (a, b) in enumerate(encoded["offset_mapping"])
        if b > a >= c0 and b <= c1
    ]
    if (
        not hist
        or not cur
        or hist != list(range(hist[0], hist[-1] + 1))
        or hist[-1] >= cur[0]
    ):
        raise ValueError("Unusable history/current token intervals")
    left, right = hist[0], hist[-1] + 1
    return {
        "prefix": ids[:left],
        "history": ids[left:right],
        "suffix": ids[right:],
        "current_in_suffix": [i - right for i in cur],
        "native_ids": ids,
    }


def aligned_triplet(own, matched, pad_id, max_length=3072):
    """Keep suffix/current positions fixed, mask only history-slot padding/visibility."""
    if (
        own["prefix"] != matched["prefix"]
        or own["suffix"] != matched["suffix"]
        or own["current_in_suffix"] != matched["current_in_suffix"]
    ):
        raise ValueError("Current/prefix/suffix tokens differ between histories")
    slot = max(len(own["history"]), len(matched["history"]))
    prefix, suffix = own["prefix"], own["suffix"]
    length = len(prefix) + slot + len(suffix)
    if not prefix or not slot or not suffix or length + 16 > max_length:
        raise ValueError("Aligned prompt exceeds fixed token budget or is empty")
    positions = [len(prefix) + slot + i for i in own["current_in_suffix"]]
    if not positions or max(positions) >= length:
        raise ValueError("Current indices outside aligned prompt")
    rows = {}
    for arm, parts in (("own", own), ("matched", matched), ("invisible", own)):
        h = parts["history"]
        ids = prefix + h + [pad_id] * (slot - len(h)) + suffix
        mask = (
            [1] * len(prefix)
            + [int(arm != "invisible")] * len(h)
            + [0] * (slot - len(h))
            + [1] * len(suffix)
        )
        rows[arm] = {
            "ids": ids,
            "mask": mask,
            "current_positions": positions,
            "decision_position": length - 1,
            "slot": [len(prefix), len(prefix) + slot],
            "current_token_sha256": material_hash([ids[i] for i in positions]),
            "visible_tokens": sum(mask),
        }
    if len({r["current_token_sha256"] for r in rows.values()}) != 1:
        raise ValueError("Current token identity changed")
    return rows


def capture_aligned(observer, row):
    """Run a fixed-shape masked-key intervention without modifying the base observer."""
    import torch

    ids, mask = row["ids"], row["mask"]
    length = observer.pad_to_length
    positions = [row["decision_position"]]
    current = row["current_positions"]
    if (
        length is None
        or len(ids) > length
        or len(ids) != len(mask)
        or not set(mask) <= {0, 1}
        or any(
            not isinstance(i, int) or i < 0 or i >= len(ids)
            for i in positions + current
        )
        or len(set(current)) != len(current)
        or not all(mask[i] for i in positions + current)
        or positions != [len(ids) - 1]
        or not current
    ):
        raise ValueError("Invalid fixed-position history capture")
    pad_id = observer.tokenizer.pad_token_id or 0
    device = next(observer.model.parameters()).device
    tokens = torch.tensor([ids + [pad_id] * (length - len(ids))], device=device)
    attention = torch.tensor([mask + [0] * (length - len(ids))], device=device)
    with (
        observer.hooks(
            positions, pools=[("mean", current), ("std", current)]
        ) as states,
        torch.inference_mode(),
    ):
        out = observer.model(
            input_ids=tokens,
            attention_mask=attention,
            position_ids=torch.arange(length, device=device)[None],
            use_cache=False,
            return_dict=True,
        )
    observer.calls += 1
    observer.input_tokens += sum(mask)
    observer.processed_tokens += length
    logits = out.logits[0, len(ids) - 1].detach().float().cpu().numpy()
    if not np.isfinite(logits).all() or any(
        not np.isfinite(v).all() for v in states.values()
    ):
        raise ValueError("Nonfinite masked history observation")
    return states, logits
