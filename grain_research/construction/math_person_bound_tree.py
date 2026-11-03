#!/usr/bin/env python3
"""Finite direction/claimed-assessability coordinates,not a human latent fit."""

from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from . import math_behavior_predictive as B
from . import math_person_bound_response as O
from .math_item_bridge import common_prefix

CHANNELS = ("direction", "assessability")


def question(item, queried):
    """Keep the original item/actor,with one explicit three-outcome evidence rule."""
    if queried not in (-1, 1) or not item or not item.endswith("."):
        raise ValueError("Complete public item and signed queried actor required")
    actor = "R" if queried == 1 else "S"
    return (
        f'First-person behavioral statement: "I {item[0].lower() + item[1:]}"\n'
        f"From their identified comments, judge whether this statement fits writer {actor}'s "
        "usual behavior across occasions. Return 1 if the comments support it, "
        "0 if they contradict it, or 2 if they are insufficient to judge it.\nAnswer:"
    )


def tree_coordinates(logits):
    """Exact interior three-category chart;do not average before nonlinear splitting."""
    z = np.asarray(logits, float)
    if z.ndim < 1 or z.shape[-1] != 3 or not np.isfinite(z).all():
        raise ValueError("All three finite native category logits required")
    d = z[..., 1] - z[..., 0]
    a = logsumexp(z[..., :2], axis=-1) - z[..., 2]
    lp = z - logsumexp(z, axis=-1, keepdims=True)
    return {"direction": d, "assessability": a, "log_probabilities": lp}


def tree_log_probabilities(direction, assessability):
    """Invert the declared tree without clipping or fitted latent parameters."""
    d, a = np.asarray(direction, float), np.asarray(assessability, float)
    if d.shape != a.shape or not np.isfinite([d, a]).all():
        raise ValueError("Matching finite tree fields required")
    available = -np.logaddexp(0, -a)
    return np.stack(
        (
            available - np.logaddexp(0, d),
            available - np.logaddexp(0, -d),
            -np.logaddexp(0, a),
        ),
        -1,
    )


def readings(logits, bank):
    """Retain both role fields and complete category laws under the new contract."""
    z = np.asarray(logits, float)
    if z.shape != (48, 2, 2, 2, 50, 3):
        raise ValueError("All48 cases/eight corners/50 items/three categories required")
    if len(bank) != 50 or sorted(b["item"] for b in bank) != list(range(50)):
        raise ValueError("Every unique item address required")
    tree = tree_coordinates(z)
    result = {}
    for channel in CHANNELS:
        chars = O.character_accounts(tree[channel])
        keys = B.key_accounts(chars, bank)
        result[channel] = {
            "items": chars,
            **keys,
            "global_common": keys["common_support"].mean(-1),
        }
    return result, tree


def contacts(source, field):
    """Compute signed SOURCE-level/reader contacts without current-author labels."""
    s, f = np.asarray(source, float), np.asarray(field, float)
    if (
        s.shape != (3, 3, 48, 5)
        or f.shape[:2] != (48, 8)
        or not np.isfinite(s).all()
        or not np.isfinite(f).all()
    ):
        raise ValueError("Typed SOURCE levels/components and full role fields")
    if f.ndim == 2:
        return np.einsum("lcuq,um->lcmq", s, f) / 48
    if f.shape != (48, 8, 5):
        raise ValueError("Every query target required")
    return np.einsum("lcuq,umr->lcmqr", s, f) / 48


def cold(record):
    """Replay qualified numerical tree observations;no neural or application-label input."""
    if (
        set(record)
        != {"version", "cases", "source_components", "item_logits", "item_bank"}
        or record["version"] != "HM288-v1"
    ):
        raise ValueError("Only registered numerical tree records allowed")
    cases = record["cases"]
    if len(cases) != 48 or len({c["case_id"] for c in cases}) != 48:
        raise ValueError("Complete uniquely addressed dyads required")
    seen, owners = set(), {}
    fields = {
        "case_id",
        "writer_id",
        "partner_id",
        "window",
        "writer_comments",
        "partner_comments",
    }
    for c in cases:
        u, v, w = c["writer_id"], c["partner_id"], c["window"]
        if (
            set(c) != fields
            or not isinstance(u, str)
            or not isinstance(v, str)
            or not u
            or not v
            or u == v
            or type(w) is not int
            or w not in (0, 1)
            or (u, w) in seen
        ):
            raise ValueError("Distinct true author/window identities required")
        seen.add((u, w))
        if (
            len(c["writer_comments"]) != 2
            or len(c["partner_comments"]) != 2
            or len(set(c["writer_comments"] + c["partner_comments"])) != 4
        ):
            raise ValueError("All four distinct comment events required")
        for actor, events in ((u, c["writer_comments"]), (v, c["partner_comments"])):
            for event in events:
                if (
                    not isinstance(event, str)
                    or not event
                    or owners.get(event, (actor, w)) != (actor, w)
                ):
                    raise ValueError("True event ownership cannot change")
                owners[event] = actor, w
    if len({u for u, w in seen}) != 24 or {u for u, w in seen} != {
        c["partner_id"] for c in cases
    }:
        raise ValueError("Same24 authors in both dyad roles required")
    s = np.asarray(record["source_components"], float)
    if (
        s.shape != (3, 3, 48, 5)
        or not np.isfinite(s).all()
        or np.max(abs(s[:, 0] - s[:, 1] - s[:, 2])) > 1e-10
        or np.max(abs(s[0] - s[1] - s[2])) > 1e-10
        or np.max(abs(s.sum(2))) > 1e-10
    ):
        raise ValueError("SOURCE level/component/relative-author identity failed")
    result, tree = readings(record["item_logits"], record["item_bank"])
    return {
        "channels": {
            channel: {
                key: contacts(s, value[key]).tolist()
                for key in ("balanced_key", "common_support", "global_common")
            }
            for channel, value in result.items()
        },
        "category_probability_means": np.exp(tree["log_probabilities"])
        .mean(axis=(0, 1, 2, 3, 4))
        .tolist(),
        "not_individual_human_trait_or_true_uncertainty": True,
    }


def native_items(model, sequences, alphabet, device, batch_size=8):
    """Separate three-digit companion to sealed binary CPU64 cached observation."""
    import torch

    if batch_size < 1 or len(alphabet) != 3 or len(set(alphabet)) != 3:
        raise ValueError("Fixed positive batch size and three-category alphabet")
    length = common_prefix(sequences)
    states, logits, normalizers, top = [], [], [], []
    with torch.inference_mode():
        prefix = torch.tensor([sequences[0][:length]], dtype=torch.long, device=device)
        cached = model.model(
            input_ids=prefix, use_cache=True, return_dict=True
        ).past_key_values
        for start in range(0, len(sequences), batch_size):
            suffixes = [s[length:] for s in sequences[start : start + batch_size]]
            width = max(map(len, suffixes))
            ids = torch.zeros((len(suffixes), width), dtype=torch.long, device=device)
            mask = torch.zeros(
                (len(suffixes), length + width), dtype=torch.long, device=device
            )
            mask[:, :length] = 1
            for i, suffix in enumerate(suffixes):
                ids[i, : len(suffix)] = torch.tensor(suffix, device=device)
                mask[i, length : length + len(suffix)] = 1
            cache = deepcopy(cached)
            cache.batch_repeat_interleave(len(suffixes))
            output = model.model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            indices = torch.tensor([len(s) - 1 for s in suffixes], device=device)
            h = output.last_hidden_state[
                torch.arange(len(suffixes), device=device), indices
            ]
            full = model.lm_head(h).float()
            if not torch.isfinite(full).all() or not torch.isfinite(h).all():
                raise ValueError("Nonfinite native item observation")
            states.append(h.float().cpu().numpy())
            logits.append(full[:, alphabet].cpu().numpy())
            normalizers.append(logsumexp(full.cpu().numpy().astype(float), -1))
            top.append(full.argmax(-1).cpu().numpy())
    return {
        "states": np.concatenate(states),
        "logits": np.concatenate(logits).astype(float),
        "normalizer": np.concatenate(normalizers),
        "top": np.concatenate(top),
        "prefix_tokens": np.array(length),
    }
