#!/usr/bin/env python3
"""Fixed public-item response operators, with key and endorsement kept separate."""

from copy import deepcopy

import numpy as np
from scipy.special import expit


def public_bodies(decoded, count=8):
    """Recover exactly the registered public eight-excerpt body without cleanup."""
    start = "The excerpts below were written by one anonymous writer.\n\nExcerpt 1:\n"
    stop = "\nEvidence ends here.\n\nQuestion: "
    if count != 8 or decoded.count(start) != 1 or decoded.count(stop) != 1:
        raise ValueError("Exact eight-excerpt public template required")
    tail = decoded.split(start, 1)[1].split(stop, 1)[0]
    bodies = []
    for k in range(2, count + 1):
        marker = f"\n\nExcerpt {k}:\n"
        if tail.count(marker) != 1:
            raise ValueError("Missing or ambiguous public event boundary")
        body, tail = tail.split(marker, 1)
        bodies.append(body)
    if "\n\nExcerpt " in tail:
        raise ValueError("Unexpected extra public event")
    bodies.append(tail)
    if any(not b for b in bodies):
        raise ValueError("Empty public excerpt")
    return bodies


def keyed_observations(logits, items, targets):
    """Return every pole, balanced key and equal-item key without latent fitting."""
    z = np.asarray(logits, float)
    if (
        z.ndim < 2
        or z.shape[-2:] != (len(items), 2)
        or not np.isfinite(z).all()
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("Complete finite item logits and distinct targets required")
    odds = z[..., 1] - z[..., 0]
    probability = expit(odds)
    results = {"item_probability": probability, "item_log_odds": odds}
    for kind, values in (("probability", probability), ("log_odds", odds)):
        by_target = []
        for target in targets:
            high = [
                i
                for i, r in enumerate(items)
                if r["target"] == target and r["pole"] == "high"
            ]
            low = [
                i
                for i, r in enumerate(items)
                if r["target"] == target and r["pole"] == "low"
            ]
            if len(high) + len(low) != 10 or not high or not low:
                raise ValueError("A complete ten-item, two-pole key is required")
            h, l = values[..., high].mean(-1), values[..., low].mean(-1)
            offset = 0.5 if kind == "probability" else 0.0
            balanced = offset + (h - l) / 2
            common = (h + l) / 2
            imbalance = (len(high) - len(low)) / 10
            official = balanced + imbalance * (common - offset)
            by_target.append(np.stack((h, l, balanced, common, official), -1))
        results[kind] = np.stack(by_target, -2)
    return results


def law_actions(values, weights):
    """Bind whole-author item observations to two unchanged positive evidence laws."""
    y, p = np.asarray(values, float), np.asarray(weights, float)
    if (
        y.ndim < 2
        or p.shape != (y.shape[0], 2, y.shape[1])
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(-1), 1, atol=1e-9, rtol=0)
    ):
        raise ValueError(
            "Complete matching observations and positive normalized laws required"
        )
    levels = np.einsum("uak,uk...->ua...", p, y)
    return levels, levels[:, 0] - levels[:, 1]


def common_prefix(sequences):
    """Find only the exactly equal causal prefix; never merge unequal evidence."""
    if not sequences or any(not s for s in sequences):
        raise ValueError("Nonempty native sequences required")
    length = 0
    for tokens in zip(*sequences, strict=False):
        if len(set(tokens)) != 1:
            break
        length += 1
    if length == 0 or any(length == len(s) for s in sequences):
        raise ValueError("Shared prefix and nonempty distinct suffixes required")
    return length


def native_items(model, sequences, alphabet, device, batch_size=4):
    """Reuse an exact prefix with independent suffix caches, preserving native inference."""
    import torch

    if batch_size < 1 or len(alphabet) != 2 or len(set(alphabet)) != 2:
        raise ValueError("Fixed positive batch size and binary alphabet required")
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
            normalizers.append(torch.logsumexp(full.double(), -1).cpu().numpy())
            top.append(full.argmax(-1).cpu().numpy())
    return {
        "states": np.concatenate(states),
        "logits": np.concatenate(logits).astype(float),
        "normalizer": np.concatenate(normalizers),
        "top": np.concatenate(top),
        "prefix_tokens": np.array(length),
    }
