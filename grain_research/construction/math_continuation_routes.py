#!/usr/bin/env python3
"""Same-memory graph interventions for complete autoregressive observations."""

import numpy as np

from grain_research.construction import math_late_condition as L

ROUTES = {"closed": (0, 0), "direct": (0, 1), "relay": (1, 0), "native": (1, 1)}


def addresses(suffix_lengths, target_length):
    """Include the last suffix row that predicts the FIRST target token."""
    if not suffix_lengths or min(suffix_lengths) < 1 or target_length < 1:
        raise ValueError("Nonempty suffixes and complete target required")
    width = max(suffix_lengths) + target_length - 1
    rows = []
    for length in suffix_lengths:
        start = width - (length + target_length - 1)
        first_prediction = start + length - 1
        rows.append(
            (list(range(start, first_prediction)), list(range(first_prediction, width)))
        )
    return width, rows


def cut_mask(mask, prefix_length, body_start, suffix_lengths, target_length, route):
    """Close ONLY context-prefix edges into declared relay/readout rows."""
    import torch

    if route not in ROUTES or not 0 < body_start < prefix_length:
        raise ValueError("Known route and nonempty addressed context required")
    width, rows = addresses(suffix_lengths, target_length)
    if (
        mask is None
        or mask.ndim != 4
        or mask.shape[0] != len(rows)
        or mask.shape[-2:] != (width, prefix_length + width)
    ):
        raise ValueError("Addressed cached additive causal mask required")
    relay, direct = ROUTES[route]
    if relay and direct:
        return mask
    changed = mask.clone()
    body = torch.arange(body_start, prefix_length, device=mask.device)
    for i, (r, d) in enumerate(rows):
        closed = ([] if relay else r) + ([] if direct else d)
        if closed:
            dest = torch.tensor(closed, device=mask.device, dtype=torch.long)
            changed[i, :, dest[:, None], body[None, :]] = -torch.inf
    return changed


def restore_cache(observer, arrays):
    """Restore a sealed prefix without re-encoding text or fitting an answer."""
    import torch
    from transformers.cache_utils import DynamicCache

    k, v = np.asarray(arrays["keys"]), np.asarray(arrays["values"])
    if k.shape != v.shape or k.ndim != 4 or not np.isfinite(k).all():
        raise ValueError("Finite layer x KV-head x prefix x head arrays required")
    if not np.isfinite(v).all() or k.shape[0] != len(observer.model.model.layers):
        raise ValueError("Every pinned decoder layer must be present")
    cache = DynamicCache(config=observer.model.config)
    for layer in range(len(k)):
        cache.update(
            torch.tensor(k[layer][None], device=observer.model.device),
            torch.tensor(v[layer][None], device=observer.model.device),
            layer,
        )
    return cache


def observe(observer, cache, suffixes, target, body_start, route):
    """Observe a complete text with an all-layer cut and a restored global hook."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    if observer.model.config._attn_implementation != "eager":
        raise ValueError("Inspected eager attention is required")
    prefix = cache.get_seq_length()
    lengths = list(map(len, suffixes))
    _, rows = addresses(lengths, len(target))
    if route not in ROUTES or not 0 < body_start < prefix:
        raise ValueError("Unknown route or invalid source support")
    original, receipts, calls = qwen.eager_attention_forward, [], []
    layers = list(cache)

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Intervene before softmax; retain renormalization and recurrence."""
        changed = cut_mask(
            attention_mask, prefix, body_start, lengths, len(target), route
        )
        output, weights = original(
            module, query, key, value, changed, scaling=scaling, **kwargs
        )
        layer = module.layer_idx
        source_error = max(
            float((now[:, :, :prefix] - before).abs().max().item())
            for now, before in zip((key, value), layers[layer][:2], strict=True)
        )
        maxima = [0.0, 0.0]
        for group, enabled in enumerate(ROUTES[route]):
            if not enabled:
                for i, positions in enumerate(rows):
                    selected = positions[group]
                    if selected:
                        maxima[group] = max(
                            maxima[group],
                            float(
                                weights[i, :, selected, body_start:prefix]
                                .abs()
                                .max()
                                .item()
                            ),
                        )
        if source_error != 0 or max(maxima) != 0:
            raise ArithmeticError("Sealed prefix or forbidden edge changed")
        receipts.append(
            {
                "layer": layer,
                "prefix_error": source_error,
                "closed_relay_weight": maxima[0],
                "closed_direct_weight": maxima[1],
            }
        )
        calls.append(layer)
        return output, weights

    try:
        qwen.eager_attention_forward = interface
        result = L.observe_suffixes(observer, cache, suffixes, target)
    finally:
        qwen.eager_attention_forward = original
    if calls != list(range(len(observer.model.model.layers))):
        raise ValueError("Each declared layer must act exactly once")
    result["receipts"] = receipts
    return result


def decompose(values):
    """Keep all signed routes; joint is NOT the sum of two isolated readers."""
    if set(values) != set(ROUTES):
        raise ValueError("All four routes are required; no missing-cell imputation")
    v = {key: np.asarray(value, float) for key, value in values.items()}
    if len({a.shape for a in v.values()}) != 1 or not all(
        np.isfinite(a).all() for a in v.values()
    ):
        raise ValueError("Aligned finite route observations required")
    zero, d, r, n = (v[k] for k in ROUTES)
    return {
        "closed": zero,
        "direct_increment": d - zero,
        "relay_increment": r - zero,
        "interaction": n - r - d + zero,
        "native": n,
        "relay_with_direct": n - d,
        "direct_with_relay": n - r,
    }


def contacts(source, effect):
    """Author-centered SOURCE contact; not r, accuracy or information share."""
    x, y = np.asarray(source, float), np.asarray(effect, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Aligned author matrices required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("No nonfinite author values")
    return (x - x.mean(0)).T @ (y - y.mean(0)) / len(x)


def diagonal_intervals(source, effect, seed, draws=999):
    """Resample authors jointly, not tokens or two directional uses separately."""
    x, y = np.asarray(source, float), np.asarray(effect, float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Same-target author coordinates required")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        pick = rng.integers(0, len(x), len(x))
        samples.append(np.diag(contacts(x[pick], y[pick])))
    return np.quantile(samples, [0.025, 0.975], axis=0).T
