#!/usr/bin/env python3
"""Source-only downstream directions and forward GRAIN attention responses."""

from __future__ import annotations

import numpy as np

from .attention_pullback import attention_pullback
from .fixed_seed_attention import decompose_attention, rotary
from .native_material_reader import direction

QUERIES = ("neuroticism", "confirmation_seeking")
INPUT_KEYS = {"q", "k", "v", "cos", "sin", "hidden", "grain", "positions"}


def source_means(source, query, held=None):
    """Use only the fixed four-source cohort, excluding a declared held author."""
    if query not in QUERIES or (held is not None and held not in range(4)):
        raise ValueError("Unknown query or held source index")
    if set(source) != {(i, q) for i in range(4) for q in QUERIES}:
        raise ValueError("Exactly four complete source authors required")
    indices = [i for i in range(4) if i != held]
    own = np.stack([source[i, query] for i in indices])
    pooled = np.stack([source[i, q] for i in indices for q in QUERIES])
    if (
        own.ndim != 3
        or pooled.shape[1:] != own.shape[1:]
        or not np.isfinite(pooled).all()
    ):
        raise ValueError("Finite matched head cotangents required")
    return {
        "own_g": own.mean(0),
        "pooled_g": pooled.mean(0),
        "source_indices": np.asarray(indices),
    }


def directional_responses(inp, wk, wv):
    """Forward JVPs from each nonquery GRAIN direction to every head's output.

    Returned [factor,head,dimension] arrays do not use a cotangent, output score
    or gradient. They are dual to contracting a pulled-back field with D.
    """
    if set(inp) != INPUT_KEYS:
        raise ValueError("Inference accepts only forward inputs and GRAIN coordinates")
    q, k, v, cos, sin, hidden, grain, pos = (
        np.asarray(inp[n])
        for n in ("q", "k", "v", "cos", "sin", "hidden", "grain", "positions")
    )
    if (
        pos.ndim != 1
        or not np.issubdtype(pos.dtype, np.integer)
        or len(pos) != len(hidden)
        or np.any(np.diff(pos) <= 0)
        or pos.min() < 0
        or pos.max() >= len(k) - 1
    ):
        raise ValueError("Strictly ordered nonquery body positions required")
    dirs = np.stack([direction(hidden, grain, f)[0] for f in range(grain.shape[1])])
    heads, d = q.shape
    groups = heads // k.shape[1]
    qr = rotary(q, cos[-1], sin[-1])
    kr = rotary(k, cos[:, None], sin[:, None])
    masses, outputs = [], []
    for h in range(heads):
        r = attention_pullback(
            qr[h], kr[:, h // groups], v[:, h // groups], np.zeros(d)
        )
        masses.append(r["attention"])
        outputs.append(r["output"])
    lv, lk = np.zeros((len(dirs), heads, d)), np.zeros((len(dirs), heads, d))
    for f, delta in enumerate(dirs):
        dv = (delta @ wv.T).reshape(len(pos), k.shape[1], d)
        dk = rotary(
            (delta @ wk.T).reshape(len(pos), k.shape[1], d),
            cos[pos, None],
            sin[pos, None],
        )
        for h in range(heads):
            b = h // groups
            a = masses[h][pos, None]
            ds = dk[:, b] @ qr[h] / np.sqrt(d)
            lv[f, h] = (a * dv[:, b]).sum(0)
            lk[f, h] = (a * (v[pos, b] - outputs[h]) * ds[:, None]).sum(0)
    return dirs, lv, lk


def predict_attention(inp, model, weights):
    """Compute two source programs and their value/key components, without targets."""
    if set(model) != {"own_g", "pooled_g", "source_indices"} or set(weights) != {
        "k",
        "v",
    }:
        raise ValueError("Fixed source model and projection schema required")
    dirs, lv, lk = directional_responses(inp, weights["k"], weights["v"])
    result = {"response_value": lv, "response_key": lk}
    for mode in ("own", "pooled"):
        g = model[mode + "_g"]
        fields = decompose_attention(
            inp["q"],
            inp["k"],
            inp["v"],
            g,
            inp["cos"],
            inp["sin"],
            weights["k"],
            weights["v"],
        )
        for channel, response in (("value", lv), ("key", lk)):
            field = fields[channel + "_field"][inp["positions"]]
            actions = np.einsum("fhd,hd->f", response, g)
            from_field = np.einsum("fth,th->f", dirs, field)
            if not np.allclose(actions, from_field, rtol=1e-10, atol=1e-10):
                raise ValueError("Forward/pullback duality failed")
            result[mode + "_" + channel] = field
            result[mode + "_" + channel + "_actions"] = actions
    return result
