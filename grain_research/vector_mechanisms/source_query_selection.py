#!/usr/bin/env python3
"""Source query anchors for current evidence selection, without current primal Q."""

import numpy as np

from .compiled_query_transfer import QUERIES, validate_sources
from .fixed_seed_attention import rotary
from .upstream_attention_channels import normalized_directions

ARMS = ("source_own_Q", "source_other_Q", "source_shared_Q")
FIELDS = {"k", "v", "dk", "cos", "sin", "mask", "g", "scale"}


def evidence_inputs(capture, weights, shared):
    """Export pure K/V/dK and projected fixed readouts, never current Q or full x."""
    w = {k: np.asarray(v, float) for k, v in weights.items()}
    z, dz = normalized_directions(
        capture["x"], capture["d"], w["gamma"], float(w["epsilon"])
    )
    tokens, hd = len(z), int(w["head_dim"])
    kv = w["k"].shape[0] // hd
    return {
        "k": (z @ w["k"].T + w["k_bias"]).reshape(tokens, kv, hd),
        "v": (z @ w["v"].T + w["v_bias"]).reshape(tokens, kv, hd),
        "dk": (dz @ w["k"].T).reshape(len(dz), tokens, kv, hd),
        "cos": capture["cos"],
        "sin": capture["sin"],
        "mask": capture["mask"],
        "g": (np.asarray(shared, float) @ w["o"]).reshape(2, -1, hd),
        "scale": w["scale"],
    }


def source_query(x, weights):
    """Project a source-only last state before applying its recipient's RoPE."""
    w = weights
    x = np.asarray(x, float)
    if x.shape != w["gamma"].shape or not np.isfinite(x).all():
        raise ValueError("Finite source last-position state required")
    z = x * w["gamma"] / np.sqrt(np.mean(x * x) + float(w["epsilon"]))
    return (z @ w["q"].T + w["q_bias"]).reshape(-1, int(w["head_dim"]))


def anchors(bank, cohort, index, query):
    """Compile own/other/shared averages from the exact independent four-source bank."""
    slots = list(zip(bank["index"].tolist(), bank["query"].tolist(), strict=True))
    if (
        len(slots) != 8
        or set(slots) != {(i, q) for i in range(4) for q in QUERIES}
        or query not in QUERIES
    ):
        raise ValueError("Exact paired source bank and known query required")
    source = [i for i in range(4) if cohort != "source4" or i != index]
    validate_sources(cohort, index, source)
    q = np.asarray(bank["q"], float)
    if q.ndim != 3 or len(q) != 8 or not np.isfinite(q).all():
        raise ValueError("Finite pre-RoPE query bank required")
    averages = [
        q[[i for i, (s, n) in enumerate(slots) if s in source and n == name]].mean(0)
        for name in QUERIES
    ]
    own = QUERIES.index(query)
    return np.stack(
        (averages[own], averages[1 - own], (averages[0] + averages[1]) / 2)
    ), source


def key_actions(query_anchors, evidence):
    """Recompute competition and centered values for every anchor and all heads.

    Returns [anchor, mode, factor] actions. K/V/dK are pre-RoPE; the supplied
    cosine/sine rows place both anchors and keys in the recipient's coordinates.
    """
    if set(evidence) != FIELDS:
        raise ValueError("Only pure evidence fields permitted; no current Q or target")
    e = {k: np.asarray(v, float) for k, v in evidence.items()}
    q = np.asarray(query_anchors, float)
    k, v, dk, g = (e[n] for n in ("k", "v", "dk", "g"))
    if (
        q.ndim != 3
        or k.ndim != 3
        or v.shape != k.shape
        or dk.ndim != 4
        or dk.shape[1:] != k.shape
        or min(k.shape) < 1
        or min(q.shape) < 1
        or q.shape[2] != k.shape[2]
        or q.shape[1] % k.shape[1]
        or g.shape != (2, q.shape[1], q.shape[2])
        or e["cos"].shape != (len(k), k.shape[2])
        or e["sin"].shape != e["cos"].shape
        or e["mask"].shape != (len(k),)
        or np.isnan(e["mask"]).any()
        or np.isposinf(e["mask"]).any()
        or not np.isfinite(e["mask"]).any()
        or float(e["scale"]) <= 0
        or not np.isfinite(q).all()
        or not all(np.isfinite(v).all() for n, v in e.items() if n != "mask")
    ):
        raise ValueError(
            "Aligned finite source-query and recipient evidence arrays required"
        )
    cos, sin = e["cos"], e["sin"]
    q = rotary(q, cos[-1], sin[-1])
    k = rotary(k, cos[:, None], sin[:, None])
    dk = rotary(dk, cos[None, :, None], sin[None, :, None])
    result = np.zeros((len(q), 2, len(dk)))
    groups, scale = q.shape[1] // k.shape[1], float(e["scale"])
    for a in range(len(q)):
        for h in range(q.shape[1]):
            kv = h // groups
            scores = k[:, kv] @ q[a, h] * scale + e["mask"]
            attention = np.exp(scores - scores.max())
            attention /= attention.sum()
            output = attention @ v[:, kv]
            centered_value = (v[:, kv] - output) @ g[:, h].T
            ds = np.einsum("fth,h->ft", dk[:, :, kv], q[a, h]) * scale
            result[a] += ((ds * attention) @ centered_value).T
    return result
