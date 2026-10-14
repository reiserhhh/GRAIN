#!/usr/bin/env python3
"""Construct focus-conditioned observations and qualified response fields."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_recursive_collector as C


def compile_histories(histories, seeds, cos, sin, valid, weights, config, device):
    """Recurse only current tokens from immutable causal history primitives.

    Histories contain batch/layer/token/KV-head/coordinate arrays. No observed
    current state or response is an input. All pretrained maps remain explicit
    dependencies; this function is not a model-independent psychological law.
    """
    import torch
    from torch.nn import functional as f

    k, v = (np.asarray(histories[name]) for name in ("k", "v"))
    seeds, cos, sin, valid = (np.asarray(x) for x in (seeds, cos, sin, valid))
    blocks = len(weights) - 1
    if (
        k.shape != v.shape
        or k.ndim != 5
        or k.shape[1] != blocks
        or seeds.ndim != 3
        or valid.shape != (k.shape[0], k.shape[2])
        or cos.shape != sin.shape
        or len(cos) != k.shape[2] + seeds.shape[1]
        or not valid.any(1).all()
        or not all(np.isfinite(x).all() for x in (k, v, seeds, cos, sin))
    ):
        raise ValueError("Complete finite histories, fixed seeds and support required")
    batch, _, context, _, _ = k.shape
    focuses, length, hidden = seeds.shape
    h = torch.as_tensor(seeds, device=device)[None].expand(batch, -1, -1, -1)
    h = h.reshape(batch * focuses, length, hidden).clone()
    cc, ss = (torch.as_tensor(x, device=device) for x in (cos, sin))
    support = np.concatenate(
        (
            np.broadcast_to(valid[:, None], (batch, length, context)),
            np.broadcast_to(np.tri(length, dtype=bool), (batch, length, length)),
        ),
        axis=-1,
    )
    support = np.repeat(support, focuses, axis=0)
    for layer in range(blocks):
        history_k, history_v = (
            torch.as_tensor(np.repeat(x[:, layer], focuses, axis=0), device=device)
            for x in (k, v)
        )
        h = C.current_block(
            h,
            history_k,
            history_v,
            torch.as_tensor(support, device=device),
            cc,
            ss,
            weights[layer],
            config,
        )
    x = C.rms_norm(h, weights[-1]["input_layernorm.weight"], config["rms_norm_eps"])
    dim = config["hidden_size"] // config["num_attention_heads"]
    outputs = []
    for name in ("k", "v"):
        a = f.linear(
            x,
            weights[-1][f"self_attn.{name}_proj.weight"],
            weights[-1][f"self_attn.{name}_proj.bias"],
        ).reshape(batch * focuses, length, config["num_key_value_heads"], dim)
        if name == "k":
            a = C.rotary(a, cc[context:], ss[context:])
        outputs.append(
            a.detach().cpu().numpy().reshape(batch, focuses, length, -1, dim)
        )
    return tuple(outputs)


def response_fields(readings):
    """Keep all actor/focus sectors, target vectors and direct pairing responses."""
    n = np.asarray(readings, float)
    if n.shape != (3, 2, 6, 2, 2, 2, 2, 2, 6) or not np.isfinite(n).all():
        raise ValueError("Complete v/w/f/c/b/o/focus/mode/query field required")
    gain = (n[:, :, :, :, 1] - n[:, :, :, :, 0]) / 2
    even = (gain[:, :, :, :, 1] + gain[:, :, :, :, 0]) / 2
    odd = (gain[:, :, :, :, 1] - gain[:, :, :, :, 0]) / 2
    result = {"gain": gain, "actor_even": even, "actor_odd": odd}
    for name, values in (("actor_even", even), ("actor_odd", odd)):
        result[name + "_focus_even"] = (values[..., 0, :, :] + values[..., 1, :, :]) / 2
        result[name + "_focus_odd"] = (values[..., 0, :, :] - values[..., 1, :, :]) / 2
    result["gamma"] = -2 * (gain[..., 0, :, :] - gain[..., 1, :, :])
    # Equal six-family mass. Counts total8 per condition; DeltaC is twice a cycle.
    result["qualified_family_pairing"] = result["gamma"][:, :, :, :, 1].mean(3) / 24
    result["other_actor_pairing_diagnostic"] = (
        result["gamma"][:, :, :, :, 0].mean(3) / 24
    )
    return result


def verify_response_identities(readings, fields):
    """Use explicit sign loops and count integrations, not the extraction helper."""
    n = np.asarray(readings, float)
    reconstruction = pairing_error = gamma_error = 0.0
    plus, minus = np.array([[3, 1], [1, 3]]), np.array([[1, 3], [3, 1]])
    for v, w, family, c in np.ndindex(3, 2, 6, 2):
        for o, r in np.ndindex(2, 2):
            predicted = sum(
                (2 * o - 1 if actor == "odd" else 1)
                * (1 - 2 * r if focus == "odd" else 1)
                * fields[f"actor_{actor}_focus_{focus}"][v, w, family, c]
                for actor in ("even", "odd")
                for focus in ("even", "odd")
            )
            actual = (n[v, w, family, c, 1, o, r] - n[v, w, family, c, 0, o, r]) / 2
            reconstruction = max(reconstruction, float(np.max(abs(actual - predicted))))
        for o in range(2):
            table = n[v, w, family, c, :, o]
            gamma = table[0, 0] - table[0, 1] - table[1, 0] + table[1, 1]
            gamma_error = max(
                gamma_error,
                float(np.max(abs(gamma - fields["gamma"][v, w, family, c, o]))),
            )
        direct = np.zeros((2, 6))
        for c2, b, r in np.ndindex(2, 2, 2):
            direct += (
                (plus[b, r] - minus[b, r]) * n[v, w, family, c2, b, 1, r] / (8 * 2 * 6)
            )
        pairing_error = max(
            pairing_error,
            float(
                np.max(abs(direct - fields["qualified_family_pairing"][v, w, family]))
            ),
        )
    return {
        "sector_reconstruction": reconstruction,
        "gamma_corners": gamma_error,
        "qualified_count_integral": pairing_error,
    }
