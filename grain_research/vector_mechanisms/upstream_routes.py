#!/usr/bin/env python3
"""Differentiable query-site routes upstream of the frozen P/Q reader.

Attention-head outputs are measured before o_proj; MLP outputs are measured
before residual addition. Donor interventions address aligned question slots,
not arbitrary equal-numbered tokens from differently sized texts.
"""

from __future__ import annotations

import numpy as np
import torch


class _BoundaryReached(Exception):
    """End the prefix before final normalization or the language-model readout."""


def tensor_score(state, mask, function):
    """Apply the exact frozen affine reader without detaching its upstream graph."""
    device, dtype = state.device, state.dtype
    cv = lambda x: torch.as_tensor(x, device=device, dtype=dtype)
    P, Q = state[0, mask], state[0, -1]
    if P.shape != function.p_ref.shape or Q.shape != function.q_ref.shape:
        raise ValueError("Reader/state coordinate mismatch")
    G = function.config_coef + function.mean_coef / len(function.p_ref)
    return (
        cv(function.anchor)
        + ((P - cv(function.p_ref)) * cv(G)).sum()
        + (Q - cv(function.q_ref)) @ cv(function.g_final)
    )


def capture_differentiable_prefix(
    observer,
    ids,
    mask,
    function,
    *,
    block=16,
    patches=None,
    gradients=True,
    alphabet=None,
):
    """Return scoped prefix activations/derivatives and optionally a native patch score.

    Parameters stay frozen. Patches map site names to complete query-slot arrays
    in their native coordinates; head-specific patches are assembled outside.
    Native scores are optional no-grad checks, not the gradient-ranking objective.
    """
    model, layers = observer.model, observer.model.model.layers
    select = np.asarray(mask, bool)
    if (
        len(select) != len(ids)
        or not select.any()
        or select[-1]
        or len(ids) > observer.max_tokens
    ):
        raise ValueError("Invalid P/Q partition")
    if not 0 <= block < len(layers) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Frozen model and valid prefix boundary required")
    if gradients and alphabet is not None:
        raise ValueError("Native comparison is a separate no-grad intervention")
    positions = np.r_[np.flatnonzero(select), len(ids) - 1]
    device = next(model.parameters()).device
    ix = torch.as_tensor(positions, device=device)
    patches = patches or {}
    sites, handles, completed, saved = {}, [], [], {}
    valid_sites = {f"{kind}:{i}" for i in range(block + 1) for kind in ("attn", "mlp")}
    if set(patches) - valid_sites:
        raise ValueError("Unknown upstream intervention site")

    def record(name, value):
        """Preserve the graph and replace only declared question-region coordinates."""
        if name in patches:
            donor = torch.as_tensor(
                patches[name], dtype=value.dtype, device=value.device
            )
            if donor.shape != (len(positions), value.shape[-1]) or not bool(
                torch.isfinite(donor).all()
            ):
                raise ValueError("Donor shape or finiteness mismatch")
            value = torch.index_copy(value, 1, ix, donor[None])
        sites[name] = value
        return value

    def attention(name):
        def hook(_module, inputs):
            return (record(name, inputs[0]), *inputs[1:])

        return hook

    def mlp(name):
        def hook(_module, _inputs, output):
            return record(name, output)

        return hook

    def boundary(index):
        def hook(_module, _inputs, output):
            completed.append(index)
            if index == block:
                saved["state"] = output[0] if isinstance(output, tuple) else output
                if alphabet is None:
                    raise _BoundaryReached

        return hook

    tokens = torch.tensor([ids], device=device)
    observer.calls += 1
    native = None
    try:
        for i in range(block + 1):
            handles.append(
                layers[i].self_attn.o_proj.register_forward_pre_hook(
                    attention(f"attn:{i}")
                )
            )
            handles.append(layers[i].mlp.register_forward_hook(mlp(f"mlp:{i}")))
            handles.append(layers[i].register_forward_hook(boundary(i)))
        with torch.set_grad_enabled(gradients):
            embedded = (
                model.model.embed_tokens(tokens).detach().requires_grad_(gradients)
            )
            try:
                kwargs = {
                    "inputs_embeds": embedded,
                    "attention_mask": torch.ones_like(tokens),
                    "use_cache": False,
                    "return_dict": True,
                }
                if alphabet is None:
                    model.model(**kwargs)
                else:
                    from .direct_traits import native_readout

                    result = model(logits_to_keep=1, **kwargs)
                    native = native_readout(
                        result.logits[0, -1].detach().float().cpu().numpy(), alphabet
                    )
            except _BoundaryReached:
                pass
            if completed != list(range(block + 1)) or "state" not in saved:
                raise RuntimeError("Prefix termination trace changed")
            value = tensor_score(saved["state"], select, function)
            derivatives = (
                torch.autograd.grad(value, [embedded, *sites.values()])
                if gradients
                else None
            )
            result = {
                "score": float(value.detach().cpu()),
                "P": saved["state"][0, select].detach().float().cpu().numpy().copy(),
                "Q": saved["state"][0, -1].detach().float().cpu().numpy().copy(),
                "sites": {
                    name: tensor[0, ix].detach().float().cpu().numpy().copy()
                    for name, tensor in sites.items()
                },
                "trace": completed,
                "native": native,
                "gradients": {},
                "input_gradient": None,
            }
            if gradients:
                result["input_gradient"] = (
                    derivatives[0][0].detach().float().cpu().numpy().copy()
                )
                result["gradients"] = {
                    name: g[0, ix].detach().float().cpu().numpy().copy()
                    for name, g in zip(sites, derivatives[1:], strict=True)
                }
    finally:
        for handle in handles:
            handle.remove()
    if not np.isfinite(result["score"]):
        raise ValueError("Nonfinite upstream readout")
    return result


def rank_route_candidates(recipient, donor, head_dim):
    """First-order donor effects per attention head or MLP; not causal proof by ranking."""
    rows = []
    for site, activation in recipient["sites"].items():
        delta = donor["sites"][site] - activation
        gradient = recipient["gradients"][site]
        if delta.shape != gradient.shape:
            raise ValueError("Matched question-slot effects required")
        kind, layer = site.split(":")
        if kind == "attn":
            if delta.shape[1] % head_dim:
                raise ValueError("Nonintegral attention head count")
            for h in range(delta.shape[1] // head_dim):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                rows.append(
                    {
                        "site": site,
                        "head": h,
                        "layer": int(layer),
                        "kind": kind,
                        "effect_estimate": float(
                            np.sum(delta[:, sl] * gradient[:, sl])
                        ),
                    }
                )
        else:
            rows.append(
                {
                    "site": site,
                    "head": -1,
                    "layer": int(layer),
                    "kind": kind,
                    "effect_estimate": float(np.sum(delta * gradient)),
                }
            )
    return sorted(
        rows, key=lambda r: (-abs(r["effect_estimate"]), r["site"], r["head"])
    )


def assemble_patch(recipient, donor, candidates, head_dim, alpha=1.0):
    """Build simultaneous matched-slot patches with additive head edits, no in-place cache mutation."""
    patches, used = {}, set()
    for c in candidates:
        site, head = c["site"], c["head"]
        if (site, head) in used:
            raise ValueError("Duplicate route candidate")
        used.add((site, head))
        if site not in patches:
            patches[site] = recipient["sites"][site].copy()
        sl = (
            slice(None) if head == -1 else slice(head * head_dim, (head + 1) * head_dim)
        )
        patches[site][:, sl] += alpha * (
            donor["sites"][site][:, sl] - recipient["sites"][site][:, sl]
        )
    return patches
