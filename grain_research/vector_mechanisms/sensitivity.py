"""Local decoder derivatives and bounded, task-conditioned state perturbations."""

from __future__ import annotations

import numpy as np
import torch


def local_margin_gradient(observer, ids, site, yes, no, *, offload=True):
    """Differentiate one pre-answer margin with respect to a block's states.

    Model parameters are temporarily frozen, not updated. Only the selected
    block output becomes a leaf; no gradient crosses into earlier blocks.
    Saved activations can be offloaded to CPU to bound GPU memory. The caller
    must replay the forward margin against the non-gradient instrument.
    """
    if site not in observer.sites or site in ("input_embedding", "final_norm"):
        raise ValueError("Use a decoder block output")
    if not ids or len(ids) > (observer.pad_to_length or observer.max_tokens):
        raise ValueError("Invalid prompt length")
    if yes == no:
        raise ValueError("Need distinct candidate tokens")
    parameters = list(observer.model.parameters())
    flags = [p.requires_grad for p in parameters]
    device = parameters[0].device
    length = observer.pad_to_length or len(ids)
    pad = getattr(observer.tokenizer, "pad_token_id", None)
    pad = 0 if pad is None else pad
    tokens = torch.tensor([ids + [pad] * (length - len(ids))], device=device)
    mask = torch.tensor([[1] * len(ids) + [0] * (length - len(ids))], device=device)
    leaves = []

    def make_leaf(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        leaf = hidden.detach().requires_grad_(True)
        leaves.append(leaf)
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    handle = None
    try:
        for p in parameters:
            p.requires_grad_(False)
        handle = observer.sites[site].register_forward_hook(make_leaf)
        observer.calls += 1
        observer.input_tokens += len(ids)
        observer.processed_tokens += length
        from contextlib import nullcontext

        storage = (
            torch.autograd.graph.save_on_cpu(pin_memory=device.type == "cuda")
            if offload
            else nullcontext()
        )
        with torch.enable_grad(), storage:
            output = observer.model(
                input_ids=tokens,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[0, len(ids) - 1].float()
            margin = logits[yes] - logits[no]
            if len(leaves) != 1:
                raise RuntimeError("Expected one invocation of the selected block")
            (gradient,) = torch.autograd.grad(margin, leaves[0])
            state = leaves[0][0, : len(ids)].detach().float().cpu().numpy().copy()
            grad = gradient[0, : len(ids)].detach().float().cpu().numpy().copy()
            score = float(margin.detach())
        if not np.isfinite(grad).all() or not np.isfinite(state).all():
            raise RuntimeError("Non-finite local derivative")
        return state, grad, score
    finally:
        if handle is not None:
            handle.remove()
        for p, flag in zip(parameters, flags, strict=True):
            p.requires_grad_(flag)


def unit_matrix_direction(matrix):
    """Return a Frobenius-unit direction, retaining zero derivatives explicitly."""
    matrix = np.asarray(matrix, dtype=float)
    norm = float(np.linalg.norm(matrix))
    return matrix / norm if norm > 0 else np.zeros_like(matrix)


def local_prediction(gradient, current, injected):
    """Use actual injected values, not their higher-precision requested precursor."""
    arrays = [np.asarray(x, dtype=float) for x in (gradient, current, injected)]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("Local derivative and states must align")
    return float(np.sum(arrays[0] * (arrays[2] - arrays[1])))


def region_derivative_stats(state, gradient, regions):
    """Describe one coordinate system; derivative fractions are not causal shares."""
    state, gradient = np.asarray(state, dtype=float), np.asarray(gradient, dtype=float)
    if state.shape != gradient.shape or state.ndim != 2:
        raise ValueError("State and derivative must align")
    indices = [i for values in regions.values() for i in values]
    if sorted(indices) != list(range(len(state))):
        raise ValueError("Regions must partition the actual prompt exactly")
    total = float(np.sum(gradient**2))
    rows = []
    for region, positions in regions.items():
        norm = float(np.linalg.norm(gradient[positions]))
        rows.append(
            {
                "region": region,
                "tokens": len(positions),
                "gradient_norm": norm,
                "gradient_energy_fraction": norm**2 / max(total, 1e-30),
                "gradient_norm_per_sqrt_token": norm / np.sqrt(max(1, len(positions))),
                "state_norm": float(np.linalg.norm(state[positions])),
            }
        )
    return rows
