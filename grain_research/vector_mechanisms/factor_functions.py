#!/usr/bin/env python3
"""Expanded query functions with distinct question-region and final-token action."""

from __future__ import annotations

import numpy as np
import torch


def joint_gradient(observer, context, mask, kwargs, alphabet, block=16):
    """Return native score and partial derivatives for P and Q with E fixed."""
    if any(p.requires_grad for p in observer.model.parameters()):
        raise ValueError("Model parameters must be frozen")
    select = np.asarray(mask, bool)
    if (
        context.ndim != 3
        or select.shape != (context.shape[1],)
        or select[-1]
        or not select.any()
    ):
        raise ValueError("P mask must exclude Q and match the state")
    positions = np.concatenate([np.flatnonzero(select), [len(select) - 1]])
    ix = torch.as_tensor(positions, device=context.device)
    observer.calls += 1
    with torch.enable_grad():
        leaf = context[:, ix].detach().clone().requires_grad_(True)
        h = torch.index_copy(context.detach(), 1, ix, leaf)
        for layer in observer.model.model.layers[block + 1 :]:
            h = layer(h, **kwargs)
            h = h[0] if isinstance(h, tuple) else h
        logits = observer.model.lm_head(observer.model.model.norm(h[:, -1:]))[
            0, -1
        ].float()
        ids = torch.as_tensor(alphabet["token_ids"], device=logits.device)
        values = torch.as_tensor(
            alphabet["values"], dtype=logits.dtype, device=logits.device
        )
        score = torch.softmax(logits[ids], dim=0) @ values
        (g,) = torch.autograd.grad(score, leaf)
    value = float(score.detach().cpu())
    gradient = g[0].detach().float().cpu().numpy().copy()
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite partial derivative")
    return value, gradient[:-1], gradient[-1]
