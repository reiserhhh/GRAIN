#!/usr/bin/env python3
"""Mean/retained-state Jacobians and source-only affine baseline reading."""

from __future__ import annotations

import numpy as np

from .joint_readout import centered_logits


def contract(jacobian, displacement):
    """Contract candidate x coordinate Jacobians in their declared coordinates."""
    j, d = np.asarray(jacobian, float), np.asarray(displacement, float)
    if j.ndim != d.ndim + 1 or j.shape[1:] != d.shape:
        raise ValueError("Candidate Jacobian and displacement coordinates differ")
    if not np.isfinite(j).all() or not np.isfinite(d).all():
        raise ValueError("Finite derivative coordinates required")
    return j.reshape(len(j), -1) @ d.ravel()


def aggregate_jacobian(hidden_jacobian, groups, rest):
    """Sum repeated-mean token derivatives, retaining every aligned scaffold token."""
    j = np.asarray(hidden_jacobian, float)
    if j.ndim != 3 or not np.isfinite(j).all():
        raise ValueError("Finite candidate x token x hidden Jacobian required")
    positions = np.concatenate([*groups, rest])
    if (
        len(positions) != len(set(positions))
        or positions.min() < 0
        or positions.max() >= j.shape[1]
    ):
        raise ValueError("Disjoint in-range body/retained positions required")
    return np.stack([j[:, g].sum(1) for g in groups], 1), j[:, rest].copy()


def source_affine(means, retained, z0, jm, jr):
    """Compile the equal average of complete local affine source functions."""
    m, r, z, gm, gr = (np.asarray(x, float) for x in (means, retained, z0, jm, jr))
    if (
        m.ndim != 3
        or r.ndim != 3
        or z.ndim != 2
        or gm.shape != (len(m), z.shape[1], *m.shape[1:])
        or gr.shape != (len(r), z.shape[1], *r.shape[1:])
    ):
        raise ValueError("Aligned source states and candidate Jacobians required")
    if len({len(x) for x in (m, r, z, gm, gr)}) != 1 or not all(
        np.isfinite(x).all() for x in (m, r, z, gm, gr)
    ):
        raise ValueError("Finite equally sized source data required")
    intercepts = np.array(
        [
            zz - contract(a, mm) - contract(b, rr)
            for mm, rr, zz, a, b in zip(m, r, z, gm, gr, strict=True)
        ]
    )
    return {
        "intercept": centered_logits(intercepts.mean(0)),
        "jm": gm.mean(0),
        "jr": gr.mean(0),
        "source_m": m.copy(),
        "source_r": r.copy(),
        "source_z": z.copy(),
        "source_jm": gm.copy(),
        "source_jr": gr.copy(),
    }


def predict_affine(means, retained, model):
    """Return a complete baseline and source-specific directional accounts, no endpoints."""
    m, r = np.asarray(means, float), np.asarray(retained, float)
    if m.ndim != 3 or r.ndim != 3 or len(m) != len(r):
        raise ValueError("Matched author input matrices required")
    z, m_parts, r_parts = [], [], []
    for a, b in zip(m, r, strict=True):
        ma = np.array(
            [
                contract(g, a - s)
                for g, s in zip(model["source_jm"], model["source_m"], strict=True)
            ]
        )
        ra = np.array(
            [
                contract(g, b - s)
                for g, s in zip(model["source_jr"], model["source_r"], strict=True)
            ]
        )
        z.append(
            centered_logits(
                model["intercept"] + contract(model["jm"], a) + contract(model["jr"], b)
            )
        )
        m_parts.append(ma)
        r_parts.append(ra)
    ma, ra = np.array(m_parts), np.array(r_parts)
    per_source = centered_logits(np.asarray(model["source_z"])[None] + ma + ra)
    return {
        "logits": np.array(z),
        "per_source": per_source,
        "mean_action": ma.mean(1),
        "retained_action": ra.mean(1),
    }


def candidate_jacobian(
    observer, state, kwargs, alphabet, groups, rest, block=16, on_backward=None
):
    """Measure one frozen suffix graph and ten scalar backward targets, no prefix."""
    import torch

    model = observer.model
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Observer weights must be frozen")
    observer.calls += 1
    gradients = []
    with torch.enable_grad():
        leaf = state.detach().clone().requires_grad_(True)
        h = leaf
        for layer in model.model.layers[block + 1 :]:
            h = layer(h, **kwargs)
            h = h[0] if isinstance(h, tuple) else h
        full = model.lm_head(model.model.norm(h[:, -1:]))[0, -1].float()
        ids = torch.as_tensor(alphabet["token_ids"], device=full.device)
        z = full[ids]
        centered = z - z.mean()
        for k in range(len(z)):
            (g,) = torch.autograd.grad(centered[k], leaf, retain_graph=k < len(z) - 1)
            gm = torch.stack([g[0, positions].sum(0) for positions in groups])
            gr = g[0, rest]
            gradients.append(
                (gm.detach().cpu().numpy().copy(), gr.detach().cpu().numpy().copy())
            )
            if on_backward is not None:
                on_backward()
    logits = z.detach().cpu().numpy().copy()
    jm = np.array([a for a, _ in gradients])
    jr = np.array([b for _, b in gradients])
    if not all(np.isfinite(x).all() for x in (logits, jm, jr)):
        raise ValueError("Nonfinite native logit derivative")
    return logits, jm, jr
