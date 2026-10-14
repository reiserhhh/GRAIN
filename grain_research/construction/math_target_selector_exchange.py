#!/usr/bin/env python3
"""Finite within-body selector exchange, preserving allocation and values."""

from __future__ import annotations

import numpy as np


def permutation(indices, size):
    """Validate an explicit one-to-one query donor assignment."""
    p = np.asarray(indices)
    if p.shape != (size,) or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("Integer query permutation required")
    if sorted(p.tolist()) != list(range(size)):
        raise ValueError("All query donors must occur once")
    return p


def redistribute(weights, body, donor, mapping):
    """Exchange conditional body probabilities without changing outside mass."""
    p, s = np.asarray(weights, float), np.asarray(donor, float)
    ix = np.asarray(body)
    if p.ndim != 3 or ix.ndim != 1 or not len(ix):
        raise ValueError("query/head/token weights and body indices required")
    if not np.issubdtype(ix.dtype, np.integer) or len(set(ix)) != len(ix):
        raise ValueError("Distinct integer body addresses required")
    if ix.min() < 0 or ix.max() >= p.shape[-1]:
        raise ValueError("Body address outside attention support")
    if s.shape != (*p.shape[:2], len(ix)):
        raise ValueError("Donor body and recipient query/head support differ")
    if (
        not np.isfinite(p).all()
        or not np.isfinite(s).all()
        or min(p.min(), s.min()) < 0
    ):
        raise ValueError("Finite nonnegative probabilities required")
    if not np.allclose(s.sum(-1), 1, rtol=0, atol=2e-6):
        raise ValueError("Donor conditional rows must already be normalized")
    m = p[..., ix].sum(-1, keepdims=True)
    if np.any(m <= 0):
        raise ValueError("Body allocation underflow is not a zero observation")
    changed = p.copy()
    changed[..., ix] = m * s[permutation(mapping, len(p))]
    return changed


def value_update(weights, changed, values):
    """Return exact signed weighted-value action, without linearizing softmax."""
    a, b, v = (np.asarray(x, float) for x in (weights, changed, values))
    if a.shape != b.shape or a.ndim != 3 or v.shape[:3] != a.shape:
        raise ValueError("query/head/token and paired value coordinates required")
    if not all(np.isfinite(x).all() for x in (a, b, v)):
        raise ValueError("Finite attention/value arrays required")
    return np.einsum("qhi,qhid->qhd", b - a, v)


def covariances(x, y):
    """Compute the complete signed SOURCE/native matrix on matched authors."""
    a, b = (np.asarray(z, float) for z in (x, y))
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b) or len(a) < 2:
        raise ValueError("At least two paired authors required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite author observations required")
    return (a - a.mean(0)).T @ (b - b.mean(0)) / (len(a) - 1)


def paired_diagonal_ci(x, native, foreign, draws, seed):
    """Resample complete authors for conditional covariance-change intervals."""
    a, n, f = (np.asarray(z, float) for z in (x, native, foreign))
    if a.shape != n.shape or a.shape != f.shape or a.ndim != 2 or draws < 2:
        raise ValueError("Matched target arrays and finite bootstrap required")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        ix = rng.integers(len(a), size=len(a))
        samples.append(np.diag(covariances(a[ix], (f - n)[ix])))
    return np.quantile(samples, (0.025, 0.975), axis=0).T


def native_batch(observer, unit, mapping=None, donor=None):
    """Execute all-block final-slot selection exchange on one five-query batch."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    if observer.model.config._attn_implementation != "eager":
        raise ValueError("Inspected native eager interface required")
    device = next(observer.model.parameters()).device
    rows, body = unit["input_ids"], unit["body"]
    if len(rows) != 5 or any(not r for r in rows):
        raise ValueError("Five nonempty query rows required")
    if donor is not None and mapping is None:
        raise ValueError("Donor exchange requires an explicit permutation")
    if not body or len(set(body)) != len(body):
        raise ValueError("Distinct nonempty body addresses required")
    if min(body) < 0 or max(body) >= min(map(len, rows)) - 1:
        raise ValueError("Body must precede every final query row")
    length = max(map(len, rows))
    ids = torch.tensor(
        [r + [observer.tokenizer.pad_token_id] * (length - len(r)) for r in rows],
        device=device,
    )
    mask = torch.tensor(
        [[1] * len(r) + [0] * (length - len(r)) for r in rows], device=device
    )
    ii, last = (
        torch.arange(5, device=device),
        torch.tensor([len(r) - 1 for r in rows], device=device),
    )
    bi = torch.tensor(body, device=device)
    if not body or min(last.tolist()) <= max(body):
        raise ValueError("Final query row must be outside the causal body")
    original = qwen.eager_attention_forward
    conditional, masses, diagnostics, calls = [], [], [], []
    mapping = None if mapping is None else permutation(mapping, 5)

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Change final body routing only; preserve every other native output row."""
        layer = module.layer_idx
        calls.append(layer)
        output, weights = original(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
        shared = max(
            float((x[:, :, bi] - x[:1, :, bi]).abs().max().item()) for x in (key, value)
        )
        if shared > 3e-4:
            raise ValueError("Body K/V differ across queries")
        full = weights[ii, :, last].clone()
        selected = full[:, :, bi]
        mass = selected.sum(-1, keepdim=True)
        if (mass <= 0).any().item():
            raise ValueError("Body attention underflow")
        s = selected / mass
        masses.append(mass[..., 0].detach().cpu().numpy())
        receipt = {
            "shared_body_KV_error": shared,
            "installed_error": 0.0,
            "mass_error": 0.0,
            "outside_error": 0.0,
            "conditional_tv_mean": 0.0,
        }
        if donor is None:
            conditional.append(s.detach().float().cpu().numpy())
        else:
            d = torch.as_tensor(donor[layer][mapping], device=device, dtype=s.dtype)
            if d.shape != s.shape or (d.sum(-1) - 1).abs().max().item() > 2e-6:
                raise ValueError("Actual donor support/normalization differs")
            replacement = mass * d
            expanded = qwen.repeat_kv(value, module.num_key_value_groups)[:, :, bi]
            correction = ((replacement - selected).unsqueeze(-1) * expanded).sum(2)
            output[ii, last] += correction
            weights[
                ii[:, None, None],
                torch.arange(weights.shape[1], device=device)[None, :, None],
                last[:, None, None],
                bi[None, None, :],
            ] = replacement
            installed = weights[ii, :, last]
            outside = [i for i in range(length) if i not in body]
            receipt.update(
                installed_error=float(
                    (installed[:, :, bi] / mass - d).abs().max().item()
                ),
                mass_error=float(
                    (installed[:, :, bi].sum(-1) - mass[..., 0]).abs().max().item()
                ),
                outside_error=float(
                    (installed[:, :, outside] - full[:, :, outside]).abs().max().item()
                ),
                conditional_tv_mean=float(((d - s).abs().sum(-1) / 2).mean().item()),
            )
            if (
                max(
                    receipt[k]
                    for k in ("installed_error", "mass_error", "outside_error")
                )
                > 2e-6
            ):
                raise ValueError("Declared selection intervention not installed")
        diagnostics.append(receipt)
        return output, weights

    try:
        qwen.eager_attention_forward = interface
        with torch.inference_mode():
            result = observer.model.model(
                input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
            )
            state = result.last_hidden_state[ii, last]
            full_logits = (
                observer.model.lm_head(state).float().cpu().numpy().astype(float)
            )
            state = state.detach().float().cpu().numpy()
    finally:
        qwen.eager_attention_forward = original
    if calls != list(range(len(observer.model.model.layers))):
        raise ValueError("All declared blocks must execute exactly once")
    return (
        full_logits,
        state,
        np.stack(masses),
        diagnostics,
        (np.stack(conditional) if donor is None else None),
    )
