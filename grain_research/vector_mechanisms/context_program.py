#!/usr/bin/env python3
"""Frozen shallow contextual programs followed by explicit two-projection charts.

These are extracted neural programs, NOT Transformer-free operators. The
contextual prefix is retained exactly; the remaining computation is approximated
by a construction-only Jacobian and an explicit token correspondence rule.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from .projection_program import pool_pairings

CUTS = (0, 3, 7)


def cropped_config_dict(config, depth):
    """Keep layer-type metadata consistent when exporting a strict decoder config."""
    if not 1 <= depth <= config["num_hidden_layers"]:
        raise ValueError("Crop depth exceeds source model")
    result = dict(config)
    result["num_hidden_layers"] = depth
    if result.get("layer_types") is not None:
        result["layer_types"] = list(result["layer_types"][:depth])
    result["architectures"] = ["Qwen2Model"]
    return result


class _CutReached(Exception):
    """Stop before suffix, final normalization and language-model head."""


def capture_context_jacobians(observer, ids, mask, function, cuts=CUTS, block=16):
    """Pull both effective projections back to several actual early residual states."""
    import torch

    layers = observer.model.model.layers
    if (
        not cuts
        or sorted(set(cuts)) != list(cuts)
        or not 0 <= cuts[0] <= cuts[-1] < block < len(layers)
    ):
        raise ValueError("Strict early cuts and later boundary required")
    select = np.asarray(mask, bool)
    if (
        len(select) != len(ids)
        or not select.any()
        or select[-1]
        or len(ids) > observer.max_tokens
    ):
        raise ValueError("Invalid query slots")
    if any(p.requires_grad for p in observer.model.parameters()):
        raise ValueError("Freeze model weights first")
    saved, handles, trace = {}, [], []
    device = next(observer.model.parameters()).device

    def hook(index):
        def remember(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            trace.append(index)
            if index in cuts or index == block:
                saved[index] = value
            if index == block:
                raise _CutReached

        return remember

    try:
        for i in range(block + 1):
            handles.append(layers[i].register_forward_hook(hook(i)))
        tokens = torch.tensor([ids], device=device)
        embedded = (
            observer.model.model.embed_tokens(tokens).detach().requires_grad_(True)
        )
        observer.calls += 1
        with torch.enable_grad():
            try:
                observer.model.model(
                    inputs_embeds=embedded,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                )
            except _CutReached:
                pass
            if trace != list(range(block + 1)):
                raise RuntimeError("Unexpected constructor trace")
            h = saved[block][0]
            cv = lambda x: torch.as_tensor(x, device=device, dtype=h.dtype)
            g = function.config_coef + function.mean_coef / len(function.p_ref)
            values = (
                ((h[select] - cv(function.p_ref)) * cv(g)).sum(),
                (h[-1] - cv(function.q_ref)) @ cv(function.g_final),
            )
            gradients = [
                torch.autograd.grad(v, [saved[k] for k in cuts], retain_graph=a == 0)
                for a, v in enumerate(values)
            ]
            return {
                k: {
                    "hidden": saved[k][0].detach().float().cpu().numpy().copy(),
                    "gradients": np.stack(
                        [
                            gradients[a][i][0].detach().float().cpu().numpy()
                            for a in range(2)
                        ]
                    ),
                    "mask": select.copy(),
                    "projection_origin": np.asarray(
                        [float(v.detach().cpu()) for v in values]
                    ),
                }
                for i, k in enumerate(cuts)
            }
    finally:
        for handle in handles:
            handle.remove()


class CroppedContext:
    """Only embedding and the first k decoder blocks are resident and executed."""

    def __init__(self, model, cuts=CUTS, max_tokens=3072):
        """Accept a base decoder without a language-model head or suffix blocks."""
        if len(model.layers) != max(cuts) + 1 or sorted(set(cuts)) != list(cuts):
            raise ValueError("Cropped model must end at the deepest declared cut")
        self.model, self.cuts, self.max_tokens, self.calls = (
            model,
            tuple(cuts),
            max_tokens,
            0,
        )
        model.eval().requires_grad_(False)

    def capture(self, ids):
        """Execute exactly the retained blocks, returning pre-final-norm cut states."""
        import torch

        if not ids or len(ids) > self.max_tokens:
            raise ValueError("Invalid cropped input length")
        saved, handles, trace = {}, [], []

        def hook(i):
            def remember(_module, _inputs, output):
                trace.append(i)
                if i in self.cuts:
                    value = output[0] if isinstance(output, tuple) else output
                    saved[i] = value[0].detach().float().cpu().numpy().copy()
                if i == max(self.cuts):
                    raise _CutReached

            return remember

        try:
            for i, layer in enumerate(self.model.layers):
                handles.append(layer.register_forward_hook(hook(i)))
            tokens = torch.tensor([ids], device=next(self.model.parameters()).device)
            self.calls += 1
            with torch.no_grad():
                try:
                    self.model(
                        input_ids=tokens,
                        attention_mask=torch.ones_like(tokens),
                        use_cache=False,
                    )
                except _CutReached:
                    pass
            if trace != list(range(max(self.cuts) + 1)) or set(saved) != set(self.cuts):
                raise RuntimeError("Cropped execution trace differs from manifest")
            return saved
        finally:
            for handle in handles:
                handle.remove()


def contextual_projections(chart, hidden, mask):
    """Use exact query-slot correspondence plus aligned contextual evidence positions."""
    h, select = np.asarray(hidden, np.float32), np.asarray(mask, bool)
    r, g, rm = chart["hidden"], chart["gradients"], np.asarray(chart["mask"], bool)
    if (
        h.ndim != 2
        or len(h) != len(select)
        or h.shape[1] != r.shape[1]
        or select[-1]
        or rm[-1]
    ):
        raise ValueError("Context/state partition mismatch")
    if select.sum() != rm.sum() or not select.any() or not np.isfinite(h).all():
        raise ValueError("Query rows must match exactly within one query")
    re, xe = ~rm, ~select
    re[-1], xe[-1] = False, False
    current = pool_pairings(r[re], h[xe], g[:, re], "semantic_order")
    origin = chart.get("offset")
    if origin is None:
        origin = pool_pairings(r[re], r[re], g[:, re], "semantic_order")
    query_delta = np.einsum("apd,pd->a", g[:, rm], h[select] - r[rm])
    endpoint_delta = g[:, -1] @ (h[-1] - r[-1])
    z = chart["projection_origin"] + current - origin + query_delta + endpoint_delta
    xmean, rmean = h[xe].mean(0), r[re].mean(0)
    affinity = float(
        xmean @ rmean / max(float(np.linalg.norm(xmean) * np.linalg.norm(rmean)), 1e-12)
    )
    return np.asarray(z), affinity


def score_context_charts(charts, hidden, mask, anchor):
    """Retain chart-wise results and frozen mean/observable-affinity pooling rules."""
    results = [contextual_projections(c, hidden, mask) for c in charts]
    values = np.stack([v for v, _ in results])
    affinity = np.asarray([a for _, a in results])
    if len(charts) != 2:
        raise ValueError("This frozen packet compares two construction charts")
    weights = softmax(8 * affinity)
    out = {}
    for name, z in {
        "chart0": values[0],
        "chart1": values[1],
        "mean": values.mean(0),
        "affinity": weights @ values,
    }.items():
        out[name] = {
            "projection1": float(z[0]),
            "projection2": float(z[1]),
            "score": float(anchor + z.sum()),
            "chart0_weight": float(weights[0]),
        }
    return out
