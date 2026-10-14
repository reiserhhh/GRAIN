#!/usr/bin/env python3
"""HM43 fixed observable covectors, not a full digit-Jacobian reconstruction."""

from __future__ import annotations

import itertools

import numpy as np

from .math_text_query_route_cut import require_replay, require_stage, route_masks

__all__ = ["require_replay", "require_stage", "route_masks"]

GROUPS = ("body", "early_query", "remainder")
COMPONENTS = (
    *GROUPS,
    "weight_correction",
    "arithmetic_correction",
    "decoder",
    "old_roundoff",
)
LINK_COMPONENTS = COMPONENTS[:5]
FIELDS = (*COMPONENTS, "weight_delta", "arithmetic_delta", "raw_link", "link", "full")


def assert_scope(spec):
    """Keep the planned observable, resource caps and original physical tolerance."""
    fixed = {
        "packet": "HM-43",
        "layer": 16,
        "cut_layer": 17,
        "paths": 96,
        "source_paths": 48,
        "maximum_forward_calls": 96,
        "maximum_first_calls": 384,
        "maximum_second_calls": 0,
        "maximum_model_seconds": 600,
        "maximum_analysis_seconds": 180,
        "absolute_tolerance": 1e-8,
        "device": "mps",
        "attention_backend": "eager",
        "new_neural_training": False,
    }
    if any(spec.get(k) != v for k, v in fixed.items()):
        raise ValueError("fixed HM43 observable and resource scope required")
    if spec.get("groups") != list(GROUPS) or spec.get("wordings") != [0, 2]:
        raise ValueError("all three routes and both wordings required")


def dual_weights(c, jp, e):
    """Freeze the pullback of one query displacement, including float32 rounding."""
    c, jp, e = [np.asarray(a, np.float64) for a in (c, jp, e)]
    if c.shape != (jp.shape[0], jp.shape[0]) or jp.shape[1:] != e.shape:
        raise ValueError("aligned readout and physical P displacement required")
    if not all(np.isfinite(a).all() for a in (c, jp, e)):
        raise ValueError("finite fixed factors required")
    w = c.T @ (jp.reshape(len(c), -1) @ e.ravel())
    return w, w.astype(np.float32)


def route_covectors(logits, cut, leaf, hp, masks, weight, journal):
    """One scalar gradient and three masked VJPs shared by both physical H arms."""
    import torch

    if cut.ndim != 3 or leaf.shape != cut.shape or cut.shape[0] != 1:
        raise ValueError("complete aligned single-sequence state required")
    if masks.shape != (3, cut.shape[1]) or not np.array_equal(
        masks.sum(0), np.ones(cut.shape[1])
    ):
        raise ValueError("three complete disjoint position masks required")
    w = torch.as_tensor(weight, dtype=logits.dtype, device=logits.device).detach()
    if w.shape != logits.shape or not torch.isfinite(w).all():
        raise ValueError("fixed finite logit weights required")

    def finish():
        if cut.device.type == "mps":
            torch.mps.synchronize()
        journal("first", "completed")

    journal("first", "attempted")
    downstream = torch.autograd.grad((w * logits).sum(), cut, retain_graph=True)[
        0
    ].detach()
    finish()
    covectors = []
    for mask in masks:
        journal("first", "attempted")
        mt = torch.as_tensor(mask, device=cut.device)[None, :, None]
        grad = torch.autograd.grad(
            cut, leaf, grad_outputs=downstream * mt, retain_graph=True
        )[0]
        finish()
        covectors.append(grad[0, hp].detach().cpu().numpy().copy())
    return np.stack(covectors), downstream[0].cpu().numpy().copy()


def scalar_accounts(covectors, weight64, weight32, old):
    """Retain untouched raw actions and separate weight and arithmetic discrepancies."""
    d, e = np.asarray(old["delta_h"], float), np.asarray(old["delta_p"], float)
    j = np.asarray(covectors, float)
    if j.shape != (3, *d.shape[1:]) or d.shape[0] != 2:
        raise ValueError("three covectors shared by two original arms required")
    routes = j.reshape(3, -1) @ d.reshape(2, -1).T
    link = np.asarray(old["link"], float).reshape(2, -1) @ e.ravel()
    h = np.asarray(old["h_jacobian"], float).reshape(len(weight64), -1)
    weight_delta = (np.asarray(weight32, float) - weight64) @ h @ d.reshape(2, -1).T
    raw = routes.sum(0)
    arithmetic = raw - link - weight_delta
    result = {
        **dict(zip(GROUPS, routes, strict=True)),
        "weight_correction": -weight_delta,
        "arithmetic_correction": -arithmetic,
        "weight_delta": weight_delta,
        "arithmetic_delta": arithmetic,
        "raw_link": raw,
        "link": link,
    }
    for key, parent in (
        ("decoder", "decoder"),
        ("old_roundoff", "roundoff"),
        ("full", "parent_total"),
    ):
        result[key] = np.asarray(old[parent], float).reshape(2, -1) @ e.ravel()
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError("nonfinite physical actions")
    return result


def sum_errors(row):
    """Corrections are bookkeeping; raw_link remains the actual measured output."""
    return {
        "full_sum_error": abs(sum(row[k] for k in COMPONENTS) - row["full"]),
        "link_sum_error": abs(sum(row[k] for k in LINK_COMPONENTS) - row["link"]),
    }


def action_rows(path, accounts, arms):
    """Bind physical scalar actions to original crossed path and arm addresses."""
    rows = []
    for ai, arm in enumerate(arms):
        row = {k: path[k] for k in ("path", "query", "role", "wording")}
        row.update(cell=path["H"] * 4 + path["B"] * 2 + path["Q"], arm=arm)
        row.update({k: float(accounts[k][ai]) for k in FIELDS})
        row.update(sum_errors(row))
        rows.append(row)
    return rows


def pair_rows(rows, wordings):
    """Preserve wording means and half-contrasts separately."""
    if len(rows) != 2 or [r["wording"] for r in rows] != wordings:
        raise ValueError("ordered original wording pair required")
    result = []
    for measure, weights in (("mean", (0.5, 0.5)), ("contrast", (-0.5, 0.5))):
        row = {k: rows[0][k] for k in ("query", "role", "cell", "arm")}
        row.update(measure=measure)
        row.update(
            {
                k: sum(w * r[k] for w, r in zip(weights, rows, strict=True))
                for k in FIELDS
            }
        )
        row.update(sum_errors(row))
        result.append(row)
    return result


def moments(rows):
    """Report signed squared-action decomposition, not information percentages."""
    x = {k: np.array([r[k] for r in rows], float) for k in FIELDS}
    if not rows or not all(np.isfinite(a).all() for a in x.values()):
        raise ValueError("nonempty finite observations required")
    result = {"count": len(rows)}
    for key, a in x.items():
        result.update(
            {
                key + "_mean": float(a.mean()),
                key + "_ms": float(np.mean(a * a)),
                key + "_rms": float(np.sqrt(np.mean(a * a))),
            }
        )
    for a, b in itertools.combinations(COMPONENTS, 2):
        result[a + "__" + b + "_cross"] = float(2 * np.mean(x[a] * x[b]))
    result.update({k: max(sum_errors(r)[k] for r in rows) for k in sum_errors(rows[0])})
    return result
