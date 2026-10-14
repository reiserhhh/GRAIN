#!/usr/bin/env python3
"""HM42 complete-state first derivative cut, with all signed routes retained."""

from __future__ import annotations

import itertools

import numpy as np

GROUPS = ("body", "early_query", "remainder")
COMPONENTS = (*GROUPS, "cut_roundoff", "decoder", "old_roundoff")
FIELDS = (*COMPONENTS, "link", "full")


def require_stage(status, source, barrier_exists):
    """Keep application acquisition behind a complete source-only checkpoint."""
    expected = "PREPARED" if source else "SOURCE_CHECKPOINTED"
    nodes = 0 if source else 48
    if (
        status["status"] != expected
        or status["in_flight"] is not None
        or status["committed_nodes"] != nodes
        or barrier_exists == source
    ):
        raise ValueError("explicit source-first acquisition stage required")


def assert_scope(spec):
    """Refuse a different layer, observation family, fit or expanded resource cap."""
    expected = {
        "packet": "HM-42",
        "layer": 16,
        "cut_layer": 17,
        "paths": 96,
        "source_paths": 48,
        "maximum_forward_calls": 96,
        "maximum_first_calls": 3840,
        "maximum_second_calls": 0,
        "maximum_model_seconds": 1200,
        "maximum_analysis_seconds": 180,
        "absolute_tolerance": 1e-8,
        "device": "mps",
        "attention_backend": "eager",
        "new_neural_training": False,
    }
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("fixed HM42 scientific and resource scope required")
    if spec.get("groups") != list(GROUPS) or spec.get("wordings") != [0, 2]:
        raise ValueError("all three groups and both original wordings required")


def route_masks(row):
    """Partition actual sequence positions; do not classify by linguistic meaning."""
    n, start = len(row["input_ids"]), row["question_start"]
    pp = row["early_query_positions"]
    hp = row["spans"]["A_property"]["positions"]
    if (
        not 0 < start < n - 1
        or not pp
        or pp != sorted(set(pp))
        or min(pp) < start
        or max(pp) >= n - 1
        or not hp
        or len(set(hp)) != len(hp)
        or min(hp) < 0
        or max(hp) >= start
    ):
        raise ValueError("aligned property/body and pre-endpoint query required")
    masks = np.zeros((3, n), dtype=bool)
    masks[0, :start] = True
    masks[1, pp] = True
    masks[2] = ~masks[:2].any(axis=0)
    if not np.array_equal(masks.sum(axis=0), np.ones(n, dtype=int)):
        raise ValueError("cut masks must exhaust all positions exactly once")
    return masks


def route_jacobians(logits, cut, leaf, hp, masks, journal):
    """Detach downstream weights before three VJPs; no second-order differentiation."""
    import torch

    if cut.ndim != 3 or leaf.shape != cut.shape or masks.shape != (3, cut.shape[1]):
        raise ValueError("one full sequential state and three position masks required")
    if logits.ndim != 1 or cut.shape[0] != 1:
        raise ValueError("one reader sequence with scalar digit logits required")
    if not np.array_equal(masks.sum(axis=0), np.ones(cut.shape[1], dtype=int)):
        raise ValueError("disjoint complete cut required")
    weights = torch.tensor(masks, device=cut.device)[:, None, :, None]
    jacobians, cut_gradients = [], []

    def finish():
        if cut.device.type == "mps":
            torch.mps.synchronize()
        journal("first", "completed")

    for value in logits:
        journal("first", "attempted")
        downstream = torch.autograd.grad(value, cut, retain_graph=True)[0].detach()
        finish()
        cut_gradients.append(downstream[0].cpu().numpy().copy())
        routes = []
        for mask in weights:
            journal("first", "attempted")
            backward = torch.autograd.grad(
                cut,
                leaf,
                grad_outputs=downstream * mask,
                retain_graph=True,
            )[0]
            finish()
            routes.append(backward[0, hp].detach().cpu().numpy().copy())
        jacobians.append(routes)
    return np.stack(jacobians).swapaxes(0, 1), np.stack(cut_gradients)


def require_replay(actual, parent, tolerance=1e-8):
    """Keep a physical comparison gate; a cut bypass must fail rather than disappear."""
    x, y = np.asarray(actual, float), np.asarray(parent, float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("aligned finite replay arrays required")
    error = float(np.max(abs(x - y))) if x.size else 0.0
    if error > tolerance:
        raise ValueError(f"original physical replay gate failed: {error}")
    return error


def link_routes(jacobians, directions, hessian, p_jacobian):
    """Compose each H-leg cut route in the original ordered physical units."""
    j, d, c, p = [
        np.asarray(x, float) for x in (jacobians, directions, hessian, p_jacobian)
    ]
    if (
        j.ndim != 4
        or j.shape[0] != 3
        or d.ndim != 3
        or d.shape[1:] != j.shape[2:]
        or c.shape != (j.shape[1], j.shape[1])
        or p.ndim != 3
        or p.shape[0] != j.shape[1]
        or not all(np.isfinite(a).all() for a in (j, d, c, p))
    ):
        raise ValueError("finite aligned cut, directions and readout factors required")
    h_actions = np.einsum("gjhd,ahd->gaj", j, d)
    return np.tensordot(h_actions @ c.T, p, axes=(2, 0))


def action_rows(path, routes, old, groups, arms):
    """Preserve route/link/full actions and both numerical remainder accounts."""
    r = np.asarray(routes, float)
    dp = np.asarray(old["delta_p"], float)
    if r.shape != (3, 2, *dp.shape) or groups != list(GROUPS) or len(arms) != 2:
        raise ValueError("all three routes and original two physical arms required")
    route_sum = r.sum(axis=0)
    cut_roundoff = np.asarray(old["link"], float) - route_sum
    components = {
        **dict(zip(GROUPS, r, strict=True)),
        "cut_roundoff": cut_roundoff,
        "decoder": old["decoder"],
        "old_roundoff": old["roundoff"],
        "link": old["link"],
        "full": old["parent_total"],
    }
    rows = []
    for ai, arm in enumerate(arms):
        row = {
            "path": path["path"],
            "query": path["query"],
            "role": path["role"],
            "cell": path["H"] * 4 + path["B"] * 2 + path["Q"],
            "wording": path["wording"],
            "arm": arm,
        }
        for key, values in components.items():
            row[key] = float(np.sum(np.asarray(values, float)[ai] * dp))
        row["full_sum_error"] = abs(sum(row[k] for k in COMPONENTS) - row["full"])
        row["link_sum_error"] = abs(
            sum(row[k] for k in (*GROUPS, "cut_roundoff")) - row["link"]
        )
        rows.append(row)
    return rows, cut_roundoff


def pair_rows(rows, wordings):
    """Keep pair means and half-differences, not only attractive grand averages."""
    if len(rows) != 2 or [r["wording"] for r in rows] != wordings:
        raise ValueError("ordered original two-wording group required")
    result = []
    for measure, weights in (("mean", (0.5, 0.5)), ("contrast", (-0.5, 0.5))):
        row = {k: rows[0][k] for k in ("query", "role", "cell", "arm")}
        row["measure"] = measure
        row.update(
            {
                k: sum(w * r[k] for w, r in zip(weights, rows, strict=True))
                for k in FIELDS
            }
        )
        row["full_sum_error"] = abs(sum(row[k] for k in COMPONENTS) - row["full"])
        row["link_sum_error"] = abs(
            sum(row[k] for k in (*GROUPS, "cut_roundoff")) - row["link"]
        )
        result.append(row)
    return result


def moments(rows):
    """Return all signed squared-action terms with no information-percent interpretation."""
    x = {k: np.array([r[k] for r in rows], float) for k in FIELDS}
    if not rows or not all(np.isfinite(a).all() for a in x.values()):
        raise ValueError("nonempty finite action rows required")
    result = {"count": len(rows)}
    for k, a in x.items():
        result.update(
            {
                k + "_mean": float(a.mean()),
                k + "_ms": float(np.mean(a * a)),
                k + "_rms": float(np.sqrt(np.mean(a * a))),
            }
        )
    for a, b in itertools.combinations(COMPONENTS, 2):
        result[a + "__" + b + "_cross"] = float(2 * np.mean(x[a] * x[b]))
    result["full_sum_error"] = float(
        np.max(abs(sum(x[k] for k in COMPONENTS) - x["full"]))
    )
    result["link_sum_error"] = float(
        np.max(abs(sum(x[k] for k in (*GROUPS, "cut_roundoff")) - x["link"]))
    )
    return result
