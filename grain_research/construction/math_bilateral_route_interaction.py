#!/usr/bin/env python3
"""HM44 signed two-leg route action, with no marginal-independence assumption."""

from __future__ import annotations

import itertools

import numpy as np

from .math_text_query_route_cut import require_replay, route_masks

__all__ = ["require_replay", "route_masks"]
GROUPS = ("body", "early_query", "remainder")
CELLS = tuple(a + "__" + b for a, b in itertools.product(GROUPS, repeat=2))
COMPONENTS = (*CELLS, "cut_roundoff", "decoder", "old_roundoff")
FIELDS = (
    *COMPONENTS,
    *("H_" + g for g in GROUPS),
    *("P_" + g for g in GROUPS),
    "raw_link",
    "link",
    "full",
)
PATHS = (
    0,
    7,
    8,
    15,
    16,
    23,
    24,
    31,
    32,
    39,
    40,
    47,
    48,
    55,
    56,
    63,
    64,
    71,
    72,
    79,
    80,
    87,
    88,
    95,
)


def assert_scope(spec):
    """Keep the original predeclared corners, physical units and resource caps."""
    expected = {
        "packet": "HM-44",
        "layer": 16,
        "cut_layer": 17,
        "paths": 24,
        "source_paths": 12,
        "maximum_forward_calls": 24,
        "maximum_first_calls": 960,
        "maximum_second_calls": 0,
        "maximum_model_seconds": 600,
        "maximum_analysis_seconds": 180,
        "absolute_tolerance": 1e-8,
        "device": "mps",
        "attention_backend": "eager",
        "new_neural_training": False,
    }
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("fixed HM44 physical and resource scope required")
    if (
        spec.get("path_ids") != list(PATHS)
        or spec.get("groups") != list(GROUPS)
        or spec.get("wordings") != [0, 2]
    ):
        raise ValueError("predeclared24 corners and complete route family required")


def require_stage(status, source, barrier_exists):
    """New application acquisitions require all12 new source captures sealed first."""
    if (
        status["status"] != ("PREPARED" if source else "SOURCE_CHECKPOINTED")
        or status["in_flight"] is not None
        or status["committed_nodes"] != (0 if source else 12)
        or barrier_exists == source
    ):
        raise ValueError("explicit source-first finite stage required")


def bilateral_jacobians(logits, cut, leaf, hp, pp, masks, journal):
    """Each masked VJP produces both H/P legs; downstream digit gradients detached."""
    import torch

    if (
        cut.ndim != 3
        or cut.shape != leaf.shape
        or cut.shape[0] != 1
        or logits.ndim != 1
    ):
        raise ValueError("one complete sequential state required")
    if masks.shape != (3, cut.shape[1]) or not np.array_equal(
        masks.sum(0), np.ones(cut.shape[1])
    ):
        raise ValueError("three disjoint exhaustive masks required")
    if (
        not hp
        or not pp
        or set(hp) & set(pp)
        or min((*hp, *pp)) < 0
        or max((*hp, *pp)) >= cut.shape[1]
    ):
        raise ValueError("distinct finite H/P leaf positions required")

    def finish():
        if cut.device.type == "mps":
            torch.mps.synchronize()
        journal("first", "completed")

    hs, ps, ds = [], [], []
    for value in logits:
        journal("first", "attempted")
        down = torch.autograd.grad(value, cut, retain_graph=True)[0].detach()
        finish()
        ds.append(down[0].cpu().numpy().copy())
        hr, pr = [], []
        for mask in masks:
            journal("first", "attempted")
            mt = torch.as_tensor(mask, device=cut.device)[None, :, None]
            grad = torch.autograd.grad(
                cut, leaf, grad_outputs=down * mt, retain_graph=True
            )[0]
            finish()
            hr.append(grad[0, hp].detach().cpu().numpy().copy())
            pr.append(grad[0, pp].detach().cpu().numpy().copy())
        hs.append(hr)
        ps.append(pr)
    return np.stack(hs).swapaxes(0, 1), np.stack(ps).swapaxes(0, 1), np.stack(ds)


def joint_actions(hj, pj, d, e, c):
    """Apply C in its original order between physical H and P digit actions."""
    h, p, d, e, c = [np.asarray(a, float) for a in (hj, pj, d, e, c)]
    if (
        h.ndim != 4
        or p.ndim != 4
        or h.shape[:2] != p.shape[:2]
        or h.shape[0] != 3
        or d.shape != (2, *h.shape[2:])
        or e.shape != p.shape[2:]
        or c.shape != (h.shape[1], h.shape[1])
    ):
        raise ValueError("three aligned routes, two H arms and ordered C required")
    if not all(np.isfinite(a).all() for a in (h, p, d, e, c)):
        raise ValueError("finite physical route factors required")
    ha = h.reshape(3, h.shape[1], -1) @ d.reshape(2, -1).T
    pa = p.reshape(3, p.shape[1], -1) @ e.ravel()
    return np.einsum("gja,jl,kl->agk", ha, c, pa)


def scalar_accounts(joint, old):
    """Preserve raw matrix, both margins, correction and inherited decoder separately."""
    a = np.asarray(joint, float)
    if a.shape != (2, 3, 3) or not np.isfinite(a).all():
        raise ValueError("two finite3x3 physical route matrices required")
    e = np.asarray(old["delta_p"], float).ravel()
    link = np.asarray(old["link"], float).reshape(2, -1) @ e
    raw = a.sum((1, 2))
    result = {
        **dict(zip(CELLS, a.reshape(2, 9).T, strict=True)),
        "cut_roundoff": link - raw,
        "raw_link": raw,
        "link": link,
    }
    result.update({"H_" + g: a[:, i].sum(1) for i, g in enumerate(GROUPS)})
    result.update({"P_" + g: a[:, :, i].sum(1) for i, g in enumerate(GROUPS)})
    for key, parent in (
        ("decoder", "decoder"),
        ("old_roundoff", "roundoff"),
        ("full", "parent_total"),
    ):
        result[key] = np.asarray(old[parent], float).reshape(2, -1) @ e
    return result


def errors(row):
    """Raw acceptance is distinct from the correction bookkeeping identity."""
    return {
        "full_sum_error": abs(sum(row[k] for k in COMPONENTS) - row["full"]),
        "link_sum_error": abs(
            sum(row[k] for k in (*CELLS, "cut_roundoff")) - row["link"]
        ),
        "H_margin_sum_error": abs(sum(row["H_" + g] for g in GROUPS) - row["raw_link"]),
        "P_margin_sum_error": abs(sum(row["P_" + g] for g in GROUPS) - row["raw_link"]),
    }


def action_rows(path, accounts, arms):
    """Bind original physical arms and crossed material addresses to scalar actions."""
    rows = []
    for ai, arm in enumerate(arms):
        r = {k: path[k] for k in ("path", "query", "role", "wording")}
        r.update(cell=path["H"] * 4 + path["B"] * 2 + path["Q"], arm=arm)
        r.update({k: float(accounts[k][ai]) for k in FIELDS})
        r.update(errors(r))
        rows.append(r)
    return rows


def pair_rows(rows, wordings):
    """Retain both wording levels and half-contrasts for every predeclared corner."""
    if len(rows) != 2 or [r["wording"] for r in rows] != wordings:
        raise ValueError("ordered original wording pair required")
    result = []
    for measure, weights in (("mean", (0.5, 0.5)), ("contrast", (-0.5, 0.5))):
        r = {k: rows[0][k] for k in ("query", "role", "cell", "arm")}
        r.update(measure=measure)
        r.update(
            {
                k: sum(w * x[k] for w, x in zip(weights, rows, strict=True))
                for k in FIELDS
            }
        )
        r.update(errors(r))
        result.append(r)
    return result


def moments(rows):
    """Signed route36 crosses plus decoder/correction terms; no probability ratios."""
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
        result[a + "___" + b + "_cross"] = float(2 * np.mean(x[a] * x[b]))
    result["link_decoder_cross"] = float(2 * np.mean(x["link"] * x["decoder"]))
    result.update({k: max(errors(r)[k] for r in rows) for k in errors(rows[0])})
    return result
