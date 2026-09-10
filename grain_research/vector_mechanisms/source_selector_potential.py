#!/usr/bin/env python3
"""Source-only scalar selection potentials and their coupled GRAIN derivatives."""

from __future__ import annotations

import numpy as np

from .fixed_seed_attention import rotary
from .source_state_transport import features
from .upstream_attention_channels import normalized_directions

OBJECTIVES = ("logit_only", "joint")
ARMS = ("identity", *OBJECTIVES)


def allowed(mask, tokens):
    """Validate fixed attention support without treating masked tokens as samples."""
    mask = np.asarray(mask, float)
    if (
        mask.shape != (tokens,)
        or np.isnan(mask).any()
        or np.isposinf(mask).any()
        or np.isfinite(mask).sum() < 2
    ):
        raise ValueError("At least two valid fixed token addresses required")
    return np.isfinite(mask)


def gauge(x, mask):
    """Remove only an allowed-token constant; token axis is always penultimate."""
    x = np.asarray(x, float)
    use = allowed(mask, x.shape[-2])
    return x - np.mean(x[..., use, :], axis=-2, keepdims=True)


def logits(q, k, dk, layout, scale):
    """Project fixed source Q and pre-RoPE keys to all head logits/directions."""
    q, k, dk = (np.asarray(v, float) for v in (q, k, dk))
    cos, sin = (np.asarray(layout[n], float) for n in ("cos", "sin"))
    if (
        q.ndim != 2
        or k.ndim != 3
        or dk.shape != (4, *k.shape)
        or q.shape[1] != k.shape[2]
        or q.shape[0] % k.shape[1]
        or cos.shape != (len(k), k.shape[2])
        or sin.shape != cos.shape
        or not np.isfinite(scale)
        or scale <= 0
        or not all(np.isfinite(v).all() for v in (q, k, dk, cos, sin))
    ):
        raise ValueError("Exact finite GQA/RoPE key and query coordinates required")
    allowed(layout["mask"], len(k))
    qr = rotary(q, cos[-1], sin[-1])
    kr = rotary(k, cos[:, None], sin[:, None])
    dr = rotary(dk, cos[None, :, None], sin[None, :, None])
    groups = len(q) // k.shape[1]
    ell, dell = [], []
    for h in range(len(q)):
        kv = h // groups
        ell.append(kr[:, kv] @ qr[h] * scale)
        dell.append(np.einsum("fth,h->ft", dr[:, :, kv], qr[h]) * scale)
    return np.stack(ell, axis=-1), np.stack(dell, axis=-1)


def identity_keys(data, d, weights):
    """Compute keys from H16, not native attention22 states or captured current K."""
    w = weights
    if "q" in w or "q_bias" in w:
        raise ValueError("Current-query projection not permitted")
    z, dz = normalized_directions(data["state"], d, w["gamma"], float(w["epsilon"]))
    hd = int(w["head_dim"])
    shape = (len(z), -1, hd)
    return (
        (z @ w["k"].T + w["k_bias"]).reshape(shape),
        (dz @ w["k"].T).reshape((4, *shape)),
    )


def design(data, d, base):
    """Keep source scales and center the feature and tangent in the same gauge."""
    p, dp = features(data["state"], d, base["mean"], base["basis"], str(base["mode"]))
    z = gauge((p - base["center"]) / base["scale"], data["mask"])
    dz = gauge(dp / base["scale"], data["mask"])
    z[:, -3], dz[:, :, -3] = 0, 0
    return z, dz


def source_statistics(items, members, query, base, anchor, scale):
    """Fit targets use one split-specific anchor, equal authors and allowed tokens."""
    picked = [i for i in items if i["index"] in members and i["query"] == query]
    if (
        len(picked) != len(members)
        or sorted(i["index"] for i in picked) != list(members)
        or not np.array_equal(base["members"], members)
    ):
        raise ValueError("Exact permitted sources required for each query")
    width, heads = len(base["center"]), len(anchor)
    s = {
        "gx": np.zeros((width, width)),
        "gd": np.zeros((width, width)),
        "cx": np.zeros((width, heads)),
        "cd": np.zeros((width, heads)),
        "rx": 0.0,
        "rd": 0.0,
        "heads": heads,
    }
    for i in picked:
        data = i["data"]
        use = allowed(data["mask"], len(data["state"]))
        z, dz = design(data, i["d"], base)
        ell0, dell0 = logits(anchor, i["k0"], i["dk0"], data, scale)
        ell, dell = logits(anchor, i["k"], i["dk"], data, scale)
        targets = (gauge(ell - ell0, data["mask"]), gauge(dell - dell0, data["mask"]))
        for tag, zz, yy in (("x", z[None], targets[0][None]), ("d", dz, targets[1])):
            weight = 1 / (len(members) * use.sum() * len(zz))
            for a, b in zip(zz[:, use], yy[:, use], strict=True):
                s["g" + tag] += weight * (a.T @ a)
                s["c" + tag] += weight * (a.T @ b)
                s["r" + tag] += weight * float(np.sum(b * b)) / heads
    if min(s["rx"], s["rd"]) <= 1e-12:
        raise ValueError("Nonzero source residual scales required")
    return s


def loss(coef, stats):
    """Report both source-normalized losses, irrespective of the training objective."""
    return {
        tag + "_loss": float(
            1
            + (
                np.sum(coef * (stats["g" + tag] @ coef))
                - 2 * np.sum(coef * stats["c" + tag])
            )
            / (stats["heads"] * stats["r" + tag])
        )
        for tag in ("x", "d")
    }


def fit(items, members, query, base, anchor, scale):
    """Two fixed ridge solves, zero-pinned constant gauge; never select on new8."""
    s = source_statistics(items, members, query, base, anchor, scale)
    active = np.arange(len(base["center"])) != len(base["center"]) - 3
    models, training = {}, {}
    for objective in OBJECTIVES:
        eta = int(objective == "joint")
        gram = s["gx"] / (s["heads"] * s["rx"]) + eta * s["gd"] / (s["heads"] * s["rd"])
        rhs = s["cx"] / (s["heads"] * s["rx"]) + eta * s["cd"] / (s["heads"] * s["rd"])
        gram = gram[np.ix_(active, active)] + float(base["alpha"]) * np.eye(
            active.sum()
        )
        coef = np.zeros_like(rhs)
        coef[active] = np.linalg.solve(gram, rhs[active])
        if not np.isfinite(coef).all():
            raise ValueError("Nonfinite selector coefficients")
        models[objective] = {
            k: np.array(base[k], copy=True)
            for k in (
                "mean",
                "basis",
                "center",
                "scale",
                "members",
                "mode",
                "alpha",
                "rank",
            )
        }
        models[objective].update(
            coef=coef,
            anchor=anchor,
            query=np.asarray(query),
            objective=np.asarray(objective),
            source_rx=np.asarray(s["rx"]),
            source_rd=np.asarray(s["rd"]),
        )
        training[objective] = loss(coef, s) | {
            "normal_error": float(np.max(abs(gram @ coef[active] - rhs[active]))),
            "source_rx": s["rx"],
            "source_rd": s["rd"],
        }
    return models, training


def potential(data, d, weights, model, keys=None):
    """The same differentiable scalar logits generate both attention and tangent."""
    k, dk = identity_keys(data, d, weights) if keys is None else keys
    ell, dell = logits(model["anchor"], k, dk, data, float(weights["scale"]))
    z, dz = design(data, d, model)
    return gauge(ell, data["mask"]) + z @ model["coef"], gauge(
        dell, data["mask"]
    ) + dz @ model["coef"]


def readout(ell, dell, mask, value):
    """Keep native V/g explicit; return scalar scores, attention and key-only actions."""
    if set(value) != {"v", "g"}:
        raise ValueError("Only value-channel fields permitted")
    ell, dell, v, g = (
        np.asarray(x, float) for x in (ell, dell, value["v"], value["g"])
    )
    use = allowed(mask, len(ell))
    if (
        ell.ndim != 2
        or dell.shape != (4, *ell.shape)
        or v.ndim != 3
        or len(v) != len(ell)
        or ell.shape[1] % v.shape[1]
        or g.shape != (2, ell.shape[1], v.shape[2])
        or not all(np.isfinite(x).all() for x in (ell, dell, v, g))
    ):
        raise ValueError("Aligned finite scalar logits, tangents and values required")
    scores = ell + np.asarray(mask)[:, None]
    loga = scores - scores.max(axis=0)
    loga -= np.log(np.exp(loga).sum(axis=0))
    a = np.exp(loga)
    action, scalar = np.zeros((2, 4)), np.zeros(2)
    groups = ell.shape[1] // v.shape[1]
    for h in range(ell.shape[1]):
        val = v[:, h // groups] @ g[:, h].T
        mu = a[:, h] @ val
        scalar += mu
        action += ((dell[:, :, h] * a[:, h]) @ (val - mu)).T
    assert np.array_equal(a[~use], np.zeros_like(a[~use]))
    return {"attention": a, "logattention": loga, "actions": action, "scalar": scalar}
