#!/usr/bin/env python3
"""Source content-value transport with a common scalar and full K/V derivative."""

import numpy as np

from . import source_selector_potential as scalar
from .source_state_transport import features
from .upstream_attention_channels import normalized_directions

SELECTORS = ("identity", "response_coordinates")
VALUES = ("identity_value", "source_value")
COMPONENTS = ("key", "value", "total")
ARMS = tuple(f"{s}__{v}" for s in SELECTORS for v in VALUES)


def values(x, d, weights):
    """Apply fixed RMSNorm/V matrices to the supplied state and its directions."""
    if "q" in weights or "q_bias" in weights:
        raise ValueError("Current query matrices are not value-channel inputs")
    z, dz = normalized_directions(x, d, weights["gamma"], float(weights["epsilon"]))
    matrix, bias = np.asarray(weights["v"], float), np.asarray(weights["v_bias"], float)
    hd = int(weights["head_dim"])
    if matrix.ndim != 2 or matrix.shape[1] != z.shape[1] or matrix.shape[0] % hd:
        raise ValueError("Aligned value projection required")
    if bias.shape != (len(matrix),) or not all(
        np.isfinite(v).all() for v in (matrix, bias)
    ):
        raise ValueError("Finite value weights required")
    return (z @ matrix.T + bias).reshape(len(z), -1, hd), (dz @ matrix.T).reshape(
        len(dz), len(z), -1, hd
    )


def design(data, d, base):
    """Ordinary fixed source coordinates, without a per-sequence value gauge."""
    p, dp = features(data["state"], d, base["mean"], base["basis"], str(base["mode"]))
    scale = np.asarray(base["scale"], float)
    if (
        scale.shape != (p.shape[1],)
        or np.any(scale <= 0)
        or not np.isfinite(scale).all()
    ):
        raise ValueError("Finite positive source feature scale required")
    if base["center"][-3] != 0 or scale[-3] != 1:
        raise ValueError("Explicit unit bias feature required")
    return (p - base["center"]) / scale, dp / scale


def fit(items, members, query, base):
    """One source-only joint V/dV solve; never read held-out payloads."""
    picked = [i for i in items if i["index"] in members and i["query"] == query]
    if (
        len(picked) != len(members)
        or sorted(i["index"] for i in picked) != list(members)
        or not np.array_equal(base["members"], members)
    ):
        raise ValueError("Exact source membership required")
    width, output = len(base["center"]), int(np.prod(picked[0]["v"].shape[1:]))
    stats = {"heads": output, "rx": 0.0, "rd": 0.0}
    for tag in ("x", "d"):
        stats["g" + tag] = np.zeros((width, width))
        stats["c" + tag] = np.zeros((width, output))
    for item in picked:
        z, dz = design(item["data"], item["d"], base)
        use = scalar.allowed(item["data"]["mask"], len(z))
        y = (item["v"] - item["v0"]).reshape(len(z), output)
        dy = (item["dv"] - item["dv0"]).reshape(4, len(z), output)
        for tag, xx, yy in (("x", z[None], y[None]), ("d", dz, dy)):
            weight = 1 / (len(members) * use.sum() * len(xx))
            for a, b in zip(xx[:, use], yy[:, use], strict=True):
                stats["g" + tag] += weight * a.T @ a
                stats["c" + tag] += weight * a.T @ b
                stats["r" + tag] += weight * np.sum(b * b) / output
    if min(stats["rx"], stats["rd"]) <= 1e-12:
        raise ValueError("Nonzero source residual energy required")
    gx, gd = [stats["g" + t] / (output * stats["r" + t]) for t in ("x", "d")]
    rhs = sum(stats["c" + t] / (output * stats["r" + t]) for t in ("x", "d"))
    penalty = np.full(width, float(base["alpha"]))
    penalty[-3] = 0
    gram = gx + gd + np.diag(penalty)
    coef = np.linalg.solve(gram, rhs)
    if not np.isfinite(coef).all():
        raise ValueError("Nonfinite value coefficients")
    model = {
        k: np.array(base[k], copy=True)
        for k in (
            "mean",
            "basis",
            "center",
            "scale",
            "mode",
            "rank",
            "alpha",
            "members",
        )
    }
    model.update(coef=coef, query=np.asarray(query))
    training = scalar.loss(coef, stats) | {
        "normal_error": float(np.max(abs(gram @ coef - rhs))),
        "source_rx": float(stats["rx"]),
        "source_rd": float(stats["rd"]),
        "bias_column_norm": float(np.linalg.norm(coef[-3])),
    }
    return model, training


def apply(data, d, weights, model):
    """Return identity and learned V/dV from H16, with no native-value packet."""
    v, dv = values(data["state"], d, weights)
    phi, dp = design(data, d, model)
    return np.stack((v, v + (phi @ model["coef"]).reshape(v.shape))), np.stack(
        (dv, dv + (dp @ model["coef"]).reshape(dv.shape))
    )


def readout(ell, dell, mask, v, dv, reader):
    """Differentiate both selection and content; g is an explicit fixed input."""
    if set(reader) != {"g"}:
        raise ValueError("A g-only reader packet is required")
    v, dv, g = (np.asarray(x, float) for x in (v, dv, reader["g"]))
    p = scalar.readout(ell, dell, mask, {"v": v, "g": g})
    if dv.shape != (4, *v.shape) or not np.isfinite(dv).all():
        raise ValueError("Aligned finite value directions required")
    content = np.zeros((2, 4))
    groups = len(g[0]) // v.shape[1]
    for h in range(len(g[0])):
        projected = np.einsum("ftd,md->ftm", dv[:, :, h // groups], g[:, h])
        content += np.einsum("t,ftm->mf", p["attention"][:, h], projected)
    return {
        "scalar": p["scalar"],
        "actions": np.stack((p["actions"], content, p["actions"] + content)),
    }


def selector(data, d, weights, model, name):
    """Use the two frozen selector functions, not a newly selected arm."""
    if name == "response_coordinates":
        return scalar.potential(data, d, weights, model)
    if name != "identity":
        raise ValueError("Unknown selector")
    k, dk = scalar.identity_keys(data, d, weights)
    ell, dell = scalar.logits(model["anchor"], k, dk, data, float(weights["scale"]))
    return scalar.gauge(ell, data["mask"]), scalar.gauge(dell, data["mask"])


def predict(data, d, weights, smodel, vmodel, reader):
    """Four predeclared compatible functions and the retained bias-column effect."""
    v, dv = apply(data, d, weights, vmodel)
    out = []
    for s in SELECTORS:
        ell, dell = selector(data, d, weights, smodel, s)
        for j in range(2):
            out.append(readout(ell, dell, data["mask"], v[j], dv[j], reader))
    bias = np.broadcast_to(vmodel["coef"][-3].reshape(v[0].shape[1:]), v[0].shape)
    bias_read = readout(ell, dell, data["mask"], bias, np.zeros_like(dv[0]), reader)
    return {
        "scalar": np.stack([p["scalar"] for p in out]),
        "actions": np.stack([p["actions"] for p in out]),
        "v": v,
        "dv": dv,
        "bias_scalar": bias_read["scalar"],
        "bias_actions": bias_read["actions"],
    }
