#!/usr/bin/env python3
"""Source endpoint composition of frozen selection and content functions."""

import numpy as np

from . import source_value_channel as value

MODES = ("query_specific", "query_pair_mean")
COMPONENTS = value.COMPONENTS
ARMS = (*value.ARMS, "source_Q__native_KV", "native_Q__native_KV")


def readers(pair):
    """Return [query, mode, hidden] readers; no outcome-dependent weighting."""
    r = np.asarray(pair, float)
    if r.ndim != 2 or len(r) != 2 or not np.isfinite(r).all():
        raise ValueError("Two aligned finite query readers required")
    return np.stack((r, np.broadcast_to(r.mean(axis=0), r.shape)), axis=1)


def vectors(ell, dell, mask, v, dv, output):
    """Return post-output-projection primal and exact fixed-Q K/V tangents.

    No output bias is included in this declared local linear functional. Its
    tangent would be zero; this scalar is not the model's final logit contrast.
    """
    ell, dell, v, dv, o = [np.asarray(a, float) for a in (ell, dell, v, dv, output)]
    use = value.scalar.allowed(mask, len(ell))
    if (
        ell.ndim != 2
        or v.ndim != 3
        or len(v) != len(ell)
        or dell.shape != (4, *ell.shape)
        or dv.shape != (4, *v.shape)
        or ell.shape[1] % v.shape[1]
        or o.ndim != 2
        or o.shape[1] != ell.shape[1] * v.shape[2]
        or not all(np.isfinite(x).all() for x in (ell, dell, v, dv, o))
    ):
        raise ValueError("Aligned finite selector/value/output arrays required")
    logits = ell + np.asarray(mask)[:, None]
    a = np.exp(logits - logits.max(axis=0))
    a[~use] = 0
    a /= a.sum(axis=0)
    heads, hd, groups = ell.shape[1], v.shape[2], ell.shape[1] // v.shape[1]
    primal, key, content = (
        np.zeros((heads, hd)),
        np.zeros((4, heads, hd)),
        np.zeros((4, heads, hd)),
    )
    for h in range(heads):
        kv = h // groups
        primal[h] = a[:, h] @ v[:, kv]
        key[:, h] = (a[:, h] * dell[:, :, h]) @ (v[:, kv] - primal[h])
        content[:, h] = np.einsum("t,ftd->fd", a[:, h], dv[:, :, kv])
    k, c = key.reshape(4, -1) @ o.T, content.reshape(4, -1) @ o.T
    return {"primal": primal.ravel() @ o.T, "vectors": np.stack((k, c, k + c))}


def contract(packet, reader):
    """Project vector components through two explicitly declared endpoint modes."""
    r = np.asarray(reader, float)
    if r.shape != (2, packet["primal"].shape[-1]) or not np.isfinite(r).all():
        raise ValueError("Exactly two finite aligned endpoint modes required")
    return {
        "scalar": packet["primal"] @ r.T,
        "actions": np.einsum("cfh,mh->cmf", packet["vectors"], r),
    }


def predict(data, d, weights, sm, vm, reader):
    """Compose four frozen H16 functions with source-only fixed endpoint readers."""
    v, dv = value.apply(data, d, weights, vm)
    packets = []
    for selector in value.SELECTORS:
        ell, dell = value.selector(data, d, weights, sm, selector)
        for i in range(2):
            p = vectors(ell, dell, data["mask"], v[i], dv[i], weights["o"])
            packets.append(p | contract(p, reader))
    return {k: np.stack([p[k] for p in packets]) for k in packets[0]}


def account(actual, transported, compiled):
    """Exact two-link error identity, retaining signed cross energy."""
    y, p, c = np.broadcast_arrays(actual, transported, compiled)
    a, b = p - y, c - p
    return {"transport_error": a, "reader_error": b, "total_error": c - y}


def error_metrics(actual, transported, compiled):
    """Aggregate signed errors without interpreting them as causal shares."""
    e = account(actual, transported, compiled)
    a, b, total = [e[k] for k in ("transport_error", "reader_error", "total_error")]
    cross = 2 * np.mean(a * b)
    result = {k + "_mse": float(np.mean(v * v)) for k, v in e.items()}
    result["signed_cross_mse"] = float(cross)
    result["identity_error"] = float(
        abs(np.mean(total * total) - np.mean(a * a) - np.mean(b * b) - cross)
    )
    return result
