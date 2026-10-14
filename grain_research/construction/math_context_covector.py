"""HM16: source-only context covectors and symmetric endpoint action kernels."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

MODELS = ("full_linear", "receiver_tensor", "endpoint_potential", "shuffled_context")


def context_reference(reference, weights):
    """Freeze a source origin and scalar RMS, without discarding the level term."""
    r, w = np.asarray(reference, float), np.asarray(weights, float)
    if (
        r.ndim != 2
        or w.shape != (len(r),)
        or len(r) < 2
        or not np.isfinite(r).all()
        or not np.isfinite(w).all()
        or np.min(w) <= 0
    ):
        raise ValueError("finite source observations and positive weights required")
    origin = w @ r / w.sum()
    centered = r - origin
    scale2 = float(w @ np.sum(centered * centered, axis=1) / w.sum())
    return origin, float(np.sqrt(scale2)) if scale2 > 0 else 1.0, scale2


def kernel_parts(delta, context, source_delta, source_context, kind):
    """Return linear and tensor Gram terms, never allocating an ambient tensor."""
    x, c, a, b = (
        np.asarray(v, float) for v in (delta, context, source_delta, source_context)
    )
    if (
        any(v.ndim != 2 or not np.isfinite(v).all() for v in (x, c, a, b))
        or len(x) != len(c)
        or len(a) != len(b)
        or x.shape[1] != a.shape[1]
        or c.shape[1] != b.shape[1]
    ):
        raise ValueError("finite matched displacement/context rows required")
    linear = x @ a.T
    if kind == "receiver_tensor":
        coupling = linear * (c @ b.T)
    elif kind == "endpoint_potential":
        if c.shape[1] != x.shape[1]:
            raise ValueError("symmetric endpoint coordinates require equal dimensions")
        coupling = 0.5 * (linear * (c @ b.T) + (c @ a.T) * (x @ b.T))
    else:
        raise ValueError("unknown fixed context kernel")
    return linear, coupling


def fit_kernel(raw_kernel, target, weights, alpha=1.0):
    """Solve one weighted source Gram system in a frozen average-diagonal unit."""
    k, y, w = (np.asarray(v, float) for v in (raw_kernel, target, weights))
    if (
        k.ndim != 2
        or k.shape != (len(y), len(y))
        or len(y) < 2
        or y.ndim != 1
        or w.shape != y.shape
        or not all(np.isfinite(v).all() for v in (k, y, w))
        or np.min(w) <= 0
        or not np.isfinite(alpha)
        or alpha <= 0
        or not np.allclose(k, k.T, atol=1e-10, rtol=1e-12)
    ):
        raise ValueError(
            "finite symmetric source Gram/targets/positive weights required"
        )
    unit = float(w @ np.diag(k) / w.sum())
    if unit <= 0:
        raise ValueError("nonzero source observation required")
    gram = (k + k.T) / (2 * unit)
    eigen = np.linalg.eigvalsh(gram)
    tolerance = 100 * np.finfo(float).eps * max(1.0, eigen[-1]) * len(y)
    if eigen[0] < -tolerance:
        raise ValueError("source tensor Gram not positive semidefinite")
    root = np.sqrt(w)
    system = root[:, None] * gram * root[None, :] + alpha * np.eye(len(y))
    dual = np.linalg.solve(system, root * y)
    return {
        "kernel_unit": unit,
        "alpha": float(alpha),
        "intercept": 0.0,
        "source_rows": len(y),
        "source_numerical_rank": int(np.sum(eigen > tolerance)),
        "gram_eigen_min": float(eigen[0]),
        "gram_eigen_max": float(eigen[-1]),
        "gram_rank_tolerance": float(tolerance),
        "weighted_dual": root * dual,
        "eigenvalues": eigen,
        "system_residual_max": float(np.max(np.abs(system @ dual - root * y))),
    }


def apply_parts(linear, coupling, weighted_dual, kernel_unit):
    """Score frozen linear/tensor contributions in their common source unit."""
    a, b, d = (np.asarray(v, float) for v in (linear, coupling, weighted_dual))
    if (
        a.ndim != 2
        or b.shape != a.shape
        or d.shape != (a.shape[1],)
        or not all(np.isfinite(v).all() for v in (a, b, d))
        or not np.isfinite(kernel_unit)
        or kernel_unit <= 0
    ):
        raise ValueError("finite matching parts and positive source unit required")
    return a @ d / kernel_unit, b @ d / kernel_unit


def shuffled_state_mapping(metadata, seed=1617):
    """Cycle pair contexts within role/axis/order/length and available-arm strata."""
    pairs, states = {}, {}
    for r in metadata:
        key = (r["pair_id"], r["arm"])
        if key in states and states[key] != r["state_index"]:
            raise ValueError("query-dependent state index")
        states[key] = r["state_index"]
        core = tuple(r[k] for k in ("role", "axis", "order", "tokens"))
        if r["pair_id"] in pairs and pairs[r["pair_id"]]["core"] != core:
            raise ValueError("pair opportunity changes across arms")
        pairs.setdefault(r["pair_id"], {"core": core, "arms": set()})["arms"].add(
            r["arm"]
        )
    groups = defaultdict(list)
    for p, v in sorted(pairs.items()):
        groups[(*v["core"], tuple(sorted(v["arms"])))].append(p)
    rng, mapping, table = np.random.default_rng(seed), {}, []
    for core, pair_ids in sorted(groups.items()):
        cycle = rng.permutation(pair_ids).tolist()
        for j, p in enumerate(cycle):
            other = cycle[(j + 1) % len(cycle)]
            mapping[p] = other
            table.append(
                {
                    "pair_id": p,
                    "context_pair_id": other,
                    "role": core[0],
                    "axis": core[1],
                    "order": core[2],
                    "tokens": core[3],
                    "arm_profile": "+".join(core[4]),
                    "stratum_pairs": len(cycle),
                    "pair_moved": p != other,
                }
            )
    result = np.full(1 + max(states.values()), -1, dtype=int)
    for (p, arm), i in states.items():
        result[i] = states[mapping[p], arm]
    if np.min(result) < 0 or len(np.unique(result)) != len(result):
        raise ValueError("context mapping must be a complete state permutation")
    return result, sorted(table, key=lambda r: r["pair_id"])
