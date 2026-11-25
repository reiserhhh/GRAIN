#!/usr/bin/env python3
"""Learn two-view directions, then return all shared readings compatible with private energy."""

import argparse

import numpy as np

from . import liquid_instrument as li


def transform(paths, quantity):
    x = np.asarray(paths, float)
    if x.ndim != 3 or not x.size or not np.isfinite(x).all():
        raise ValueError("complete aligned numeric paths required")
    if quantity == "level":
        return x.mean(1, keepdims=True)
    if quantity == "move" and x.shape[1] > 1:
        return np.diff(x, axis=1)
    if quantity == "event":
        return x
    raise ValueError("unknown quantity or no observed transition")


def whiten(x):
    mean = x.mean((0, 1))
    d = x - mean
    flat = d.reshape(-1, d.shape[-1])
    vals, Q = np.linalg.eigh(flat.T @ flat / len(flat))
    if vals[0] <= max(1e-15, vals[-1] * 1e-10):
        raise ValueError(
            "source block is rank deficient; change the declared observed basis"
        )
    w = (Q / np.sqrt(vals)) @ Q.T
    return d @ w, mean, w


def fit(x, axes, split, basis, quantity, *, permutations=99, counts=None, event_var=None):
    """Independent units in axis 0. Null shuffles whole persons in one view at matched slots."""
    noise = None
    if counts is not None or event_var is not None:
        from .stream_precision import eb_apply, eb_fit
        if quantity != "event" or counts is None or event_var is None or len(counts)!=2 or len(event_var)!=2:
            raise ValueError("precision chart requires event means, two count arrays and two variance vectors")
        blocks = np.split(np.asarray(x,float),[split],axis=-1)
        noise = [eb_fit(a,n,v) for a,n,v in zip(blocks,counts,event_var,strict=True)]
        x = np.concatenate([eb_apply(a,n,r) for a,n,r in zip(blocks,counts,noise,strict=True)],axis=-1)
    x = transform(x, quantity)
    if (
        not basis
        or not 0 < split < x.shape[-1]
        or len(axes) != x.shape[-1]
        or len(set(axes)) != len(axes)
        or permutations < 19
        or len(x) < 4
    ):
        raise ValueError("invalid paired reference")
    a, ma, wa = whiten(x[:, :, :split])
    b, mb, wb = whiten(x[:, :, split:])

    def cov(a, b):
        return np.einsum("nti,ntj->ij", a, b) / (len(a) * a.shape[1])

    u, roots, vt = np.linalg.svd(cov(a, b), full_matrices=True)
    rng = np.random.default_rng(517)
    cutoff = float(
        np.quantile(
            [
                np.linalg.svd(cov(a, b[rng.permutation(len(b))]), compute_uv=False)[0]
                for _ in range(permutations)
            ],
            0.95,
            method="higher",
        )
    )
    k = int(np.sum(roots > cutoff))
    for j in range(len(roots)):
        sign = -1 if u[np.abs(u[:, j]).argmax(), j] < 0 else 1
        u[:, j] *= sign
        vt[j] *= sign
    ref = {
        "version": "grain-common-range-1",
        **({"precision_chart":noise} if noise is not None else {}),
        "quantity": quantity,
        "axes": list(axes),
        "split": split,
        "basis": basis,
        "loadings": [(wa @ u).tolist(), (wb @ vt.T).tolist()],
        "means": [ma.tolist(), mb.tolist()],
        "rank": k,
        "roots": roots.tolist(),
        "largest_null_root_p95": cutoff,
        "source_people": len(x),
        "permutations": permutations,
        "clock_reference": "common raw chart across readers, not refitted on held people",
    }
    ref["coordinate_id"] = li.digest(ref)
    return ref


def coords(ref, views, counts=None):
    if len(views) != 2:
        raise ValueError("provide two blocks; null denotes a whole unobserved block")
    out = []
    if ("precision_chart" in ref) != (counts is not None) or (counts is not None and len(counts)!=2):
        raise ValueError("counts must match frozen chart contract")
    for j,(x, mu, w) in enumerate(zip(views, ref["means"], ref["loadings"], strict=True)):
        if x is None:
            out.append(None)
            continue
        x = np.asarray(x, float)
        if x.shape != (len(mu),) or not np.isfinite(x).all():
            raise ValueError("incomplete or wrong field block")
        if counts is not None:
            from .stream_precision import eb_apply
            x = eb_apply(x,counts[j],ref["precision_chart"][j])
        out.append((x - mu) @ np.array(w))
    if all(a is None for a in out):
        raise ValueError("no observed field")
    return out


def geometry(v, k):
    seen = [a for a in v if a is not None]
    center = np.mean([a[:k] for a in seen], axis=0)
    private = [None if a is None else a - np.pad(center, (0, len(a) - k)) for a in v]
    emin = sum(float(p @ p) for p in private if p is not None)
    return center, private, emin, len(seen)


def witness(v, k, s):
    return [None if a is None else (a - np.pad(s, (0, len(a) - k))).tolist() for a in v]


def estimate(ref, views, budget, *, counts=None):
    """Exact Euclidean ball of shared coordinates; private effects need not be independent."""
    if budget is not None and (not np.isfinite(budget) or budget < 0):
        raise ValueError("invalid private norm allowance")
    k = ref["rank"]
    v = coords(ref, views, counts)
    c, _, e, n = geometry(v, k)
    zero = np.zeros(k)
    out = {
        "rank": k,
        "coordinate_id": ref["coordinate_id"],
        "observed_blocks": n,
        "minimum_private_norm": float(np.sqrt(e)),
        "norm_for_entirely_private": float(np.sqrt(e + n * (c @ c))),
        "center": c.tolist(),
        "canonical_observations": [None if a is None else a.tolist() for a in v],
        "minimum_private_witness": witness(v, k, c),
        "entirely_private_witness": witness(v, k, zero),
        "budget": budget,
        "radius": None,
    }
    if budget is None:
        return {
            **out,
            "status": "unrestricted_shared"
            if k
            else "no_paired_direction_in_reference",
        }
    rem = budget**2 - e
    if rem < -1e-12 * max(1.0, e, budget**2):
        return {**out, "status": "outside_private_budget"}
    rad = float(np.sqrt(max(0.0, rem) / n)) if k else 0.0
    bounds = np.column_stack((c - rad, c + rad)).tolist()
    return {
        **out,
        "status": "compatible",
        "radius": rad,
        "axis_bounds": bounds,
        "zero_shared_compatible": bool(np.linalg.norm(c) <= rad + 1e-12),
    }


def motion(a, b):
    """Minimum and maximum compatible common displacement, with witnesses on the center line."""
    if a["rank"] != b["rank"] or a["coordinate_id"] != b["coordinate_id"]:
        raise ValueError("inconsistent shared chart")
    if a["status"] != "compatible" or b["status"] != "compatible":
        return {"status": "no_bounded_pair"}
    c, d = np.array(a["center"]), np.array(b["center"])
    r, s = a["radius"], b["radius"]
    if not len(c):
        return {"status": "no_paired_direction_in_reference"}
    dist = float(np.linalg.norm(d - c))
    direction = (d - c) / dist if dist else np.eye(len(c))[0]
    t = c + min(dist, r) * direction
    mins = [c + r * direction, d - s * direction] if dist > r + s else [t, t]
    maxs = [c - r * direction, d + s * direction]
    return {
        "status": "bounded",
        "same_source_possible": bool(dist <= r + s),
        "min_move": max(0.0, dist - r - s),
        "max_move": dist + r + s,
        "witness_min": [v.tolist() for v in mins],
        "witness_max": [v.tolist() for v in maxs],
    }


def calibrate(ref, paths, *, quantile=0.90, counts=None):
    x = transform(paths, ref["quantity"])
    cut = ref["split"]
    if counts is not None:
        from .stream_precision import counts as validate
        counts = [validate(n,x.shape[:2]).ravel() for n in counts]
    vals = [
        geometry(coords(ref, [v[:cut], v[cut:]],None if counts is None else [n[i] for n in counts]), ref["rank"])[2]
        for i,v in enumerate(x.reshape(-1, x.shape[-1]))
    ]
    return float(np.quantile(np.sqrt(vals), quantile, method="higher"))


def score(ref, doc):
    if (
        doc["axes"] != ref["axes"]
        or doc["basis"] != ref["basis"]
        or doc["quantity"] != ref["quantity"]
    ):
        raise ValueError("different sensor, quantity or coordinates")
    if (
        "view_clock_digest" in ref
        and doc.get("view_clock_digest") != ref["view_clock_digest"]
    ):
        raise ValueError("different clock")
    out = []
    previous = None
    ns = doc.get("counts", [None]*len(doc["observations"]))
    for record,n in zip(doc["observations"],ns,strict=True):
        if record is None or all(v is None for v in record):
            out.append({"state": {"status": "no_observation"}, "from_previous": None})
            previous = None
            continue
        est = estimate(ref, record, doc["budget"], counts=n)
        out.append(
            {
                "state": est,
                "from_previous": motion(previous, est)
                if previous is not None
                else None,
            }
        )
        previous = est
    return {
        "reference": li.digest(ref),
        "results": out,
        "norm": "Euclidean norm of private deviations in both source-whitened blocks",
        "assumption": "a private budget constrains the explanation, not a confidence level; without it all common coordinates are possible",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("reference", "input", "out"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    reference = li.load(a.reference)
    li.save(a.out, score(reference.get("chart", reference), li.load(a.input)))
