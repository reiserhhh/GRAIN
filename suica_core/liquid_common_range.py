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


def whiten(x, *, weights=None, noise_cov=None):
    if weights is None and noise_cov is None:
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

    w_arr = (
        np.ones(x.shape[:2], dtype=float)
        if weights is None
        else np.asarray(weights, float)
    )
    if w_arr.ndim == 1 and len(w_arr) == len(x):
        w_arr = np.broadcast_to(w_arr[:, None], x.shape[:2])
    if w_arr.shape != x.shape[:2] or not np.isfinite(w_arr).all() or np.any(w_arr <= 0):
        raise ValueError("weights must be positive and match (N, T)")
    wn = w_arr / float(np.sum(w_arr))
    mean = np.einsum("nt,ntd->d", wn, x)
    d = x - mean
    cov_mat = np.einsum("nt,nti,ntj->ij", wn, d, d)
    if noise_cov is not None:
        nc = np.asarray(noise_cov, float)
        if nc.shape != (x.shape[-1], x.shape[-1]) or not np.isfinite(nc).all():
            raise ValueError("noise_cov must be finite (D, D)")
        cov_mat = cov_mat - nc
    cov_mat = 0.5 * (cov_mat + cov_mat.T)
    vals, Q = np.linalg.eigh(cov_mat)
    if vals[0] <= max(1e-15, vals[-1] * 1e-10):
        raise ValueError(
            "source block is rank deficient; change the declared observed basis"
        )
    floor = max(1e-12, float(np.max(vals)) * 1e-8)
    vals = np.maximum(vals, floor)
    w = (Q / np.sqrt(vals)) @ Q.T
    return d @ w, mean, w


def _align_view_counts(counts, x_raw, quantity, split):
    c = np.asarray(counts, float)
    if c.ndim == 2:
        c = np.stack([c, c], axis=-1)
    if c.ndim != 3 or c.shape[0] != len(x_raw) or c.shape[2] != 2:
        raise ValueError("counts must have shape (N, T) or (N, T, 2)")
    if not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("counts must be positive and finite")
    if quantity == "level":
        if c.shape[1] != x_raw.shape[1]:
            raise ValueError("one count per original time step required")
        return c.shape[1] ** 2 / np.sum(1 / c, axis=1, keepdims=True)
    if quantity == "move" and x_raw.shape[1] > 1:
        if c.shape[1] == x_raw.shape[1]:
            return (c[:, 1:] * c[:, :-1]) / (c[:, 1:] + c[:, :-1])
        if c.shape[1] == x_raw.shape[1] - 1:
            return c
    if quantity == "event" and c.shape[1] == x_raw.shape[1]:
        return c
    raise ValueError("counts time axis must match input or transformed paths")


def fit(
    x,
    axes,
    split,
    basis,
    quantity,
    *,
    permutations=99,
    counts=None,
    noise_cov=None,
    event_var=None,
):
    """Independent units in axis 0. Supports optional heteroscedastic precision weighting."""
    noise = None
    if event_var is not None:
        if noise_cov is not None:
            raise ValueError("choose shrinkage or covariance correction")
        from .stream_precision import eb_apply, eb_fit

        if (
            quantity != "event"
            or counts is None
            or event_var is None
            or len(counts) != 2
            or len(event_var) != 2
        ):
            raise ValueError(
                "precision chart requires event means, two count arrays and two variance vectors"
            )
        blocks = np.split(np.asarray(x, float), [split], axis=-1)
        noise = [
            eb_fit(a, n, v) for a, n, v in zip(blocks, counts, event_var, strict=True)
        ]
        x = np.concatenate(
            [eb_apply(a, n, r) for a, n, r in zip(blocks, counts, noise, strict=True)],
            axis=-1,
        )
        counts = None
    x_raw = np.asarray(x, float)
    x_t = transform(x_raw, quantity)
    if (
        not basis
        or not 0 < split < x_t.shape[-1]
        or len(axes) != x_t.shape[-1]
        or len(set(axes)) != len(axes)
        or permutations < 19
        or len(x_t) < 4
    ):
        raise ValueError("invalid paired reference")

    if counts is None and noise_cov is None:
        a, ma, wa = whiten(x_t[:, :, :split])
        b, mb, wb = whiten(x_t[:, :, split:])

        def cov(a, b):
            return np.einsum("nti,ntj->ij", a, b) / (len(a) * a.shape[1])

        u, roots, vt = np.linalg.svd(cov(a, b), full_matrices=True)
        rng = np.random.default_rng(517)
        cutoff = float(
            np.quantile(
                [
                    np.linalg.svd(cov(a, b[rng.permutation(len(b))]), compute_uv=False)[
                        0
                    ]
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
            **({"precision_chart": noise} if noise is not None else {}),
            "quantity": quantity,
            "axes": list(axes),
            "split": split,
            "basis": basis,
            "loadings": [(wa @ u).tolist(), (wb @ vt.T).tolist()],
            "means": [ma.tolist(), mb.tolist()],
            "rank": k,
            "roots": roots.tolist(),
            "largest_null_root_p95": cutoff,
            "source_people": len(x_t),
            "permutations": permutations,
            "clock_reference": "common raw chart across readers, not refitted on held people",
        }
        ref["coordinate_id"] = li.digest(ref)
        return ref

    if counts is None:
        c_aligned = np.ones((x_t.shape[0], x_t.shape[1], 2), dtype=float)
    else:
        c_aligned = _align_view_counts(counts, x_raw, quantity, split)
    c1, c2 = c_aligned[:, :, 0], c_aligned[:, :, 1]
    w1 = c1 / (c1 + 1.0)
    w2 = c2 / (c2 + 1.0)
    nc1, nc2 = noise_cov if noise_cov is not None else (None, None)
    eff_nc1 = (
        None
        if nc1 is None
        else (float(np.sum((w1 / w1.sum()) / c1)) * np.asarray(nc1, float))
    )
    eff_nc2 = (
        None
        if nc2 is None
        else (float(np.sum((w2 / w2.sum()) / c2)) * np.asarray(nc2, float))
    )
    a, ma, wa = whiten(x_t[:, :, :split], weights=w1, noise_cov=eff_nc1)
    b, mb, wb = whiten(x_t[:, :, split:], weights=w2, noise_cov=eff_nc2)

    def wcov(aa, bb, ww1, ww2):
        wp = np.sqrt(ww1 * ww2)
        wp /= float(np.sum(wp))
        return np.einsum("nt,nti,ntj->ij", wp, aa, bb)

    u, roots, vt = np.linalg.svd(wcov(a, b, w1, w2), full_matrices=True)
    rng = np.random.default_rng(517)
    null_roots = []
    for _ in range(permutations):
        perm = rng.permutation(len(b))
        null_roots.append(
            np.linalg.svd(wcov(a, b[perm], w1, w2[perm]), compute_uv=False)[0]
        )
    cutoff = float(np.quantile(null_roots, 0.95, method="higher"))
    k = int(np.sum(roots > cutoff))
    for j in range(len(roots)):
        sign = -1 if u[np.abs(u[:, j]).argmax(), j] < 0 else 1
        u[:, j] *= sign
        vt[j] *= sign
    ref = {
        "version": "grain-common-range-1",
        **({"precision_chart": noise} if noise is not None else {}),
        "quantity": quantity,
        "axes": list(axes),
        "split": split,
        "basis": basis,
        "heteroscedastic": True,
        "loadings": [(wa @ u).tolist(), (wb @ vt.T).tolist()],
        "means": [ma.tolist(), mb.tolist()],
        "rank": k,
        "roots": roots.tolist(),
        "largest_null_root_p95": cutoff,
        "source_people": len(x_t),
        "permutations": permutations,
        "clock_reference": "common raw chart across readers, not refitted on held people",
    }
    ref["coordinate_id"] = li.digest(ref)
    return ref


def coords(ref, views, counts=None):
    if len(views) != 2:
        raise ValueError("provide two blocks; null denotes a whole unobserved block")
    out = []
    if ("precision_chart" in ref) != (counts is not None) or (
        counts is not None and len(counts) != 2
    ):
        raise ValueError("counts must match frozen chart contract")
    for j, (x, mu, w) in enumerate(
        zip(views, ref["means"], ref["loadings"], strict=True)
    ):
        if x is None:
            out.append(None)
            continue
        x = np.asarray(x, float)
        if x.shape != (len(mu),) or not np.isfinite(x).all():
            raise ValueError("incomplete or wrong field block")
        if counts is not None:
            from .stream_precision import eb_apply

            x = eb_apply(x, counts[j], ref["precision_chart"][j])
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


def estimate(
    ref,
    views,
    budget,
    *,
    counts=None,
    event_sigma=None,
    z_alpha=1.6448536269514722,
):
    """Exact Euclidean ball of shared coordinates; optionally separates sampling underpowering (G14)."""
    if budget is not None and (not np.isfinite(budget) or budget < 0):
        raise ValueError("invalid private norm allowance")
    k = ref["rank"]
    if event_sigma is not None and (
        counts is None or not np.isfinite(z_alpha) or z_alpha < 0
    ):
        raise ValueError(
            "sampling diagnostic requires counts and a nonnegative finite multiplier"
        )
    if "precision_chart" in ref and event_sigma is not None:
        raise ValueError("do not double count uncertainty")
    v = coords(ref, views, counts if "precision_chart" in ref else None)
    c, _, e, n = geometry(v, k)
    if counts is not None and event_sigma is not None:
        if len(counts) != 2 or len(event_sigma) != 2:
            raise ValueError("counts and event_sigma must align with two views")
        prec = []
        shared_obs = []
        noise_sq_sum = 0.0
        for j, a in enumerate(v):
            if a is None:
                continue
            cj = float(counts[j])
            if not np.isfinite(cj) or cj <= 0:
                raise ValueError("positive count required for each observed block")
            sj = float(event_sigma[j])
            if not np.isfinite(sj) or sj < 0:
                raise ValueError("event_sigma must be non-negative and finite")
            sv_j = (sj**2) / cj
            noise_sq_sum += float(len(a)) * sv_j
            b_prior = 0.5 * (float(budget) ** 2) if budget is not None else 0.5
            prec.append(1.0 / max(1e-9, b_prior + sv_j))
            shared_obs.append(a[:k])
        prec_arr = np.asarray(prec, dtype=float)
        prec_arr /= float(np.sum(prec_arr))
        center_w = np.sum(
            prec_arr[:, None] * np.asarray(shared_obs, dtype=float), axis=0
        )
    else:
        center_w = c
        noise_sq_sum = 0.0

    zero = np.zeros(k)
    out = {
        "rank": k,
        "coordinate_id": ref["coordinate_id"],
        "observed_blocks": n,
        "minimum_private_norm": float(np.sqrt(e)),
        "norm_for_entirely_private": float(np.sqrt(e + n * (c @ c))),
        "center": c.tolist(),
        **(
            {"precision_weighted_point": center_w.tolist()}
            if event_sigma is not None
            else {}
        ),
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
    if counts is None or event_sigma is None:
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

    eff_sq = float(budget) ** 2 + (float(z_alpha) ** 2) * noise_sq_sum
    if e <= float(budget) ** 2 + 1e-12 * max(1.0, e, float(budget) ** 2):
        status = "compatible"
        structural_exceeded = False
        rad = float(np.sqrt(max(0.0, float(budget) ** 2 - e) / n)) if k else 0.0
    elif e <= eff_sq + 1e-12 * max(1.0, e, eff_sq):
        status = "sampling_underpowered"
        structural_exceeded = True
        rad = float(np.sqrt(max(0.0, eff_sq - e) / n)) if k else 0.0
    else:
        return {
            **out,
            "status": "outside_private_budget",
            "sampling_adjusted_budget": float(np.sqrt(eff_sq)),
            "structural_budget_exceeded": True,
        }
    bounds = np.column_stack((c - rad, c + rad)).tolist()
    return {
        **out,
        "status": status,
        "sampling_adjusted_budget": float(np.sqrt(eff_sq)),
        "structural_budget_exceeded": structural_exceeded,
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

        counts = [validate(n, x.shape[:2]).ravel() for n in counts]
    vals = [
        geometry(
            coords(
                ref,
                [v[:cut], v[cut:]],
                None if counts is None else [n[i] for n in counts],
            ),
            ref["rank"],
        )[2]
        for i, v in enumerate(x.reshape(-1, x.shape[-1]))
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
    ns = doc.get("counts", [None] * len(doc["observations"]))
    for record, n in zip(doc["observations"], ns, strict=True):
        if record is None or all(v is None for v in record):
            out.append({"state": {"status": "no_observation"}, "from_previous": None})
            previous = None
            continue
        est = estimate(
            ref,
            record,
            doc["budget"],
            counts=n,
            event_sigma=doc.get("event_sigma"),
            z_alpha=doc.get("z_alpha", 1.6448536269514722),
        )
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
