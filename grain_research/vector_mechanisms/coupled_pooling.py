#!/usr/bin/env python3
"""A source-centered pooled quadratic nested in the existing compatible scalar."""

from __future__ import annotations

import numpy as np

from . import full_context_geometry as base
from .paired_response_objective import delta_energy

ARMS = ("matched_local", "coupled_quadratic")
VIEWS = ("original", "shifted", "plus", "minus")


def pooled(z):
    """Mean and one fixed positional moment, without author-level whitening."""
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("At least two finite token coordinate rows required")
    p = np.linspace(-1.0, 1.0, len(z))
    return np.r_[z.mean(0), (p[:, None] * z).mean(0)], p


def quadratic(z, center):
    """Return scalar coefficient features and derivatives with respect to pooled v."""
    v, p = pooled(z)
    v = v - center
    i, j = np.triu_indices(len(v))
    phi = v[i] * v[j]
    phi[i == j] *= 0.5
    derivative = np.zeros((len(i), len(v)))
    rows = np.arange(len(i))
    derivative[rows, i] = v[j]
    off = i != j
    derivative[rows[off], j[off]] = v[i[off]]
    return phi, derivative, p


def reference(original_bodies, lasts, frozen):
    """Use original source views only; do not recenter on transformed/current authors."""
    if len(original_bodies) != len(frozen["members"]) or len(lasts) != len(
        original_bodies
    ):
        raise ValueError("Original source membership required")
    values = [
        pooled(base.design(b, l, frozen)[0])[0]
        for b, l in zip(original_bodies, lasts, strict=True)
    ]
    return dict(
        frozen,
        pool_center=np.mean(values, axis=0),
        local_width=np.array(len(frozen["coef"])),
    )


def design(body, last, model):
    """Return the original scalar prefix and, only for the joint arm, its quadratic."""
    z, p, phi = base.design(body, last, model)
    if str(model["arm"]) == ARMS[1]:
        phi = np.r_[phi, quadratic(z, model["pool_center"])[0]]
    return z, p, phi


def density_blocks(z, p, last, model):
    """Stream T*dF/dz designs; quadratic derivatives use the whole fixed current view."""
    joint = str(model["arm"]) == ARMS[1]
    if joint:
        _, dv, pp = quadratic(z, model["pool_center"])
        rank = z.shape[1]
    for rows, local in base.density_blocks(z, p, last, model):
        if joint:
            extra = dv[:, :rank].T[None] + pp[rows, None, None] * dv[:, rank:].T[None]
            yield rows, np.concatenate((local, extra), axis=2)
        else:
            yield rows, local


def predict(body, last, model):
    """Compute score and BODY density from one scalar, with no native target arguments."""
    z, p, phi = design(body, last, model)
    if phi.shape != model["coef"].shape:
        raise ValueError("Scalar coefficient layout differs")
    return float(phi @ model["coef"]), np.concatenate(
        [d @ model["coef"] for _, d in density_blocks(z, p, last, model)]
    )


def scalar(body, last, model):
    """Evaluate score only for independent finite differences."""
    return float(design(body, last, model)[2] @ model["coef"])


def last_gradient(body, last, model):
    """The pooled quadratic has no last-state dependence; retain the local chain rule."""
    return base.last_gradient(
        body, last, dict(model, coef=model["coef"][: int(model["local_width"])])
    )


def layout(records, members):
    """Require exactly the four declared views for every known source author."""
    mapping = {(r["author"], r["view"]): i for i, r in enumerate(records)}
    expected = {(int(a), v) for a in members for v in VIEWS}
    if set(mapping) != expected or len(mapping) != len(records):
        raise ValueError(
            "Exactly four unique views per original source author required"
        )
    return [[mapping[int(a), v] for v in VIEWS] for a in members]


def targets(records, model):
    """Project native gradients in the unchanged source U/scale coordinate contract."""
    values = []
    for r in records:
        b, g = r["body"], r["gradient"]
        if b.shape != g.shape or any(
            not np.isfinite(x).all() for x in (b, g, r["last"], np.asarray(r["score"]))
        ):
            raise ValueError("Finite aligned source data required")
        values.append(len(b) * (g @ model["basis"]) * model["scale"])
    return values


def energies(records, model):
    """Separate source-normalized last-change and untouched-B change targets."""
    groups = layout(records, model["members"])
    ts = targets(records, model)
    last, cross = [], []
    for i, j, plus, minus in groups:
        if not np.array_equal(records[i]["body"], records[j]["body"]):
            raise ValueError("Last-only pair changed BODY")
        cut = len(records[i]["body"]) // 2
        for k in (plus, minus):
            np.testing.assert_array_equal(
                records[k]["body"][cut:], records[i]["body"][cut:]
            )
            np.testing.assert_array_equal(records[k]["last"], records[i]["last"])
        last.append((ts[j] - ts[i]) / 2)
        cross.append((ts[plus][cut:] - ts[minus][cut:]) / 2)
    return delta_energy(last, float(model["density_scale"])), delta_energy(
        cross, float(model["density_scale"])
    )


def normal_equations(records, model, scales):
    """Build value/field/paired normal equations; keep component constants explicit."""
    groups = layout(records, model["members"])
    ts = targets(records, model)
    coords = [design(r["body"], r["last"], model) for r in records]
    width = len(coords[0][2])
    grams = np.zeros((4, width, width))
    rhs = np.zeros((4, width))
    constants = np.zeros(4)
    sl, sk = float(model["score_scale"]), float(model["density_scale"])
    weight = 1 / len(records)

    def add(channel, a, target, normalization):
        a = a.reshape(-1, width) * normalization
        b = target.ravel() * normalization
        grams[channel] += a.T @ a
        rhs[channel] += a.T @ b
        constants[channel] += b @ b

    for r, (z, p, phi), t in zip(records, coords, ts, strict=True):
        add(0, phi, np.asarray(r["score"]), np.sqrt(weight) / sl)
        for rows, d in density_blocks(z, p, r["last"], model):
            add(1, d, t[rows], np.sqrt(weight / t.size) / sk)
    for i, j, plus, minus in groups:
        for channel, left, right, cut, energy in (
            (2, i, j, 0, scales[0]),
            (3, minus, plus, len(ts[i]) // 2, scales[1]),
        ):
            delta = (ts[right][cut:] - ts[left][cut:]) / 2
            streams = [
                density_blocks(*coords[k][:2], records[k]["last"], model)
                for k in (left, right)
            ]
            for (s0, a), (s1, b) in zip(*streams, strict=True):
                if s0 != s1:
                    raise ValueError("Paired block layout differs")
                start, end = max(s0.start, cut), s0.stop
                if start >= end:
                    continue
                d = (b[start - s0.start :] - a[start - s0.start :]) / 2
                if channel == 3 and np.any(d[:, :, : int(model["local_width"])]):
                    raise ValueError("Local cross-token derivative is not exactly zero")
                add(
                    channel,
                    d,
                    delta[start - cut : end - cut],
                    1 / np.sqrt(len(groups) * delta.size * energy),
                )
    return grams, rhs, constants


def objective(records, model, scales):
    """Recompute all losses directly from predictions, independent of Gram arithmetic."""
    groups = layout(records, model["members"])
    ts = targets(records, model)
    predictions = [predict(r["body"], r["last"], model) for r in records]
    value = (
        np.mean(
            [
                (p[0] - r["score"]) ** 2
                for p, r in zip(predictions, records, strict=True)
            ]
        )
        / float(model["score_scale"]) ** 2
    )
    field = (
        np.mean(
            [np.mean((p[1] - t) ** 2) for p, t in zip(predictions, ts, strict=True)]
        )
        / float(model["density_scale"]) ** 2
    )
    paired = [[], []]
    for i, j, plus, minus in groups:
        for channel, left, right, cut in (
            (0, i, j, 0),
            (1, minus, plus, len(ts[i]) // 2),
        ):
            e = (
                predictions[right][1] - predictions[left][1] - ts[right] + ts[left]
            ) / 2
            paired[channel].append(np.mean(e[cut:] ** 2))
    return np.array(
        [
            value,
            field,
            np.mean(paired[0]) / scales[0],
            np.mean(paired[1]) / scales[1],
            float(model["alpha"]) * np.sum(model["coef"][1:] ** 2),
        ]
    )


def fit(records, frozen):
    """Two nested convex solves on identical data, with an independently rebuilt control."""
    groups = layout(records, frozen["members"])
    original = [records[g[0]] for g in groups]
    ref = reference(
        [r["body"] for r in original], [r["last"] for r in original], frozen
    )
    scales = energies(records, ref)
    if not np.isclose(
        scales[0], float(frozen["source_delta_energy"]), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Frozen last-response normalization changed")
    joint = dict(ref, arm=np.array(ARMS[1]))
    gg, rr, cc = normal_equations(records, joint, scales)
    local = dict(ref, arm=np.array(ARMS[0]))
    lg, lr, lc = normal_equations(records, local, scales)
    size = int(ref["local_width"])
    prefix_error = float(
        max(
            np.max(abs(gg[:, :size, :size] - lg)),
            np.max(abs(rr[:, :size] - lr)),
            np.max(abs(cc - lc)),
        )
    )
    if prefix_error > 1e-8:
        raise ValueError("Independent matched control normal prefix differs")
    models, diagnostics = [], []
    for template, gs, rs, cs in ((local, lg, lr, lc), (joint, gg, rr, cc)):
        gram, rhs = gs.sum(0), rs.sum(0)
        penalty = np.eye(len(rhs)) * float(ref["alpha"])
        penalty[0, 0] = 0
        coef = np.linalg.solve(gram + penalty, rhs)
        model = dict(
            template,
            coef=coef,
            gram=gram,
            rhs=rhs,
            source_cross_energy=np.array(scales[1]),
        )
        loss = objective(records, model, scales)
        algebra = np.r_[
            cs - 2 * (rs @ coef) + np.einsum("i,kij,j->k", coef, gs, coef),
            coef @ penalty @ coef,
        ]
        error = float(np.max(abs(loss - algebra)))
        stationarity = float(np.max(abs((gram + penalty) @ coef - rhs)))
        if error > 1e-7 or stationarity > 1e-8 or not np.isfinite(coef).all():
            raise ValueError("Same-scalar objective/normal equations differ")
        models.append(model)
        diagnostics.append(
            {
                "candidate": str(model["arm"]),
                "parameters": len(coef),
                "source_authors": len(groups),
                "source_views": len(records),
                "source_last_energy": scales[0],
                "source_cross_energy": scales[1],
                "matched_prefix_error": prefix_error,
                "objective_reconstruction_error": error,
                "normal_equation_error": stationarity,
                "source_objective": float(loss.sum()),
            }
            | dict(
                zip(
                    (
                        "source_score_loss",
                        "source_density_loss",
                        "source_last_loss",
                        "source_cross_loss",
                        "source_penalty",
                    ),
                    map(float, loss),
                    strict=True,
                )
            )
        )
    if diagnostics[1]["source_objective"] > diagnostics[0]["source_objective"] + 1e-8:
        raise ValueError("Nested coupled objective increased")
    return models, diagnostics
