#!/usr/bin/env python3
"""Native full-function sensitivity outside the current frozen GRAIN moments."""

import numpy as np
import pandas as pd

MAPS = ("old_mean_pca32", "body_mean", "grain_moments")
VIEWS = ("neuroticism", "confirmation_seeking", "query_mean", "query_difference")
TABLES = (
    "input_geometry",
    "gradient_geometry",
    "responses",
    "query_geometry",
    "magnitude_links",
    "comparisons",
)
COUNTS = [2, 24, 120, 18, 32, 8]


def ratio(a, b):
    """Keep undefined zero-denominator quantities separate from measured zeros."""
    return float(a / b) if b > 0 else np.nan


def basis(grain):
    """Return the thin token-index basis of intercept and four centered patterns."""
    g = np.asarray(grain, float)
    if g.ndim != 2 or g.shape[1] != 4 or len(g) < 2 or not np.isfinite(g).all():
        raise ValueError("Finite BODY rows by four fixed GRAIN patterns required")
    a = np.column_stack((np.ones(len(g)), g - g.mean(axis=0)))
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    cutoff = max(a.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > cutoff))
    q = u[:, :rank]
    return (
        a,
        q,
        {
            "tokens": len(g),
            "rank": rank,
            "null_token_dimension": len(g) - rank,
            "condition": float(s[0] / s[rank - 1]),
        },
    )


def project(x, q, pca, method):
    """Apply nested orthogonal feature-gradient projectors without a dense T matrix."""
    x = np.asarray(x, float)
    if (
        x.ndim != 2
        or q.shape[0] != len(x)
        or pca.shape[0] != x.shape[1]
        or not np.isfinite(x).all()
    ):
        raise ValueError("Aligned finite BODY matrix and feature bases required")
    if method == "old_mean_pca32":
        mean = (x.mean(axis=0) @ pca) @ pca.T
        return np.broadcast_to(mean, x.shape)
    if method == "body_mean":
        return np.broadcast_to(x.mean(axis=0), x.shape)
    if method != "grain_moments":
        raise ValueError("Unknown frozen feature map")
    return x.copy() if q.shape[1] == len(x) else q @ (q.T @ x)


def cosine(a, b):
    """Cosine is undefined for a zero vector; do not invent zero similarity."""
    aa, bb = float(np.sum(a * a)), float(np.sum(b * b))
    return ratio(float(np.sum(a * b)), np.sqrt(aa * bb))


def analyze(grain, directions, gradients, pca):
    """Analyze one paired author, retaining signed actions and target dependence."""
    g, ds, jj, u = [np.asarray(x, float) for x in (grain, directions, gradients, pca)]
    a, q, geom = basis(g)
    if (
        jj.ndim != 3
        or jj.shape[:2] != (2, len(g))
        or ds.shape != (4, *jj.shape[1:])
        or u.ndim != 2
        or u.shape[0] != jj.shape[-1]
    ):
        raise ValueError(
            "Two paired gradients, four BODY directions and PCA basis required"
        )
    if not all(np.isfinite(x).all() for x in (ds, jj, u)):
        raise ValueError("Finite paired fields required")
    np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-10)
    checks = {"orthogonal_basis": float(np.max(abs(q.T @ q - np.eye(q.shape[1]))))}

    def check(name, discrepancy, scale=1.0):
        value = float(abs(discrepancy)) / max(float(scale), np.finfo(float).tiny)
        checks[name] = max(checks.get(name, 0.0), value)

    fields = (jj[0], jj[1], jj.mean(axis=0), jj[0] - jj[1])
    dmean = ds.mean(axis=1)
    dcoeff = {
        "old_mean_pca32": (dmean @ u) * np.sqrt(len(g)),
        "body_mean": dmean * np.sqrt(len(g)),
        "grain_moments": np.einsum("tk,fth->fkh", q, ds),
    }
    denergy = np.sum(ds**2, axis=(1, 2))
    dvisible = {k: np.sum(v.reshape(4, -1) ** 2, axis=1) for k, v in dcoeff.items()}
    dnull = {k: np.maximum(0.0, denergy - v) for k, v in dvisible.items()}
    current_null_moment = np.einsum("tk,fth->fkh", a, ds) - np.einsum(
        "lk,fkh->flh", a.T @ q, dcoeff["grain_moments"]
    )
    check(
        "direction_feature_null",
        np.linalg.norm(current_null_moment),
        max(1.0, np.linalg.norm(a) * np.linalg.norm(ds)),
    )
    rows = {k: [] for k in TABLES}
    rows["input_geometry"].append(geom)
    store = {}
    for view, field in zip(VIEWS, fields, strict=True):
        total = float(np.sum(field**2))
        energies = []
        for method in MAPS:
            visible = project(field, q, u, method)
            null = field - visible
            store[view, method] = (field, visible, null)
            ve, ne = float(np.sum(visible**2)), float(np.sum(null**2))
            check("gradient_energy", ve + ne - total, max(1.0, total))
            check("gradient_orthogonality", np.sum(visible * null), max(1.0, total))
            witness = float(np.sum(field * null) / np.sqrt(ne)) if ne > 0 else 0.0
            check(
                "optimal_null_witness", witness - np.sqrt(ne), max(1.0, np.sqrt(total))
            )
            if method == "grain_moments":
                check(
                    "current_feature_null",
                    np.linalg.norm(a.T @ null),
                    max(1.0, np.linalg.norm(a) * np.linalg.norm(field)),
                )
            rows["gradient_geometry"].append(
                {
                    "query": view,
                    "map": method,
                    "native_energy": total,
                    "visible_energy": ve,
                    "null_energy": ne,
                    "null_fraction": ratio(ne, total),
                    "retained_fraction": ratio(ve, total),
                    "optimal_null_response": witness,
                    "structural_null_zero": q.shape[1] == len(g)
                    if method == "grain_moments"
                    else False,
                }
            )
            energies.append(ve)
            fc = {
                "old_mean_pca32": (field.mean(axis=0) @ u) * np.sqrt(len(g)),
                "body_mean": field.mean(axis=0) * np.sqrt(len(g)),
                "grain_moments": q.T @ field,
            }[method]
            native_actions = np.einsum("th,fth->f", field, ds)
            null_actions = np.einsum("th,fth->f", null, ds)
            for fi, d in enumerate(ds):
                native, retained, missed = (
                    float(native_actions[fi]),
                    float(np.sum(fc * dcoeff[method][fi])),
                    float(null_actions[fi]),
                )
                check(
                    "signed_action_sum",
                    retained + missed - native,
                    max(1.0, abs(native)),
                )
                check(
                    "direction_energy",
                    dvisible[method][fi] + dnull[method][fi] - denergy[fi],
                    max(1.0, denergy[fi]),
                )
                rows["responses"].append(
                    {
                        "query": view,
                        "map": method,
                        "factor": f"F{fi + 1:02d}",
                        "native": native,
                        "visible": retained,
                        "null": missed,
                        "direction_energy": float(denergy[fi]),
                        "null_direction_energy": float(dnull[method][fi]),
                    }
                )
        check(
            "nested_retention",
            max(0.0, energies[0] - energies[1], energies[1] - energies[2]),
            max(1.0, total),
        )
        rows["comparisons"].append(
            {
                "query": view,
                "native_energy": total,
                "old_visible_energy": energies[0],
                "mean_visible_energy": energies[1],
                "current_visible_energy": energies[2],
                "mean_expansion_fraction": ratio(energies[1] - energies[0], total),
                "grain_increment_fraction": ratio(energies[2] - energies[1], total),
            }
        )
        residual = store[view, "grain_moments"][2]
        power = np.sum(residual**2, axis=1)
        centered_power = power - power.mean()
        for fi in range(4):
            gc = g[:, fi] - g[:, fi].mean()
            rows["magnitude_links"].append(
                {
                    "query": view,
                    "factor": f"F{fi + 1:02d}",
                    "within_token_r": cosine(gc, centered_power),
                    "power_covariance": float(np.mean(gc * centered_power)),
                    "linear_moment_norm": float(np.linalg.norm(gc @ residual / len(g))),
                }
            )
    for method in MAPS:
        for pi, part in enumerate(("native", "visible", "null")):
            n, c = [store[v, method][pi] for v in VIEWS[:2]]
            mean, diff = (n + c) / 2, n - c
            energy = float(np.sum(n**2) + np.sum(c**2))
            common, contrast = float(2 * np.sum(mean**2)), float(np.sum(diff**2) / 2)
            check("paired_energy", common + contrast - energy, max(1.0, energy))
            rows["query_geometry"].append(
                {
                    "map": method,
                    "part": part,
                    "pair_energy": energy,
                    "common_energy": common,
                    "difference_energy": contrast,
                    "common_fraction": ratio(common, energy),
                    "cosine": cosine(n, c),
                }
            )
    if max(checks.values()) > 1e-8:
        raise ValueError("Moment-map geometry or contraction identity failed")
    return rows, checks


def aggregate(raw):
    """Publish cohort summaries, never pool token rows as independent people."""
    result = {}
    keys = {
        "input_geometry": [],
        "gradient_geometry": ["query", "map"],
        "responses": ["query", "map", "factor"],
        "query_geometry": ["map", "part"],
        "magnitude_links": ["query", "factor"],
        "comparisons": ["query"],
    }
    for table in TABLES:
        data = raw[table]
        if table == "responses":
            all_four = data.copy()
            all_four["factor"] = "all4"
            data = pd.concat((data, all_four), ignore_index=True)
        rows = []
        for index, part in data.groupby(["cohort", *keys[table]], sort=True):
            index = index if isinstance(index, tuple) else (index,)
            meta = dict(zip(["cohort", *keys[table]], index, strict=True))
            meta["authors"] = int(part.author_key.nunique())
            if table == "input_geometry":
                values = {
                    f"{k}_{s}": float(getattr(part[k], s)())
                    for k in ("tokens", "rank", "null_token_dimension", "condition")
                    for s in ("min", "median", "max")
                }
            elif table == "gradient_geometry":
                values = {
                    k: float(part[k].sum())
                    for k in ("native_energy", "visible_energy", "null_energy")
                }
                values.update(
                    null_fraction=ratio(values["null_energy"], values["native_energy"]),
                    retained_fraction=ratio(
                        values["visible_energy"], values["native_energy"]
                    ),
                    median_author_null_fraction=float(part.null_fraction.median()),
                    null_response_rms=float(
                        np.sqrt(np.mean(part.optimal_null_response**2))
                    ),
                    zero_native_authors=int((part.native_energy == 0).sum()),
                )
            elif table == "responses":
                n, v, z = [
                    part[k].to_numpy(float) for k in ("native", "visible", "null")
                ]
                en = float(n @ n)
                values = {
                    "directional_observations": len(n),
                    "native_rms": float(np.sqrt(np.mean(n**2))),
                    "visible_rms": float(np.sqrt(np.mean(v**2))),
                    "null_rms": float(np.sqrt(np.mean(z**2))),
                    "null_projection": ratio(float(z @ n), en),
                    "visible_projection": ratio(float(v @ n), en),
                    "null_squared_ratio": ratio(float(z @ z), en),
                    "visible_squared_ratio": ratio(float(v @ v), en),
                    "signed_cross_ratio": ratio(float(2 * v @ z), en),
                    "null_direction_fraction": ratio(
                        float(part.null_direction_energy.sum()),
                        float(part.direction_energy.sum()),
                    ),
                }
            elif table == "query_geometry":
                values = {
                    k: float(part[k].sum())
                    for k in ("pair_energy", "common_energy", "difference_energy")
                }
                values.update(
                    common_fraction=ratio(
                        values["common_energy"], values["pair_energy"]
                    ),
                    mean_author_cosine=float(part.cosine.mean()),
                    median_author_cosine=float(part.cosine.median()),
                    valid_cosines=int(part.cosine.notna().sum()),
                )
            elif table == "magnitude_links":
                values = {
                    "mean_author_r": float(part.within_token_r.mean()),
                    "median_author_r": float(part.within_token_r.median()),
                    "positive_author_count": int((part.within_token_r > 0).sum()),
                    "valid_authors": int(part.within_token_r.notna().sum()),
                    "mean_power_covariance": float(part.power_covariance.mean()),
                    "max_linear_moment_norm": float(part.linear_moment_norm.max()),
                }
            else:
                values = {
                    k: float(part[k].sum())
                    for k in (
                        "native_energy",
                        "old_visible_energy",
                        "mean_visible_energy",
                        "current_visible_energy",
                    )
                }
                values.update(
                    mean_expansion_fraction=ratio(
                        values["mean_visible_energy"] - values["old_visible_energy"],
                        values["native_energy"],
                    ),
                    grain_increment_fraction=ratio(
                        values["current_visible_energy"]
                        - values["mean_visible_energy"],
                        values["native_energy"],
                    ),
                    median_grain_increment=float(
                        part.grain_increment_fraction.median()
                    ),
                )
            rows.append(meta | values)
        result[table] = pd.DataFrame(rows)
    return result
