#!/usr/bin/env python3
"""Apply a frozen V6 author bank to original or liquid comment coordinates."""

import hashlib

import numpy as np
import pandas as pd

KEYS = ["user_id", "half"]


def user_key(user):
    return hashlib.sha256(str(user).encode()).hexdigest()


def segments(g):
    """Work on time-ordered rows; an unsampled position is a reset, not elapsed time."""
    if not g.order.is_monotonic_increasing or not g.order.is_unique:
        raise ValueError("duplicate or unordered opportunities within author/half")
    if g.user_id.nunique() != 1 or g.half.nunique() != 1:
        raise ValueError("do not join people or halves")
    run, cond = g.run_id.to_numpy(), g.condition.to_numpy()
    cut = (
        (np.diff(g.order) != 1)
        | (np.diff(g.run_pos) != 1)
        | (run[1:] != run[:-1])
        | (cond[1:] != cond[:-1])
    )
    return np.split(np.arange(len(g)), 1 + np.flatnonzero(cut))


def nuisance(meta, bank):
    q = meta[bank["q_columns"]].to_numpy(float)
    late = meta.half.eq("late").to_numpy(float)[:, None]
    if not meta.half.isin(["early", "late"]).all() or not np.isfinite(q).all():
        raise ValueError(
            "require original early/late assignment and finite surface features"
        )
    fold = meta.user_id.map(
        {u: bank["source_folds"].get(user_key(u), -1) for u in meta.user_id.unique()}
    ).to_numpy(int)
    models = bank["nuisance"]
    dim = len(models[0]["coef"][0])
    out = np.zeros((len(meta), dim))
    for i, m in enumerate(models):
        ix = np.flatnonzero((fold == i) | (fold < 0))
        if not len(ix):
            continue
        z = (q[ix] - m["mean"]) / m["scale"]
        design = np.column_stack([np.ones(len(ix)), z, late[ix], z * late[ix]])
        pred = design @ m["coef"]
        for (c, h), loc in (
            meta.iloc[ix].groupby(["condition", "half"], sort=False).indices.items()
        ):
            offset = m["cell"].get(h, {}).get(c)
            if offset is not None:
                pred[loc] += offset
        out[ix] += pred * np.where(fold[ix] < 0, 1 / len(models), 1)[:, None]
    return out


def object_frames(meta, res, zmap, cfg, pc):
    res = np.asarray(res, float)
    if res.ndim != 2 or len(meta) != len(res) or not np.isfinite(res).all():
        raise ValueError("invalid residual observations")
    names = [f"svd_{i + 1:02d}" for i in range(res.shape[1])]
    sc = ["static::" + i for i in names]
    hc = ["hybrid::" + i for i in names]
    pcx = (res - pc["mean"]) @ np.asarray(pc["components"]).T
    d = pcx.shape[1]
    dc = (
        [f"dynamic::log_variance_{i + 1}" for i in range(d)]
        + [f"dynamic::phi_{i + 1}_{j + 1}" for i in range(d) for j in range(d)]
        + [f"dynamic::roughness_{i + 1}" for i in range(d)]
        + [f"dynamic::changepoint_{i + 1}" for i in range(d)]
    )
    sr, hr, dr = [], [], []
    for (u, h), indices in meta.groupby(KEYS, sort=False).indices.items():
        ix = indices[np.argsort(meta.order.to_numpy()[indices], kind="stable")]
        g = meta.iloc[ix]
        cond, codes = np.unique(g.condition.to_numpy(str), return_inverse=True)
        counts = np.bincount(codes)
        cell = np.zeros((len(cond), len(sc)))
        np.add.at(cell, codes, res[ix])
        cell /= counts[:, None]
        center = cell.mean(0)
        z = np.array([zmap.get(c, np.nan) for c in cond])
        ok = np.isfinite(z)
        info = [u, h, len(g), len(cond)]
        sr.append([*info, float(z[ok].mean()) if ok.any() else np.nan, *center])
        if (
            len(g) >= cfg["hybrid_min_comments"]
            and ok.sum() >= cfg["hybrid_min_conditions"]
            and counts.max() / len(g) <= cfg["hybrid_max_condition_share"]
        ):
            dz = z[ok] - z[ok].mean()
            den = dz @ dz + cfg["hybrid_ridge_alpha"]
            if dz @ dz / den >= cfg["hybrid_min_information"]:
                hr.append([*info, *(dz @ (cell[ok] - center) / den)])
        seqs = [
            pcx[ix[sl]]
            for sl in segments(g)
            if len(sl) >= cfg["dynamic_min_run_length"]
        ]
        n = sum(len(seq) - 1 for seq in seqs)
        if len(seqs) < cfg["dynamic_min_runs"] or n < cfg["dynamic_min_transitions"]:
            continue
        ss = [s - s.mean(0) for s in seqs]
        cov = sum(s.T @ s for s in ss) / (n + len(ss))
        phi = (sum(s[1:].T @ s[:-1] for s in ss) / n) @ np.linalg.inv(
            cov + 0.1 * np.eye(d)
        )
        rough = np.sqrt(sum((np.diff(s, axis=0) ** 2).sum(0) for s in ss) / n)
        cp = np.median(
            [
                np.max(abs(np.cumsum(s, axis=0)[:-1]), axis=0) / np.sqrt(len(s))
                for s in ss
            ],
            axis=0,
        )
        dr.append(
            [
                *info,
                *np.log(np.maximum(np.linalg.eigvalsh(cov), 1e-8))[::-1],
                *phi.ravel(),
                *rough,
                *cp,
            ]
        )
    cols = KEYS + ["meta_comments", "meta_conditions"]
    return (
        pd.DataFrame(sr, columns=cols + ["meta_opportunity_choice"] + sc),
        pd.DataFrame(hr, columns=cols + hc),
        pd.DataFrame(dr, columns=cols + dc),
    )


def joined(target, predictors):
    df = target.copy()
    names = []
    for j, p in enumerate(predictors):
        cols = [c for c in p if "::" in c]
        rename = {c: f"p{j}::{c}" for c in cols}
        df = df.merge(
            p[KEYS + cols].rename(columns=rename),
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        names.extend(rename.values())
    return df, names


def innovation(target, predictors, model):
    df, cols = joined(target, predictors)
    if cols != model["predictors"]:
        raise ValueError("incompatible predictor columns")
    x = (df[cols].to_numpy(float) - model["mean"]) / model["scale"]
    df[model["targets"]] = df[model["targets"]].to_numpy(float) - (
        x @ model["coef"] + model["bias"]
    )
    return df[target.columns]


def project(frame, model):
    x = frame[model["cols"]].to_numpy(float)
    x = np.where(np.isfinite(x), x, model["center"])
    return ((x - model["center"]) / model["scale"]) @ model["dirs"]


def scores(meta, observations, bank, *, baseline=None):
    if baseline is None:
        baseline = nuisance(meta, bank)
    if np.shape(observations) != baseline.shape:
        raise ValueError("observation and correction must use matching axes")
    s, h, d = object_frames(
        meta, observations - baseline, bank["opportunities"], bank["config"], bank["pc"]
    )
    h = innovation(h, [s], bank["conditional"]["hybrid"])
    di = innovation(d, [s, h], bank["conditional"]["dynamic"])
    out = {}
    for family, frame in (("static", s), ("opportunity_response", h), ("dynamic", di)):
        a = project(frame, bank["families"][family])
        df = frame[KEYS].copy()
        for j in range(a.shape[1]):
            df[f"{family}_F{j + 1}"] = a[:, j]
        out[family] = df
    raw = s.merge(d, on=KEYS, validate="one_to_one")
    sh = bank["shared"]
    both = (project(raw, sh["static"]) + project(raw, sh["dynamic"])) / 2
    df = raw[KEYS].copy()
    for j in range(both.shape[1]):
        df[f"shared_hybrid_F{j + 1}"] = both[:, j]
    out["shared_hybrid"] = df
    return out
