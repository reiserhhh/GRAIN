#!/usr/bin/env python3
"""Apply released V6 opportunity and response-slope estimators to frozen liquid tracks."""

import argparse

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import factor_discovery as fd
from . import liquid_instrument as li

CHANNELS = ("native", "fixed_half", "liquid", "anchored_signed")
META = ["user_id", "half"]


def frame(data, channel):
    labels = data["node_names"]
    parts = []
    for key, ep in data["episodes"].items():
        x = np.asarray(ep[channel], dtype=float)
        if x.shape != (len(ep["slots"]), len(labels)) or not np.isfinite(x).all():
            raise ValueError("invalid slice/node coordinates")
        df = pd.DataFrame(x, columns=labels)
        for k, v in (
            ("user_id", ep["author"]),
            ("half", ep["view"]),
            ("condition", ep["condition"]),
            ("episode", key),
        ):
            df[k] = v
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["slice_index"] = full.groupby([*META, "condition"], sort=False).cumcount()
    return full


def pack(model):
    return {
        "feature_cols": model.feature_cols,
        "global_mean": model.global_mean.tolist(),
        "condition_mean": {k: v.tolist() for k, v in model.condition_mean.items()},
        "condition_z": {k: v.tolist() for k, v in model.condition_z.items()},
        "supported_conditions": sorted(model.supported_conditions),
        "axis_loadings": model.axis_loadings.tolist(),
        "axis_explained_variance": model.axis_explained_variance.tolist(),
    }


def unpack(obj):
    return fd.OpportunityModel(
        obj["feature_cols"],
        np.array(obj["global_mean"]),
        {k: np.array(v) for k, v in obj["condition_mean"].items()},
        {k: np.array(v) for k, v in obj["condition_z"].items()},
        set(obj["supported_conditions"]),
        np.array(obj["axis_loadings"]),
        np.array(obj["axis_explained_variance"]),
    )


def matrices(df, model):
    out = fd.build_family_features(df, model)
    # The release assumes that one person/half/condition is one continuous path.
    # Preserve its estimate, and separately supply a path that resets at episode boundaries.
    dyn = []
    for (who, view), g in df.groupby(META, sort=False):
        if len(g) < 8:
            continue
        seq = [
            e[model.feature_cols].to_numpy()
            - fd._condition_mean(model, str(e["condition"].iloc[0]))
            for _, e in g.groupby("episode", sort=False)
        ]
        desc = fd._pooled_path_descriptors(seq)
        row = dict(zip(META, (who, view), strict=True))
        row.update(
            {
                f"{k}::{node}": v
                for k, a in desc.items()
                for node, v in zip(model.feature_cols, a, strict=True)
            }
        )
        dyn.append(row)
    out["dynamic_episode_reset"] = pd.DataFrame(dyn)
    return out


def learn_axes(mats):
    axes = {}
    for name, table in mats.items():
        ax = {"columns": [], "k": 0}
        axes[name] = ax
        if table.empty:
            continue
        a, b = [
            table[table.half == v].set_index("user_id").drop(columns="half")
            for v in ("A", "B")
        ]
        ids = sorted(set(a.index) & set(b.index))
        if len(ids) < 4:
            continue
        x, y = [df.loc[ids].to_numpy(float) for df in (a, b)]
        z = np.vstack([x, y])
        valid = np.flatnonzero(np.isfinite(z).all(0) & (z.std(0) > 1e-8))
        if len(valid) < 2:
            continue
        m = fd.fit_stable_crossview(x[:, valid], y[:, valid])
        axes[name] = {
            "columns": list(a.columns[valid]),
            "k": m.n_factors,
            "center": m.center.tolist(),
            "scale": m.scale.tolist(),
            "directions": m.rotated_directions.tolist(),
        }
    return axes


def fit(data, native_reference):
    if (
        data["reference_digest"] != li.digest(native_reference)
        or data["input_digest"] != native_reference["source_digest"]
    ):
        raise ValueError(
            "fit requires the original source scored by its own frozen reader"
        )
    src = frame(data, "native")
    cols = data["node_names"]
    support = src.groupby("condition").user_id.nunique().to_dict()
    ref = {
        "version": "liquid-opportunity-v1",
        "reader_digest": data["reference_digest"],
        "source_input_digest": data["input_digest"],
        "node_names": cols,
        "source_authors_per_condition": support,
    }
    if sum(v >= 20 for v in support.values()) < 5:
        return {
            **ref,
            "status": "insufficient_source",
            "needed": "5 conditions each with 20 source authors for 3 opportunity axes",
        }
    base = fd.fit_opportunity_model(src, cols, set(src.user_id))
    coords = np.array([base.condition_z[c] for c in sorted(base.supported_conditions)])
    ref["source_opportunity_rank"] = int(np.linalg.matrix_rank(coords, tol=1e-9))
    if (
        not np.isfinite(base.axis_explained_variance).all()
        or ref["source_opportunity_rank"] < 3
    ):
        return {
            **ref,
            "status": "insufficient_geometry",
            "needed": "three linearly independent source opportunity coordinates",
        }
    ref.update(status="fitted", models={}, axes={})
    for channel in CHANNELS:
        f = src if channel == "native" else frame(data, channel)
        cells = f.groupby(["user_id", "condition"])[cols].mean()
        obj = pack(base)
        obj["condition_mean"] = {
            c: row.tolist()
            for c, row in cells.groupby(level="condition").mean().iterrows()
        }
        obj["global_mean"] = cells.mean().tolist()
        ref["models"][channel] = obj
        ref["axes"][channel] = learn_axes(matrices(f, unpack(obj)))
    return ref


def score(data, ref):
    if (
        ref["version"] != "liquid-opportunity-v1"
        or data["reference_digest"] != ref["reader_digest"]
        or data["node_names"] != ref["node_names"]
    ):
        raise ValueError("different reader or node coordinates")
    out = {
        "status": ref["status"],
        "opportunity_reference_digest": li.digest(ref),
        "input_digest": data["input_digest"],
    }
    if ref["status"] != "fitted":
        return {
            **out,
            "reason": ref["needed"],
            "source_authors_per_condition": ref["source_authors_per_condition"],
        }
    out["channels"] = {}
    for channel in CHANNELS:
        model = unpack(ref["models"][channel])
        tabs = matrices(frame(data, channel), model)
        out["channels"][channel] = {}
        for name, t in tabs.items():
            rows = []
            ax = ref["axes"][channel][name]
            for rec in t.to_dict(orient="records"):
                vals = {
                    k: float(v) if np.isfinite(v) else None
                    for k, v in rec.items()
                    if k not in META
                }
                z = [vals.get(c) for c in ax["columns"]]
                vec = None
                if ax["k"] and all(v is not None for v in z):
                    vec = (
                        (np.array(z) - ax["center"])
                        / ax["scale"]
                        @ np.array(ax["directions"])
                    ).tolist()
                rows.append(
                    {
                        **{k: rec[k] for k in META},
                        "features": vals,
                        "factor_scores": vec,
                    }
                )
            out["channels"][channel][name] = rows
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "score"))
    for name in ("input", "reference", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    with threadpool_limits(1):
        li.save(
            a.out,
            (fit if a.mode == "fit" else score)(li.load(a.input), li.load(a.reference)),
        )


if __name__ == "__main__":
    main()
