#!/usr/bin/env python3
"""Same-material query relations and exact learned evidence-mixture accounting."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.natural_transport import (
    correlation,
    fit_operator,
    paired_summary,
    removal_coordinate,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.run_grain_natural_transport import FACTORS, QUERIES


def query_relations(out):
    """Compare queries only on identical authors/views inside each source population."""
    fp = BASE / "factor_path_routes_v1/source_r2/private_factor_views.csv"
    pp = BASE / "dynamic_path_channels_v1/private_path_views.csv"
    factors = pd.read_csv(fp)
    paths = pd.read_csv(pp)
    observations = []
    maps = {}
    for group, queries in (("big5_only", tuple(BIG5)), ("mbti_only", tuple(AXES))):
        g = factors[factors.source_group == group].sort_values(["author_id", "view"])
        names = g.author_id.unique()
        gx = g[FACTORS].to_numpy().reshape(-1, 3, len(FACTORS))
        if len(names) != (213 if group == "big5_only" else 63):
            raise ValueError("Incomplete declared source population")
        for q in queries:
            joined = g.merge(
                paths[paths["query"] == q][["author_id", "view", "configuration"]],
                on=["author_id", "view"],
                validate="one_to_one",
            )
            c = joined.configuration.to_numpy().reshape(-1, 3)
            model = fit_operator(gx, c, "between")
            maps[q] = model
            represented = gx @ model["coef"] + model["intercept"]
            residue = c - represented
            observations.extend(
                {
                    "author_id": uid,
                    "group": group,
                    "query": q,
                    "view": v,
                    "configuration": c[i, v],
                    "grain": represented[i, v],
                    "residual": residue[i, v],
                }
                for i, uid in enumerate(names)
                for v in range(3)
            )
    obs = pd.DataFrame(observations)
    rows = []
    errors = []
    for group, part in obs.groupby("group"):
        for level in ("author", "within"):
            if level == "author":
                data = (
                    part.groupby(["author_id", "query"])[
                        ["configuration", "grain", "residual"]
                    ]
                    .mean()
                    .reset_index()
                )
                index = ["author_id"]
            else:
                data = part.copy()
                data[["configuration", "grain", "residual"]] -= data.groupby(
                    ["author_id", "query"]
                )[["configuration", "grain", "residual"]].transform("mean")
                index = ["author_id", "view"]
            tables = {
                c: data.pivot(index=index, columns="query", values=c).sort_index(axis=1)
                for c in ("configuration", "grain", "residual")
            }
            if any(t.isna().any().any() for t in tables.values()):
                raise ValueError("Cannot compare unmatched query rows")
            c, p, r = [
                tables[k].to_numpy() for k in ("configuration", "grain", "residual")
            ]
            centered = [a - a.mean(0) for a in (c, p, r)]
            a, b, d = centered
            error = np.max(
                np.abs(
                    a.T @ a / len(a) - (b.T @ b + d.T @ d + b.T @ d + d.T @ b) / len(a)
                )
            )
            errors.append(float(error))
            queries = list(tables["configuration"].columns)
            for q1, q2 in itertools.combinations(queries, 2):
                row = {
                    "group": group,
                    "level": level,
                    "query1": q1,
                    "query2": q2,
                    "authors": part.author_id.nunique(),
                }
                for field, table in tables.items():
                    if level == "author":
                        stat = paired_summary(table[q1], table[q2], calibrated=False)
                        row.update(
                            {
                                field + "_" + k: v
                                for k, v in stat.items()
                                if k in ("r", "r_lower", "r_upper")
                            }
                        )
                    else:
                        row[field + "_r"] = correlation(table[q1], table[q2])
                rows.append(row)
    obs.to_csv(out / "private_query_configuration_views.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "query_relations.csv", index=False)
    write(out / "query_part_maps.json", maps)
    return {
        "factor_sha256": sha(fp),
        "function_sha256": sha(pp),
        "source_populations": {"big5_only": 213, "mbti_only": 63},
        "number_of_edges": len(rows),
        "covariance_identity_max_error": max(errors),
        "no_between_population_covariance": True,
        "reference_fit_on_source": True,
    }


def evidence_maps(source, out):
    """Compare learned and semantic evidence weights on the same frozen deletions."""
    if read(source / "status.json")["status"] != "COMPLETE":
        raise ValueError("Complete extraction run required")
    with np.load(source / "private_features.npz", allow_pickle=False) as z:
        flat, offsets, poles = z["bags"], z["offsets"], z["poles"]
    meta = pd.read_csv(source / "private_feature_rows.csv")
    lookup = {
        r["author_id"]: r
        for r in read(BASE / "multiquery_sensitivity_s1_r1/private_material.json")
        if r["view"] == 0
    }
    deltas = pd.read_csv(
        BASE / "logic_natural_transport_v1/private_predictions.csv"
    ).set_index(["author_id", "query", "edit"])
    direct = pd.read_csv(source / "private_held_changes.csv")
    records, weights_summary = [], []
    max_error = 0.0
    for qi, q in enumerate(QUERIES):
        model_path = source / f"private_{q}_e5_mean.joblib"
        model = joblib.load(model_path)["input"]
        weights, _bias = model.linear_instance_map()
        unit = poles[qi] / np.linalg.norm(poles[qi], axis=1, keepdims=True)
        contrast = unit[1] - unit[0]
        weights_summary.append(
            {
                "query": q,
                "configuration_to_semantic_cosine": float(
                    weights[0]
                    @ contrast
                    / (np.linalg.norm(weights[0]) * np.linalg.norm(contrast))
                ),
                "residual_to_semantic_cosine": float(
                    weights[1]
                    @ contrast
                    / (np.linalg.norm(weights[1]) * np.linalg.norm(contrast))
                ),
            }
        )
        for i, r in meta[(meta.group == "held") & (meta.edit == "full")].iterrows():
            bag = flat[offsets[i] : offsets[i + 1]]
            psi = bag @ weights.T
            semantic = (
                bag
                / np.maximum(np.linalg.norm(bag, axis=1, keepdims=True), 1e-12)
                @ contrast
            )
            for edit in ("e1", "e2"):
                prior = deltas.loc[(r.author_id, q, edit)]
                for j, estimate in enumerate(("predicted_c", "predicted_r")):
                    account = removal_coordinate(psi[:, j], lookup[r.author_id][edit])
                    actual = direct[
                        (direct.author_id == r.author_id)
                        & (direct["query"] == q)
                        & (direct.arm == "e5_mean")
                        & (direct.edit == edit)
                        & (direct.estimate == estimate)
                    ].delta
                    if len(actual) != 1:
                        raise ValueError("Ambiguous direct prediction counterpart")
                    err = abs(account["delta"] - actual.iloc[0])
                    max_error = max(max_error, err)
                    records.append(
                        dict(
                            author_id=r.author_id,
                            query=q,
                            edit=edit,
                            coordinate=estimate,
                            native_delta=prior.native_delta,
                            **account,
                        )
                    )
                account = removal_coordinate(semantic, lookup[r.author_id][edit])
                records.append(
                    dict(
                        author_id=r.author_id,
                        query=q,
                        edit=edit,
                        coordinate="semantic_pole",
                        native_delta=prior.native_delta,
                        **account,
                    )
                )
    if max_error > 1e-10:
        raise ValueError("Instance-functional deletion does not replay whole program")
    frame = pd.DataFrame(records)
    summary = []
    for (q, edit, coordinate), part in frame.groupby(["query", "edit", "coordinate"]):
        part = part.sort_values("author_id")
        summary.append(
            dict(
                query=q,
                edit=edit,
                coordinate=coordinate,
                **paired_summary(part.native_delta, part.delta, calibrated=False),
            )
        )
    frame.to_csv(out / "private_evidence_mixtures.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "evidence_mixture_metrics.csv", index=False)
    pd.DataFrame(weights_summary).to_csv(
        out / "functional_semantic_directions.csv", index=False
    )
    return {
        "max_removal_replay_error": max_error,
        "extraction_manifest_sha256": sha(source / "feature_manifest.json"),
        "instance_rule": "psi_q(h)=w_q^T h; availability contrast not causal human trait influence",
    }


def matched_view_diagnostics(source, out):
    """Compare source single-view fidelity with held single-view observations."""
    frame = pd.read_csv(source / "private_source_oof.csv")
    rows = []
    for (q, arm), part in frame[frame.view == 0].groupby(["query", "arm"]):
        part = part.sort_values("author_id")
        if part.author_id.duplicated().any():
            raise ValueError("One observation per source author required")
        for estimate, target in (
            ("predicted_c", "actual_c"),
            ("predicted_r", "actual_r"),
            ("derived_r", "actual_r"),
        ):
            rows.append(
                {
                    "query": q,
                    "arm": arm,
                    "estimate": estimate,
                    "scope": "source_author_oof_view0",
                    "comments_per_observation": 12,
                    **paired_summary(part[target], part[estimate]),
                }
            )
    pd.DataFrame(rows).to_csv(out / "source_view0_metrics.csv", index=False)
    return {
        "source_oof_sha256": sha(source / "private_source_oof.csv"),
        "source_unit": "one view of twelve comments per held-out source author",
        "source_author_mean_is_different_unit": True,
    }


def residual_operator_account(source, out):
    """Decompose direct and post-subtracted residuals in the same fixed program."""
    data = pd.read_csv(source / "private_held_predictions.csv")
    native = pd.read_csv(BASE / "logic_natural_transport_v1/private_predictions.csv")
    rows, private = [], []
    errors = []
    for (q, arm), group in data.groupby(["query", "arm"]):
        full = group[group.edit == "full"].set_index("author_id").sort_index()
        for edit in ("e1", "e2"):
            changed = group[group.edit == edit].set_index("author_id").loc[full.index]
            delta = (
                changed[["predicted_c", "predicted_r", "derived_r", "grain_part"]]
                - full[["predicted_c", "predicted_r", "derived_r", "grain_part"]]
            )
            proxy = delta.predicted_c - delta.predicted_r
            gap = delta.grain_part - proxy
            error = np.max(np.abs(delta.predicted_r - delta.derived_r - gap))
            errors.append(float(error))
            endpoint = (
                native[(native["query"] == q) & (native.edit == edit)]
                .set_index("author_id")
                .loc[full.index]
                .native_delta
            )
            for name, values in (
                ("grain_actual", delta.grain_part),
                ("grain_proxy", proxy),
                ("grain_transport_gap", gap),
            ):
                rows.append(
                    {
                        "query": q,
                        "arm": arm,
                        "edit": edit,
                        "channel": name,
                        "rms": float(np.sqrt(np.mean(values**2))),
                        **paired_summary(endpoint, values, calibrated=False),
                    }
                )
            private.extend(
                {
                    "author_id": uid,
                    "query": q,
                    "arm": arm,
                    "edit": edit,
                    "grain_proxy": proxy.loc[uid],
                    "grain_transport_gap": gap.loc[uid],
                }
                for uid in full.index
            )
    if max(errors) > 1e-10:
        raise ValueError("Residual subtraction identity failed")
    pd.DataFrame(rows).to_csv(out / "residual_operator_metrics.csv", index=False)
    pd.DataFrame(private).to_csv(
        out / "private_residual_operator_values.csv", index=False
    )
    return {"max_subtraction_identity_error": max(errors), "additional_model_fits": 0}


def main():
    """Run two bounded analyses without retraining the fixed input programs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=BASE / "logic_configuration_extraction_v1"
    )
    p.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_configuration_relations_v1"
    )
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private output required")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "status.json", {"status": "RUNNING"})
    try:
        with threadpool_limits(limits=2):
            q = query_relations(out)
            e = evidence_maps(args.source.resolve(), out)
            v = matched_view_diagnostics(args.source.resolve(), out)
            r = residual_operator_account(args.source.resolve(), out)
        write(
            out / "manifest.json",
            {
                "query_relations": q,
                "evidence_maps": e,
                "matched_view_diagnostics": v,
                "residual_operator_account": r,
                "script_sha256": sha(__file__),
                "new_neural_calls": 0,
                "human_labels_read": False,
            },
        )
        write(out / "status.json", {"status": "COMPLETE"})
        print(
            pd.read_csv(out / "evidence_mixture_metrics.csv").to_string(index=False),
            flush=True,
        )
    except Exception as exc:
        write(
            out / "status.json", {"status": "FAILED", "error_type": type(exc).__name__}
        )
        raise


if __name__ == "__main__":
    main()
