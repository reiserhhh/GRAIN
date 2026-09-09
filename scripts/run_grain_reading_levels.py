#!/usr/bin/env python3
"""Cache-only nine-query reading levels, source-OOF and real-material changes."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.natural_transport import (
    correlation,
    fit_operator,
    paired_summary,
)
from grain_research.vector_mechanisms.reading_levels import (
    affine_error_account,
    fit_readers,
    within_change,
    within_predictions,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_conditional_edits import load_panel
from scripts.run_grain_natural_transport import FACTORS
from scripts.run_grain_natural_transport import SOURCE as M29

PLAN = ROOT / "docs/GRAIN_LOGIC_READING_LEVEL_PLAN_V1.md"
SOURCE = BASE / "logic_configuration_extraction_v1"
QUERIES = (*AXES, *BIG5)
OLD_QUERIES = ("TF_cont", "neuroticism")


def load_inputs():
    """Reuse exact numeric bags; old-query replay later verifies both populations."""
    if read(SOURCE / "status.json")["status"] != "COMPLETE":
        raise ValueError("Complete input cache required")
    meta = pd.read_csv(SOURCE / "private_feature_rows.csv")
    with np.load(SOURCE / "private_features.npz", allow_pickle=False) as z:
        flat, offsets = z["bags"], z["offsets"]
    bags = [flat[offsets[i] : offsets[i + 1]] for i in range(len(meta))]
    paths = pd.read_csv(BASE / "dynamic_path_channels_v1/private_path_views.csv")
    if len(meta) != 864 or meta[meta.group != "held"].author_id.nunique() != 276:
        raise ValueError("Incorrect source/held feature panel")
    return meta, bags, paths


def construct(out, meta, bags, paths):
    """Fit model targets only, with identical five-fold author partitions."""
    rows, changes, levels, geometry, losses = [], [], [], [], []
    for q in QUERIES:
        group = "mbti_only" if q in AXES else "big5_only"
        data = meta[meta.group == group].sort_values(["author_id", "view"]).copy()
        data["feature_index"] = data.index
        data = data.merge(
            paths[paths["query"] == q][["author_id", "view", "configuration"]],
            on=["author_id", "view"],
            validate="one_to_one",
        )
        ids = data.author_id.unique()
        if (
            len(ids) != (63 if q in AXES else 213)
            or data.groupby("author_id").size().ne(3).any()
        ):
            raise ValueError("Complete three-view author panel required")
        xg = data[FACTORS].to_numpy().reshape(-1, 3, len(FACTORS))
        yc = data.configuration.to_numpy().reshape(-1, 3)
        splits = list(KFold(5, shuffle=True, random_state=20260910).split(ids))
        splits.append((np.arange(len(ids)), np.array([], int)))
        for fold, (train, test) in enumerate(splits):
            tr = np.concatenate([np.arange(i * 3, (i + 1) * 3) for i in train])
            training = data.iloc[tr]
            gm = fit_operator(xg[train], yc[train], "between")
            grain = training[FACTORS].to_numpy() @ gm["coef"] + gm["intercept"]
            y = np.column_stack(
                [training.configuration, training.configuration - grain]
            )
            ix = training.feature_index.to_numpy(int)
            readers = fit_readers([bags[i] for i in ix], y)
            losses.append({"query": q, "fold": fold, **readers["loss_account"]})
            slopes = {name: readers[name].head.coef_ for name in ("pooled", "between")}
            slopes["within"] = readers["within_coef"]
            for a, b in itertools.combinations(slopes, 2):
                for k, target in enumerate(("c", "r")):
                    v, w = slopes[a][k], slopes[b][k]
                    geometry.append(
                        {
                            "query": q,
                            "fold": fold,
                            "target": target,
                            "a": a,
                            "b": b,
                            "cosine": float(
                                v @ w / (np.linalg.norm(v) * np.linalg.norm(w))
                            ),
                        }
                    )
            if fold < 5:
                te = np.concatenate([np.arange(i * 3, (i + 1) * 3) for i in test])
                testing = data.iloc[te]
                ti = testing.feature_index.to_numpy(int)
                test_bags = [bags[i] for i in ti]
                tg = testing[FACTORS].to_numpy() @ gm["coef"] + gm["intercept"]
                truth = np.column_stack(
                    [testing.configuration, testing.configuration - tg]
                )
                for arm in ("pooled", "between", "within"):
                    prediction = (
                        within_predictions(readers, test_bags)
                        if arm == "within"
                        else readers[arm].predict(test_bags, None)
                    )
                    observed = truth.reshape(-1, 3, 2)
                    if arm == "within":
                        observed = observed - observed.mean(1, keepdims=True)
                    observed = observed.reshape(-1, 2)
                    rows.extend(
                        {
                            "author_id": r.author_id,
                            "view": int(r.view),
                            "query": q,
                            "arm": arm,
                            "fold": fold,
                            "actual_c": observed[i, 0],
                            "actual_r": observed[i, 1],
                            "predicted_c": prediction[i, 0],
                            "predicted_r": prediction[i, 1],
                        }
                        for i, (_, r) in enumerate(testing.iterrows())
                    )
            else:
                joblib.dump(
                    {
                        "query": q,
                        "readers": readers,
                        "grain": gm,
                        "within_level_defined": False,
                    },
                    out / f"private_{q}_readers.joblib",
                )
                if q not in OLD_QUERIES:
                    continue
                h = meta[meta.group == "held"].copy()
                h["feature_index"] = h.index
                full = h[h.edit == "full"].sort_values("author_id")
                fb = [bags[i] for i in full.feature_index]
                for arm in ("pooled", "between"):
                    p = readers[arm].predict(fb, None)
                    levels.extend(
                        {
                            "author_id": r.author_id,
                            "query": q,
                            "arm": arm,
                            "predicted_c": p[i, 0],
                            "predicted_r": p[i, 1],
                            "grain_part": float(
                                r[FACTORS].to_numpy(float) @ gm["coef"]
                                + gm["intercept"]
                            ),
                        }
                        for i, (_, r) in enumerate(full.iterrows())
                    )
                for edit in ("e1", "e2"):
                    edited = (
                        h[h.edit == edit].set_index("author_id").loc[full.author_id]
                    )
                    eb = [bags[i] for i in edited.feature_index]
                    dg = (edited[FACTORS].to_numpy() - full[FACTORS].to_numpy()) @ gm[
                        "coef"
                    ]
                    for arm in ("pooled", "between", "within"):
                        d = (
                            within_change(readers, fb, eb)
                            if arm == "within"
                            else readers[arm].predict(eb, None)
                            - readers[arm].predict(fb, None)
                        )
                        changes.extend(
                            {
                                "author_id": uid,
                                "query": q,
                                "arm": arm,
                                "edit": edit,
                                "predicted_c_delta": d[i, 0],
                                "predicted_r_delta": d[i, 1],
                                "grain_delta": dg[i],
                            }
                            for i, uid in enumerate(full.author_id)
                        )
        print(f"Completed source-only readers: {q}", flush=True)
    table = pd.DataFrame(rows)
    old = pd.read_csv(SOURCE / "private_source_oof.csv").query("arm=='e5_mean'")
    match = table[(table.arm == "pooled") & table["query"].isin(OLD_QUERIES)].merge(
        old,
        on=["author_id", "view", "query", "fold"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    errors = [
        float((match[k + "_new"] - match[k + "_old"]).abs().max())
        for k in ("actual_c", "actual_r", "predicted_c", "predicted_r")
    ]
    if len(match) != len(old) or max(errors) > 1e-10:
        raise ValueError("Prior pooled programs were not reproduced")
    table.to_csv(out / "private_source_oof.csv", index=False)
    pd.DataFrame(changes).to_csv(out / "private_held_changes.csv", index=False)
    pd.DataFrame(levels).to_csv(out / "private_held_levels.csv", index=False)
    pd.DataFrame(geometry).to_csv(out / "reader_geometry.csv", index=False)
    write(out / "loss_accounts.json", losses)
    write(
        out / "prediction_freeze.json",
        {
            "files": {
                name: sha(out / name)
                for name in (
                    "private_source_oof.csv",
                    "private_held_changes.csv",
                    "private_held_levels.csv",
                )
            },
            "old_pooled_rows": len(match),
            "old_pooled_max_error": max(errors),
            "human_labels_read": False,
            "max_loss_identity_error": max(r["max_error"] for r in losses),
        },
    )


def native_targets():
    """Restore exact two-query natural-edit targets from existing P/Q caches."""
    native, states, _, _ = load_panel()
    sm = read(M29 / "manifest.json")
    result = []
    with np.load(
        Path(sm["functions"]) / "nine_query_functions.npz", allow_pickle=False
    ) as z:
        funcs = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in OLD_QUERIES
        }
    for i, r in native.iterrows():
        f = funcs[r["query"]]
        full = f.parts(states[i]["full_P"], states[i]["full_Q"])["configuration"]
        changed = f.parts(states[i]["edited_P"], states[i]["edited_Q"])["configuration"]
        result.append(
            {
                "author_id": r.author_id,
                "query": r["query"],
                "c_full": full,
                "c_delta": changed - full,
            }
        )
    return pd.DataFrame(result).set_index(["author_id", "query"])


def evaluate(out):
    """Read native endpoints and existing criteria only after predictions freeze."""
    freeze = read(out / "prediction_freeze.json")
    for name, expected in freeze["files"].items():
        if sha(out / name) != expected:
            raise ValueError("Predictions changed")
    oof = pd.read_csv(out / "private_source_oof.csv")
    criteria = pd.read_csv(M29 / "private_external_links.csv").set_index(
        ["author_id", "query"]
    )
    if criteria.index.duplicated().any():
        raise ValueError("Nonunique criterion")
    metrics, links, accounts = [], [], []
    for (q, arm), part in oof.groupby(["query", "arm"]):
        part = part.sort_values(["author_id", "view"])
        mean = part.groupby(["author_id", "query"])[
            ["actual_c", "actual_r", "predicted_c", "predicted_r", "fold"]
        ].mean()
        y = criteria.loc[mean.index].y_true.to_numpy()
        if not np.isfinite(y).all():
            raise ValueError("Incomplete criterion join")
        for t in ("c", "r"):
            actual, predicted = "actual_" + t, "predicted_" + t
            if arm != "within":
                for level, group in (
                    ("author_mean", mean),
                    ("view0", part[part.view == 0]),
                ):
                    metrics.append(
                        {
                            "query": q,
                            "arm": arm,
                            "level": level,
                            "target": t,
                            **paired_summary(group[actual], group[predicted]),
                        }
                    )
                links.append(
                    {
                        "query": q,
                        "arm": arm,
                        "target": t,
                        **association(mean[predicted], y, binary=q in AXES),
                    }
                )
                for level in ("pooled_oof", "within_cv_fold"):
                    v = mean[[actual, predicted]].to_numpy()
                    yy = y.copy()
                    if level == "within_cv_fold":
                        for fold in mean.fold.unique():
                            mask = mean.fold.to_numpy() == fold
                            v[mask] -= v[mask].mean(0)
                            yy[mask] -= yy[mask].mean()
                    accounts.append(
                        {
                            "query": q,
                            "arm": arm,
                            "target": t,
                            "level": level,
                            **affine_error_account(v[:, 0], v[:, 1], yy),
                        }
                    )
            a = part[actual] - part.groupby("author_id")[actual].transform("mean")
            p = part[predicted] - part.groupby("author_id")[predicted].transform("mean")
            metrics.append(
                {
                    "query": q,
                    "arm": arm,
                    "level": "within",
                    "target": t,
                    "n": part.author_id.nunique(),
                    "r": correlation(a, p),
                }
            )
    pd.DataFrame(metrics).to_csv(out / "source_fidelity.csv", index=False)
    cl = pd.DataFrame(links)
    cl["q_bh"] = bh_adjust(cl.p)
    cl.to_csv(out / "criterion_links.csv", index=False)
    pd.DataFrame(accounts).to_csv(out / "affine_error_accounting.csv", index=False)
    targets = native_targets()
    native = pd.read_csv(
        BASE / "logic_natural_transport_v1/private_predictions.csv"
    ).set_index(["author_id", "query", "edit"])
    dm, lm = [], []
    for (q, arm, edit), part in pd.read_csv(out / "private_held_changes.csv").groupby(
        ["query", "arm", "edit"]
    ):
        part = part.sort_values("author_id")
        n = native.xs((q, edit), level=("query", "edit")).loc[part.author_id]
        target = targets.xs(q, level="query").loc[part.author_id]
        for t in ("c", "r"):
            p = part[f"predicted_{t}_delta"]
            dm.append(
                {
                    "query": q,
                    "arm": arm,
                    "edit": edit,
                    "estimate": t,
                    "target": "native_delta",
                    **paired_summary(n.native_delta, p, calibrated=False),
                }
            )
            if edit == "e1":
                actual = target.c_delta.to_numpy() - (
                    part.grain_delta.to_numpy() if t == "r" else 0
                )
                dm.append(
                    {
                        "query": q,
                        "arm": arm,
                        "edit": edit,
                        "estimate": t,
                        "target": "component_delta",
                        **paired_summary(actual, p),
                    }
                )
    for (q, arm), part in pd.read_csv(out / "private_held_levels.csv").groupby(
        ["query", "arm"]
    ):
        part = part.sort_values("author_id")
        t = targets.xs(q, level="query").loc[part.author_id]
        for term in ("c", "r"):
            actual = t.c_full.to_numpy() - (
                part.grain_part.to_numpy() if term == "r" else 0
            )
            lm.append(
                {
                    "query": q,
                    "arm": arm,
                    "target": term,
                    **paired_summary(actual, part["predicted_" + term]),
                }
            )
    pd.DataFrame(dm).to_csv(out / "held_change_metrics.csv", index=False)
    pd.DataFrame(lm).to_csv(out / "held_level_metrics.csv", index=False)
    print(
        cl[["query", "arm", "target", "pearson_r", "auc"]].to_string(index=False),
        flush=True,
    )


def main():
    """Run a bounded author-level reading experiment; preserve every packet."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=BASE / "logic_reading_levels_v1")
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private output directory required")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write(out / "status.json", {"status": "RUNNING", "stage": "fit"})
    try:
        with threadpool_limits(limits=2):
            construct(out, *load_inputs())
            write(out / "status.json", {"status": "RUNNING", "stage": "evaluate"})
            evaluate(out)
        write(
            out / "manifest.json",
            {
                "plan_sha256": sha(PLAN),
                "script_sha256": sha(__file__),
                "core_sha256": sha(
                    ROOT / "grain_research/vector_mechanisms/reading_levels.py"
                ),
                "source_manifest_sha256": sha(SOURCE / "feature_manifest.json"),
                "criterion_sha256": sha(M29 / "private_external_links.csv"),
                "new_neural_calls": 0,
                "new_criterion_fits": 0,
                "queries": list(QUERIES),
                "views": 3,
            },
        )
        write(
            out / "status.json",
            {"status": "COMPLETE", "seconds": time.monotonic() - start},
        )
    except Exception as exc:
        write(
            out / "status.json", {"status": "FAILED", "error_type": type(exc).__name__}
        )
        raise


if __name__ == "__main__":
    main()
