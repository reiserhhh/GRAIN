#!/usr/bin/env python3
"""M22: bounded cached teacher-readout comparisons, then source-label accounting."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.nonlinear_readout import (
    METHODS,
    fit_readers,
    grouped_folds,
    paired_mse_interval,
    select_readers,
)
from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    covariance_split,
    feature_cache,
    key,
    paired_intervals,
    read,
    scalar_metrics,
    sha,
    teacher_arrays,
    write,
)

CONNECTION = BASE / "response_student_source_connection_r2"
PANEL = BASE / "multiquery_sensitivity_s1_r1"
PREFIX = BASE / "prefix_source_connection_v1"
PLAN = ROOT / "docs/GRAIN_LINEAR_NONLINEAR_READOUT_PLAN_V1.md"
QUERIES = ("TF_cont", "neuroticism")


def prefix_features(ids):
    """Read hash-verified query states; mean pools deliberately keep the probe small."""
    if read(PREFIX / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("M10 query-state collection is incomplete")
    material = BASE / "source_remainder_s9_r1/private_material.json"
    if sha(material) != read(PREFIX / "manifest.json")["source_material_sha256"]:
        raise ValueError("M10 membership changed")
    expected = {(r["author_id"], r["query"]) for r in read(material)}
    source = PREFIX / "private_scores_before_outcomes.csv"
    frozen = read(PREFIX / "scores_frozen.json")
    if sha(source) not in frozen.values():
        raise ValueError("M10 scoring target hash changed")
    scores = pd.read_csv(
        source, dtype={"author_id": str}, usecols=["author_id", "query", "native", "PQ"]
    )
    if scores.duplicated(["author_id", "query"]).any():
        raise ValueError("Duplicate M10 author/query")
    if set(zip(scores["author_id"], scores["query"], strict=True)) != expected:
        raise ValueError("M10 target membership mismatch")
    vectors, provenance = {}, {}
    for path in sorted((PREFIX / "private_packets").glob("author_*.json")):
        record = read(path)
        array_path = path.with_suffix(".npz")
        if sha(array_path) != record["arrays_sha256"]:
            raise ValueError("M10 vector cache changed")
        pair = (record["author_id"], record["query"])
        if pair not in expected or pair in vectors or pair[0] not in set(ids):
            raise ValueError("Unexpected, duplicate or unmatched prefix author")
        with np.load(array_path, allow_pickle=False) as packet:
            p, q = packet["P"], packet["Q"]
            if p.ndim != 3 or q.ndim != 2 or len(p) != 3 or len(q) != 3:
                raise ValueError("Require complete three-view query packets")
            vector = np.concatenate((p.mean((0, 1)), q.mean(0)))
            if not np.isfinite(vector).all():
                raise ValueError("Invalid prefix vector")
            vectors[pair] = vector
        provenance[str(array_path.relative_to(ROOT))] = record["arrays_sha256"]
    if set(vectors) != expected:
        raise ValueError("Missing prefix authors")
    return scores, vectors, provenance


def cases(ids, features, targets, prefix_scores, prefix_vectors):
    """Construct same-material comparisons; never substitute old for new teacher scores."""
    for query in QUERIES:
        for family in ("mean", "query"):
            names, x, y = teacher_arrays(ids, features[query][family], targets, query)
            yield "source_all", family, query, names, x, y, None
        scores = prefix_scores[prefix_scores["query"] == query].set_index("author_id")
        names = np.asarray([uid for uid in ids if uid in scores.index])
        if len(names) != len(scores):
            raise ValueError("Missing matched author")
        lookup = {uid: i for i, uid in enumerate(ids)}
        indices = [lookup[uid] for uid in names]
        y = scores.loc[names, "native"].to_numpy()
        fixed = scores.loc[names, "PQ"].to_numpy()
        for family in ("mean", "query", "prefix_pq_mean"):
            x = (
                np.asarray([prefix_vectors[uid, query] for uid in names])
                if family == "prefix_pq_mean"
                else features[query][family][indices]
            )
            yield "prefix_matched", family, query, names, x, y, fixed


def summarize(out, frame, outcomes, draws=1000):
    """Join external outcomes only to frozen OOF scores; report every prespecified arm."""
    frame = frame.merge(outcomes, on=["author_id", "query"], validate="many_to_one")
    if frame.y_true.isna().any():
        raise ValueError("Missing source outcomes")
    metrics, comparisons, covariances = [], [], []
    group_cols = ["cohort", "family", "query", "control"]
    for keys, group in frame.groupby(group_cols):
        meta = dict(zip(group_cols, keys, strict=True))
        pivot = group.pivot(index="author_id", columns="method", values="prediction")
        base = group[group.method == "ridge"].set_index("author_id").loc[pivot.index]
        for method in pivot.columns:
            p = pivot[method].to_numpy()
            y, teacher = base.y_true.to_numpy(), base.teacher.to_numpy()
            if not np.isfinite(p).all():
                raise ValueError("Incomplete OOF method")
            fidelity = scalar_metrics(teacher, p)
            external = scalar_metrics(y, p, meta["query"] == "TF_cont")
            metrics.append(
                {
                    **meta,
                    "method": method,
                    "authors": len(p),
                    "teacher_r": fidelity["pearson_r"],
                    "teacher_mse": float(np.mean((teacher - p) ** 2)),
                    "external_r": external["pearson_r"],
                    "external_auc": external.get("auc"),
                    "prediction_sd": float(np.std(p)),
                }
            )
            covariances.append(
                {**meta, "method": method, **covariance_split(y, teacher, p)}
            )
            controls = ["ridge"]
            if method == "trees":
                controls.append("stumps")
            if method == "ridge_trees":
                controls.append("ridge_stumps")
            for control in controls:
                if method == control:
                    continue
                b = pivot[control].to_numpy()
                interval = paired_intervals(y, p, b, meta["query"] == "TF_cont", draws)
                fold_gains = []
                for fold in range(5):
                    mask = base.fold.to_numpy() == fold
                    fold_gains.append(
                        float(
                            np.mean(
                                (teacher[mask] - b[mask]) ** 2
                                - (teacher[mask] - p[mask]) ** 2
                            )
                        )
                    )
                comparisons.append(
                    {
                        **meta,
                        "method": method,
                        "baseline": control,
                        **paired_mse_interval(teacher, p, b, draws),
                        "positive_mse_folds": int(np.sum(np.asarray(fold_gains) > 0)),
                        "external_delta": interval["delta"],
                        "external_delta_lower": interval["delta_lower"],
                        "external_delta_upper": interval["delta_upper"],
                    }
                )
    for name, rows in (
        ("metrics", metrics),
        ("paired_comparisons", comparisons),
        ("covariance_account", covariances),
    ):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    frame.to_csv(out / "private_outcome_connection.csv", index=False)
    return metrics, comparisons


def main():
    """Run one finite cached probe; preserve partial work on failures/time boundaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-fit-seconds", type=float, default=1200)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--skip-prefix", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE)
        or args.max_fit_seconds <= 0
        or args.bootstrap_draws < 1
    ):
        raise ValueError("Use a private results path and positive budgets")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_models").mkdir()
    start = time.monotonic()
    source = CONNECTION / "private_predictions.csv"
    manifest = {
        "protocol": "GRAIN_LINEAR_NONLINEAR_M22_V1",
        "date": "2026-09-08",
        "source_sha256": sha(source),
        "plan_sha256": sha(PLAN),
        "script_sha256": sha(__file__),
        "module_sha256": sha(
            ROOT / "grain_research/vector_mechanisms/nonlinear_readout.py"
        ),
        "dependency_versions": {
            v: importlib.metadata.version(v)
            for v in ("numpy", "scikit-learn", "lightgbm", "scipy")
        },
        "cache_module_sha256": sha(
            ROOT / "grain_research/vector_mechanisms/readout_cache_cpu.py"
        ),
        "methods": METHODS,
        "rank": 64,
        "alphas": [1, 10, 100],
        "trees": [100, 300],
        "selection": "inner3 teacher MSE only; outer5 s4_outer",
        "fit_target": "cached teacher, never questionnaire",
        "neural_calls": 0,
        "bridge_meps_lockbox_read": False,
        "max_fit_seconds": args.max_fit_seconds,
        "bootstrap_draws": args.bootstrap_draws,
        "scope": "exposed discovery authors, paired OOF intervals conditional on fits",
    }
    write(out / "manifest.json", manifest)
    rows, selections = [], []
    try:
        with threadpool_limits(limits=2):
            ids, features = feature_cache(CONNECTION, PANEL)
            targets = pd.read_csv(
                source,
                dtype={"author_id": str},
                usecols=["author_id", "query", "native_query"],
            )
            if args.skip_prefix:
                prefix_scores = pd.DataFrame(
                    columns=["author_id", "query", "native", "PQ"]
                )
                prefix_vectors, hashes = {}, {}
            else:
                prefix_scores, prefix_vectors, hashes = prefix_features(ids)
            write(out / "private_prefix_hashes.json", hashes)
            write(
                out / "input_hashes.json",
                {
                    "e5_cache_index": sha(CONNECTION / "private_cache_hashes.json"),
                    "query_anchor": sha(PANEL / "private_encoder.npz"),
                    "prefix_manifest": sha(PREFIX / "manifest.json")
                    if not args.skip_prefix
                    else None,
                    "prefix_targets": sha(PREFIX / "private_scores_before_outcomes.csv")
                    if not args.skip_prefix
                    else None,
                },
            )
            fit_start = time.monotonic()
            for cohort, family, query, names, x, teacher, fixed in cases(
                ids, features, targets, prefix_scores, prefix_vectors
            ):
                if not len(names):
                    continue
                if len(set(names)) != len(names):
                    raise ValueError("Author aggregation is not one row per author")
                outer = grouped_folds(names)
                controls = (
                    ("observed", "permuted")
                    if cohort == "source_all" and family == "mean"
                    else ("observed",)
                )
                for control in controls:
                    y = teacher.copy()
                    if control == "permuted":
                        y = np.random.default_rng(731).permutation(y)
                    for fold in range(5):
                        train = np.array(
                            sorted(
                                np.flatnonzero(outer != fold),
                                key=lambda j: key(names[j], f"s4_{fold}"),
                            )
                        )
                        test = np.flatnonzero(outer == fold)
                        specs = select_readers(
                            x[train], y[train], names[train], f"s4_inner_{fold}"
                        )
                        readers = fit_readers(x[train], y[train], specs)
                        chosen = min(
                            specs,
                            key=lambda m: (specs[m]["inner_mse"], METHODS.index(m)),
                        )
                        predictions = {}
                        for method, reader in readers.items():
                            a, b = reader.parts(x[test])
                            predictions[method] = (a + b, a, b)
                            selections.append(
                                {
                                    "cohort": cohort,
                                    "family": family,
                                    "query": query,
                                    "control": control,
                                    "fold": fold,
                                    "method": method,
                                    **specs[method],
                                    "chosen": chosen,
                                }
                            )
                        predictions["inner_chosen"] = predictions[chosen]
                        predictions["teacher_mean"] = (
                            np.full(len(test), y[train].mean()),
                            np.zeros(len(test)),
                            np.zeros(len(test)),
                        )
                        if fixed is not None and control == "observed":
                            predictions["frozen_reference_PQ"] = (
                                fixed[test],
                                np.zeros(len(test)),
                                np.zeros(len(test)),
                            )
                        joblib.dump(
                            {
                                "readers": readers,
                                "specs": specs,
                                "chosen": chosen,
                                "train_authors": names[train],
                                "test_authors": names[test],
                            },
                            out
                            / "private_models"
                            / f"{cohort}_{family}_{query}_{control}_{fold}.joblib",
                            compress=3,
                        )
                        for method, (p, a, b) in predictions.items():
                            rows.extend(
                                {
                                    "cohort": cohort,
                                    "family": family,
                                    "query": query,
                                    "control": control,
                                    "fold": fold,
                                    "method": method,
                                    "author_id": names[j],
                                    "teacher": float(y[j]),
                                    "original_teacher": float(teacher[j]),
                                    "prediction": float(pred),
                                    "linear_part": float(lin),
                                    "nonlinear_part": float(nonlin),
                                }
                                for j, pred, lin, nonlin in zip(
                                    test, p, a, b, strict=True
                                )
                            )
                        pd.DataFrame(rows).to_csv(
                            out / "private_partial.csv", index=False
                        )
                        write(
                            out / "run_status.json",
                            {
                                "status": "RUNNING",
                                "cohort": cohort,
                                "family": family,
                                "query": query,
                                "control": control,
                                "fold": fold,
                                "fit_seconds": time.monotonic() - fit_start,
                                "neural_calls": 0,
                            },
                        )
                        print(
                            f"M22 {cohort}/{family}/{query}/{control} fold {fold + 1}/5: {chosen}",
                            flush=True,
                        )
                        if time.monotonic() - fit_start > args.max_fit_seconds:
                            raise TimeoutError(
                                "M22 fit budget reached; partial artifacts retained"
                            )
        frame = pd.DataFrame(rows)
        if frame.duplicated(
            ["cohort", "family", "query", "control", "method", "author_id"]
        ).any():
            raise ValueError("Duplicate OOF prediction")
        frozen = out / "private_oof_before_outcomes.csv"
        frame.to_csv(frozen, index=False)
        write(
            out / "oof_frozen.json", {"sha256": sha(frozen), "outcome_selection": False}
        )
        pd.DataFrame(selections).to_csv(out / "selected_models.csv", index=False)
        if sha(source) != manifest["source_sha256"]:
            raise ValueError("Outcomes changed during fit")
        outcomes = pd.read_csv(
            source, dtype={"author_id": str}, usecols=["author_id", "query", "y_true"]
        )
        metrics, comparisons = summarize(out, frame, outcomes, args.bootstrap_draws)
        write(
            out / "summary.json",
            {
                "status": "COMPLETE",
                "seconds": time.monotonic() - start,
                "new_neural_calls": 0,
                "questionnaire_fit_or_selection": False,
                "oof_sha256": sha(frozen),
                "metrics": pd.DataFrame(metrics)
                .astype(object)
                .where(pd.notna(pd.DataFrame(metrics)), None)
                .to_dict("records"),
                "comparison_count": len(comparisons),
            },
        )
        write(
            out / "run_status.json",
            {
                "status": "COMPLETE",
                "seconds": time.monotonic() - start,
                "neural_calls": 0,
            },
        )
        table = pd.DataFrame(metrics)
        (out / "report.md").write_text(
            "# M22 cached nonlinear readout\n\nAll results are teacher-selected author OOF; outcomes joined afterwards.\nIntervals condition on these fitted readers; no new neural calls.\n\n"
            + table.to_markdown(index=False, floatfmt=".4f")
            + "\n",
            encoding="utf-8",
        )
        print(table.to_string(index=False), flush=True)
    except BaseException as exc:
        if rows:
            pd.DataFrame(rows).to_csv(out / "private_partial.csv", index=False)
        write(
            out / "run_status.json",
            {
                "status": "PAUSED_BUDGET"
                if isinstance(exc, TimeoutError)
                else "FAILED",
                "error_type": type(exc).__name__,
                "seconds": time.monotonic() - start,
            },
        )
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
