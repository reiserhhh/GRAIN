#!/usr/bin/env python3
"""M23: localize query-state readability using a fixed linear reader family."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_coordinates import (
    FAMILIES,
    CoordinateReader,
)
from grain_research.vector_mechanisms.nonlinear_readout import (
    grouped_folds,
    paired_mse_interval,
)
from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    corr,
    covariance_split,
    key,
    paired_intervals,
    read,
    scalar_metrics,
    sha,
    write,
)
from scripts.run_grain_nonlinear_readout import PREFIX, prefix_features

M22 = BASE / "nonlinear_readout_m22_v4"
SOURCE = BASE / "response_student_source_connection_r2/private_predictions.csv"
PLAN = ROOT / "docs/GRAIN_MAINLINE_PIECES_GOAL_V4.md"


def load_inputs():
    """Verify the M22/M10 chain; read no external-label column during construction."""
    oof_path = M22 / "private_oof_before_outcomes.csv"
    if sha(oof_path) != read(M22 / "oof_frozen.json")["sha256"]:
        raise ValueError("M22 OOF changed")
    old = pd.read_csv(oof_path, dtype={"author_id": str})
    old = old[
        (old.cohort == "prefix_matched")
        & (old.family == "prefix_pq_mean")
        & (old.method == "ridge")
        & (old.control == "observed")
    ]
    if old.duplicated(["author_id", "query"]).any():
        raise ValueError("Duplicate M22 source")
    ids = old.author_id.to_numpy()
    scores, native_means, hashes = prefix_features(ids)
    packets = {}
    for path in sorted((PREFIX / "private_packets").glob("author_*.json")):
        r = read(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            packets[r["author_id"], r["query"]] = (
                z["P"].astype(float),
                z["Q"].astype(float),
            )
    selected = pd.read_csv(M22 / "selected_models.csv")
    selected = selected[
        (selected.cohort == "prefix_matched")
        & (selected.family == "prefix_pq_mean")
        & (selected.method == "ridge")
        & (selected.control == "observed")
    ]
    if len(selected) != 10 or selected.duplicated(["query", "fold"]).any():
        raise ValueError("Expected ten frozen fold alphas")
    alpha = {
        (r["query"], int(r["fold"])): float(r["alpha"])
        for r in selected.to_dict("records")
    }
    return old, scores, packets, native_means, hashes, alpha


def analyze(out, frame, views, contributions, draws):
    """Join previously exposed outcomes only after predictions have been frozen."""
    outcomes = pd.read_csv(
        SOURCE, dtype={"author_id": str}, usecols=["author_id", "query", "y_true"]
    )
    joined = frame.merge(outcomes, on=["author_id", "query"], validate="many_to_one")
    if joined.y_true.isna().any():
        raise ValueError("Missing source outcome")
    metrics, comparisons, accounts, reliability = [], [], [], []
    for (query, family), g in joined.groupby(["query", "family"]):
        y, p, t = g.y_true.to_numpy(), g.prediction.to_numpy(), g.teacher.to_numpy()
        external = scalar_metrics(y, p, query == "TF_cont")
        metrics.append(
            {
                "query": query,
                "family": family,
                "authors": len(g),
                "teacher_r": corr(t, p),
                "teacher_mse": float(np.mean((t - p) ** 2)),
                "external_r": external["pearson_r"],
                "external_auc": external["auc"],
                "score_sd": float(np.std(p)),
            }
        )
        accounts.append({"query": query, "family": family, **covariance_split(y, t, p)})
        base = joined[(joined["query"] == query) & (joined.family == "p_mean_q")]
        base = base.set_index("author_id").loc[g.author_id]
        comparisons.append(
            {
                "query": query,
                "family": family,
                "baseline": "p_mean_q",
                **paired_mse_interval(t, p, base.prediction.to_numpy(), draws),
                **paired_intervals(
                    y, p, base.prediction.to_numpy(), query == "TF_cont", draws
                ),
            }
        )
        v = views[(views["query"] == query) & (views.family == family)]
        v = v.pivot(index="author_id", columns="view", values="prediction")
        for a, b in ((0, 1), (0, 2), (1, 2)):
            reliability.append(
                {
                    "query": query,
                    "family": family,
                    "view_a": a,
                    "view_b": b,
                    "authors": len(v),
                    "view_score_r": corr(v[a], v[b]),
                }
            )
    contrib = contributions.merge(
        outcomes, on=["author_id", "query"], validate="many_to_one"
    )
    ledger = []
    cov = lambda a, b: float(np.mean((a - np.mean(a)) * (b - np.mean(b))))
    for query, g in contrib.groupby("query"):
        for part in ("reference", "p_mean", "p_config", "q"):
            a = g[part].to_numpy()
            ledger.append(
                {
                    "query": query,
                    "part": part,
                    "authors": len(g),
                    "rms": float(np.sqrt(np.mean(a * a))),
                    "teacher_covariance": cov(g.teacher.to_numpy(), a),
                    "source_covariance": cov(g.y_true.to_numpy(), a),
                    "source_r": corr(g.y_true.to_numpy(), a),
                }
            )
    for name, data in (
        ("metrics", metrics),
        ("paired_comparisons", comparisons),
        ("covariance_account", accounts),
        ("view_consistency", reliability),
        ("contribution_ledger", ledger),
    ):
        pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
    joined.to_csv(out / "private_outcome_connection.csv", index=False)
    return metrics


def main():
    """Run one six-representation packet, retaining partial files on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-fit-seconds", type=float, default=900)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE)
        or args.max_fit_seconds <= 0
        or args.bootstrap_draws < 1
    ):
        raise ValueError("Private output and positive budgets required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_models").mkdir()
    start = time.monotonic()
    manifest = {
        "protocol": "GRAIN_M23_CONDITIONAL_COORDINATES_V1",
        "date": "2026-09-09",
        "plan_sha256": sha(PLAN),
        "script_sha256": sha(__file__),
        "module_sha256": sha(
            ROOT / "grain_research/vector_mechanisms/conditional_coordinates.py"
        ),
        "source_outcomes_sha256": sha(SOURCE),
        "families": FAMILIES,
        "alpha_rule": "reuse M22 prefix mean Ridge alpha per outer fold for all arms",
        "rank": 64,
        "new_neural_calls": 0,
        "human_label_fit": False,
        "max_fit_seconds": args.max_fit_seconds,
        "bootstrap_draws": args.bootstrap_draws,
    }
    write(out / "manifest.json", manifest)
    rows, vrows, crows, trows, replays = [], [], [], [], []
    try:
        with threadpool_limits(limits=2):
            old, scores, packets, native_means, hashes, alphas = load_inputs()
            write(out / "private_input_hashes.json", hashes)
            for query in ("TF_cont", "neuroticism"):
                prior = old[old["query"] == query].set_index("author_id")
                names = np.array(sorted(prior.index))
                target = (
                    scores[scores["query"] == query].set_index("author_id").loc[names]
                )
                y = target.native.to_numpy()
                if not np.allclose(y, prior.loc[names, "teacher"], atol=1e-14, rtol=0):
                    raise ValueError("Teacher target changed")
                pv = np.asarray([packets[uid, query][0] for uid in names])
                qv = np.asarray([packets[uid, query][1] for uid in names])
                p, q = pv.mean(1), qv.mean(1)
                folds = grouped_folds(names)
                if not np.array_equal(folds, prior.loc[names, "fold"].to_numpy()):
                    raise ValueError("Outer author membership changed")
                for fold in range(5):
                    train = np.array(
                        sorted(
                            np.flatnonzero(folds != fold),
                            key=lambda j: key(names[j], f"s4_{fold}"),
                        )
                    )
                    test = np.flatnonzero(folds == fold)
                    # Replay the historical native-precision means before canonical float64 analysis.
                    old_model = joblib.load(
                        M22
                        / "private_models"
                        / f"prefix_matched_prefix_pq_mean_{query}_observed_{fold}.joblib"
                    )
                    xold = np.asarray([native_means[uid, query] for uid in names[test]])
                    replay = old_model["readers"]["ridge"].predict(xold)
                    replay_error = float(
                        np.max(np.abs(replay - prior.loc[names[test], "prediction"]))
                    )
                    if replay_error > 1e-8:
                        raise ValueError("Frozen M22 scores do not replay")
                    for family in FAMILIES:
                        reader = CoordinateReader.fit(
                            p[train], q[train], y[train], family, alphas[query, fold]
                        )
                        pred = reader.predict(p[test], q[test])
                        view_pred = reader.predict(
                            pv[test].reshape(-1, *p.shape[1:]),
                            qv[test].reshape(-1, q.shape[1]),
                        ).reshape(len(test), 3)
                        if np.max(np.abs(view_pred.mean(1) - pred)) > 1e-7:
                            raise ValueError("Affine view aggregation failed")
                        joblib.dump(
                            {
                                "reader": reader,
                                "query": query,
                                "train_authors": names[train],
                                "test_authors": names[test],
                                "question_tokens": read(
                                    BASE / "complete_level_response_v1/manifest.json"
                                )["question_tokens"][query],
                            },
                            out / "private_models" / f"{query}_{family}_{fold}.joblib",
                            compress=3,
                        )
                        for k, j in enumerate(test):
                            rows.append(
                                {
                                    "author_id": names[j],
                                    "query": query,
                                    "family": family,
                                    "fold": fold,
                                    "alpha": reader.alpha,
                                    "teacher": float(y[j]),
                                    "prediction": float(pred[k]),
                                    "unroll_error": reader.unroll_error,
                                }
                            )
                            vrows.extend(
                                {
                                    "author_id": names[j],
                                    "query": query,
                                    "family": family,
                                    "fold": fold,
                                    "view": v,
                                    "prediction": float(view_pred[k, v]),
                                }
                                for v in range(3)
                            )
                        if family == "p_mean_q":
                            replays.append(
                                {
                                    "query": query,
                                    "fold": fold,
                                    "old_replay_error": replay_error,
                                    "canonical_float64_delta": float(
                                        np.max(np.abs(pred - replay))
                                    ),
                                }
                            )
                        if family == "p_full_q":
                            parts, token, config = reader.full_parts(p[test], q[test])
                            for k, j in enumerate(test):
                                crows.append(
                                    {
                                        "author_id": names[j],
                                        "query": query,
                                        "fold": fold,
                                        "teacher": float(y[j]),
                                        "prediction": float(pred[k]),
                                        **{
                                            name: float(v[k])
                                            for name, v in parts.items()
                                        },
                                    }
                                )
                            for t in range(token.shape[1]):
                                trows.append(
                                    {
                                        "query": query,
                                        "fold": fold,
                                        "token_index": t,
                                        "authors": len(test),
                                        "token_rms": float(
                                            np.sqrt(np.mean(token[:, t] ** 2))
                                        ),
                                        "config_rms": float(
                                            np.sqrt(np.mean(config[:, t] ** 2))
                                        ),
                                        "token_teacher_cov": float(
                                            np.cov(token[:, t], y[test], ddof=0)[0, 1]
                                        ),
                                    }
                                )
                        pd.DataFrame(rows).to_csv(
                            out / "private_partial.csv", index=False
                        )
                        seconds = time.monotonic() - start
                        write(
                            out / "run_status.json",
                            {
                                "status": "RUNNING",
                                "query": query,
                                "fold": fold,
                                "family": family,
                                "seconds": seconds,
                            },
                        )
                        print(
                            f"M23 {query} fold {fold + 1}/5 {family}: {seconds:.1f}s",
                            flush=True,
                        )
                        if seconds > args.max_fit_seconds:
                            raise TimeoutError(
                                "M23 resource boundary; partial results retained"
                            )
                del pv, qv, p, q
        frame, views, contributions = (
            pd.DataFrame(rows),
            pd.DataFrame(vrows),
            pd.DataFrame(crows),
        )
        if frame.duplicated(["author_id", "query", "family"]).any() or len(
            frame
        ) != 277 * len(FAMILIES):
            raise ValueError("Incomplete or duplicate OOF rows")
        for name, data in (
            ("private_oof_before_outcomes", frame),
            ("private_view_scores", views),
            ("private_contributions", contributions),
            ("token_contribution_map", pd.DataFrame(trows)),
            ("replay_audit", pd.DataFrame(replays)),
        ):
            data.to_csv(out / f"{name}.csv", index=False)
        frozen = out / "private_oof_before_outcomes.csv"
        write(
            out / "oof_frozen.json",
            {"sha256": sha(frozen), "human_outcomes_read": False},
        )
        if sha(SOURCE) != manifest["source_outcomes_sha256"]:
            raise ValueError("Source outcomes changed during fit")
        metrics = analyze(out, frame, views, contributions, args.bootstrap_draws)
        summary = {
            "status": "COMPLETE",
            "seconds": time.monotonic() - start,
            "authors": 277,
            "new_neural_calls": 0,
            "oof_rows": len(frame),
            "max_unroll_error": float(frame.unroll_error.max()),
            "max_m22_replay_error": max(v["old_replay_error"] for v in replays),
            "max_precision_delta": max(v["canonical_float64_delta"] for v in replays),
            "metrics": metrics,
        }
        write(out / "summary.json", summary)
        write(
            out / "run_status.json",
            {k: v for k, v in summary.items() if k != "metrics"},
        )
        print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    except BaseException as exc:
        write(
            out / "run_status.json",
            {"status": "FAILED_RETAINED", "seconds": time.monotonic() - start},
        )
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
