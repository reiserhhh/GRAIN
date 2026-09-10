#!/usr/bin/env python3
"""Cache-only audit of whole-function outputs, tangent units and paired metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_whole_path_assembly import (
    BASE,
    CANDIDATES,
    ENDPOINT,
    QUERIES,
    apply_map,
    array_hash,
    arrays,
    body_directions,
    frozen,
    read,
    ref,
    sha,
    source_set,
    verify,
    verify_hashes,
    write,
)


def audit(out):
    """Recompute contractions and metrics without calling any model or fitting."""
    m, pf = verify(out), frozen(out, "predictions")
    verify_hashes(ROOT, m["evaluation"])
    jobs = read(out / "private_jobs.json")
    contraction_error, packets = 0.0, 0
    for j in jobs:
        models = {
            o: arrays(out / "private_models" / f"{o}_{ref(j)}_pointwise.npz")
            for o in CANDIDATES[1:]
        }
        for key in j["query_keys"]:
            data = arrays(out / "private_inputs" / (key + ".npz"))
            expected = arrays(out / "private_expected" / (key + ".npz"))
            p = arrays(out / "predictions" / (key + ".npz"))
            x, d = data["state"].astype(float), body_directions(data)
            if p["gradient"].shape != (3, *x.shape):
                raise ValueError("Complete candidate gradient inventory required")
            for ai, o in enumerate(CANDIDATES):
                mx, md = (x, d) if ai == 0 else apply_map(x, d, models[o])[:2]
                for a, name in ((mx, "x"), (md, "d")):
                    if array_hash(a) != str(expected[name + "_hash"][ai]) or array_hash(
                        a.astype(np.float32)
                    ) != str(p[name + "32_hash"][ai]):
                        raise ValueError(
                            "Original/observer-boundary state or tangent changed"
                        )
                tau = np.sum(
                    p["gradient"][ai].astype(float)[None]
                    * md.astype(np.float32).astype(float),
                    axis=(1, 2),
                )
                contraction_error = max(
                    contraction_error, float(abs(tau - p["tau"][ai]).max())
                )
            packets += 1
    if packets != 24 or contraction_error > 1e-11:
        raise ValueError("Cached gradient contractions differ")
    summaries = [
        read(out / n / "summary.json") for n in ("analysis", "replay_analysis")
    ]
    for folder, r in zip(("analysis", "replay_analysis"), summaries, strict=True):
        verify_hashes(out / folder, r["files"])
    if summaries[0]["decisions"] != summaries[1]["decisions"] or set(
        summaries[0]["files"]
    ) != set(summaries[1]["files"]):
        raise ValueError("Replay inventory or decisions differ")
    for name in summaries[0]["files"]:
        if sha(out / "analysis" / name) != sha(out / "replay_analysis" / name):
            raise ValueError("Cached table replay differs")
    scores = pd.read_csv(out / "analysis/private_scores.csv")
    actions = pd.read_csv(out / "analysis/private_actions.csv")
    source_scores = {}
    for j in [j for j in jobs if j["cohort"] == "source4"]:
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            source_scores[j["index"], q] = float(
                arrays(ENDPOINT / "capture_source" / (k + ".npz"))["score"]
            )
    for j in jobs:
        folder = "capture_source" if j["cohort"] == "source4" else "capture_new"
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            target = arrays(ENDPOINT / folder / (key + ".npz"))
            pred = arrays(out / "predictions" / (key + ".npz"))
            s = scores[
                (scores.author_key == j["key"]) & (scores.query_name == q)
            ].set_index("candidate")
            np.testing.assert_allclose(
                s.loc[list(CANDIDATES), "predicted"], pred["score"], atol=1e-14
            )
            np.testing.assert_allclose(s.actual, float(target["score"]), atol=1e-14)
            const = np.mean(
                [source_scores[i, q] for i in source_set(j["cohort"], j["index"])]
            )
            np.testing.assert_allclose(s.constant, const, atol=1e-14)
            a = actions[(actions.author_key == j["key"]) & (actions.query_name == q)]
            for ai, o in enumerate(CANDIDATES):
                g = a[a.candidate == o].sort_values("factor")
                np.testing.assert_allclose(g.actual, target["tau"], atol=1e-14)
                np.testing.assert_allclose(g.predicted, pred["tau"][ai], atol=1e-14)
    metric_error = 0.0
    for full, private, key, fields in (
        (
            scores,
            "private_score_contrasts.csv",
            ["author_key", "cohort", "candidate"],
            ["actual", "predicted", "constant"],
        ),
        (
            actions,
            "private_action_contrasts.csv",
            ["author_key", "cohort", "candidate", "factor"],
            ["actual", "predicted"],
        ),
    ):
        a, b = [
            full[full.query_name == q].set_index(key)[fields].sort_index()
            for q in QUERIES
        ]
        tab = (
            pd.read_csv(out / "analysis" / private).set_index(key)[fields].sort_index()
        )
        np.testing.assert_allclose(tab, a - b, atol=1e-13)
    for private, public, groups, score in (
        ("private_scores", "scores", ["cohort", "query_name", "candidate"], True),
        ("private_score_contrasts", "score_contrasts", ["cohort", "candidate"], True),
        ("private_actions", "actions", ["cohort", "query_name", "candidate"], False),
        (
            "private_actions",
            "factor_actions",
            ["cohort", "query_name", "candidate", "factor"],
            False,
        ),
        (
            "private_action_contrasts",
            "action_contrasts",
            ["cohort", "candidate"],
            False,
        ),
        (
            "private_action_contrasts",
            "factor_contrasts",
            ["cohort", "candidate", "factor"],
            False,
        ),
    ):
        raw = pd.read_csv(out / "analysis" / (private + ".csv"))
        result = pd.read_csv(out / "analysis" / (public + ".csv")).set_index(groups)
        for key, g in raw.groupby(groups, sort=True):
            y, p = g.actual.to_numpy(), g.predicted.to_numpy()
            error, prefix = p - y, "score" if score else "action"
            check = {
                prefix + "_r": np.corrcoef(y, p)[0, 1],
                prefix + "_rmse": np.sqrt(np.mean(error**2)),
            }
            if score:
                check.update(
                    score_mse_over_source_constant=np.mean(error**2)
                    / np.mean((y - g.constant.to_numpy()) ** 2),
                    mean_error=np.mean(error),
                    error_variance=np.var(error),
                )
            else:
                check["action_mse_over_zero"] = np.mean(error**2) / np.mean(y**2)
            metric_error = max(
                metric_error,
                max(abs(float(result.loc[key, k]) - v) for k, v in check.items()),
            )
    for name in summaries[0]["files"]:
        if name.endswith("error_changes.csv"):
            e = pd.read_csv(out / "analysis" / name)
            diff = (
                e.candidate_mse
                - e.baseline_mse
                - e.linear_error_term
                - e.correction_energy
            )
            metric_error = max(metric_error, float(abs(diff).max()))
    if metric_error > 1e-10:
        raise ValueError("Independent metric audit failed")
    if (pf["reverse_pairs"], pf["jvps"], pf["fd_forwards"]) != (72, 2, 4) or len(
        pf["checks"]
    ) != 2:
        raise ValueError("Native complete/check inventory differs")
    for r in pf["checks"]:
        for field, atol, rtol in (
            ("jvp", m["jvp_atol"], m["jvp_rtol"]),
            ("fd", m["fd_atol"], m["fd_rtol"]),
        ):
            np.testing.assert_allclose(r[field], r["reverse"], atol=atol, rtol=rtol)
    if (
        pf["extra_denials"]
        or pf["current_target_reads"]
        or pf["fits"]
        or pf["human_labels_read"]
        or pf["denial_probes"] != 4
    ):
        raise ValueError("Prediction dependency contract differs")
    if pf["native_seconds"] > 180 or pf["numeric_seconds"] > 60:
        raise ValueError("Execution budget exceeded")
    r = {
        "status": "PASS",
        "primary_commit": m["commit"],
        "candidate_packets": packets,
        "gradient_contraction_max_error": contraction_error,
        "independent_metric_max_error": float(metric_error),
        "cached_replayed_tables": len(summaries[0]["files"]),
        "reverse_pairs": 72,
        "jvps": 2,
        "fd_forwards": 4,
        "primary_native_seconds": pf["native_seconds"],
        "numeric_seconds": pf["numeric_seconds"],
        "loading_seconds": pf["loading_seconds"],
        "audit_native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "decisions": summaries[0]["decisions"],
        "audit_code_sha256": sha(Path(__file__)),
    }
    write(out / "replay_audit.json", r)
    print(r, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        audit(BASE / "logic_whole_path_assembly_v1")
