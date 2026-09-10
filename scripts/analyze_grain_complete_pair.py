#!/usr/bin/env python3
"""Evaluate frozen conditional scores and localize complete-interface error."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.comment_carrier import score
from grain_research.vector_mechanisms.complete_pair import ARMS, error_account
from grain_research.vector_mechanisms.joint_readout import centered_logits
from scripts.analyze_grain_conditional_energy import author_draws, ci, corr, within
from scripts.run_grain_complete_pair import OLD, PARENT, verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def analyze(out, name="analysis"):
    """Use native endpoints only after numerical prediction artifacts are frozen."""
    verified(out)
    dest = out / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "analyzer_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "new_neural_calls": 0,
            "human_labels_read": False,
            "bootstrap": "2000 complete-author draws, seed2501, pointwise fixed-prediction intervals",
        },
    )
    membership = read(out / "private_membership.json")
    by_author = {r["author_id"]: r for r in membership["authors"]}
    old = {
        (r["author_id"], r["pair"], r["arm"]): r
        for r in read(PARENT / "private_joint_predictions.json")
    }
    scores, components, terms = [], [], []
    replay, algebra = 0.0, 0.0
    for fold in membership["folds"]:
        number = fold["fold"]
        if set(fold["source"]) & set(fold["held"]):
            raise ValueError("Evaluation author in training membership")
        with np.load(out / f"private_fold{number}_predictions.npz") as z:
            prediction = dict(z)
        with np.load(out / f"private_fold{number}_model.npz") as z:
            base_mean = z["base_mean"].copy()
        for j, author in enumerate(fold["held"]):
            source = by_author[author]
            packet = read(
                OLD / "private_packets" / f"author_{source['index']:03d}.json"
            )
            endpoints = packet["scores"]
            values = packet["alphabet"]["values"]
            z0 = centered_logits(endpoints["zero"]["candidate_logits"])
            inc = np.array(
                [
                    centered_logits(endpoints[f"s{i}"]["candidate_logits"]) - z0
                    for i in (0, 1, 11)
                ]
            )
            for p, (pair, idx) in enumerate((("adjacent", 1), ("distant", 2))):
                actual = centered_logits(endpoints[pair]["candidate_logits"])
                additive = z0 + inc[0] + inc[idx]
                mixed = actual - additive
                for arm in ARMS:
                    meta = {
                        "author_id": author,
                        "fold": number,
                        "pair": pair,
                        "arm": arm,
                    }
                    cached = old[author, pair, arm]
                    pmix = prediction[f"{arm}_mixed"][j, p]
                    padd = prediction[f"{arm}_additive"][j, p]
                    pinc = prediction[f"{arm}_singleton"][j]
                    pbase = prediction["base"][j]
                    replay = max(
                        replay,
                        float(np.max(abs(pmix - cached["predicted_mixed"]))),
                        float(np.max(abs(mixed - cached["actual_mixed"]))),
                        float(np.max(abs(additive - centered_logits(cached["z_add"])))),
                        abs(score(actual, values) - cached["actual_score"]),
                    )
                    assemblies = {
                        "oracle_additive_old_mixed": additive + pmix,
                        "complete_predicted": prediction[f"{arm}_full"][j, p],
                        "predicted_additive_oracle_mixed": padd + mixed,
                        "predicted_additive": padd,
                        "source_base_mean_diagnostic": base_mean,
                    }
                    true_score = score(actual, values)
                    for label, pred in assemblies.items():
                        scores.append(
                            meta
                            | {
                                "assembly": label,
                                "actual": true_score,
                                "predicted": score(pred, values),
                                "logit_mse": float(np.mean((actual - pred) ** 2)),
                                "local_interaction": true_score
                                - score(additive, values),
                                "base_actual": score(z0, values),
                            }
                        )
                    comparisons = {
                        "base": (z0, pbase),
                        "singleton_i": (inc[0], pinc[0]),
                        "singleton_j": (inc[idx], pinc[idx]),
                        "additive": (additive, padd),
                        "mixed": (mixed, pmix),
                        "full": (actual, assemblies["complete_predicted"]),
                    }
                    for key, (target, pred) in comparisons.items():
                        components.append(
                            meta
                            | {
                                "component": key,
                                "mse": float(np.mean((target - pred) ** 2)),
                                "target_energy": float(np.mean(target**2)),
                                "predicted_energy": float(np.mean(pred**2)),
                            }
                        )
                    errors = {
                        "base": (pbase - z0)[None],
                        "i": (pinc[0] - inc[0])[None],
                        "j": (pinc[idx] - inc[idx])[None],
                        "mixed": (pmix - mixed)[None],
                    }
                    account = error_account(errors)
                    algebra = max(algebra, abs(float(account["identity_error"][0])))
                    terms.append(meta | {k: float(v[0]) for k, v in account.items()})
    if replay > 1e-12 or algebra > 1e-12:
        raise ValueError("Native quartet/mixed prediction/error algebra mismatch")
    s, c, t = map(pd.DataFrame, (scores, components, terms))
    s.to_csv(dest / "private_score_predictions.csv", index=False)
    metrics, paired = [], []
    for arm in ARMS:
        for scope in ("adjacent", "distant", "pooled"):
            g = s[s.arm == arm]
            if scope != "pooled":
                g = g[g.pair == scope]
            selected = {}
            for assembly, h in g.groupby("assembly"):
                h = h.sort_values(["author_id", "pair"]).reset_index(drop=True)
                y, pred = h.actual.to_numpy(), h.predicted.to_numpy()
                draws = author_draws(h.author_id)
                lo, hi, nvalid = ci([corr(y[ix], pred[ix]) for ix in draws])
                mse = float(np.mean((y - pred) ** 2))
                metrics.append(
                    {
                        "arm": arm,
                        "pair": scope,
                        "assembly": assembly,
                        "authors": h.author_id.nunique(),
                        "pairs": len(h),
                        "score_rmse": np.sqrt(mse),
                        "score_mae": np.mean(abs(y - pred)),
                        "score_r": corr(y, pred),
                        "r_low": lo,
                        "r_high": hi,
                        "r_bootstrap_n": nvalid,
                        "within_fold_r": corr(within(y, h.fold), within(pred, h.fold)),
                        "actual_score_sd": y.std(),
                        "predicted_score_sd": pred.std(),
                        "score_mse_over_global_variance": mse / y.var(),
                        "native_interaction_rms": np.sqrt(
                            np.mean(h.local_interaction**2)
                        ),
                        "logit_rmse": np.sqrt(h.logit_mse.mean()),
                    }
                )
                selected[assembly] = h
            main = selected["complete_predicted"]
            draws = author_draws(main.author_id)
            for other in (
                "predicted_additive",
                "oracle_additive_old_mixed",
                "predicted_additive_oracle_mixed",
                "source_base_mean_diagnostic",
            ):
                h = selected[other]
                if not main[["author_id", "pair"]].equals(h[["author_id", "pair"]]):
                    raise ValueError("Paired comparison index mismatch")
                delta = (
                    (main.actual - main.predicted) ** 2 - (h.actual - h.predicted) ** 2
                ).to_numpy()
                lo, hi, _ = ci([delta[ix].mean() for ix in draws])
                paired.append(
                    {
                        "arm": arm,
                        "pair": scope,
                        "comparator": other,
                        "mse_delta_complete_minus_other": delta.mean(),
                        "low": lo,
                        "high": hi,
                        "authors": main.author_id.nunique(),
                    }
                )
    pd.DataFrame(metrics).to_csv(dest / "score_comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(dest / "paired_score_mse.csv", index=False)
    summary_c = (
        c.groupby(["arm", "pair", "component"])[
            ["mse", "target_energy", "predicted_energy"]
        ]
        .mean()
        .reset_index()
    )
    summary_c["mse_over_zero"] = summary_c.mse / summary_c.target_energy
    summary_c.to_csv(dest / "component_errors.csv", index=False)
    t.drop(columns=["author_id", "fold"]).groupby(
        ["arm", "pair"]
    ).mean().reset_index().to_csv(dest / "error_cross_terms.csv", index=False)
    files = [
        "private_score_predictions.csv",
        "score_comparison.csv",
        "paired_score_mse.csv",
        "component_errors.csv",
        "error_cross_terms.csv",
    ]
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "native_endpoint_replay_max": replay,
            "error_identity_max": algebra,
            "human_labels_read": False,
            "new_neural_calls": 0,
            "assembly_rows": len(s),
            "metric_rows": len(metrics),
            "paired_comparisons": len(paired),
            "files": {p: sha(dest / p) for p in files},
        },
    )
    print(
        {"status": "COMPLETE", "metric_rows": len(metrics), "native_replay_max": replay}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="logic_complete_pair_v1")
    p.add_argument("--name", default="analysis")
    a = p.parse_args()
    analyze(BASE / a.source, a.name)
