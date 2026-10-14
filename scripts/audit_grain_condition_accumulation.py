#!/usr/bin/env python3
"""Independent arithmetic and source-only checks of M01/M02 packet outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_accumulation import profile_fit
from scripts.prepare_grain_predecessor_reference import digest, write
from scripts.run_grain_condition_accumulation import read_conditions, verify


def check_metrics(pred, summary):
    """Independently aggregate every result cell from saved individual predictions."""
    keys = ["query", "sampling", "condition_arm", "method", "target"]
    max_error = 0.0
    for row in summary.itertuples():
        mask = np.ones(len(pred), bool)
        for k in keys:
            mask &= pred[k].eq(getattr(row, k))
        f = pred[mask]
        assert len(f) == row.authors
        valid = f.dropna(subset=["estimate"])
        assert len(valid) == row.observed
        error = valid.estimate.to_numpy() - valid.truth.to_numpy()
        max_error = max(
            max_error,
            abs(np.sqrt(error @ error / len(error)) - row.rmse),
            abs(error.mean() - row.bias),
        )
        if np.isfinite(row.pearson_r):
            corr = np.corrcoef(valid.truth, valid.estimate)[0, 1]
            max_error = max(max_error, abs(corr - row.pearson_r))
        interval = valid.dropna(subset=["lower", "upper"])
        if len(interval):
            coverage = np.mean(
                (interval.lower <= interval.truth) & (interval.truth <= interval.upper)
            )
            max_error = max(max_error, abs(coverage - row.coverage95))
    assert max_error < 1e-10, max_error
    return max_error


def paired_comparisons(source, scope):
    """Paired bootstrap over simulated authors within seeds, fixed wording bank."""
    original = pd.read_csv(source / "profile_predictions.csv")
    changed = pd.read_csv(scope / "profile_predictions.csv")
    rows = []
    rng = np.random.default_rng(91399)
    for query in original["query"].unique():
        for target in ("level", "response"):
            for contrast in ("condition_scope", "event_weighting"):
                base = original[
                    original["query"].eq(query)
                    & original.target.eq(target)
                    & original.condition_arm.eq("learned")
                ]
                if contrast == "condition_scope":
                    a = base[
                        base.sampling.eq("independent")
                        & base.method.eq("event_cluster")
                    ]
                    b = changed[
                        changed["query"].eq(query)
                        & changed.target.eq(target)
                        & changed.condition_arm.eq("learned")
                        & changed.sampling.eq("independent")
                        & changed.method.eq("event_cluster")
                    ]
                else:
                    a = base[
                        base.sampling.eq("selective_repetition")
                        & base.method.eq("report_cluster")
                    ]
                    b = base[
                        base.sampling.eq("selective_repetition")
                        & base.method.eq("event_cluster")
                    ]
                paired = (
                    a.merge(
                        b,
                        on=["seed", "author"],
                        suffixes=("_old", "_new"),
                        validate="one_to_one",
                    )
                    .sort_values(["seed", "author"])
                    .reset_index(drop=True)
                )
                np.testing.assert_allclose(
                    paired.truth_old, paired.truth_new, atol=0, rtol=0
                )
                e1 = (paired.estimate_old - paired.truth_old).to_numpy() ** 2
                e2 = (paired.estimate_new - paired.truth_new).to_numpy() ** 2
                indexes = [g.index.to_numpy() for _, g in paired.groupby("seed")]
                boot = []
                for _ in range(999):
                    ix = np.concatenate(
                        [rng.choice(g, len(g), replace=True) for g in indexes]
                    )
                    boot.append(np.sqrt(e1[ix].mean()) - np.sqrt(e2[ix].mean()))
                lower, upper = np.quantile(boot, [0.025, 0.975])
                rows.append(
                    {
                        "query": query,
                        "target": target,
                        "contrast": contrast,
                        "authors": len(paired),
                        "rmse_reduction": np.sqrt(e1.mean()) - np.sqrt(e2.mean()),
                        "lower": lower,
                        "upper": upper,
                        "positive_seeds": sum(
                            np.mean(e1[ix]) > np.mean(e2[ix]) for ix in indexes
                        ),
                        "seeds": len(indexes),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(source / "paired_comparisons.csv", index=False)
    return result


def main():
    """Check hashes, all metrics, source-only fit and WLS via a second library."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--scope", type=Path)
    args = p.parse_args()
    config, bank = verify(args.source)
    reports = {}
    for directory in [args.source] + ([args.scope] if args.scope else []):
        s = json.loads((directory / "summary.json").read_text())
        for name, sha in s["files"].items():
            assert digest(directory / name) == sha, name
        predictions = pd.read_csv(directory / "profile_predictions.csv")
        metrics = pd.read_csv(directory / "metrics.csv")
        reports[directory.name] = {
            "metric_rows": len(metrics),
            "prediction_rows": len(predictions),
            "metric_max_error": check_metrics(predictions, metrics),
        }
        vectors = np.load(directory / "vectors.npy")
        nli = pd.read_csv(args.source / "public_nli.csv")
        scores, fit = read_conditions(config, bank, vectors, nli)
        changed = bank.copy()
        held = changed.partition.eq("held")
        changed.loc[held, "condition"] *= -1
        changed.loc[held, "sign"] *= -1
        replay, fit2 = read_conditions(config, changed, vectors, nli)
        assert fit == fit2, "Held truth reached training"
        np.testing.assert_array_equal(scores.condition_hat, replay.condition_hat)
        np.testing.assert_array_equal(scores.evidence_hat, replay.evidence_hat)
        saved = pd.read_csv(directory / "public_scored_bank.csv")
        np.testing.assert_allclose(
            scores.condition_hat, saved.condition_hat, rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            scores.evidence_hat, saved.evidence_hat, rtol=1e-7, atol=1e-7
        )
    rng = np.random.default_rng(1973)
    ix = np.repeat(np.arange(16), np.arange(16) % 5 + 1)
    z = rng.normal(size=16)[ix]
    y = 0.2 + 0.4 * z + rng.normal(size=16)[ix] + rng.normal(scale=0.05, size=len(ix))
    max_error = 0.0
    for balance in (False, True):
        weights = 1 / np.bincount(ix)[ix] if balance else np.ones(len(ix))
        result = profile_fit(z, y, ix, balance_events=balance, cluster_errors=True)
        independent = sm.WLS(y, sm.add_constant(z), weights=weights).fit(
            cov_type="cluster", cov_kwds={"groups": ix, "use_correction": True}
        )
        difference = max(
            np.max(abs(independent.params - result["beta"])),
            np.max(abs(independent.cov_params() - result["covariance"])),
        )
        assert difference < 1e-12, difference
        max_error = max(max_error, difference)
    if args.scope:
        print(paired_comparisons(args.source, args.scope).to_string(index=False))
    write(
        args.source / "audit.json",
        {
            "status": "PASS",
            "packets": reports,
            "held_label_mutation_invariant": True,
            "statsmodels_max_error": max_error,
            "scope": "All aggregate metric cells and reader fits; independent covariance fixture. No new neural calls, no blind human validation.",
        },
    )
    print(json.dumps(reports, indent=2))
    print(
        f"PASS: held-label invariance; independent WLS covariance max error {max_error:.3g}"
    )


if __name__ == "__main__":
    main()
