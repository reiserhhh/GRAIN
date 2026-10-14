#!/usr/bin/env python3
"""M01/M02: learned text conditions and event-dependent personal profiles.

Prepare/collect/analyze stages freeze inputs before outcomes. All text is public
authored mechanism material, not generated human personality ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_accumulation import (
    calibration_fit,
    profile_fit,
    resolve_revisions,
)
from scripts.prepare_grain_predecessor_reference import digest, write
from scripts.run_grain_entailment_reference import MODEL as NLI_MODEL
from scripts.run_grain_entailment_reference import REVISION as NLI_REVISION
from scripts.run_grain_evidence_calibration import infer_pairs
from scripts.run_grain_paraphrase_reference import MODEL, REVISION, encode_batches


def build_bank(config):
    """Cross outcome signs with conditions; split complete wording families."""
    rows = []
    for q in config["queries"]:
        for partition, contexts, responses in (
            ("source", range(4), range(2)),
            ("held", range(4, 8), range(2, 4)),
        ):
            for z in (-1, 1):
                for sign in (-1, 1):
                    for ci in contexts:
                        for ri in responses:
                            text = f"{q['conditions'][str(z)][ci]}, {q['responses'][str(sign)][ri]}"
                            rows.append(
                                {
                                    "case_id": f"{q['id']}:{partition}:{z}:{sign}:{ci}:{ri}",
                                    "query": q["id"],
                                    "partition": partition,
                                    "condition": z,
                                    "sign": sign,
                                    "context_variant": ci,
                                    "response_variant": ri,
                                    "text": text,
                                }
                            )
    return pd.DataFrame(rows)


def prepare(output, config_path):
    """Freeze source code, config and public material before neural collection."""
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    output.mkdir(parents=True)
    config = json.loads(config_path.read_text())
    write(output / "config.json", config)
    build_bank(config).to_csv(output / "public_bank.csv", index=False)
    files = [Path(__file__), ROOT / "grain_research/evidence_accumulation.py"]
    files += [
        ROOT / "scripts" / f
        for f in (
            "run_grain_paraphrase_reference.py",
            "run_grain_evidence_calibration.py",
            "run_grain_entailment_reference.py",
        )
    ]
    write(
        output / "contract.json",
        {
            "version": config["version"],
            "files": {str(p.relative_to(ROOT)): digest(p) for p in files},
            "config_sha256": digest(output / "config.json"),
            "bank_sha256": digest(output / "public_bank.csv"),
            "encoder": MODEL,
            "encoder_revision": REVISION,
            "nli": NLI_MODEL,
            "nli_revision": NLI_REVISION,
            "design": "Source labels fit condition/calibration only; held labels grade only. Known event IDs are design metadata. All material public synthetic; no human labels.",
            "arms": ["oracle", "learned", "no_condition"],
            "sampling": ["independent", "repeated", "selective_repetition"],
            "estimators": ["report_iid", "report_cluster", "event_cluster"],
        },
    )


def verify(output):
    """Reject changed collection inputs or model/scoring implementation."""
    c = json.loads((output / "contract.json").read_text())
    for path, expected in c["files"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Changed implementation: {path}")
    if (
        digest(output / "config.json") != c["config_sha256"]
        or digest(output / "public_bank.csv") != c["bank_sha256"]
    ):
        raise ValueError("Changed frozen material")
    return json.loads((output / "config.json").read_text()), pd.read_csv(
        output / "public_bank.csv"
    )


def collect(output):
    """Run only the frozen local encoder and paired behavioral-event NLI."""
    config, bank = verify(output)
    if (output / "reader_manifest.json").exists():
        print(
            "Reader collection already complete; analyze uses verified caches.",
            flush=True,
        )
        return
    start = time.monotonic()
    vectors, _ = encode_batches(
        bank.text.tolist(), output / "encoder", MODEL, REVISION, cap=128
    )
    if json.loads((output / "encoder/contract.json").read_text())["truncated_texts"]:
        raise ValueError("No truncation allowed for the designed material")
    np.save(output / "vectors.npy", vectors)
    hypotheses = {q["id"]: q for q in config["queries"]}
    pairs = []
    for r in bank.itertuples():
        for key in ("positive", "negative"):
            pairs.append(
                {
                    "audit_id": r.case_id + "/" + key,
                    "case_id": r.case_id,
                    "key": key,
                    "sentence": r.text,
                    "item_text": hypotheses[r.query][key + "_hypothesis"],
                }
            )
    pairs = pd.DataFrame(pairs)
    print(f"Reading {len(bank)} public texts / {len(pairs)} NLI pairs", flush=True)
    result = pairs.merge(infer_pairs(pairs), on="audit_id", validate="one_to_one")
    result.to_csv(output / "public_nli.csv", index=False)
    write(
        output / "reader_manifest.json",
        {
            "files": {p: digest(output / p) for p in ("vectors.npy", "public_nli.csv")},
            "seconds": time.monotonic() - start,
        },
    )


def read_conditions(config, bank, vectors, nli):
    """Fit source-only condition estimates and evidence calibration, not traits."""
    rows, fitted = [], {}
    for query, f in bank.groupby("query", sort=True):
        source = f.index[f.partition.eq("source")].to_numpy()
        ix = f.index.to_numpy()
        scaler = StandardScaler().fit(vectors[source])
        model = Ridge(alpha=config["condition_alpha"]).fit(
            scaler.transform(vectors[source]), bank.loc[source, "condition"]
        )
        zhat = np.clip(model.predict(scaler.transform(vectors[ix])), -1, 1)
        table = nli.pivot(
            index="case_id", columns="key", values="p_entailment"
        ).reindex(f.case_id)
        margin = (table.positive - table.negative).to_numpy()
        mask = f.partition.eq("source").to_numpy()
        calibration = calibration_fit(margin[mask], f.sign.to_numpy()[mask])
        scores = calibration[0] + calibration[1] * margin
        g = f.copy()
        g["condition_hat"], g["event_margin"], g["evidence_hat"] = zhat, margin, scores
        rows.append(g)
        fitted[query] = {
            "source_case_ids": bank.loc[source, "case_id"].tolist(),
            "condition_coef": model.coef_.tolist(),
            "condition_intercept": float(model.intercept_),
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "calibration": calibration.tolist(),
        }
    return pd.concat(rows).sort_index(), fitted


def draw_reports(rng, pool, level, response, sampling):
    """Generate balanced known opportunities; duplicate event identity is explicit."""
    records = []
    for condition in (-1, 1):
        count = 24 if sampling == "independent" else 6
        directions = np.where(
            rng.random(count) < (1 + level + response * condition) / 2, 1, -1
        )
        copies = np.ones(count, dtype=int)
        if sampling == "repeated":
            copies[:] = 4
        elif sampling == "selective_repetition":
            copies[int(np.argmax(directions))] += 18
        for event, (sign, times) in enumerate(zip(directions, copies, strict=True)):
            choices = pool[pool.condition.eq(condition) & pool.sign.eq(sign)]
            for _ in range(times):
                row = choices.iloc[int(rng.integers(len(choices)))].to_dict()
                row["event_id"] = f"{condition}/{event}"
                records.append(row)
    return pd.DataFrame(records)


def simulation(config, scores):
    """Measure profile recovery and CI behavior across independent author draws."""
    results = []
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        for query, qrows in scores[scores.partition.eq("held")].groupby(
            "query", sort=True
        ):
            for author in range(config["authors_per_seed"]):
                level, response = rng.uniform(-0.35, 0.35), rng.uniform(-0.45, 0.45)
                truth = np.array([level, response])
                for sampling in ("independent", "repeated", "selective_repetition"):
                    reports = draw_reports(rng, qrows, level, response, sampling)
                    for condition_arm in ("oracle", "learned"):
                        z = (
                            reports.condition.to_numpy()
                            if condition_arm == "oracle"
                            else reports.condition_hat.to_numpy()
                        )
                        for method, balance, cluster in (
                            ("report_iid", False, False),
                            ("report_cluster", False, True),
                            ("event_cluster", True, True),
                        ):
                            fit = profile_fit(
                                z,
                                reports.evidence_hat,
                                reports.event_id,
                                balance_events=balance,
                                cluster_errors=cluster,
                            )
                            for k, target in enumerate(("level", "response")):
                                results.append(
                                    {
                                        "seed": seed,
                                        "query": query,
                                        "author": author,
                                        "sampling": sampling,
                                        "condition_arm": condition_arm,
                                        "method": method,
                                        "target": target,
                                        "truth": truth[k],
                                        "estimate": fit["beta"][k] if fit else np.nan,
                                        "lower": fit["lower"][k] if fit else np.nan,
                                        "upper": fit["upper"][k] if fit else np.nan,
                                        "events": reports.event_id.nunique(),
                                        "reports": len(reports),
                                        "effective_events": fit[
                                            "effective_events_perfect_dependence"
                                        ]
                                        if fit
                                        else np.nan,
                                    }
                                )
                    for target, value, actual in (
                        ("level", reports.evidence_hat.mean(), level),
                        ("response", 0.0, response),
                    ):
                        results.append(
                            {
                                "seed": seed,
                                "query": query,
                                "author": author,
                                "sampling": sampling,
                                "condition_arm": "no_condition",
                                "method": "pooled_mean",
                                "target": target,
                                "truth": actual,
                                "estimate": value,
                                "lower": np.nan,
                                "upper": np.nan,
                                "events": reports.event_id.nunique(),
                                "reports": len(reports),
                                "effective_events": np.nan,
                            }
                        )
    return pd.DataFrame(results)


def aggregate_metrics(predictions, by_seed=True):
    """Summarize authors, never count repeated reports as independent persons."""
    keys = ["query", "sampling", "condition_arm", "method", "target"]
    if by_seed:
        keys = ["seed"] + keys
    rows = []
    for key, f in predictions.groupby(keys, sort=True):
        valid = f.estimate.notna()
        g = f[valid]
        delta = g.estimate - g.truth
        intervals = g.lower.notna() & g.upper.notna()
        r = (
            float(pearsonr(g.truth, g.estimate).statistic)
            if len(g) > 2 and g.estimate.std() > 1e-12
            else None
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "authors": len(f),
                "observed": len(g),
                "pearson_r": r,
                "rmse": float(np.sqrt(np.mean(delta**2))),
                "bias": float(delta.mean()),
                "coverage95": float(
                    ((g.truth >= g.lower) & (g.truth <= g.upper))[intervals].mean()
                )
                if intervals.any()
                else None,
                "mean_interval_width": float((g.upper - g.lower)[intervals].mean())
                if intervals.any()
                else None,
                "mean_effective_events": float(g.effective_events.mean())
                if g.effective_events.notna().any()
                else None,
            }
        )
    return pd.DataFrame(rows)


def analyze(output):
    """Execute finite simulations from hash-verified readers and freeze outputs."""
    config, bank = verify(output)
    manifest = json.loads((output / "reader_manifest.json").read_text())
    for p, sha in manifest["files"].items():
        if digest(output / p) != sha:
            raise ValueError("Reader cache changed")
    start = time.monotonic()
    scores, fitted = read_conditions(
        config,
        bank,
        np.load(output / "vectors.npy"),
        pd.read_csv(output / "public_nli.csv"),
    )
    write(output / "fitted_source.json", fitted)
    scores.to_csv(output / "public_scored_bank.csv", index=False)
    reader_rows = []
    for (query, partition), f in scores.groupby(["query", "partition"]):
        reader_rows.append(
            {
                "query": query,
                "partition": partition,
                "n": len(f),
                "condition_accuracy": float(
                    np.mean(np.sign(f.condition_hat) == f.condition)
                ),
                "condition_rmse": float(
                    np.sqrt(np.mean((f.condition_hat - f.condition) ** 2))
                ),
                "event_sign_accuracy": float(
                    np.mean(np.sign(f.evidence_hat) == f.sign)
                ),
                "event_rmse": float(np.sqrt(np.mean((f.evidence_hat - f.sign) ** 2))),
            }
        )
    pd.DataFrame(reader_rows).to_csv(output / "reader_metrics.csv", index=False)
    prediction = simulation(config, scores)
    prediction.to_csv(output / "profile_predictions.csv", index=False)
    aggregate_metrics(prediction).to_csv(output / "metrics_by_seed.csv", index=False)
    metrics = aggregate_metrics(prediction, False)
    metrics.to_csv(output / "metrics.csv", index=False)
    revisions = pd.DataFrame(
        {
            "author": ["a"] * 3,
            "event": ["x", "y", "x"],
            "version": [1, 1, 2],
            "active": [True] * 3,
            "score": [1.0, 1.0, -1.0],
        }
    )
    current, history = resolve_revisions(revisions)
    write(
        output / "revision_witness.json",
        {
            "naive_all_records": float(history.score.mean()),
            "latest_event_mean": float(current.score.mean()),
            "history_rows": len(history),
            "active_events": len(current),
        },
    )
    public = [
        "public_scored_bank.csv",
        "reader_metrics.csv",
        "profile_predictions.csv",
        "metrics_by_seed.csv",
        "metrics.csv",
        "fitted_source.json",
        "revision_witness.json",
    ]
    write(
        output / "summary.json",
        {
            "status": "finite_packet_complete",
            "public_texts": len(bank),
            "simulated_authors_per_query": len(config["seeds"])
            * config["authors_per_seed"],
            "prediction_rows": len(prediction),
            "reader_metrics": reader_rows,
            "analysis_seconds": time.monotonic() - start,
            "collection_seconds": manifest["seconds"],
            "files": {p: digest(output / p) for p in public},
            "scope": "Known event identity; learned two-condition reader; authored held wording. No new humans, questionnaire labels or open-domain condition discovery.",
        },
    )
    print(metrics.to_string(index=False), flush=True)


def main():
    """Stage CLI; use fresh output for independent experiment changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "collect", "analyze"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/grain_condition_accumulation.json",
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output, args.config)
    elif args.stage == "collect":
        collect(args.output)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
