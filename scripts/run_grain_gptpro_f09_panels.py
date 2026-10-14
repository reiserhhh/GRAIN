#!/usr/bin/env python3
"""Full frozen main/process rival panels, with equivalence-class scoring."""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f09_detection import score_oof
from grain_research.gptpro_foundation_v3.f09_rivals import (
    EVEN,
    ODD,
    candidates,
    fit_family,
    predict_family,
    probability,
)


def main():
    """Generate disjoint role streams; pass only observed prefixes to estimators."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    counts = (
        {"source": 12, "validation": 6, "application": 12}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    protocol = {
        "seeds": seeds,
        "counts": counts,
        "panels": ["main", "process"],
        "targets": ["even12_to_odd12", "odd6_to_odd6"],
        "second_target": "uses only first6 odd observations,no even observations",
        "class_prior": "uniform over observational classes;uniform atoms within class",
        "tie_rule": "lexical first class within1e-14",
        "application_selection": False,
        "classification_ineligible": "M4 not in declared candidate family",
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    metrics = []
    people = 0
    even = np.repeat(EVEN, 3, axis=0)
    odd = np.repeat(ODD, 3, axis=0)
    with gzip.open(args.output / "author_packets.jsonl.gz", "wt") as stream:
        for seed in seeds:
            calibration = np.load(args.calibration / f"calibration_{seed}.npz")[
                "statistics"
            ]
            for panel_index, panel in enumerate(("main", "process")):
                spec = candidates(panel == "process")
                for role_index, (role, n) in enumerate(counts.items()):
                    params, noise = [
                        np.random.default_rng(s)
                        for s in np.random.SeedSequence(
                            [seed, 9, 903, panel_index, role_index]
                        ).spawn(2)
                    ]
                    classes = np.resize(np.arange(len(spec["class_keys"])), n)
                    params.shuffle(classes)
                    outside = np.zeros(n, dtype=bool)
                    if panel == "main" and role == "application":
                        outside[params.choice(n, n // 4, replace=False)] = True
                    for i, cls in enumerate(classes):
                        indices = np.flatnonzero(spec["classes"] == cls)
                        family, beta = spec["atoms"][int(params.choice(indices))]
                        if outside[i]:
                            family = "M4"
                        author = (
                            "u_"
                            + hashlib.sha256(
                                f"{seed}|{panel}|{role}|{i}".encode()
                            ).hexdigest()[:16]
                        )
                        pdev = probability(even, family, beta)
                        pfuture = probability(odd, family, beta)
                        dev = (noise.random(12) < pdev).astype(int)
                        future = (noise.random(12) < pfuture).astype(int)
                        packet = {
                            "author_id": author,
                            "seed": seed,
                            "role": role,
                            "panel": panel,
                            "family": family,
                            "beta": beta,
                            "outside": bool(outside[i]),
                            "class_truth": None if outside[i] else int(cls),
                            "development": dev.tolist(),
                            "future": future.tolist(),
                            "future_probability": pfuture.tolist(),
                            "scores": [],
                        }
                        for target, ztrain, ytrain, ztest, ytest, truthp in (
                            ("even12_to_odd12", even, dev, odd, future, pfuture),
                            (
                                "odd6_to_odd6",
                                odd[:6],
                                future[:6],
                                odd[6:],
                                future[6:],
                                pfuture[6:],
                            ),
                        ):
                            fitted = fit_family(ztrain, ytrain, panel == "process")
                            prediction = predict_family(fitted, ztest)
                            cp = fitted["class_posterior"]
                            chosen = int(np.flatnonzero(cp >= cp.max() - 1e-14)[0])
                            row = {
                                "author_id": author,
                                "seed": seed,
                                "role": role,
                                "panel": panel,
                                "target": target,
                                "outside": bool(outside[i]),
                                "class_correct": None
                                if outside[i]
                                else bool(chosen == cls),
                                "true_class_mass": None
                                if outside[i]
                                else float(cp[cls]),
                                "brier": float(np.mean((prediction - ytest) ** 2)),
                                "log_loss": float(
                                    np.mean(
                                        -np.log(
                                            np.where(ytest, prediction, 1 - prediction)
                                        )
                                    )
                                ),
                                "probability_mse": float(
                                    np.mean((prediction - truthp) ** 2)
                                ),
                                "events": len(ytest),
                            }
                            metrics.append(row)
                            packet["scores"].append(
                                {
                                    **row,
                                    "posterior": fitted["posterior"].tolist(),
                                    "class_posterior": cp.tolist(),
                                    "prediction": prediction.tolist(),
                                }
                            )
                        if panel == "main":
                            result = score_oof(
                                dev.reshape(4, 3).sum(axis=1),
                                future.reshape(4, 3).sum(axis=1),
                                calibration,
                            )
                            packet["out_of_family"] = {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in result.items()
                            }
                        stream.write(json.dumps(packet) + "\n")
                        people += 1
    with (args.output / "metrics_by_author.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    summaries = []
    for panel in ("main", "process"):
        for role in counts:
            for target in protocol["targets"]:
                for outside in (False, True):
                    rows = [
                        r
                        for r in metrics
                        if (r["panel"], r["role"], r["target"], r["outside"])
                        == (panel, role, target, outside)
                    ]
                    if not rows:
                        continue
                    summaries.append(
                        {
                            "panel": panel,
                            "role": role,
                            "target": target,
                            "outside": outside,
                            "n": len(rows),
                            "class_accuracy": None
                            if outside
                            else float(np.mean([r["class_correct"] for r in rows])),
                            "brier": float(np.mean([r["brier"] for r in rows])),
                            "probability_mse": float(
                                np.mean([r["probability_mse"] for r in rows])
                            ),
                        }
                    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "status": "FULL_PANELS_COMPUTED_AUDIT_PENDING",
                "authors": people,
                "targets": len(metrics),
                "results": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "authors": people,
                "target_rows": len(metrics),
                "application": [r for r in summaries if r["role"] == "application"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
