#!/usr/bin/env python3
"""Independent precision-formula audit of two source-constructed instruments."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_local_instrument_construction import (
    DEFAULT,
    frozen_bundles,
    loadlines,
    verify,
)
from scripts.run_grain_query_moment_operator import read, sha, write


def independent_score(raw, margins, bundle):
    """Use explicit posterior precision, not the application's covariance conditioning."""
    import hashlib

    spec = bundle["spec"]
    unique = {r["event"]: r for r in raw if r["scope"] == spec["allowed_scope"]}
    rows = list(unique.values())
    cf = bundle["calibration"]["coef"]
    values = np.array(
        [
            (
                margins[
                    hashlib.sha256(
                        (spec["query"] + "\0" + r["text"]).encode()
                    ).hexdigest()
                ]
                - cf[0]
                - cf[2] * r["condition"]
            )
            / (cf[1] + cf[3] * r["condition"])
            for r in rows
        ]
    )
    draws = defaultdict(list)
    for fitted in bundle["bootstrap"]:
        if spec["kind"] == "conditional_profile":
            mean, prior = np.asarray(fitted["mean"]), np.asarray(fitted["covariance"])
            h = np.array(
                [[r["condition"] == -1, r["condition"] == 1] for r in rows], float
            )
            noise = np.array([fitted["noise"][int(r["condition"] == 1)] for r in rows])
            targets = {
                "level": [0.5, 0.5],
                "condition": [-0.5, 0.5],
                "low_condition": [1.0, 0.0],
                "high_condition": [0.0, 1.0],
            }
        else:
            times = np.array(sorted({r["time"] for r in rows}))
            n = len(times)
            state_cov = fitted["stationary"] * np.exp(
                -fitted["rate"] * abs(times[:, None] - times[None, :])
            )
            prior = block_diag(fitted["covariance"], state_cov)
            mean = np.r_[fitted["mean"], np.zeros(n)]
            h = np.zeros((len(rows), n + 3))
            for i, row in enumerate(rows):
                h[i, :3] = [1.0, row["condition"], row["partner"]]
                h[i, 3 + int(np.flatnonzero(times == row["time"])[0])] = 1
            noise = np.full(
                len(rows), fitted["noise"] * bundle["pilot_reports_per_occasion"]
            )
            targets = {
                name: np.eye(n + 3)[i]
                for i, name in enumerate(("level", "condition", "partner"))
            }
            targets["current_state"] = np.eye(n + 3)[-1]
            last = next(r for r in rows if r["time"] == times[-1])
            targets["current_response"] = targets["current_state"].copy()
            targets["current_response"][:3] = [1.0, last["condition"], last["partner"]]
            gap = np.diff(times)
            w = np.r_[gap[0] / 2, (gap[:-1] + gap[1:]) / 2, gap[-1] / 2] / (
                times[-1] - times[0]
            )
            targets["window_state"] = np.r_[np.zeros(3), w]
        precision = np.linalg.inv(prior)
        post = np.linalg.inv(precision + h.T @ (h / noise[:, None]))
        estimate = post @ (precision @ mean + h.T @ (values / noise))
        for name, target in targets.items():
            t = np.array(target)
            draws[name].append((float(t @ estimate), float(t @ post @ t)))
    return {
        name: {
            "score": np.mean(np.asarray(a)[:, 0]),
            "within_variance": np.mean(np.asarray(a)[:, 1]),
            "construction_variance": np.var(np.asarray(a)[:, 0], ddof=1),
        }
        for name, a in draws.items()
    }


def audit(out):
    """Recompute every application target and metrics; verify exact cold input separation."""
    contract, cfg = verify(out)
    frozen_bundles(out)
    for stage in ("application", "evaluation"):
        meta = read(out / stage / "freeze.json")
        for name, expected in meta["files"].items():
            if sha(out / stage / name) != expected:
                raise ValueError("Changed recorded stage: " + name)
    source_freeze = read(out / "instruments/freeze.json")
    app_freeze = read(out / "application/freeze.json")
    if source_freeze["extra_denials"] or app_freeze["extra_denials"]:
        raise ValueError("Undeclared constructor/scorer access")
    if (
        not contract["prepared_time_ns"]
        < source_freeze["frozen_time_ns"]
        < app_freeze["frozen_time_ns"]
    ):
        raise ValueError("Construction/application chronology invalid")
    margins = read(out / "reader_cache.json")["margins"]
    participants = {}
    count, max_error = 0, 0.0
    for job in contract["jobs"]:
        raw = loadlines(out / "raw" / (job["key"] + ".jsonl"))
        ids = {r["author"] for r in raw}
        if len(ids) != job["authors"]:
            raise ValueError("Author count disagrees with contract")
        for author in ids:
            if author in participants:
                raise ValueError("Author reused across evidence partitions")
            participants[author] = job["key"]
        if job["partition"] == "pilot":
            continue
        bundle = read(out / "instruments" / f"{job['instrument']}_{job['seed']}.json")
        if ids.intersection(bundle["pilot_authors"]):
            raise ValueError("Scored person used as source pilot")
        saved = read(out / "application" / (job["key"] + ".json"))
        grouped = defaultdict(list)
        for row in raw:
            grouped[row["author"]].append(row)
        for author, rows in grouped.items():
            independently = independent_score(rows, margins, bundle)
            for name, values in independently.items():
                actual = saved[author]["targets"][name]
                for field, expected in values.items():
                    error = abs(actual[field] - expected)
                    max_error = max(max_error, error)
                    np.testing.assert_allclose(
                        actual[field], expected, atol=1e-8, rtol=1e-7
                    )
                np.testing.assert_allclose(
                    actual["variance"],
                    values["within_variance"] + values["construction_variance"],
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    actual["score"],
                    actual["source_term"] + actual["evidence_term"],
                    atol=1e-10,
                )
                count += 1
    frame = pd.read_csv(out / "evaluation/predictions_with_targets.csv")
    for file, keys in (
        ("metrics.csv", ["instrument", "world", "truth_kind", "target", "arm"]),
        (
            "metrics_by_seed.csv",
            ["seed", "instrument", "world", "truth_kind", "target", "arm"],
        ),
    ):
        table = pd.read_csv(out / "evaluation" / file).set_index(keys)
        for key, group in frame.groupby(keys):
            row = table.loc[key]
            error = group.score.to_numpy() - group.truth.to_numpy()
            np.testing.assert_allclose(
                [row.rmse, row.bias],
                [np.sqrt(np.mean(error**2)), np.mean(error)],
                atol=1e-12,
            )
            if row.n != len(group):
                raise ValueError("Metric author count is wrong")
            if np.isfinite(group.variance).all():
                expected = np.mean(
                    abs(error) <= 1.959963984540054 * np.sqrt(group.variance.to_numpy())
                )
                np.testing.assert_allclose(row.coverage95, expected, atol=1e-12)
    ablation = read(out / "application/ablations.json")
    for case in ablation:
        original = case["original"]
        for key in ("exact_repeat", "record_order_only"):
            variant = case["variants"][key]
            if variant["qualified_events"] != original["qualified_events"]:
                raise ValueError("Repeated/reordered records changed effective events")
            for name in original["targets"]:
                np.testing.assert_allclose(
                    variant["targets"][name]["score"],
                    original["targets"][name]["score"],
                    atol=1e-10,
                )
        if case["variants"]["unqualified_scope"]["targets"]:
            raise ValueError("Unqualified evidence was turned into a score")
    cold = []
    for spec in cfg["instruments"]:
        folder = out / "cold" / spec["id"]
        meta = read(folder / "manifest.json")
        for name, expected in meta["files"].items():
            if sha(folder / name) != expected:
                raise ValueError("Changed cold output")
        if (
            meta["extra_denials"]
            or meta["cached_margins_read"]
            or meta["evaluation_truth_read"]
        ):
            raise ValueError("Cold application leaked cached/evaluation values")
        payload = read(out / "cold_inputs" / (spec["id"] + ".json"))
        bundle = read(out / "instruments" / f"{spec['id']}_{cfg['seeds'][0]}.json")
        reference = independent_score(payload["records"], margins, bundle)
        author = payload["records"][0]["author"]
        actual = read(folder / "predictions.json")[author]
        max_diff = max(
            abs(actual["targets"][name]["score"] - v["score"])
            for name, v in reference.items()
        )
        if max_diff > 1e-5:
            raise ValueError("Raw NLI and cached scoring differ beyond FP32 tolerance")
        for value in actual["targets"].values():
            np.testing.assert_allclose(
                value["score"],
                value["source_term"]
                + sum(e["contribution"] for e in value["evidence"]),
                atol=1e-10,
            )
        cold.append(
            {
                "instrument": spec["id"],
                "pairs": meta["pairs"],
                "max_score_difference": max_diff,
            }
        )
    result = {
        "status": "PASS",
        "independent_person_target_checks": count,
        "distinct_simulated_authors": len(participants),
        "max_precision_formula_error": max_error,
        "cold": cold,
        "independent_builder": False,
        "psychological_gold": False,
    }
    write(out / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        audit(args.output.resolve())
