#!/usr/bin/env python3
"""Independent information-form replay of frozen history scores and operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_history import (
    active_records,
    append_innovation,
    restore_summary,
    score_records,
)
from grain_research.evidence_history_innovation import square_root_append
from scripts.run_grain_evidence_history import DEFAULT, PREVIOUS, TARGETS
from scripts.run_grain_joint_conditional_observation import dump, read, sha


def replay(history, naive=False):
    """Alternative explicit state machine, not the production active-record call."""
    active, seen, all_rows = {}, {}, []
    for item in history:
        r = dict(item)
        kind, name = r["kind"], r["record_id"]
        if kind in ["new", "repeat", "revise"]:
            if kind == "revise":
                del active[r["parent_id"]]
            active[name] = r
            all_rows.append(r)
        elif kind == "retract":
            del active[r["parent_id"]]
        elif kind == "alias":
            all_rows.append(dict(seen[r["parent_id"]], record_id=name))
        seen[name] = r
    return all_rows if naive else list(active.values())


def independent_program(model, records, arm):
    """Information-form inverse of R, distinct from production total-K inverse."""
    z = np.array([r["condition"] for r in records], float)
    x = np.c_[np.ones(len(z)), z]
    c = np.asarray(model["components"])
    r = np.zeros((len(z), len(z)))
    for i, a in enumerate(records):
        for j, b in enumerate(records):
            r[i, j] = (
                c[2] * (a["block_id"] == b["block_id"])
                + c[3] * (a["event_id"] == b["event_id"])
                + c[4] * (a["record_id"] == b["record_id"])
            )
    t = np.eye(len(z))
    if arm == "all_rows":
        r = np.eye(len(z)) * c[2:].sum()
    elif arm == "event_only":
        names = list(dict.fromkeys(row["event_id"] for row in records))
        ids = np.array([row["event_id"] for row in records])
        t = np.array([(ids == name) / (ids == name).sum() for name in names])
        x = t @ x
        r = np.diag(np.diag(t @ r @ t.T))
    prior_information = np.diag(1 / c[:2])
    ri_x = solve(r, x, assume_a="pos")
    posterior_cov = np.linalg.inv(prior_information + x.T @ ri_x)
    w = posterior_cov @ ri_x.T @ t
    offset = posterior_cov @ prior_information @ model["mean"]
    return TARGETS @ w, TARGETS @ offset, TARGETS @ posterior_cov @ TARGETS.T


def witnesses():
    """Same present state, different future updates, and recoverable source summary."""
    model = {"mean": np.zeros(2), "components": [1, 1, 0.3, 0.5, 0.02]}
    records = [
        {
            "record_id": f"r{i}",
            "event_id": f"e{i}",
            "block_id": f"b{i // 2}",
            "condition": -1 if i < 2 else 1,
            "kind": "new",
            "value_index": i,
        }
        for i in range(4)
    ]
    a, b = np.array([[-1, 1, -1, 1.0]]), np.array([[1, -1, -1, 1.0]])
    initial_a, initial_b = (
        score_records(model, a, records),
        score_records(model, b, records),
    )
    np.testing.assert_allclose(initial_a["mean"], initial_b["mean"], atol=1e-12)
    replacement = dict(
        records[0], record_id="fix", kind="revise", parent_id="r0", value_index=4
    )
    summary = {
        "format": "versioned_numeric_evidence_v1",
        "history": records,
        "values": {str(i): float(a[0, i]) for i in range(4)},
    }
    restored = restore_summary(summary)
    active = active_records(restored["history"] + [replacement])["records"]
    av, bv = np.c_[a, 1], np.c_[b, 1]
    index = [r["value_index"] for r in active]
    after_a = score_records(model, av[:, index], active)
    after_b = score_records(model, bv[:, index], active)
    retained = dict(summary["values"], **{"4": 1.0})
    from_summary = score_records(model, [[retained[str(i)] for i in index]], active)
    np.testing.assert_allclose(after_a["mean"], from_summary["mean"])
    separation = float(np.linalg.norm(after_a["mean"] - after_b["mean"]))
    assert separation > 0.3
    fresh_same = dict(records[0], record_id="new", event_id="e4")
    fresh_independent = dict(fresh_same, block_id="b_new")
    dependent = append_innovation(model, a, records, [1], fresh_same)
    independent = append_innovation(model, a, records, [1], fresh_independent)
    duplicate = dict(records[0], record_id="alias", origin_id="r0")
    copied = append_innovation(model, a, records, a[:, 0], duplicate)
    np.testing.assert_allclose(copied["mean"], initial_a["mean"], atol=1e-12)
    # Scalar-only summaries are indistinguishable before the same correction;
    # at least one squared Euclidean error must be >= separation^2 / 4.
    return {
        "current_mean": initial_a["mean"],
        "current_covariance": initial_a["covariance"],
        "revision_A": after_a["mean"],
        "revision_B": after_b["mean"],
        "scalar_summary_minimax_squared_error_lower_bound": separation**2 / 4,
        "source_summary_exact": True,
        "same_block_new_event_mean": dependent["mean"],
        "new_block_new_event_mean": independent["mean"],
        "same_block_innovation_variance": dependent["innovation_variance"],
        "new_block_innovation_variance": independent["innovation_variance"],
        "duplicate_gain": copied["gain"],
        "working_gaussian_example_not_human_data": True,
    }


def main(out):
    """Validate every source fit, stream program, prediction and metric separately."""
    repair_path = out / "evaluate/square_root_repair_freeze.json"
    repair_files = [
        ROOT / "grain_research/evidence_history_innovation.py",
        Path(__file__).resolve(),
    ]
    if not repair_path.exists():
        dump(
            repair_path,
            {
                "scope": "Post-result numerical repair; original r1 batch programs unchanged",
                "original_audit": read(out / "evaluate/independent_audit.json"),
                "sha256": {str(path): sha(path) for path in repair_files},
            },
        )
    for path, digest in read(repair_path)["sha256"].items():
        assert sha(path) == digest, path
    for path, digest in read(out / "freeze.json").items():
        assert sha(path) == digest, path
    for folder in ["generate", "apply"]:
        for name, digest in read(out / folder / "seal.json").items():
            assert sha(out / folder / name) == digest
    assert sha(out / "fit/models.json") == read(out / "fit/seal.json")["models_sha256"]
    cfg, schedule = read(out / "config.json"), read(out / "schedule.json")
    models, programs = read(out / "fit/models.json"), read(out / "apply/programs.json")
    metrics = pd.read_csv(out / "evaluate/metrics_by_seed.csv")
    checks, worst, innovation_checks, values_checked = 0, 0.0, 0, 0
    innovation_worst, innovation_tolerance_exceedances = 0.0, 0
    root_worst, root_covariance_worst, metric_checks = 0.0, 0.0, 0
    with np.load(out / "generate/source.npz", allow_pickle=False) as source:
        records = schedule["source"]
        z = np.array([r["condition"] for r in records])
        basis = np.array(
            [
                [[1.0 for b in records] for a in records],
                np.outer(z, z),
                [[a["block_id"] == b["block_id"] for b in records] for a in records],
                [[a["event_id"] == b["event_id"] for b in records] for a in records],
                np.eye(len(records)),
            ],
            float,
        )
        design = basis.reshape(5, -1).T
        bank = pd.read_csv(PREVIOUS / "public_bank.csv")
        for seed in cfg["seeds"]:
            model, y = models[str(seed)], source[f"s{seed}_values"]
            assert bank.iloc[source[f"s{seed}_cases"].ravel()].source.all()
            empirical = np.cov(y, rowvar=False)
            residual = design @ model["raw_components"] - empirical.ravel()
            gradient = design.T @ residual
            assert np.max(np.abs(gradient)) < 1e-8
            mean = np.linalg.solve(
                np.c_[np.ones(len(z)), z].T @ np.c_[np.ones(len(z)), z],
                np.c_[np.ones(len(z)), z].T @ y.mean(axis=0),
            )
            np.testing.assert_allclose(mean, model["mean"], atol=1e-12)
            checks += 1
    with (
        np.load(out / "generate/evidence.npz", allow_pickle=False) as evidence,
        np.load(out / "apply/predictions.npz", allow_pickle=False) as pred,
        np.load(out / "generate/truth.npz", allow_pickle=False) as truth,
    ):
        for key, p in programs.items():
            seed, step_word, arm = key.split("_", 2)
            step = int(step_word[4:])
            part = schedule["stream"][:step]
            records = replay(part, naive=arm == "all_rows")
            assert [r["record_id"] for r in records] == p["record_ids"]
            w, offset, covariance = independent_program(models[seed[1:]], records, arm)
            np.testing.assert_allclose(w, p["weights"], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(covariance, p["covariance"], atol=1e-7)
            for style in ["source_style", "held_style"]:
                y = evidence[f"{seed}_{style}_values"][
                    :, [r["value_index"] for r in records]
                ]
                value = y @ w.T + offset
                actual_saved = pred[f"{seed}_{style}_{step_word}_{arm}"]
                error = np.max(np.abs(value - actual_saved))
                worst = max(worst, float(error))
                np.testing.assert_allclose(value, actual_saved, atol=1e-7, rtol=1e-6)
                actual = truth[f"{seed}_beta"] @ TARGETS.T
                reported = metrics[
                    (metrics.seed == int(seed[1:]))
                    & (metrics.step == step)
                    & (metrics.arm == arm)
                    & (metrics["style"] == style)
                ]
                for j, target in enumerate(cfg["source_targets"]):
                    error = value[:, j] - actual[:, j]
                    stats = {
                        "mse": np.mean(error**2),
                        "bias": np.mean(error),
                        "correlation": np.corrcoef(value[:, j], actual[:, j])[0, 1],
                        "working_variance": covariance[j, j],
                        "coverage": np.mean(
                            np.abs(error) <= norm.ppf(0.975) * np.sqrt(covariance[j, j])
                        ),
                    }
                    row = reported[reported.target == target].iloc[0]
                    for name, expected in stats.items():
                        np.testing.assert_allclose(expected, row[name], atol=1e-7)
                        metric_checks += 1
                values_checked += value.size
                if (
                    arm == "versioned"
                    and step > 1
                    and part[-1]["kind"] in ["new", "repeat"]
                ):
                    old = replay(part[:-1])
                    raw = evidence[f"{seed}_{style}_values"]
                    innovation = append_innovation(
                        models[seed[1:]],
                        raw[:, [r["value_index"] for r in old]],
                        old,
                        raw[:, part[-1]["value_index"]],
                        part[-1],
                    )
                    incremental = innovation["mean"] @ TARGETS.T
                    innovation_worst = max(
                        innovation_worst, float(np.max(np.abs(incremental - value)))
                    )
                    innovation_tolerance_exceedances += int(
                        not np.allclose(incremental, value, atol=1e-7, rtol=1e-6)
                    )
                    root = square_root_append(
                        models[seed[1:]],
                        raw[:, [r["value_index"] for r in old]],
                        old,
                        raw[:, part[-1]["value_index"]],
                        part[-1],
                    )
                    root_value = root["mean"] @ TARGETS.T
                    root_covariance = TARGETS @ root["covariance"] @ TARGETS.T
                    root_worst = max(
                        root_worst, float(np.max(np.abs(root_value - value)))
                    )
                    root_covariance_worst = max(
                        root_covariance_worst,
                        float(np.max(np.abs(root_covariance - covariance))),
                    )
                    np.testing.assert_allclose(root_value, value, atol=1e-7, rtol=1e-6)
                    np.testing.assert_allclose(root_covariance, covariance, atol=1e-9)
                    innovation_checks += 1
            checks += 1
    for name in ["fit", "apply"]:
        denied = read(out / name / "guard.json")["denied"]
        assert str(out / "generate/truth.npz") in denied
    result = {
        "status": "PASS_WITH_REPAIRED_INCREMENTAL"
        if innovation_tolerance_exceedances
        else "PASS",
        "programs": len(programs),
        "source_fits": len(models),
        "equation_checks": checks,
        "conditional_innovation_checks": innovation_checks,
        "prediction_values": values_checked,
        "metric_checks": metric_checks,
        "worst_prediction_error": worst,
        "incremental_max_error": innovation_worst,
        "incremental_tolerance_exceedances": innovation_tolerance_exceedances,
        "square_root_prediction_max_error": root_worst,
        "square_root_covariance_max_error": root_covariance_worst,
        "incremental_note": "Frozen application uses full conditioning, not the cancellation-sensitive incremental helper; keep deviations visible rather than relax the original tolerance",
        "same_agent_independent_formula_not_D02": True,
    }
    dump(out / "evaluate/independent_audit.json", result)
    dump(out / "evaluate/update_witnesses.json", witnesses())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    main(parser.parse_args().output.resolve())
