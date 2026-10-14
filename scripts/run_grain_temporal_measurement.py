#!/usr/bin/env python3
"""Finite M04/M08/M09 integration, with inherited text observations and no training."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.temporal_measurement import (
    interaction_pairs,
    posterior,
    score_target,
    state_covariance,
    target_vector,
)
from scripts.run_grain_query_moment_operator import read, sha, write

BASE = ROOT / "results/grain_vector_mechanisms"
BANK = BASE / "mechanism_todo_v1/condition_accumulation_r1/public_scored_bank.csv"
COMMENTS = (
    BASE / "dynamic_path_channels_v1/within_comment/private_comment_descriptors.csv"
)
VIEWS = BASE / "vm05_qualification_r2/common_views.json"
TURNS = BASE / "vm06_meps_windows_r3/private_turn_integrity.csv"
TARGETS = (
    "level",
    "condition",
    "partner",
    "current_state",
    "current_response",
    "window_state",
    "window_response",
    "conditional_profile",
)


def prepare(out, config_path):
    """Freeze all construction rules before simulation or natural application."""
    config = read(config_path)
    if out.exists():
        raise FileExistsError("Fresh output directory required")
    prior = read(TURNS.parent / "summary.json")
    if sha(TURNS) != prior["outputs_sha256"][TURNS.name]:
        raise ValueError("MEPS metadata changed")
    bank = pd.read_csv(BANK)
    if set(bank.partition) != {"source", "held"}:
        raise ValueError("Frozen source and held wording required")
    files = [
        Path(__file__),
        ROOT / "grain_research/temporal_measurement.py",
        ROOT / "docs/GRAIN_TEMPORAL_MEASUREMENT_PACKET_V1.md",
        config_path,
        BANK,
        COMMENTS,
        VIEWS,
        TURNS,
        TURNS.parent / "summary.json",
    ]
    out.mkdir(parents=True)
    write(out / "config.json", config)
    write(
        out / "contract.json",
        {
            "version": config["version"],
            "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "config_sha256": sha(out / "config.json"),
            "queries": sorted(bank["query"].unique().tolist()),
            "fitting": "No new fit; inherited source text reader, declared synthetic state prior",
            "human_labels": False,
            "new_neural_calls": 0,
        },
    )


def verify(out):
    """Verify frozen code, source reader and metadata, with no outcomes used."""
    contract = read(out / "contract.json")
    for name, expected in contract["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed bound input: " + name)
    if sha(out / "config.json") != contract["config_sha256"]:
        raise ValueError("Configuration changed")
    return read(out / "config.json"), contract


def clocks(world, n):
    """Different actual sampling clocks with the same number of observations."""
    if world == "regular":
        gaps = np.ones(n - 1)
    elif world == "clustered":
        gaps = np.where(np.arange(n - 1) % 8 == 7, 15.0, 0.08)
    else:
        gaps = np.resize(np.array([0.15, 0.6, 1.5, 4.0]), n - 1)
    return np.r_[0, np.cumsum(gaps)]


def simulate_panel(config, bank, rng, world, beta, latent, z, modality):
    """Use known states to generate observations; never pass truth to the scorer."""
    n, queries = len(z), sorted(bank["query"].unique())
    y, mu, zh, u = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n)
    clipped, case_rows = 0, []
    groups = {
        (q, c, s): g
        for (q, c, s), g in bank[bank.partition.eq("held")].groupby(
            ["query", "condition", "sign"]
        )
    }
    for i in range(n):
        if world == "constant_partner":
            u[i] = 1.0
        elif world == "feedback_partner":
            u[i] = 0.5 + 0.6 * (y[i - 1].mean() if i else 0) + rng.normal(0, 0.3)
        elif world == "shared_shock":
            u[i] = 0.7 * latent[i, 0] / np.sqrt(
                config["stationary"][0][0]
            ) + rng.normal(0, 0.2)
        else:
            u[i] = rng.choice([-1.0, 1.0])
        for k, query in enumerate(queries):
            mu[i, k] = beta[3 * k : 3 * k + 3] @ [1, z[i, k], u[i]] + latent[i, k]
            if modality == "gaussian":
                y[i, k] = mu[i, k] + rng.normal(
                    0, 1 / np.sqrt(config["events_per_occasion"])
                )
                zh[i, k] = z[i, k]
            else:
                bounded = np.clip(mu[i, k], -0.95, 0.95)
                clipped += int(bounded != mu[i, k])
                values, conditions = [], []
                for event in range(config["events_per_occasion"]):
                    sign = 1 if rng.random() < (bounded + 1) / 2 else -1
                    pool = groups[(query, z[i, k], sign)]
                    row = pool.iloc[int(rng.integers(len(pool)))]
                    values.append(row.evidence_hat)
                    conditions.append(row.condition_hat)
                    case_rows.append(
                        {
                            "occasion": i,
                            "channel": k,
                            "event": event,
                            "case_id": row.case_id,
                        }
                    )
                y[i, k], zh[i, k] = np.mean(values), np.mean(conditions)
    return y, zh, u, mu, clipped, case_rows


def model_parameters(config, arm):
    """Only the named mechanism is changed between fixed reference arms."""
    a = np.asarray(config["drift"], float)
    if arm == "independent_state":
        a = -1e6 * np.eye(2)
    return {
        "drift": a,
        "stationary": np.asarray(config["stationary"]),
        "noise": np.eye(2) / config["events_per_occasion"],
        "profile_mean": np.asarray(config["profile_mean"]),
        "profile_cov": np.diag(np.square(config["profile_sd"])),
    }


def simulation(out):
    """Score every fixed world and preserve failures rather than select winners."""
    config, _ = verify(out)
    if (out / "simulation_summary.json").exists():
        print("Simulation complete; no repeat", flush=True)
        return
    start, rows, kernel_rows, observed_rows, case_rows = (
        time.process_time(),
        [],
        [],
        [],
        [],
    )
    bank = pd.read_csv(BANK)
    clipped, total, largest_kernel_error, panels = 0, 0, 0.0, 0
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        for world in config["worlds"]:
            t = clocks(world, config["occasions"])
            covariance = state_covariance(t, config["drift"], config["stationary"])
            root = np.linalg.cholesky(covariance)
            for author in range(config["authors_per_seed"]):
                beta = (
                    np.asarray(config["profile_mean"])
                    + rng.normal(size=6) * config["profile_sd"]
                )
                latent = (root @ rng.normal(size=2 * len(t))).reshape(-1, 2)
                z = rng.choice([-1.0, 1.0], size=(len(t), 2))
                for modality in ("gaussian", "decoded_text"):
                    panel_id = f"{seed}:{world}:{author}:{modality}"
                    y, zh, u, mu, cuts, cases = simulate_panel(
                        config, bank, rng, world, beta, latent, z, modality
                    )
                    clipped += cuts
                    total += len(t) * 2 if modality == "decoded_text" else 0
                    case_rows.extend({"panel": panel_id, **case} for case in cases)
                    for i in range(len(t)):
                        row = {
                            "panel": panel_id,
                            "seed": seed,
                            "world": world,
                            "author": author,
                            "modality": modality,
                            "occasion": i,
                            "time": t[i],
                            "partner": u[i],
                        }
                        for k in range(2):
                            row.update(
                                {
                                    f"y{k}": y[i, k],
                                    f"z{k}": z[i, k],
                                    f"zh{k}": zh[i, k],
                                    f"state{k}": latent[i, k],
                                    f"mu{k}": mu[i, k],
                                }
                            )
                        row.update({f"beta{k}": beta[k] for k in range(6)})
                        observed_rows.append(row)
                    for n in config["prefixes"]:
                        for arm in config["arms"]:
                            tt = (
                                np.arange(n, dtype=float)
                                if arm == "order_clock"
                                else t[:n]
                            )
                            uu = np.zeros(n) if arm == "no_partner" else u[:n]
                            fit = posterior(
                                tt, y[:n], zh[:n], uu, **model_parameters(config, arm)
                            )
                            truth = np.r_[beta, latent[:n].ravel()]
                            for k in range(2):
                                for target in TARGETS:
                                    zz = (
                                        zh[n - 1, k]
                                        if target == "current_response"
                                        else 0.5
                                        if target == "conditional_profile"
                                        else 0
                                    )
                                    pu = uu[-1] if target == "current_response" else 0
                                    w = target_vector(fit, target, k, z=zz, partner=pu)
                                    score = score_target(fit, w)
                                    # Evaluate every arm against the SAME actual-time and condition target.
                                    true_fit = {**fit, "times": t[:n] - t[0]}
                                    true_z = (
                                        z[n - 1, k]
                                        if target == "current_response"
                                        else zz
                                    )
                                    true_u = (
                                        u[n - 1] if target == "current_response" else 0
                                    )
                                    wt = target_vector(
                                        true_fit, target, k, z=true_z, partner=true_u
                                    )
                                    truth_value = float(wt @ truth)
                                    error = abs(
                                        score["offset"]
                                        + score["kernel"] @ y[:n].ravel()
                                        - score["score"]
                                    )
                                    largest_kernel_error = max(
                                        largest_kernel_error, error
                                    )
                                    rows.append(
                                        {
                                            "panel": panel_id,
                                            "seed": seed,
                                            "world": world,
                                            "modality": modality,
                                            "arm": arm,
                                            "prefix": n,
                                            "channel": k,
                                            "target": target,
                                            "score": score["score"],
                                            "truth": truth_value,
                                            "lower": score["lower"],
                                            "upper": score["upper"],
                                            "variance": score["variance"],
                                        }
                                    )
                                    if (
                                        seed == config["seeds"][0]
                                        and author == 0
                                        and n == config["occasions"]
                                        and arm == "joint"
                                    ):
                                        for j, value in enumerate(score["kernel"]):
                                            kernel_rows.append(
                                                {
                                                    "panel": panel_id,
                                                    "target": target,
                                                    "channel": k,
                                                    "evidence_channel": j % 2,
                                                    "occasion": j // 2,
                                                    "lag": t[n - 1] - t[j // 2],
                                                    "weight": value,
                                                }
                                            )
                        for k in range(2):
                            for arm, value in (
                                ("raw_current", y[n - 1, k]),
                                ("history_mean", y[:n, k].mean()),
                            ):
                                for target, truth_value in (
                                    ("current_response", mu[n - 1, k]),
                                    ("level", beta[3 * k]),
                                ):
                                    rows.append(
                                        {
                                            "panel": panel_id,
                                            "seed": seed,
                                            "world": world,
                                            "modality": modality,
                                            "arm": arm,
                                            "prefix": n,
                                            "channel": k,
                                            "target": target,
                                            "score": value,
                                            "truth": truth_value,
                                            "lower": np.nan,
                                            "upper": np.nan,
                                            "variance": np.nan,
                                        }
                                    )
                    panels += 1
                    if time.process_time() - start > config["max_cpu_seconds"]:
                        raise TimeoutError("Finite simulation CPU budget exceeded")
            print(
                {
                    "seed": seed,
                    "world": world,
                    "panels": panels,
                    "cpu_seconds": round(time.process_time() - start, 2),
                },
                flush=True,
            )
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(observed_rows).to_csv(out / "simulation_observations.csv", index=False)
    pd.DataFrame(case_rows).to_csv(out / "public_event_addresses.csv", index=False)
    pd.DataFrame(kernel_rows).to_csv(out / "kernels.csv", index=False)
    frame = pd.DataFrame(rows)
    groups = ["world", "modality", "arm", "prefix", "channel", "target"]
    metric_rows = []
    for keys, group in frame.groupby(groups):
        error = group.score - group.truth
        finite = group.lower.notna()
        metric_rows.append(
            {
                **dict(zip(groups, keys)),
                "n": len(group),
                "rmse": np.sqrt(np.mean(error**2)),
                "bias": error.mean(),
                "pearson": np.corrcoef(group.score, group.truth)[0, 1]
                if group.score.std() > 1e-12
                else np.nan,
                "coverage": (
                    (group.truth[finite] >= group.lower[finite])
                    & (group.truth[finite] <= group.upper[finite])
                ).mean(),
                "mean_variance": group.variance.mean(),
            }
        )
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    write(
        out / "simulation_summary.json",
        {
            "status": "complete",
            "panels": panels,
            "prediction_rows": len(rows),
            "clipped_text_means": clipped,
            "text_mean_count": total,
            "max_kernel_error": largest_kernel_error,
            "cpu_seconds": time.process_time() - start,
            "new_neural_calls": 0,
            "files": {
                p: sha(out / p)
                for p in (
                    "predictions.csv",
                    "simulation_observations.csv",
                    "public_event_addresses.csv",
                    "kernels.csv",
                    "metrics.csv",
                )
            },
        },
    )


def natural(out):
    """Apply a declared working time kernel; MEPS remains metadata validation only."""
    config, _ = verify(out)
    if (out / "natural_summary.json").exists():
        print("Natural application complete; no repeat", flush=True)
        return
    comments = pd.read_csv(COMMENTS)
    wanted = set(comments.author_id)
    metadata = {}
    for author in read(VIEWS):
        if author["author_id"] in wanted:
            for v, view in enumerate(author["views"]):
                for j, comment in enumerate(view):
                    metadata[(author["author_id"], v, j)] = (
                        float(comment["timestamp"]) / 86400
                    )
    rows, rejected, directions = [], [], []
    rate = np.log(2) / config["natural_working_state_half_life_days"]
    for uid, group in comments.groupby("author_id"):
        group = group.sort_values(["view", "comment"])
        ts = np.array([metadata[(uid, r.view, r.comment)] for r in group.itertuples()])
        if np.any(np.diff(ts) <= 0):
            rejected.append({"author_id": uid, "reason": "non_strict_observed_times"})
            continue
        y = group[["level::whole:F01", "level::whole:F02"]].to_numpy(float)
        params = {
            "drift": -rate * np.eye(2),
            "stationary": np.eye(2) * 0.5,
            "noise": np.eye(2) * 0.5,
            "profile_mean": np.zeros(6),
            "profile_cov": np.diag([1, 0.1, 0.1] * 2),
        }
        for arm, clock in (
            ("actual_days", ts - ts[0]),
            ("comment_index", np.arange(len(ts))),
        ):
            fit = posterior(clock, y, np.zeros_like(y), np.zeros(len(y)), **params)
            for k in range(2):
                for target in (
                    "level",
                    "current_state",
                    "current_response",
                    "window_state",
                ):
                    score = score_target(fit, target_vector(fit, target, k))
                    weights = score.pop("kernel").reshape(-1, 2)[:, k]
                    rows.append(
                        {
                            "author_id": uid,
                            "arm": arm,
                            "channel": k,
                            "target": target,
                            "comments": len(y),
                            "span_days": ts[-1] - ts[0],
                            "kernel_l1": abs(weights).sum(),
                            **score,
                        }
                    )
        directions.append(
            {
                "author_id": uid,
                "comments": len(y),
                "span_days": ts[-1] - ts[0],
                "median_gap_days": np.median(np.diff(ts)),
            }
        )
    pd.DataFrame(rows).to_csv(out / "private_natural_scores.csv", index=False)
    pd.DataFrame(rejected, columns=["author_id", "reason"]).to_csv(
        out / "private_time_rejections.csv", index=False
    )
    pd.DataFrame(directions).to_csv(out / "private_natural_clocks.csv", index=False)
    turns = pd.read_csv(TURNS)
    pairs, coverage = [], []
    for key, group in turns.groupby(["participant_code", "view_kind", "condition"]):
        paired = interaction_pairs(group.to_dict("records"))
        coverage.append(
            {
                "participant_code": key[0],
                "view_kind": key[1],
                "condition": key[2],
                "turns": len(group),
                "pairs": len(paired),
                "user_turns": int(group.role.eq("user").sum()),
            }
        )
        for left, right in paired:
            pairs.append(
                {
                    "participant_code": key[0],
                    "view_kind": key[1],
                    "condition": key[2],
                    "assistant_sequence": left["turn_sequence"],
                    "user_sequence": right["turn_sequence"],
                    "assistant_chars": left["source_chars"],
                    "user_chars": right["source_chars"],
                }
            )
    pd.DataFrame(pairs).to_csv(out / "private_meps_pairs.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "private_meps_pair_coverage.csv", index=False)
    aggregate = (
        pd.DataFrame(coverage)
        .groupby("view_kind")
        .agg(
            views=("pairs", "size"),
            participants=("participant_code", "nunique"),
            pairs=("pairs", "sum"),
            views_with_3pairs=("pairs", lambda x: int((x >= 3).sum())),
            user_turns=("user_turns", "sum"),
        )
    )
    aggregate.to_csv(out / "meps_pair_coverage.csv")
    natural_scores = pd.DataFrame(rows)
    comparison = natural_scores.pivot(
        index=["author_id", "channel", "target"], columns="arm", values="score"
    )
    comparison["difference"] = comparison.actual_days - comparison.comment_index
    comparison.reset_index().groupby(["channel", "target"]).agg(
        authors=("author_id", "nunique"),
        mean_difference=("difference", "mean"),
        mean_absolute_difference=("difference", lambda x: abs(x).mean()),
    ).to_csv(out / "natural_clock_comparison.csv")
    write(
        out / "natural_summary.json",
        {
            "status": "complete",
            "scored_authors": len(directions),
            "rejected_authors": len(rejected),
            "input_authors": len(wanted),
            "meps_pairs": len(pairs),
            "meps_labels_read": False,
            "meps_fit": False,
            "new_neural_calls": 0,
            "psychological_time_parameters_calibrated": False,
            "scope": config["natural_scope"],
            "files": {
                p.name: sha(p)
                for p in out.glob("*.csv")
                if ("natural" in p.name or "meps" in p.name or "rejections" in p.name)
            },
        },
    )
    print({"natural_authors": len(directions), "meps_pairs": len(pairs)}, flush=True)


def main():
    """Prepare once, run one fixed simulation and validate metadata locally."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "simulate", "natural"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_temporal_measurement.json"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == "prepare":
            prepare(args.output, args.config)
        elif args.stage == "simulate":
            simulation(args.output)
        else:
            natural(args.output)


if __name__ == "__main__":
    main()
