#!/usr/bin/env python3
"""Run bounded M03/M04/M10 mechanisms using an immutable text-evidence bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.choice_design import (
    anchored_cells,
    choice_coordinates,
    choose_allocation,
    learn_cell_moments,
    linear_cell_score,
)

TARGETS = {
    "level": np.array([0.5, 0.5]),
    "response": np.array([-0.5, 0.5]),
    "positive_condition": np.array([0.0, 1.0]),
}


def jsonable(value):
    """Convert nested numeric artifacts to portable JSON values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(path):
    """Hash a small source or evidence artifact before/after execution."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_evidence(bank, condition, signs, rng):
    """Sample fixed actual NLI readings at specified simulated event directions."""
    z, signs = np.asarray(condition), np.asarray(signs)
    evidence, addresses = np.empty(z.shape), np.empty(z.shape, dtype=object)
    for cell in (-1, 1):
        for sign in (-1, 1):
            mask = (z == cell) & (signs == sign)
            pool = bank[(bank.condition == cell) & (bank.sign == sign)]
            if pool.empty:
                raise ValueError("Incomplete frozen evidence bank")
            selected = pool.iloc[rng.integers(0, len(pool), size=int(mask.sum()))]
            evidence[mask] = selected.evidence_hat.to_numpy()
            addresses[mask] = selected.case_id.to_numpy()
    return evidence, addresses


def reader_moments(bank, a_bound, b_bound):
    """Analytic population moments, used ONLY in oracle/evaluation arms.

    Uniform independent a/b generate signed Bernoulli events. Each outcome selects
    one of the fixed bank scores uniformly. No held-author sample statistics enter.
    """
    alpha, beta, second = [], [], []
    for cell in (-1, 1):
        minus = bank[
            (bank.condition == cell) & (bank.sign == -1)
        ].evidence_hat.to_numpy()
        plus = bank[(bank.condition == cell) & (bank.sign == 1)].evidence_hat.to_numpy()
        alpha.append((minus.mean() + plus.mean()) / 2)
        beta.append((plus.mean() - minus.mean()) / 2)
        second.append(((minus**2).mean() + (plus**2).mean()) / 2)
    alpha, beta, second = map(np.asarray, (alpha, beta, second))
    va, vb = a_bound**2 / 3, b_bound**2 / 3
    latent = np.array([[va + vb, va - vb], [va - vb, va + vb]])
    covariance = latent * np.outer(beta, beta)
    return {
        "mean": alpha,
        "covariance": covariance,
        "noise": second - alpha**2 - np.diag(covariance),
        "reader_gain": beta,
    }


def generate_responses(rng, bank, authors, events, a_bound=0.45, b_bound=0.15):
    """Make independent fixed-condition events and return truth separately."""
    a = rng.uniform(-a_bound, a_bound, authors)
    b = rng.uniform(-b_bound, b_bound, authors)
    latent_cells = a[:, None] + b[:, None] * np.array([-1, 1])
    z = np.broadcast_to(np.array([-1, 1])[None, :, None], (authors, 2, events))
    sign = np.where(rng.random(z.shape) < (1 + latent_cells[..., None]) / 2, 1, -1)
    evidence, addresses = draw_evidence(bank, z, sign, rng)
    return evidence, latent_cells, addresses


def metric_table(frame, keys):
    """Report both latent numeric and fixed-reader truths, never human validity."""
    rows = []
    for group, part in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, group if isinstance(group, tuple) else (group,)))
        row["scheduled"] = len(part)
        valid = part[np.isfinite(part.prediction)]
        row["n"] = len(valid)
        for truth_name in ("truth", "reader_truth"):
            pair = valid[np.isfinite(valid[truth_name])]
            p, t = pair.prediction.to_numpy(), pair[truth_name].to_numpy()
            row[truth_name + "_r"] = (
                np.corrcoef(p, t)[0, 1]
                if len(p) > 2 and min(np.std(p), np.std(t)) > 1e-12
                else np.nan
            )
            row[truth_name + "_rmse"] = (
                np.sqrt(np.mean((p - t) ** 2)) if len(p) else np.nan
            )
            row[truth_name + "_bias"] = np.mean(p - t) if len(p) else np.nan
        if "variance" in valid and len(valid):
            radius = 1.96 * np.sqrt(valid.variance)
            row["working_reader_coverage"] = np.mean(
                abs(valid.prediction - valid.reader_truth) <= radius
            )
            row["mean_working_variance"] = valid.variance.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def choice_study(config, bank):
    """Compare equal-budget free, assigned and two-phase observations."""
    rows, samples, support = [], [], []
    for qi, query in enumerate(config["queries"]):
        held = bank[(bank["query"] == query) & (bank.partition == "held")]
        reader = reader_moments(held, 0.3, 0.3)
        for wi, world in enumerate(config["choice_worlds"]):
            for seed in config["seeds"]:
                rng = np.random.default_rng(np.random.SeedSequence([seed, qi, wi, 31]))
                for author in range(config["held_authors"]):
                    gamma = rng.normal(0, 1, 3)
                    offset = rng.normal(0, 0.8, 3)
                    if world == "opportunity_confounded":
                        offset += 1.5 * gamma
                    a, b = rng.uniform(-0.3, 0.3, 2)
                    if world == "shared_response":
                        b = 0.3 * np.tanh(gamma[0])
                    state_strength = 0.3 if world == "state_selected" else 0.0
                    budget = config["observation_budget"]
                    state = rng.choice([-1, 1], budget)
                    probability = expit(
                        offset[0] + gamma[0] + (2.0 * state if state_strength else 0)
                    )
                    free_z = np.where(rng.random(budget) < probability, 1, -1)
                    nodes = np.where(free_z == -1, 1, 2)
                    free_tail = np.where(
                        rng.random(budget) < expit(offset[nodes] + gamma[nodes]), 1, -1
                    )
                    free_expected = a + b * free_z + state_strength * state
                    free_sign = np.where(
                        rng.random(budget) < (1 + free_expected) / 2, 1, -1
                    )
                    free_y, free_addr = draw_evidence(held, free_z, free_sign, rng)
                    fixed_z = np.tile([-1, 1], budget // 2)
                    fixed_state = rng.choice([-1, 1], budget)
                    fixed_expected = a + b * fixed_z + state_strength * fixed_state
                    fixed_sign = np.where(
                        rng.random(budget) < (1 + fixed_expected) / 2, 1, -1
                    )
                    fixed_y, fixed_addr = draw_evidence(held, fixed_z, fixed_sign, rng)
                    truth_cells = a + b * np.array([-1, 1])
                    read_cells = reader["mean"] + reader["reader_gain"] * truth_cells
                    header = {
                        "query": query,
                        "world": world,
                        "seed": seed,
                        "author": author,
                    }
                    for design in ("free", "fixed", "two_phase"):
                        nf = (
                            budget
                            if design == "free"
                            else (0 if design == "fixed" else budget // 2)
                        )
                        nx = budget - nf
                        z = np.r_[free_z[:nf], fixed_z[:nx]]
                        y = np.r_[free_y[:nf], fixed_y[:nx]]
                        addr = np.r_[free_addr[:nf], fixed_addr[:nx]]
                        h = {**header, "design": design}
                        if author < 2:
                            samples.extend(
                                {
                                    **h,
                                    "event": i,
                                    "phase": "free" if i < nf else "fixed",
                                    "condition": int(cell),
                                    "evidence": value,
                                    "case_id": address,
                                }
                                for i, (cell, value, address) in enumerate(
                                    zip(z, y, addr)
                                )
                            )
                        for known in (False, True):
                            score, count = choice_coordinates(
                                free_z[:nf], free_tail[:nf], offset if known else None
                            )
                            for j, name in enumerate(
                                ("choice_root", "choice_tail_minus", "choice_tail_plus")
                            ):
                                rows.append(
                                    {
                                        **h,
                                        "estimator": "known_opportunity"
                                        if known
                                        else "observed_choice",
                                        "target": name,
                                        "prediction": score[j],
                                        "truth": gamma[j],
                                        "reader_truth": np.nan,
                                        "observations": int(count[j]),
                                    }
                                )
                        support.append(
                            {
                                **h,
                                "free_events": nf,
                                "fixed_events": nx,
                                "negative_cell": int((z == -1).sum()),
                                "positive_cell": int((z == 1).sum()),
                            }
                        )
                        rows.append(
                            {
                                **h,
                                "estimator": "pooled",
                                "target": "level",
                                "prediction": y.mean(),
                                "truth": a,
                                "reader_truth": read_cells.mean(),
                                "observations": len(y),
                            }
                        )
                        arms = {"all_conditional": (z, y)}
                        if design == "two_phase":
                            arms["assigned_only"] = (fixed_z[:nx], fixed_y[:nx])
                        for estimator, (az, ay) in arms.items():
                            score, _ = anchored_cells(az, ay)
                            for j, name in enumerate(("level", "response")):
                                rows.append(
                                    {
                                        **h,
                                        "estimator": estimator,
                                        "target": name,
                                        "prediction": score[j],
                                        "truth": (a, b)[j],
                                        "reader_truth": TARGETS[name] @ read_cells,
                                        "observations": len(ay),
                                    }
                                )
    return pd.DataFrame(rows), pd.DataFrame(samples), pd.DataFrame(support)


def observations(pool, counts, *, repeat=False, naive=False):
    """Extract common random events; repeated copies keep event identity unless naive."""
    z = np.r_[np.full(counts[0], -1), np.full(counts[1], 1)]
    y = np.r_[pool[0, : counts[0]], pool[1, : counts[1]]]
    ids = np.array([f"{cell}:{i}" for cell, n in enumerate(counts) for i in range(n)])
    if repeat:
        z, y, ids = np.tile(z, 2), np.tile(y, 2), np.tile(ids, 2)
    if naive:
        ids = np.arange(len(z)).astype(str)
    return z, y, ids


def design_study(config, bank, output):
    """Learn on source only, then compare designs on new simulated authors."""
    rows, risks, plans, fits = [], [], [], {}
    for qi, query in enumerate(config["queries"]):
        source = bank[(bank["query"] == query) & (bank.partition == "source")]
        for seed in config["seeds"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, qi, 71]))
            pilot, _, _ = generate_responses(
                rng, source, config["source_authors"], config["source_events_per_cell"]
            )
            source_moments = learn_cell_moments(pilot)
            key = f"{query}_{seed}"
            np.save(output / f"pilot_{key}.npy", pilot)
            fits[key] = source_moments
            for wi, world in enumerate(config["design_worlds"]):
                rng = np.random.default_rng(np.random.SeedSequence([seed, qi, wi, 72]))
                partition = "source" if world == "matched_reader" else "held"
                held = bank[(bank["query"] == query) & (bank.partition == partition)]
                b_bound = 0.4 if world == "response_shift" else 0.15
                oracle = reader_moments(held, 0.45, b_bound)
                budget = config["design_budget"]
                pools, latent, _ = generate_responses(
                    rng, held, config["held_authors"], budget, b_bound=b_bound
                )
                reader_truth = oracle["mean"] + oracle["reader_gain"] * latent
                np.savez_compressed(
                    output / f"held_{key}_{world}.npz",
                    evidence=pools,
                    latent=latent,
                    reader_truth=reader_truth,
                )
                header = {"query": query, "seed": seed, "world": world}
                for name, target in TARGETS.items():
                    selected, trace = choose_allocation(source_moments, target, budget)
                    ideal, _ = choose_allocation(oracle, target, budget)
                    plans.append(
                        {
                            **header,
                            "target": name,
                            "source_counts": selected.tolist(),
                            "oracle_counts": ideal.tolist(),
                            "trace": trace,
                        }
                    )
                    for author, pool in enumerate(pools):
                        random_minus = 2 + int(rng.binomial(budget - 4, 0.5))
                        allocations = {
                            "source_design": selected,
                            "oracle_design": ideal,
                            "oracle_full": ideal,
                            "balanced": np.array([budget // 2, budget // 2]),
                            "random": np.array([random_minus, budget - random_minus]),
                            "repeat_aware": np.array([budget // 4, budget // 4]),
                            "repeat_naive": np.array([budget // 4, budget // 4]),
                            "one_condition": np.array([0, budget]),
                        }
                        for policy, counts in allocations.items():
                            repeat = policy.startswith("repeat_")
                            args = observations(
                                pool,
                                counts,
                                repeat=repeat,
                                naive=policy == "repeat_naive",
                            )
                            fit = linear_cell_score(
                                *args,
                                oracle if policy == "oracle_full" else source_moments,
                            )
                            rows.append(
                                {
                                    **header,
                                    "author": author,
                                    "target": name,
                                    "policy": policy,
                                    "prediction": target @ fit["score"],
                                    "truth": target @ latent[author],
                                    "reader_truth": target @ reader_truth[author],
                                    "variance": target @ fit["covariance"] @ target,
                                    "observed_term": target @ fit["observed"],
                                    "borrowed_mean_term": target @ fit["borrowed"],
                                    "minus_count": counts[0],
                                    "plus_count": counts[1],
                                    "reports": fit["reports"],
                                    "unique_events": fit["unique_events"],
                                    "unobserved_minus_weight_from_plus": fit[
                                        "data_weights"
                                    ][0, 1]
                                    if counts[0] == 0
                                    else np.nan,
                                }
                            )
                    # This evaluation-only allocation surface never revises the design.
                    for minus in range(2, budget - 1):
                        counts = np.array([minus, budget - minus])
                        predicted, variance = [], []
                        for pool in pools:
                            fit = linear_cell_score(
                                *observations(pool, counts), source_moments
                            )
                            predicted.append(target @ fit["score"])
                            variance.append(target @ fit["covariance"] @ target)
                        risks.append(
                            {
                                **header,
                                "target": name,
                                "minus_count": minus,
                                "plus_count": budget - minus,
                                "predicted_risk": np.mean(variance),
                                "actual_reader_mse": np.mean(
                                    (predicted - reader_truth @ target) ** 2
                                ),
                                "actual_numeric_mse": np.mean(
                                    (predicted - latent @ target) ** 2
                                ),
                                "selected": bool(np.array_equal(counts, selected)),
                                "oracle_selected": bool(np.array_equal(counts, ideal)),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(risks), plans, fits


def comparisons(frame, arm, keys, pairs, bootstraps):
    """Paired author-within-seed RMSE changes, conditional on the fixed word bank."""
    rows = []
    rng = np.random.default_rng(91390)
    for group, data in frame.groupby(keys):
        head = dict(zip(keys, group))
        for old, new in pairs:
            left = data[data[arm] == old].set_index(["seed", "author"])
            right = data[data[arm] == new].set_index(["seed", "author"])
            shared = left.index.intersection(right.index)
            if not len(shared):
                continue
            left, right = left.loc[shared], right.loc[shared]
            valid = np.isfinite(left.prediction) & np.isfinite(right.prediction)
            left, right = left[valid], right[valid]
            if not len(left):
                continue
            for truth in ("truth", "reader_truth"):
                if not np.isfinite(left[truth]).all():
                    continue
                error_old = (left.prediction - left[truth]).to_numpy() ** 2
                error_new = (right.prediction - right[truth]).to_numpy() ** 2
                indices = [
                    np.where(left.index.get_level_values("seed") == seed)[0]
                    for seed in left.index.get_level_values("seed").unique()
                ]
                bootstrap = []
                for _ in range(bootstraps):
                    take = np.concatenate(
                        [rng.choice(ix, len(ix), replace=True) for ix in indices]
                    )
                    bootstrap.append(
                        np.sqrt(error_old[take].mean())
                        - np.sqrt(error_new[take].mean())
                    )
                rows.append(
                    {
                        **head,
                        "old": old,
                        "new": new,
                        "truth_kind": truth,
                        "paired_n": len(left),
                        "rmse_reduction": np.sqrt(error_old.mean())
                        - np.sqrt(error_new.mean()),
                        "lower": np.quantile(bootstrap, 0.025),
                        "upper": np.quantile(bootstrap, 0.975),
                        "positive_seeds": sum(
                            error_old[ix].mean() > error_new[ix].mean()
                            for ix in indices
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run(config_path, output, smoke=False):
    """Freeze inputs, execute bounded packets and retain complete result tables."""
    start = time.monotonic()
    config = json.loads(config_path.read_text())
    if smoke:
        config.update(
            seeds=[91311], held_authors=8, source_authors=16, bootstrap_replicates=19
        )
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing experiment")
    output.mkdir(parents=True)
    bank_path = ROOT / config["bank"]
    inputs = [
        config_path,
        Path(__file__),
        ROOT / "grain_research/choice_design.py",
        ROOT / "docs/GRAIN_CHOICE_DESIGN_PACKET_V1.md",
        bank_path,
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    (output / "contract.json").write_text(
        json.dumps({"config": config, "input_hashes": hashes, "smoke": smoke}, indent=2)
        + "\n"
    )
    bank = pd.read_csv(bank_path)
    if set(bank.partition) != {"source", "held"}:
        raise ValueError("Explicit source/held wording partitions required")
    choice, samples, support = choice_study(config, bank)
    choice.to_csv(output / "choice_predictions.csv", index=False)
    samples.to_csv(output / "event_address_samples.csv", index=False)
    support.to_csv(output / "choice_support.csv", index=False)
    choice_keys = ["query", "world", "design", "estimator", "target"]
    metric_table(choice, choice_keys).to_csv(output / "choice_metrics.csv", index=False)
    metric_table(choice, choice_keys + ["seed"]).to_csv(
        output / "choice_metrics_by_seed.csv", index=False
    )
    print("Choice/response packet complete:", len(choice), "parameter rows", flush=True)
    design, risk, plans, fits = design_study(config, bank, output)
    design.to_csv(output / "design_predictions.csv", index=False)
    risk.to_csv(output / "allocation_risk.csv", index=False)
    (output / "source_fits.json").write_text(
        json.dumps(jsonable(fits), indent=2) + "\n"
    )
    (output / "allocation_plans.json").write_text(
        json.dumps(jsonable(plans), indent=2) + "\n"
    )
    design_keys = ["query", "world", "policy", "target"]
    metric_table(design, design_keys).to_csv(output / "design_metrics.csv", index=False)
    metric_table(design, design_keys + ["seed"]).to_csv(
        output / "design_metrics_by_seed.csv", index=False
    )
    comparisons(
        design,
        "policy",
        ["query", "world", "target"],
        [
            ("balanced", "source_design"),
            ("random", "source_design"),
            ("source_design", "oracle_full"),
            ("repeat_aware", "source_design"),
        ],
        config["bootstrap_replicates"],
    ).to_csv(output / "design_comparisons.csv", index=False)
    choice["arm"] = choice.design + "/" + choice.estimator
    pairs = [
        ("free/all_conditional", "fixed/all_conditional"),
        ("two_phase/all_conditional", "two_phase/assigned_only"),
        ("free/observed_choice", "free/known_opportunity"),
    ]
    comparisons(
        choice,
        "arm",
        ["query", "world", "target"],
        pairs,
        config["bootstrap_replicates"],
    ).to_csv(output / "choice_comparisons.csv", index=False)
    for path, expected in hashes.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError("Input changed during execution: " + path)
    files = {
        p.name: digest(p)
        for p in output.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    summary = {
        "status": "complete",
        "elapsed_seconds": time.monotonic() - start,
        "choice_prediction_rows": len(choice),
        "design_prediction_rows": len(design),
        "risk_rows": len(risk),
        "new_neural_calls": 0,
        "new_human_data": 0,
        "files": files,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)


def main():
    """CLI for reproducible smoke/full runs; no downloads or remote services."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_choice_design.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.output, args.smoke)


if __name__ == "__main__":
    main()
