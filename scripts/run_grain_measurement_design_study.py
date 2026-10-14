#!/usr/bin/env python3
"""Execute V3 Study A / V4 F1-F2 / V5 G1 standard-mechanism subexperiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.contextual_measurement import (
    basis,
    gaussian_scores,
    posterior_covariance,
    response_coordinates,
    target_variance_gain,
)


def generate_world(rng, n, observations, world, shifted=False):
    """Generate target, choice, state, expression and reader as distinct variables."""
    target, identity, other = rng.normal(size=(3, n))
    opportunity = rng.normal(scale=1.4, size=n)
    state = rng.normal(size=(n, observations))
    if world == "opportunity_shift" and shifted:
        opportunity += 2.0
    choice_strength = 1.3 if world == "choice_only" else 0.0
    logit = opportunity[:, None] + choice_strength * target[:, None]
    if world == "state_selection":
        logit = logit + 1.3 * state
    z = np.where(rng.random((n, observations)) < expit(logit), 1.0, -1.0)
    if world == "nonlinear":
        z = rng.uniform(-1, 1, size=z.shape)
    if world in ("interaction", "reader_loss"):
        mean = identity[:, None] + target[:, None] * z
    elif world == "nonlinear":
        mean = identity[:, None] + target[:, None] * (z**2 - 1 / 3)
    elif world == "opportunity_shift":
        mean = target[:, None] + 1.5 * z
    elif world == "choice_only":
        mean = identity[:, None] + 0.2 * other[:, None] * z
    elif world == "state_selection":
        mean = target[:, None] * (1 + 0.7 * z) + state
    elif world == "null":
        mean = identity[:, None] + other[:, None] * z
    elif world == "level":
        mean = target[:, None] + 0.2 * other[:, None] * z
    else:
        raise ValueError("Unknown world: " + world)
    expressed = mean + rng.normal(scale=0.8, size=z.shape)
    read = expressed.copy()
    if world == "reader_loss":
        # Sign/attribution loss is a separate channel, not nuisance subtraction.
        read *= rng.choice([-1.0, 1.0], size=z.shape, p=[0.45, 0.55])
    count = (z > 0).sum(axis=1)
    choice_logit = np.log((count + 0.5) / (observations - count + 0.5))
    return {
        "target": target,
        "z": z,
        "read": read,
        "expressed": expressed,
        "choice": choice_logit,
        "known_opportunity_choice": choice_logit - opportunity,
    }


def features(data, penalty):
    """Keep pooled, conditional, choice and ideal-reader interfaces separate."""
    linear = response_coordinates(data["z"], data["read"], 1, penalty)
    quadratic = response_coordinates(data["z"], data["read"], 2, penalty)
    return {
        "pooled": data["read"].mean(axis=1)[:, None],
        "conditional_linear": linear,
        "conditional_quadratic": quadratic,
        "conditional_choice_unknown": np.column_stack([quadratic, data["choice"]]),
        "conditional_choice_known": np.column_stack(
            [quadratic, data["known_opportunity_choice"]]
        ),
        "oracle_reader_conditional": response_coordinates(
            data["z"], data["expressed"], 2, penalty
        ),
    }


def reference_comparison(config):
    """Fit only development authors; use fixed settings on disjoint generated authors."""
    rows = []
    for index, world in enumerate(config["worlds"]):
        for seed in config["seeds"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            train = generate_world(
                rng, config["train_authors"], config["observations"], world
            )
            held = generate_world(
                rng, config["held_authors"], config["observations"], world, shifted=True
            )
            train_features, held_features = [
                features(d, config["response_penalty"]) for d in (train, held)
            ]
            for method in train_features:
                model = make_pipeline(
                    StandardScaler(), Ridge(alpha=config["ridge_alpha"])
                )
                model.fit(train_features[method], train["target"])
                pred = model.predict(held_features[method])
                rows.append(
                    {
                        "world": world,
                        "seed": seed,
                        "method": method,
                        "held_authors": len(pred),
                        "r": np.corrcoef(pred, held["target"])[0, 1],
                        "mse": np.mean((pred - held["target"]) ** 2),
                        "bias": np.mean(pred - held["target"]),
                        "meaning": "known numeric target, not human personality validity",
                    }
                )
    return pd.DataFrame(rows)


def design_path(config, degree, policy, rng):
    """Choose future observations without reading authors' target values or outcomes."""
    prior = np.eye(degree + 1) * config["design_prior_variance"]
    target = np.eye(degree + 1)[1]  # response slope at z=0
    candidates = np.asarray(config["design_candidates"])
    z = [-0.25, 0.25, -0.25, 0.25]
    for _ in range(config["design_budget"]):
        if policy == "random":
            choice = rng.choice(candidates)
        elif policy == "quietest":
            choice = candidates[np.argmin(0.25 + candidates**2)]
        else:
            covariance = posterior_covariance(
                basis(z, degree), 0.25 + np.asarray(z) ** 2, prior
            )
            gains = [
                target_variance_gain(covariance, target, basis(c, degree), 0.25 + c * c)
                for c in candidates
            ]
            choice = candidates[np.argmax(gains)]
        z.append(float(choice))
    return np.asarray(z), prior, target


def design_experiment(config):
    """Compare predicted design risk with simulation risk, including misspecification."""
    rows, allocations = [], []
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed + 9000)
        beta = rng.normal(
            scale=np.sqrt(config["design_prior_variance"]),
            size=(config["design_authors"], 4),
        )
        for world in ("linear", "cubic"):
            for degree in (1, 3):
                if world == "linear" and degree == 3:
                    continue
                for policy in ("random", "quietest", "target_variance"):
                    z, prior, target = design_path(config, degree, policy, rng)
                    noise = 0.25 + z**2
                    generating = beta[:, :2] @ basis(z, 1).T
                    if world == "cubic":
                        generating += beta[:, 2:] @ basis(z, 3)[:, 2:].T
                    observation = generating + rng.normal(
                        size=generating.shape
                    ) * np.sqrt(noise)
                    score, covariance = gaussian_scores(
                        basis(z, degree), observation, noise, prior
                    )
                    error = score @ target - beta[:, 1]
                    predicted = float(target @ covariance @ target)
                    rows.append(
                        {
                            "seed": seed,
                            "world": world,
                            "fit_degree": degree,
                            "policy": policy,
                            "n_authors": len(beta),
                            "observations": len(z),
                            "predicted_target_mse": predicted,
                            "observed_target_mse": np.mean(error**2),
                            "coverage95": np.mean(
                                np.abs(error) <= 1.95996398454 * np.sqrt(predicted)
                            ),
                            "model_correct": (world == "linear" and degree == 1)
                            or (world == "cubic" and degree == 3),
                        }
                    )
                    allocations.append(
                        {
                            "seed": seed,
                            "world": world,
                            "fit_degree": degree,
                            "policy": policy,
                            "conditions": z.tolist(),
                        }
                    )
    return pd.DataFrame(rows), allocations


def main():
    """Save the frozen recipe, numerical outputs and exact code provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/grain_contextual_measurement.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text())
    (args.output / "config.yaml").write_text(args.config.read_text())
    sources = [
        args.config,
        Path(__file__),
        ROOT / "grain_research/contextual_measurement.py",
    ]
    contract = {
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "data": "new simulated authors only; no raw human data, LLM or neural training",
        "claims": "standard-model reference and pre-specified counterexamples, not a new theorem",
        "selection": "fixed hyperparameters, no held-author or seed-based selection",
        "upstream_gap": "evidence/condition values generated directly; learning them from text remains open",
    }
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    try:
        if not all(isinstance(w, str) for w in config["worlds"]):
            raise ValueError("World names must be strings; quote YAML null")
        with threadpool_limits(limits=2):
            comparison = reference_comparison(config)
            comparison.to_csv(args.output / "reference_by_seed.csv", index=False)
            design, allocations = design_experiment(config)
    except Exception:
        (args.output / "error_log.txt").write_text(traceback.format_exc())
        (args.output / "summary.json").write_text(
            '{"status": "FAILED", "partial_outputs_retained": true}\n'
        )
        raise
    comparison.to_csv(args.output / "reference_by_seed.csv", index=False)
    comparison.groupby(["world", "method"])[["r", "mse", "bias"]].agg(
        ["mean", "std"]
    ).to_csv(args.output / "reference_summary.csv")
    design.to_csv(args.output / "design_by_seed.csv", index=False)
    design.groupby(["world", "fit_degree", "policy"])[
        ["predicted_target_mse", "observed_target_mse", "coverage95"]
    ].mean().to_csv(args.output / "design_summary.csv")
    (args.output / "design_allocations.json").write_text(
        json.dumps(allocations, indent=2) + "\n"
    )
    summary = {
        "status": "COMPLETE",
        "worlds": len(config["worlds"]),
        "seeds": len(config["seeds"]),
        "reference_rows": len(comparison),
        "design_rows": len(design),
        "theory_scope": "numeric mechanism and design only; text evidence learning and empirical validity open",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
