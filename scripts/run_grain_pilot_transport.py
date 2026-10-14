#!/usr/bin/env python3
"""Bounded independent-pilot transport study; no new LLM calls or human labels."""

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
from grain_research.choice_design import learn_cell_moments, linear_cell_score
from grain_research.pilot_transport import (
    calibration,
    fit_pilot,
    restore_units,
    score_batch,
    total_uncertainty,
)
from scripts.run_grain_choice_design import (
    TARGETS as CELL_TARGETS,
)
from scripts.run_grain_choice_design import (
    generate_responses,
    observations,
    reader_moments,
)
from scripts.run_grain_query_moment_operator import read, sha, write

BASE = ROOT / "results/grain_vector_mechanisms"
BANK = BASE / "mechanism_todo_v1/condition_accumulation_r1/public_scored_bank.csv"
COMMENTS = (
    BASE / "dynamic_path_channels_v1/within_comment/private_comment_descriptors.csv"
)


def prepare(out, config):
    """Freeze source files, explicit worlds and allocation before outcomes exist."""
    if out.exists():
        raise FileExistsError("Fresh packet directory required")
    files = [
        Path(__file__),
        config,
        ROOT / "grain_research/pilot_transport.py",
        ROOT / "grain_research/temporal_measurement.py",
        ROOT / "grain_research/choice_design.py",
        ROOT / "scripts/run_grain_choice_design.py",
        ROOT / "docs/GRAIN_PILOT_TRANSPORT_PACKET_V1.md",
        BANK,
        COMMENTS,
    ]
    out.mkdir(parents=True)
    write(out / "config.json", read(config))
    write(
        out / "contract.json",
        {
            "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "config_hash": sha(out / "config.json"),
            "human_labels": False,
            "meps_access": False,
            "new_neural_calls": 0,
        },
    )


def verify(out, stage):
    """Refuse changed inputs or overwriting completed/partial stage outputs."""
    contract = read(out / "contract.json")
    for path, digest in contract["files"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    if sha(out / "config.json") != contract["config_hash"]:
        raise ValueError("Frozen config changed")
    folder = out / stage
    if folder.exists():
        raise FileExistsError("Inspect existing stage; do not restart over it")
    folder.mkdir()
    return read(out / "config.json"), folder


def world_parameters(world):
    """Declared construction parameters, never passed to learned-model fitting."""
    mean = np.array([0.0, 0.12, 0.18])
    scale = np.diag([0.14, 0.10, 0.10])
    covariance = scale @ np.array([[1, 0.25, 0], [0.25, 1, 0.15], [0, 0.15, 1]]) @ scale
    p, rate = 0.0256, 0.3
    if world == "population_shift":
        mean = np.array([0.22, -0.12, 0.18])
        covariance = np.diag([0.23, 0.22, 0.12]) ** 2
    if world == "dynamics_shift":
        p, rate = 0.0625, 0.045
    return {
        "mean": mean,
        "covariance": covariance,
        "stationary": p,
        "rate": rate,
        "noise": 0.16 / 8,
    }


def meter(y, z, world):
    """Observation change, explicitly separate from latent population change."""
    if world == "affine_meter":
        return 0.6 + 1.8 * y
    if world == "conditional_meter":
        return 0.6 + 0.15 * z + (1.8 + 0.6 * z) * y
    return y.copy()


def generate(rng, n, repetitions, t, z, u, params, world):
    """Return noisy replicate observations and a separate evaluation-only truth."""
    profile = rng.multivariate_normal(params["mean"], params["covariance"], n)
    k = params["stationary"] * np.exp(-params["rate"] * np.abs(t[:, None] - t[None, :]))
    state = rng.multivariate_normal(np.zeros(len(t)), k, n)
    value = profile @ np.c_[np.ones(len(t)), z, u].T + state
    base = value[:, :, None] + rng.normal(0, 0.4, (n, len(t), repetitions))
    y = meter(base, z[None, :, None], world)
    return y, {"profile": profile, "state": state, "response": value}


def target_truth(truth, targets, t):
    """Evaluation-only target values in the original declared observation unit."""
    result = []
    gaps = np.diff(t)
    weights = np.r_[gaps[0] / 2, (gaps[:-1] + gaps[1:]) / 2, gaps[-1] / 2] / (
        t[-1] - t[0]
    )
    for kind in targets:
        if kind in ("level", "condition", "partner"):
            result.append(
                truth["profile"][:, ("level", "condition", "partner").index(kind)]
            )
        elif kind == "current_state":
            result.append(truth["state"][:, -1])
        elif kind == "current_response":
            result.append(truth["response"][:, -1])
        elif kind == "window_state":
            result.append(truth["state"] @ weights)
        elif kind == "conditional_profile":
            result.append(truth["profile"] @ np.array([1, 0.5, 0]))
        else:
            raise ValueError(kind)
    return np.stack(result, axis=1)


def records(
    head, arm, prediction, variance, truth, targets, *, within=None, between=None
):
    """Flatten preselected targets without pooling them as independent authors."""
    rows = []
    for i in range(len(prediction)):
        for j, target in enumerate(targets):
            rows.append(
                {
                    **head,
                    "author": i,
                    "target": target,
                    "arm": arm,
                    "score": prediction[i, j],
                    "variance": variance[i, j],
                    "within_variance": variance[i, j]
                    if within is None
                    else within[i, j],
                    "construction_variance": 0 if between is None else between[i, j],
                    "truth": truth[i, j],
                }
            )
    return rows


def metrics(frame):
    """Report measurement error and interval coverage, not personality accuracy."""
    rows = []
    keys = [k for k in ("world", "query", "seed", "target", "arm") if k in frame]
    for group, part in frame.groupby(keys):
        error = part.score.to_numpy() - part.truth.to_numpy()
        rows.append(
            {
                **dict(zip(keys, group)),
                "n": len(part),
                "rmse": np.sqrt(np.mean(error**2)),
                "bias": np.mean(error),
                "coverage95": np.mean(
                    abs(error) <= 1.959963984540054 * np.sqrt(part.variance)
                ),
                "mean_variance": part.variance.mean(),
                "within_variance": part.within_variance.mean(),
                "construction_variance": part.construction_variance.mean(),
            }
        )
    return pd.DataFrame(rows)


def gaussian(out):
    """Learn population and time parameters; apply only to separate authors."""
    cfg, folder = verify(out, "gaussian")
    start = time.process_time()
    rows, fits, boots, identities = [], {}, {}, []
    n, r = cfg["occasions"], cfg["replicates"]
    t = np.r_[0, np.cumsum(np.resize([0.2, 0.8, 1.5, 3.5], n - 1))]
    z = np.resize([-1, -1, 1, 1, -1, 1], n).astype(float)
    u = np.resize([-1, 1, -1, 1, 1, -1], n).astype(float)
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        source, _ = generate(
            rng,
            cfg["source_authors"],
            r,
            t,
            z,
            u,
            world_parameters("unchanged"),
            "unchanged",
        )
        source_fit = fit_pilot(source, z, u, t, maxiter=cfg["max_optimizer_iterations"])
        fits[f"source_{seed}"] = source_fit
        np.savez_compressed(
            folder / f"source_{seed}.npz",
            observations=source,
            times=t,
            condition=z,
            partner=u,
        )
        identities.extend(
            {
                "seed": seed,
                "world": "source",
                "partition": "source",
                "id": f"{seed}:source:{i}",
            }
            for i in range(len(source))
        )
        for wi, world in enumerate(cfg["worlds"]):
            rng = np.random.default_rng(np.random.SeedSequence([seed, wi, 2]))
            params = world_parameters(world)
            pilot, _ = generate(rng, cfg["pilot_authors"], r, t, z, u, params, world)
            evaluation, truth = generate(
                rng, cfg["evaluation_authors"], r, t, z, u, params, world
            )
            anchors = rng.uniform(-1, 1, cfg["paired_anchors"])
            anchor_z = np.resize([-1.0, 1.0], len(anchors))
            # Both anchor and evaluation meters are deterministic here. Recalibration
            # uncertainty is therefore numeric zero for correctly specified maps.
            anchor_new = meter(anchors, anchor_z, world)
            global_map = calibration(anchors, anchor_new, anchor_z)
            conditional_map = calibration(
                anchors, anchor_new, anchor_z, conditional=True
            )
            pilot_global = restore_units(pilot, z[None, :, None], global_map)
            pilot_conditional = restore_units(pilot, z[None, :, None], conditional_map)
            global_fit = fit_pilot(
                pilot_global, z, u, t, maxiter=cfg["max_optimizer_iterations"]
            )
            local_fit = fit_pilot(
                pilot_conditional, z, u, t, maxiter=cfg["max_optimizer_iterations"]
            )
            key = f"{seed}_{world}"
            fits[key] = {
                "global_map": global_map,
                "conditional_map": conditional_map,
                "global_fit": global_fit,
                "local_fit": local_fit,
            }
            np.savez_compressed(
                folder / f"pilot_{key}.npz",
                observations=pilot,
                times=t,
                condition=z,
                partner=u,
                anchors=anchors,
                anchor_condition=anchor_z,
                anchor_new=anchor_new,
            )
            np.savez_compressed(
                folder / f"evaluation_{key}.npz",
                observations=evaluation,
                profile=truth["profile"],
                state=truth["state"],
                response=truth["response"],
            )
            for partition, count in (
                ("pilot", len(pilot)),
                ("evaluation", len(evaluation)),
            ):
                identities.extend(
                    {
                        "seed": seed,
                        "world": world,
                        "partition": partition,
                        "id": f"{seed}:{world}:{partition}:{i}",
                    }
                    for i in range(count)
                )
            truth_values = target_truth(truth, cfg["targets"], t)
            restored_global = restore_units(
                evaluation, z[None, :, None], global_map
            ).mean(axis=2)
            restored_conditional = restore_units(
                evaluation, z[None, :, None], conditional_map
            ).mean(axis=2)
            arms = {
                "source_raw": (evaluation.mean(axis=2), source_fit),
                "source_anchor": (restored_global, source_fit),
                "pilot_global": (restored_global, global_fit),
                "pilot_conditional": (restored_conditional, local_fit),
                "oracle": (restored_conditional, params),
            }
            for arm, (data, model) in arms.items():
                prediction, variance = score_batch(data, z, u, t, model, cfg["targets"])
                rows.extend(
                    records(
                        {"seed": seed, "world": world},
                        arm,
                        prediction,
                        variance,
                        truth_values,
                        cfg["targets"],
                    )
                )
            bp, bv, traces = [], [], []
            for b in range(cfg["bootstrap_constructions"]):
                ii = rng.integers(0, len(pilot), len(pilot))
                ai = rng.integers(0, len(anchors), len(anchors))
                mapping = calibration(
                    anchors[ai], anchor_new[ai], anchor_z[ai], conditional=True
                )
                restored = restore_units(pilot[ii], z[None, :, None], mapping)
                model = fit_pilot(
                    restored, z, u, t, maxiter=cfg["max_optimizer_iterations"]
                )
                evaluation_fixed = restore_units(
                    evaluation, z[None, :, None], mapping
                ).mean(axis=2)
                prediction, variance = score_batch(
                    evaluation_fixed, z, u, t, model, cfg["targets"]
                )
                bp.append(prediction)
                bv.append(variance)
                traces.append(
                    {
                        "resampled_authors": ii,
                        "resampled_anchors": ai,
                        "fit": model,
                        "map": mapping,
                    }
                )
            mixture = total_uncertainty(bp, bv)
            rows.extend(
                records(
                    {"seed": seed, "world": world},
                    "pilot_mixture",
                    mixture["score"],
                    mixture["variance"],
                    truth_values,
                    cfg["targets"],
                    within=mixture["within"],
                    between=mixture["between"],
                )
            )
            np.savez_compressed(
                folder / f"bootstrap_scores_{key}.npz", scores=bp, variances=bv
            )
            boots[key] = traces
            write(
                folder / "progress.json",
                {"last_completed": key, "cpu_seconds": time.process_time() - start},
            )
            if time.process_time() - start > cfg["max_cpu_seconds"]:
                raise TimeoutError(
                    "Preserved current worlds; stop rather than start another fit"
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "predictions.csv", index=False)
    metrics(frame).to_csv(folder / "metrics_by_seed.csv", index=False)
    pooled = frame.drop(columns="seed")
    metrics(pooled).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(identities).to_csv(folder / "partitions.csv", index=False)
    write(folder / "fits.json", fits)
    write(folder / "bootstrap_fits.json", boots)
    all_fits = (
        [f for key, f in fits.items() if key.startswith("source_")]
        + [
            v[k]
            for key, v in fits.items()
            if not key.startswith("source_")
            for k in ("global_fit", "local_fit")
        ]
        + [b["fit"] for group in boots.values() for b in group]
    )
    write(
        folder / "summary.json",
        {
            "stage": "complete",
            "rows": len(rows),
            "models": len(all_fits),
            "optimizer_nonconverged": sum(not f["success"] for f in all_fits),
            "models_at_bounds": sum(f["bound_count"] > 0 for f in all_fits),
            "cpu_seconds": time.process_time() - start,
            "source_truth_used": False,
            "human_labels": False,
            "oracle_arm": "Evaluation reference only",
            "anchor_uncertainty": "Deterministic same-unit maps in Gaussian worlds; true conditional map bootstrap variation is numerical",
        },
    )


def text(out):
    """Real frozen reader observations with new simulated-person pilot moments."""
    cfg, folder = verify(out, "text")
    start = time.process_time()
    bank = pd.read_csv(BANK)
    rows, fits, partitions, samples = [], {}, [], []
    for qi, query in enumerate(cfg["queries"]):
        source_bank = bank[(bank["query"] == query) & (bank.partition == "source")]
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, qi, 5]))
            source, _, addresses = generate_responses(
                rng, source_bank, cfg["source_authors"], cfg["text_events_per_cell"]
            )
            source_fit = learn_cell_moments(source)
            key = f"{query}_{seed}"
            np.savez_compressed(
                folder / f"source_{key}.npz",
                observations=source,
                addresses=addresses.astype(str),
            )
            fits[f"source_{key}"] = source_fit
            partitions.extend(
                {
                    "query": query,
                    "seed": seed,
                    "world": "source",
                    "partition": "source",
                    "id": f"{key}:source:{i}",
                }
                for i in range(len(source))
            )
            for wi, world in enumerate(cfg["text_worlds"]):
                rng = np.random.default_rng(np.random.SeedSequence([seed, qi, wi, 6]))
                part = "source" if world == "same_wording" else "held"
                target_bank = bank[(bank["query"] == query) & (bank.partition == part)]
                bound = 0.4 if world == "response_shift" else 0.15
                pilot, _, addr_p = generate_responses(
                    rng,
                    target_bank,
                    cfg["pilot_authors"],
                    cfg["text_events_per_cell"],
                    b_bound=bound,
                )
                evaluation, latent, addr_e = generate_responses(
                    rng,
                    target_bank,
                    cfg["evaluation_authors"],
                    cfg["text_events_per_cell"],
                    b_bound=bound,
                )
                model = learn_cell_moments(pilot)
                ideal = reader_moments(target_bank, 0.45, bound)
                truth = ideal["mean"] + ideal["reader_gain"] * latent
                condition_id = f"{key}_{world}"
                fits[condition_id] = model
                boot_models = [
                    learn_cell_moments(pilot[rng.integers(0, len(pilot), len(pilot))])
                    for _ in range(cfg["bootstrap_constructions"])
                ]
                np.savez_compressed(
                    folder / f"pilot_{condition_id}.npz",
                    observations=pilot,
                    addresses=addr_p.astype(str),
                )
                np.savez_compressed(
                    folder / f"evaluation_{condition_id}.npz",
                    observations=evaluation,
                    addresses=addr_e.astype(str),
                    latent=latent,
                    reader_truth=truth,
                )
                fits[f"bootstrap_{condition_id}"] = boot_models
                for partition, count in (
                    ("pilot", len(pilot)),
                    ("evaluation", len(evaluation)),
                ):
                    partitions.extend(
                        {
                            "query": query,
                            "seed": seed,
                            "world": world,
                            "partition": partition,
                            "id": f"{condition_id}:{partition}:{i}",
                        }
                        for i in range(count)
                    )
                # Only four independent reports per condition for scoring. Pilot
                # design is intentionally richer than a new-person observation.
                count = np.array([4, 4])
                for i, pool in enumerate(evaluation):
                    args = observations(pool, count)
                    for arm, fitted in (
                        ("source", source_fit),
                        ("local_pilot", model),
                        ("oracle", ideal),
                    ):
                        result = linear_cell_score(*args, fitted)
                        for name, target in CELL_TARGETS.items():
                            rows.append(
                                {
                                    "query": query,
                                    "seed": seed,
                                    "world": world,
                                    "author": i,
                                    "target": name,
                                    "arm": arm,
                                    "score": target @ result["score"],
                                    "variance": target @ result["covariance"] @ target,
                                    "truth": target @ truth[i],
                                    "within_variance": target
                                    @ result["covariance"]
                                    @ target,
                                    "construction_variance": 0,
                                }
                            )
                    predictions, variances = [], []
                    for fitted in boot_models:
                        result = linear_cell_score(*args, fitted)
                        predictions.append(
                            [t @ result["score"] for t in CELL_TARGETS.values()]
                        )
                        variances.append(
                            [
                                t @ result["covariance"] @ t
                                for t in CELL_TARGETS.values()
                            ]
                        )
                    mix = total_uncertainty(predictions, variances)
                    for j, name in enumerate(CELL_TARGETS):
                        rows.append(
                            {
                                "query": query,
                                "seed": seed,
                                "world": world,
                                "author": i,
                                "target": name,
                                "arm": "pilot_mixture",
                                "score": mix["score"][j],
                                "variance": mix["variance"][j],
                                "truth": list(CELL_TARGETS.values())[j] @ truth[i],
                                "within_variance": mix["within"][j],
                                "construction_variance": mix["between"][j],
                            }
                        )
                samples.append(
                    {
                        "query": query,
                        "seed": seed,
                        "world": world,
                        "source_fit_floor": source_fit["floor_count"],
                        "local_fit_floor": model["floor_count"],
                        "bootstrap_floor_models": sum(
                            f["floor_count"] > 0 for f in boot_models
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "predictions.csv", index=False)
    metrics(frame).to_csv(folder / "metrics_by_seed.csv", index=False)
    metrics(frame.drop(columns="seed")).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(partitions).to_csv(folder / "partitions.csv", index=False)
    pd.DataFrame(samples).to_csv(folder / "covariance_floors.csv", index=False)
    write(folder / "fits.json", fits)
    write(
        folder / "summary.json",
        {
            "stage": "complete",
            "rows": len(rows),
            "cpu_seconds": time.process_time() - start,
            "target": "Frozen reader conditional expectations in the target wording world",
            "fit_input": "Independent pilot persons' text-reader observations, not their latent parameters",
            "conditions": "Externally assigned two-cell design, not open-text condition inference",
            "new_neural_calls": 0,
        },
    )


def natural(out):
    """Fit a separate descriptive observation reference; preserve old GRAIN scores."""
    cfg, folder = verify(out, "natural")
    import hashlib

    data = pd.read_csv(COMMENTS)
    features = ["level::whole:F01", "level::whole:F02"]
    means = data.groupby("author_id")[features].mean()
    source = np.array(
        [
            hashlib.sha256(("pilot-reference-v1:" + str(a)).encode()).digest()[0] < 128
            for a in means.index
        ]
    )
    if min(source.sum(), (~source).sum()) < 8:
        raise ValueError("Insufficient independent reference/evaluation authors")
    rng = np.random.default_rng(91389)
    ref = means.to_numpy()[source]
    mu, sd = ref.mean(axis=0), ref.std(axis=0, ddof=1)
    ii = rng.integers(0, len(ref), (999, len(ref)))
    mus, sds = ref[ii].mean(axis=1), ref[ii].std(axis=1, ddof=1)
    rows = []
    for author, row in means[~source].iterrows():
        chunk = data[data.author_id == author][features].to_numpy()
        jj = rng.integers(0, len(chunk), (999, len(chunk)))
        material = chunk[jj].mean(axis=1)
        # Total variance over independent reference and material bootstraps.
        # Conditional on a reference, material variance rescales by 1/sd_ref^2.
        for j, feature in enumerate(features):
            score = (row[feature] - mu[j]) / sd[j]
            ref_scores = (material[:, j].mean() - mus[:, j]) / sds[:, j]
            within = np.mean(material[:, j].var(ddof=1) / sds[:, j] ** 2)
            between = ref_scores.var(ddof=1)
            rows.append(
                {
                    "author_id": author,
                    "feature": feature,
                    "comments": len(chunk),
                    "raw_observation_mean": row[feature],
                    "source_relative_position": score,
                    "material_variance": within,
                    "reference_variance": between,
                    "total_variance": within + between,
                }
            )
    scored = pd.DataFrame(rows)
    scored.to_csv(folder / "private_relative_positions.csv", index=False)
    pd.DataFrame(
        {
            "author_id": means.index,
            "partition": np.where(source, "source", "evaluation"),
        }
    ).to_csv(folder / "private_partitions.csv", index=False)
    np.savez_compressed(
        folder / "source_reference.npz",
        mean=mu,
        sd=sd,
        bootstrap_mean=mus,
        bootstrap_sd=sds,
    )
    summary = []
    for j, feature in enumerate(features):
        part = scored[scored.feature == feature]
        summary.append(
            {
                "feature": feature,
                "source_authors": int(source.sum()),
                "evaluation_authors": int((~source).sum()),
                "source_observation_mean": mu[j],
                "source_observation_sd": sd[j],
                "evaluation_raw_mean": part.raw_observation_mean.mean(),
                "evaluation_position_mean": part.source_relative_position.mean(),
                "mean_material_variance": part.material_variance.mean(),
                "mean_reference_variance": part.reference_variance.mean(),
                "mean_total_variance": part.total_variance.mean(),
            }
        )
    pd.DataFrame(summary).to_csv(folder / "reference_summary.csv", index=False)
    write(
        folder / "summary.json",
        {
            "stage": "complete",
            "source_authors": int(source.sum()),
            "evaluation_authors": int((~source).sum()),
            "comments": len(data),
            "refitted_old_scores": False,
            "psychological_norm": False,
            "scope": cfg["natural_scope"],
            "meps_access": False,
        },
    )


def main():
    """Run one frozen stage at a time; preserve artifacts instead of silent retries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "gaussian", "text", "natural"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_pilot_transport.json"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == "prepare":
            prepare(args.output, args.config)
        else:
            globals()[args.stage](args.output)
    print(f"Completed {args.stage}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
