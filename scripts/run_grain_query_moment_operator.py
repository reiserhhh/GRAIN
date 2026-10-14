#!/usr/bin/env python3
"""Compile M05 source query moments, apply without targets, then evaluate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.query_moment_operator import (
    apply_operator,
    fit_operator,
    observation_moments,
    within_direction,
)
from grain_research.vector_mechanisms.native_material_reader import direction

BASE = ROOT / "results/grain_vector_mechanisms"
INPUT = BASE / "logic_complete_function_extraction_v1"
NATIVE = BASE / "logic_fixed_seed_attention_v1"
PLAN = ROOT / "docs/GRAIN_QUERY_MOMENT_OPERATOR_PLAN_V1.md"
QUERIES = ("neuroticism", "confirmation_seeking")
ARMS = ("linear", "quadratic", "within_augmented")


def read(path):
    """Read a private numeric/metadata artifact without printing raw material."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Stream source/artifact hashes."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    """Write standard JSON with missing values explicitly null."""

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            v = v.item()
        return None if isinstance(v, float) and not np.isfinite(v) else v

    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def arrays(path):
    """Load numeric arrays without permitting pickled objects."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def current(job):
    """Load current pure state/GRAIN; never native target fields."""
    path = INPUT / "private_inputs" / (job["query_keys"][0] + ".npz")
    data = arrays(path)
    if set(data) != {"state", "grain", "positions", "cos", "sin", "mask"}:
        raise ValueError("Unexpected data in pure application input")
    return {
        "h": data["state"][data["positions"]].astype(float),
        "g": data["grain"].astype(float),
    }


def teacher(job, qi):
    """Open an explicitly requested teacher field only in source/evaluation stages."""
    key = job["query_keys"][qi]
    gradient = arrays(NATIVE / "private_packets" / (key + ".npz"))["body_gradient"]
    value = read(NATIVE / "private_packets" / (key + ".json"))["contrast"]
    return {"j": gradient.astype(float), "value": float(value)}


def verify(out):
    """Verify local bindings; target contents are not deserialized here."""
    contract = read(out / "contract.json")
    for name, expected in contract["inputs"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed frozen input: " + name)
    return contract


def prepare(out):
    """Freeze twelve reused authors, source membership and the candidate formula."""
    old = BASE / "logic_grain_cell_residual_v1"
    if read(old / "audit.json")["status"] != "PASS":
        raise ValueError("Completed NEXT91 audit required")
    jobs = read(INPUT / "private_jobs.json")
    if len(jobs) != 12 or sum(j["cohort"] == "source4" for j in jobs) != 4:
        raise ValueError("Frozen source4/reused8 membership required")
    paths = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/query_moment_operator.py",
        INPUT / "private_jobs.json",
        old / "audit.json",
    ]
    prior_hashes = read(old / "manifest.json")["inputs"]
    for job in jobs:
        first = current(job)
        for key in job["query_keys"]:
            ip = INPUT / "private_inputs" / (key + ".npz")
            data = arrays(ip)
            np.testing.assert_array_equal(first["h"], data["state"][data["positions"]])
            np.testing.assert_array_equal(first["g"], data["grain"])
            paths.extend(
                [
                    ip,
                    NATIVE / "private_packets" / (key + ".json"),
                    NATIVE / "private_packets" / (key + ".npz"),
                ]
            )
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for key, expected in hashes.items():
        if key in prior_hashes and prior_hashes[key] != expected:
            raise ValueError("NEXT91 binding changed: " + key)
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "contract.json",
        {
            "protocol": "M05_QUERY_MOMENT_OPERATOR_V1",
            "jobs": jobs,
            "inputs": hashes,
            "rank": 4,
            "alpha": 1e-6,
            "seed": 91351,
            "epsilon": 1e-4,
            "max_cpu_seconds": 120,
            "native_calls": 0,
        },
    )
    print("Prepared source4/reused8 and two queries", flush=True)


def train(out):
    """Construct from source gradients only; no held states are fitted."""
    contract = verify(out)
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    started = time.process_time()
    sources = [j for j in contract["jobs"] if j["cohort"] == "source4"]
    for qi, query in enumerate(QUERIES):
        panels = [{**current(j), **teacher(j, qi)} for j in sources]
        model = fit_operator(
            panels,
            rank=contract["rank"],
            alpha=contract["alpha"],
            seed=contract["seed"],
        )
        np.savez(dst / (query + ".npz"), **model)
    elapsed = time.process_time() - started
    if elapsed > contract["max_cpu_seconds"]:
        raise TimeoutError("Source compile exceeded numerical budget")
    write(
        dst / "freeze.json",
        {
            "files": {q + ".npz": sha(dst / (q + ".npz")) for q in QUERIES},
            "source_authors": 4,
            "cpu_seconds": elapsed,
        },
    )
    print({"stage": "train", "cpu_seconds": elapsed}, flush=True)


def load_models(out):
    """Read immutable compiled matrices, not teacher caches."""
    meta = read(out / "models/freeze.json")
    for name, expected in meta["files"].items():
        if sha(out / "models" / name) != expected:
            raise ValueError("Compiled model changed")
    return {q: arrays(out / "models" / (q + ".npz")) for q in QUERIES}


def directions(data, models):
    """Old four directions and three new moment-null directions, all input-only."""
    g, h = data["g"], data["h"]
    values = {f"F{i + 1:02d}": direction(h, g, i)[0] for i in range(4)}
    for query, model in models.items():
        values["within_" + query] = within_direction(g, h, model["query_basis"])
    basis = np.linalg.qr(np.random.default_rng(91352).normal(size=(h.shape[1], 4)))[0]
    values["within_random"] = within_direction(g, h, basis)
    return values


def apply(out):
    """Compute all candidate scores and actions without opening evaluation targets."""
    contract, models = verify(out), load_models(out)
    dst = out / "application"
    dst.mkdir(exist_ok=False)
    rows, states = [], []
    started = time.process_time()
    max_null = max_derivative = 0.0
    eps = contract["epsilon"]
    for job in contract["jobs"]:
        data = current(job)
        ds = directions(data, models)
        for query, model in models.items():
            values, fields = apply_operator(data["g"], data["h"], model)
            moment = observation_moments(data["g"], data["h"], model)
            state = {
                "author_key": job["key"],
                "cohort": job["cohort"],
                "query": query,
                "within_energy": np.trace(moment["within_covariance"]),
                "within_score": values[2]
                - values[1]
                - (model["offsets"][2] - model["offsets"][1]),
            }
            states.append(state)
            for name, d in ds.items():
                plus = apply_operator(data["g"], data["h"] + eps * d, model)[0]
                minus = apply_operator(data["g"], data["h"] - eps * d, model)[0]
                effects = np.einsum("ath,th->a", fields, d)
                error = np.max(abs((plus - minus) / (2 * eps) - effects))
                max_derivative = max(max_derivative, error)
                null = np.linalg.norm(
                    observation_moments(data["g"], d, model)["quadratic"]
                )
                if name.startswith("within_"):
                    max_null = max(max_null, null)
                for ai, arm in enumerate(ARMS):
                    rows.append(
                        {
                            "author_key": job["key"],
                            "cohort": job["cohort"],
                            "query": query,
                            "direction": name,
                            "arm": arm,
                            "score": values[ai],
                            "effect": effects[ai],
                            "finite_difference": (plus[ai] - minus[ai]) / (2 * eps),
                            "quadratic_moment_derivative_norm": null,
                        }
                    )
    pd.DataFrame(rows).to_csv(dst / "private_predictions.csv", index=False)
    pd.DataFrame(states).to_csv(dst / "private_observations.csv", index=False)
    elapsed = time.process_time() - started
    if (
        elapsed + read(out / "models/freeze.json")["cpu_seconds"]
        > contract["max_cpu_seconds"]
    ):
        raise TimeoutError("Application exceeded numerical budget")
    if max_null > 1e-8 or max_derivative > 1e-6:
        raise ArithmeticError("Moment nullspace or scalar derivative identity failed")
    write(
        dst / "freeze.json",
        {
            "files": {
                name: sha(dst / name)
                for name in ("private_predictions.csv", "private_observations.csv")
            },
            "max_null": max_null,
            "max_derivative": max_derivative,
            "cpu_seconds": elapsed,
            "teacher_calls": 0,
        },
    )
    print({"stage": "apply", "rows": len(rows), "cpu_seconds": elapsed}, flush=True)


def evaluate(out):
    """Only after freezing predictions, compare old native fields and author positions."""
    contract, models = verify(out), load_models(out)
    freeze = read(out / "application/freeze.json")
    for name, expected in freeze["files"].items():
        if sha(out / "application" / name) != expected:
            raise ValueError("Predictions changed before evaluation")
    dst = out / "evaluation"
    dst.mkdir(exist_ok=False)
    pred = pd.read_csv(out / "application/private_predictions.csv")
    truth, fields = [], []
    for job in contract["jobs"]:
        data = current(job)
        ds = directions(data, models)
        for qi, query in enumerate(QUERIES):
            actual = teacher(job, qi)
            _, estimates = apply_operator(data["g"], data["h"], models[query])
            for ai, arm in enumerate(ARMS):
                j, p = actual["j"], estimates[ai]
                fields.append(
                    {
                        "author_key": job["key"],
                        "cohort": job["cohort"],
                        "query": query,
                        "arm": arm,
                        "field_error_squared": np.sum((p - j) ** 2),
                        "native_field_energy": np.sum(j * j),
                        "cosine": np.sum(p * j)
                        / (np.linalg.norm(p) * np.linalg.norm(j)),
                    }
                )
            truth.extend(
                {
                    "author_key": job["key"],
                    "query": query,
                    "direction": name,
                    "native_score": actual["value"],
                    "native_effect": np.sum(actual["j"] * d),
                }
                for name, d in ds.items()
            )
    merged = pred.merge(
        pd.DataFrame(truth),
        on=["author_key", "query", "direction"],
        validate="many_to_one",
    )
    merged.to_csv(dst / "private_comparison.csv", index=False)
    pd.DataFrame(fields).to_csv(dst / "private_fields.csv", index=False)
    rows = []
    for (cohort, query, arm), group in merged.groupby(["cohort", "query", "arm"]):
        points = group.drop_duplicates("author_key")
        for scope in ("old4", "within2", "random"):
            selected = (
                group[group.direction.str.startswith("F")]
                if scope == "old4"
                else group[group.direction.isin(["within_" + q for q in QUERIES])]
                if scope == "within2"
                else group[group.direction == "within_random"]
            )
            y, p = selected.native_effect.to_numpy(), selected.effect.to_numpy()
            rows.append(
                {
                    "cohort": cohort,
                    "query": query,
                    "arm": arm,
                    "scope": scope,
                    "authors": len(points),
                    "effect_rmse": np.sqrt(np.mean((p - y) ** 2)),
                    "effect_signed_projection": p @ y / (y @ y) if y @ y else np.nan,
                    "native_effect_rms": np.sqrt(np.mean(y * y)),
                    "predicted_effect_rms": np.sqrt(np.mean(p * p)),
                    "score_rmse": np.sqrt(
                        np.mean((points.score - points.native_score) ** 2)
                    ),
                    "score_r": points.score.corr(points.native_score),
                }
            )
    pd.DataFrame(rows).to_csv(dst / "summary.csv", index=False)
    fs = pd.DataFrame(fields)
    aggregate = (
        fs.groupby(["cohort", "query", "arm"])[
            ["field_error_squared", "native_field_energy", "cosine"]
        ]
        .agg(
            {
                "field_error_squared": "sum",
                "native_field_energy": "sum",
                "cosine": "mean",
            }
        )
        .reset_index()
    )
    aggregate["field_nmse"] = (
        aggregate.field_error_squared / aggregate.native_field_energy
    )
    aggregate.to_csv(dst / "field_summary.csv", index=False)
    observations = pd.read_csv(out / "application/private_observations.csv")
    positions = []
    for query, group in observations.groupby("query"):
        source = group[group.cohort == "source4"]
        for name in ("within_energy", "within_score"):
            mean, std = source[name].mean(), source[name].std(ddof=1)
            for r in group.itertuples():
                positions.append(
                    {
                        "author_key": r.author_key,
                        "cohort": r.cohort,
                        "query": query,
                        "observation": name,
                        "value": getattr(r, name),
                        "source_standardized": (getattr(r, name) - mean) / std,
                        "reference_n": len(source),
                    }
                )
    pd.DataFrame(positions).to_csv(dst / "private_author_positions.csv", index=False)
    write(
        dst / "summary.json",
        {
            "status": "complete",
            "finite_hidden_state_sensor": True,
            "new_native_calls": 0,
            "gradient_is_evaluation_only_for_new8": True,
            "source_authors": 4,
            "reused_authors": 8,
            "files": {
                p.name: sha(p)
                for p in dst.iterdir()
                if p.is_file() and not p.name.startswith("._")
            },
        },
    )
    print("Evaluation complete; cold raw-text stage remains", flush=True)


def main():
    """Separate construction, blind application and evaluation CLI processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "apply", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
