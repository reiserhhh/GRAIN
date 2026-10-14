#!/usr/bin/env python3
"""Freeze, execute and replay HM142 without model or teacher calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_multiview_law as M

SPEC = ROOT / "configs/grain_multiview_law_hm142.json"
CODE = (
    "configs/grain_multiview_law_hm142.json",
    "docs/GRAIN_MULTIVIEW_LAW_PLAN_HM142.md",
    "grain_research/construction/math_multiview_law.py",
    "scripts/run_grain_multiview_law_hm142.py",
    "tests/test_grain_multiview_law.py",
)


def read(path):
    """Read a named local JSON artifact."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash the complete artifact without mutating it."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Create new artifacts exclusively, preserving every previous attempt."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def archive(path, **arrays):
    """Create an exclusive binary numerical archive."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def table(path, rows):
    """Store all predeclared numerical cases, not just passing outcomes."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    """Freeze source-only alphabets and all constructor dependencies."""
    if out.exists():
        raise FileExistsError("Existing/failed runs are retained, not overwritten")
    spec = read(SPEC)
    source = ROOT / spec["source_dictionary"]
    prior = read(ROOT / spec["source_manifest"])
    if sha(source) != prior["dictionary_sha256"]:
        raise ValueError("Changed frozen HM140 dictionary")
    with np.load(source, allow_pickle=False) as a:
        alphabet = a["menus"][:9].reshape(3, 3, 15).copy()
    distances = [
        np.linalg.norm(a[i] - a[j])
        for a in alphabet
        for i, j in itertools.combinations(range(3), 2)
    ]
    if min(distances) <= 1e-10:
        raise ValueError("Frozen alphabet has coinciding category vectors")
    out.mkdir(parents=True)
    archive(out / "alphabet.npz", alphabet=alphabet)
    write(
        out / "manifest.json",
        {
            "packet": "HM142",
            "code": {str(ROOT / p): sha(ROOT / p) for p in CODE},
            "inputs": {str(source): sha(source)},
            "alphabet_sha256": sha(out / "alphabet.npz"),
            "scope": "Finite GRAIN vector alphabet;anonymous unknown law,not natural text or psychological meaning",
        },
    )


def check(out):
    """Verify frozen code/source provenance before execution or replay."""
    manifest = read(out / "manifest.json")
    for path, digest in (manifest["code"] | manifest["inputs"]).items():
        if sha(path) != digest:
            raise ValueError("Changed HM142 dependency: " + path)
    if sha(out / "alphabet.npz") != manifest["alphabet_sha256"]:
        raise ValueError("Changed frozen alphabet")


def sample(rng, labels, emissions):
    """Generate independent categorical events conditional on hidden component."""
    return np.stack(
        [
            (rng.random(len(labels))[:, None] > np.cumsum(m[:, labels].T, axis=1)).sum(
                axis=1
            )
            for m in emissions
        ],
        axis=1,
    ).astype(np.int8)


def pack_model(model):
    """Return only source-constructed parameters, without oracle component names."""
    return {
        "weights": model["weights"],
        "emissions": np.stack(model["emissions"]),
        "eigenvalues": model["eigenvalues"],
    }


def oracle_alignment(model, emissions):
    """Find the evaluation-only label permutation after cold scoring is frozen."""
    return min(
        itertools.permutations(range(3)),
        key=lambda perm: sum(
            np.sum((a[:, perm] - b) ** 2)
            for a, b in zip(model["emissions"], emissions, strict=True)
        ),
    )


def predictive_nll(first_posterior, future, weights, emissions):
    """Evaluate cold next-bundle law, retaining prior as an honest comparator."""
    likelihood = np.ones((len(future), 3))
    for v, m in enumerate(emissions):
        likelihood *= m[future[:, v]]
    predicted = (first_posterior * likelihood).sum(axis=1)
    prior = likelihood @ weights
    difference = np.log(predicted) - np.log(prior)
    return float(difference.mean()), float(
        difference.std(ddof=1) / np.sqrt(len(future))
    )


def cross_occasion_norm(first, second):
    """Measure categorical within-author between-occasion covariance, not r."""
    a, b = np.eye(3)[first[:, 0]], np.eye(3)[second[:, 0]]
    covariance = a.T @ b / len(a) - np.outer(a.mean(axis=0), b.mean(axis=0))
    return float(np.linalg.norm(covariance))


def population(emissions, weights, spec):
    """Enumerate informative, ambiguous, nonidentifying and copied-event laws."""
    changed_p, changed_m, rotation = M.pairwise_alias(
        weights, emissions, spec["pairwise_alias_angle"]
    )
    uninformative = [np.full((3, 3), 1 / 3)] * 3
    bad_third = [emissions[0], emissions[1], np.full((3, 3), 1 / 3)]
    copied = [np.eye(3)] * 3
    worlds = {
        "ordinary": (weights, emissions),
        "pairwise_alias": (changed_p, changed_m),
        "uninformative": (weights, uninformative),
        "nonseparating_third": (weights, bad_third),
        "copied_one_event": (weights, copied),
    }
    rows, arrays = [], {"rotation": rotation}
    for name, (p, matrices) in worlds.items():
        t = M.joint_law(p, matrices)
        arrays[name] = t
        try:
            fitted = M.fit_law(t, 3, spec["contraction"])
            perm = oracle_alignment(fitted, matrices)
            error = max(
                np.max(abs(a[:, perm] - b))
                for a, b in zip(fitted["emissions"], matrices, strict=True)
            )
            row = {
                "world": name,
                "status": "IDENTIFIED_ANONYMOUS_LAW",
                "emission_max_error": float(error),
                "reconstruction_TV": fitted["reconstruction_TV"],
                "reason": "Anonymous mixture,not psychological cause",
            }
        except M.IdentificationError as exc:
            row = {
                "world": name,
                "status": "NONIDENTIFYING_FIXED_CONSTRUCTOR",
                "emission_max_error": None,
                "reconstruction_TV": None,
                "reason": str(exc),
            }
        rows.append(row)
    first, other = arrays["ordinary"], arrays["pairwise_alias"]
    categories = np.array(list(itertools.product(range(3), repeat=3)))
    diagnostics = {
        "pairwise_max_error": max(
            float(np.max(abs(first.sum(axis=v) - other.sum(axis=v)))) for v in range(3)
        ),
        "triple_TV": float(abs(first - other).sum() / 2),
        "posterior_alias_max": float(
            np.max(
                abs(
                    M.posterior(categories, weights, emissions)
                    - M.posterior(categories, changed_p, changed_m)
                )
            )
        ),
        "rotation_orthogonality_error": float(
            np.max(abs(rotation.T @ rotation - np.eye(3)))
        ),
        "source_author_vs_occasion_tensor_error": 0.0,
        "copied_event_scope": "Exact spectral fit is possible;event provenance rejects three encodings of one event as three independent views",
    }
    return rows, arrays, diagnostics


def composition_checks(weights, emissions):
    """Test actual likelihood composition and declared copied-event exclusion."""
    rows = [
        {"event": str(v), "view": v, "category": v, "qualification": "self_actual"}
        for v in range(3)
    ]
    unique = M.qualified_posterior(rows, weights, emissions)
    repeated = M.qualified_posterior(rows * 5, weights, emissions)
    naive = M.posterior(np.array([[0, 1, 2]]), weights, emissions)[0]
    try:
        M.empirical_tensor(np.array([[0, 0, 0]]), (3, 3, 3), ["same"] * 3)
    except ValueError:
        copied_rejected = True
    else:
        copied_rejected = False
    return {
        "event_repeat_probability_error": float(
            np.max(abs(unique["posterior"] - repeated["posterior"]))
        ),
        "direct_composition_probability_error": float(
            np.max(abs(unique["posterior"] - naive))
        ),
        "unique_events": unique["unique_events"],
        "repeated_mentions": repeated["repeated_mentions"],
        "copied_event_views_rejected": copied_rejected,
        "scope": "Declared event identities and conditional independence are distinct assumptions",
    }


def run(out):
    """Learn source laws, seal cold scores, then evaluate oracle distinctions."""
    check(out)
    if (out / "execution_started.json").exists():
        raise FileExistsError("Research run already started;no automatic replay")
    write(
        out / "execution_started.json",
        {
            "status": "STARTED",
            "resumption": "Existing completed seed archives are retained;no automatic restart",
        },
    )
    spec = read(SPEC)
    start = time.perf_counter()
    weights = np.asarray(spec["weights"])
    emissions = tuple(np.asarray(a) for a in spec["emissions"])
    with np.load(out / "alphabet.npz", allow_pickle=False) as a:
        alphabet = a["alphabet"].copy()
    pop_rows, pop_arrays, diagnostics = population(emissions, weights, spec)
    table(out / "population.csv", pop_rows)
    archive(out / "population.npz", **pop_arrays)
    write(out / "identification_witness.json", diagnostics)
    write(out / "composition_checks.json", composition_checks(weights, emissions))
    metrics = []
    for seed in spec["seeds"]:
        if time.perf_counter() - start > spec["max_CPU_wall_seconds"]:
            raise TimeoutError("Original CPU cap reached;preserve completed outputs")
        directory = out / str(seed)
        directory.mkdir()
        rng = np.random.default_rng(seed)
        source_h = rng.choice(3, spec["source_authors_per_seed"], p=weights)
        source = sample(rng, source_h, emissions)
        # Run the actual frozen-alphabet vector observation channel. No native
        # teacher, fresh embedding, hidden label or ID is used by this decoder.
        observed_vectors = np.stack(
            [alphabet[v, source[:, v]] for v in range(3)], axis=1
        )
        decoded = M.decode_vectors(observed_vectors, alphabet)
        cold_h = rng.choice(3, spec["application_authors_per_seed"], p=weights)
        first = sample(rng, cold_h, emissions)
        stable_second = sample(rng, cold_h, emissions)
        occasion_h = rng.choice(3, len(cold_h), p=weights)
        occasion_second = sample(rng, occasion_h, emissions)
        archive(
            directory / "observations.npz",
            source=decoded,
            first=first,
            stable_second=stable_second,
            occasion_second=occasion_second,
        )
        archive(
            directory / "oracle_evaluation_only.npz",
            source_h=source_h,
            cold_h=cold_h,
            occasion_h=occasion_h,
            weights=weights,
            emissions=np.stack(emissions),
        )
        observed = M.empirical_tensor(
            decoded, tuple(spec["categories_per_view"]), spec["event_views"]
        )
        archive(directory / "observed_tensor.npz", tensor=observed)
        try:
            fitted = M.fit_law(observed, 3, spec["contraction"])
        except M.IdentificationError as exc:
            write(
                directory / "failure.json",
                {"status": "FIXED_CONSTRUCTOR_FAILURE", "reason": str(exc)},
            )
            metrics.append(
                {
                    "seed": seed,
                    "status": "FAILURE_RETAINED",
                    "emission_RMSE": None,
                    "weight_max_error": None,
                    "sample_reconstruction_TV": None,
                    "posterior_RMSE": None,
                    "stable_predictive_log_gain": None,
                    "stable_gain_SE": None,
                    "occasion_predictive_log_gain": None,
                    "occasion_gain_SE": None,
                    "stable_cross_occasion_cov_norm": None,
                    "occasion_cross_occasion_cov_norm": None,
                    "vector_decode_error": int(np.sum(decoded != source)),
                }
            )
            continue
        archive(directory / "source_model.npz", **pack_model(fitted))
        cold = M.posterior(first, fitted["weights"], fitted["emissions"])
        archive(directory / "cold_scores.npz", posterior=cold)
        write(
            directory / "cold_seal.json",
            {
                p.name: sha(p)
                for p in (
                    directory / "source_model.npz",
                    directory / "cold_scores.npz",
                    directory / "observations.npz",
                )
            },
        )
        # Oracle parameters first enter interpretation below, after source model
        # and all cold class scores are frozen. They never alter the estimator.
        perm = oracle_alignment(fitted, emissions)
        oracle = M.posterior(first, weights, emissions)
        stable_gain, stable_se = predictive_nll(
            cold, stable_second, fitted["weights"], fitted["emissions"]
        )
        state_gain, state_se = predictive_nll(
            cold, occasion_second, fitted["weights"], fitted["emissions"]
        )
        row = {
            "seed": seed,
            "status": "IDENTIFIED_ANONYMOUS_LAW",
            "emission_RMSE": float(
                np.sqrt(
                    np.mean(
                        [
                            (a[:, perm] - b) ** 2
                            for a, b in zip(fitted["emissions"], emissions, strict=True)
                        ]
                    )
                )
            ),
            "weight_max_error": float(
                np.max(abs(fitted["weights"][list(perm)] - weights))
            ),
            "sample_reconstruction_TV": fitted["reconstruction_TV"],
            "posterior_RMSE": float(np.sqrt(np.mean((cold[:, perm] - oracle) ** 2))),
            "stable_predictive_log_gain": stable_gain,
            "stable_gain_SE": stable_se,
            "occasion_predictive_log_gain": state_gain,
            "occasion_gain_SE": state_se,
            "stable_cross_occasion_cov_norm": cross_occasion_norm(first, stable_second),
            "occasion_cross_occasion_cov_norm": cross_occasion_norm(
                first, occasion_second
            ),
            "vector_decode_error": int(np.sum(decoded != source)),
        }
        write(directory / "metrics.json", row)
        metrics.append(row)
    table(out / "metrics.csv", metrics)
    write(
        out / "result.json",
        {
            "packet": "HM142",
            "status": "FINITE_UNKNOWN_LAW_CONSTRUCTOR_EXECUTED",
            "population": pop_rows,
            "witness": diagnostics,
            "sample_metrics": metrics,
            "source_authors": spec["source_authors_per_seed"] * len(spec["seeds"]),
            "application_authors": spec["application_authors_per_seed"]
            * len(spec["seeds"]),
            "CPU_wall_seconds": time.perf_counter() - start,
            "scientific_scope": "Unknown finite anonymous observation law,not named psychological cause;no native teacher fit",
            "original_scope_complete": False,
            "new_neural_API_training_private_calls": 0,
        },
    )
    write(
        out / "output_seal.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )


def verify(out):
    """Read-only all-tensor/model/cold-score replay, not another research trial."""
    check(out)
    for name, digest in read(out / "output_seal.json").items():
        if sha(out / name) != digest:
            raise ValueError("Changed execution artifact: " + name)
    spec = read(SPEC)
    errors = {
        "tensor": 0.0,
        "model": 0.0,
        "cold_scores": 0.0,
        "Bayes_enumeration": 0.0,
        "metrics": 0.0,
        "population_enumeration": 0.0,
    }
    weights = np.asarray(spec["weights"])
    emissions = tuple(np.asarray(a) for a in spec["emissions"])
    pop_rows, pop_arrays, diagnostics = population(emissions, weights, spec)
    saved_result = read(out / "result.json")
    if (
        saved_result["population"] != pop_rows
        or read(out / "identification_witness.json") != diagnostics
    ):
        raise ValueError("Population result replay mismatch")
    with np.load(out / "population.npz", allow_pickle=False) as a:
        for name, value in pop_arrays.items():
            if np.max(abs(a[name] - value)) > spec["exact_tolerance"]:
                raise ValueError("Population archive replay mismatch")
    direct_tensor = np.zeros((3, 3, 3))
    for h, i, j, k in itertools.product(range(3), repeat=4):
        direct_tensor[i, j, k] += (
            weights[h] * emissions[0][i, h] * emissions[1][j, h] * emissions[2][k, h]
        )
    errors["population_enumeration"] = float(
        np.max(abs(direct_tensor - pop_arrays["ordinary"]))
    )
    if read(out / "composition_checks.json") != composition_checks(weights, emissions):
        raise ValueError("Event composition replay mismatch")
    replayed = 0
    for seed in spec["seeds"]:
        directory = out / str(seed)
        with np.load(directory / "observations.npz", allow_pickle=False) as a:
            source, first = a["source"].copy(), a["first"].copy()
            stable_second, occasion_second = (
                a["stable_second"].copy(),
                a["occasion_second"].copy(),
            )
        with np.load(directory / "observed_tensor.npz", allow_pickle=False) as a:
            observed = a["tensor"].copy()
        counts = np.bincount(
            np.ravel_multi_index(tuple(source.T), (3, 3, 3)), minlength=27
        ).reshape(3, 3, 3)
        errors["tensor"] = max(
            errors["tensor"], float(np.max(abs(counts / len(source) - observed)))
        )
        if (directory / "failure.json").exists():
            continue
        fitted = M.fit_law(observed, 3, spec["contraction"])
        with np.load(directory / "source_model.npz", allow_pickle=False) as a:
            for key, value in pack_model(fitted).items():
                errors["model"] = max(
                    errors["model"], float(np.max(abs(a[key] - value)))
                )
        with np.load(directory / "cold_scores.npz", allow_pickle=False) as a:
            original = a["posterior"].copy()
        scores = M.posterior(first, fitted["weights"], fitted["emissions"])
        errors["cold_scores"] = max(
            errors["cold_scores"], float(np.max(abs(scores - original)))
        )
        direct = np.tile(fitted["weights"], (len(first), 1))
        for v, matrix in enumerate(fitted["emissions"]):
            direct *= matrix[first[:, v]]
        direct /= direct.sum(axis=1, keepdims=True)
        errors["Bayes_enumeration"] = max(
            errors["Bayes_enumeration"], float(np.max(abs(direct - scores)))
        )
        perm = oracle_alignment(fitted, emissions)
        oracle = M.posterior(first, weights, emissions)
        metrics = read(directory / "metrics.json")
        reconstructed = M.joint_law(fitted["weights"], fitted["emissions"])
        values = {
            "emission_RMSE": float(
                np.sqrt(
                    np.mean(
                        [
                            (a[:, perm] - b) ** 2
                            for a, b in zip(fitted["emissions"], emissions, strict=True)
                        ]
                    )
                )
            ),
            "weight_max_error": float(
                np.max(abs(fitted["weights"][list(perm)] - weights))
            ),
            "sample_reconstruction_TV": float(abs(observed - reconstructed).sum() / 2),
            "posterior_RMSE": float(np.sqrt(np.mean((scores[:, perm] - oracle) ** 2))),
        }
        # Alternative direct predictive mixture enumeration, not the metric helper.
        for key, future in (("stable", stable_second), ("occasion", occasion_second)):
            likelihood = np.tile(np.ones(3), (len(future), 1))
            for v, matrix in enumerate(fitted["emissions"]):
                likelihood *= matrix[future[:, v]]
            gain = np.log(
                np.sum(scores * likelihood, axis=1)
                / np.sum(fitted["weights"] * likelihood, axis=1)
            )
            values[key + "_predictive_log_gain"] = float(gain.mean())
            values[key + "_gain_SE"] = float(gain.std(ddof=1) / np.sqrt(len(gain)))
            counts = np.zeros((3, 3))
            np.add.at(counts, (first[:, 0], future[:, 0]), 1)
            joint = counts / len(first)
            values[key + "_cross_occasion_cov_norm"] = float(
                np.linalg.norm(joint - np.outer(joint.sum(axis=1), joint.sum(axis=0)))
            )
        for key, value in values.items():
            errors["metrics"] = max(errors["metrics"], abs(metrics[key] - value))
        for row in saved_result["sample_metrics"]:
            if row["seed"] == seed and row != metrics:
                raise ValueError("Result-to-seed metric mismatch")
        replayed += len(first) * 3
    with (out / "metrics.csv").open() as stream:
        for row, expected in zip(
            csv.DictReader(stream), saved_result["sample_metrics"], strict=True
        ):
            for key, value in expected.items():
                if value is None:
                    if row[key] != "":
                        raise ValueError("Failure row CSV mismatch")
                elif isinstance(value, str):
                    if row[key] != value:
                        raise ValueError("CSV status mismatch")
                else:
                    errors["metrics"] = max(
                        errors["metrics"], abs(float(row[key]) - value)
                    )
    if max(errors.values()) > spec["exact_tolerance"]:
        raise ValueError("Numerical replay tolerance exceeded")
    return {
        "status": "READONLY_NUMERICAL_REPLAY",
        "errors": errors,
        "cold_probability_values_replayed": replayed,
        "scope": "Same-agent alternative counting/direct Bayes,not independent blind replication",
    }


def main():
    """Execute exactly one explicit lifecycle stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "verify"])
    args = parser.parse_args()
    out = ROOT / read(SPEC)["out"]
    with threadpool_limits(limits=1):
        result = {"prepare": prepare, "run": run, "verify": verify}[args.mode](out)
    print(json.dumps(result or {"mode": args.mode, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
