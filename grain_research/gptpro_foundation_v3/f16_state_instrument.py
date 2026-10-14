#!/usr/bin/env python3
"""Source-frozen T2 scoring with actual personal and source uncertainty."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .f13_contract import measurement_result
from .f13_dynamic_parameter import filter_parameter_batch, laplace_person
from .f13_uncertainty import validate_covariance

READER_HASH = hashlib.sha256(b"synthetic_numeric_identity_reader_v1").hexdigest()
FORMULA = "T2_source_laplace_mixture_v1"


def code_hash():
    """Bind all repository mathematical dependencies used by this scorer."""
    root = Path(__file__).parent
    names = [
        "f16_state_instrument.py",
        "f13_dynamic_parameter.py",
        "f05_profiles.py",
        "f05_recovery.py",
        "f05.py",
        "f13_contract.py",
        "f13_uncertainty.py",
    ]
    return hashlib.sha256(
        b"".join((root / name).read_bytes() for name in names)
    ).hexdigest()


def digest(path):
    """Return an input file's SHA256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_model(model):
    """Require finite nonsingular parameter prior and supported noise units."""
    mu = np.asarray(model["prior"]["mu"], float)
    covariance = validate_covariance(model["prior"]["Sigma"])
    logs = np.asarray(model["log_sigma"], float)
    if mu.shape != (3,) or covariance.shape != (3, 3) or logs.shape != (2,):
        raise ValueError("wrong source parameter shapes")
    if not np.isfinite(mu).all() or not np.isfinite(logs).all():
        raise ValueError("nonfinite source parameters")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("parameter prior must be positive definite")
    if np.any(logs < np.log(0.05)) or np.any(logs > np.log(5)):
        raise ValueError("noise outside frozen source fit bounds")


def validate_records(rows, world):
    """Reject mixed authors, duplicate events and coerced or unordered times."""
    if not rows:
        return
    allowed = {"author_id", "event_id", "episode_id", "role", "world", "t", "z", "y"}
    if any(set(r) - allowed for r in rows):
        raise ValueError("undeclared observation fields")
    if len({r["author_id"] for r in rows}) != 1 or len({r["role"] for r in rows}) != 1:
        raise ValueError("one author and role required")
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate event")
    previous = None
    for r in rows:
        if r["world"] != world or r["role"] not in (
            "source",
            "validation",
            "application",
        ):
            raise ValueError("wrong observation world or role")
        t = r["t"]
        if type(t) is not int or t < 0 or (previous is not None and t <= previous):
            raise ValueError("strictly increasing integer time required")
        if any(
            type(r[k]) not in (int, float) or not np.isfinite(r[k]) for k in ("z", "y")
        ):
            raise ValueError("finite numeric condition and observation required")
        if r["z"] not in (-1, 1):
            raise ValueError("unsupported condition")
        previous = t


def source_template(brief, source_file, bootstrap_root, seed, count=200, draws=200):
    """Verify actual source fits and embed them; perform no source refitting."""
    if brief != {"target_id": "T2", "world": "core", "target": "current_state"}:
        raise ValueError("unsupported T2 brief")
    if count < 1 or draws < 2:
        raise ValueError(
            "positive source count and at least two personal draws required"
        )
    rows = [json.loads(line) for line in Path(source_file).read_text().splitlines()]
    if not rows or any(r["role"] != "source" or r["world"] != "core" for r in rows):
        raise ValueError("core source records required")
    source_ids = {r["author_id"] for r in rows}
    fits, hashes = [], {"source": digest(source_file)}
    for b in range(count):
        path = Path(bootstrap_root) / f"fit_{b:03d}.json"
        artifact = json.loads(path.read_text())
        if (
            artifact["status"] != "FITTED"
            or artifact["seed"] != seed
            or artifact["world"] != "core"
            or artifact["bootstrap"] != b
            or set(artifact["source_ids"]) != source_ids
        ):
            raise ValueError("source bootstrap identity or fit failure")
        model = {k: artifact["fit"][k] for k in ("prior", "log_sigma")}
        validate_model(model)
        fits.append({"bootstrap": b, "model": model})
        hashes[f"bootstrap_{b:03d}"] = digest(path)
    return {
        "target_id": "T2",
        "object_class": "dynamic_state",
        "formula_version": FORMULA,
        "world_contract": "F05_core_single_state",
        "world": "core",
        "seed": seed,
        "reader_hash": READER_HASH,
        "scorer_code_hash": code_hash(),
        "source_hash": hashes,
        "unit": "latent response units",
        "observed_role": "synthetic observed response prefix",
        "invalidating_result": "state/intercept ambiguity, model misspecification or failed approximation invalidates reliable state interpretation",
        "support_rule": "latent state is model inferred; empty prefix is explicitly prior-only",
        "error_model": "empirical source ensemble plus personal Gaussian Laplace and conditional state covariance; not exact psychological posterior",
        "explanation_rule": "conditional latent state in the F05 law; no emotion/personality name inferred",
        "reference_mu": None,
        "reference_sigma": None,
        "reference_prefix": 48,
        "source_reference_authors": len(source_ids),
        "personal_draws": draws,
        "source_models": fits,
    }


def state_moments(rows, bundle):
    """Integrate real state estimates; never drop an unsupported source fit."""
    validate_records(rows, bundle["world"])
    records = []
    for item in bundle["source_models"]:
        model = item["model"]
        validate_model(model)
        if rows:
            posterior = laplace_person(rows, model)
            if posterior["status"] != "LAPLACE_APPROXIMATION":
                return {
                    "status": "UNSUPPORTED_PERSONAL_FIT",
                    "bootstrap": item["bootstrap"],
                    "failure": posterior,
                    "components": None,
                }
            center = np.asarray(posterior["fit"]["psi"])
            covariance = np.asarray(posterior["parameter_covariance"])
        else:
            center = np.asarray(model["prior"]["mu"])
            covariance = np.asarray(model["prior"]["Sigma"])
        recipe = [bundle["seed"], 16, 2, item["bootstrap"]]
        normal = np.random.default_rng(np.random.SeedSequence(recipe)).normal(
            size=(bundle["personal_draws"], 3)
        )
        parameters = center + normal @ np.linalg.cholesky(covariance).T
        if rows:
            means, variances = filter_parameter_batch(
                parameters, rows, model["log_sigma"]
            )
        else:
            rho = 0.99 * np.tanh(parameters[:, 2])
            means = np.zeros(len(parameters))
            variances = np.exp(2 * model["log_sigma"][0]) / (1 - rho**2)
        records.append(
            {
                "bootstrap": item["bootstrap"],
                "mean": float(means.mean()),
                "event_state": float(variances.mean()),
                "personal_parameter": float(np.var(means)),
                "parameter_center": center.tolist(),
                "parameter_covariance": covariance.tolist(),
            }
        )
    centers = np.array([r["mean"] for r in records])
    components = {
        "event_state": float(np.mean([r["event_state"] for r in records])),
        "personal_parameter": float(
            np.mean([r["personal_parameter"] for r in records])
        ),
        "source": float(np.var(centers)),
    }
    return {
        "status": "MOMENTS_COMPUTED",
        "score": float(centers.mean()),
        "variance": sum(components.values()),
        "components": components,
        "source_components": records,
        "prior_only": not rows,
    }


def finalize_reference(template, source_results):
    """Freeze norms from the same source-only scorer, preserving failures."""
    if len(source_results) != template["source_reference_authors"]:
        raise ValueError("incomplete source reference")
    if any(r["status"] != "MOMENTS_COMPUTED" for r in source_results):
        raise ValueError("unsupported source reference cannot be silently omitted")
    values = np.array([r["score"] for r in source_results])
    sd = float(values.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("nondegenerate reference required")
    return dict(template, reference_mu=float(values.mean()), reference_sigma=sd)


def validate_bundle(bundle):
    """Validate the declared target, frozen references and complete ensemble."""
    required = {
        "target_id",
        "object_class",
        "invalidating_result",
        "formula_version",
        "observed_role",
        "unit",
        "reference_mu",
        "reference_sigma",
        "source_hash",
        "reader_hash",
        "world_contract",
        "scorer_code_hash",
        "support_rule",
        "error_model",
        "explanation_rule",
        "source_models",
        "personal_draws",
        "seed",
    }
    if not required <= bundle.keys():
        raise ValueError("incomplete bundle")
    if (
        bundle["scorer_code_hash"] != code_hash()
        or bundle["formula_version"] != FORMULA
        or bundle["target_id"] != "T2"
        or bundle["object_class"] != "dynamic_state"
        or bundle["world_contract"] != "F05_core_single_state"
        or bundle["world"] != "core"
        or bundle["reader_hash"] != READER_HASH
    ):
        raise ValueError("incompatible scorer/target/world/reader")
    if (
        not np.isfinite([bundle["reference_mu"], bundle["reference_sigma"]]).all()
        or bundle["reference_sigma"] <= 0
    ):
        raise ValueError("invalid reference")
    fits = bundle["source_models"]
    if not fits or [f["bootstrap"] for f in fits] != list(range(len(fits))):
        raise ValueError("incomplete ordered source ensemble")


def cold_score(raw, bundle):
    """Score a new prefix using only the self-contained bundle and input."""
    validate_bundle(bundle)
    if set(raw) != {"world_contract", "reader_hash", "records"}:
        raise ValueError("undeclared envelope fields")
    if (
        raw["world_contract"] != bundle["world_contract"]
        or raw["reader_hash"] != READER_HASH
    ):
        raise ValueError("wrong input world/reader")
    moments = state_moments(raw["records"], bundle)
    if moments["status"] != "MOMENTS_COMPUTED":
        return moments
    result = measurement_result(
        moments["score"],
        moments["variance"],
        {"target": "T2", "world": bundle["world_contract"], "method": FORMULA},
        "model_borrowed",
    )
    result.update(
        status=moments["status"],
        components=moments["components"],
        source_components=moments["source_components"],
        prior_only=moments["prior_only"],
        reference_score=(moments["score"] - bundle["reference_mu"])
        / bundle["reference_sigma"],
        reference_variance=moments["variance"] / bundle["reference_sigma"] ** 2,
        source_uncertainty_included=True,
        personal_approximation="Gaussian_Laplace",
        prefix_observations=len(raw["records"]),
    )
    return result


def main():
    """Trusted-bundle cold CLI; source files and outcome truth are not inputs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if digest(args.bundle) != args.sha256:
        raise ValueError("untrusted bundle hash")
    bundle = json.loads(args.bundle.read_text())
    with args.output.open("x") as stream:
        for line in args.input.read_text().splitlines():
            result = cold_score(json.loads(line), bundle)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
