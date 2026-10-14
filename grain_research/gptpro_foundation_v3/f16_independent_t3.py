#!/usr/bin/env python3
"""Independent numeric T3 construction from the 2026-09-23 handbook only.

No repository imports. Source fitting and application evaluation are deliberately
absent. IDs validate grouping; they never enter a mathematical calculation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

FORMULA_VERSION = "independent_t3_v1"
READER_HASH = hashlib.sha256(b"F16-T3:numeric-y-z-t:stationary-AR1:v1").hexdigest()
SETTINGS = {
    "optimizer": "L-BFGS-B",
    "rho_starts": [0.2, 0.8],
    "maxiter": 100,
    "ftol": 1e-9,
    "gtol": 1e-6,
    "hessian_step": 1e-4,
    "rng": "numpy.random.PCG64(SeedSequence([seed,16,3,bootstrap]))",
    "ddof_mixtures": 0,
    "ddof_reference": 1,
}


class ContractError(ValueError):
    """An input, source identity or frozen-bundle contract was violated."""


def strict_json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def object_hash(value):
    return hashlib.sha256(strict_json_bytes(value)).hexdigest()


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _finite_array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid {label}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ContractError(f"invalid {label}: expected finite shape {shape}")
    return result


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def prepare_input(value, bundle):
    if not isinstance(value, dict) or set(value) != {"world", "reader_hash", "prefix"}:
        raise ContractError("cold input must contain only world, reader_hash, prefix")
    if value["world"] != bundle["world_contract"]["world"]:
        raise ContractError("wrong world")
    if value["reader_hash"] != bundle["reader_hash"]:
        raise ContractError("wrong reader_hash")
    rows = value["prefix"]
    if not isinstance(rows, list) or len(rows) > bundle["max_prefix"]:
        raise ContractError("prefix must be a list no longer than the frozen prefix")
    required = {"author_id", "role", "world", "event_id", "episode_id", "t", "z", "y"}
    events = set()
    identity = None
    previous = None
    prepared = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ContractError(
                "observed rows must have exactly the eight documented fields"
            )
        if any(
            not isinstance(row[key], str) or not row[key]
            for key in ("author_id", "role", "world", "event_id", "episode_id")
        ):
            raise ContractError("grouping identifiers must be nonempty strings")
        current = tuple(
            row[key] for key in ("author_id", "world", "role", "episode_id")
        )
        if identity is None:
            identity = current
        if current != identity:
            raise ContractError("mixed authors/worlds/roles/episodes")
        if row["world"] != value["world"] or row["role"] not in bundle["observed_role"]:
            raise ContractError("wrong observation world or role")
        if row["event_id"] in events:
            raise ContractError("duplicate event")
        events.add(row["event_id"])
        t = row["t"]
        if type(t) is not int or (previous is not None and t <= previous):
            raise ContractError("times must be strictly increasing integers")
        for key in ("z", "y"):
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)):
                raise ContractError(f"{key} must be numeric")
            if not math.isfinite(row[key]):
                raise ContractError(f"{key} must be finite")
        prepared.append(
            (0 if previous is None else t - previous, float(row["z"]), float(row["y"]))
        )
        previous = t
    return prepared


def filter_parameters(parameters, observations, log_sigma):
    """Stationary initialization, exact gap propagation; vectorized over draws."""
    psi = np.atleast_2d(np.asarray(parameters, dtype=float))
    q, r = np.exp(2 * np.asarray(log_sigma, dtype=float))
    rho = 0.99 * np.tanh(psi[:, 2])
    stationary = q / (1 - rho * rho)
    mean = np.zeros(len(psi))
    variance = stationary.copy()
    loglik = np.zeros(len(psi))
    for index, (gap, z, y) in enumerate(observations):
        if index:
            transition = rho**gap
            mean = transition * mean
            variance = transition**2 * variance + stationary * (1 - transition**2)
        innovation = y - psi[:, 0] - psi[:, 1] * z - mean
        predictive = variance + r
        loglik -= 0.5 * (
            math.log(2 * math.pi) + np.log(predictive) + innovation**2 / predictive
        )
        gain = variance / predictive
        mean += gain * innovation
        variance = r * variance / predictive
    return mean, variance, loglik


def objective_and_gradient(psi, observations, mu, precision, log_sigma):
    """Independent forward derivatives of the scalar Kalman likelihood."""
    a, b, eta = map(float, psi)
    q, r = map(float, np.exp(2 * np.asarray(log_sigma)))
    tanh_eta = math.tanh(eta)
    rho = 0.99 * tanh_eta
    drho = 0.99 * (1 - tanh_eta * tanh_eta)
    stationary = q / (1 - rho * rho)
    dstationary = 2 * q * rho * drho / (1 - rho * rho) ** 2
    mean = ma = mb = me = 0.0
    variance, pe = stationary, dstationary
    nll = ga = gb = ge = 0.0
    for index, (gap, z, y) in enumerate(observations):
        if index:
            transition = rho**gap
            dt = gap * rho ** (gap - 1) * drho
            ma, mb, me = transition * ma, transition * mb, transition * me + dt * mean
            mean *= transition
            pe = (
                transition**2 * pe
                + 2 * transition * dt * (variance - stationary)
                + dstationary * (1 - transition**2)
            )
            variance = transition**2 * variance + stationary * (1 - transition**2)
        innovation = y - a - b * z - mean
        ea, eb, ee = -1 - ma, -z - mb, -me
        predictive = variance + r
        scaled = innovation / predictive
        nll += 0.5 * (
            math.log(2 * math.pi) + math.log(predictive) + innovation * scaled
        )
        ga += scaled * ea
        gb += scaled * eb
        ge += scaled * ee + 0.5 * (1 / predictive - scaled * scaled) * pe
        gain = variance / predictive
        d_gain = r * pe / predictive**2
        mean += gain * innovation
        ma, mb, me = (
            ma + gain * ea,
            mb + gain * eb,
            me + d_gain * innovation + gain * ee,
        )
        variance = r * variance / predictive
        pe *= r * r / predictive**2
    delta = np.asarray(psi) - mu
    prior_gradient = precision @ delta
    return nll + 0.5 * float(delta @ prior_gradient), np.array(
        [ga, gb, ge]
    ) + prior_gradient


def central_hessian(function, point, step=1e-4):
    point = np.asarray(point, dtype=float)
    hessian = np.empty((3, 3))
    center = function(point)
    basis = np.eye(3) * step
    for i in range(3):
        hessian[i, i] = (
            function(point + basis[i]) - 2 * center + function(point - basis[i])
        ) / step**2
        for j in range(i):
            cross = (
                function(point + basis[i] + basis[j])
                - function(point + basis[i] - basis[j])
                - function(point - basis[i] + basis[j])
                + function(point - basis[i] - basis[j])
            ) / (4 * step**2)
            hessian[i, j] = hessian[j, i] = cross
    return hessian


def personal_draws(observations, model, seed, n_draws):
    prior = model["fit"]["prior"]
    mu = np.asarray(prior["mu"], dtype=float)
    sigma = np.asarray(prior["Sigma"], dtype=float)
    xi = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, 16, 3, model["bootstrap"]]))
    ).standard_normal((n_draws, 3))
    if not observations:
        return mu + xi @ np.linalg.cholesky(sigma).T, {"prior_only": True, "starts": []}
    precision = np.linalg.inv(sigma)
    z = np.asarray([row[1] for row in observations])
    y = np.asarray([row[2] for row in observations])
    slope = float(
        np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]), y, rcond=None)[0][1]
    )

    def objective(psi):
        return objective_and_gradient(
            psi, observations, mu, precision, model["fit"]["log_sigma"]
        )

    candidates, starts = [], []
    for rho in SETTINGS["rho_starts"]:
        result = minimize(
            objective,
            [float(np.mean(y)), slope, math.atanh(rho / 0.99)],
            method="L-BFGS-B",
            jac=True,
            options={key: SETTINGS[key] for key in ("maxiter", "ftol", "gtol")},
        )
        finite = bool(np.isfinite(result.fun) and np.isfinite(result.x).all())
        starts.append(
            {
                "rho": rho,
                "success": bool(result.success),
                "finite": finite,
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "objective": float(result.fun) if finite else None,
            }
        )
        if result.success and finite:
            candidates.append(result)
    diagnostics = {"prior_only": False, "starts": starts}
    if not candidates:
        diagnostics["reason"] = "NO_CONVERGED_PERSONAL_MAP"
        return None, diagnostics
    result = min(candidates, key=lambda value: value.fun)
    hessian = central_hessian(
        lambda psi: objective(psi)[0], result.x, SETTINGS["hessian_step"]
    )
    if not np.isfinite(hessian).all():
        diagnostics["reason"] = "NONFINITE_HESSIAN"
        return None, diagnostics
    diagnostics.update(
        {
            "map": result.x.tolist(),
            "hessian_eigenvalues": np.linalg.eigvalsh(hessian).tolist(),
        }
    )
    try:
        np.linalg.cholesky(hessian)
        lower = np.linalg.cholesky(np.linalg.inv(hessian))
    except np.linalg.LinAlgError:
        diagnostics["reason"] = "NONPOSITIVE_HESSIAN"
        return None, diagnostics
    draws = result.x + xi @ lower.T
    if not np.isfinite(draws).all():
        diagnostics["reason"] = "NONFINITE_PARAMETER_DRAWS"
        return None, diagnostics
    return draws, diagnostics


def predictive_moments(parameters, terminal_mean, terminal_variance, log_sigma):
    """Scalar target and a separately assembled full four-observation covariance."""
    psi = np.atleast_2d(np.asarray(parameters, dtype=float))
    rho = 0.99 * np.tanh(psi[:, 2])
    q, r = np.exp(2 * np.asarray(log_sigma, dtype=float))
    powers = rho[:, None] ** np.arange(1, 5)
    average_loading = powers.mean(axis=1)
    score = psi[:, 0] + terminal_mean * average_loading
    state_variance = terminal_variance * average_loading**2
    innovation_variance = np.zeros(len(psi))
    for j in range(1, 5):
        innovation_variance += (
            q / 16 * np.sum(rho[:, None] ** np.arange(5 - j), axis=1) ** 2
        )
    noise_variance = np.full(len(psi), r / 4)
    scalar = state_variance + innovation_variance + noise_variance
    covariance = np.empty((len(psi), 4, 4))
    for k in range(1, 5):
        for h in range(1, 5):
            cell = terminal_variance * rho ** (k + h)
            for j in range(1, min(k, h) + 1):
                cell = cell + q * rho ** (k - j) * rho ** (h - j)
            if k == h:
                cell = cell + r
            covariance[:, k - 1, h - 1] = cell
    projected = covariance.sum(axis=(1, 2)) / 16
    error = float(np.max(np.abs(scalar - projected)))
    if not np.allclose(scalar, projected, rtol=1e-11, atol=1e-12):
        raise ArithmeticError("scalar/full-covariance predictive variance mismatch")
    return (
        score,
        scalar,
        {
            "terminal_state": state_variance,
            "future_innovation": innovation_variance,
            "future_observation": noise_variance,
            "covariance_max_abs_error": error,
        },
    )


def _score(value, bundle):
    observations = prepare_input(value, bundle)
    per_source = []
    for model in bundle["ensemble"]:
        base = {"bootstrap": model["bootstrap"], "source_status": model["status"]}
        if model["status"] != "FITTED":
            per_source.append(
                {**base, "supported": False, "reason": "SOURCE_NOT_FITTED"}
            )
            continue
        try:
            draws, diagnostics = personal_draws(
                observations, model, bundle["seed"], bundle["personal_draws"]
            )
            if draws is None:
                per_source.append({**base, "supported": False, **diagnostics})
                continue
            mean, variance, _ = filter_parameters(
                draws, observations, model["fit"]["log_sigma"]
            )
            scores, conditional, components = predictive_moments(
                draws, mean, variance, model["fit"]["log_sigma"]
            )
            personal = float(np.var(scores, ddof=0))
            event_state = float(np.mean(conditional))
            result = {
                **base,
                "supported": True,
                "mean": float(np.mean(scores)),
                "event_state": event_state,
                "personal_parameter": personal,
                "within_total": event_state + personal,
                "terminal_state": float(np.mean(components["terminal_state"])),
                "future_innovation": float(np.mean(components["future_innovation"])),
                "future_observation": float(np.mean(components["future_observation"])),
                "covariance_max_abs_error": components["covariance_max_abs_error"],
                "diagnostics": diagnostics,
            }
            strict_json_bytes(result)
            if result["event_state"] < 0 or result["personal_parameter"] < 0:
                raise ArithmeticError("negative variance")
            per_source.append(result)
        except (
            FloatingPointError,
            OverflowError,
            ArithmeticError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            per_source.append(
                {
                    **base,
                    "supported": False,
                    "reason": "NUMERICAL_FAILURE",
                    "detail": str(exc),
                }
            )
    supported = len(per_source) == bundle["source_bootstraps"] and all(
        row["supported"] for row in per_source
    )
    result = {
        "target_id": "T3",
        "supported": supported,
        "prior_only": not observations,
        "observed_count": len(observations),
        "source_count": len(per_source),
        "unsupported_count": sum(not row["supported"] for row in per_source),
        "raw_score": None,
        "variance_components": None,
        "interval": None,
        "source_standardized_score": None,
        "source_standardized_variance": None,
        "approximation_status": "WORKING_BOOTSTRAP_LAPLACE_MOMENTS_NOT_EXACT_POSTERIOR",
        "calibration_status": "NOT_EVALUATED",
        "per_source": per_source,
    }
    if not supported:
        return result
    components = {
        key: float(np.mean([row[key] for row in per_source]))
        for key in (
            "event_state",
            "personal_parameter",
            "terminal_state",
            "future_innovation",
            "future_observation",
        )
    }
    components["source"] = float(np.var([row["mean"] for row in per_source], ddof=0))
    components["total"] = (
        components["event_state"]
        + components["personal_parameter"]
        + components["source"]
    )
    raw_score = float(np.mean([row["mean"] for row in per_source]))
    radius = 1.96 * math.sqrt(components["total"])
    result.update(
        {
            "raw_score": raw_score,
            "variance_components": components,
            "interval": [raw_score - radius, raw_score + radius],
        }
    )
    if bundle["reference_sigma"] is not None:
        result["source_standardized_score"] = (
            raw_score - bundle["reference_mu"]
        ) / bundle["reference_sigma"]
        result["source_standardized_variance"] = (
            components["total"] / bundle["reference_sigma"] ** 2
        )
    return result


def _verify_ensemble(ensemble, seed, world, source_ids, expected):
    if len(ensemble) != expected:
        raise ContractError("wrong number of source bootstrap fits; cannot renormalize")
    if [model.get("bootstrap") for model in ensemble] != list(range(expected)):
        raise ContractError("source bootstrap indices must be complete and ordered")
    for model in ensemble:
        if model.get("seed") != seed or model.get("world") != world:
            raise ContractError("source fit seed/world mismatch")
        ids = model.get("source_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != len(source_ids)
            or set(ids) != set(source_ids)
        ):
            raise ContractError("source identity mismatch")
        if not isinstance(model.get("status"), str):
            raise ContractError("missing source fit status")
        if model["status"] != "FITTED":
            continue
        prior = model.get("fit", {}).get("prior", {})
        _finite_array(prior.get("mu"), (3,), "prior mean")
        sigma = _finite_array(prior.get("Sigma"), (3, 3), "prior covariance")
        if not np.allclose(sigma, sigma.T, rtol=0, atol=1e-12):
            raise ContractError("nonsymmetric source prior")
        try:
            np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError as exc:
            raise ContractError("nonpositive source prior") from exc
        noise = _finite_array(model["fit"].get("log_sigma"), (2,), "log_sigma")
        with np.errstate(over="ignore", under="ignore"):
            variances = np.exp(2 * noise)
        if not np.isfinite(variances).all() or (variances <= 0).any():
            raise ContractError("nonpositive/nonfinite source noise variance")


def _draft_bundle(brief, manifest):
    if brief.get("T3", {}).get("target_id") != "T3" or brief["T3"].get(
        "future_conditions"
    ) != [1, -1, 1, -1]:
        raise ContractError("wrong target or future conditions")
    counts = {
        key: _positive_integer(brief.get(key), key)
        for key in (
            "source_authors",
            "reference_prefix",
            "application_prefix",
            "source_bootstraps",
            "personal_draws",
        )
    }
    if counts["source_authors"] < 2:
        raise ContractError("sample reference SD requires at least two authors")
    seed, world = manifest["seed"], manifest["world"]
    if seed not in brief["seeds"] or world != brief["world"]:
        raise ContractError("manifest seed/world not in frozen brief")
    records = manifest["source_prefixes"]
    source_ids = sorted(records)
    if len(source_ids) != counts["source_authors"]:
        raise ContractError("wrong source author count")
    if object_hash(manifest["provenance"]) != manifest["source_hash"]:
        raise ContractError("source manifest hash mismatch")
    _verify_ensemble(
        manifest["ensemble"], seed, world, source_ids, counts["source_bootstraps"]
    )
    bundle = {
        "target_id": "T3",
        "object_class": "predictive_state",
        "formula_version": FORMULA_VERSION,
        "invalidating_result": "Any unsupported source score, hash/identity mismatch or failed covariance equality blocks reference construction.",
        "observed_role": ["source", "application"],
        "unit": brief["T3"]["unit"],
        "reference_mu": None,
        "reference_sigma": None,
        "source_hash": manifest["source_hash"],
        "reader_hash": READER_HASH,
        "world_contract": {"world": world, "synthetic_only": True},
        "scorer_code_hash": code_hash(),
        "support_rule": "Every frozen replicate must succeed; no clipping, Hessian repair, fallback score or renormalization. Empty prefix is prior-only.",
        "error_model": "stationary Gaussian AR(1) state plus independent Gaussian observation noise; bootstrap/Laplace moment mixture",
        "explanation_rule": "Report event/state, personal-parameter and source uncertainty separately; interval calibration NOT_EVALUATED; not human H06.",
        "seed": seed,
        "ensemble": manifest["ensemble"],
        "source_provenance": manifest["provenance"],
        "ensemble_hash": object_hash(manifest["ensemble"]),
        "source_ids": source_ids,
        "brief_hash": object_hash(brief),
        "algorithm": dict(SETTINGS),
        "future_conditions": brief["T3"]["future_conditions"],
        "source_bootstraps": counts["source_bootstraps"],
        "personal_draws": counts["personal_draws"],
        "max_prefix": max(counts["reference_prefix"], counts["application_prefix"]),
        "reference_prefix": counts["reference_prefix"],
        "source_authors": counts["source_authors"],
        "construction_status": "CHECKPOINT_ONLY",
        "calibration_status": "NOT_EVALUATED",
    }
    for author, rows in records.items():
        if len(rows) != counts["reference_prefix"] or any(
            row["author_id"] != author or row["role"] != "source" for row in rows
        ):
            raise ContractError(
                "source reference needs exactly the documented source prefix per author"
            )
        prepare_input(
            {"world": world, "reader_hash": READER_HASH, "prefix": rows}, bundle
        )
    return bundle


def build_instrument(
    brief,
    source_manifest,
    *,
    checkpoint=None,
    checkpoint_callback=None,
    max_new_authors=None,
):
    """Build source reference, or return a visibly non-scoreable checkpoint.

    The callback receives a complete checkpoint after every ten new authors,
    on an unsupported author, and at the requested stopping boundary.
    """
    bundle = _draft_bundle(brief, source_manifest)
    identity = {
        "source_hash": bundle["source_hash"],
        "scorer_code_hash": bundle["scorer_code_hash"],
        "brief_hash": bundle["brief_hash"],
        "ensemble_hash": bundle["ensemble_hash"],
        "seed": bundle["seed"],
    }
    records = []
    if checkpoint is not None:
        if checkpoint.get("identity") != identity:
            raise ContractError(
                "checkpoint source/code/brief/ensemble identity mismatch"
            )
        records = list(checkpoint["records"])
        if checkpoint.get("records_hash") != object_hash(records):
            raise ContractError("checkpoint records hash mismatch")
        if len(records) > len(bundle["source_ids"]):
            raise ContractError("too many checkpoint records")
        for index, record in enumerate(records):
            author = bundle["source_ids"][index]
            if record["author_id"] != author or record["input_hash"] != object_hash(
                source_manifest["source_prefixes"][author]
            ):
                raise ContractError("checkpoint author/input mismatch")
            if not record["score"]["supported"]:
                bundle["construction_status"] = "UNSUPPORTED_SOURCE_REFERENCE"
                return bundle, checkpoint
    if max_new_authors is not None:
        _positive_integer(max_new_authors, "max_new_authors")

    def snapshot():
        return {
            "identity": identity,
            "records": records,
            "records_hash": object_hash(records),
            "completed_authors": len(records),
            "required_authors": bundle["source_authors"],
            "construction_status": bundle["construction_status"],
        }

    for new_count, author in enumerate(bundle["source_ids"][len(records) :], start=1):
        rows = source_manifest["source_prefixes"][author]
        score = _score(
            {
                "world": source_manifest["world"],
                "reader_hash": READER_HASH,
                "prefix": rows,
            },
            bundle,
        )
        records.append(
            {"author_id": author, "input_hash": object_hash(rows), "score": score}
        )
        if not score["supported"]:
            bundle["construction_status"] = "UNSUPPORTED_SOURCE_REFERENCE"
        if checkpoint_callback is not None and (
            new_count % 10 == 0 or not score["supported"]
        ):
            checkpoint_callback(snapshot())
        if not score["supported"] or (
            max_new_authors is not None and new_count >= max_new_authors
        ):
            break
    if len(records) == bundle["source_authors"] and all(
        row["score"]["supported"] for row in records
    ):
        values = np.asarray([row["score"]["raw_score"] for row in records])
        reference_sigma = float(np.std(values, ddof=1))
        if not math.isfinite(reference_sigma) or reference_sigma <= 0:
            bundle["construction_status"] = "UNSUPPORTED_SOURCE_REFERENCE"
        else:
            bundle.update(
                {
                    "reference_mu": float(np.mean(values)),
                    "reference_sigma": reference_sigma,
                    "reference_records_hash": object_hash(records),
                    "construction_status": "SOURCE_REFERENCE_FROZEN",
                }
            )
    final_checkpoint = snapshot()
    if checkpoint_callback is not None:
        checkpoint_callback(final_checkpoint)
    return bundle, final_checkpoint


def verify_bundle(bundle, expected_source_hash=None):
    if bundle.get("construction_status") != "SOURCE_REFERENCE_FROZEN":
        raise ContractError(
            "source reference is not frozen; checkpoint is not a cold instrument"
        )
    if bundle.get("scorer_code_hash") != code_hash():
        raise ContractError("scorer implementation hash mismatch")
    if (
        bundle.get("formula_version") != FORMULA_VERSION
        or bundle.get("algorithm") != SETTINGS
    ):
        raise ContractError("formula/algorithm mismatch")
    if bundle.get("target_id") != "T3" or bundle.get("future_conditions") != [
        1,
        -1,
        1,
        -1,
    ]:
        raise ContractError("target mismatch")
    if bundle.get("reader_hash") != READER_HASH:
        raise ContractError("reader mismatch")
    if object_hash(bundle["source_provenance"]) != bundle["source_hash"]:
        raise ContractError("source hash mismatch")
    if (
        expected_source_hash is not None
        and bundle["source_hash"] != expected_source_hash
    ):
        raise ContractError("externally expected source hash mismatch")
    if object_hash(bundle["ensemble"]) != bundle["ensemble_hash"]:
        raise ContractError("embedded source ensemble hash mismatch")
    _verify_ensemble(
        bundle["ensemble"],
        bundle["seed"],
        bundle["world_contract"]["world"],
        bundle["source_ids"],
        bundle["source_bootstraps"],
    )
    if (
        not math.isfinite(bundle["reference_mu"])
        or not math.isfinite(bundle["reference_sigma"])
        or bundle["reference_sigma"] <= 0
    ):
        raise ContractError("invalid frozen reference")


def cold_score(input, bundle):
    verify_bundle(bundle)
    return _score(input, bundle)


def load_bundle(path, expected_sha256, expected_source_hash=None):
    """Authenticate exact bytes before parsing; never open embedded source paths."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ContractError("external expected SHA256 required")
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ContractError("bundle SHA256 mismatch")

    def reject_constant(value):
        raise ContractError(f"non-JSON constant: {value}")

    bundle = json.loads(payload, parse_constant=reject_constant)
    verify_bundle(bundle, expected_source_hash)
    return bundle
