#!/usr/bin/env python3
"""Explicit target-clock adapter for frozen numerical T2/T3 instruments.

Only declared numerical observations enter here. Deciding whether natural text
is attributable evidence is a separate reader/history operation. Neither source
fits nor fixed reference norms are changed by this adapter.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import f16_independent_t3 as t3
from . import f16_state_instrument as t2

VERSION = "F16_explicit_target_clock_v1"
ROW_FIELDS = {"author_id", "role", "world", "event_id", "episode_id", "t", "z", "y"}


def object_hash(value):
    """Digest serializable values without permitting nonfinite JSON numbers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def code_hash():
    """Bind this adapter; each underlying scorer separately checks dependencies."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_base(base):
    """Authenticate the frozen target and its complete numerical source ensemble."""
    if base.get("target_id") == "T2":
        t2.validate_bundle(base)
        for item in base["source_models"]:
            t2.validate_model(item["model"])
    elif base.get("target_id") == "T3":
        t3.verify_bundle(base)
    else:
        raise ValueError("explicit clock adapter supports only T2/T3")
    if type(base["personal_draws"]) is not int or base["personal_draws"] < 2:
        raise ValueError("at least two personal integration draws required")


def build_adapter(base):
    """Wrap, not refit or edit, a source-frozen instrument."""
    validate_base(base)
    return {
        "schema_version": VERSION,
        "adapter_code_hash": code_hash(),
        "base_bundle_hash": object_hash(base),
        "base_bundle": copy.deepcopy(base),
        "target_clock": "requested nonnegative integer cutoff, not last retained event",
        "reference_rule": "original complete-prefix source reference; no batch or missingness renormalization",
    }


def verify_adapter(bundle):
    """Reject changed code or nested source bundle before reading observations."""
    if (
        bundle.get("schema_version") != VERSION
        or bundle.get("adapter_code_hash") != code_hash()
        or bundle.get("base_bundle_hash") != object_hash(bundle["base_bundle"])
    ):
        raise ValueError("adapter/source bundle identity mismatch")
    validate_base(bundle["base_bundle"])


def validate_input(raw, base):
    """Keep a single episode, actual time gaps and the requested target explicit."""
    if not isinstance(raw, dict) or set(raw) != {
        "world",
        "reader_hash",
        "target_time",
        "records",
    }:
        raise ValueError("wrong explicit-clock envelope")
    if raw["world"] != "core" or raw["reader_hash"] != base["reader_hash"]:
        raise ValueError("wrong observation world or reader")
    cutoff = raw["target_time"]
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("target_time must be a nonnegative integer")
    rows = raw["records"]
    if not isinstance(rows, list) or len(rows) > base.get("max_prefix", 48):
        raise ValueError("records outside frozen prefix budget")
    identity, previous, events = None, None, set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise ValueError("exactly eight numeric observation fields required")
        for key in ("author_id", "role", "world", "event_id", "episode_id"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError("nonempty grouping identifiers required")
        current = tuple(row[k] for k in ("author_id", "role", "world", "episode_id"))
        if identity is None:
            identity = current
        if (
            current != identity
            or row["world"] != "core"
            or row["role"] not in ("source", "validation", "application")
        ):
            raise ValueError("mixed or unsupported author/role/world/episode")
        time = row["t"]
        if type(time) is not int or time < 0 or time > cutoff:
            raise ValueError("observation outside requested target history")
        if previous is not None and time <= previous:
            raise ValueError("strict increasing observation times required")
        if row["event_id"] in events:
            raise ValueError("duplicate event")
        if any(
            type(row[k]) not in (int, float) or not np.isfinite(row[k])
            for k in ("z", "y")
        ):
            raise ValueError("finite numeric observations required")
        if row["z"] not in (-1, 1):
            raise ValueError("condition outside frozen core schedule")
        events.add(row["event_id"])
        previous = time
    return rows, cutoff


def propagate(mean, variance, parameters, log_sigma, steps):
    """Propagate each conditional state with no fabricated measurement update."""
    if type(steps) is not int or steps < 0:
        raise ValueError("nonnegative integer propagation gap required")
    rho = 0.99 * np.tanh(np.asarray(parameters, float)[:, 2])
    mean, variance = np.asarray(mean, float), np.asarray(variance, float)
    if mean.shape != rho.shape or variance.shape != rho.shape:
        raise ValueError("one state moment per parameter draw required")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
        or np.any(variance < 0)
    ):
        raise ValueError("invalid conditional state moments")
    if steps == 0:
        return mean.copy(), variance.copy()
    stationary = np.exp(2 * log_sigma[0]) / (1 - rho * rho)
    loading = rho**steps
    return mean * loading, variance * loading**2 + stationary * (1 - loading**2)


def parameter_draws(base, item, rows):
    """Reuse original source-conditioned inference and its exact integration RNG."""
    if base["target_id"] == "T3":
        observations = prepare_observations(rows)
        return t3.personal_draws(
            observations, item, base["seed"], base["personal_draws"]
        )
    model = item["model"]
    if rows:
        posterior = t2.laplace_person(rows, model)
        if posterior["status"] != "LAPLACE_APPROXIMATION":
            return None, posterior
        center = np.asarray(posterior["fit"]["psi"])
        covariance = np.asarray(posterior["parameter_covariance"])
    else:
        center = np.asarray(model["prior"]["mu"])
        covariance = np.asarray(model["prior"]["Sigma"])
    normal = np.random.default_rng(
        np.random.SeedSequence([base["seed"], 16, 2, item["bootstrap"]])
    ).normal(size=(base["personal_draws"], 3))
    return center + normal @ np.linalg.cholesky(covariance).T, {
        "parameter_center": center.tolist(),
        "parameter_covariance": covariance.tolist(),
    }


def prepare_observations(rows):
    """Convert already-validated records without compressing missing time steps."""
    return [
        (0 if i == 0 else row["t"] - rows[i - 1]["t"], float(row["z"]), float(row["y"]))
        for i, row in enumerate(rows)
    ]


def conditional_moments(target, parameters, rows, log_sigma, cutoff):
    """Return requested-time and old-last-record moments for the same parameters."""
    if rows and target == "T2":
        mean, variance = t2.filter_parameter_batch(parameters, rows, log_sigma)
    else:
        mean, variance, _ = t3.filter_parameters(
            parameters, prepare_observations(rows), log_sigma
        )
    lag = cutoff - rows[-1]["t"] if rows else 0
    shifted_mean, shifted_variance = propagate(
        mean, variance, parameters, log_sigma, lag
    )
    if target == "T3":
        old_mean, old_variance, _ = t3.predictive_moments(
            parameters, mean, variance, log_sigma
        )
        shifted_mean, shifted_variance, _ = t3.predictive_moments(
            parameters, shifted_mean, shifted_variance, log_sigma
        )
    else:
        old_mean, old_variance = mean, variance
    return shifted_mean, shifted_variance, old_mean, old_variance


def mixture(parts, prefix=""):
    """Keep conditional, personal-parameter and source uncertainty separate."""
    means = np.array([p[prefix + "mean"] for p in parts])
    components = {
        "event_state": float(np.mean([p[prefix + "event_state"] for p in parts])),
        "personal_parameter": float(
            np.mean([p[prefix + "personal_parameter"] for p in parts])
        ),
        "source": float(np.var(means)),
    }
    return float(means.mean()), components, sum(components.values())


def cold_score(raw, bundle, audit_callback=None):
    """Score only observed history at its requested clock; no truth/source files."""
    verify_adapter(bundle)
    base = bundle["base_bundle"]
    rows, cutoff = validate_input(raw, base)
    target = base["target_id"]
    ensemble = base["source_models"] if target == "T2" else base["ensemble"]
    parts = []
    for item in ensemble:
        entry = {"bootstrap": item["bootstrap"], "supported": False}
        if target == "T3" and item["status"] != "FITTED":
            parts.append({**entry, "reason": "SOURCE_NOT_FITTED"})
            continue
        try:
            parameters, diagnostics = parameter_draws(base, item, rows)
            if parameters is None:
                parts.append(
                    {
                        **entry,
                        "reason": "PERSONAL_FIT_UNSUPPORTED",
                        "diagnostics": diagnostics,
                    }
                )
                continue
            model = item["model"] if target == "T2" else item["fit"]
            m, v, old_m, old_v = conditional_moments(
                target, parameters, rows, model["log_sigma"], cutoff
            )
            if not all(np.isfinite(a).all() for a in (m, v, old_m, old_v)) or np.any(
                v < 0
            ):
                raise ArithmeticError("nonfinite or negative conditional moments")
            entry.update(
                supported=True,
                mean=float(m.mean()),
                event_state=float(v.mean()),
                personal_parameter=float(np.var(m)),
                old_mean=float(old_m.mean()),
                old_event_state=float(old_v.mean()),
                old_personal_parameter=float(np.var(old_m)),
            )
            if audit_callback is not None:
                audit_callback(item["bootstrap"], parameters, model["log_sigma"], m, v)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            entry.update(supported=False, reason="NUMERICAL_FAILURE", detail=str(exc))
        parts.append(entry)
    supported = all(p["supported"] for p in parts) and bool(parts)
    result = {
        "schema_version": VERSION,
        "target_id": target,
        "supported": supported,
        "target_time": cutoff,
        "last_observed_time": rows[-1]["t"] if rows else None,
        "propagation_steps": cutoff - rows[-1]["t"] if rows else None,
        "prior_only": not rows,
        "observed_count": len(rows),
        "source_count": len(parts),
        "unsupported_count": sum(not p["supported"] for p in parts),
        "future_times": list(range(cutoff + 1, cutoff + 5)) if target == "T3" else None,
        "future_conditions": base.get("future_conditions"),
        "score": None,
        "variance": None,
        "components": None,
        "interval": None,
        "reference_score": None,
        "reference_variance": None,
        "reference_rule": bundle["reference_rule"],
        "calibration_status": "NOT_EVALUATED",
        "support": "prior_only_model_borrowed"
        if not rows
        else "history_conditioned_model_borrowed",
        "interval_kind": "future_realization_predictive"
        if target == "T3"
        else "conditional_state_working",
        "old_last_observation_target": None,
        "per_source": parts,
    }
    if supported:
        score, components, variance = mixture(parts)
        old_score, _, old_variance = mixture(parts, "old_")
        radius = 1.96 * np.sqrt(variance)
        result.update(
            score=score,
            variance=variance,
            components=components,
            interval=[float(score - radius), float(score + radius)],
            reference_score=(score - base["reference_mu"]) / base["reference_sigma"],
            reference_variance=variance / base["reference_sigma"] ** 2,
            old_last_observation_target={
                "score": old_score,
                "variance": old_variance,
                "meaning": "diagnostic only: different target if trailing evidence is missing",
            }
            if rows
            else None,
        )
    return result


def main():
    """Cold CLI requires an externally expected hash of the complete bundle."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sha256", required=True)
    args = p.parse_args()
    if hashlib.sha256(args.bundle.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("untrusted adapter bundle")
    bundle = json.loads(args.bundle.read_text())
    with args.input.open() as inputs, args.output.open("x") as outputs:
        for line in inputs:
            outputs.write(
                json.dumps(cold_score(json.loads(line), bundle), allow_nan=False) + "\n"
            )
            outputs.flush()


if __name__ == "__main__":
    main()
