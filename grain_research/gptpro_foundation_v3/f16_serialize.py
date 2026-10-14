#!/usr/bin/env python3
"""Versioned public calibrated F16 bundles and a sealed cold scoring process.

A bundle binds one target, clock, condition plan, unit, changed reader, source
ensemble, calibration data provenance, every calibration draw, the integration
rule and an explicit reference policy. The cold scorer reads only that bundle
and an observation envelope. Numeric Cartesian source-by-pilot integration and
categorical paired source/channel Monte Carlo remain different estimators.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import f16_dynamic_text as dyn_text
from . import f16_dynamic_transport as dyn_law
from . import f16_t1_source as t1_source
from . import f16_t1_text as t1_text
from . import f16_t1_transport as t1_law
from .f12_transport import fit_transport, transport_score

VERSION = "F16_public_calibrated_bundle_v1"
CODE_FILES = (
    "f16_serialize.py",
    "f16_t1_source.py",
    "f16_t1_transport.py",
    "f12_transport.py",
    "f16_dynamic_transport.py",
    "f16_t1_text.py",
    "f16_dynamic_text.py",
)
CLASSES = {
    "T1": "conditional_response_function",
    "T2": "dynamic_state",
    "T3": "predictive_state",
}
ESTIMANDS = {
    "T1": "m(+1)-m(-1): conditional endpoint contrast at target time 31",
    "T2": "E[s_47 | observed prefix]: current latent state, not an observed emotion",
    "T3": "E[mean(y_48..y_51) | prefix, plan +1,-1,+1,-1]; interval is predictive and includes future noise",
}
ROUTE_TARGETS = {
    "numeric_t1": ("T1",),
    "numeric_dynamic": ("T2", "T3"),
    "text_t1": ("T1",),
    "text_dynamic": ("T2", "T3"),
}
INTEGRATION = {
    "numeric_t1": "cartesian_source_by_pilot_exact_v1",
    "numeric_dynamic": "cartesian_source_by_pilot_eta_gauss_hermite64_v1",
    "text_t1": "paired_source_channel_monte_carlo_qmc_v1",
    "text_dynamic": "paired_source_channel_monte_carlo_qmc_adf_v1",
}
OBSERVATION_CONTRACTS = {
    "numeric_t1": "numeric_changed_reader_four_condition_v1",
    "numeric_dynamic": "numeric_changed_reader_complete_prefix48_v1",
    "text_t1": "F04_L_thresholded_text_eligibility_v1",
    "text_dynamic": dyn_text.CONTRACT,
}
REFERENCE_KINDS = {
    "numeric_t1": "INHERITED_COMMON_UNIT",
    "numeric_dynamic": "FRESH_SOURCE_SCORED",
    "text_t1": "FRESH_SOURCE_SCORED",
    "text_dynamic": "FRESH_SOURCE_SCORED",
}
STALE_TEXT_FIELDS = (
    "channel_counts",
    "calibration",
    "reader_hash",
    "reader_bundle_sha256",
)
CONTRACT_FIELDS = (
    "bundle_id",
    "target_id",
    "reader_id",
    "unit",
    "target_time",
    "condition_plan",
    "observation_contract",
)
TOP_LEVEL = {
    "schema_version",
    "bundle_id",
    "route",
    "estimator",
    "seed",
    "target",
    "observation",
    "source",
    "calibration",
    "integration",
    "reference_policy",
    "known_validation",
    "legacy_reconciliation",
    "engine",
    "code_manifest",
}


class Refusal(ValueError):
    """A declared construction or bundle-level refusal with a stable code."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def json_bytes(value):
    """Canonical finite JSON; no pickle or executable payload."""
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def object_hash(value):
    """Hash the full canonical JSON object."""
    return hashlib.sha256(json_bytes(value)).hexdigest()


def code_manifest():
    """Hash the serializer and every engine module it executes."""
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in CODE_FILES
    }


def body_id(bundle):
    """Identity of every field except the identity itself."""
    return object_hash({k: v for k, v in bundle.items() if k != "bundle_id"})


def target_block(route, target, unit):
    """Declare the object, clock and condition plan before any scoring."""
    if target not in ROUTE_TARGETS[route]:
        raise Refusal("WRONG_TARGET", f"{target} is not produced by {route}")
    t1 = target == "T1"
    return {
        "target_id": target,
        "object_class": CLASSES[target],
        "estimand": ESTIMANDS[target],
        "unit": unit,
        "time_contract": {
            "unit": "simulator_step",
            "target_time": 31 if t1 else 47,
            "rule": "fixed-time source-frozen target;not calendar time;not an arbitrary-time norm",
        },
        "condition_plan": {
            "observed_support": [-1.0, -1 / 3, 1 / 3, 1.0] if t1 else [-1.0, 1.0],
            "future": [1, -1, 1, -1] if target == "T3" else [],
        },
    }


def reader_id(observation):
    """Recompute changed-reader identity from its definition and cache digest."""
    return object_hash(
        {
            "route": observation["route"],
            "definition": observation["definition"],
            "cache_sha256": observation.get("cache_sha256"),
        }
    )


def text_materials_digest(materials):
    """Digest the actual readout cache used by the engine."""
    return object_hash(materials)


def regenerate_channel(calibration):
    """Rebuild every text channel draw from declared counts and generator."""
    counts = np.asarray(calibration["counts"], float)
    generator = calibration["generator"]
    number = generator["draws"]
    if calibration["estimator"] == "mixture":
        return t1_text.channel_draws(
            counts, np.random.SeedSequence(generator["seed_sequence"]), number
        )
    law = (counts + 0.5) / (counts + 0.5).sum(-1, keepdims=True)
    return np.repeat(law[None], number, axis=0)


def build_bundle(
    *,
    route,
    estimator,
    seed,
    target,
    unit,
    observation,
    source,
    calibration,
    reference,
    engine,
    known_validation,
    legacy_reconciliation=None,
    integration_extra=None,
):
    """Assemble one public bundle, refusing non-operational constructions."""
    if route not in ROUTE_TARGETS:
        raise Refusal("UNKNOWN_ROUTE", route)
    if estimator not in ("mixture", "plugin"):
        raise Refusal("UNKNOWN_ESTIMATOR", estimator)
    provenance = calibration.get("data_provenance", {})
    if provenance.get("role") in ("application", "oracle"):
        raise Refusal(
            "APPLICATION_CALIBRATION_NOT_OPERATIONAL",
            "calibration data role is an information reference",
        )
    if calibration.get("status") not in ("READY", "SUPPORTED"):
        raise Refusal(
            calibration.get("status", "UNKNOWN_STATUS"), "calibration unusable"
        )
    if route.startswith("numeric"):
        plugin = calibration["plugin"]
        if plugin["status"] != "READY":
            raise Refusal(plugin["status"], "plugin calibrator refused")
        if estimator == "mixture":
            bad = [d["status"] for d in calibration["draws"] if d["status"] != "READY"]
            if bad or not calibration["draws"]:
                raise Refusal(
                    "INCOMPLETE_CALIBRATION_DRAWS",
                    f"{len(bad)} rejected bootstrap calibrators;no survivor renormalization",
                )
        else:
            calibration = dict(calibration, draws=[], bootstrap_indices=[])
    if reference.get("application_derived") is not False:
        raise Refusal(
            "APPLICATION_REFERENCE_FORBIDDEN", "reference must be source-only"
        )
    if reference["kind"] != REFERENCE_KINDS[route]:
        raise Refusal("WRONG_REFERENCE_POLICY", reference["kind"])
    observation = dict(observation, reader_id=reader_id(observation))
    integration = {"rule": INTEGRATION[route], "estimator": estimator}
    integration.update(integration_extra or {})
    bundle = {
        "schema_version": VERSION,
        "route": route,
        "estimator": estimator,
        "seed": seed,
        "target": target_block(route, target, unit),
        "observation": observation,
        "source": source,
        "calibration": calibration,
        "integration": integration,
        "reference_policy": reference,
        "known_validation": known_validation,
        "legacy_reconciliation": legacy_reconciliation,
        "engine": engine,
        "code_manifest": code_manifest(),
    }
    bundle["bundle_id"] = body_id(bundle)
    verify(bundle)
    return bundle


def verify(bundle):
    """Re-derive, rather than trust, every identity and calibration object."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != VERSION:
        version = bundle.get("schema_version") if isinstance(bundle, dict) else None
        raise Refusal("WRONG_BUNDLE_VERSION", str(version))
    if set(bundle) != TOP_LEVEL:
        raise Refusal("UNDECLARED_FIELDS", str(sorted(set(bundle) ^ TOP_LEVEL)))
    if bundle["bundle_id"] != body_id(bundle):
        raise Refusal("BUNDLE_HASH_MISMATCH", "body changed after identity")
    if bundle["code_manifest"] != code_manifest():
        raise Refusal("CODE_CHANGED", "engine or serializer code changed")
    route, estimator = bundle["route"], bundle["estimator"]
    if route not in ROUTE_TARGETS or estimator not in ("mixture", "plugin"):
        raise Refusal("UNKNOWN_ROUTE", route)
    target = bundle["target"]["target_id"]
    if bundle["target"] != target_block(route, target, bundle["target"]["unit"]):
        raise Refusal("TARGET_CONTRACT_CHANGED", target)
    integration = bundle["integration"]
    if (
        integration["rule"] != INTEGRATION[route]
        or integration["estimator"] != estimator
    ):
        raise Refusal("WRONG_INTEGRATION_RULE", integration["rule"])
    observation = bundle["observation"]
    if observation["reader_id"] != reader_id(observation):
        raise Refusal("WRONG_READER", "reader identity does not match definition")
    reference = bundle["reference_policy"]
    if reference["kind"] != REFERENCE_KINDS[route] or reference["application_derived"]:
        raise Refusal("WRONG_REFERENCE_POLICY", reference["kind"])
    values = [reference["mu"], reference["sigma"]]
    if not np.isfinite(values).all() or reference["sigma"] <= 0:
        raise Refusal("WRONG_REFERENCE_POLICY", "finite positive reference required")
    calibration, engine = bundle["calibration"], bundle["engine"]
    if calibration.get("data_provenance", {}).get("role") != "pilot":
        raise Refusal("APPLICATION_CALIBRATION_NOT_OPERATIONAL", "pilot role required")
    if route.startswith("numeric"):
        _verify_numeric(bundle, calibration, engine, reference)
    else:
        _verify_text(bundle, calibration, engine, reference, observation)


def _verify_numeric(bundle, calibration, engine, reference):
    """Validate calibrators, source ensemble and inherited/fresh reference."""
    route, estimator = bundle["route"], bundle["estimator"]
    if calibration["kind"] != "numeric_affine_pilot":
        raise Refusal("WRONG_CALIBRATION_KIND", calibration["kind"])
    if (
        estimator == "mixture"
        and len(calibration["draws"]) != calibration["draw_count"]
    ):
        raise Refusal("INCOMPLETE_CALIBRATION_DRAWS", "draw count mismatch")
    if estimator == "plugin" and calibration["draws"]:
        raise Refusal("UNDECLARED_FIELDS", "plugin bundle carries unused draws")
    for fit in [calibration["plugin"], *calibration["draws"]]:
        if fit["status"] != "READY" or fit["mode"] != calibration["mode"]:
            raise Refusal(fit["status"], "calibrator not ready or mode mismatch")
        width = 2 if fit["mode"] == "global" else 4
        coef = np.asarray(fit["coefficients"], float)
        if coef.shape != (width,) or not np.isfinite(coef).all():
            raise Refusal("INVALID_CALIBRATOR", "coefficient shape")
    _verify_refit(calibration, estimator)
    if route == "numeric_t1":
        try:
            t1_source.verify_bundle(engine)
        except (ValueError, KeyError, np.linalg.LinAlgError) as error:
            raise Refusal("SOURCE_ENGINE_INVALID", str(error)) from error
        if values_differ(
            [reference["mu"], reference["sigma"]],
            [engine["reference_mu"], engine["reference_sigma"]],
        ):
            raise Refusal("WRONG_REFERENCE_POLICY", "inherited reference altered")
        return
    models = engine["source_models"]
    if len(models) != engine["source_components"] or not models:
        raise Refusal("SOURCE_ENGINE_INVALID", "incomplete source ensemble")
    for model in models:
        mu = np.asarray(model["prior"]["mu"], float)
        cov = np.asarray(model["prior"]["Sigma"], float)
        logs = np.asarray(model["log_sigma"], float)
        if mu.shape != (3,) or cov.shape != (3, 3) or logs.shape != (2,):
            raise Refusal("SOURCE_ENGINE_INVALID", "dynamic parameter shape")
        if not np.isfinite(np.r_[mu, cov.ravel(), logs]).all():
            raise Refusal("SOURCE_ENGINE_INVALID", "nonfinite source model")
        if not np.allclose(cov, cov.T, atol=1e-12, rtol=0):
            raise Refusal("SOURCE_ENGINE_INVALID", "asymmetric prior")
        np.linalg.cholesky(cov)
    if engine["eta_nodes"] != 64:
        raise Refusal("WRONG_INTEGRATION_RULE", "declared 64-node eta quadrature")


def _same_fit(left, right, tolerance=1e-11):
    """Compare a refitted calibrator with the serialized one."""
    if left["status"] != right["status"] or left["mode"] != right["mode"]:
        return False
    if left["status"] != "READY":
        return True
    a = np.asarray(left["coefficients"], float)
    b = np.asarray(right["coefficients"], float)
    return (
        a.shape == b.shape
        and float(np.max(np.abs(a - b))) <= tolerance
        and list(np.asarray(left["conditions"], float))
        == list(np.asarray(right["conditions"], float))
        and np.allclose(left["new_range"], right["new_range"], rtol=0, atol=tolerance)
    )


def _verify_refit(calibration, estimator):
    """Re-derive every serialized calibrator from the embedded pilot fit pairs."""
    pairs = calibration["pilot_fit_pairs"]
    old = np.asarray(pairs["old"], float)
    new = np.asarray(pairs["new"], float)
    z = np.asarray(pairs["conditions"], float)
    if old.shape != new.shape or old.ndim != 2 or old.shape[1] != len(z):
        raise Refusal("CALIBRATION_DRAWS_NOT_REPRODUCIBLE", "pilot pair shape")
    tiled = np.tile(z, len(old))
    mode = calibration["mode"]
    if not _same_fit(
        fit_transport(old.ravel(), new.ravel(), tiled, mode), calibration["plugin"]
    ):
        raise Refusal("CALIBRATION_DRAWS_NOT_REPRODUCIBLE", "plugin refit differs")
    if estimator == "mixture":
        indices = np.asarray(calibration["bootstrap_indices"], int)
        if indices.shape != (len(calibration["draws"]), len(old)):
            raise Refusal("CALIBRATION_DRAWS_NOT_REPRODUCIBLE", "bootstrap shape")
        for ix, draw in zip(indices, calibration["draws"], strict=True):
            refit = fit_transport(old[ix].ravel(), new[ix].ravel(), tiled, mode)
            if not _same_fit(refit, draw):
                raise Refusal(
                    "CALIBRATION_DRAWS_NOT_REPRODUCIBLE", "bootstrap refit differs"
                )


def values_differ(left, right):
    """Exact comparison of frozen JSON numbers."""
    return [float(x) for x in left] != [float(x) for x in right]


def _verify_text(bundle, calibration, engine, reference, observation):
    """Authenticate changed reader cache, regenerate channel draws, reject stale fields."""
    route, target = bundle["route"], bundle["target"]["target_id"]
    stale = [k for k in STALE_TEXT_FIELDS if k in engine]
    if stale:
        raise Refusal("STALE_LEGACY_FIELD_PRESENT", ",".join(stale))
    try:
        if route == "text_t1":
            t1_text.verify_bundle(engine)
        else:
            dyn_text.validate_bundle(engine)
    except (ValueError, KeyError, np.linalg.LinAlgError) as error:
        raise Refusal("SOURCE_ENGINE_INVALID", str(error)) from error
    if observation["cache_sha256"] != text_materials_digest(engine["materials"]):
        raise Refusal("WRONG_READER", "readout cache differs from declared reader")
    for item in engine["materials"].values():
        if item["grain_qualification"] != item["nli_qualification"]:
            raise Refusal(
                "WRONG_READER", "legacy slot alias does not carry the changed reader"
            )
    if calibration["kind"] != "categorical_channel_pilot":
        raise Refusal("WRONG_CALIBRATION_KIND", calibration["kind"])
    if calibration["estimator"] != bundle["estimator"]:
        raise Refusal("WRONG_CALIBRATION_KIND", "estimator mismatch")
    counts = np.asarray(calibration["counts"], float)
    if counts.shape != (2, 2, 4) or (counts.sum(-1) == 0).any():
        raise Refusal("MISSING_ANCHOR", "unsupported pole/qualification strata")
    regenerated = regenerate_channel(calibration)
    for key in ("hybrid_joint", "nli_joint"):
        if not np.array_equal(regenerated, np.asarray(engine["channel_draws"][key])):
            raise Refusal(
                "CALIBRATION_DRAWS_NOT_REPRODUCIBLE",
                "engine channel draws differ from declared counts/generator",
            )
    if route == "text_t1":
        frozen = [engine["reference_mu"], engine["reference_sigma"]]
    else:
        frozen = [
            engine["reference"][target]["mu"],
            engine["reference"][target]["sigma"],
        ]
        other = engine["reference"]["T3" if target == "T2" else "T2"]
        if reference.get("companion") != other:
            raise Refusal(
                "WRONG_REFERENCE_POLICY", "companion target reference mismatch"
            )
    if values_differ([reference["mu"], reference["sigma"]], frozen):
        raise Refusal("WRONG_REFERENCE_POLICY", "engine/public reference mismatch")


def request_contract(bundle):
    """Exactly what an observation envelope must match."""
    target = bundle["target"]
    return {
        "bundle_id": bundle["bundle_id"],
        "target_id": target["target_id"],
        "reader_id": bundle["observation"]["reader_id"],
        "unit": target["unit"],
        "target_time": target["time_contract"]["target_time"],
        "condition_plan": target["condition_plan"],
        "observation_contract": OBSERVATION_CONTRACTS[bundle["route"]],
    }


def make_request(bundle, data, explain="summary"):
    """Wrap observed data only; latent truth is never part of the envelope."""
    return {"contract": request_contract(bundle), "data": data, "explain": explain}


def _reject(code, detail, bundle, data=None):
    """Per-request refusal row; the batch continues."""
    author = data.get("author_id") if isinstance(data, dict) else None
    return {
        "status": "REJECTED",
        "code": code,
        "detail": detail,
        "author_id": author,
        "bundle_id": bundle["bundle_id"],
    }


def _check_request(request, bundle):
    """Return a rejection (code, detail) or None, before any inference."""
    if not isinstance(request, dict) or set(request) != {"contract", "data", "explain"}:
        return "MALFORMED_REQUEST", "fields must be contract,data,explain"
    if request["explain"] not in ("summary", "deletion"):
        return "MALFORMED_REQUEST", "explain must be summary or deletion"
    contract, expected = request["contract"], request_contract(bundle)
    if not isinstance(contract, dict) or set(contract) != set(CONTRACT_FIELDS):
        return "MALFORMED_REQUEST", "contract fields"
    codes = {
        "bundle_id": "WRONG_VERSION",
        "target_id": "WRONG_TARGET",
        "reader_id": "WRONG_READER",
        "unit": "WRONG_UNIT",
        "target_time": "WRONG_CLOCK",
        "condition_plan": "WRONG_CONDITION_PLAN",
        "observation_contract": "WRONG_OBSERVATION_CONTRACT",
    }
    for key in CONTRACT_FIELDS:
        if contract[key] != expected[key]:
            return codes[key], f"{key} differs from bundle"
    if not isinstance(request["data"], dict):
        return "MALFORMED_REQUEST", "data envelope"
    return None


def _reference_block(bundle, score, variance):
    """Position against the declared source reference only."""
    ref = bundle["reference_policy"]
    return {
        "policy": ref["kind"],
        "mu": ref["mu"],
        "sigma": ref["sigma"],
        "position": (score - ref["mu"]) / ref["sigma"],
        "position_variance": variance / ref["sigma"] ** 2,
        "scope": ref["scope"],
    }


def _numeric_records(data, horizon, conditions, complete):
    """Validate numeric changed-reader observations; truth fields rejected."""
    if set(data) != {"author_id", "records"}:
        return None, ("HIDDEN_OR_UNDECLARED_FIELD", "data must be author_id,records")
    if not isinstance(data["author_id"], str) or not data["author_id"]:
        return None, ("MALFORMED_REQUEST", "author identity required")
    rows = data["records"]
    if not isinstance(rows, list) or not rows:
        return None, ("PRIOR_ONLY_NOT_TRANSPORTED", "no observations to transport")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"t", "z", "y"}:
            return None, ("HIDDEN_OR_UNDECLARED_FIELD", "record fields must be t,z,y")
        t = row["t"]
        if type(t) is not int or not 0 <= t <= horizon or t in seen:
            return None, (
                "WRONG_CLOCK",
                "event time outside target clock or duplicated",
            )
        seen.add(t)
        if isinstance(row["z"], bool) or row["z"] not in conditions:
            return None, ("UNSUPPORTED_CONDITION", f"condition {row['z']}")
        if isinstance(row["y"], bool) or not isinstance(row["y"], (int, float)):
            return None, ("MALFORMED_REQUEST", "numeric observation")
        if not np.isfinite(row["y"]):
            return None, ("MALFORMED_REQUEST", "finite observation required")
    rows = sorted(rows, key=lambda r: r["t"])
    if complete and [r["t"] for r in rows] != list(range(horizon + 1)):
        return None, (
            "INCOMPLETE_CLOCK_SUPPORT",
            "this compiled estimator requires one event per step 0..47",
        )
    return rows, None


def _score_numeric_t1(bundle, items):
    """Exact Cartesian source x pilot integration with additive explanations."""
    engine, cal = bundle["engine"], bundle["calibration"]
    fits = [cal["plugin"]] if bundle["estimator"] == "plugin" else cal["draws"]
    out, groups = {}, {}
    for index, data in items:
        rows, error = _numeric_records(data, 31, t1_source.CONDITIONS, False)
        if error:
            out[index] = _reject(*error, bundle, data)
            continue
        z = np.array([r["z"] for r in rows], float)
        if any(not np.any(z == c) for c in t1_source.CONDITIONS):
            out[index] = _reject(
                "INCOMPLETE_CONDITION_SUPPORT",
                "transport kernel requires all four conditions",
                bundle,
                data,
            )
            continue
        new = np.array([r["y"] for r in rows], float)
        ok = transport_score(cal["plugin"], new, z)[1].all()
        for fit in fits:
            ok &= transport_score(fit, new, z)[1].all()
        if not ok:
            out[index] = _reject(
                "OUTSIDE_CALIBRATED_SUPPORT",
                "a calibrator does not support every observation",
                bundle,
                data,
            )
            continue
        groups.setdefault(tuple(z), []).append((index, data, rows, new))
    for key, members in groups.items():
        z = np.array(key)
        kernels = t1_law.source_kernels(engine, z)
        corrected = np.stack(
            [np.stack([transport_score(f, m[3], z)[0] for m in members]) for f in fits]
        )
        result = t1_law.integrate(kernels, t1_law.cell_means(corrected, z))
        counts = np.array([(z == c).sum() for c in t1_source.CONDITIONS], float)
        omega = kernels["weights"].mean(0) / counts
        cell = np.array([t1_source.CONDITIONS.index(v) for v in z])
        intercept = float(kernels["offsets"].mean())
        for a, (index, data, rows, _) in enumerate(members):
            calibrated = corrected[:, a].mean(0)
            contributions = omega[cell] * calibrated
            value = float(result["score"][a])
            variance = float(result["variance"][a])
            out[index] = {
                "status": "SCORED",
                "bundle_id": bundle["bundle_id"],
                "author_id": data["author_id"],
                "target_id": "T1",
                "estimator": bundle["estimator"],
                "score": value,
                "variance": variance,
                "lower": float(result["lower"][a]),
                "upper": float(result["upper"][a]),
                "components": {
                    k: float(result[k][a]) for k in ("conditional", "source", "pilot")
                },
                "unit": bundle["target"]["unit"],
                "target_time": 31,
                "support": {
                    "status": "SUPPORTED",
                    "observed_events": len(rows),
                    "condition_counts": counts.tolist(),
                    "calibrators": len(fits),
                    "integration_components": len(fits) * len(engine["source_models"]),
                },
                "reference": _reference_block(bundle, value, variance),
                "explanation": {
                    "kind": "exact_additive_linear_decomposition",
                    "intercept": intercept,
                    "events": [
                        {
                            "t": r["t"],
                            "z": r["z"],
                            "observed": r["y"],
                            "calibrated_mean": float(calibrated[i]),
                            "weight": float(omega[cell[i]]),
                            "contribution": float(contributions[i]),
                        }
                        for i, r in enumerate(rows)
                    ],
                    "identity_error": abs(
                        intercept + float(contributions.sum()) - value
                    ),
                    "note": "score=intercept+sum(weight*calibrated mean);fixed support;not a psychological attribution",
                },
            }
    return out


def _dynamic_attribution(kernels, basis, coefficients):
    """Posterior-weight-conditional linear attribution for T2/T3 means."""
    x = np.einsum("atd,jd->jat", basis, coefficients)
    scale = len(coefficients) * len(kernels)
    intercept = np.zeros((len(basis), 2))
    contributions = np.zeros((len(basis), 2, basis.shape[1]))
    for kernel in kernels:
        ig = np.einsum("ktv,avd->katd", kernel["inverse"], basis, optimize=True)
        quad = np.einsum("atd,kath->kadh", basis, ig, optimize=True)
        linear = np.einsum("atd,kt->kad", basis, kernel["inv_mu"], optimize=True)
        ll = -0.5 * np.einsum(
            "jd,kadh,jh->kja", coefficients, quad, coefficients, optimize=True
        )
        ll += (
            np.einsum("jd,kad->kja", coefficients, linear)
            + kernel["constant"][:, None, None]
        )
        logw = ll + kernel["log_weights"][:, None, None]
        weights = np.exp(logw - logsumexp(logw, axis=0, keepdims=True))
        loading = np.einsum(
            "kja,klt->jalt", weights, kernel["target_weight"], optimize=True
        )
        contributions += np.einsum("jalt,jat->alt", loading, x) / scale
        intercept += np.einsum("kja,kl->al", weights, kernel["target_offset"]) / scale
    return intercept, contributions, x.mean(0)


def _score_numeric_dynamic(bundle, items, chunk=25):
    """Compile the declared eta quadrature per schedule; Cartesian source x pilot."""
    engine, cal = bundle["engine"], bundle["calibration"]
    target = bundle["target"]["target_id"]
    t = 0 if target == "T2" else 1
    support_fits = [cal["plugin"], *cal["draws"]]
    select = slice(0, 1) if bundle["estimator"] == "plugin" else slice(1, None)
    out, groups = {}, {}
    for index, data in items:
        rows, error = _numeric_records(data, 47, (-1.0, 1.0), True)
        if error:
            out[index] = _reject(*error, bundle, data)
            continue
        groups.setdefault(tuple(r["z"] for r in rows), []).append((index, data, rows))
    for key, members in groups.items():
        z = np.array(key, float)
        new = np.array([[r["y"] for r in m[2]] for m in members], float)
        basis, coef, support = dyn_law.inverse_basis(new, z, support_fits)
        for a in np.flatnonzero(~support):
            index, data, _ = members[a]
            out[index] = _reject(
                "OUTSIDE_CALIBRATED_SUPPORT",
                "a calibrator does not support this trajectory",
                bundle,
                data,
            )
        if not support.any():
            continue
        kept = [m for m, ok in zip(members, support, strict=True) if ok]
        kernels = [
            dyn_law.compile_source(m, z, engine["eta_nodes"])
            for m in engine["source_models"]
        ]
        chosen = coef[select]
        result = dyn_law.collapse(
            dyn_law.integrate(kernels, basis, chosen), slice(None)
        )
        pieces = [
            _dynamic_attribution(kernels, basis[s : s + chunk], chosen)
            for s in range(0, len(basis), chunk)
        ]
        intercept = np.concatenate([p[0] for p in pieces])
        contributions = np.concatenate([p[1] for p in pieces])
        calibrated = np.concatenate([p[2] for p in pieces])
        for a, (index, data, rows) in enumerate(kept):
            value = float(result["mean"][a, t])
            variance = float(result["covariance"][a, t, t])
            out[index] = {
                "status": "SCORED",
                "bundle_id": bundle["bundle_id"],
                "author_id": data["author_id"],
                "target_id": target,
                "estimator": bundle["estimator"],
                "score": value,
                "variance": variance,
                "lower": value - 1.96 * np.sqrt(variance),
                "upper": value + 1.96 * np.sqrt(variance),
                "components": {
                    k: float(result[k][a, t, t])
                    for k in ("state_future", "personal", "source", "pilot")
                },
                "joint": {
                    "mean_T2_T3": result["mean"][a].tolist(),
                    "covariance_T2_T3": result["covariance"][a].tolist(),
                },
                "unit": bundle["target"]["unit"],
                "target_time": 47,
                "support": {
                    "status": "SUPPORTED",
                    "observed_events": len(rows),
                    "clock": "complete 0..47",
                    "calibrators": len(chosen),
                    "integration_components": int(len(chosen) * len(kernels)),
                    "min_effective_nodes": float(result["min_effective_nodes"][a]),
                },
                "reference": _reference_block(bundle, value, variance),
                "explanation": {
                    "kind": "posterior_weight_conditional_linear_attribution",
                    "intercept": float(intercept[a, t]),
                    "events": [
                        {
                            "t": r["t"],
                            "z": r["z"],
                            "observed": r["y"],
                            "calibrated_mean": float(calibrated[a, i]),
                            "contribution": float(contributions[a, t, i]),
                        }
                        for i, r in enumerate(rows)
                    ],
                    "identity_error": abs(
                        float(intercept[a, t] + contributions[a, t].sum()) - value
                    ),
                    "note": "exact at the observed eta weights;not a deletion or counterfactual effect",
                },
            }
    return out


def _text_events(bundle, raw):
    """Readable evidence trace: materials, categories and conditions."""
    engine = bundle["engine"]
    rows, stale = t1_text.replay_history(raw, engine)
    events = []
    for row in rows:
        material = engine["materials"][row["material_id"]]
        events.append(
            {
                "t": row["t"],
                "z": row["z"],
                "material_id": row["material_id"],
                "category": t1_text.category(
                    material["pole_readout"], material["grain_qualification"]
                ),
            }
        )
    return events, stale


def _text_error_code(text):
    """Stable codes for engine-side input refusals."""
    if "unregistered" in text:
        return "UNREGISTERED_TEXT"
    if "unexpected input" in text or "truth" in text:
        return "HIDDEN_OR_UNDECLARED_FIELD"
    if "time" in text or "future" in text or "clock" in text:
        return "WRONG_CLOCK"
    return "INVALID_HISTORY"


def _score_text(bundle, items):
    """Paired source/channel Monte Carlo engines, unchanged; explanation added."""
    engine, target = bundle["engine"], bundle["target"]["target_id"]
    out, valid = {}, []
    for index, data in items:
        try:
            if set(data) != {"author_id", "target_time", "history", "world_contract"}:
                raise ValueError("unexpected input fields, including latent truth")
            expected = bundle["target"]["time_contract"]["target_time"]
            if data["target_time"] != expected:
                raise ValueError("requested time outside source-frozen target")
            t1_text.replay_history(data, engine)
            if bundle["route"] == "text_dynamic":
                dyn_text.prepare([data], engine)
        except (ValueError, KeyError, TypeError) as error:
            out[index] = _reject(_text_error_code(str(error)), str(error), bundle, data)
            continue
        valid.append((index, data))
    if not valid:
        return out
    inputs = [d for _, d in valid]
    if bundle["route"] == "text_t1":
        rows = t1_text.cold_scores(inputs, engine)
        projected = rows
    else:
        rows = dyn_text.cold_scores(inputs, engine)
        projected = [r["targets"][target] for r in rows]
    for (index, data), row, proj in zip(valid, rows, projected, strict=True):
        events, stale = _text_events(bundle, data)
        if bundle["route"] == "text_t1":
            components = {
                "personal_conditional": row["personal_conditional_variance"],
                "source_and_reader": row["source_and_reader_variance"],
            }
        else:
            components = dict(proj["components"])
        out[index] = {
            "status": "SCORED",
            "bundle_id": bundle["bundle_id"],
            "author_id": data["author_id"],
            "target_id": target,
            "estimator": bundle["estimator"],
            "score": proj["score"],
            "variance": proj["variance"],
            "lower": proj["lower"],
            "upper": proj["upper"],
            "components": components,
            "unit": bundle["target"]["unit"],
            "target_time": data["target_time"],
            "support": {
                "status": "SUPPORTED",
                "observed_events": len(events),
                "prior_only": not events,
                "stale_annotations": stale,
                "engine_support": row.get("support", "state_filter"),
                "integration_components": engine["source_components"],
                "pairing": "source b with channel draw b;not Cartesian",
            },
            "reference": _reference_block(bundle, proj["score"], proj["variance"]),
            "explanation": {"kind": "evidence_trace", "events": events},
            "engine_output": row,
        }
    return out


def text_deletion_effects(bundle, raw):
    """Exact score change from withdrawing each active report (or T1 cell count)."""
    engine, target = bundle["engine"], bundle["target"]["target_id"]
    if bundle["route"] == "text_t1":
        counts, _ = t1_text.counts_from_input(raw, engine)
        cells = [tuple(int(v) for v in ix) for ix in np.argwhere(counts > 0)]
        variants = [counts]
        for k, c in cells:
            reduced = counts.copy()
            reduced[k, c] -= 1
            variants.append(reduced)
        scores = t1_text.score_counts(np.stack(variants), engine)
        base = scores[0]["score"]
        return [
            {
                "condition": t1_source.CONDITIONS[k],
                "category": c,
                "count": int(counts[k, c]),
                "delete_one_effect": base - s["score"],
            }
            for (k, c), s in zip(cells, scores[1:], strict=True)
        ]
    last = max(op["recorded_at"] for op in raw["history"])
    rows, _ = t1_text.replay_history(raw, engine)
    active = [
        op["record_id"]
        for op in raw["history"]
        if op["op"] in ("new", "revise") and any(op["observation"] == r for r in rows)
    ]
    variants = [raw]
    for rid in active:
        variant = copy.deepcopy(raw)
        variant["history"].append(
            {
                "record_id": f"withdraw::{rid}",
                "op": "retract",
                "recorded_at": last,
                "parent_id": rid,
            }
        )
        variants.append(variant)
    scores = dyn_text.score_panel(variants, engine)
    base = scores[0]["targets"][target]["score"]
    return [
        {"record_id": rid, "delete_one_effect": base - s["targets"][target]["score"]}
        for rid, s in zip(active, scores[1:], strict=True)
    ]


def score(bundle, requests):
    """Cold scoring from the bundle and observation envelopes only."""
    verify(bundle)
    out, accepted = {}, []
    for index, request in enumerate(requests):
        problem = _check_request(request, bundle)
        if problem:
            data = request.get("data") if isinstance(request, dict) else None
            out[index] = _reject(*problem, bundle, data)
        else:
            accepted.append((index, request["data"]))
    route = bundle["route"]
    if route == "numeric_t1":
        out.update(_score_numeric_t1(bundle, accepted))
    elif route == "numeric_dynamic":
        out.update(_score_numeric_dynamic(bundle, accepted))
    else:
        out.update(_score_text(bundle, accepted))
        for index, data in accepted:
            if (
                out[index]["status"] == "SCORED"
                and requests[index]["explain"] == "deletion"
            ):
                out[index]["explanation"]["deletion_effects"] = text_deletion_effects(
                    bundle, data
                )
    return [out[i] for i in range(len(requests))]


def seal(allowed_files, log):
    """Forbid undeclared file reads, sockets and subprocesses in this process."""
    roots = {
        os.path.realpath(p)
        for p in (
            sys.prefix,
            sys.base_prefix,
            sys.exec_prefix,
            sys.base_exec_prefix,
            str(Path(__file__).parent),
        )
    }
    files = {os.path.realpath(p) for p in allowed_files}
    blocked = (
        "socket.connect",
        "socket.bind",
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
    )

    def hook(event, args):
        if event == "open":
            path = args[0]
            if isinstance(path, int) or path is None:
                return
            real = os.path.realpath(os.fsdecode(path))
            ok = real in files or any(
                real == r or real.startswith(r + os.sep) for r in roots
            )
            log.append([real, ok])
            if not ok:
                raise PermissionError(
                    f"sealed cold scorer refused undeclared file {real}"
                )
        elif event in blocked:
            raise PermissionError(f"sealed cold scorer refused {event}")

    sys.addaudithook(hook)


def main():
    """Sealed CLI: trusted bundle hash, observation envelopes in, scores out."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("score")
    for name in ("bundle", "input", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--sha256", required=True)
    run.add_argument("--probe", type=Path, help="test only: attempt an undeclared read")
    args = parser.parse_args()
    manifest = Path(str(args.output) + ".manifest.json")
    log = []
    seal([args.bundle, args.input, args.output, manifest], log)
    probe = None
    if args.probe is not None:
        try:
            args.probe.read_bytes()
            probe = {"path": str(args.probe), "refused": False}
        except PermissionError as error:
            probe = {"path": str(args.probe), "refused": True, "error": str(error)}
    payload = args.bundle.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.sha256:
        raise Refusal("BUNDLE_HASH_MISMATCH", "file bytes differ from trusted hash")
    requests = [json.loads(line) for line in args.input.read_text().splitlines()]
    rows = score(json.loads(payload), requests)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json_bytes(row).decode() + "\n")
    with manifest.open("x") as stream:
        stream.write(
            json_bytes(
                {
                    "bundle_sha256": args.sha256,
                    "requests": len(requests),
                    "scored": sum(r["status"] == "SCORED" for r in rows),
                    "rejected": sum(r["status"] == "REJECTED" for r in rows),
                    "files_opened": sorted({p for p, _ in log}),
                    "refused_opens": sorted({p for p, ok in log if not ok}),
                    "probe": probe,
                    "python": sys.version,
                    "numpy": np.__version__,
                }
            ).decode()
        )


if __name__ == "__main__":
    main()
