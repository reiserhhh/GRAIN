#!/usr/bin/env python3
"""Target-compatible dynamic inference from noisy categorical text observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import qmc

from .f16_t1_text import category, factorize_channel, oracle_channel, replay_history

VERSION = "F16_dynamic_text_ADF_prior_QMC_v1"
CONTRACT = "F05_core_thresholded_text_eligibility_v1"
PLAN = [1, -1, 1, -1]
ARMS = (
    "hybrid_joint",
    "nli_joint",
    "hybrid_factorized",
    "oracle",
    "naive_gaussian",
    "eligible_continuous",
)
TARGETS = np.array([[1.0, 0, 0, 0, 0], [0, 0.25, 0.25, 0.25, 0.25]])


def code_hash():
    """Bind the scorer and its imported text/history dependencies."""
    root = Path(__file__).parent
    names = ["f16_dynamic_text.py", "f16_t1_text.py", "f16_t1_source.py"]
    return hashlib.sha256(b"".join((root / x).read_bytes() for x in names)).hexdigest()


def probit_update(mean, variance, offset, noise, channel, categories):
    """Exact first two moments for one affine-probit update of Gaussian states."""
    c = np.asarray(categories, int)
    if np.any((c < 0) | (c > 3)) or noise <= 0 or np.any(variance < 0):
        raise ValueError(
            "valid categories, nonnegative state and positive noise required"
        )
    e = np.asarray(channel)
    c0, c1 = 0.5 * e[0, 1, c], 0.5 * e[1, 1, c]
    unrelated = 0.25 * (e[0, 0, c] + e[1, 0, c])
    scale2 = noise + variance
    k = (offset + mean) / np.sqrt(scale2)
    normalizer = c0 * ndtr(-k) + c1 * ndtr(k) + unrelated
    if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
        raise ValueError("impossible/underflowed category likelihood")
    slope = (c1 - c0) * np.exp(-0.5 * k * k) / np.sqrt(2 * np.pi * scale2)
    derivative = slope / normalizer
    curvature = -k * slope / np.sqrt(scale2) / normalizer
    updated_mean = mean + variance * derivative
    updated_var = variance + variance**2 * (curvature - derivative**2)
    if np.any(updated_var < 0) or not np.isfinite(updated_var).all():
        raise ValueError("invalid ADF variance; no flooring fallback")
    return updated_mean, updated_var, np.log(normalizer)


def parameter_nodes(model, seed, index, power, scramble=0):
    """Integrate the frozen source parameter prior, not application truth."""
    s = int(np.random.SeedSequence([seed, 164, index, scramble]).generate_state(1)[0])
    u = qmc.Sobol(3, scramble=True, seed=s).random_base2(power)
    n = ndtri(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
    return (
        np.asarray(model["prior"]["mu"])
        + n @ np.linalg.cholesky(model["prior"]["Sigma"]).T
    )


def filter_nodes(
    parameters, conditions, observations, weights, mask, logs, channel, arm, target_time
):
    """Filter regular-clock arrays, retaining empty steps and parameter likelihoods."""
    p = np.asarray(parameters)
    z, obs, precision, mask = map(np.asarray, (conditions, observations, weights, mask))
    if not (z.shape == obs.shape == precision.shape == mask.shape):
        raise ValueError("aligned author/time arrays required")
    if z.ndim != 2 or z.shape[1] != target_time + 1 or target_time < 0:
        raise ValueError("requested clock must match array support")
    rho = 0.99 * np.tanh(p[:, 2:3])
    q, r = np.exp(2 * np.asarray(logs))
    stationary = q / (1 - rho**2)
    mean = np.zeros((len(p), len(z)))
    variance = np.broadcast_to(stationary, mean.shape).copy()
    loglik = np.zeros_like(mean)
    for t in range(target_time + 1):
        if t:
            mean *= rho
            variance = rho**2 * variance + q
        visible = mask[:, t]
        if not visible.any():
            continue
        m, v = mean[:, visible], variance[:, visible]
        offset = p[:, 0:1] + p[:, 1:2] * z[visible, t]
        if arm in ("naive_gaussian", "eligible_continuous"):
            w = precision[visible, t]
            # A zero precision supplies no evidence and must not enter likelihood.
            active = w > 0
            rv = r / np.where(active, w, 1.0)
            total = v + rv
            innovation = obs[visible, t] - offset - m
            m1 = m + v / total * innovation
            v1 = v * rv / total
            ll = -0.5 * (np.log(2 * np.pi * total) + innovation**2 / total)
            m1, v1, ll = (
                np.where(active, m1, m),
                np.where(active, v1, v),
                np.where(active, ll, 0),
            )
        else:
            m1, v1, ll = probit_update(
                m, v, offset, r, channel, obs[visible, t].astype(int)
            )
        mean[:, visible], variance[:, visible] = m1, v1
        loglik[:, visible] += ll
    return mean, variance, loglik


def future_moments(parameters, state_mean, state_var, logs, plan):
    """Full state/future joint moments, including each shared innovation loading."""
    p = np.asarray(parameters)
    rho = 0.99 * np.tanh(p[:, 2])
    q, r = np.exp(2 * np.asarray(logs))
    loading = rho[:, None] ** np.arange(5)
    offsets = np.column_stack(
        [np.zeros(len(p)), p[:, 0, None] + p[:, 1, None] * np.asarray(plan)]
    )
    means = state_mean[:, :, None] * loading[:, None, :] + offsets[:, None, :]
    covariance = (
        state_var[:, :, None, None]
        * (loading[:, :, None] * loading[:, None, :])[:, None]
    )
    innovations = np.zeros((len(p), 5, 5))
    for j in range(1, 5):
        vector = np.zeros((len(p), 5))
        vector[:, j:] = rho[:, None] ** np.arange(5 - j)
        innovations += q * vector[:, :, None] * vector[:, None, :]
        innovations[:, j, j] += r
    return means, covariance + innovations[:, None]


def prepare(inputs, bundle, arm="hybrid_joint", evaluator=None):
    """Resolve observed text history into clock-indexed reader values only."""
    if not inputs:
        raise ValueError("nonempty author panel required")
    time = inputs[0]["target_time"]
    if type(time) is not int or not 0 <= time <= 63:
        raise ValueError("supported integer target clock required")
    shapes = (len(inputs), time + 1)
    z, values, weight, mask = (
        np.zeros(shapes),
        np.zeros(shapes),
        np.zeros(shapes),
        np.zeros(shapes, bool),
    )
    supports = []
    for i, raw in enumerate(inputs):
        if raw["target_time"] != time:
            raise ValueError("batch requires common requested clock")
        rows, stale = replay_history(raw, bundle)
        for row in rows:
            t = row["t"]
            if row["z"] not in (-1, 1) or mask[i, t]:
                raise ValueError(
                    "one event per clock and binary design condition required"
                )
            mat = bundle["materials"][row["material_id"]]
            z[i, t], mask[i, t], weight[i, t] = row["z"], True, 1
            if arm in ("oracle", "eligible_continuous"):
                if evaluator is None:
                    raise ValueError("oracle references unavailable in cold scorer")
                ev = evaluator[(raw["author_id"], row["event_id"])]
                if ev["material_id"] != row["material_id"]:
                    raise ValueError("reference rendering mismatch")
                if arm == "oracle":
                    values[i, t] = 2 * ev["rendered_pole"] + ev["qualification"]
                else:
                    values[i, t], weight[i, t] = ev["y"], ev["qualification"]
            elif arm == "naive_gaussian":
                values[i, t], weight[i, t] = (
                    2 * mat["pole_readout"] - 1,
                    mat["grain_qualification"],
                )
            else:
                qkey = (
                    "nli_qualification" if arm == "nli_joint" else "grain_qualification"
                )
                values[i, t] = category(mat["pole_readout"], mat[qkey])
        supports.append(
            {
                "active_materials": [r["material_id"] for r in rows],
                "stale_annotations": stale,
                "observed_events": len(rows),
                "last_observed_time": rows[-1]["t"] if rows else None,
                "prior_only": not rows,
            }
        )
    return (z, values, weight, mask), supports


def validate_bundle(bundle):
    """Verify frozen source/calibration shapes, readable materials and units."""
    if (
        bundle["version"] != VERSION
        or bundle["scorer_code_hash"] != code_hash()
        or bundle["world_contract"] != CONTRACT
        or bundle["future_conditions"] != PLAN
    ):
        raise ValueError("incompatible code, observation law or target plan")
    n = len(bundle["source_models"])
    if n != bundle["source_components"] or n < 1 or not 1 <= bundle["qmc_power"] <= 14:
        raise ValueError("incomplete source/integration ensemble")
    for model in bundle["source_models"]:
        mu, cov, logs = (
            np.asarray(model["prior"]["mu"]),
            np.asarray(model["prior"]["Sigma"]),
            np.asarray(model["log_sigma"]),
        )
        if (
            mu.shape != (3,)
            or cov.shape != (3, 3)
            or logs.shape != (2,)
            or not np.isfinite(np.r_[mu, cov.ravel(), logs]).all()
        ):
            raise ValueError("invalid dynamic source parameters")
        if not np.allclose(cov, cov.T, atol=1e-12, rtol=0):
            raise ValueError("asymmetric prior")
        np.linalg.cholesky(cov)
    for key in ("hybrid_joint", "nli_joint"):
        e = np.asarray(bundle["channel_draws"][key])
        if (
            e.shape != (n, 2, 2, 4)
            or not np.isfinite(e).all()
            or np.any(e < 0)
            or not np.allclose(e.sum(-1), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError("invalid categorical channel")
    allowed = {
        "text",
        "text_sha256",
        "pole_readout",
        "nli_qualification",
        "grain_qualification",
    }
    for item in bundle["materials"].values():
        if (
            set(item) != allowed
            or hashlib.sha256(item["text"].encode()).hexdigest() != item["text_sha256"]
        ):
            raise ValueError("cache has latent labels or wrong text hash")
        category(item["pole_readout"], item["grain_qualification"])
        category(item["pole_readout"], item["nli_qualification"])
    for target in ("T2", "T3"):
        ref = bundle["reference"][target]
        if not np.isfinite([ref["mu"], ref["sigma"]]).all() or ref["sigma"] <= 0:
            raise ValueError("invalid frozen reference")


def score_panel(inputs, bundle, arm="hybrid_joint", evaluator=None, scramble=0):
    """Integrate conditional states and parameters without mixing source feedback."""
    validate_bundle(bundle)
    if arm not in ARMS:
        raise ValueError("unknown arm")
    arrays, support = prepare(inputs, bundle, arm, evaluator)
    centers, within_parts, personal_parts, ess_parts = [], [], [], []
    for b, model in enumerate(bundle["source_models"]):
        nodes = parameter_nodes(model, bundle["seed"], b, bundle["qmc_power"], scramble)
        if arm == "oracle":
            channel = oracle_channel()
        else:
            key = "nli_joint" if arm == "nli_joint" else "hybrid_joint"
            channel = np.asarray(bundle["channel_draws"][key][b])
            if arm == "hybrid_factorized":
                channel = factorize_channel(channel)
        mean, var, loglik = filter_nodes(
            nodes, *arrays, model["log_sigma"], channel, arm, inputs[0]["target_time"]
        )
        weights = np.exp(loglik - logsumexp(loglik, axis=0))
        means, cov = future_moments(nodes, mean, var, model["log_sigma"], PLAN)
        center = np.einsum("na,nak->ak", weights, means)
        within = np.einsum("na,nakl->akl", weights, cov)
        diff = means - center[None]
        personal = np.einsum("na,nak,nal->akl", weights, diff, diff)
        centers.append(center)
        within_parts.append(within)
        personal_parts.append(personal)
        ess_parts.append(1 / np.sum(weights**2, axis=0))
    centers = np.asarray(centers)
    mean = centers.mean(0)
    diff = centers - mean[None]
    source = np.einsum("bak,bal->akl", diff, diff) / len(centers)
    components = {
        "conditional_state_future": np.mean(within_parts, axis=0),
        "personal_parameter": np.mean(personal_parts, axis=0),
        "source_reader": source,
    }
    total = sum(components.values())
    projected_mean = mean @ TARGETS.T
    projected_var = np.einsum("tk,akl,tl->at", TARGETS, total, TARGETS)
    results = []
    for i, raw in enumerate(inputs):
        targets = {}
        for t, name in enumerate(("T2", "T3")):
            m, v = float(projected_mean[i, t]), float(projected_var[i, t])
            if v < 0:
                raise ValueError("negative target variance")
            ref = bundle["reference"][name]
            targets[name] = {
                "score": m,
                "variance": v,
                "lower": m - 1.96 * np.sqrt(v),
                "upper": m + 1.96 * np.sqrt(v),
                "reference_score": (m - ref["mu"]) / ref["sigma"],
                "components": {
                    k: float(TARGETS[t] @ v0[i] @ TARGETS[t])
                    for k, v0 in components.items()
                },
            }
        results.append(
            {
                "arm": arm,
                "target_time": raw["target_time"],
                "future_conditions": PLAN,
                "targets": targets,
                "joint_mean": mean[i].tolist(),
                "joint_covariance": total[i].tolist(),
                "joint_components": {k: v[i].tolist() for k, v in components.items()},
                "qmc_ess_min": float(np.min(ess_parts, axis=0)[i]),
                "approximation": "ADF-state/prior-QMC-parameter/cut-source-reader;moment-normal intervals",
                **support[i],
            }
        )
    return results


def cold_scores(inputs, bundle):
    """Batch-independent source-frozen primary scoring with bounded memory."""
    out = []
    for start in range(0, len(inputs), 48):
        out.extend(score_panel(inputs[start : start + 48], bundle))
    return out


def main():
    """Score authenticated bundle and histories in a standalone process."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    payload = args.bundle.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.sha256:
        raise ValueError("bundle hash mismatch")
    inputs = [json.loads(x) for x in args.input.read_text().splitlines()]
    results = cold_scores(inputs, json.loads(payload))
    with args.output.open("x") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
