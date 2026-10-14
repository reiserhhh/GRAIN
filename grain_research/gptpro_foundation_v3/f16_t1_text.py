#!/usr/bin/env python3
"""Source-calibrated categorical text observations for the four-cell T1 target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from .f16_t1_source import CONDITIONS, CONTRAST, posterior

VERSION = "F16_T1_probit_text_joint_channel_v1"
ARMS = ("hybrid_joint", "nli_joint", "hybrid_factorized")


def code_hash():
    """Bind this observation and integration implementation to its bundle."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def category(pole, qualification):
    """Use a fixed, explicit observation alphabet, not posterior probabilities."""
    values = np.asarray([pole, qualification], float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("finite reader outputs in [0,1] required")
    return 2 * int(pole >= 0.5) + int(qualification >= 0.5)


def channel_counts(rows, arm):
    """Calibrate on development texts only, disjoint from source Ridge fitting."""
    counts = np.zeros((2, 2, 4))
    seen = set()
    for row in rows:
        if row["role"] != "validation" or row["material_id"] in seen:
            raise ValueError("unique validation calibration texts required")
        seen.add(row["material_id"])
        b, q = row["pole"], row["qualification"]
        if b not in (0, 1) or q not in (0, 1):
            raise ValueError("binary designed calibration labels required")
        key = "nli_qualification" if arm == "nli_joint" else "grain_qualification"
        counts[int(b), int(q), category(row["pole_readout"], row[key])] += 1
    if (counts.sum(axis=2) == 0).any():
        raise ValueError("all pole/eligibility strata need calibration support")
    return counts


def channel_draws(counts, seed, number):
    """Draw source-only categorical error laws with Jeffreys row priors."""
    counts = np.asarray(counts, float)
    if counts.shape != (2, 2, 4) or (counts < 0).any() or not np.isfinite(counts).all():
        raise ValueError("invalid channel counts")
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.dirichlet(counts[b, q] + 0.5, size=number)
            for b in (0, 1)
            for q in (0, 1)
        ],
        axis=1,
    ).reshape(number, 2, 2, 4)


def factorize_channel(channel):
    """Remove within-channel dependence while preserving both output marginals."""
    joint = np.asarray(channel, float).reshape(2, 2, 2, 2)
    return (joint.sum(-1)[..., :, None] * joint.sum(-2)[..., None, :]).reshape(2, 2, 4)


def event_probabilities(means, sigma2, channel, eligible_rate=0.5):
    """Marginalize true pole and report eligibility without reading either truth."""
    m, e = np.asarray(means, float), np.asarray(channel, float)
    if (
        e.shape != (2, 2, 4)
        or not np.isfinite(e).all()
        or (e < 0).any()
        or not np.allclose(e.sum(-1), 1, atol=1e-12, rtol=0)
        or not np.isfinite(m).all()
        or not np.isfinite(sigma2)
        or sigma2 <= 0
        or not 0 <= eligible_rate <= 1
    ):
        raise ValueError("normalized channel and finite positive-noise law required")
    p0, p1 = ndtr(-m / np.sqrt(sigma2)), ndtr(m / np.sqrt(sigma2))
    return eligible_rate * (p0[..., None] * e[0, 1] + p1[..., None] * e[1, 1]) + (
        (1 - eligible_rate) / 2 * (e[0, 0] + e[1, 0])
    )


def oracle_channel():
    """Known material labels are a reference arm, never a real-reader input."""
    e = np.zeros((2, 2, 4))
    for b in (0, 1):
        for q in (0, 1):
            e[b, q, 2 * b + q] = 1
    return e


def posterior_from_nodes(counts, nodes, sigma2, channel):
    """Condition fixed prior nodes on joint category counts; return numerical ESS."""
    counts = np.asarray(counts, float)
    if counts.ndim != 3 or counts.shape[1:] != (4, 4):
        raise ValueError("author by four-condition by four-category counts required")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("nonnegative finite counts required")
    likelihood = event_probabilities(nodes, sigma2, channel).reshape(len(nodes), 16)
    log_l = np.log(np.maximum(likelihood, np.finfo(float).tiny))
    log_w = log_l @ counts.reshape(len(counts), 16).T
    log_w -= log_w.max(axis=0)
    w = np.exp(log_w)
    w /= w.sum(axis=0)
    delta = nodes @ CONTRAST
    means = delta @ w
    variances = (delta * delta) @ w - means * means
    if (variances < -1e-10).any():
        raise ValueError("negative numerical posterior variance")
    return means, np.maximum(variances, 0), 1 / np.sum(w * w, axis=0)


def score_counts(counts, bundle, arm="hybrid_joint", scramble=0):
    """Integrate each source/reader component separately, then equal-weight cuts."""
    if arm not in (*ARMS, "oracle"):
        raise ValueError("unknown observation arm")
    models = bundle["source_models"]
    mean_parts, var_parts, ess_parts = [], [], []
    for b, model in enumerate(models):
        sobol_seed = int(
            np.random.SeedSequence([bundle["seed"], 161, b, scramble]).generate_state(
                1
            )[0]
        )
        uniforms = qmc.Sobol(4, scramble=True, seed=sobol_seed).random_base2(
            bundle["qmc_power"]
        )
        normals = ndtri(np.clip(uniforms, np.finfo(float).eps, 1 - np.finfo(float).eps))
        nodes = np.asarray(model["mu"]) + normals @ np.linalg.cholesky(model["Sigma"]).T
        if arm == "oracle":
            channel = oracle_channel()
        else:
            key = "nli_joint" if arm == "nli_joint" else "hybrid_joint"
            channel = np.asarray(bundle["channel_draws"][key][b])
            if arm == "hybrid_factorized":
                channel = factorize_channel(channel)
        mean, var, ess = posterior_from_nodes(counts, nodes, model["sigma2"], channel)
        mean_parts.append(mean)
        var_parts.append(var)
        ess_parts.append(ess)
    means, within, ess = map(np.asarray, (mean_parts, var_parts, ess_parts))
    score = means.mean(0)
    personal, calibration = within.mean(0), means.var(axis=0, ddof=0)
    variance = personal + calibration
    results = []
    for i in range(len(counts)):
        results.append(
            {
                "target_id": "T1",
                "arm": arm,
                "score": float(score[i]),
                "variance": float(variance[i]),
                "lower": float(score[i] - 1.96 * np.sqrt(variance[i])),
                "upper": float(score[i] + 1.96 * np.sqrt(variance[i])),
                "personal_conditional_variance": float(personal[i]),
                "source_and_reader_variance": float(calibration[i]),
                "qmc_ess_min": float(ess[:, i].min()),
                "qmc_ess_median": float(np.median(ess[:, i])),
                "nodes_per_component": 2 ** bundle["qmc_power"],
                "source_components": len(models),
                "observed_events": int(np.sum(counts[i])),
                "prior_only": bool(np.sum(counts[i]) == 0),
                "support": "both_endpoint_texts"
                if counts[i, 0].sum() and counts[i, 3].sum()
                else "model_borrowed",
                "uncertainty_scope": "cut-source-bootstrap/reader-Dirichlet mixture;moment-normal interval;QMC error separate",
            }
        )
    return results


def numeric_comparator(rows, bundle, naive=False):
    """Use real eligible values or explicitly misspecified fractional Gaussian data."""
    counts, sums = np.zeros(4), np.zeros(4)
    for row in rows:
        k = CONDITIONS.index(row["z"])
        weight = row["grain_qualification"] if naive else row["qualification"]
        value = 2 * row["pole_readout"] - 1 if naive else row["y"]
        counts[k] += weight
        sums[k] += weight * value
    parts = [posterior(counts, sums, model) for model in bundle["source_models"]]
    means = np.array([CONTRAST @ m for m, _ in parts])
    within = np.array([CONTRAST @ v @ CONTRAST for _, v in parts])
    value, variance = float(means.mean()), float(within.mean() + means.var())
    return {
        "score": value,
        "variance": variance,
        "lower": value - 1.96 * np.sqrt(variance),
        "upper": value + 1.96 * np.sqrt(variance),
    }


def replay_history(raw, bundle):
    """Resolve text reports before reading; aliases/summaries add no observations."""
    if set(raw) != {"author_id", "target_time", "history", "world_contract"}:
        raise ValueError("unexpected input fields, including latent truth")
    if raw["world_contract"] != bundle["world_contract"]:
        raise ValueError("wrong text observation world")
    if not isinstance(raw["author_id"], str) or not raw["author_id"]:
        raise ValueError("author identity required")
    if type(raw["target_time"]) is not int or raw["target_time"] < 0:
        raise ValueError("nonnegative scoring clock required")
    active, events, seen, annotations = {}, {}, set(), []
    previous = -1
    for op in raw["history"]:
        kind = op.get("op")
        fields = {"record_id", "op", "recorded_at"}
        if kind in ("new", "revise"):
            fields.add("observation")
        if kind in ("revise", "retract", "alias"):
            fields.add("parent_id")
        if kind == "summary":
            fields.add("parents")
        if (
            kind not in ("new", "revise", "retract", "alias", "summary")
            or set(op) != fields
        ):
            raise ValueError("invalid text history operation")
        rid, at = op["record_id"], op["recorded_at"]
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("duplicate or missing report identity")
        if type(at) is not int or not previous <= at <= raw["target_time"]:
            raise ValueError("future or unordered report")
        previous = at
        seen.add(rid)
        if kind in ("new", "revise"):
            row = op["observation"]
            if set(row) != {"author_id", "event_id", "t", "z", "material_id"}:
                raise ValueError(
                    "text observations must not contain numeric or qualification truth"
                )
            if row["author_id"] != raw["author_id"] or row["z"] not in CONDITIONS:
                raise ValueError("wrong author or condition")
            if type(row["t"]) is not int or not 0 <= row["t"] <= at:
                raise ValueError("future or invalid event time")
            if row["material_id"] not in bundle["materials"]:
                raise ValueError("unregistered text reading")
            event = row["event_id"]
            if not isinstance(event, str) or not event:
                raise ValueError("explicit event identity required")
            context = (row["author_id"], row["t"], row["z"])
            if kind == "new" and event in events:
                raise ValueError("duplicate event requires revision/alias")
            if kind == "revise":
                parent = active.get(op["parent_id"])
                if (
                    parent is None
                    or parent["event_id"] != event
                    or events[event] != context
                ):
                    raise ValueError(
                        "revision must replace the active same event/context"
                    )
                del active[op["parent_id"]]
            events[event], active[rid] = context, dict(row)
        elif kind in ("alias", "retract"):
            if op["parent_id"] not in active:
                raise ValueError("missing active parent")
            if kind == "retract":
                del active[op["parent_id"]]
            else:
                annotations.append((rid, [op["parent_id"]]))
        else:
            parents = op["parents"]
            if (
                not isinstance(parents, list)
                or not parents
                or len(set(parents)) != len(parents)
                or not set(parents) <= active.keys()
            ):
                raise ValueError("summary needs distinct active reports")
            annotations.append((rid, parents))
    rows = sorted(active.values(), key=lambda r: (r["t"], r["event_id"]))
    return rows, [
        rid for rid, parents in annotations if not set(parents) <= active.keys()
    ]


def counts_from_input(raw, bundle, arm="hybrid_joint"):
    """Read only current text addresses and pinned reader outputs, never labels."""
    rows, stale = replay_history(raw, bundle)
    counts = np.zeros((4, 4))
    for row in rows:
        material = bundle["materials"][row["material_id"]]
        qkey = "nli_qualification" if arm == "nli_joint" else "grain_qualification"
        c = category(material["pole_readout"], material[qkey])
        counts[CONDITIONS.index(row["z"]), c] += 1
    return counts, {
        "active_materials": [r["material_id"] for r in rows],
        "stale_annotations": stale,
    }


def verify_bundle(bundle):
    """Authenticate the observation law and reject impossible calibration arrays."""
    if bundle["version"] != VERSION or bundle["scorer_code_hash"] != code_hash():
        raise ValueError("changed text observation formula")
    if bundle["world_contract"] != "F04_L_thresholded_text_eligibility_v1":
        raise ValueError("wrong text observation contract")
    if bundle.get("qualification_rate") != 0.5:
        raise ValueError("the frozen material design requires eligibility rate .5")
    if type(bundle["qmc_power"]) is not int or not 1 <= bundle["qmc_power"] <= 16:
        raise ValueError("bounded integer integration resolution required")
    n = len(bundle["source_models"])
    if n != bundle["source_components"] or n < 1:
        raise ValueError("incomplete source models")
    for source in bundle["source_models"]:
        mu, cov = np.asarray(source["mu"]), np.asarray(source["Sigma"])
        if (
            mu.shape != (4,)
            or cov.shape != (4, 4)
            or not np.isfinite(mu).all()
            or not np.isfinite(cov).all()
        ):
            raise ValueError("finite four-cell source prior required")
        if (
            not np.allclose(cov, cov.T, rtol=0, atol=1e-12)
            or not np.isfinite(source["sigma2"])
            or source["sigma2"] <= 0
        ):
            raise ValueError("symmetric prior and positive source noise required")
        np.linalg.cholesky(cov)
    for key in ("nli_joint", "hybrid_joint"):
        e = np.asarray(bundle["channel_draws"][key])
        if e.shape != (n, 2, 2, 4):
            raise ValueError("incomplete reader uncertainty")
        for row in e:
            event_probabilities(np.zeros((1, 4)), 1.0, row)
    for item in bundle["materials"].values():
        if set(item) != {
            "text",
            "text_sha256",
            "pole_readout",
            "nli_qualification",
            "grain_qualification",
        }:
            raise ValueError(
                "reader cache must not contain calibration or target truth"
            )
        if hashlib.sha256(item["text"].encode()).hexdigest() != item["text_sha256"]:
            raise ValueError("text address/hash mismatch")
        category(item["pole_readout"], item["grain_qualification"])
        category(item["pole_readout"], item["nli_qualification"])
    if (
        not np.isfinite(bundle["reference_mu"])
        or not np.isfinite(bundle["reference_sigma"])
        or bundle["reference_sigma"] <= 0
    ):
        raise ValueError("frozen positive source reference required")


def cold_scores(inputs, bundle):
    """Score only bundle and each author's text-event history; norms stay frozen."""
    verify_bundle(bundle)
    prepared = [counts_from_input(raw, bundle) for raw in inputs]
    results = score_counts(np.stack([x[0] for x in prepared]), bundle)
    for result, (_, support) in zip(results, prepared, strict=True):
        result.update(support)
        result["reference_score"] = (result["score"] - bundle["reference_mu"]) / bundle[
            "reference_sigma"
        ]
        result["reference_variance"] = (
            result["variance"] / bundle["reference_sigma"] ** 2
        )
    return results


def main():
    """Standalone authenticated JSON scoring; no source files or target truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    payload = args.bundle.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.sha256:
        raise ValueError("bundle hash mismatch")
    inputs = [json.loads(line) for line in args.input.read_text().splitlines()]
    results = cold_scores(inputs, json.loads(payload))
    with args.output.open("x") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
