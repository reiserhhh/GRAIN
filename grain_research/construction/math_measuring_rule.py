#!/usr/bin/env python3
"""Common-law evidence updates separated from observation-model replacement."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_source_event_coupling as S
from grain_research.construction.math_relational_diagnostic_update import BITS

RULE = "HM275-source-label-conditional-measuring-rule-v1"


def normalized(prior, log_channels):
    """Condition one declared positive source law without choosing a target."""
    p, h = np.asarray(prior, float), np.asarray(log_channels, float)
    if (
        p.ndim != 1
        or len(p) < 2
        or h.shape[-1:] != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(h).all()
        or np.any(p <= 0)
        or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("A positive normalized prior and finite channels are required")
    a = h + np.log(p)
    return np.exp(a - logsumexp(a, axis=-1, keepdims=True))


def target_odds(profile):
    """Read all five binary SOURCE keys without treating other readouts as probabilities."""
    p = np.asarray(profile, float)
    if p.shape[-1:] != (32,) or np.any(p <= 0) or not np.isfinite(p).all():
        raise ValueError("Full strictly positive32-profile law required")
    if np.max(abs(p.sum(axis=-1) - 1)) > 1e-12:
        raise ValueError(
            "Nonexhaustive candidate likelihoods are not profile probabilities"
        )
    logp = np.log(p)
    return np.stack(
        [
            logsumexp(logp[..., BITS[:, q] == 1], axis=-1)
            - logsumexp(logp[..., BITS[:, q] == 0], axis=-1)
            for q in range(5)
        ],
        axis=-1,
    )


def source_rule(logs, codes, relink):
    """Compile four views of the same SOURCE target, retaining distinct references."""
    original = S.profile_readings(logs, codes, relink)
    k, g = np.asarray(logs, float), np.asarray(codes)
    marginal = logsumexp(k, axis=-1) - np.log(k.shape[-1])
    counts = np.bincount(g, minlength=32)
    single = np.stack(
        [
            logsumexp(marginal[:, g == code], axis=1) - np.log(counts[code])
            for code in range(32)
        ],
        axis=-1,
    )
    singles = normalized(original["profile_prior"], single)
    profiles = np.stack(
        (
            singles[:, 0],
            singles[:, 1],
            original["profile"][:, 0],
            original["profile"][:, 1],
        ),
        axis=1,
    )
    targets = profiles @ BITS
    odds = target_odds(profiles)
    left, right, full, product = np.moveaxis(targets, 1, 0)
    lo, ro, fo, jo = np.moveaxis(odds, 1, 0)
    return {
        "profiles": profiles,
        "targets": targets,
        "log_odds": odds,
        "source_reference": original["profile_prior"] @ BITS,
        "evidence_after_left": full - left,
        "evidence_after_right": full - right,
        "model_replacement": full - product,
        "equal_singleton_departure": full - (left + right) / 2,
        "evidence_log_BF_after_left": fo - lo,
        "evidence_log_BF_after_right": fo - ro,
        "model_log_odds_change": fo - jo,
        "old_full_and_product": original["targets"][:, :2],
    }


def cold(bundle, record):
    """Score one addressed pair; behavior likelihoods and application outcomes are excluded."""
    S.cold(bundle, record)  # Reuse the registered chronology and evidence-ID contract.
    logs = S.event_logs(
        np.asarray([e["vector"] for e in record["events"]], float)[None],
        bundle["prototypes"],
        float(bundle["variance"]),
    )
    values = source_rule(logs, bundle["codes"], bundle["relink"])
    return {
        "rule_id": RULE,
        "frame_key": record["frame_key"],
        "criterion": "authorized_SOURCE_binary_label_keys_not_independent_behavior",
        "reference": "frozen_SOURCE_empirical_full32_profile_population",
        "profile_order": [
            "left_marginal",
            "right_marginal",
            "observed_pair_F",
            "author_product_J",
        ],
        "accounts": {
            key: value.tolist() if key == "source_reference" else value[0].tolist()
            for key, value in values.items()
            if key != "old_full_and_product"
        },
        "native_IPIP_reader_not_a_probability_of_SOURCE_labels": True,
        "overlapping_pairs_not_independent": True,
        "new_LLM_API_training_or_application_labels": False,
    }


def world(accuracy, rho, prior):
    """Construct a complete declared two-event law with fixed singleton channels."""
    a, r, p = float(accuracy), float(rho), float(prior)
    if not all(np.isfinite(x) for x in (a, r, p)) or not (0 < a < 1 and 0 < p < 1):
        raise ValueError("Finite emission accuracy and nondegenerate prior required")
    v, k = a * (1 - a), r * a * (1 - a)
    high = np.array([[a * a + k, v - k], [v - k, (1 - a) ** 2 + k]])
    if np.any(high <= 0):
        raise ValueError(
            "This registered census requires strictly positive joint support"
        )
    channels = np.stack((high[::-1, ::-1], high))
    source = np.array([1 - p, p])[:, None, None] * channels
    joint = source.sum(axis=0)
    eta = source[1] / joint
    sx, sz = source.sum(axis=2), source.sum(axis=1)
    ex, ez = sx[1] / sx.sum(axis=0), sz[1] / sz.sum(axis=0)
    product = channels.sum(axis=2)[:, :, None] * channels.sum(axis=1)[:, None, :]
    jp = np.array([1 - p, p])[:, None, None] * product
    ej = jp[1] / jp.sum(axis=0)
    return {
        "joint_target_observation": source,
        "observation_mass": joint,
        "full": eta,
        "left": np.broadcast_to(ex[:, None], (2, 2)).copy(),
        "right": np.broadcast_to(ez[None, :], (2, 2)).copy(),
        "product": ej,
    }


def world_accounts(accuracy, rho, prior):
    """Enumerate perfect/wrong-key contacts and common-law innovation risk identities."""
    w = world(accuracy, rho, prior)
    joint, mass, eta = w["joint_target_observation"], w["observation_mass"], w["full"]
    source_change = eta - w["product"]
    average = (w["left"] + w["right"]) / 2
    readers = {"perfect_same_target": eta, "complement_wrong_key": 1 - eta}
    singles = {"perfect_same_target": average, "complement_wrong_key": 1 - average}

    def risk(value):
        return float(sum(np.sum(joint[y] * (y - value) ** 2) for y in (0, 1)))

    rows = []
    for name, value in readers.items():
        departure = value - singles[name]
        rows.append(
            {
                "rho": rho,
                "reader": name,
                "contact": float(np.sum(mass * source_change * departure)),
                "target_squared_risk": risk(value),
                "regret_against_declared_Y": risk(value) - risk(eta),
                "joint_over_own_singleton_mean_risk_gain": risk(singles[name])
                - risk(value),
            }
        )
    innovation = eta - w["left"]
    tower = np.sum(mass * innovation, axis=1) / mass.sum(axis=1)
    energy = float(np.sum(mass * innovation**2))
    return (
        w,
        rows,
        {
            "rho": rho,
            "conditional_innovation_mean": tower.tolist(),
            "innovation_energy": energy,
            "single_left_minus_full_risk": risk(w["left"]) - risk(eta),
            "singleton_mean_minus_full_risk": risk(average) - risk(eta),
            "full_risk": risk(eta),
            "product_risk": risk(w["product"]),
            "prior_minus_full_risk": risk(float(prior)) - risk(eta),
        },
    )
