#!/usr/bin/env python3
"""Preserve singleton target meaning while changing multi-event dependence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_joint_evidence as J

VERSION = "HM175_target_profile_composition_v1"
MODELS = ("target_iid", "target_profile", "shared_author")


def code_hash():
    """Identify the scientific profile constructor, not an optimized reader."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def profiles(p, y):
    """Pool only inside observed full five-bit labels, preserving their mass."""
    p, y = J.validate_source(p, y)
    labels, inverse, number = np.unique(
        y.astype(int), axis=0, return_inverse=True, return_counts=True
    )
    mass = np.array([p[inverse == c].mean(0) for c in range(len(labels))])
    return labels, number, number / len(p), mass, inverse


def build_bundle(parent):
    """Build a SOURCE-only hierarchy without class selection or smoothing."""
    p, y = J.validate_bundle(parent)
    labels, number, weight, mass, _ = profiles(p, y)
    return {
        "version": VERSION,
        "code_hash": code_hash(),
        "parent_bundle": parent,
        "profile_labels": labels.tolist(),
        "profile_counts": number.tolist(),
        "profile_weights": weight.tolist(),
        "profile_mass": mass.tolist(),
        "observation_law": "empirical_full_target_profile;conditional_iid_given_profile",
        "smoothing": 0,
    }


def validate_bundle(bundle):
    """Reject extra target outcomes, altered means or a changed implementation."""
    if bundle.get("version") != VERSION or bundle.get("code_hash") != code_hash():
        raise ValueError("Unknown profile implementation")
    rebuilt = build_bundle(bundle["parent_bundle"])
    if set(bundle) != set(rebuilt) or bundle != rebuilt:
        raise ValueError("Changed profile law or extra fitting field")
    return bundle


def evaluate(bundle, cells):
    """Evaluate profile products and old comparators on the SAME observed events."""
    old = J.evaluate(bundle["parent_bundle"], cells)
    labels = np.asarray(bundle["profile_labels"])
    weight = np.asarray(bundle["profile_weights"])
    mass = np.asarray(bundle["profile_mass"])
    ll = np.log(weight) + J.log_likelihoods(mass, J.counts(cells))
    joints = np.array(
        [[logsumexp(ll[labels[:, q] == g]) for g in (0, 1)] for q in range(5)]
    )
    norm = logsumexp(joints, axis=-1)
    with np.errstate(invalid="ignore"):
        p = np.exp(joints[:, 1] - norm)
        odds = joints[:, 1] - joints[:, 0]
    return {
        "probabilities": np.stack(
            [old["probabilities"][1], p, old["probabilities"][2]]
        ),
        "binary_log_odds": np.stack(
            [old["binary_log_odds"][0], odds, old["binary_log_odds"][1]]
        ),
        "posterior_supported": np.stack(
            [
                old["posterior_supported"][0],
                np.isfinite(norm),
                old["posterior_supported"][1],
            ]
        ),
        "finite_log_odds": np.stack(
            [old["finite_log_odds"][0], np.isfinite(odds), old["finite_log_odds"][1]]
        ),
        "events": old["events"],
    }


def two_event_accounts(bundle):
    """Separate within-profile and between-profile dependence with signed entries."""
    parent = bundle["parent_bundle"]
    p, y = (
        np.asarray(parent["source_mass"]),
        np.asarray(parent["source_binary_targets"]),
    )
    labels, weight = (
        np.asarray(bundle["profile_labels"]),
        np.asarray(bundle["profile_weights"]),
    )
    mass, prior = np.asarray(bundle["profile_mass"]), np.asarray(parent["prior"])
    law = np.empty((3, 5, 2, 8, 8))
    for q in range(5):
        for g in (0, 1):
            idx, c = y[:, q] == g, labels[:, q] == g
            rho = np.asarray(parent["rho1" if g else "rho0"])[:, q]
            law[0, q, g] = (prior[q] if g else 1 - prior[q]) * np.outer(rho, rho)
            law[1, q, g] = (mass[c].T * weight[c]) @ mass[c]
            law[2, q, g] = p[idx].T @ p[idx] / len(p)
    within, between = law[2] - law[1], law[1] - law[0]
    signed = lambda x: (
        x[:, 1] / prior[:, None, None] - x[:, 0] / (1 - prior[:, None, None])
    )
    return {
        "joint_laws": law,
        "within_profile": within,
        "between_profiles": between,
        "total_dependence": law[2] - law[0],
        "signed_within": signed(within),
        "signed_between": signed(between),
        "signed_total": signed(law[2] - law[0]),
    }


def invariants(bundle):
    """Check the full label prior and every singleton before evidence composition."""
    parent = bundle["parent_bundle"]
    labels, weight, mass = (
        np.asarray(bundle[k])
        for k in ("profile_labels", "profile_weights", "profile_mass")
    )
    p, y = (
        np.asarray(parent["source_mass"]),
        np.asarray(parent["source_binary_targets"]),
    )
    singleton = mass.T @ (weight[:, None] * labels)
    return {
        "joint_profile_prior_error": float(
            max(
                abs(w - np.mean(np.all(y == c, axis=1)))
                for c, w in zip(labels, weight, strict=True)
            )
        ),
        "target_prior_error": float(np.max(np.abs(weight @ labels - y.mean(0)))),
        "one_event_marginal_error": float(np.max(np.abs(weight @ mass - p.mean(0)))),
        "one_event_target_joint_error": float(
            np.max(np.abs(singleton - p.T @ y / len(p)))
        ),
        "singleton_target_mean_error": float(
            np.max(
                np.abs(
                    singleton / (weight @ mass)[:, None] - parent["event_probability"]
                )
            )
        ),
    }


def cold_record(bundle, record):
    """Score outcome-free distinct events, retaining missing joint support."""
    if set(record) != {"author_key", "events"}:
        raise ValueError("Cold input excludes outcomes")
    events = record["events"]
    if any(set(e) != {"event_id", "cell"} for e in events):
        raise ValueError("Only identified event cells allowed")
    ids = [e["event_id"] for e in events]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Distinct event identities required")
    r = evaluate(bundle, [e["cell"] for e in events])
    clean = lambda v: np.where(np.isfinite(v), v, None).tolist()
    return {
        "author_key": record["author_key"],
        "events": r["events"],
        "status": "SCORED" if events else "NO_OBSERVATIONS",
        "probabilities": clean(r["probabilities"]),
        "binary_log_odds": clean(r["binary_log_odds"]),
        "posterior_supported": r["posterior_supported"].tolist(),
        "finite_log_odds": r["finite_log_odds"].tolist(),
    }
