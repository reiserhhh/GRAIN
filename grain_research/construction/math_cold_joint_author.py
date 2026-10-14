#!/usr/bin/env python3
"""Source-frozen same-unit observations, with inspectable author aggregation."""

from __future__ import annotations

import hashlib
import json

import numpy as np

VERSION = "GRAIN_cold_joint_author_v1"
TARGETS = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
    "anthropomorphism",
)
QUALIFICATIONS = ("focal_report", "other_report", "unresolved")


def canonical_hash(value):
    """Authenticate strict JSON values independently of whitespace."""
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def finite_matrix(values, columns):
    """Validate finite observation coordinates without repairing missing values."""
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != columns or not np.isfinite(x).all():
        raise ValueError("finite observation matrix with declared columns required")
    return x


def lift(g, q):
    """Retain same-unit g/query and query/query products BEFORE author averaging."""
    g, q = finite_matrix(g, 4), finite_matrix(q, 6)
    if len(g) != len(q):
        raise ValueError("joint observations require equal event identities")
    cross = np.einsum("ni,nj->nij", g, q).reshape(len(g), 24)
    pairs = np.column_stack(
        [q[:, j] * q[:, k] for j in range(6) for k in range(j + 1, 6)]
    )
    return np.column_stack((g, q, cross, pairs))


def feature_names():
    """Address all fixed features without assigning psychological construct names."""
    grain = [f"G{j + 1:02d}" for j in range(4)]
    return (
        grain
        + list(TARGETS)
        + [f"{g}*{q}" for g in grain for q in TARGETS]
        + [f"{TARGETS[j]}*{TARGETS[k]}" for j in range(6) for k in range(j + 1, 6)]
    )


def source_bundle(grain, query, lexical, observer_id, provenance):
    """Calibrate only source observation units, without labels or a teacher fit."""
    g, q = finite_matrix(grain, 4), finite_matrix(query, 6)
    if len(g) != len(q) or len(g) < 2 or np.any((q < 0) | (q > 1)):
        raise ValueError("at least two paired source observations required")
    gm, qm = g.mean(0), q.mean(0)
    gs, qs = g.std(0, ddof=1), q.std(0, ddof=1)
    if np.any(gs < 1e-8) or np.any(qs < 1e-8):
        raise ValueError("degenerate source reading channels")
    phi = lift((g - gm) / gs, (q - qm) / qs)
    sd = phi.std(0, ddof=1)
    bundle = {
        "version": VERSION,
        "observer_id": observer_id,
        "lexical": lexical,
        "grain_reference_id": canonical_hash(lexical),
        "source_units": len(g),
        "grain_center": gm.tolist(),
        "grain_scale": gs.tolist(),
        "query_center": qm.tolist(),
        "query_scale": qs.tolist(),
        "feature_center": phi.mean(0).tolist(),
        "feature_scale": np.maximum(sd, 1e-8).tolist(),
        "source_degenerate_features": np.flatnonzero(sd < 1e-8).tolist(),
        "feature_names": feature_names(),
        "source_conditions": ["unspecified"],
        "provenance": provenance,
        "not_latent_inverse_or_personality_calibration": True,
    }
    bundle["calibration_sha256"] = canonical_hash(bundle)
    return bundle


def verify_bundle(bundle):
    """Bind numeric calibration and observation identities before any cold use."""
    if bundle.get("version") != VERSION:
        raise ValueError("unknown bundle version")
    unsigned = {k: v for k, v in bundle.items() if k != "calibration_sha256"}
    if canonical_hash(unsigned) != bundle["calibration_sha256"]:
        raise ValueError("calibration drift")
    if canonical_hash(bundle["lexical"]) != bundle["grain_reference_id"]:
        raise ValueError("lexical observer drift")
    if bundle["feature_names"] != feature_names():
        raise ValueError("feature schema drift")


def observed_features(bundle, grain, query):
    """Apply fixed units; application batches cannot change the coordinate system."""
    g, q = finite_matrix(grain, 4), finite_matrix(query, 6)
    if len(g) != len(q) or np.any((q < 0) | (q > 1)):
        raise ValueError("paired bounded query observations required")
    gc = (g - bundle["grain_center"]) / bundle["grain_scale"]
    qc = (q - bundle["query_center"]) / bundle["query_scale"]
    phi = lift(gc, qc)
    return (phi - bundle["feature_center"]) / bundle["feature_scale"], gc, qc


def cold_score(bundle, record):
    """Score only supplied records, preserving choice mass, qualifications and units."""
    verify_bundle(bundle)
    if set(record) != {"author_key", "events"}:
        raise ValueError("only author_key and events are cold inputs")
    seen, eligible, counts = set(), [], dict.fromkeys(QUALIFICATIONS, 0)
    for event in record["events"]:
        if set(event) != {
            "event_id",
            "condition",
            "qualification",
            "observer_id",
            "grain",
            "query",
        }:
            raise ValueError(
                "unexpected event input; no truth or counterfactual tables"
            )
        identity = event["event_id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("unique nonempty event identity required")
        seen.add(identity)
        if event["observer_id"] != bundle["observer_id"]:
            raise ValueError("query observer does not match the source")
        if not isinstance(event["condition"], str) or not event["condition"]:
            raise ValueError("explicit condition required")
        qual = event["qualification"]
        if qual not in counts:
            raise ValueError("explicit evidence qualification required")
        counts[qual] += 1
        # Validate unqualified observations too, without treating them as a zero score.
        finite_matrix([event["grain"]], 4)
        raw_query = finite_matrix([event["query"]], 6)
        if np.any((raw_query < 0) | (raw_query > 1)):
            raise ValueError("bounded query observation required")
        if qual == "focal_report":
            eligible.append(event)
    result = {
        "author_key": record["author_key"],
        "qualification_counts": counts,
        "events": len(record["events"]),
        "eligible_events": len(eligible),
        "calibration_sha256": bundle["calibration_sha256"],
    }
    if not eligible:
        return result | {
            "status": "UNOBSERVED",
            "profile": None,
            "conditional_profiles": {},
            "contributions": [],
        }
    phi, g, q = observed_features(
        bundle, [e["grain"] for e in eligible], [e["query"] for e in eligible]
    )
    cov = g.T @ q / len(g) - np.outer(g.mean(0), q.mean(0))
    conditional = {}
    conditions = [e["condition"] for e in eligible]
    for name in sorted(set(conditions)):
        mask = np.asarray([c == name for c in conditions])
        conditional[name] = {
            "events": int(mask.sum()),
            "share": float(mask.mean()),
            "profile": phi[mask].mean(0).tolist(),
            "outside_source_condition_support": name not in bundle["source_conditions"],
        }
    return result | {
        "status": "OBSERVED_PROFILE",
        "profile": phi.mean(0).tolist(),
        "grain_query_covariance": cov.tolist(),
        "conditional_profiles": conditional,
        "contributions": [
            {
                "event_id": e["event_id"],
                "condition": e["condition"],
                "contribution": (v / len(phi)).tolist(),
            }
            for e, v in zip(eligible, phi, strict=True)
        ],
        "scope": "published focal reports; explicit supplied qualification; not psychological scores",
    }
