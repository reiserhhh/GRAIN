#!/usr/bin/env python3
"""Frozen source-law author measurements and addressed evidence actions.

The unit is one sampled early/late event pair. Averaging its conditional
reading is a reference-relative functional, not a full-history posterior.
"""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_observed_family as F
from grain_research.construction import math_route_relation as M

VERSION = "HM194_frozen_pair_measurement_v1"
REGIME = "self_selected_publication_early_late_current_events"
TARGETS = ("E", "N", "A", "C", "O")


def compile_instrument(parent_bundle, reference, thresholds, representation_id):
    """Compile all three existing laws once, without new labels or fitting."""
    if not isinstance(representation_id, str) or not representation_id:
        raise ValueError("A frozen representation identifier is required")
    if parent_bundle.get("version") != M.VERSION:
        raise ValueError("The frozen three-law source bundle is required")
    M.query_pair(
        parent_bundle,
        [
            {"event_id": "source-schema-a", "window": 0, "cell": 0},
            {"event_id": "source-schema-b", "window": 1, "cell": 0},
        ],
    )
    laws = np.asarray(parent_bundle["aggregate_profile_event_laws"], float)
    probe = M.actions(laws)
    rho, cut = np.asarray(reference, float), np.asarray(thresholds, float)
    if (
        rho.shape != (8, 8)
        or not np.isfinite(rho).all()
        or np.any(rho < 0)
        or not np.isclose(rho.sum(), 1, atol=1e-12, rtol=0)
        or cut.shape != (2,)
        or not np.isfinite(cut).all()
    ):
        raise ValueError(
            "Normalized pair reference and frozen two-coordinate cuts required"
        )
    if np.any((rho > 0) & ~probe["supported"]):
        raise ValueError("Reference has unsupported evidence; do not renormalize it")
    reading = np.where(probe["supported"][None, None], probe["after"], 0)
    route_mass = rho.reshape(2, 4, 2, 4).sum((1, 3))
    route_mean = np.full((3, 5, 2, 2), np.nan)
    for a in (0, 1):
        for b in (0, 1):
            sl = (slice(4 * a, 4 * a + 4), slice(4 * b, 4 * b + 4))
            if route_mass[a, b] > 0:
                route_mean[:, :, a, b] = (reading[:, :, sl[0], sl[1]] * rho[sl]).sum(
                    (2, 3)
                ) / route_mass[a, b]
    return {
        "version": VERSION,
        "regime": REGIME,
        "representation_id": representation_id,
        "targets": list(TARGETS),
        "models": list(M.MODELS),
        "thresholds": cut.tolist(),
        "reference_pair_law": rho.tolist(),
        "supported": probe["supported"].tolist(),
        "readings": [
            [
                [[None if not np.isfinite(x) else float(x) for x in row] for row in t]
                for t in model
            ]
            for model in probe["after"]
        ],
        "reference_reading": np.einsum("mqab,ab->mq", reading, rho).tolist(),
        "route_reference_mass": route_mass.tolist(),
        "route_reference_reading": [
            [
                [[None if not np.isfinite(x) else float(x) for x in row] for row in t]
                for t in model
            ]
            for model in route_mean
        ],
        "source_interpretation": "binary_Big5_calibration_NOT_new_personality_factors",
        "measurement_interpretation": "pair_functional_NOT_full_history_posterior_or_state_change",
    }


def instrument_arrays(bundle):
    """Check the registered reading/reference units without changing them."""
    keys = {
        "version",
        "regime",
        "representation_id",
        "targets",
        "models",
        "thresholds",
        "reference_pair_law",
        "supported",
        "readings",
        "reference_reading",
        "route_reference_mass",
        "route_reference_reading",
        "source_interpretation",
        "measurement_interpretation",
    }
    if (
        set(bundle) != keys
        or bundle["version"] != VERSION
        or bundle["regime"] != REGIME
        or bundle["targets"] != list(TARGETS)
        or bundle["models"] != list(M.MODELS)
        or bundle["source_interpretation"]
        != "binary_Big5_calibration_NOT_new_personality_factors"
        or bundle["measurement_interpretation"]
        != "pair_functional_NOT_full_history_posterior_or_state_change"
    ):
        raise ValueError("Unknown measurement instrument or target interpretation")
    psi = np.asarray(bundle["readings"], float)
    support = np.asarray(bundle["supported"], bool)
    rho = np.asarray(bundle["reference_pair_law"], float)
    ref = np.asarray(bundle["reference_reading"], float)
    mass = np.asarray(bundle["route_reference_mass"], float)
    route = np.asarray(bundle["route_reference_reading"], float)
    if (
        psi.shape != (3, 5, 8, 8)
        or support.shape != (8, 8)
        or rho.shape != (8, 8)
        or ref.shape != (3, 5)
        or mass.shape != (2, 2)
        or route.shape != (3, 5, 2, 2)
        or not np.isfinite(rho).all()
        or np.any(rho < 0)
        or not np.isclose(rho.sum(), 1, atol=1e-12, rtol=0)
        or np.any((rho > 0) & ~support)
        or not np.isfinite(psi[:, :, support]).all()
        or np.any((psi[:, :, support] < 0) | (psi[:, :, support] > 1))
        or not np.allclose(mass, rho.reshape(2, 4, 2, 4).sum((1, 3)))
    ):
        raise ValueError("Inconsistent reading units or supported reference")
    safe = np.where(support[None, None], psi, 0)
    if not np.allclose(ref, np.einsum("mqab,ab->mq", safe, rho), atol=1e-12, rtol=0):
        raise ValueError("Reference was recalibrated")
    for a in (0, 1):
        for b in (0, 1):
            sl = (slice(4 * a, 4 * a + 4), slice(4 * b, 4 * b + 4))
            if mass[a, b] > 0:
                expected = (safe[:, :, sl[0], sl[1]] * rho[sl]).sum((2, 3)) / mass[a, b]
                if not np.allclose(route[:, :, a, b], expected, atol=1e-12, rtol=0):
                    raise ValueError("Conditional reference was recalibrated")
            elif np.isfinite(route[:, :, a, b]).any():
                raise ValueError("An unobserved route cannot have a reference")
    F.states(np.zeros((0, 15)), bundle["thresholds"])
    return psi, support, ref, mass, route


def addressed_cells(bundle, packet):
    """Encode current vector events only; check identity and chronological scope."""
    if (
        set(packet) != {"author_key", "regime", "representation_id", "events"}
        or not isinstance(packet["author_key"], str)
        or not packet["author_key"]
        or packet["regime"] != bundle["regime"]
        or packet["representation_id"] != bundle["representation_id"]
    ):
        raise ValueError("Author, regime and representation must match the instrument")
    events = packet["events"]
    required = {"event_id", "window", "condition", "timestamp", "coordinates"}
    for e in events:
        if (
            set(e) != required
            or not isinstance(e["event_id"], str)
            or not e["event_id"]
            or type(e["window"]) is not int
            or e["window"] not in (0, 1)
            or type(e["condition"]) is not int
            or e["condition"] not in (0, 1)
            or type(e["timestamp"]) not in (int, float)
            or not np.isfinite(e["timestamp"])
        ):
            raise ValueError(
                "Addressed vector events only; labels and native answers forbidden"
            )
    if len({e["event_id"] for e in events}) != len(events):
        raise ValueError("Repeated event identity, not independent new evidence")
    groups = [[e for e in events if e["window"] == w] for w in (0, 1)]
    if all(groups) and max(e["timestamp"] for e in groups[0]) > min(
        e["timestamp"] for e in groups[1]
    ):
        raise ValueError("Early and late windows overlap in time")
    cells = []
    for group in groups:
        x = (
            np.asarray([e["coordinates"] for e in group], float)
            if group
            else np.zeros((0, 15))
        )
        if x.shape != (len(group), 15):
            raise ValueError("Exactly15 coordinates per addressed event required")
        s = F.states(x, bundle["thresholds"])
        cells.append(s + 4 * np.array([e["condition"] for e in group], int))
    return groups, cells


def score(bundle, packet):
    """Measure the same events under all laws and retain every signed account."""
    psi, support, ref, reference_mass, route = instrument_arrays(bundle)
    groups, (a, b) = addressed_cells(bundle, packet)
    base = {"author_key": packet["author_key"], "counts": [len(a), len(b)]}
    if not len(a) or not len(b):
        return base | {"status": "MISSING_WINDOW", "relative": None}
    missing = np.argwhere(~support[np.ix_(a, b)])
    if len(missing):
        return base | {
            "status": "UNOBSERVED_SOURCE_PAIRS",
            "relative": None,
            "missing_pairs": sorted({(int(a[i]), int(b[j])) for i, j in missing}),
        }
    empirical = np.outer(
        np.bincount(a, minlength=8) / len(a), np.bincount(b, minlength=8) / len(b)
    )
    mass = empirical.reshape(2, 4, 2, 4).sum((1, 3))
    if np.any((mass > 0) & (reference_mass == 0)):
        return base | {"status": "UNOBSERVED_REFERENCE_ROUTES", "relative": None}
    values = psi[:, :, a[:, None], b[None, :]]
    mean = values.mean((2, 3))
    relative = mean - ref
    route_safe = np.where(reference_mass[None, None] > 0, route, 0)
    choice = np.einsum("mqab,ab->mq", route_safe, mass - reference_mass)
    within = mean - np.einsum("mqab,ab->mq", route_safe, mass)
    row, column = values.mean(3), values.mean(2)
    event_delta = None
    joint_delta = None
    if len(a) > 1 and len(b) > 1:
        event_delta = np.concatenate(
            (
                (mean[..., None] - row) / (len(a) - 1),
                (mean[..., None] - column) / (len(b) - 1),
            ),
            axis=2,
        )
        joint_delta = (
            values - row[..., :, None] - column[..., None, :] + mean[..., None, None]
        ) / ((len(a) - 1) * (len(b) - 1))
    return base | {
        "status": "SUPPORTED",
        "pair_average_NOT_history_posterior": mean,
        "relative": relative,
        "reference_route_composition": choice,
        "within_route_departure": within,
        "law_change_routed": relative[1] - relative[0],
        "law_change_within": relative[2] - relative[1],
        "law_change_total": relative[2] - relative[0],
        "empirical_pair_law": empirical,
        "route_composition": mass,
        "event_ids": [e["event_id"] for group in groups for e in group],
        "event_deletion_delta": event_delta,
        "joint_deletion_interaction": joint_delta,
    }
