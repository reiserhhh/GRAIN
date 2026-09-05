"""Label-free frozen geometry comparators and their exact information boundary."""

from __future__ import annotations

import numpy as np

from suica_core.v7_geometry import score_geometry_bundle
from suica_core.v8_bridge import canonical_orbit_distance_signatures
from suica_core.v8_external_connection import canonical_scale_residual

READY = "GEOMETRY_PROFILE_READY"


def profile_features(bundle, features, unit_counts, *, relative_tolerance=1e-8):
    """Preserve the intermediate, identified, sorted and row-normalized channels.

    Returned values outside the frozen support envelope are diagnostics only.
    Consumers must select the separately recorded ready cohort before scoring.
    No transform is fitted to the query authors or their psychological labels.
    """
    x = np.asarray(features, dtype=float)
    counts = np.asarray(unit_counts)
    if (
        x.ndim != 2
        or x.shape[1] != len(bundle.feature_names)
        or not np.isfinite(x).all()
        or counts.shape != (len(x),)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != counts.astype(int))
    ):
        raise ValueError("Invalid complete features or observation counts")
    q = (x - np.asarray(bundle.feature_center)) @ np.asarray(bundle.metric_whitener)
    scored = score_geometry_bundle(bundle, x, unit_counts=counts.astype(int))
    canonical, names, diagnostics = canonical_orbit_distance_signatures(
        q,
        np.asarray(bundle.reference_landmarks),
        relative_tolerance=relative_tolerance,
    )
    sorted_profile = np.asarray(scored["landmark_distance_profile"])
    error = float(np.max(abs(np.sort(canonical, axis=1) - sorted_profile)))
    if error > 1e-9:
        raise ValueError("Canonical and sorted distances disagree")
    return {
        "intermediate": x,
        "whitened": q,
        "canonical_distances": canonical,
        "sorted_distances": sorted_profile,
        "canonical_shape": canonical_scale_residual(canonical),
        "reference_radius": np.asarray(scored["reference_radius"]),
        "status": np.asarray(scored["status"]),
    }, diagnostics | {"canonical_names": names, "distance_replay_max": error}


def ready_cohorts(status_by_arm):
    """Keep raw support distinct from the outcome-blind three-arm intersection."""
    if set(status_by_arm) != {"raw", "cue_mask", "placebo_mask"}:
        raise ValueError("Require exactly the three frozen text arms")
    values = {k: np.asarray(v) for k, v in status_by_arm.items()}
    if (
        any(v.ndim != 1 for v in values.values())
        or len({len(v) for v in values.values()}) != 1
    ):
        raise ValueError("Arm support vectors are not aligned")
    raw = values["raw"] == READY
    common = np.logical_and.reduce([v == READY for v in values.values()])
    return {"raw_ready": raw, "all_arms_ready": common}


def landmark_information_audit(points, landmarks):
    """Verify which affine components identified landmark distances can encode.

    For A_j = l_j-l_0, squared distance differences recover A_j dot (q-l_0).
    They recover the projection onto span(A), plus the perpendicular norm, not
    the perpendicular direction. A reflected synthetic point is a witness of
    non-injectivity when the affine landmark span is not full dimensional.
    """
    q, l = np.asarray(points, dtype=float), np.asarray(landmarks, dtype=float)
    if (
        q.ndim != 2
        or l.ndim != 2
        or len(q) < 1
        or len(l) < 2
        or q.shape[1] != l.shape[1]
        or not np.isfinite(q).all()
        or not np.isfinite(l).all()
    ):
        raise ValueError("Expected finite compatible point/landmark matrices")
    a = l[1:] - l[0]
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    tolerance = max(a.shape) * np.finfo(float).eps * (s[0] if len(s) else 0)
    rank = int(np.sum(s > tolerance))
    basis = vt[:rank]
    centered = q - l[0]
    parallel = (centered @ basis.T) @ basis
    perpendicular = centered - parallel
    d2 = np.sum((q[:, None, :] - l[None, :, :]) ** 2, axis=2)
    rhs = (d2[:, [0]] - d2[:, 1:] + np.sum(a * a, axis=1)) / 2
    recovered = np.linalg.lstsq(a, rhs.T, rcond=None)[0].T
    recovered_perp2 = d2[:, 0] - np.sum(recovered * recovered, axis=1)
    rebuilt = (
        np.sum((recovered[:, None, :] - (l - l[0])[None, :, :]) ** 2, axis=2)
        + recovered_perp2[:, None]
    )
    witness = None
    if rank < q.shape[1]:
        direction = vt[rank]
        pair = np.stack([l[0] + direction, l[0] - direction])
        distances = np.linalg.norm(pair[:, None, :] - l[None, :, :], axis=2)
        witness = {
            "synthetic_only": True,
            "point_separation": float(np.linalg.norm(pair[0] - pair[1])),
            "identified_distance_max_difference": float(
                np.max(abs(distances[0] - distances[1]))
            ),
        }
    return {
        "ambient_dimensions": q.shape[1],
        "landmarks": len(l),
        "affine_span_rank": rank,
        "rank_tolerance": float(tolerance),
        "singular_values": s.tolist(),
        "perpendicular_dimensions": q.shape[1] - rank,
        "generic_unidentified_angular_degrees": max(0, q.shape[1] - rank - 1),
        "identified_distance_continuous_rank_upper_bound": min(q.shape[1], rank + 1),
        "parallel_reconstruction_max_error": float(np.max(abs(recovered - parallel))),
        "perpendicular_norm_squared_max_error": float(
            np.max(abs(recovered_perp2 - np.sum(perpendicular**2, axis=1)))
        ),
        "distance_squared_reconstruction_max_error": float(np.max(abs(rebuilt - d2))),
        "noninjectivity_witness": witness,
        "interpretation": "Ambient-space information bound, not proof of lost personality information or observed-manifold dimension",
    }
