"""Geometry comparison tests use generated vectors, never human labels."""

from types import SimpleNamespace

import numpy as np
import pytest

from grain_research.vector_mechanisms.external_profiles import (
    READY,
    landmark_information_audit,
    profile_features,
    ready_cohorts,
)
from suica_core.v7_geometry import GeometryBundle, fit_geometry_bundle


def test_landmark_difference_identity_and_unidentified_directions():
    rng = np.random.default_rng(18)
    landmarks = rng.normal(size=(5, 12))
    points = rng.normal(size=(50, 12))
    audit = landmark_information_audit(points, landmarks)
    assert audit["affine_span_rank"] == 4
    assert audit["generic_unidentified_angular_degrees"] == 7
    assert audit["parallel_reconstruction_max_error"] < 1e-12
    assert audit["distance_squared_reconstruction_max_error"] < 1e-12
    assert audit["noninjectivity_witness"]["point_separation"] == pytest.approx(2)
    assert audit["noninjectivity_witness"]["identified_distance_max_difference"] < 1e-12


def test_full_affine_rank_landmarks_identify_query_coordinates():
    rng = np.random.default_rng(21)
    audit = landmark_information_audit(rng.normal(size=(8, 3)), rng.normal(size=(6, 3)))
    assert audit["perpendicular_dimensions"] == 0
    assert audit["noninjectivity_witness"] is None
    assert audit["parallel_reconstruction_max_error"] < 1e-12


def test_arm_cohort_requires_common_support_without_reclassifying_raw():
    result = ready_cohorts(
        {
            "raw": [READY, READY, "NO"],
            "cue_mask": ["NO", READY, READY],
            "placebo_mask": [READY, READY, READY],
        }
    )
    assert result["raw_ready"].tolist() == [True, True, False]
    assert result["all_arms_ready"].tolist() == [False, True, False]
    with pytest.raises(ValueError, match="aligned"):
        ready_cohorts({"raw": [READY], "cue_mask": [], "placebo_mask": [READY]})


def test_profile_rejects_nonfinite_input_before_imputing_into_formal_score():
    with pytest.raises(ValueError, match="complete features"):
        profile_features(SimpleNamespace(feature_names=["a", "b"]), [[np.nan, 1]], [36])


def test_profile_is_query_batch_independent_and_retains_refusal():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(40, 6))
    # Fit only the synthetic reference, not query subjects.
    fitted = fit_geometry_bundle(
        x,
        feature_names=[f"x{i}" for i in range(6)],
        operator={"kind": "synthetic"},
        representation={"kind": "synthetic"},
        runtime_artifact={"kind": "synthetic"},
        reference_population={"n": 40},
        min_units_for_score=24,
    )
    bundle = (
        fitted
        if isinstance(fitted, GeometryBundle)
        else GeometryBundle.from_dict(fitted)
    )
    a, _ = profile_features(bundle, x[:2], [36, 0])
    b, _ = profile_features(bundle, x[:1], [36])
    for key in ("canonical_distances", "canonical_shape", "sorted_distances"):
        np.testing.assert_allclose(a[key][:1], b[key], atol=1e-12)
    assert a["status"][1] != READY
    assert np.max(abs(a["canonical_shape"].mean(1))) < 1e-12
