#!/usr/bin/env python3
"""Focused synthetic checks of the new SOURCE-to-observed-behavior interface."""

import numpy as np
import pytest

from grain_research.construction import math_author_belief as B
from grain_research.construction import math_behavior_pushforward as P


def bundle():
    """Supply a small nondegenerate source law, not a human personality truth."""
    paths = np.array([[[0, 1, 0], [1, 0, 1]], [[1, 1, 0], [0, 0, 1]]])
    targets = np.array([[[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]] * 2)
    result = B.compile_source(paths, targets, np.arange(2), states=2)
    result["behavior"] = np.array([[1.0, 4.0], [5.0, 2.0]])
    return result


def test_metadata_constant_and_distinct():
    """Check exact behavioral units and opportunity-dependent constants."""
    constant = P.behavioral_observation([3, 3, 3], [0, 0, 0], [2, 2, 2])
    np.testing.assert_allclose(constant, [np.log(4), 0, 0, 0, 0, 0, 1 / 3])
    distinct = P.behavioral_observation([0, 0, 0], [0, 1, 2], [1, 2, 3])
    np.testing.assert_allclose(distinct, [0, 0, np.log(2), 0, np.log(3), 1, 1])


def test_profile_projection_identity():
    """Both mode/profile terms survive exact reconstruction."""
    b = bundle()
    b["codes"][:] = 0
    w = np.array([0.9, 0.1])
    r = P.pushforward(b, w)
    np.testing.assert_allclose(r["full_mode"], w @ b["behavior"])
    np.testing.assert_allclose(r["actual_profile"], b["behavior"].mean(0))
    np.testing.assert_allclose(
        r["actual_profile"] + r["within_profile"][0], r["full_mode"]
    )
    assert abs(r["within_profile"][0, 0]) > 1


def test_relabeling_does_not_change_full_mode():
    """Meaning grouping may change; the label-blind mode readout must not."""
    b = bundle()
    first = P.pushforward(b, np.array([0.3, 0.7]))
    b["codes"][:] = 0
    second = P.pushforward(b, np.array([0.3, 0.7]))
    np.testing.assert_array_equal(first["full_mode"], second["full_mode"])
    assert not np.allclose(first["actual_profile"], second["actual_profile"])


def test_source_order_equivariance():
    """Reorder source identities and values together, preserving the measuring law."""
    b = bundle()
    first = P.infer(b, np.array([0, 1, 0]))
    for key in ("alpha0", "alpha", "behavior"):
        b[key] = b[key][::-1]
    b["codes"] = b["codes"][:, ::-1]
    second = P.infer(b, np.array([0, 1, 0]))
    np.testing.assert_allclose(first["full_mode"], second["full_mode"])


def test_risk_zero_and_known_positive():
    """The gain can be positive, zero or negative; no forced success."""
    y = np.array([[1.0, 2.0], [-1.0, -2.0]])
    np.testing.assert_array_equal(
        P.risk_gain(y, np.zeros_like(y), np.zeros(2), np.ones(2)), 0
    )
    assert P.risk_gain(y, y, np.zeros(2), np.ones(2)).mean() > 0
    assert P.risk_gain(y, -y, np.zeros(2), np.ones(2)).mean() < 0
    with pytest.raises(ValueError):
        P.risk_gain(y, y, np.zeros(2), np.zeros(2))


def test_missing_profile_and_invalid_weights():
    """Unsupported profiles are marked, not conflated with neutral human meaning."""
    table, support = P.profile_table(np.ones((2, 1)), np.array([0, 31]))
    assert support.sum() == 2 and table.shape == (32, 1)
    with pytest.raises(ValueError):
        P.pushforward(bundle(), np.array([0.1, 0.1]))


def test_rejects_nonchronological_and_negative_metadata():
    """Do not rewrite bad opportunities as harmless observed behavior."""
    with pytest.raises(ValueError):
        P.behavioral_observation([1, -1], [0, 1], [0, 1])
    with pytest.raises(ValueError):
        P.behavioral_observation([1, 1], [1, 0], [0, 1])


def test_cold_rejects_future_truth_and_personality_labels():
    """Application truth cannot enter the measuring input."""
    record = {
        "author_key": "opaque",
        "episodes": [
            {
                "events": [
                    {"event_id": "a", "timestamp": 1, "cell": 0},
                    {"event_id": "b", "timestamp": 2, "cell": 1},
                ]
            }
        ],
    }
    bad = record | {"personality": [0, 1, 0, 1, 0]}
    with pytest.raises(ValueError):
        P.cold(bundle(), bad)
    bad = record | {
        "episodes": record["episodes"]
        + [
            {
                "events": [
                    {"event_id": "c", "timestamp": 3, "cell": 0},
                ]
            }
        ]
    }
    with pytest.raises(ValueError):
        P.cold(bundle(), bad)
