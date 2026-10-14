#!/usr/bin/env python3
"""Focused common-target/reference tests, not personality-accuracy acceptance."""

import numpy as np
import pytest

from grain_research.construction import math_measuring_rule as M
from grain_research.construction import math_source_event_coupling as S


@pytest.mark.parametrize("rho", [0, 0.5, -0.25])
def test_complete_law_and_tower(rho):
    """Each fixed world preserves marginals and common-law refinement identities."""
    w, _, a = M.world_accounts(0.7, rho, 0.5)
    np.testing.assert_allclose(w["joint_target_observation"].sum(), 1, atol=1e-15)
    np.testing.assert_allclose(w["left"][:, 0], [0.7, 0.3], atol=1e-15)
    np.testing.assert_allclose(a["conditional_innovation_mean"], 0, atol=1e-15)
    assert a["innovation_energy"] == pytest.approx(a["single_left_minus_full_risk"])


def test_negative_contact_can_mean_perfect_target_reading():
    """A negative old-style contact does not identify a target-mismatch defect."""
    _, rows, a = M.world_accounts(0.7, 0.5, 0.5)
    assert rows[0]["contact"] < 0
    assert rows[0]["regret_against_declared_Y"] == 0
    assert a["singleton_mean_minus_full_risk"] > 0
    assert rows[1]["contact"] > 0
    assert rows[1]["regret_against_declared_Y"] > 0


def test_conditional_independence_and_complementarity():
    """Correlation signs change contact without changing the criterion or singleton law."""
    w, rows, _ = M.world_accounts(0.7, 0, 0.5)
    np.testing.assert_allclose(w["full"], w["product"], atol=1e-15)
    assert abs(rows[0]["contact"]) < 1e-15
    _, rows, _ = M.world_accounts(0.7, -0.25, 0.5)
    assert rows[0]["contact"] > 0
    assert rows[1]["contact"] < 0


@pytest.mark.parametrize(
    "a,rho,p", [(0.7, 1, 0.5), (0.7, -1, 0.5), (0.7, 0, 1), (float("nan"), 0, 0.5)]
)
def test_invalid_worlds_rejected(a, rho, p):
    """No zero-support smoothing or invalid prior rescue occurs."""
    with pytest.raises(ValueError):
        M.world(a, rho, p)


def fixture():
    """Give every profile one author and two positive addressed source pairs."""
    codes = np.arange(32)
    prototypes = np.arange(32 * 2 * 2 * 2, dtype=float).reshape(32, 2, 2, 2) / 100
    bundle = {
        "prototypes": prototypes,
        "codes": codes,
        "relink": np.tile([1, 0], (32, 1)),
        "variance": 0.4,
    }
    record = {
        "frame_key": "test-pair",
        "sampling_regime": "next_cached_record_within_registered_window",
        "events": [
            {
                "event_id": f"test-{i}",
                "vector": [0.2 + i / 10, 0.5],
                "time": i,
                "window": 0,
                "condition": i,
            }
            for i in range(2)
        ],
    }
    return bundle, record


def test_source_rule_matches_original_and_telescopes():
    """Consolidation changes interpretation, not the old F/J numerical definitions."""
    b, record = fixture()
    result = M.cold(b, record)["accounts"]
    targets = np.asarray(result["targets"])
    old = S.cold(b, record)["laws"]
    np.testing.assert_allclose(targets[2], old["observed_pair"]["targets"])
    np.testing.assert_allclose(targets[3], old["within_author_product"]["targets"])
    np.testing.assert_allclose(
        targets[0] + result["evidence_after_left"], targets[2], atol=1e-15
    )
    np.testing.assert_allclose(
        targets[3] + result["model_replacement"], targets[2], atol=1e-15
    )
    assert (
        np.max(
            abs(np.asarray(result["evidence_after_left"]) - result["model_replacement"])
        )
        > 1e-3
    )


def test_duplicate_event_and_label_injection_rejected():
    """The cold route retains the existing event/label exclusion boundary."""
    b, r = fixture()
    r["events"][1]["event_id"] = r["events"][0]["event_id"]
    with pytest.raises(ValueError):
        M.cold(b, r)
    b, r = fixture()
    r["answer"] = 1
    with pytest.raises(ValueError):
        M.cold(b, r)


def test_native_candidate_mass_is_not_target_posterior():
    """A balanced or incomplete likelihood list cannot masquerade as this profile law."""
    with pytest.raises(ValueError):
        M.target_odds(np.full(32, 0.001))
    with pytest.raises(ValueError):
        M.target_odds(np.full(50, 0.02))
    with pytest.raises(ValueError):
        M.normalized([0.1, 0.1], np.zeros(2))
