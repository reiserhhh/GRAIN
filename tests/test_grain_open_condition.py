#!/usr/bin/env python3
"""Open-condition assignment tests on synthetic vectors and a tiny bundle."""

import numpy as np
import pytest

from grain_research import open_condition as oc


def make_bundle():
    fitted = {"mean": [0.1, -0.2], "covariance": [[0.04, 0.01], [0.01, 0.05]], "noise": [0.3, 0.3]}
    return {"bootstrap": [fitted, {**fitted, "mean": [0.12, -0.18]}]}


def test_assigner_separates_and_is_calibrated_at_prototypes():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (10, 5)) + np.array([1, 0, 0, 0, 0])
    b = rng.normal(0, 0.1, (10, 5)) + np.array([0, 1, 0, 0, 0])
    for posterior in ("isotropic", "lda1d"):
        model = oc.learn_prototype_assigner(np.r_[a, b], np.r_[-np.ones(10), np.ones(10)], posterior=posterior)
        pi = oc.assign(np.r_[a, b], model)
        assert (pi[:10] < 0.5).all() and (pi[10:] > 0.5).all()
        at_minus = oc.assign(np.array([model["mu_minus"]]), model)[0]
        at_plus = oc.assign(np.array([model["mu_plus"]]), model)[0]
        assert at_minus < 0.5 < at_plus
    lda = oc.learn_prototype_assigner(np.r_[a, b], np.r_[-np.ones(10), np.ones(10)], posterior="lda1d")
    assert np.mean(np.abs(oc.assign(np.r_[a, b], lda) - 0.5)) > 0.4


def test_soft_readout_reduces_to_hard_at_certainty():
    bundle = make_bundle()
    y = np.array([0.8, -0.6, 0.7, -0.5])
    hard = oc.score_person_open(y, np.array([0.0, 1.0, 0.0, 1.0]), bundle, mode="soft")
    fitted = bundle["bootstrap"][0]
    from grain_research.contextual_instrument import PROFILE_TARGETS, gaussian_readout

    design = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    result = gaussian_readout(fitted["mean"], fitted["covariance"], design, np.array(fitted["noise"])[[0, 1, 0, 1]])
    t = PROFILE_TARGETS["level"]
    expected_first = float(t @ result["offset"] + (t @ result["kernel"]) @ y)
    soft_first, _ = oc.soft_profile_readout(y, np.array([0.0, 1.0, 0.0, 1.0]), fitted)
    assert float(t @ soft_first["offset"] + (t @ soft_first["kernel"]) @ y) == pytest.approx(expected_first, abs=1e-9)
    assert hard["status"] == "open_condition_working_measurement"


def test_uncertain_assignment_inflates_variance_and_abstain_drops():
    bundle = make_bundle()
    y = np.array([0.8, -0.6, 0.7, -0.5])
    certain = oc.score_person_open(y, np.array([0.01, 0.99, 0.01, 0.99]), bundle)
    unsure = oc.score_person_open(y, np.array([0.5, 0.5, 0.5, 0.5]), bundle)
    assert unsure["targets"]["condition"]["variance"] > certain["targets"]["condition"]["variance"]
    dropped = oc.score_person_open(y, np.array([0.5, 0.5, 0.9, 0.1]), bundle, mode="abstain", tau=0.2)
    assert dropped["used_events"] == 2
    empty = oc.score_person_open(y, np.array([0.5] * 4), bundle, mode="abstain", tau=0.2)
    assert empty["status"] == "insufficient_qualified_evidence"


def test_block_correlated_inflation_matches_independent_when_clauses_unique():
    bundle = make_bundle()
    y = np.array([0.8, -0.6, 0.7, -0.5])
    pi = np.array([0.3, 0.8, 0.4, 0.9])
    independent = oc.score_person_open(y, pi, bundle)
    unique = oc.score_person_open(y, pi, bundle, clause_ids=np.array([0, 1, 2, 3]))
    for t in independent["targets"]:
        assert unique["targets"][t]["score"] == pytest.approx(independent["targets"][t]["score"], abs=1e-9)
        assert unique["targets"][t]["variance"] == pytest.approx(independent["targets"][t]["variance"], abs=1e-9)
    shared = oc.score_person_open(y, pi, bundle, clause_ids=np.array([0, 0, 0, 0]))
    assert shared["clause_correlated"]
    assert shared["targets"]["condition"]["score"] != pytest.approx(independent["targets"]["condition"]["score"], abs=1e-12)
