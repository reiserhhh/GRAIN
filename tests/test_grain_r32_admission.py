"""R32 admission rule, triangulation identification and dynamic-parameter fields."""

import numpy as np

from grain_research.gptpro_foundation_v3.r32_admission import (
    Candidate,
    admit,
    check_dynamic_parameter,
    triangulate,
    two_indicator_counterexample,
)


def test_admission_requires_replication_and_condition():
    ok = Candidate("x", "dispositional", True, {"unique_experimental_signature": True})
    assert admit(ok)["admitted"]
    assert not admit(Candidate("x", "dispositional", False, {"unique_experimental_signature": True}))["admitted"]
    assert not admit(Candidate("x", "dispositional", True, {}))["admitted"]


def test_increment_needs_all_shortcut_controls():
    c = Candidate("x", "dispositional", True, {"incremental_cross_situational_prediction": True},
                  {"topic": True, "demographic": True, "language_proficiency": True, "social_desirability": False})
    assert not admit(c)["admitted"]
    c.shortcut_controls["social_desirability"] = True
    assert admit(c)["admitted"]


def test_triad_identified_and_dyad_not():
    lam = np.array([0.8, 0.6, 0.4])
    cov = np.outer(lam, lam) + np.diag(1 - lam ** 2)
    assert np.allclose(triangulate(cov)["loadings"], lam)
    a, b = two_indicator_counterexample(0.8, 0.5, 1.5)
    assert np.allclose(a, b)


def test_dynamic_parameter_fields():
    assert check_dynamic_parameter({"operation": "o", "rival": "r", "prediction": "p", "boundary": "b"}) == ["criterion"]


def test_instrument_phenomenon_is_not_admitted_as_construct():
    c = Candidate("reader", "formative_index", True, {"unique_experimental_signature": True}, domain="instrument")
    out = admit(c)
    assert not out["admitted"] and out["status"] == "INSTRUMENT_PHENOMENON_NOT_A_CONSTRUCT"
