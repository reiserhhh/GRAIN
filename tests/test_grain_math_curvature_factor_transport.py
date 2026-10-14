#!/usr/bin/env python3
"""Analytic witnesses for ordered factors, source pairing and finite resource rules."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_curvature_factor_transport as F


def bank():
    """Provide noncommuting, indefinite source and application factors."""
    rng = np.random.default_rng(4101)
    c = np.diag([1.0, -2.0, 0.5])
    items = [
        (c + rng.normal(size=(3, 3)), rng.normal(size=(3, 4)), rng.normal(size=(3, 5)))
        for _ in range(2)
    ]
    ref = F.source_reference(items, ["source", "source"])
    act = tuple(rng.normal(size=x.shape) for x in items[0])
    return ref, act, rng.normal(size=(2, 4))


def test_all_eight_ordered_terms():
    ref, act, d = bank()
    cube = F.crossed(ref, act, d)
    terms = F.mobius(cube)
    np.testing.assert_allclose(terms, F.direct_terms(ref, act, d), atol=1e-12)
    np.testing.assert_allclose(terms.sum(axis=0), cube[7], atol=1e-12)
    for mask in range(8):
        np.testing.assert_allclose(
            sum(terms[s] for s in range(8) if s & mask == s), cube[mask]
        )


def test_order_is_not_commutative():
    c, h, p = (
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([[1.0, 3.0], [0.0, 1.0]]),
        np.eye(2),
    )
    d = np.array([[2.0, 1.0]])
    np.testing.assert_allclose(F.contract(c, h, p, d).ravel(), p.T @ c @ h @ d.ravel())
    assert not np.allclose(F.contract(c, h, p, d).ravel(), p.T @ h @ c @ d.ravel())


def test_affine_score_has_zero_link():
    np.testing.assert_array_equal(
        F.contract(np.zeros((2, 2)), np.eye(2), np.eye(2), np.ones((2, 2))), 0
    )


def test_indefinite_link_is_retained():
    np.testing.assert_array_equal(
        F.contract(np.diag([1.0, -1.0]), np.eye(2), np.eye(2), np.eye(2)),
        np.diag([1.0, -1.0]),
    )


def test_common_logit_shift_gauge():
    n = 3
    c = np.eye(n) - np.ones((n, n)) / n
    h, p, d = np.eye(n), np.eye(n), np.array([[1.0, 2.0, 4.0]])
    x = F.contract(c, h, p, d)
    np.testing.assert_allclose(
        F.contract(c, h + np.ones((n, n)), p + 2 * np.ones((n, n)), d), x
    )


def test_score_and_input_units():
    _ref, (c, h, p), d = bank()
    x = F.contract(c, h, p, d)
    np.testing.assert_allclose(F.contract(c * 7, h / 3, p, d * 3), x * 7)
    np.testing.assert_allclose(
        F.contract(c, h, p, np.stack([d[0], d[0] * 9])), np.stack([x[0], x[0] * 9])
    )


def test_source_pairing_cannot_be_replaced_by_means():
    a = np.eye(2)
    ref = F.source_reference([(a, a, a), (a, -a, -a)], ["source", "source"])
    d = np.eye(2)
    np.testing.assert_array_equal(F.contract(ref["c"], ref["h"], ref["p"], d), 0)
    np.testing.assert_array_equal(F.paired_source(ref, d), a)


def test_source_only_and_distinct_query_layouts():
    ref, act, d = bank()
    with pytest.raises(ValueError, match="source wordings"):
        F.source_reference([act, act], ["source", "application"])
    with pytest.raises(ValueError, match="layouts"):
        F.crossed(ref, (act[0], act[1], act[2][:, :2]), d)


def test_signed_cross_moments_not_additive_variance():
    r = dict.fromkeys(F.FIELDS, 0.0)
    r.update(full=1.0, link=3.0, decoder=-2.0, m0=3.0)
    m = F.moment_summary([r])
    assert m["m0_ms"] == 9 and m["decoder_ms"] == 4
    assert m["m0__decoder_cross"] == -12 and m["full_ms"] == 1
    assert m["full_identity_error"] == 0


def test_alternate_inclusion_and_association():
    from scripts.verify_grain_math_curvature_factor_transport import inclusion, ordered

    ref, act, d = bank()
    cube = F.crossed(ref, act, d)
    np.testing.assert_allclose(inclusion(cube), F.mobius(cube), atol=1e-12)
    np.testing.assert_allclose(ordered(*act, d), cube[7], atol=1e-12)


def test_fail_closed_scope_and_no_application_reader():
    from scripts.run_grain_math_curvature_factor_transport import read_node

    spec = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/grain_math_curvature_factor_transport_v1.json"
        ).read_text()
    )
    F.assert_scope(spec)
    for k, v in (
        ("maximum_model_calls", 1),
        ("maximum_analysis_seconds", 301),
        ("source_bundles", 23),
    ):
        with pytest.raises(ValueError):
            F.assert_scope({**spec, k: v})
    with pytest.raises(ValueError, match="source-first"):
        read_node(Path("absent"), {"role": "application"}, "application")


def test_budget_charges_before_stop(tmp_path):
    from scripts.run_grain_math_curvature_factor_transport import Budget

    (tmp_path / "run_status.json").write_text(json.dumps({"analysis_seconds": 300.0}))
    b = Budget(tmp_path, started=0.0)
    with pytest.raises(TimeoutError):
        b.checkpoint()
    assert (
        json.loads((tmp_path / "run_status.json").read_text())["analysis_seconds"] > 300
    )


def test_relative_slots_record_absolute_offset():
    from scripts.run_grain_math_curvature_factor_transport import layout

    side = {
        "spans": {"A_property": {"positions": [90, 91, 92, 93]}},
        "early_query_positions": [120, 121],
    }
    p = {"left": side, "right": side}
    assert layout(p)["left"]["h_positions"] == [90, 91, 92, 93]
    bad = {**p, "right": {**side, "early_query_positions": [119, 120]}}
    with pytest.raises(ValueError, match="coordinates"):
        layout(bad)
