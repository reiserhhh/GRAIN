#!/usr/bin/env python3
"""Exact quotient witnesses and source-only physical compensation checks."""

import numpy as np
import pytest

from grain_research.construction.math_cancellation_transport import (
    assert_scope,
    compensation,
    compensation_basis,
    complement,
    parts,
    random_complement,
    signed_summary,
    transport_sum,
)


def test_equal_transport_preserves_invisible_pair():
    u = np.array([2.0, -3.0])
    g = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(transport_sum(u, -u, g, g), np.zeros(2))


def test_unequal_transport_reveals_source_invisible_pair():
    u = np.array([2.0, -3.0])
    np.testing.assert_array_equal(u - u, np.zeros(2))
    revealed = transport_sum(u, -u, np.eye(2) * 2, np.eye(2))
    np.testing.assert_array_equal(revealed, u)
    assert np.linalg.norm(revealed) > 0


def test_nonlinear_fiber_map_need_not_descend():
    def mapping(l, d):
        return l * l + d

    assert 2 + (-2) == 0 + 0
    assert mapping(2, -2) != mapping(0, 0)


def test_compensation_common_and_antisymmetric():
    u = np.eye(4)[:, :1]
    l, d = np.arange(8).reshape(4, 2), np.arange(8, 16).reshape(4, 2)
    c = compensation(l, d, u)
    np.testing.assert_allclose(c["c"] + c["e"], c["r_link"])
    np.testing.assert_allclose(-c["c"] + c["e"], c["r_decoder"])
    np.testing.assert_allclose(2 * c["e"], complement(l + d, u))


def test_source_basis_rank_gauge_and_units():
    rng = np.random.default_rng(40)
    u = np.eye(12)[:, :3]
    x = rng.normal(size=(12, 5))
    r = np.zeros_like(x)
    a = compensation_basis(x, ["source"] * 5, u, r)
    b = compensation_basis(x * 7, ["source"] * 5, u, r)
    np.testing.assert_allclose(
        a["basis"] @ a["basis"].T, b["basis"] @ b["basis"].T, atol=1e-12
    )
    np.testing.assert_allclose(a["basis"].T @ u, 0, atol=1e-12)
    q = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    np.testing.assert_allclose(
        a["basis"] @ a["basis"].T, (a["basis"] @ q) @ (a["basis"] @ q).T, atol=1e-12
    )


def test_application_and_nonorthogonal_rejected():
    u = np.eye(6)[:, :1]
    with pytest.raises(ValueError, match="application"):
        compensation_basis(np.eye(6), ["application"] * 6, u, np.zeros((6, 6)))
    with pytest.raises(ValueError, match="orthonormal"):
        complement(np.eye(6), u * 2)


def test_zero_and_roundoff_resolution_labels_do_not_tune_rank():
    u = np.eye(8)[:, :2]
    x = np.zeros((8, 3))
    zero = compensation_basis(x, ["source"] * 3, u, x)
    assert zero["rank"] == 0
    x[2:5] = np.eye(3) * 1e-10
    noisy = compensation_basis(x, ["source"] * 3, u, np.ones_like(x) * 1e-9)
    assert noisy["rank"] == 3 and not noisy["resolved_against_proxy"].any()
    assert noisy["proxy_is_certified_component_error_bound"] is False


def test_random_control_dimensions_seed_and_zero():
    u = np.eye(30)[:, :4]
    a = random_complement(u, 5, 4007)
    np.testing.assert_array_equal(a, random_complement(u, 5, 4007))
    assert not np.array_equal(a, random_complement(u, 5, 4008))
    np.testing.assert_allclose(u.T @ a, 0, atol=1e-12)
    assert random_complement(u, 0, 1).shape == (30, 0)
    with pytest.raises(ValueError, match="rank"):
        random_complement(u, 27, 1)


def test_physical_parts_signed_cancellation_and_component_units():
    u, w, wr = np.eye(4)[:, :1], np.eye(4)[:, 1:2], np.eye(4)[:, 2:3]
    l = np.array([[1.0, 3.0, 2.0, 0.0], [2.0, 6.0, 0.0, 1.0]])
    d = np.array([[0.0, -2.0, -2.0, 0.0], [0.0, -5.0, 0.0, -1.0]])
    r = np.zeros_like(l)
    p = parts(l + d, l, d, r, np.array([1.0, 2.0, 3.0, 4.0]), u, w, wr)
    np.testing.assert_allclose(p["full"], p["u"] + p["w"] + p["remaining"])
    np.testing.assert_allclose(
        p["missing_u"], 2 * p["e_action"] + p["roundoff_complement"]
    )
    scaled = parts(
        (l + d) * 7, l * 7, d * 7, r, np.array([1.0, 2.0, 3.0, 4.0]) / 7, u, w, wr
    )
    np.testing.assert_allclose(scaled["full"], p["full"])
    with pytest.raises(ValueError, match="same dimension"):
        parts(l + d, l, d, r, np.ones(4), u, w, np.empty((4, 0)))


def test_signed_summary_and_unresolved_denominator():
    row = dict.fromkeys(
        (
            "full",
            "u",
            "w",
            "remaining",
            "missing_u",
            "random_w",
            "random_remaining",
            "link_complement",
            "decoder_complement",
            "roundoff_complement",
            "c_action",
            "e_action",
        ),
        0.0,
    )
    row.update(
        full=1.0, u=0.0, w=3.0, remaining=-2.0, missing_u=1.0, random_remaining=1.0
    )
    s = signed_summary([row])
    assert s["missing_ms_reduction"] == -3 and s["w_remaining_cross"] == -12
    row.update(full=1e-12, u=0, w=3e-12, remaining=-2e-12, missing_u=1e-12)
    assert signed_summary([row])["remaining_over_full_rms"] is None


def test_source_first_scope_and_budget():
    spec = {"maximum_analysis_seconds": 300, "maximum_model_calls": 0}
    assert_scope(spec, source_nodes=48)
    with pytest.raises(ValueError, match="barrier"):
        assert_scope(spec, application=True, barrier=False)
    with pytest.raises(ValueError, match="48"):
        assert_scope(spec, source_nodes=47)
    with pytest.raises(ValueError, match="budget"):
        assert_scope({**spec, "maximum_model_calls": 1})


def test_reader_refuses_application_before_opening_files(tmp_path):
    from scripts.run_grain_math_cancellation_transport import arrays

    with pytest.raises(ValueError, match="source-first"):
        arrays(tmp_path, {"role": "application"}, "application", False)
    with pytest.raises(ValueError, match="source-first"):
        arrays(tmp_path, {"role": "application"}, "source", False)


def test_budget_charges_failure_time_without_reset(tmp_path, monkeypatch):
    from scripts import run_grain_math_cancellation_transport as runner

    runner.write_json(tmp_path / "run_status.json", {"analysis_seconds": 299.0})
    monkeypatch.setattr(runner.time, "monotonic", lambda: 20.0)
    budget = runner.Budget(tmp_path, started=18.0)
    with pytest.raises(TimeoutError, match="300"):
        budget.checkpoint()
    assert runner.read_json(tmp_path / "run_status.json")["analysis_seconds"] == 301.0
