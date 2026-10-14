"""Independent Gaussian, fallback and source-budget checks for restored F04."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f04 import CONDITIONS, generate_world, grouped
from grain_research.gptpro_foundation_v3.f04_recovery import (
    fit_source_halves,
    fit_source_lmm,
    fit_source_moments,
    generate_response_world,
    observed_arrays,
    score_response,
    score_source_halves,
)


def data():
    return generate_world(
        "L", 20260915, counts={"source": 40, "validation": 0, "application": 2}
    )[0]


def test_blup_matches_precision_form_and_zero_prior_variance():
    rows = data()
    prefix = next(iter(grouped([r for r in rows if r["role"] == "application"]).values()))[:24]
    fit = {
        "mode": "linear",
        "mu": [0.1, 0.4],
        "Sigma": [[1, 0.1], [0.1, 0.25]],
        "sigma2": 0.25,
    }
    result = score_response(prefix, fit)
    x, y = observed_arrays(prefix)
    precision = np.linalg.inv(fit["Sigma"])
    expected = np.linalg.solve(
        precision + x.T @ x / 0.25, precision @ fit["mu"] + x.T @ y / 0.25
    )
    np.testing.assert_allclose(result["coefficients"], expected, atol=1e-10)
    np.testing.assert_allclose(
        result["source_term"] + result["observed_term"], result["means"], atol=1e-12
    )
    zero = score_response(prefix, {**fit, "Sigma": np.zeros((2, 2)).tolist()})
    np.testing.assert_array_equal(zero["coefficients"], fit["mu"])


def test_four_cell_sampling_noise_is_subtracted():
    source = [r for r in data() if r["role"] == "source"]
    fit = fit_source_moments(source, "four_cell")
    np.testing.assert_allclose(
        fit["sampling_covariance"], np.eye(4) * fit["sigma2"] / 8, atol=1e-12
    )
    assert fit["noise_df"] == 40 * (32 - 4)
    with pytest.raises(ValueError, match="source-only"):
        fit_source_moments(data())


def test_reml_is_really_fitted_and_source_only():
    rows = data()
    source = [r for r in rows if r["role"] == "source"]
    fit = fit_source_lmm(source)
    assert not fit["fallback"]
    assert fit["attempts"][0]["converged"]
    assert abs(fit["mu"][1] - 0.5) < 0.25
    assert 0.1 < fit["sigma2"] < 0.5
    with pytest.raises(ValueError, match="source"):
        fit_source_lmm(rows)


def test_reml_failure_is_explicit_and_only_retries_once(monkeypatch):
    calls = []

    def fail(self, **kwargs):
        calls.append(kwargs["method"])
        raise np.linalg.LinAlgError("controlled test failure")

    monkeypatch.setattr(
        "grain_research.gptpro_foundation_v3.f04_recovery.MixedLM.fit", fail
    )
    result = fit_source_lmm([r for r in data() if r["role"] == "source"])
    assert calls == ["lbfgs", "powell"]
    assert result["fallback"] and "failure" in result["method"]


def test_old_halves_preserve_actual_noise_budget():
    rows = data()
    source = [r for r in rows if r["role"] == "source"]
    prefix = next(iter(grouped([r for r in rows if r["role"] == "application"]).values()))[:24]
    fit = fit_source_halves(source)
    assert fit["half_events"] == 4
    result = score_source_halves(prefix, fit)
    np.testing.assert_allclose(result["source_noise_scale"], np.sqrt(4 / 6))
    assert result["means"].shape == (4,)


def test_rank_one_is_not_claimed_to_identify_slope_or_duplicate_evidence():
    rows = data()
    prefix = [
        r for r in rows if r["role"] == "application" and r["z"] == CONDITIONS[0]
    ][:8]
    fit = {
        "mode": "linear",
        "mu": [0, 0.5],
        "Sigma": np.eye(2).tolist(),
        "sigma2": 0.25,
    }
    result = score_response(prefix, fit)
    assert result["data_rank"] == 1
    assert not result["data_identifies_all_coefficients"]
    with pytest.raises(ValueError, match="duplicate"):
        score_response(prefix + [prefix[0]], fit)


def test_corrected_worlds_pair_intercepts_and_fix_constant_slope():
    counts = {"source": 6, "validation": 0, "application": 2}
    _, truth = generate_response_world("S", 20260915, counts=counts)
    for index in range(0, len(truth), 2):
        assert truth[index]["a"] == truth[index + 1]["a"]
        assert (truth[index]["b"], truth[index + 1]["b"]) == (-0.5, 0.5)
    _, constant = generate_response_world("0", 20260915, counts=counts)
    assert all(t["b"] == 0.5 for t in constant)


def test_noise_law_comparison_and_nested_designs_share_parameters():
    counts = {"source": 4, "validation": 0, "application": 2}
    first, truth = generate_response_world("L", 20260915, counts=counts, events=8)
    longer, other = generate_response_world("L", 20260915, counts=counts, events=32)
    assert first == [r for r in longer if r["t"] < 8]
    assert truth == other
    _, heavy = generate_response_world("H", 20260915, counts=counts)
    assert [t["a"] for t in truth] == [t["a"] for t in heavy]
    assert [t["sigma2"] for t in truth] == [t["sigma2"] for t in heavy]
