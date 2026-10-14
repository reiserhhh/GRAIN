#!/usr/bin/env python3
"""Finite conditional-response covariance and personal-accounting tests."""

import numpy as np
import pytest

from grain_research.response_sharing import (
    apply_function,
    compile_function,
    learn_response,
    project_psd,
    source_neighbors,
)


def example():
    """A common response direction with opposite effects at two conditions."""
    v = np.array([-1, 1, 0, -0.5, 0.5])
    k = 0.02 * np.ones((5, 5)) + 0.2 * np.outer(v, v) + 0.01 * np.eye(5)
    return {
        "mean": np.zeros(5),
        "kernel": k,
        "noise": 0.03 * np.eye(5),
        "geometry_kernel": np.eye(5),
    }


def test_pairing_separates_person_and_measurement_variation():
    rng = np.random.default_rng(85)
    m = example()
    f = rng.multivariate_normal(np.zeros(5), m["kernel"], size=12000)
    y = f[:, :, None] + rng.normal(scale=np.sqrt(0.03), size=(len(f), 5, 2))
    learned = learn_response(y, np.arange(5))
    np.testing.assert_allclose(learned["kernel"], m["kernel"], atol=0.012)
    np.testing.assert_allclose(learned["noise"], m["noise"], atol=0.002)
    unpaired = learn_response(y, np.arange(5), permutation=rng.permutation(len(y)))
    assert np.trace(unpaired["kernel"]) < 0.05 * np.trace(learned["kernel"])


def test_signed_response_covariance_is_positive_semidefinite():
    k = example()["kernel"]
    assert k[0, 1] < 0
    assert np.linalg.eigvalsh(k).min() > 0
    p = np.eye(5) - np.ones((5, 5)) / 5
    np.testing.assert_allclose((p @ k @ p).sum(0), 0, atol=1e-12)


def test_function_and_accounts_equal_conditional_formula():
    m, obs = example(), [0, 2, 4]
    y = np.array([[0.1, 0.3, -0.2], [0.5, -0.2, 0.1]])
    p = compile_function(m, obs)
    k, r = m["kernel"], m["noise"]
    s = (k + r)[np.ix_(obs, obs)]
    expected = y @ np.linalg.inv(s) @ k[obs]
    scored = apply_function(p, y)
    np.testing.assert_allclose(scored["value"], expected, atol=1e-12)
    np.testing.assert_allclose(scored["personal_response"].mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(
        p["covariance"], k - k[:, obs] @ np.linalg.inv(s) @ k[obs], atol=1e-12
    )


def test_level_only_does_not_claim_zero_omitted_response_error():
    m = example()
    p = compile_function(m, [0, 2, 4], "level_only")
    r = apply_function(p, [[0.4, 0, -0.2]])
    np.testing.assert_allclose(r["personal_response"], 0, atol=1e-12)
    contrast = np.array([0, -1, 0, 1, 0])
    np.testing.assert_allclose(
        contrast @ p["covariance"] @ contrast, contrast @ m["kernel"] @ contrast
    )


def test_source_only_uncertainty_is_not_zero():
    m = example()
    p = compile_function(m, [2], "source_mean")
    np.testing.assert_allclose(p["weights"], 0)
    np.testing.assert_allclose(p["covariance"], m["kernel"])


def test_added_observation_cannot_increase_working_posterior_variance():
    m = example()
    first = compile_function(m, [0, 2, 4])
    second = compile_function(m, [0, 1, 2, 4])
    assert np.linalg.eigvalsh(first["covariance"] - second["covariance"]).min() > -1e-12


def test_same_level_is_not_same_response_and_reversed_levels_can_share():
    source = np.array([[-1, 0, 1], [3, 2, 1]])
    app = np.array([[1, 0, -1]])
    rng = np.random.default_rng(11)
    assert source_neighbors(source, app, "level", rng)[0] == 0
    assert source_neighbors(source, app, "response", rng)[0] == 1


def test_psd_and_input_errors():
    k, eig = project_psd([[1, 2], [2, 1]])
    assert eig.min() < 0
    assert np.linalg.eigvalsh(k).min() > -1e-12
    with pytest.raises(ValueError):
        compile_function(example(), [0, 0])
    with pytest.raises(ValueError):
        learn_response(np.zeros((2, 5, 2)), np.arange(5))
    with pytest.raises(ValueError):
        apply_function(compile_function(example(), [0, 2]), [[1]])


def test_centering_alone_creates_negative_off_diagonals():
    """Negative entries after mean removal do not prove opposite causal responses."""
    p = np.eye(5) - np.ones((5, 5)) / 5
    independent = 0.3 * np.eye(5)
    np.testing.assert_allclose(p @ independent @ p, 0.3 * p, atol=1e-12)
    assert (p @ independent @ p)[0, 1] == pytest.approx(-0.06)


def test_common_origin_change_preserves_personal_departure():
    m = example()
    y = np.array([[0.2, 0.1, -0.3]])
    first = apply_function(compile_function(m, [0, 2, 4]), y)
    moved = dict(m, mean=m["mean"] + 3)
    second = apply_function(compile_function(moved, [0, 2, 4]), y + 3)
    np.testing.assert_allclose(second["value"], first["value"] + 3, atol=1e-12)
    np.testing.assert_allclose(
        second["personal_response"], first["personal_response"], atol=1e-12
    )
