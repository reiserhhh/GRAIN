"""Independent finite-channel checks, including a genuine infeasible order."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f15 import (
    garbling_certificate,
    mutual_information,
)


def test_identity_to_fair_has_actual_garbling():
    a = np.eye(2)
    b = np.full((2, 2), 0.5)
    result = garbling_certificate(a, b)
    assert result["status"] == "FEASIBLE"
    kernel = np.asarray(result["kernel"])
    np.testing.assert_allclose(a @ kernel, b, atol=1e-12)
    np.testing.assert_allclose(kernel.sum(axis=1), 1, atol=1e-12)
    # The old doubly-stochastic LP could incorrectly return this permutation.
    assert not np.allclose(kernel, [[0, 1], [1, 0]])


def test_fair_to_identity_has_independently_checkable_farkas_witness():
    a = np.full((2, 2), 0.5)
    b = np.eye(2)
    result = garbling_certificate(a, b)
    assert result["status"] == "INFEASIBLE_CERTIFIED"
    y = np.asarray(result["farkas_y"])
    rows = []
    rhs = []
    for state in range(2):
        for out in range(2):
            row = np.zeros((2, 2))
            row[:, out] = a[state]
            rows.append(row.ravel())
            rhs.append(b[state, out])
    for obs in range(2):
        row = np.zeros((2, 2))
        row[obs] = 1
        rows.append(row.ravel())
        rhs.append(1)
    assert np.min(np.array(rows).T @ y) >= -1e-10
    assert np.asarray(rhs) @ y < -1e-10


def test_information_and_data_processing():
    assert mutual_information(np.eye(2), [0.5, 0.5]) == pytest.approx(np.log(2))
    assert mutual_information(np.full((2, 2), 0.5), [0.5, 0.5]) == pytest.approx(0)
    a = np.array([[0.8, 0.1, 0.1], [0.1, 0.3, 0.6]])
    kernel = np.array([[0.6, 0.4], [0.2, 0.8], [0.7, 0.3]])
    b = a @ kernel
    result = garbling_certificate(a, b)
    assert result["status"] == "FEASIBLE"
    np.testing.assert_allclose(a @ np.asarray(result["kernel"]), b, atol=1e-10)
    assert mutual_information(a, [0.3, 0.7]) >= mutual_information(b, [0.3, 0.7])


@pytest.mark.parametrize("bad", [[[0.2, 0.2]], [[-1, 2]], [[float("nan"), 1]]])
def test_invalid_channel_rejected(bad):
    with pytest.raises(ValueError):
        garbling_certificate(bad, [[0.5, 0.5]])
