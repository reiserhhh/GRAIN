#!/usr/bin/env python3
"""Finite-law and operation checks independent of private data or model calls."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_joint_diagnostic as M


def test_positive_equal_class_counts():
    for c in (-1, 1):
        x = M.population_counts(c)
        assert np.all(x > 0)
        np.testing.assert_array_equal(x.sum(1), [100, 100])
        np.testing.assert_array_equal(x.sum(0), np.full(8, 25))


def test_all_lower_class_marginals_and_reference_laws_identical():
    lo, hi = (M.lower_marginals(c) for c in (-1, 1))
    assert set(lo) == set(hi)
    for indices in lo:
        np.testing.assert_allclose(lo[indices], hi[indices], atol=1e-15)
        np.testing.assert_allclose(lo[indices], 2 ** (-len(indices)), atol=1e-15)


@pytest.mark.parametrize("observed", [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)])
def test_every_lower_posterior_remains_prior(observed):
    for b, c in itertools.product(M.POINTS, (-1, 1)):
        assert M.posterior(b, c, observed) == 0.5


def test_full_posterior_is_joint_not_additive_likelihood():
    for b, c in itertools.product(M.POINTS, (-1, 1)):
        p = M.posterior(b, c)
        np.testing.assert_allclose(p, 0.5 + 0.3 * c * np.prod(b))
        np.testing.assert_allclose(np.log(p / (1 - p)), c * np.prod(b) * np.log(4))


def test_full_four_factor_character_and_orthogonality():
    matrix = M.character_matrix()
    np.testing.assert_array_equal(matrix.T @ matrix, np.eye(16) * 16)
    field = np.array([M.posterior(row[:3], row[3]) for row in M.FIELD_POINTS])
    beta = M.coefficients(field.reshape(2, 2, 2, 2))
    wanted = np.zeros(16)
    wanted[[0, 15]] = [0.5, 0.3]
    np.testing.assert_allclose(beta, wanted, atol=1e-15)
    np.testing.assert_allclose(M.reconstruct(beta).ravel(), field)


def test_lower_order_observer_has_zero_high_order_term():
    b = M.FIELD_POINTS
    field = 0.5 + 0.1 * b[:, 0] + 0.07 * b[:, 1] * b[:, 3]
    assert abs(M.coefficients(field.reshape(2, 2, 2, 2))[15]) < 1e-15


def test_reference_inversion_and_configuration_parity_contrast():
    field = np.array([M.posterior(row[:3], row[3]) for row in M.FIELD_POINTS])
    np.testing.assert_allclose(
        M.parity_contrasts(field.reshape(2, 2, 2, 2)), [-0.6, 0.6]
    )


def test_signed_effects_are_not_information_percentages():
    result = M.effect_comparison([-1, 1], [1, -1])
    assert result["cosine"] == -1
    assert result["signed_projection"] == -1
    assert result["relative_error"] == 2
    assert M.effect_comparison([0, 0], [0, 0])["cosine"] is None


@pytest.mark.parametrize("observed", [(0, 0), (3,), (-1,)])
def test_bad_observation_scope_fails(observed):
    with pytest.raises(ValueError):
        M.posterior([-1, 1, -1], 1, observed)


def test_invalid_reference_and_incomplete_field_fail():
    with pytest.raises(ValueError):
        M.population_counts(0)
    with pytest.raises(ValueError):
        M.coefficients(np.ones((2, 2, 2)))


def fake_rows():
    """Make a numeric-only native support fixture, never a fabricated model result."""
    rows = []
    for bank, b, c in itertools.product(range(2), list(M.POINTS), (-1, 1)):
        reference = np.where(c * np.prod(M.POINTS, axis=1) == 1, 10, 20).tolist()
        rows.append(
            {
                "id": len(rows),
                "bank": bank,
                "b": b.tolist(),
                "c": c,
                "action_text": str((bank, b.tolist())),
                "history": list(range(2, 13)),
                "current": [14],
                "reference": list(range(5, 13)),
                "reference_slot": list(range(5, 13)),
                "padding": [],
                "input_ids": [101, 102] + (b + 4).tolist() + reference + [103, 104],
            }
        )
    return rows


def test_complete_design_and_donor_event_identity():
    from scripts import run_grain_joint_diagnostic_hm145 as R

    rows = fake_rows()
    design = R.fixed_design(rows)
    assert len(design) == 132
    assert sum(d["mode"] == "capture" for d in design) == 32
    assert sum(d["mode"] == "noop" for d in design) == 4
    for entry in design[36:]:
        row, donor = (rows[entry[key]] for key in ("row", "donor"))
        assert row["bank"] == donor["bank"] and row["b"] == donor["b"]
        assert row["c"] == -donor["c"]


def test_incomplete_support_or_nonreference_edit_fails():
    from scripts import run_grain_joint_diagnostic_hm145 as R

    rows = fake_rows()
    with pytest.raises(ValueError):
        R.validate_rows(rows[:-1])
    rows[0]["input_ids"][2] = 909
    with pytest.raises(ValueError):
        R.validate_rows(rows)


def test_runtime_status_can_update_without_overwriting_primitives(tmp_path):
    from scripts import run_grain_joint_diagnostic_hm145 as R

    path = tmp_path / "status.json"
    R.status_write(path, {"attempts": 0})
    R.status_write(path, {"attempts": 1})
    assert R.H.load(path) == {"attempts": 1}
    frozen = tmp_path / "primitive.json"
    R.H.write(frozen, {"x": 1})
    with pytest.raises(FileExistsError):
        R.H.write(frozen, {"x": 2})
    assert R.H.load(frozen) == {"x": 1}
