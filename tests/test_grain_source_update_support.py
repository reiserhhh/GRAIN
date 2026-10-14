#!/usr/bin/env python3
"""Construction identities, support discipline and noninvariant context checks."""

import itertools

import numpy as np
import pytest

from grain_research.construction.math_source_update_support import (
    defect_terms,
    energy_account,
    mask_memory,
    operation_slots,
    source_increment,
    transported_memory,
)


def fixture():
    """Use a tiny source orbit and disjoint clause supports, not neural output."""
    rng = np.random.default_rng(162)
    cube = rng.normal(size=(2, 2, 1, 1, 8, 2))
    row = {
        "e": -1,
        "n": 1,
        "history": list(range(8)),
        "scopes": {
            "extraversion": [1, 2],
            "neuroticism": [4, 5],
        },
    }
    return cube, row


@pytest.mark.parametrize("axis", ["e", "n"])
def test_source_complete_exchange_and_inverse(axis):
    """Full increments equal source exchange; label-aware inverse closes exactly."""
    cube, _ = fixture()
    for e, n in itertools.product((-1, 1), repeat=2):
        i, j = (e + 1) // 2, (n + 1) // 2
        partner = cube[1 - i, j] if axis == "e" else cube[i, 1 - j]
        delta = source_increment(cube, e, n, axis, "full")
        np.testing.assert_allclose(cube[i, j] + delta, partner, atol=1e-14)
        inverse = source_increment(
            cube, -e if axis == "e" else e, -n if axis == "n" else n, axis, "full"
        )
        np.testing.assert_allclose(delta + inverse, 0, atol=1e-14)


def test_support_is_conditioned_on_axis_not_outcome():
    """Closure and other-clause supports are registered time addresses."""
    _, row = fixture()
    assert operation_slots(row, "e", "full_closure") == list(range(1, 8))
    assert operation_slots(row, "n", "relation_closure") == list(range(4, 8))
    assert operation_slots(row, "e", "relation_other") == [4, 5]
    with pytest.raises(ValueError):
        operation_slots(row, "e", "fitted_mask")


@pytest.mark.parametrize(
    "operation", ["relation_other", "full_other", "relation_closure", "full_closure"]
)
def test_exact_three_term_defect_and_cross_energy(operation):
    """Restore omissions without discarding a role-dependent context remainder."""
    cube, row = fixture()
    base = cube[0, 1] + 0.2
    partner = cube[1, 1] - 0.4
    terms = defect_terms(base, partner, cube, row, "e", operation)
    account = energy_account(terms)
    assert account["account_max_error"] < 1e-14
    energy = sum(
        v for k, v in account.items() if k.endswith("_energy") and k != "total_energy"
    )
    energy += sum(v for k, v in account.items() if "_x_" in k)
    assert energy == pytest.approx(account["total_energy"])
    assert np.max(abs(terms["context"] - 0.6)) < 1e-14


def test_native_cast_and_outside_support_unchanged():
    """A transported operation uses only the source increment on fixed rows."""
    cube, row = fixture()
    base = cube[0, 1].astype(np.float32)
    donor, slots = transported_memory(base, cube, row, "e", "full_other")
    outside = sorted(set(range(8)) - set(slots))
    assert donor.dtype == np.float32
    np.testing.assert_array_equal(base[:, :, outside], donor[:, :, outside])


def test_context_invariance_not_source_cycle_decides_natural_update():
    """Source inverse identity can hold while application exchange differs."""
    cube, row = fixture()
    cube[:, :, :, :, :1] = 0
    base = cube[0, 1] + 0.2
    target = cube[1, 1] + 0.2
    terms = defect_terms(base, target, cube, row, "e", "full_closure")
    np.testing.assert_allclose(terms["total"], 0, atol=1e-14)
    changed = defect_terms(base, target + 0.7, cube, row, "e", "full_closure")
    np.testing.assert_allclose(changed["total"], -0.7, atol=1e-14)


@pytest.mark.parametrize("slots", [[], [0, 0], [-1], [8]])
def test_invalid_support_is_not_silently_repaired(slots):
    """Missing support must not masquerade as an unchanged successful update."""
    with pytest.raises(ValueError):
        mask_memory(np.zeros((1, 1, 8, 2)), slots)
