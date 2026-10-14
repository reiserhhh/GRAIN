#!/usr/bin/env python3
"""Check independent routing arithmetic, not success of the real intervention."""

import numpy as np
import pytest

from scripts.audit_grain_binding_router_hm128 import response


def test_personal_swap_unit_response_preserves_common_and_third():
    baseline = np.array([[0.8] * 6, [0.2] * 6, [0.5] * 6])
    natural = baseline[[1, 0, 2]]
    account = response(baseline, natural, natural)
    assert account["towards_natural_fraction"] == pytest.approx(1)
    assert account["common_norm"] == pytest.approx(0)
    assert account["untouched_norm"] == 0


def test_common_response_is_not_assignment_transfer():
    baseline = np.array([[0.8] * 6, [0.2] * 6, [0.5] * 6])
    changed = baseline + 0.1
    account = response(baseline, changed, baseline[[1, 0, 2]])
    assert account["towards_natural_fraction"] == pytest.approx(0, abs=1e-15)
    assert account["common_norm"] == pytest.approx(np.sqrt(6) * 0.1)
    assert account["untouched_norm"] == pytest.approx(np.sqrt(6) * 0.1)


def test_reference_without_assignment_change_stays_undefined():
    baseline = np.ones((3, 6))
    assert (
        response(baseline, baseline + 0.1, baseline)["towards_natural_fraction"] is None
    )
