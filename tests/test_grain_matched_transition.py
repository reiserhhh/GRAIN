#!/usr/bin/env python3
"""Synthetic operation/support tests, not personality validity tests."""

import numpy as np
import pytest

from grain_research.construction.math_matched_transition import (
    action_components,
    support_account,
    transition_address,
)
from grain_research.construction.math_response_function import response_components


def messages(body="fixed article"):
    """Return a small explicit four-message observation protocol."""
    return [
        {"role": "system", "content": "neutral style"},
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": body},
        {"role": "user", "content": "A: do; B: defer"},
    ]


def test_same_destination_does_not_define_same_transition():
    assert transition_address(messages(), "email", "diary") != transition_address(
        messages(), "review", "diary"
    )


def test_new_evidence_body_does_not_change_declared_operation():
    assert transition_address(messages("one"), "email", "diary") == (
        transition_address(messages("two"), "email", "diary")
    )


def test_system_and_order_remain_part_of_observation_operation():
    before = messages()
    after = messages()
    after[3]["content"] = "A: defer; B: do"
    assert transition_address(before, "email", "diary") != transition_address(
        after, "email", "diary"
    )
    after = messages()
    after[0]["content"] = "different style"
    assert transition_address(before, "email", "diary") != transition_address(
        after, "email", "diary"
    )


def test_support_does_not_impute_absent_operations():
    account = support_account(["a", "b"], ["a", "c", "c"])
    assert account["shared_unique"] == 1
    assert account["supported_application_units"] == 1
    assert account["unsupported_application_units"] == 2


def test_action_covariance_is_difference_of_endpoint_covariances():
    rng = np.random.default_rng(168)
    x, start, end = rng.normal(size=(3, 12, 4))
    groups = np.repeat([0, 1, 2], 4)
    origin, full, within, between = action_components(x, start, end, groups)
    np.testing.assert_allclose(full, within + between, atol=1e-15)
    first, last = (response_components(x, y, groups) for y in (start, end))
    np.testing.assert_array_equal(origin, first[0])
    for i, actual in enumerate((full, within, between), 1):
        np.testing.assert_allclose(actual, last[i] - first[i], atol=1e-15)


def test_constant_log_odds_action_has_zero_total_witness():
    x = np.arange(24.0).reshape(6, 4)
    start = x[:, :2]
    _, full, within, between = action_components(
        x, start, start + 3, np.repeat([0, 1], 3)
    )
    np.testing.assert_array_equal(full, 0)
    np.testing.assert_array_equal(within, 0)
    np.testing.assert_array_equal(between, 0)


def test_missing_endpoints_and_wrong_roles_are_rejected():
    with pytest.raises(ValueError):
        action_components(np.ones((4, 2)), np.ones((4, 2)), np.ones((3, 2)), [0] * 4)
    bad = messages()
    bad[2]["role"] = "user"
    with pytest.raises(ValueError):
        transition_address(bad, "email", "diary")
