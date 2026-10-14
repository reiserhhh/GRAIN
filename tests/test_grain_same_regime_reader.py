#!/usr/bin/env python3
"""Finite measurement identities; no data, model or private text required."""

import numpy as np
import pytest

from grain_research.construction.math_same_regime_reader import (
    cold,
    contacts,
    reader_parts,
    source_parts,
)


def test_singleton_mixture_reader_has_no_join_or_order():
    rng = np.random.default_rng(274)
    a, b = rng.normal(size=(2, 9, 5))
    x = np.stack(((a + b) / 2, (a + b) / 2, a, b), 1)
    parts = reader_parts(x)
    np.testing.assert_allclose(parts["join"], 0, atol=1e-15)
    np.testing.assert_allclose(parts["odd"], 0, atol=1e-15)


def test_order_and_join_are_separate_observable_channels():
    even, odd = np.ones((6, 5)) * 2, np.ones((6, 5)) * 0.3
    x = np.stack((even + odd, even - odd, np.ones((6, 5)), np.ones((6, 5))), 1)
    parts = reader_parts(x)
    np.testing.assert_allclose(parts["odd"], odd)
    np.testing.assert_allclose(parts["join"], 1)


def test_complete_contact_identity_keeps_singleton_support():
    rng = np.random.default_rng(8)
    source, reader = rng.normal(size=(13, 2, 5)), rng.normal(size=(13, 4, 5))
    c = contacts(source, reader)
    np.testing.assert_allclose(
        c["raw"], c["join"] + c["order"] + c["single"], atol=1e-15
    )


def test_exchanging_both_observers_preserves_contact():
    rng = np.random.default_rng(5)
    source, reader = rng.normal(size=(13, 2, 5)), rng.normal(size=(13, 4, 5))
    a, b = contacts(source, reader), contacts(source[:, ::-1], reader[:, [1, 0, 3, 2]])
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-15)


def test_exchange_only_reader_changes_order_not_join():
    rng = np.random.default_rng(2)
    source, reader = rng.normal(size=(4, 2, 5)), rng.normal(size=(4, 4, 5))
    a, b = contacts(source, reader), contacts(source, reader[:, [1, 0, 3, 2]])
    np.testing.assert_allclose(a["join"], b["join"])
    np.testing.assert_allclose(a["order"], -b["order"])


@pytest.mark.parametrize("value", [np.zeros((3, 3, 5)), np.full((3, 4, 5), np.nan)])
def test_missing_context_is_not_zero(value):
    with pytest.raises(ValueError):
        reader_parts(value)


def test_source_requires_the_actual_display_order_pair():
    with pytest.raises(ValueError):
        source_parts(np.zeros((2, 5)))


def test_no_empty_matching_population():
    with pytest.raises(ValueError):
        contacts(np.zeros((0, 2, 5)), np.zeros((0, 4, 5)))


def test_complete_cold_packet_needs_no_model():
    bank = [
        {"item": j, "target": j // 10, "pole": 1 if j % 10 < 5 else -1}
        for j in range(50)
    ]
    packet = {
        "version": "HM274-v1",
        "bank": bank,
        "chosen": np.full((48, 4, 50), -6).tolist(),
        "partition": np.zeros((48, 4, 50)).tolist(),
        "lengths": [3] * 50,
    }
    output = cold(packet)
    for scale in ("sum", "per_token"):
        np.testing.assert_allclose(
            output["accounts"][scale]["balanced_key"]["parts"]["join"], 0
        )
    packet["chosen"][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        cold(packet)


def test_incomplete_behavior_bank_is_not_accepted():
    packet = {
        "version": "HM274-v1",
        "bank": [],
        "chosen": np.full((48, 4, 49), -6).tolist(),
        "partition": np.zeros((48, 4, 49)).tolist(),
        "lengths": [3] * 49,
    }
    with pytest.raises(ValueError):
        cold(packet)
