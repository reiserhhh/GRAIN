#!/usr/bin/env python3
"""Partition/assembly tests in finite synthetic worlds; no neural or private data."""

import numpy as np
import pytest

from grain_research.construction import math_capsule_assembly as A


def test_partition_preserves_original_slot_content_and_contract():
    text = "intro\n" + "\n\n".join(f"Excerpt {i}:\ntext{i}" for i in range(1, 5))
    text += "\nEvidence ends here.\ncriterion\nAnswer:"
    left, parts = A.partition_prompt(text, 0)
    right, again = A.partition_prompt(text, 1)
    assert parts == again
    assert "text1" in left and "text2" in left and "text3" not in left
    assert "text3" in right and "text4" in right and "text1" not in right
    assert "Excerpt 3:" in right and "Excerpt 1:" not in right
    assert left.endswith("Evidence ends here.\ncriterion\nAnswer:")
    assert right.endswith("Evidence ends here.\ncriterion\nAnswer:")


def test_same_event_multiple_carriers_are_not_independent_events():
    blocks = A.event_partition(["e1", "e2", "e3", "e4"])
    assert blocks == [["e1", "e2"], ["e3", "e4"]]
    with pytest.raises(ValueError):
        A.event_partition(["e1", "e1", "e3", "e4"])


def test_linear_compiler_linear_reader_is_null():
    a, b = np.array([0.3, -0.7]), np.array([0.2, 0.4])
    summed, coupling = A.assemble(np.stack([a, b]), a + b)
    np.testing.assert_array_equal(coupling, np.zeros(2))
    accounts = A.action_decomposition(a + b, a, b, summed)
    for x in accounts.values():
        np.testing.assert_allclose(x, 0, atol=1e-16)


def test_nonlinear_reader_does_not_fabricate_compiler_coupling():
    a, b = np.array([0.3]), np.array([0.2])
    summed, coupling = A.assemble(np.stack([a, b]), a + b)
    accounts = A.action_decomposition((a + b) ** 3, a**3, b**3, summed**3)
    np.testing.assert_array_equal(coupling, [0])
    np.testing.assert_array_equal(accounts["compiler"], [0])
    assert accounts["reader"][0] > 0


def test_compiler_and_reader_effects_can_cancel():
    a, b = np.array([0.3]), np.array([0.2])
    accounts = A.action_decomposition([0.2], a, b, [0.8])
    assert accounts["compiler"][0] < 0 < accounts["reader"][0]
    np.testing.assert_allclose(
        accounts["total"], accounts["compiler"] + accounts["reader"]
    )


def test_unobserved_or_ambiguous_inputs_rejected():
    with pytest.raises(ValueError):
        A.partition_prompt("Excerpt 1:\nonly one", 0)
    with pytest.raises(ValueError):
        A.assemble(np.zeros((3, 2)), np.zeros(2))
    with pytest.raises(ValueError):
        A.action_decomposition([np.nan], [0], [0], [0])
