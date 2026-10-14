#!/usr/bin/env python3
"""Companion interface checks; do not modify the sealed HM156 scientific files."""

import itertools

import numpy as np
import pytest

from scripts import run_grain_target_carrier_closure_hm156 as R


def toy_memories():
    """Supply a causal two-event memory with an explicitly mixed constant tail."""
    rows, natural = [], {}
    for e, n in itertools.product((-1, 1), repeat=2):
        row = {
            "id": len(rows),
            "bank": 0,
            "condition": 0,
            "order": 0,
            "e": e,
            "n": n,
            "history": list(range(8)),
            "scopes": {"extraversion": [1, 2], "neuroticism": [3, 4, 5]},
        }
        key = np.array(
            [
                0,
                e,
                e,
                e + n,
                e + n,
                e + n,
                e + 2 * n + 3 * e * n,
                e + 2 * n + 3 * e * n,
            ],
            dtype=np.float32,
        ).reshape(1, 1, 8, 1)
        natural[row["id"]] = {"history_key": key, "history_value": 2 * key}
        rows.append(row)
    indexed = {(r["bank"], r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    return rows, natural, indexed


def test_upstream_other_is_null_but_later_other_changes():
    rows, natural, indexed = toy_memories()
    n, slots, _ = R.donor_arguments(rows[0], "n_other", natural, indexed)
    assert slots == [1, 2]
    np.testing.assert_array_equal(n[0], natural[0]["history_key"])
    e, slots, _ = R.donor_arguments(rows[0], "e_other", natural, indexed)
    assert slots == [3, 4, 5]
    assert np.max(abs(e[0] - natural[0]["history_key"])) == 2


def test_closure_matches_natural_single_flip():
    rows, natural, indexed = toy_memories()
    for mode, donor in (("e_closure", 2), ("n_closure", 1)):
        actual, _, _ = R.donor_arguments(rows[0], mode, natural, indexed)
        for a, name in zip(actual, ("history_key", "history_value"), strict=True):
            np.testing.assert_array_equal(a, natural[donor][name])


def test_real_and_additive_tail_are_distinct_operations():
    rows, natural, indexed = toy_memories()
    real, slots, _ = R.donor_arguments(rows[0], "both_tail_real", natural, indexed)
    additive, _, rounding = R.donor_arguments(
        rows[0], "both_tail_additive", natural, indexed
    )
    assert slots == [6, 7]
    assert rounding == 0
    np.testing.assert_array_equal(
        real[0][:, :, slots] - additive[0][:, :, slots], np.full((1, 1, 2, 1), 12)
    )
    np.testing.assert_array_equal(
        real[0][:, :, :6], natural[0]["history_key"][:, :, :6]
    )


def test_unknown_operation_is_not_silently_double_flip():
    rows, natural, indexed = toy_memories()
    with pytest.raises(ValueError):
        R.donor_arguments(rows[0], "unknown", natural, indexed)
