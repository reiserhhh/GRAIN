#!/usr/bin/env python3
"""Synthetic contracts for fixed-layer ownership exchange and source transport."""

import numpy as np
import pytest

from grain_research.construction import math_relational_carrier as M


def fake_rows():
    """Build unequal property lengths with identical high-first token addresses."""
    rows = []
    for role in ("source", "application"):
        for content in ("x", "y", "z"):
            for person in ("A", "B"):
                for b in (0, 1):
                    positions = [2, 3, 7, 8, 9] if b == 0 else [7, 8, 9, 2, 3]
                    rows.append(
                        {
                            "unit": len(rows),
                            "role": role,
                            "content": content,
                            "query": content,
                            "person": person,
                            "behaviour": b,
                            "order": b,
                            "appraisal": "none",
                            "property_positions": positions,
                            "property_a_count": 2 if b == 0 else 3,
                            "body_endpoint": 10,
                            "input_ids": [
                                1,
                                100 + b,
                                42,
                                43,
                                5,
                                6,
                                101 - b,
                                70,
                                71,
                                72,
                                10,
                                ord(person),
                            ],
                        }
                    )
    return rows


def fixtures():
    """Construct controlled state differences, never real data or a model."""
    rows = {r["unit"]: r for r in fake_rows()}
    captures = {}
    for r in rows.values():
        # Content amplitudes differ but averaging must give equal content weight.
        content = ("x", "y", "z").index(r["content"]) + 1
        values = np.asarray(
            [[p + r["behaviour"] * content, p / 2] for p in r["property_positions"]],
            dtype=np.float32,
        )
        captures[r["unit"]] = {
            "property": values,
            "endpoint": np.ones(2, np.float32) * r["unit"],
            "query_end": np.ones(2, np.float32) * (r["unit"] + 1),
        }
    pair_rows = M.pairs(list(rows.values()), ["x", "y", "z"])
    shared, components = M.source_vectors(
        pair_rows,
        rows,
        {k: v for k, v in captures.items() if rows[k]["role"] == "source"},
        ["x", "y", "z"],
    )
    return rows, captures, pair_rows, shared, components


def test_pair_scope_unequal_lengths_and_alignment():
    rows, _, pair_rows, _, _ = fixtures()
    assert len(pair_rows) == 24
    for p in pair_rows:
        proof = M.aligned_pair(rows[p["receiver"]], rows[p["donor"]])
        assert proof["high"] == [2, 3]
        assert proof["low"] == [7, 8, 9]


@pytest.mark.parametrize(
    "field", ["person", "input_ids", "property_positions", "order"]
)
def test_misalignment_never_resampled(field):
    rows = fake_rows()
    a, b = rows[0], dict(rows[1])
    b[field] = {
        "person": "B",
        "input_ids": [0] * 12,
        "property_positions": [8, 9, 10, 2, 3],
        "order": 0,
    }[field]
    with pytest.raises(ValueError):
        M.aligned_pair(a, b)


def test_source_equal_content_and_person_means():
    _, _, _, shared, components = fixtures()
    np.testing.assert_array_equal(components[:, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(shared, [[2, 0], [2, 0]])


def test_application_not_read_to_construct_source():
    rows, arrays, pairs, shared, _ = fixtures()
    for unit in arrays:
        if rows[unit]["role"] == "application":
            arrays[unit]["property"][:] = np.nan
    again, _ = M.source_vectors(pairs, rows, arrays, ["x", "y", "z"])
    np.testing.assert_array_equal(shared, again)
    with pytest.raises(ValueError):
        M.source_vectors(pairs[4:], rows, arrays, ["x", "y", "z"])


def test_random_norm_and_seed_and_zero():
    shared = np.array([[1, 2, 3], [3, 4, 5]], float)
    r = M.random_vectors(shared, 49)
    np.testing.assert_allclose(
        np.linalg.norm(r, axis=1), np.linalg.norm(shared, axis=1)
    )
    np.testing.assert_array_equal(r, M.random_vectors(shared, 49))
    np.testing.assert_array_equal(M.random_vectors(shared * 0, 49), shared * 0)


@pytest.mark.parametrize("arm", M.ARMS)
def test_exact_patch_outside_unchanged_and_direction(arm):
    rows, arrays, _, shared, _ = fixtures()
    random = M.random_vectors(shared, 49)
    for a, b in ((rows[0], rows[1]), (rows[1], rows[0])):
        p, values = M.patch_values(
            a, b, arrays[a["unit"]], arrays[b["unit"]], shared, random, arm
        )
        full = np.zeros((12, 2), np.float32)
        full[a["property_positions"]] = arrays[a["unit"]]["property"]
        full[a["body_endpoint"]] = arrays[a["unit"]]["endpoint"]
        full[-1] = arrays[a["unit"]]["query_end"]
        after = full.copy()
        after[p] = values
        np.testing.assert_array_equal(
            after[[i for i in range(12) if i not in p]],
            full[[i for i in range(12) if i not in p]],
        )
        if arm in ("source_mean", "random"):
            vector = shared if arm == "source_mean" else random
            np.testing.assert_allclose(
                after[2] - full[2], (1 - 2 * a["behaviour"]) * vector[0], atol=1e-6
            )
        if arm == "noop":
            np.testing.assert_array_equal(after, full)


def test_effects_constant_null_not_fake_ratio():
    assert M.effects(0.4, 0.4, 0.4) == {
        "full_change": 0.0,
        "patch_change": 0.0,
        "remaining": 0.0,
        "same_direction": False,
        "ratio": None,
    }
    assert M.effects(0.8, 0.2, 0.5)["ratio"] == pytest.approx(0.5)


def test_call_lifecycle_and_bounded_resources():
    state = {"status": "PREPARED", "in_flight": None, "attempted_forward_calls": 0}
    spec = {"max_forwards": 384, "max_model_seconds": 600}
    assert M.call_allowed(state, spec, 599)
    assert not M.call_allowed(state, spec, 600)
    assert not M.call_allowed({**state, "in_flight": 0}, spec, 1)
    assert not M.call_allowed({**state, "status": "COLLECTED"}, spec, 1)
    assert not M.call_allowed({**state, "attempted_forward_calls": 384}, spec, 1)
