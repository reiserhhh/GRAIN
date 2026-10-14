#!/usr/bin/env python3
"""Reference anchors, affine-function averages and invariant evidence differences."""

import numpy as np
import pytest

from grain_research.affine_reference import (
    combine_affine_references,
    score_affine_reference,
)


def charts():
    """Synthetic two-block affine charts without human labels."""
    rng = np.random.default_rng(61)
    return [
        {
            "Z": rng.normal(size=(5, 3)),
            "h": rng.normal(size=3),
            "G": rng.normal(size=(5, 3)),
            "g": rng.normal(size=3),
            "b": np.asarray(rng.normal()),
        }
        for _ in range(4)
    ]


def test_chart_mean_matches_whole_function_average():
    source = charts()
    ref = combine_affine_references(source, [1, 2, 3, 4])
    p, q = np.ones((5, 3)), np.arange(3)
    direct = np.dot(
        ref["weights"],
        [
            r["b"] + np.sum(r["G"] * (p - r["Z"])) + r["g"] @ (q - r["h"])
            for r in source
        ],
    )
    np.testing.assert_allclose(score_affine_reference(ref, p, q, "chart_mean"), direct)
    np.testing.assert_allclose(
        score_affine_reference(ref, p, q) - direct, ref["mean_anchor_minus_chart_mean"]
    )


def test_reference_mean_anchor_is_empirical_mean_anchor():
    source = charts()
    ref = combine_affine_references(source, [1, 2, 3, 4])
    scored = [score_affine_reference(ref, r["Z"], r["h"]) for r in source]
    np.testing.assert_allclose(ref["weights"] @ scored, ref["b_mean_anchor"])


def test_endpoint_response_identical_under_anchor_change():
    ref = combine_affine_references(charts())
    p, q = np.ones((5, 3)), np.ones(3)
    responses = [
        score_affine_reference(ref, 2 * p, -q, mode)
        - score_affine_reference(ref, p, q, mode)
        for mode in ("mean_anchor", "chart_mean")
    ]
    np.testing.assert_allclose(responses[0], responses[1])


def test_translation_of_states_preserves_both_conventions():
    source = charts()
    changed = [{**r, "Z": r["Z"] + 3.5, "h": r["h"] - 1.5} for r in source]
    left, right = combine_affine_references(source), combine_affine_references(changed)
    p, q = np.ones((5, 3)), np.ones(3)
    for mode in ("mean_anchor", "chart_mean"):
        np.testing.assert_allclose(
            score_affine_reference(left, p, q, mode),
            score_affine_reference(right, p + 3.5, q - 1.5, mode),
        )


def test_single_reference_and_constant_operators_coincide():
    source = charts()
    assert (
        abs(combine_affine_references(source[:1])["mean_anchor_minus_chart_mean"])
        < 1e-14
    )
    for r in source:
        r["G"], r["g"] = source[0]["G"], source[0]["g"]
    assert (
        abs(combine_affine_references(source)["mean_anchor_minus_chart_mean"]) < 1e-14
    )


@pytest.mark.parametrize(
    "weights", [[0, 0, 0, 0], [1, -1, 1, 1], [1, 2], [1, np.nan, 1, 1]]
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        combine_affine_references(charts(), weights)


def test_wrong_convention_or_shape_rejected():
    ref = combine_affine_references(charts())
    with pytest.raises(ValueError):
        score_affine_reference(ref, np.zeros((3, 5)), np.zeros(3))
    with pytest.raises(ValueError):
        score_affine_reference(ref, np.zeros((5, 3)), np.zeros(3), "guess")
