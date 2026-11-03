#!/usr/bin/env python3
"""Focused finite-law tests with explicit nonpsychological synthetic outputs."""

import numpy as np
import pytest

from grain_research.construction import math_conditional_publication as C


def fixture():
    """Supply all addresses and labels without a guaranteed generalization result."""
    rng = np.random.default_rng(292)
    states = np.tile(np.arange(8), (32, 2, 2))
    labels = (np.arange(32)[:, None] >> np.arange(5)) & 1
    outputs = rng.normal(size=(*states.shape, 3)) + labels[:, None, None, :3]
    return states, outputs, np.array([labels, labels[::-1]])


def test_field_reconstruction_and_reference_zero():
    """Source centering defines units but does not remove any component."""
    a, b, _ = fixture()
    frame = C.reference(a, b)
    fields = C.fields(frame, a, b)
    np.testing.assert_allclose(
        fields["joint"], fields["opportunity"] + fields["response"], atol=1e-14
    )
    np.testing.assert_allclose(fields["joint"].mean((0, 1)), 0, atol=1e-14)


def test_missing_author_support_not_invented_conditional_mean():
    """Zero occupied mass does not certify a person's unobserved response as zero."""
    a, b, _ = fixture()
    frame = C.reference(a, b)
    f = C.fields(frame, np.zeros((1, 1, 2), int), np.ones((1, 1, 2, 3)))
    assert f["support"].sum() == 1
    np.testing.assert_array_equal(f["response"].reshape(8, 3)[1:], 0)
    assert "conditional_author_mean" not in f


def test_criterion_residual_group_means_zero():
    """Residual labels qualify within-profile variation without deleting shared parts."""
    _, _, ys = fixture()
    e = C.criterion_residuals(ys[0])
    for q in range(5):
        code = np.delete(ys[0], q, 1) @ (1 << np.arange(4))
        for g in np.unique(code):
            assert abs(e[code == g, q].mean()) < 1e-14
    np.testing.assert_array_equal(C.criterion_residuals(np.ones((3, 5))), 0)


def test_same_covector_score_and_covariance_add():
    """Score/covariance closure does not imply additive correlations."""
    a, b, ys = fixture()
    rule = C.compile_rule(a, b, ys)
    scores = C.score(rule, C.fields(rule, a, b))
    np.testing.assert_allclose(
        scores["joint"], scores["opportunity"] + scores["response"], atol=1e-13
    )
    cov = lambda x: ((x - x.mean(0)) * (ys[0] - ys[0].mean(0))).mean(0)
    np.testing.assert_allclose(
        cov(scores["joint"][:, 0, 0]),
        cov(scores["opportunity"][:, 0, 0]) + cov(scores["response"][:, 0, 0]),
        atol=1e-13,
    )


def test_batch_composition_does_not_change_score():
    """Frozen-reference scores do not depend on who else is measured."""
    a, b, ys = fixture()
    rule = C.compile_rule(a, b, ys)
    all_scores = C.score(rule, C.fields(rule, a, b))["joint"]
    one = C.score(rule, C.fields(rule, a[:1], b[:1]))["joint"]
    np.testing.assert_allclose(one[0], all_scores[0])


def test_cold_rejects_labels_and_bad_binding():
    """Publications must be an observed window,not labeled inverse-model answers."""
    a, b, ys = fixture()
    rule = C.compile_rule(a, b, ys)
    pair = {
        "origin_id": "a",
        "destination_id": "b",
        "origin_time": 1,
        "destination_time": 2,
        "origin_cell": 0,
        "next_span": 5,
        "same_venue": True,
    }
    record = {"author_key": "opaque", "pairs": [pair]}
    result = C.cold(rule, record)
    assert result["scores"]["joint"].shape == (1, 1, 2, 5)
    with pytest.raises(ValueError):
        C.cold(rule, record | {"true_traits": [1, 0, 1, 0, 1]})
    with pytest.raises(ValueError):
        C.cold(rule, record | {"pairs": [pair, pair]})


def test_missing_source_support_and_constant_output_rejected():
    """No silently neutral population reference or invented behavioral scale."""
    a, b, _ = fixture()
    with pytest.raises(ValueError):
        C.reference(np.zeros_like(a), b)
    with pytest.raises(ValueError):
        C.reference(a, np.ones_like(b))


def test_metadata_units_and_no_cross_window_edge():
    """Observed spans/gaps/transitions keep their actual field meanings."""
    np.testing.assert_allclose(
        C.pair_observations([1, 3, 7], [0, 1, 4], [1, 1, 2]),
        [[np.log(4), np.log(2), 1], [np.log(8), np.log(4), 0]],
    )
    with pytest.raises(ValueError):
        C.pair_observations([1, 3], [1, 1], [1, 2])
