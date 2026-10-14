#!/usr/bin/env python3
"""Synthetic finite nulls and bindings for the natural-text link observer."""

import numpy as np
import pytest

from grain_research.construction import math_author_predictive_link as M


def test_first_pair_and_missing_scope():
    """Selection uses only conditions/lengths, never target labels or outcomes."""
    n = np.full((3, 2, 2), 12)
    c = np.array([[[0, 1], [1, 0]]] * 3)
    assert np.array_equal(M.select_pairs(n, c, 16), [[0, 1]] * 3)
    with pytest.raises(ValueError):
        M.select_pairs(n, c, 4)


def test_stranger_is_derangement_and_condition_matched():
    """All authors retain one foreign context and nobody donates to themself."""
    x = np.random.default_rng(257).normal(size=(4, 2, 3))
    n, c = np.full((4, 2), 32), np.array([[0, 1]] * 4)
    donor, event, costs = M.foreign_matching(x, n, c, np.zeros(4, int), 0, 0)
    assert len(set(donor)) == 4 and (donor != np.arange(4)).all()
    assert (event == 0).all() and np.isfinite(costs).all()


def test_no_match_is_failure_not_silent_relaxation():
    """An unavailable stranger cannot be replaced by self or another condition."""
    with pytest.raises(ValueError):
        M.foreign_matching(np.ones((2, 1, 2)), [[32], [32]], [[0], [1]], [0, 0], 0, 0)


def test_criterion_blind_context_has_zero_target_link():
    """Generic context gain alone must not be called target evidence."""
    z = np.random.default_rng(4).normal(size=(2, 4, 11, 7))
    z[:, :, 1:] = z[:, :, :1] + np.arange(10)[None, None, :, None]
    a = np.zeros_like(z)
    r = M.accounts(z, a)
    assert abs(r["stranger"]).max() < 1e-14
    assert abs(r["stranger_neutral"]).max() > 0


def test_criterion_context_interaction_and_normalizer_are_retained():
    """Positive and opposing vocabulary competition survive the exact account."""
    z, a = np.zeros((1, 4, 11, 3)), np.zeros((1, 4, 11, 3))
    z[:, 1, 1] = 3
    a[:, 1, 1] = 5
    r = M.accounts(z, a)
    assert np.array_equal(r["stranger"][0, 0], [-2] * 3)
    assert np.array_equal(r["stranger_numerator"][0, 0], [3] * 3)
    assert np.array_equal(r["stranger_normalizer"][0, 0], [5] * 3)


def test_common_logit_gauge_and_order_control():
    """A common logit shift cancels; token-bag identity is not order identity."""
    rng = np.random.default_rng(8)
    z, a = rng.normal(size=(2, 4, 11, 5)), rng.normal(size=(2, 4, 11, 5))
    g = rng.normal(size=(2, 4, 11, 5))
    assert np.allclose(
        M.accounts(z, a)["stranger"], M.accounts(z + g, a + g)["stranger"]
    )
    x = [12, 5, 19, 5]
    reverse = x[::-1]
    assert sorted(x) == sorted(reverse) and x != reverse


def test_contact_is_numeric_and_preserves_negative_cells():
    """Cold measurement has no labels/models and must not rectify its contacts."""
    x = np.arange(12).reshape(4, 3)
    result = M.contact(x, -x[:, :2])
    assert result.shape == (3, 2) and (result < 0).all()


def test_wrong_or_nonfinite_input_rejected():
    """Incomplete voice banks cannot masquerade as a completed experiment."""
    with pytest.raises(ValueError):
        M.accounts(np.zeros((1, 4, 10, 2)), np.zeros((1, 4, 10, 2)))
    z = np.zeros((1, 4, 11, 2))
    z[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        M.accounts(z, np.zeros_like(z))


def test_cold_score_replays_signed_accounts_without_text():
    """Numeric transport retains target order and all signed context operations."""
    from scripts.score_grain_author_predictive_link_cold import score

    rng = np.random.default_rng(25)
    z, a = rng.normal(size=(1, 4, 11, 1)), rng.normal(size=(1, 4, 11, 1))
    result = score(
        {
            "version": "HM257-v1",
            "targets": list(M.TARGETS),
            "units": [
                {
                    "unit_id": 7,
                    "tokens": 23,
                    "chosen": z[0, ..., 0].tolist(),
                    "partition": a[0, ..., 0].tolist(),
                }
            ],
        }
    )
    assert result["rows"][0]["unit_id"] == 7
    for key in ("stranger", "order", "carry"):
        assert np.allclose(result["rows"][0][key], M.accounts(z, a)[key][0, :, 0])
    with pytest.raises(ValueError):
        score({"version": "unknown", "targets": list(M.TARGETS)})


def test_prompt_preserves_native_context_and_requests_token_list():
    """Template return types cannot silently replace the observed native tokens."""
    from scripts.run_grain_author_predictive_link_hm257 import prompt

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            return [11, 12, 99, 10]

        def convert_tokens_to_ids(self, text):
            assert text == "<|im_end|>"
            return 99

        def encode(self, text, **kwargs):
            return [10] if text == "\n" else [77, 88]

    tokenizer = Tokenizer()
    original = [23, 45, 23]
    assert prompt(tokenizer, M.VOICES[0], original) == [
        11,
        12,
        23,
        45,
        23,
        99,
        10,
        77,
        88,
    ]
    assert original == [23, 45, 23]
