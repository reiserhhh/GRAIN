#!/usr/bin/env python3
"""Synthetic compatibility checks; no learned model or private data."""

import numpy as np
import pytest
from scipy.special import expit

from grain_research.construction.math_probability_bridge import (
    compatibility,
    finite_forward,
    inverse_orders,
    literal_cycle,
    splice_passage,
)


def test_compatible_joint_for_any_prior():
    a = np.log([[0.1, 0.6], [0.3, 0.1], [0.6, 0.3]])
    f = finite_forward(a)
    r = f["finite_odds"] + np.log(4)
    result = compatibility(f["raw_odds"], r)
    assert result["cycle_max"] < 1e-14
    for i in range(3):
        for j in range(3):
            assert abs(literal_cycle(f["log_forward"], expit(r), i, j)) < 1e-14


def test_prior_cannot_rescue_incompatible_shape():
    l = np.array([-2.0, 0.0, 2.0])
    for prior in (-15.0, 0.0, 19.0):
        c = compatibility(l, l + prior + np.array([0.0, 1.0, 0.0]))
        assert c["constant_uniform_error_bound"] == pytest.approx(0.5)
        assert c["cycle_max"] == pytest.approx(1)


def test_restricting_support_changes_prior_not_cycles():
    f = finite_forward([[-100.0, -105.0], [-102.0, -103.0], [-103.0, -100.0]])
    np.testing.assert_allclose(np.exp(f["log_forward"]).sum(0), 1)
    assert not np.allclose(f["raw_equal_prior"], f["finite_equal_prior"])
    c = compatibility(f["raw_odds"], f["finite_odds"])
    assert c["cycle_max"] < 1e-13


def test_order_decoding_preserves_signed_component():
    z = np.zeros((3, 2, 2, 2))
    z[:, :, 0, 0] = 3
    z[:, :, 1, 1] = 1
    r = inverse_orders(z)
    np.testing.assert_array_equal(r["mean"], 2)
    np.testing.assert_array_equal(r["order"], 1)
    np.testing.assert_allclose(r["orders"][..., 0], r["mean"] + r["order"])
    np.testing.assert_allclose(r["orders"][..., 1], r["mean"] - r["order"])


def test_equal_inverse_decisions_need_not_equal_writer_law():
    c = compatibility([-2, 0, 2], [-1, 0, 1])
    assert c["cycle_max"] == 2


def test_log_product_and_odds_cycle_are_identical():
    f = finite_forward([[-5.0, -4.0], [-3.0, -7.0], [-6.0, -2.0]])
    r = np.array([0.3, -0.5, 0.9])
    c = compatibility(f["raw_odds"], r)
    for i in range(3):
        for j in range(3):
            assert literal_cycle(f["log_forward"], expit(r), i, j) == pytest.approx(
                c["cycles"][i, j]
            )


@pytest.mark.parametrize("bad", [[0], [0, float("nan")]])
def test_bad_support_rejected(bad):
    with pytest.raises(ValueError):
        compatibility(bad, bad)


def test_stable_log_normalization_does_not_underflow():
    f = finite_forward([[-1e5, -1e5 - 2], [-1e5 - 3, -1e5]])
    assert np.isfinite(f["finite_odds"]).all()
    np.testing.assert_allclose(np.exp(f["log_forward"]).sum(0), 1, atol=1e-10)


def test_exact_historical_tokens_are_inserted_not_reencoded():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "BEGIN" + messages[1]["content"] + "END"

        def encode(self, text, **kwargs):
            return list(map(ord, text))

    ids, start = splice_passage(Tokenizer(), "prefix", [999, 998], "suffix", "s", 100)
    assert ids[start : start + 2] == [999, 998]
    with pytest.raises(ValueError):
        splice_passage(Tokenizer(), "prefix", [999, 998], "suffix", "s", 2)
