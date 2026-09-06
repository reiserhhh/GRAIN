#!/usr/bin/env python3
"""Synthetic boundaries and contrast tests for the independent domain packet."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.domain_context import (
    qualify_case,
    sextile_spans,
    spread_indices,
    user_parts,
)
from grain_research.vector_mechanisms.response_order import permute_prefix
from scripts.run_grain_domain_context import CONDITIONS, summarize


class CharTokenizer:
    """Expose exact roles and offset positions without external model files."""

    all_special_ids = ()

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not add_generation_prompt
        text = "".join("[" + m["role"] + "]\n" + m["content"] + "\n" for m in messages)
        return {"input_ids": self.encode(text)} if tokenize else text


def test_sextile_roundtrip_and_declared_gap():
    text = "First section is here. Another section follows. Then more information. " + (
        " More careful sentences follow here." * 10
    )
    spans = sextile_spans(text)
    assert "".join(text[a:b] for a, b in spans) == text
    assert spans[2][1] < spans[5][0]
    assert all(text[a].isspace() for a, b in spans[1:])


def test_single_word_cannot_invent_six_spans():
    with pytest.raises(ValueError):
        sextile_spans("abcdefghi")


def test_roles_current_and_order_bag():
    t = CharTokenizer()
    history = "This is prior writing from the same source person."
    current = "This is later writing by the source person."
    prefix, body, ix, _ = user_parts(t, current, history)
    _, alone, _, _ = user_parts(t, current)
    assert body == alone == list(map(ord, current))
    assert "[assistant]" not in "".join(map(chr, prefix))
    shuffled, meta = permute_prefix(prefix, ix, "D0", "block8", 731)
    assert sorted(shuffled) == sorted(prefix)
    assert meta["outside_span_changed"] is False
    assert meta["inverse_max_token_error"] == 0
    qualified = qualify_case({"current": current, "context": history}, t)
    assert qualified["current_tokens"] == len(current)


def test_complete_budget_rejects_without_truncation():
    with pytest.raises(ValueError):
        qualify_case({"current": "x" * 300, "context": "h" * 50}, CharTokenizer())
    with pytest.raises(ValueError):
        spread_indices(5)
    assert len(set(spread_indices(21))) == 6


def test_all_conditions_and_interaction_identity():
    rows = []
    values = dict(zip(CONDITIONS, [4.0, 3.0, 2.0, 1.0, 2.0, 1.5]))
    for code in ["A", "B", "C"]:
        for condition in CONDITIONS:
            rows.append(
                {
                    "corpus": "test",
                    "language": "en",
                    "case_code": code,
                    "condition": condition,
                    "tail_logp": values[condition],
                    "mean_logp": values[condition],
                }
            )
    result = summarize(pd.DataFrame(rows))
    s = result[result.score == "tail_logp"].set_index("endpoint")
    assert s.loc["actual_minus_matched", "mean"] == 1
    assert s.loc["order_interaction", "mean"] == 0.5
    assert s.loc["common_block8_penalty", "mean"] == 1.75
    with pytest.raises(ValueError):
        summarize(pd.DataFrame(rows[:-1]))
    np.testing.assert_array_equal(spread_indices(6), np.arange(6))
