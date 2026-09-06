"""Synthetic-only invariants for response-context material and arithmetic."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.response_context import (
    context_derangements,
    cosine,
    native_parts,
    packed_sequence,
    paired_interval,
    select_responses,
)


class CharacterChatTokenizer:
    """A tiny native template with mapping-valued tokenization results."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not add_generation_prompt
        text = "".join("[" + m["role"] + "]\n" + m["content"] + "\n" for m in messages)
        return {"input_ids": self.encode(text)} if tokenize else text

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_native_body_is_identical_and_accepts_mapping_api():
    tokenizer = CharacterChatTokenizer()
    _, actual = native_parts(tokenizer, "a current response", "a previous response")
    _, alone = native_parts(tokenizer, "a current response")
    assert actual == alone == tokenizer.encode("a current response")


def test_last_eligible_pair_selected_without_truncation():
    tokenizer = CharacterChatTokenizer()
    turns = [
        {"role": "assistant", "text": "a" * 24, "source_sequence": 0},
        {"role": "user", "text": "x" * 24, "source_sequence": 1},
        {"role": "assistant", "text": "b" * 24, "source_sequence": 2},
        {"role": "user", "text": "y" * 20, "source_sequence": 3},
        {"role": "assistant", "text": "c" * 24, "source_sequence": 4},
        {"role": "user", "text": "ok", "source_sequence": 5},
    ]
    views = [
        {
            "metadata": {"view_kind": "free_chat", "participant_code": "P001"},
            "turns": turns,
        }
    ]
    cases, coverage = select_responses(views, tokenizer)
    assert cases[0]["response_sequence"] == 3 and cases[0]["response"] == "y" * 20
    assert coverage[0]["length_rejected"] == 1


def test_assignments_are_complete_fixed_point_free_and_reproducible():
    a, b = context_derangements([20, 21, 70, 71, 150, 151])
    x, y = context_derangements([20, 21, 70, 71, 150, 151])
    for z in (a, b):
        assert sorted(z) == list(range(6))
        assert not (z == np.arange(6)).any()
    assert np.array_equal(a, x) and np.array_equal(b, y)
    assert set(zip(range(6), a)) == {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)}


def test_current_tokens_positions_and_valid_sequence_preserved():
    prefix, body = [12, 13, 14], [99, 98, 97]
    padded, mask, indices = packed_sequence(prefix, body, 0)
    compact, cmask, ci = packed_sequence(prefix, body, 0, compact=True)
    assert len(padded) == len(compact) == 1024 and indices[0] == 768
    assert [padded[i] for i in indices] == [compact[i] for i in ci] == body
    assert np.array_equal(
        np.asarray(padded)[np.asarray(mask, bool)],
        np.asarray(compact)[np.asarray(cmask, bool)],
    )
    _, _, other = packed_sequence([44] * 90, body, 0)
    assert indices == other
    with pytest.raises(ValueError):
        packed_sequence([1] * 769, body, 0)


def test_fixed_material_interval_and_cosine():
    r = paired_interval([0.1] * 12)
    assert r["lower"] == pytest.approx(0.1) and r["upper"] == pytest.approx(0.1)
    assert r["positive_count"] == 12
    assert cosine([1, 0], [1, 0]) == 1
    assert cosine([1, 0], [0, 1]) == 0
    with pytest.raises(ValueError):
        paired_interval([np.nan])
