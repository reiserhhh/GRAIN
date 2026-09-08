#!/usr/bin/env python3
"""Alignment, finite-response accounting, and actual decoder-suffix replay tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.direct_traits import render
from grain_research.vector_mechanisms.full_state_edit import (
    align_excerpts,
    align_states,
    capture_boundary,
    grid_components,
    mix_states,
    replay_suffix,
)


def char_encoding(text):
    return {
        "input_ids": list(map(ord, text)),
        "offset_mapping": [(i, i + 1) for i in range(len(text))],
    }


def test_alignment_keeps_bodies_and_explicitly_rebases_numbering():
    texts = ["first content", "same content", "same content", "last content"]
    full, spans = render(texts, "neuroticism")
    edit, espans = render([texts[1], texts[3]], "neuroticism")
    mapping, audit = align_excerpts(
        full, edit, spans, espans, [1, 3], char_encoding(full), char_encoding(edit)
    )
    assert "".join(full[i] for i in mapping) == edit
    assert audit["renumbered_headers"] == 2
    assert mapping[espans[0][0]] == spans[1][0]
    assert mapping[espans[1][0]] == spans[3][0]
    with pytest.raises(ValueError):
        align_excerpts(
            full, edit, spans, espans, [0, 2], char_encoding(full), char_encoding(edit)
        )


def test_identity_mapping_and_zero_edit():
    text, spans = render(["one", "two"], "TF_cont")
    enc = char_encoding(text)
    mapping, audit = align_excerpts(text, text, spans, spans, [0, 1], enc, enc)
    np.testing.assert_array_equal(mapping, np.arange(len(text)))
    assert audit["monotone"] and not audit["renumbered_headers"]
    assert all(
        value == 0 for value in grid_components(0.4, 0.4, 0.4, 0.4, 0.4).values()
    )


def test_unmatched_interior_body_token_is_not_hidden_as_boundary():
    text, spans = render(["abcdef", "second"], "TF_cont")
    original, corrupt = char_encoding(text), char_encoding(text)
    corrupt["input_ids"][spans[0][0] + 3] += 1000
    with pytest.raises(ValueError, match="interior body"):
        align_excerpts(text, text, spans, spans, [0, 1], original, corrupt)


def test_four_cell_interaction_is_not_added_to_allocations_twice():
    v = grid_components(0.1, 0.2, 0.3, 0.4, 0.8)
    assert v["interaction"] == pytest.approx(0.3)
    assert v["rebase"] + v["context_main"] + v["query_main"] + v[
        "interaction"
    ] == pytest.approx(v["natural"])
    assert v["rebase"] + v["context_allocated"] + v["query_allocated"] == pytest.approx(
        v["natural"]
    )


def test_mix_never_mutates_and_selects_only_intended_positions():
    a, b = torch.zeros((1, 4, 3)), torch.ones((1, 4, 3))
    h = mix_states(a, b, 0, 1)
    assert h[:, :-1].sum() == 0 and h[:, -1].sum() == 3
    assert a.sum() == 0 and b.sum() == 12


def test_fresh_boundary_is_preserved_not_dropped_or_mislabelled():
    full = torch.arange(12.0).reshape(1, 4, 3)
    edited = torch.ones((1, 3, 3)) * 99
    aligned = align_states(full, edited, [0, -1, 3])
    torch.testing.assert_close(aligned[:, 1], edited[:, 1])
    torch.testing.assert_close(aligned[:, 0], full[:, 0])
    torch.testing.assert_close(aligned[:, -1], full[:, -1])
    assert edited.sum() == 891


def test_small_qwen_suffix_reproduces_full_native_scores_without_prefix_calls():
    torch.manual_seed(9)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=19,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    observer = SimpleNamespace(model=model, calls=0)
    alphabet = {"token_ids": [2, 7], "values": [0.0, 1.0], "labels": ["a", "b"]}
    h, kw, original = capture_boundary(observer, [3, 4, 5, 6], alphabet)

    def forbidden(*_args):
        raise AssertionError("Suffix replay executed prefix")

    hook = model.model.layers[0].register_forward_pre_hook(forbidden)
    try:
        replay = replay_suffix(observer, h, kw, alphabet)
    finally:
        hook.remove()
    assert replay["conditional_score"] == pytest.approx(
        original["conditional_score"], abs=1e-7
    )
    assert observer.calls == 2
