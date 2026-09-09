#!/usr/bin/env python3
"""Portable input and dependency boundaries on synthetic material only."""

from types import SimpleNamespace

import numpy as np
import pytest

from grain_research.vector_mechanisms.navigation_runtime import (
    checked_path,
    encode_texts,
)


class CharacterTokenizer:
    """Deterministic fake tokenizer whose offsets are character coordinates."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(
            ids=list(map(ord, text)), offsets=[(i, i + 1) for i in range(len(text))]
        )


def recipe():
    """Include a question marker in the evidence without treating it as the query."""
    return {
        "prefix": "Writer\nExcerpt 1:\n",
        "separator": "\n\nExcerpt {index}:\n",
        "suffix": "\nEnd\nQuestion: Target\nAnswer:",
        "question_offset_in_suffix": len("\nEnd\n"),
    }


def test_raw_recipe_query_positions_do_not_depend_on_excerpt_content():
    tok, spec = CharacterTokenizer(), recipe()
    expected = list(map(ord, "Question: Target\nAnswer"))
    for texts in (["Hello"], ["Question: fake\n", "日本語", "third"]):
        ids, mask = encode_texts(tok, texts, spec, expected, maximum=1000)
        assert np.asarray(ids)[mask].tolist() == expected
        assert mask[-1] == np.bool_(False)
        assert "日本語" in "".join(map(chr, ids)) if len(texts) > 1 else True


def test_raw_input_rejects_silent_truncation_and_wrong_question():
    tok, spec = CharacterTokenizer(), recipe()
    expected = list(map(ord, "Question: Target\nAnswer"))
    for texts in ([], [""], ["x", 2], "not a list"):
        with pytest.raises(ValueError):
            encode_texts(tok, texts, spec, expected)
    with pytest.raises(ValueError, match="capacity"):
        encode_texts(tok, ["x" * 200], spec, expected, maximum=100)
    with pytest.raises(ValueError, match="coordinates"):
        encode_texts(tok, ["x"], spec, [0])


def test_bundle_paths_do_not_escape_through_parent_or_symlink(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "good.json").write_text("{}")
    (tmp_path / "outside.json").write_text("{}")
    (root / "alias.json").symlink_to(tmp_path / "outside.json")
    assert checked_path(root, "good.json") == root / "good.json"
    for relative in ("../outside.json", "alias.json", "missing.json"):
        with pytest.raises(ValueError):
            checked_path(root, relative)
