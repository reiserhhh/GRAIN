#!/usr/bin/env python3
"""Synthetic arithmetic and material tests for the new HM46 response experiment."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_construct_routing as M

ROOT = Path(__file__).resolve().parents[1]


def test_materials_are_complete_and_disjoint():
    """Six contents, two independent examples and all eight cells stay visible."""
    spec = json.loads(
        (ROOT / "configs/grain_math_construct_routing_v1.json").read_text()
    )
    docs = M.documents(spec)
    assert len(docs) == 192
    assert len({r["body"] for r in docs}) == 192
    for r in docs:
        a, b = r["property_interval"]
        assert r["body"][a:b] == r["property"]
        assert r["body"].count(r["property"]) == 1
        assert (
            "only imagined" in r["body"] or "imagined scenario" in r["body"]
        ) == bool(r["imagined"])
        assert ("my colleague" in r["body"]) == bool(r["other"])


def test_cube_orthogonality_and_inverse():
    """Every interaction is retained even when marginal effects cancel."""
    h = M.characters()
    np.testing.assert_array_equal(h @ h.T, 8 * np.eye(8))
    values = np.random.default_rng(1).normal(size=(2, 2, 2, 5))
    beta, check = M.split_cube(values)
    assert beta.shape == (8, 5)
    assert check["reconstruction_max_error"] < 1e-14
    assert check["parseval_error"] < 1e-14


def test_binding_interaction_not_a_main_effect():
    """An owner-by-polarity effect survives despite zero polarity marginal."""
    values = np.array([(-1) ** (p + o) for p, o, _ in M.BITS]).reshape(2, 2, 2)
    beta, _ = M.split_cube(values)
    assert beta[M.TERMS.index("polarity")] == 0
    assert beta[M.TERMS.index("polarity*other")] == 1


def test_cube_rejects_missing_and_nonfinite_cells():
    """Unobserved conditions cannot be silently zero-filled."""
    with pytest.raises(ValueError):
        M.split_cube(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        M.split_cube(np.full((2, 2, 2), np.nan))


def test_spectrum_identity_rank_one_and_zero():
    """Output rank describes the actual matrix rather than a hand-set factor count."""
    assert M.spectrum(np.eye(6))["effective_rank"] == pytest.approx(6)
    assert M.spectrum(np.ones((6, 6)))["leading_squared_share"] == pytest.approx(1)
    assert M.spectrum(np.zeros((6, 6)))["leading_squared_share"] is None


def test_random_norm_and_reproducibility():
    """Random carrier controls match delta norm without sharing its direction."""
    delta = np.random.default_rng(2).normal(size=(4, 1536))
    a, b = M.norm_matched(delta, 10), M.norm_matched(delta, 10)
    np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(np.linalg.norm(delta))
    np.testing.assert_array_equal(M.norm_matched(np.zeros((3, 4)), 1), 0)


def test_token_intersections_are_addressed():
    """Boundary-spanning tokens are retained, special empty offsets are not."""
    assert M.token_positions([(0, 0), (0, 3), (3, 7), (7, 8)], 2, 7) == [1, 2]
    with pytest.raises(ValueError):
        M.token_positions([(0, 0)], 1, 4)


def test_fixed_query_scope():
    """Six query definitions are fixed; no outcome-guided fallback question exists."""
    for target in (*M.BIG5, "anthropomorphism"):
        assert "0 to 9" in M.query(target)
    with pytest.raises(ValueError):
        M.query("unknown")


def test_render_accepts_mapping_template_tokens():
    """Current Transformers returns a mapping even with tokenize=True."""

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, **_kwargs):
            text = "SYS:" + messages[0]["content"] + "USR:" + messages[1]["content"]
            return {"input_ids": list(range(len(text)))} if tokenize else text

        def __call__(self, text, **_kwargs):
            return {
                "input_ids": list(range(len(text))),
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

    rendered, encoded, offset = M.render("a unique body", "extraversion", Tokenizer())
    assert rendered[offset : offset + 13] == "a unique body"
    assert len(encoded["input_ids"]) == len(rendered)
