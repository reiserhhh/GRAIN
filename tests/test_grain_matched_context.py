#!/usr/bin/env python3
"""Numerical and source-token controls for matched context construction."""

import numpy as np
import pytest

from grain_research.matched_context import observations, pool_context


class TinyTokenizer:
    """Deterministic character tokenizer that preserves original offsets."""

    cls_token_id = 1001
    sep_token_id = 1002

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(x) for x in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def encode(self, text, **kwargs):
        ids = [ord(x) for x in text]
        return [1001, *ids, 1002] if kwargs.get("add_special_tokens") else ids

    def build_inputs_with_special_tokens(self, ids):
        return [1001, *ids, 1002]


def test_partition_and_offsets():
    text = "a b! c\n" * 31
    packets, info = observations(text, TinyTokenizer(), budget=101, piece_size=29)
    whole, pieces = packets[0], packets[1:]
    assert [x for p in pieces for x in p["ids"][8:-1]] == whole["ids"][8:-1]
    assert "".join(text[p["char_lo"] : p["char_hi"]] for p in pieces) == text[:101]
    assert sum(p["hi"] - p["lo"] for p in pieces) == 101
    assert info["omitted_tokens"] == len(text) - 101
    assert not info["complete_document"]


def test_short_document_identity():
    packets, info = observations("short", TinyTokenizer())
    assert packets[0]["ids"] == packets[1]["ids"]
    assert info["complete_document"] and info["pieces"] == 1
    h = np.array([0.6, 0.8])
    f, radius = pool_context(h, h[None, :], [5])
    assert radius == 1
    for k in ("whole", "piece_equal", "piece_mass", "piece_mass_unit"):
        np.testing.assert_allclose(f[k], h)


def test_geometric_decomposition_in_anchor_space():
    rng = np.random.default_rng(31)
    pieces = rng.normal(size=(4, 12))
    pieces /= np.linalg.norm(pieces, axis=1)[:, None]
    whole = rng.normal(size=12)
    whole /= np.linalg.norm(whole)
    d = rng.normal(size=(12, 7))
    f, _ = pool_context(whole, pieces, [128, 128, 128, 116])
    total, a, b = (
        (f["whole"] - f["piece_mass"]) @ d,
        f["direction"] @ d,
        f["radial"] @ d,
    )
    np.testing.assert_allclose(total, a + b, atol=1e-14)
    np.testing.assert_allclose(total @ total, a @ a + b @ b + 2 * a @ b, atol=1e-14)


def test_weighting_changes_only_when_unequal():
    p = np.eye(3)
    f, _ = pool_context(p[0], p, [2, 2, 2])
    np.testing.assert_allclose(f["piece_equal"], f["piece_mass"])
    f, _ = pool_context(p[0], p, [1, 1, 8])
    np.testing.assert_allclose(f["piece_mass"], [0.1, 0.1, 0.8])


@pytest.mark.parametrize("masses", [[0, 1], [-1, 2], [np.nan, 1], [1]])
def test_bad_masses(masses):
    with pytest.raises(ValueError):
        pool_context(np.array([1.0, 0]), np.eye(2), masses)


def test_zero_direction_rejected():
    with pytest.raises(ValueError, match="Zero pooled"):
        pool_context([1.0, 0], [[1.0, 0], [-1.0, 0]], [1, 1])


def test_nonunit_rejected():
    with pytest.raises(ValueError, match="unit normalized"):
        pool_context([2.0, 0], np.eye(2), [1, 1])


@pytest.mark.parametrize("budget,piece", [(0, 1), (1, 0), (-1, 5)])
def test_bad_budgets(budget, piece):
    with pytest.raises(ValueError):
        observations("x", TinyTokenizer(), budget, piece)
