#!/usr/bin/env python3
"""Bilinear identities, original-address partitioning and source-only fitting."""

from itertools import permutations

import numpy as np
import pytest

from grain_research.local_bilinear import (
    ARMS,
    ConstructionFeatures,
    author_blocks,
    chunk_spans,
    keyed_directions,
)


@pytest.mark.parametrize("text", ["", "One", " a  b\nc\t d ", "日本語 の 文です。"])
def test_chunks_preserve_every_original_character(text):
    spans = chunk_spans(text, 2)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(len(text[a:b].split()) <= 2 for a, b in spans)


def test_covariance_and_complete_permutation_identity():
    g = np.array([[1.0, 3.0], [-2.0, 4.0], [6.0, 0.0]])
    s = np.array([[0.3, 0.8], [0.9, 0.1], [0.7, 0.2]])
    b, counts, energy = author_blocks(g, s, s, [0, 0, 0])
    centered = (g - g.mean(0)).T @ (s - s.mean(0)) / 3
    np.testing.assert_allclose((b["paired"] - b["outer"])[0], centered.ravel())
    every = [g.T @ s[list(p)] / 3 for p in permutations(range(3))]
    np.testing.assert_allclose(np.mean(every, axis=0), b["outer"][0].reshape(2, 2))
    assert counts.tolist() == [3]
    assert energy[0] == pytest.approx(np.linalg.norm(centered))


def test_same_marginals_can_have_different_pairing():
    g = np.array([[1.0], [-1.0]])
    s = np.array([[2.0], [-2.0]])
    a, _, _ = author_blocks(g, s, s, [0, 0])
    b, _, _ = author_blocks(g, s[::-1], s[::-1], [0, 0])
    for key in ("semantic", "grain", "affinity", "outer"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["paired"][0, 0] == 2.0 and b["paired"][0, 0] == -2.0


def test_single_unit_cannot_gain_colocation_information():
    b, counts, energy = author_blocks([[1.0, 2.0]], [[3.0, 4.0]], [[0.2]], [0])
    for name in ("paired", "shuffle924", "shuffle925", "shuffle926"):
        np.testing.assert_array_equal(b[name], b["outer"])
    assert counts.tolist() == [1] and energy[0] == 0.0


def test_shuffle_is_deterministic_and_keeps_all_units():
    rng = np.random.default_rng(22)
    g, h = rng.normal(size=(20, 3)), rng.normal(size=(20, 4))
    a, _, _ = author_blocks(g, h, h[:, :2], [0] * 10 + [1] * 10)
    b, _, _ = author_blocks(g, h, h[:, :2], [0] * 10 + [1] * 10)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError, match="Complete ordered author"):
        author_blocks(g, h, h, [1] * 20)


def test_keyed_directions_are_cosine_mean_contrasts():
    q = np.array([[3.0, 0.0], [0.0, 2.0]])
    d, names = keyed_directions(
        q, [{"facet": "A", "key": 1}, {"facet": "A", "key": -1}]
    )
    np.testing.assert_array_equal(d[:, 0], [1.0, -1.0])
    assert names == ["A"]
    with pytest.raises(ValueError, match="Both item"):
        keyed_directions(q[:1], [{"facet": "A", "key": 1}])


@pytest.mark.parametrize("arm", [a for a in ARMS if a != "word_char"])
def test_source_scaling_and_single_batch_invariance(arm):
    rng = np.random.default_rng(14)
    g, h = rng.normal(size=(40, 3)), rng.normal(size=(40, 5))
    blocks, _, _ = author_blocks(g, h, h[:, :2], np.repeat(np.arange(10), 4))
    docs = ["work study" for _ in range(10)]
    source = np.arange(8)
    features = ConstructionFeatures(arm).fit(blocks, docs, source)
    frozen_mean = features.scaler_.mean_.copy()
    one = features.transform(blocks, docs, [8])
    many = features.transform(blocks, docs, [9, 8, 8])
    np.testing.assert_allclose(one[0], many[1])
    np.testing.assert_allclose(one[0], many[2])
    np.testing.assert_array_equal(frozen_mean, features.scaler_.mean_)
    perturbed = {k: v.copy() for k, v in blocks.items()}
    for values in perturbed.values():
        values[8:] += 500
    again = ConstructionFeatures(arm).fit(perturbed, docs, source)
    np.testing.assert_allclose(again.scaler_.mean_, frozen_mean)


def test_word_vocabulary_does_not_learn_held_terms():
    docs = ["work calm talk"] * 5 + ["heldonlyword"] * 2
    f = ConstructionFeatures("word_char").fit({}, docs, np.arange(5))
    assert "heldonlyword" not in f.text_.transformer_list[0][1].vocabulary_
    assert f.transform({}, docs, [5, 6]).shape[0] == 2
