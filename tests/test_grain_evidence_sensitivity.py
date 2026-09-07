#!/usr/bin/env python3
"""Reference fitting and information ladder checks for comment sensitivity."""

import numpy as np

from scripts.fit_grain_evidence_sensitivity import (
    feature_blocks,
    fit_scale,
    scale_blocks,
)


def test_context_features_preserve_the_content_and_structure_blocks():
    rng = np.random.default_rng(12)
    h, lengths = rng.normal(size=(12, 8)), np.arange(1, 13)
    structure = feature_blocks(h, lengths, "position_length")
    content = feature_blocks(h, lengths, "comment_content")
    context = feature_blocks(h, lengths, "comment_context")
    assert np.array_equal(structure[0], content[0])
    assert all(np.array_equal(a, b) for a, b in zip(content, context))
    assert np.allclose(context[-1], h * h.mean(0))


def test_scales_do_not_learn_from_held_out_comments():
    rng = np.random.default_rng(3)
    blocks = [rng.normal(size=(12, 8))]
    training = np.arange(8)
    scales = fit_scale(blocks, training)
    single = scale_blocks(blocks, [9], scales)
    blocks[0][11] = 1e9
    assert np.allclose(single, scale_blocks(blocks, [9, 10, 11], scales)[0])
    assert np.array_equal(scales[0][0], blocks[0][training].mean(0))
