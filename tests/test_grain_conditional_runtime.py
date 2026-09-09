#!/usr/bin/env python3
"""Numerical and raw-rendering contracts for the portable conditional program."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.conditional_models import (
    fit_response_kernel,
    predict_response_kernel,
)
from grain_research.vector_mechanisms.conditional_runtime import (
    conditional_scores,
    encode_with_evidence,
)


def test_scores_exact_and_batch_independent():
    """Compiled single-row scoring equals the source kernel for both model classes."""
    rng = np.random.default_rng(918)
    x, r = rng.normal(size=(12, 8)), rng.normal(size=(12, 3))
    inputs = {
        "x": x,
        "reader": rng.normal(size=(8, 3)),
        "bias": rng.normal(size=3),
        "anchor": np.asarray(0.5),
    }
    models = {
        str(k): fit_response_kernel(
            x @ x.T, r, np.arange(12), [(0, 1), (2, 3)], degree=k, secant=True
        )
        for k in (1, 2)
    }
    a, b = rng.normal(size=(2, 8))
    first = conditional_scores(inputs, models, a)
    conditional_scores(inputs, models, b)
    assert first == conditional_scores(inputs, models, a)
    for name, model in models.items():
        expected = (
            0.5
            + (a @ inputs["reader"] + inputs["bias"]).sum()
            + predict_response_kernel(model, (a @ x.T)[None]).sum()
        )
        np.testing.assert_allclose(first[name], expected, atol=1e-12)
    with pytest.raises(ValueError):
        conditional_scores(inputs, models, np.full(8, np.nan))


class Characters:
    """A one-character tokenizer isolates prompt-boundary arithmetic."""

    def encode(self, text, add_special_tokens=False):
        from types import SimpleNamespace

        return SimpleNamespace(
            ids=[ord(c) for c in text], offsets=[(i, i + 1) for i in range(len(text))]
        )


def test_raw_renderer_keeps_evidence_out_of_query():
    """Only excerpt bodies enter the extra summaries, not separators/questions."""
    spec = {
        "recipe": {
            "prefix": "P:",
            "separator": "/{index}:",
            "suffix": "|Q?X",
            "question_offset_in_suffix": 1,
        },
        "question_tokens": [ord("Q"), ord("?")],
    }
    ids, q, b = encode_with_evidence(Characters(), ["abc", "de"], spec)
    assert "".join(chr(i) for i in np.asarray(ids)[b]) == "abcde"
    assert not (q & b).any()
    assert not b[-1]
    with pytest.raises(ValueError):
        encode_with_evidence(Characters(), ["abc"], spec, maximum=3)
