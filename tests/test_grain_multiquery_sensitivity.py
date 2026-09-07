#!/usr/bin/env python3
"""Synthetic safeguards for common edits and differentiable native readouts."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.direct_traits import native_readout, render
from scripts.run_grain_multiquery_sensitivity import (
    common_edits,
    conditional_torch,
    edit_indices,
    query_prompt,
)


def test_common_edits_are_shared_disjoint_reproducible_and_length_matched():
    rng = np.random.default_rng(8)
    vectors, poles = rng.normal(size=(12, 9)), rng.normal(size=(3, 2, 9))
    lengths = np.full(12, 100)
    a, b, scores, match = common_edits(vectors, poles, lengths)
    assert not set(a) & set(b)
    assert 1 <= len(a) == len(b) <= 3
    assert set(a) == set(np.argmax(abs(scores), axis=0))
    assert match["within_tolerance"]
    assert (a, b) == common_edits(vectors, poles, lengths)[:2]
    edits = edit_indices(12, a, b)
    assert edits["joint"] == sorted(set(edits["e1"]) & set(edits["e2"]))


def test_invalid_edit_overlap_is_rejected():
    with pytest.raises(ValueError):
        edit_indices(12, [0], [0])


@pytest.mark.parametrize("task", ["TF_cont", "neuroticism"])
def test_original_prompts_are_unchanged(task):
    assert query_prompt(["First excerpt.", "Second excerpt."], task) == render(
        ["First excerpt.", "Second excerpt."], task
    )


def test_narrow_query_does_not_claim_an_attachment_diagnosis():
    prompt, _ = query_prompt(["Some text."], "confirmation_seeking")
    assert "reassurance" in prompt and "attachment" not in prompt


def test_torch_readout_and_gradient_match_native_and_finite_difference():
    alphabet = {"labels": ["T", "F"], "token_ids": [1, 3], "values": [0.0, 1.0]}
    z = torch.tensor(
        [0.2, 0.7, -1.0, 0.1, 0.9], dtype=torch.float64, requires_grad=True
    )
    s = conditional_torch(z, alphabet)
    (g,) = torch.autograd.grad(s, z)
    assert np.isclose(
        float(s.detach()),
        native_readout(z.detach().numpy(), alphabet)["conditional_score"],
    )
    eps = 1e-5
    d = torch.tensor([0.0, -0.3, 0.0, 0.4, 0.0], dtype=torch.float64)
    finite = (
        conditional_torch(z + eps * d, alphabet)
        - conditional_torch(z - eps * d, alphabet)
    ) / (2 * eps)
    assert np.isclose(float(finite.detach()), float(g @ d), atol=1e-9)
