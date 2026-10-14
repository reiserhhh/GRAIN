#!/usr/bin/env python3
"""Public key identities, unit binding and native prefix-cache equivalence."""

import numpy as np
import pytest

from grain_research.construction import math_item_bridge as M
from grain_research.vector_mechanisms import direct_traits as D


def items():
    """Provide two complete toy keys with the actual six/four pole imbalance."""
    return [
        {"target": t, "pole": p} for t in ("A", "C") for p in ["high"] * 6 + ["low"] * 4
    ]


def test_exact_public_eight_event_parser_and_extra_event_rejection():
    body = [f"Public statement {i}." for i in range(8)]
    user, _ = D.render(body, "agreeableness")
    assert M.public_bodies(user) == body
    nine, _ = D.render(body + ["extra"], "agreeableness")
    seven, _ = D.render(body[:-1], "agreeableness")
    with pytest.raises(ValueError):
        M.public_bodies(nine)
    with pytest.raises(ValueError):
        M.public_bodies(seven)


def test_official_balanced_and_common_identity_and_probability_bounds():
    rng = np.random.default_rng(255)
    z = rng.normal(size=(12, 16, 20, 2))
    q = M.keyed_observations(z, items(), ["A", "C"])
    for key, offset in (("probability", 0.5), ("log_odds", 0)):
        h, l, b, u, official = np.moveaxis(q[key], -1, 0)
        assert np.allclose(b, offset + (h - l) / 2)
        assert np.allclose(official, b + 0.2 * (u - offset))
        assert np.allclose(official, 0.6 * h + 0.4 * (2 * offset - l))
    assert np.all((q["probability"][..., 2] >= 0) & (q["probability"][..., 2] <= 1))


def test_indiscriminate_endorsement_moves_common_not_balanced_key():
    z = np.zeros((2, 16, 20, 2))
    z[:, 8:, :, 1] = 2
    q = M.keyed_observations(z, items(), ["A", "C"])
    assert np.allclose(q["probability"][..., 2], 0.5)
    assert np.all(q["probability"][:, 8:, :, 3] > 0.5)
    assert np.all(q["probability"][:, 8:, :, 4] > 0.5)


def test_reverse_key_changes_key_not_common_and_keeps_all_items():
    rng = np.random.default_rng(10)
    z = rng.normal(size=(20, 2))
    old = M.keyed_observations(z, items(), ["A", "C"])
    inverse = [{**r, "pole": "low" if r["pole"] == "high" else "high"} for r in items()]
    new = M.keyed_observations(z, inverse, ["A", "C"])
    assert np.allclose(old["probability"][..., 2] + new["probability"][..., 2], 1)
    assert np.allclose(old["log_odds"][..., 2], -new["log_odds"][..., 2])
    assert np.allclose(old["probability"][..., 3], new["probability"][..., 3])


def test_positive_law_levels_and_signed_action_are_not_independent_items():
    rng = np.random.default_rng(25)
    p = rng.dirichlet(np.ones(16), size=(12, 2))
    y = rng.normal(size=(12, 16, 2, 5))
    levels, action = M.law_actions(y, p)
    assert levels.shape == (12, 2, 2, 5)
    assert np.allclose(action, np.einsum("uk,ukqj->uqj", p[:, 0] - p[:, 1], y))
    assert np.allclose(M.law_actions(np.ones_like(y), p)[1], 0)
    with pytest.raises(ValueError):
        M.law_actions(y, p[:, :, :-1])
    with pytest.raises(ValueError):
        M.keyed_observations(y, items(), ["A", "C"])
    with pytest.raises(ValueError):
        M.keyed_observations(np.zeros((19, 2)), items()[:-1], ["A", "C"])


def test_prefix_is_exact_and_mixed_contexts_are_not_merged():
    assert M.common_prefix([[1, 2, 3, 9], [1, 2, 4, 8]]) == 2
    with pytest.raises(ValueError):
        M.common_prefix([[1], [1]])
    with pytest.raises(ValueError):
        M.common_prefix([[1, 2], [3, 2]])


def test_cache_equals_direct_on_local_synthetic_model_with_unequal_suffixes():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(255)
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    sequences = [[1, 3, 5, 7, 11], [1, 3, 5, 8], [1, 3, 5, 9, 12, 14]]
    q = M.native_items(model, sequences, [0, 1], "cpu", batch_size=2)
    with torch.inference_mode():
        for i, sequence in enumerate(sequences):
            h = model.model(
                input_ids=torch.tensor([sequence]), use_cache=False
            ).last_hidden_state[0, -1]
            assert np.allclose(q["states"][i], h.numpy(), atol=1e-6)
            assert np.allclose(
                q["logits"][i], model.lm_head(h)[[0, 1]].numpy(), atol=1e-6
            )
