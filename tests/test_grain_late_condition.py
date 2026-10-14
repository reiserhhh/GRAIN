#!/usr/bin/env python3
"""Finite accounts and tiny-model tests for target-blind prefix reuse."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_late_condition as M
from scripts.score_grain_late_condition_cold import score


def test_same_context_has_no_target_interaction():
    """A duplicated context cannot create a signed target/context contrast."""
    z = np.random.default_rng(8).normal(size=(1, 11, 7))
    assert (
        np.max(abs(M.accounts(np.repeat(z, 2, 0), np.zeros((2, 11, 7)))["effect"])) == 0
    )


def test_additive_target_blind_null():
    """Generic context gains must cancel even if target descriptions differ."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(2, 1, 9)) + rng.normal(size=(1, 11, 9))
    assert np.max(abs(M.accounts(z, np.zeros_like(z))["effect"])) < 1e-14


def test_signed_normalizer_and_cold():
    """Vocabulary competition and negative actions cannot be omitted."""
    z = np.zeros((2, 11, 4))
    a = np.zeros_like(z)
    z[0, 1] = 1
    a[0, 1] = 2
    result = M.accounts(z, a)
    assert result["effect"][0].sum() == -4
    packet = {
        "version": "HM258-v1",
        "units": [
            {
                "unit_id": 0,
                "chosen": z.sum(-1).tolist(),
                "partition": a.sum(-1).tolist(),
            }
        ],
    }
    assert score(packet)["rows"][0]["effect"][0] == -4


@pytest.mark.parametrize("shape", [(2, 10, 8), (4, 11, 8), (2, 11)])
def test_incomplete_family_rejected(shape):
    """Missing targets or a changed sampling unit cannot silently pass."""
    with pytest.raises(ValueError):
        M.accounts(np.zeros(shape), np.zeros(shape))


def test_nonfinite_rejected():
    """A failed observation is not imputed zero."""
    x = np.zeros((2, 11, 3))
    x[1, 3, 1] = np.nan
    with pytest.raises(ValueError):
        M.accounts(x, np.zeros_like(x))


def test_native_template_places_all_targets_after_context():
    """Pinned tokenizer preserves raw context IDs before every target suffix."""
    from transformers import AutoTokenizer

    from scripts.run_grain_joint_persona_style_hm150 import MODEL

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    context = tokenizer.encode("A short archived comment.", add_special_tokens=False)
    prefix, suffixes = M.observation_parts(tokenizer, context)
    assert prefix[-len(context) :] == context
    assert "Hypothetical voice:" not in tokenizer.decode(prefix)
    assert len({tuple(s) for s in suffixes}) == 11
    assert all("Hypothetical voice:" in tokenizer.decode(s) for s in suffixes)


def test_tiny_qwen_cache_native_and_immutability():
    """CPU random weights test ragged suffixes and frozen prefix, no downloads."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(2)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=39,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            attn_implementation="eager",
        )
    ).eval()
    observer = type("Observer", (), {"model": model})()
    prefix, suffixes, target = [2, 3, 4, 5], [[6, 7, 8], [9]], [11, 12, 13]
    cache = M.encode_context(observer, prefix)
    before = copy.deepcopy(M.cache_arrays(cache))
    result = M.observe_suffixes(observer, cache, suffixes, target)
    for i, suffix in enumerate(suffixes):
        with torch.inference_mode():
            logits = model(
                torch.tensor([prefix + suffix + target[:-1]]), use_cache=False
            ).logits[0, len(prefix + suffix) - 1 :]
        expected = (
            logits.gather(1, torch.tensor(target)[:, None])[:, 0].detach().numpy()
        )
        assert np.max(abs(expected - result["chosen"][i])) < 1e-6
        assert (
            np.max(
                abs(
                    torch.logsumexp(logits, -1).detach().numpy()
                    - result["partition"][i]
                )
            )
            < 1e-6
        )
    for key, value in M.cache_arrays(cache).items():
        assert np.array_equal(value, before[key])
    repeated = M.observe_suffixes(observer, cache, suffixes[::-1], target)
    assert np.max(abs(repeated["chosen"] - result["chosen"][::-1])) < 1e-6
