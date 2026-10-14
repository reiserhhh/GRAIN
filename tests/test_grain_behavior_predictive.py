#!/usr/bin/env python3
"""Behavior-key identities and native/cache ragged completion tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_behavior_predictive as M
from grain_research.construction import math_late_condition as L

ROOT = Path(__file__).resolve().parents[1]


def bank():
    """Read only the public licensed item inventory, not study questionnaires."""
    return M.item_bank(
        json.loads((ROOT / "configs/grain_behavior_anchor_hm251.json").read_text())
    )


def test_complete_keys_common_gain_and_identity():
    """Common support is retained and can contaminate imbalanced equal-item keys."""
    rows = bank()
    result = M.key_accounts(np.full((3, 50), 2.0), rows)
    assert np.max(abs(result["balanced_key"])) == 0
    assert np.max(abs(result["common_support"] - 2)) == 0
    assert result["equal_item_key"][0, 1] == pytest.approx(1.2)
    rng = np.random.default_rng(1)
    result = M.key_accounts(rng.normal(size=(7, 50)), rows)
    assert (
        np.max(
            abs(
                result["equal_item_key"]
                - result["balanced_key"]
                - result["imbalance"] * result["common_support"]
            )
        )
        < 1e-14
    )


def test_cold_same_evidence_and_logit_gauge():
    """No same-evidence effect; common token logit origins do not change readings."""
    rows = bank()
    z = np.full((48, 2, 50), -8.0)
    a = np.full_like(z, 2.0)
    packet = {
        "version": "HM261-v1",
        "chosen": z.tolist(),
        "partition": a.tolist(),
        "lengths": [3] * 50,
        "bank": rows,
    }
    first = M.cold_score(packet)
    assert np.max(abs(np.asarray(first["item_delta"]))) == 0
    gauge = np.random.default_rng(3).normal(size=z.shape)
    packet.update(chosen=(z + gauge).tolist(), partition=(a + gauge).tolist())
    assert np.max(abs(np.asarray(M.cold_score(packet)["item_delta"]))) < 1e-14


@pytest.mark.parametrize("values", [np.zeros((48, 49)), np.full((48, 50), np.nan)])
def test_partial_or_nonfinite_not_zero(values):
    """Missing behavioral candidates cannot silently become neutral readings."""
    with pytest.raises(ValueError):
        M.key_accounts(values, bank())


def test_pinned_neutral_prefix_and_complete_end_token():
    """The same prefix has no named trait or persona pole; complete targets end."""
    from transformers import AutoTokenizer

    from scripts.run_grain_joint_persona_style_hm150 import MODEL

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    context = tok.encode("A short archived comment.", add_special_tokens=False)
    prefix, suffix = M.observation_parts(tok, context)
    assert prefix[-len(context) :] == context
    decoded = tok.decode(prefix[: -len(context)] + suffix).lower()
    assert all(term not in decoded for term in M.TARGETS)
    assert "hypothetical voice" not in decoded
    end = tok.convert_tokens_to_ids("<|im_end|>")
    targets = [
        tok.encode(row["completion"], add_special_tokens=False) + [end]
        for row in bank()
    ]
    assert len(targets) == 50 and all(t[-1] == end for t in targets)
    assert len({tuple(t) for t in targets}) == 50


def test_tiny_ragged_native_cache_no_mutation():
    """Random CPU decoder checks first-word address, EOS, left padding and order."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(4)
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
    prefix, suffix, targets = [2, 3, 4], [5, 6], [[8, 9, 1], [10, 1], [11, 12, 13, 1]]
    cache = L.encode_context(observer, prefix)
    before = L.cache_arrays(cache)
    result = M.observe_candidates(observer, cache, suffix, targets)
    for target, row in zip(targets, result, strict=True):
        with torch.inference_mode():
            logits = (
                model(torch.tensor([prefix + suffix + target[:-1]]), use_cache=False)
                .logits[0, len(prefix + suffix) - 1 :]
                .double()
            )
        expected = logits.gather(1, torch.tensor(target)[:, None])[:, 0].numpy()
        assert np.max(abs(expected - row["chosen"])) < 1e-6
        assert (
            np.max(abs(torch.logsumexp(logits, -1).numpy() - row["partition"])) < 1e-6
        )
        assert len(row["states"]) == len(target)
    assert all(np.array_equal(v, L.cache_arrays(cache)[k]) for k, v in before.items())
    reverse = M.observe_candidates(observer, cache, suffix, targets[::-1])
    assert all(
        np.max(abs(a["chosen"] - b["chosen"])) < 1e-6
        for a, b in zip(result, reverse[::-1], strict=True)
    )
