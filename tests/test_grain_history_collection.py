#!/usr/bin/env python3
"""History production controls and readout algebra, without private data or downloads."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.context import QUESTIONS
from grain_research.vector_mechanisms.external_material import (
    ARMS,
    pack_features,
    unpack_features,
)
from grain_research.vector_mechanisms.history_collection import (
    KINDS,
    SITES,
    collect_history_author,
    history_channel,
    qualification_packet,
)
from grain_research.vector_mechanisms.source_history import aligned_triplet
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.collect_grain_source_history import replay_qualification, require_technical


def mini_observer():
    """Random small decoder with the same named hooks, not a real-checkpoint qualification."""
    torch.manual_seed(731)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    ).eval()
    model.config._attn_implementation = "eager"
    tok = SimpleNamespace(
        pad_token_id=0, encode=lambda text, **kw: [21 if text == "Yes" else 22]
    )
    obs = DecoderObserver(model, tok, max_tokens=64, pad_to_length=48)
    obs.sites = {
        "input_embedding": model.model.embed_tokens,
        "block_04": model.model.layers[0],
        "block_10": model.model.layers[1],
        "block_16": model.model.layers[2],
        "final_norm": model.model.norm,
    }
    return obs


def toy_layouts():
    """Three views and late questions; identical text arms permit lossless reuse."""
    layouts = {}
    for vi in range(3):
        for arm in ARMS:
            for ti, task in enumerate(QUESTIONS):
                own = {
                    "prefix": [2, 3],
                    "history": [4, 5, 6],
                    "suffix": [7 + vi, 8 + vi, 29 + ti, 10],
                    "current_in_suffix": [0, 1],
                }
                other = own | {"history": [11, 12, 13, 14, 15]}
                for h, row in aligned_triplet(own, other, 0, max_length=48).items():
                    layouts[vi, arm, task, h] = row
    return layouts


def test_production_capture_reuses_only_identical_layout_and_preserves_arrays():
    obs = mini_observer()
    layouts = toy_layouts()
    arrays, meta = collect_history_author(layouts, obs, repeat=True)
    assert len(arrays) == 90 and meta["logical_layouts"] == 54
    assert meta["unique_layouts"] == 18 and meta["forwards"] == 19
    assert (
        meta["causal_prefix_max"]
        == meta["input_embedding_max"]
        == meta["replay_max"]
        == 0
    )
    stored, aliases = pack_features(arrays)
    restored = unpack_features(stored, aliases)
    assert aliases
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], restored[key])
    for site in SITES:
        for kind in KINDS:
            assert history_channel(restored, "raw", site, kind).shape == (32,)
    np.testing.assert_array_equal(
        history_channel(arrays, "raw", "input_embedding", "current_context"),
        np.zeros(32),
    )
    assert (
        np.linalg.norm(history_channel(arrays, "raw", "block_10", "current_context"))
        > 0
    )
    bad = copy.deepcopy(layouts)
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError, match="Incomplete"):
        collect_history_author(bad, obs)
    with pytest.raises(TimeoutError):
        collect_history_author(layouts, obs, deadline=0)


def test_readout_is_variation_of_view_mean_effect_not_token_variance_subtraction():
    features = {}
    for arm in ARMS:
        for task in QUESTIONS:
            for hi, h in enumerate(("own", "matched", "invisible")):
                for site in SITES:
                    x = np.zeros((3, 3, 2))
                    x[:, 1] = np.array([[1, 4], [2, 3], [6, 9]]) * (1 + hi)
                    x[:, 2] = 1000 + hi
                    x[:, 0] = (1 if task == "behavior" else 2) * x[:, 1]
                    features[f"{arm}__{task}__{h}__{site}"] = x
    effect = (
        features["raw__behavior__own__block_10"][:, 1]
        - features["raw__behavior__matched__block_10"][:, 1]
    )
    expected = np.r_[effect.mean(0), effect.std(0)]
    np.testing.assert_allclose(
        history_channel(features, "raw", "block_10", "current_context"), expected
    )
    changed = copy.deepcopy(features)
    for x in changed.values():
        x[:, 2] *= 10
    np.testing.assert_array_equal(
        history_channel(features, "raw", "block_10", "current_context"),
        history_channel(changed, "raw", "block_10", "current_context"),
    )
    np.testing.assert_allclose(
        history_channel(features, "raw", "block_10", "decision_task_interaction"),
        expected,
    )


def test_saved_qualification_comparison_rejects_any_moved_component():
    arrays, metrics = collect_history_author(toy_layouts(), mini_observer())
    metrics["candidate_tokens"] = {"21": "yes_logit", "22": "no_logit"}
    parent = {}
    for name, x in arrays.items():
        arm, task, h, site = name.split("__")
        for vi in range(3):
            parent[f"case0_view{vi}_{arm}_{task}_{h}_{site}"] = x[vi].copy()
    for r in metrics["readouts"]:
        x = np.zeros(64)
        x[21] = r["yes_logit"]
        x[22] = r["no_logit"]
        parent[
            f"case0_view{r['view']}_{r['text_arm']}_{r['task']}_{r['history_arm']}_logits"
        ] = x
    n, error = replay_qualification(arrays, metrics, 0, parent)
    assert n == 378 and error == 0
    parent[next(iter(parent))][0, 0] += 1
    with pytest.raises(ValueError, match="differs"):
        replay_qualification(arrays, metrics, 0, parent)


def test_token_preflight_cannot_satisfy_real_neural_gate(monkeypatch):
    from grain_research.vector_mechanisms import history_collection

    monkeypatch.setattr(
        history_collection,
        "verified_packet",
        lambda _: (
            {"protocol": "VM05_HISTORY_INSTRUMENT_V1", "preflight_only": True},
            {},
        ),
    )
    with pytest.raises(ValueError, match="real history qualification"):
        qualification_packet("unused", {}, {}, "unused")


def test_technical_model_signature_comes_from_verified_output(monkeypatch):
    from scripts import collect_grain_source_history as runner

    manifest = {
        "protocol": "VM05_HISTORY_COLLECTION_V1",
        "mode": "qualification_replay",
        "qualification_identity": "q",
        "material_identity": "m",
        "code_sha256": {"x": "sha"},
        "identity": "technical",
    }
    summary = {
        "authors": 6,
        "token_layouts_replayed": 324,
        "qualified_state_replay_comparison_groups": 2268,
        "qualified_state_replay_max": 0,
        "real_neural_collection": True,
        "labels_read": False,
        "bridge_outcomes_read": False,
        "model_signature": "fixed",
    }
    monkeypatch.setattr(runner, "verified_packet", lambda _: (manifest, summary))
    out = require_technical(
        "present", {"identity": "q"}, {"identity": "m"}, {"x": "sha"}
    )
    assert out["model_signature"] == "fixed" and out["identity"] == "technical"
    with pytest.raises(ValueError, match="Invalid"):
        require_technical(
            "present", {"identity": "q"}, {"identity": "m"}, {"x": "changed"}
        )
