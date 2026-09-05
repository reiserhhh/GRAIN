"""Synthetic tests for observer mechanics; no downloads or private records."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.analysis import (
    factorial_interactions,
    grouped_bootstrap,
)
from grain_research.vector_mechanisms.controls import build_controls, validate_controls
from grain_research.vector_mechanisms.inventory import inspect_csv


def test_controls_have_disjoint_scenario_splits():
    rows = build_controls()
    info = validate_controls(rows)
    assert info["cases"] == 192
    assert info["independent_scenarios"] == 12
    assert {r["group_id"] for r in rows if r["split"] == "discovery"}.isdisjoint(
        r["group_id"] for r in rows if r["split"] != "discovery"
    )


def test_controls_reject_pseudoreplicated_splits():
    rows = build_controls()
    rows[0]["split"] = "heldout_scenario"
    with pytest.raises(ValueError, match="crosses"):
        validate_controls(rows)


def test_cluster_bootstrap_does_not_count_variants_as_subjects():
    result = grouped_bootstrap([1, 1, 1, 0], ["a", "a", "a", "b"])
    assert result["clusters"] == 2
    assert result["mean"] == 0.5


def test_interaction_additive_and_nonlinear():
    frame = pd.DataFrame(
        {
            "family": ["composition"] * 4,
            "group_id": ["a"] * 4,
            "split": ["discovery"] * 4,
            "variant": range(4),
            "margin": [0, 2, 3, 5],
        }
    )
    assert factorial_interactions(frame).iloc[0].logodds_interaction == 0
    frame.loc[3, "margin"] = 7
    assert factorial_interactions(frame).iloc[0].logodds_interaction == 2


def test_inventory_missing_identifier_is_error(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("text,label\nhello,1\n")
    with pytest.raises(ValueError, match="missing required ID"):
        inspect_csv(path, id_field="user_id")


class TinyTokenizer:
    chat_template = "synthetic"

    def encode(self, text, **_kwargs):
        return {"Yes": [3], "No": [4], "two tokens": [5, 6]}[text]


@pytest.fixture
def observer():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from grain_research.vector_mechanisms.trace import DecoderObserver

    torch.manual_seed(731)
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    return DecoderObserver(model, TinyTokenizer(), max_tokens=64)


def test_hook_pass_through_and_recovery(observer):
    from grain_research.vector_mechanisms.trace import Patch

    ids = [1, 2, 7]
    baseline = observer._forward(ids).logits.detach().numpy()[0, -1]
    states, logits = observer.capture(ids)
    np.testing.assert_allclose(logits, baseline, atol=1e-6)
    _, identity = observer.capture(
        ids, patch=Patch("block_00", 2, states["block_00"][0])
    )
    np.testing.assert_allclose(identity, logits, atol=1e-6)
    observer.capture(ids, patch=Patch("block_00", 2, np.ones(16)))
    _, restored = observer.capture(ids)
    np.testing.assert_allclose(restored, logits, atol=1e-6)
    assert not observer.sites["block_00"]._forward_hooks


def test_input_embedding_is_context_invariant(observer):
    first, _ = observer.capture([1, 2, 7])
    second, _ = observer.capture([2, 1, 7])
    np.testing.assert_array_equal(first["input_embedding"], second["input_embedding"])


def test_chat_template_batch_encoding_is_unwrapped(observer):
    from transformers import BatchEncoding

    observer.tokenizer.apply_chat_template = lambda *a, **k: BatchEncoding(
        {"input_ids": [1, 2, 7]}
    )
    assert observer.prompt_ids("system", "user") == [1, 2, 7]


def test_chat_template_malformed_tokens_fail_early(observer):
    observer.tokenizer.apply_chat_template = lambda *a, **k: ["input_ids"]
    with pytest.raises(ValueError, match="flat integer"):
        observer.prompt_ids("system", "user")


def test_content_position_precedes_future_question(observer):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "alpha": 1, "beta": 2, "Question": 3}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    observer.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]"
    )
    observer.tokenizer.chat_template = "{{ messages[1]['content'] }}"
    assert observer.content_position("", "alpha beta Question", "alpha beta") == 1
    with pytest.raises(ValueError, match="exactly once"):
        observer.content_position("", "alpha alpha Question", "alpha")


def test_generated_answer_does_not_contaminate_prompt_state(observer):
    before, _ = observer.capture([1, 2, 7])
    after, _ = observer.capture([1, 2, 7, 3, 4], positions=[2])
    for site in before:
        np.testing.assert_allclose(before[site], after[site], atol=1e-6)


def test_complete_candidate_likelihood(observer):
    import torch

    output = observer._forward([1, 2, 7, 5, 6])
    expected = (
        torch.log_softmax(output.logits[0, 2:4].float(), dim=-1)[[0, 1], [5, 6]]
        .sum()
        .item()
    )
    assert observer.score([1, 2, 7], "two tokens") == pytest.approx(expected, abs=1e-6)


def test_single_token_readout_uses_identical_prefill(observer):
    import torch

    from grain_research.vector_mechanisms.trace import Patch

    ids = [1, 2, 7]
    states, logits = observer.capture(ids)
    score = observer.score(ids, "Yes")
    expected = torch.log_softmax(torch.as_tensor(logits), dim=-1)[3].item()
    assert score == pytest.approx(expected, abs=1e-6)
    self_score = observer.score(
        ids, "Yes", patch=Patch("block_00", 2, states["block_00"][0])
    )
    assert self_score == pytest.approx(score, abs=1e-6)


def test_yes_no_logodds_is_an_unembedding_projection(observer):
    import torch

    states, logits = observer.capture([1, 2, 7])
    head = observer.model.get_output_embeddings()
    direction = (head.weight[3] - head.weight[4]).detach().float().numpy()
    bias = float((head.bias[3] - head.bias[4]).item()) if head.bias is not None else 0.0
    reconstructed = states["final_norm"][0] @ direction + bias
    assert reconstructed == pytest.approx(float(logits[3] - logits[4]), abs=1e-6)
    logp = torch.log_softmax(torch.as_tensor(logits), dim=-1)
    assert float(logp[3] - logp[4]) == pytest.approx(reconstructed, abs=1e-6)


def test_subspace_algebra_and_validation():
    from grain_research.vector_mechanisms.trace import patch_delta

    current, donor = np.zeros(3), np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_equal(patch_delta(current, donor, np.eye(3)), donor)
    np.testing.assert_array_equal(patch_delta(donor, donor, np.eye(3)), current)
    np.testing.assert_array_equal(
        patch_delta(current, donor, np.eye(3)[:, :1]), [2, 0, 0]
    )
    with pytest.raises(ValueError, match="orthonormal"):
        patch_delta(current, donor, np.ones((3, 1)))


def test_invalid_site_and_hook_exception_cleanup(observer):
    from grain_research.vector_mechanisms.trace import Patch

    with pytest.raises(ValueError, match="Unknown patch site"):
        observer.capture([1, 2], patch=Patch("missing", 1, np.zeros(16)))
    with pytest.raises(ValueError, match="outside sequence"):
        observer.capture([1, 2], patch=Patch("block_00", 4, np.zeros(16)))
    assert not observer.sites["block_00"]._forward_hooks


def test_alpha_zero_does_not_move_logits(observer):
    from grain_research.vector_mechanisms.trace import Patch

    _, base = observer.capture([1, 2, 7])
    _, changed = observer.capture(
        [1, 2, 7], patch=Patch("block_00", 2, np.ones(16), alpha=0)
    )
    np.testing.assert_allclose(base, changed, atol=1e-6)


def test_masked_static_padding_preserves_readout_and_prefix(observer):
    states, logits = observer.capture([1, 2, 7])
    margin = observer.margin([1, 2, 7])
    observer.pad_to_length = 16
    padded, padded_logits = observer.capture([1, 2, 7])
    np.testing.assert_allclose(logits, padded_logits, atol=1e-6)
    assert observer.margin([1, 2, 7]) == pytest.approx(margin, abs=1e-6)
    extended, _ = observer.capture([1, 2, 7, 8, 9], positions=[2])
    for site in states:
        np.testing.assert_allclose(padded[site], extended[site], atol=1e-6)
    with pytest.raises(ValueError, match="fixed execution length"):
        observer.capture([1] * 17)
    from grain_research.vector_mechanisms.trace import Patch

    with pytest.raises(ValueError, match="outside sequence"):
        observer.capture([1, 2, 7], patch=Patch("block_00", 10, np.zeros(16)))


def test_contrast_basis_preserves_shared_mean_direction():
    from grain_research.vector_mechanisms.interventions import contrast_basis

    differences = np.array([[10.0, 0.1, 0.0], [10.0, -0.1, 0.0], [10.0, 0.0, 0.1]])
    basis = contrast_basis(differences, 1)
    assert abs(basis[0, 0]) > 0.99
    np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1e-6)
    with pytest.raises(ValueError, match="numerical"):
        contrast_basis(np.zeros((4, 3)), 2)


def test_random_intervention_matches_candidate_norm():
    from grain_research.vector_mechanisms.interventions import matched_random_effect

    delta, candidate = np.arange(1.0, 21.0), np.ones(20)
    first = matched_random_effect(delta, candidate, 2, "fixed")
    second = matched_random_effect(delta, candidate, 2, "fixed")
    np.testing.assert_array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(np.linalg.norm(candidate))
    assert not np.allclose(first, candidate)


def test_threshold_can_fix_output_offset_without_changing_rank():
    from grain_research.vector_mechanisms.interventions import fit_margin_threshold

    threshold = fit_margin_threshold([1, 2, 5, 6], [0, 0, 1, 1])
    assert 2 < threshold < 5
    with pytest.raises(ValueError, match="both classes"):
        fit_margin_threshold([1, 2], [0, 0])


def test_context_bundle_has_real_temporal_and_thread_separation():
    from grain_research.vector_mechanisms.data_views import (
        select_context_bundle,
        validate_context_bundle,
    )

    comments = [
        {
            "author_id": "a",
            "comment_id": str(i),
            "thread_id": f"t{i}",
            "timestamp": i,
            "body": f"Event number {i}",
        }
        for i in range(8)
    ]
    bundle = select_context_bundle(comments)
    assert [r["timestamp"] for r in bundle["history"]] == [3, 4, 5]
    assert [r["timestamp"] for r in bundle["current"]] == [6, 7]
    bundle["history"][0]["thread_id"] = bundle["current"][0]["thread_id"]
    with pytest.raises(ValueError, match="overlap"):
        validate_context_bundle(bundle)


def test_context_bundle_rejects_missing_author_and_insufficient_threads():
    from grain_research.vector_mechanisms.data_views import select_context_bundle

    with pytest.raises(ValueError, match="author ID"):
        select_context_bundle([{"body": "text"}])
    comments = [
        {
            "author_id": "a",
            "comment_id": str(i),
            "thread_id": "same",
            "timestamp": i,
            "body": f"Event number {i}",
        }
        for i in range(8)
    ]
    assert select_context_bundle(comments) is None
