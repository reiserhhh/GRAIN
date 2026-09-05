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


def test_dual_cka_agrees_with_primal_formula():
    from grain_research.vector_mechanisms.multicomment import linear_cka_dual

    rng = np.random.default_rng(14)
    x, y = rng.normal(size=(12, 3)), rng.normal(size=(12, 5))
    xc, yc = x - x.mean(0), y - y.mean(0)
    expected = np.linalg.norm(xc.T @ yc) ** 2 / (
        np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc)
    )
    assert linear_cka_dual(x, y) == pytest.approx(expected)
    assert linear_cka_dual(x, x) == pytest.approx(1.0)


def test_finite_population_quadratic_moment_matches_all_subsets():
    from itertools import combinations

    from grain_research.vector_mechanisms.multicomment import (
        finite_population_mean_radius,
    )

    h = np.array([[[1.0, 2.0], [3.0, 1.0], [-2.0, 4.0], [0.0, -1.0]]])
    center, metric = np.array([0.1, 0.5]), np.array([[2.0, 0.2], [0.2, 1.0]])
    for n in (1, 2, 3, 4):
        base, inflation = finite_population_mean_radius(h, n, center, metric)
        actual = []
        for indices in combinations(range(4), n):
            d = h[0, list(indices)].mean(0) - center
            actual.append(d @ metric @ d)
        assert base[0] + inflation[0] == pytest.approx(np.mean(actual))
    assert inflation[0] == 0


def test_explicit_intervals_handle_repeated_and_nested_quotations():
    from grain_research.vector_mechanisms.multicomment import interval_token_indices
    from scripts.run_grain_vector_mechanism_multicomment_qwen import render_evidence

    comments = [{"body": "same words"}, {"body": "I quoted: same words"}]
    user, _, spans = render_evidence(comments, "behavior")
    assert [user[a:b] for a, b in spans] == [c["body"] for c in comments]
    offsets = [(i, i + 1) for i in range(len(user))]
    indices = interval_token_indices(offsets, spans)
    assert len(indices) == sum(len(c["body"]) for c in comments)
    with pytest.raises(ValueError, match="Overlapping"):
        interval_token_indices(offsets, [(1, 5), (4, 8)])


def test_fast_permutation_matches_original_null():
    from grain_research.vector_mechanisms.comparison import shuffled_author_null
    from scripts.summarize_grain_vector_multicomment import fast_null

    values = np.random.default_rng(9).normal(size=(8, 3, 4))
    a = fast_null(values, draws=19, seed=17)
    b = shuffled_author_null(values, draws=19, seed=17)
    assert a == pytest.approx(b)


def test_common_groups_do_not_drop_or_duplicate_records():
    from grain_research.vector_mechanisms.multicomment import common_groups, pool_groups

    groups = common_groups(["a" * i for i in (5, 10, 15, 20)], [len], max_tokens=32)
    assert groups == [[0, 1], [2], [3]]
    assert np.allclose(
        pool_groups(np.array([[2, 4], [4, 6], [6, 8]]), groups, 4), [3.5, 5.5]
    )
    with pytest.raises(ValueError, match="Single comment"):
        common_groups(["a" * 33], [len], max_tokens=32)
    with pytest.raises(ValueError, match="exactly once"):
        pool_groups(np.ones((2, 2)), [[0, 1], [1, 2]], 3)


def test_view_layout_has_identical_material_and_distinct_exposure():
    from grain_research.vector_mechanisms.multicomment import view_layout

    bundle = {"views": [list(range(i * 12, (i + 1) * 12)) for i in range(3)]}
    c, i = view_layout(bundle, "chronological"), view_layout(bundle, "interleaved")
    assert c != i
    assert (
        sorted(x for g in c for x in g)
        == sorted(x for g in i for x in g)
        == list(range(36))
    )


def test_multicomment_views_are_disjoint_and_label_blind():
    from grain_research.vector_mechanisms.cohort import select_multicomment_bundle
    from grain_research.vector_mechanisms.data_views import normalized_text_key

    rows = [
        {
            "author_id": "a",
            "comment_id": str(i),
            "thread_id": str(i),
            "timestamp": i + 1,
            "body": f"distinct words {i}",
        }
        for i in range(50)
    ]
    a = select_multicomment_bundle(rows)
    b = select_multicomment_bundle(list(reversed(rows)))
    assert a == b
    assert [len(v) for v in a["views"]] == [12, 12, 12]
    assert len({c["thread_id"] for v in a["views"] for c in v}) == 36
    forbidden = [normalized_text_key(c["body"]) for c in rows[:20]]
    assert select_multicomment_bundle(rows, forbidden_texts=forbidden) is None
    assert select_multicomment_bundle(rows[:35]) is None


def test_multicomment_views_reject_bad_author_or_thread():
    from grain_research.vector_mechanisms.cohort import (
        select_multicomment_bundle,
        validate_multicomment_bundle,
    )

    rows = [
        {
            "author_id": "a",
            "comment_id": str(i),
            "thread_id": str(i),
            "timestamp": i + 1,
            "body": f"unique item {i}",
        }
        for i in range(36)
    ]
    b = select_multicomment_bundle(rows)
    b["views"][2][0]["thread_id"] = b["views"][0][0]["thread_id"]
    with pytest.raises(ValueError, match="thread_id"):
        validate_multicomment_bundle(b, units=12)
    rows[0]["author_id"] = "b"
    with pytest.raises(ValueError, match="mix authors"):
        select_multicomment_bundle(rows)


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
    pytest.importorskip("torch")
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


def test_pooled_capture_is_exact_and_cannot_read_padding(observer):
    ids = [1, 2, 7, 8]
    full, _ = observer.capture(ids, positions=list(range(4)))
    pooled, _ = observer.capture(
        ids, positions=[3], pools=[("mean", [1, 2]), ("std", [1, 2])]
    )
    for site in full:
        np.testing.assert_allclose(
            pooled[site][1], full[site][[1, 2]].mean(axis=0), atol=1e-6
        )
        np.testing.assert_allclose(
            pooled[site][2], full[site][[1, 2]].std(axis=0), atol=1e-6
        )
    with pytest.raises(ValueError, match="Pool positions"):
        observer.capture(ids, pools=[("mean", [4])])


def test_context_tasks_have_identical_evidence_prefix():
    from grain_research.vector_mechanisms.context import render_context

    a, anchor_a = render_context("prior event", "current event", "behavior")
    b, anchor_b = render_context("prior event", "current event", "disposition")
    assert a.split("Question:")[0] == b.split("Question:")[0]
    assert anchor_a == anchor_b


def test_donor_matching_excludes_future_self_overlap_and_split_crossing():
    from grain_research.vector_mechanisms.context import match_donors

    def bundle(uid, start):
        comments = [
            {
                "author_id": uid,
                "comment_id": f"{uid}{i}",
                "thread_id": f"t{uid}{i}",
                "timestamp": start + i,
                "body": f"{uid} event {i}",
                "subreddit": "common",
            }
            for i in range(5)
        ]
        return {
            "author_id": uid,
            "split": "dev",
            "history": comments[:3],
            "current": comments[3:],
        }

    a, b, future = bundle("a", 100), bundle("b", 99), bundle("f", 1000)
    matches = match_donors([a, b, future], len)
    assert {r.get("donor_id") for r in matches if r["author_id"] == "a"} == {"b"}
    b["history"][0]["thread_id"] = a["current"][0]["thread_id"]
    matches = match_donors([a, b, future], len)
    assert all(
        r["status"] == "NO_ELIGIBLE_PRIOR_DONOR"
        for r in matches
        if r["author_id"] == "a"
    )
    future["split"] = "test"
    with pytest.raises(ValueError, match="crosses"):
        match_donors([a, future], len)


def test_author_rank_is_half_for_shared_input_and_one_for_identity():
    from grain_research.vector_mechanisms.context import paired_author_rank

    assert (
        paired_author_rank(np.ones((4, 8)), np.ones((4, 8)))["author_rank_auc"] == 0.5
    )
    assert paired_author_rank(np.eye(4), np.eye(4))["author_rank_auc"] == 1


def test_anchored_partition_reconstructs_additive_and_interaction_responses():
    from grain_research.vector_mechanisms.context import anchored_decomposition

    ref, context, current = np.ones(3), np.arange(3.0), np.array([2.0, 3.0, 4.0])
    additive = ref + context + current
    parts = anchored_decomposition(additive, ref + context, ref + current, ref)
    np.testing.assert_allclose(parts["interaction"], 0)
    full = additive + context * current
    parts = anchored_decomposition(full, ref + context, ref + current, ref)
    np.testing.assert_allclose(parts["interaction"], context * current)
    np.testing.assert_allclose(sum(parts.values()), full)
    with pytest.raises(ValueError, match="identical shapes"):
        anchored_decomposition(full, [1.0, 2.0], current, ref)


def test_node_rank_matches_original_directed_view_statistic():
    from grain_research.vector_mechanisms.comparison import (
        pair_concordance,
        weighted_rank,
    )
    from grain_research.vector_mechanisms.context import paired_author_rank

    values = np.random.default_rng(18).normal(size=(6, 3, 10))
    expected = np.mean(
        [
            paired_author_rank(values[:, a], values[:, b])["author_rank_auc"]
            for a, b in ((0, 1), (0, 2), (1, 2))
        ]
    )
    assert weighted_rank(pair_concordance(values), np.ones(6))[0] == pytest.approx(
        expected
    )


def test_node_bootstrap_never_counts_cloned_author_as_impostor():
    from grain_research.vector_mechanisms.comparison import (
        node_bootstrap_weights,
        pair_concordance,
        weighted_rank,
    )

    values = np.repeat(np.eye(4)[:, None, :], 3, axis=1)
    weights = node_bootstrap_weights(4, draws=20)
    np.testing.assert_allclose(weighted_rank(pair_concordance(values), weights), 1)
    np.testing.assert_allclose(
        weighted_rank(pair_concordance(np.ones((4, 3, 4))), weights), 0.5
    )
    with pytest.raises(ValueError, match="distinct author"):
        weighted_rank(pair_concordance(values), [4, 0, 0, 0])


def test_conditional_rank_excludes_self_and_reports_empty_support():
    from grain_research.vector_mechanisms.comparison import (
        conditional_rank,
        pair_concordance,
    )

    values = np.repeat(np.eye(4)[:, None, :], 3, axis=1)
    concordance = pair_concordance(values)
    np.testing.assert_allclose(
        conditional_rank(concordance, np.ones((3, 4, 4)), np.ones(4)), 1
    )
    assert np.isnan(
        conditional_rank(concordance, np.zeros((3, 4, 4)), np.ones(4))
    ).all()


def test_binding_quartets_have_same_words_but_opposite_relational_answers():
    import re
    from collections import Counter

    from grain_research.vector_mechanisms.binding import build_binding_controls

    rows = build_binding_controls()
    assert len(rows) == 288
    assert len({r["case_id"] for r in rows}) == 288
    groups = {}
    for row in rows:
        groups.setdefault((row["group_id"], row["order"], row["task"]), []).append(row)
    for quartet in groups.values():
        assert len(quartet) == 4
        counts = [Counter(re.findall(r"\w+", r["passage"])) for r in quartet]
        assert all(c == counts[0] for c in counts)
        assert sum(r["expected_yes"] for r in quartet) == 2


def test_binding_second_difference_is_zero_for_additive_records():
    from grain_research.vector_mechanisms.binding import factorial_interaction

    additive = {
        (a, b): np.array([1.0 + 2 * a + b, a - b]) for a in (0, 1) for b in (0, 1)
    }
    np.testing.assert_allclose(factorial_interaction(additive), 0)
    assert (
        factorial_interaction({(a, b): int(a == b) for a in (0, 1) for b in (0, 1)})
        == 2
    )
    with pytest.raises(ValueError, match="quartet"):
        factorial_interaction({(0, 0): 1})


def test_leakage_matching_uses_flag_values_not_nonempty_dictionary():
    from scripts.summarize_grain_vector_baselines import condition_masks

    authors = ["a", "b", "c"]
    rows = [
        {
            "author_id": u,
            "history_index": i,
            "word_count": 60,
            "subreddit": "source",
            "leakage_flags": {"clue": u == "c"},
        }
        for u in authors
        for i in range(3)
    ]
    vectors = {
        k: np.repeat(np.eye(3)[:, None, :], 3, axis=1)
        for k in ("bge_large", "e5_large")
    }
    masks = condition_masks(rows, authors, vectors)
    assert masks["no_clue_author"].sum() == 6


def test_additive_record_probe_cannot_solve_balanced_xnor():
    from scripts.summarize_grain_vector_binding import fit_probe

    rows = [(g, a, b) for g in range(4) for a in (0, 1) for b in (0, 1)]
    x = np.array([[a, b, 1, g] for g, a, b in rows], dtype=float)
    y = np.array([a == b for g, a, b in rows], dtype=int)
    train = np.array([g < 3 for g, a, b in rows])
    np.testing.assert_array_equal(fit_probe(x, y, train, ~train), 0)


def test_span_self_patch_and_full_sequence_donor_equivalence(observer):
    from grain_research.vector_mechanisms.trace import Patch

    ids, donor_ids = [1, 2, 7], [2, 1, 7]
    positions = [0, 1, 2]
    original, logits = observer.capture(ids, positions=positions)
    donor, donor_logits = observer.capture(donor_ids, positions=positions)
    _, self_logits = observer.capture(
        ids, patch=Patch("block_00", positions, original["block_00"])
    )
    np.testing.assert_allclose(self_logits, logits, atol=1e-6)
    _, changed = observer.capture(
        ids, patch=Patch("block_00", positions, donor["block_00"])
    )
    np.testing.assert_allclose(changed, donor_logits, atol=1e-6)
    _, restored = observer.capture(ids)
    np.testing.assert_allclose(restored, logits, atol=1e-6)


def test_span_patch_rejects_duplicate_and_broadcast_donors(observer):
    from grain_research.vector_mechanisms.trace import Patch

    with pytest.raises(ValueError, match="unique"):
        observer.capture([1, 2, 7], patch=Patch("block_00", [0, 0], np.zeros((2, 16))))
    with pytest.raises(ValueError, match="Donor shape"):
        observer.capture([1, 2, 7], patch=Patch("block_00", [0, 1], np.zeros(16)))
    assert not observer.sites["block_00"]._forward_hooks


def test_span_projected_patch_algebra(observer):
    from grain_research.vector_mechanisms.trace import Patch, patch_delta

    ids = [1, 2, 7]
    states, _ = observer.capture(ids, positions=[0, 1])
    donor = states["block_00"] + 1
    basis = np.eye(16)[:, :2]
    effect = patch_delta(states["block_00"], donor, basis)
    np.testing.assert_allclose(effect[:, 2:], 0)
    _, a = observer.capture(ids, patch=Patch("block_00", [0, 1], donor, basis=basis))
    _, b = observer.capture(
        ids, patch=Patch("block_00", [0, 1], states["block_00"] + effect)
    )
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_bfloat16_full_replacement_is_copy_not_subtract_add(observer):
    import torch

    from grain_research.vector_mechanisms.trace import Patch

    observer.model.to(torch.bfloat16)
    donor = np.full((3, 16), 0.0001)
    expected = torch.as_tensor(donor, dtype=torch.bfloat16).float().numpy()
    captured, _ = observer.capture(
        [1, 2, 7], positions=[0, 1, 2], patch=Patch("block_00", [0, 1, 2], donor)
    )
    np.testing.assert_array_equal(captured["block_00"], expected)


def test_relational_kernel_is_psd_and_adds_the_missing_interaction():
    from grain_research.vector_mechanisms.binding import relational_grams

    a = np.array([[-1.0], [-1.0], [1.0], [1.0]])
    b = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    truth = np.array([1.0, -1.0, -1.0, 1.0])
    kernels = relational_grams(a, b, np.ones(4, dtype=bool))
    for matrix in kernels.values():
        assert np.linalg.eigvalsh(matrix).min() > -1e-10
    np.testing.assert_allclose(kernels["additive"] @ truth, 0)
    assert (kernels["tensor_product"] @ truth > 0).tolist() == [
        True,
        False,
        False,
        True,
    ]
