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


def test_scope_controls_separate_performed_reported_and_quoted_actions():
    from grain_research.vector_mechanisms.scope import build_scope_controls

    rows = build_scope_controls()
    assert len(rows) == 288
    assert len({r["case_id"] for r in rows}) == 288
    for task in ("enacted_guidance", "reported_guidance"):
        own = [r for r in rows if r["task"] == task]
        assert sum(r["expected_yes"] for r in own) == 48
        assert all(
            r["expected_yes"] == 0 for r in own if r["variant"] == "other_quoted"
        )
    a = {r["group_id"] for r in rows if r["split"] == "scope_development"}
    b = {r["group_id"] for r in rows if r["split"] == "scope_holdout"}
    assert len(a) == 8 and len(b) == 4 and not a & b


def test_memory_summary_citation_is_not_semantic_acceptance():
    import json

    from grain_research.vector_mechanisms.memory import parse_summary, summary_text

    text = "I watched while my neighbor invited people to a long walk."
    raw = json.dumps(
        {"facts": [{"excerpt": 1, "fact": "The writer invited people.", "quote": text}]}
    )
    audit = parse_summary(raw, [{"body": text}])
    assert audit["facts"][0]["status"] == "EXACT_QUOTE_SUPPORTED"
    assert "No supported" in summary_text(audit, reviewed=True)
    audit["facts"][0]["semantic_review"] = "REJECT"
    assert "No supported" in summary_text(audit, reviewed=True)
    assert parse_summary("garbage", [])["status"] == "PARSE_FAILED"


def test_memory_controls_are_scoped_and_mixed_polarity():
    from grain_research.vector_mechanisms.memory import update_controls, update_message

    rows = update_controls()
    assert len(rows) == 8
    assert {r["expected_yes"] for r in rows} == {0, 1}
    for r in rows:
        for placement in ("before", "after"):
            text = update_message(
                "unrelated natural writing", r["text"], placement=placement
            )
            assert "fabricated experimental event" in text
            assert text.count("unrelated natural writing") == 1
            assert "final account of Record Z7" in text
            statement = update_message(
                "background",
                r["text"],
                placement=placement,
                negative_question=True,
                question_style="statement",
            )
            assert "Is that statement supported" in statement
            assert "the writer did not personally initiate" in statement


def test_summary_repair_does_not_hide_invalid_generation_or_excess_facts():
    import json

    from grain_research.vector_mechanisms.memory import recover_summary

    fact = {
        "excerpt": 1,
        "fact": "A report.",
        "quote": "One two three four five six seven eight.",
    }
    raw = json.dumps({"facts": [fact]})[:-1]
    repaired = recover_summary(raw, [{"body": fact["quote"]}])
    assert repaired["strict_status"] == "PARSE_FAILED"
    assert repaired["repair"] == "APPEND_ONE_CLOSING_OBJECT_BRACE"
    assert repaired["facts"][0]["semantic_review"] == "PENDING"
    excess = recover_summary(
        json.dumps({"facts": [fact] * 7})[:-1], [{"body": fact["quote"]}]
    )
    assert excess["status"] == "SCHEMA_FAILED"
    assert excess["facts"] == []


def test_memory_current_has_identical_right_boundary_in_every_placement():
    from grain_research.vector_mechanisms.memory import (
        CURRENT_END,
        natural_message,
        update_message,
    )

    current = "An emoticon at the boundary :("
    prompts = [natural_message(None, current, "behavior")]
    prompts += [
        update_message(current, "Record Z7: test.", placement=p)
        for p in ("before", "after")
    ]
    for prompt in prompts:
        assert current + CURRENT_END in prompt


def test_memory_material_keeps_natural_and_fabricated_truth_separate():
    from scripts.run_grain_vector_mechanism_memory import make_material

    history = [{"body": f"Evidence number {i}."} for i in range(12)]
    case = {
        "case_id": "c",
        "author_id": "a",
        "current_index": 0,
        "history": history,
        "current": {"body": "Unchanged current."},
        "retrieval": {
            task: {"indices": [0, 4, 8]} for task in ("behavior", "disposition")
        },
        "random3": [1, 5, 9],
        "donors": {
            key: {"status": "MATCHED", "history": history}
            for key in ("other_length", "other_condition")
        },
    }
    rows = make_material([case], {"a": {"audit": {"facts": []}}})
    natural = [r for r in rows if r["kind"] == "natural"]
    controlled = [r for r in rows if r["kind"] == "synthetic_on_natural"]
    assert len(natural) == 26
    assert len(controlled) == 32
    assert all(r["expected_yes"] is None for r in natural)
    assert sum(r["expected_yes"] for r in controlled) == 16
    for variant in {r["arm"] for r in controlled}:
        for placement in ("before", "after"):
            pair = [
                r
                for r in controlled
                if r["arm"] == variant and r["placement"] == placement
            ]
            assert sum(r["expected_yes"] for r in pair) == 1


def test_memory_analysis_pairs_features_by_integer_row_not_case_string(
    tmp_path, monkeypatch
):
    import json

    from grain_research.vector_mechanisms.controls import material_hash
    from scripts.run_grain_vector_mechanism_memory import make_material
    from scripts.summarize_grain_vector_memory import main

    history = [{"body": f"Some unchanging words in excerpt {i}."} for i in range(12)]
    review = {
        "reviews": [
            {
                "author_id": uid,
                "audit": {
                    "strict_status": "PARSED",
                    "status": "PARSED",
                    "repair": "NONE",
                    "facts": [
                        {
                            "fact_id": 0,
                            "status": "EXACT_QUOTE_SUPPORTED",
                            "semantic_review": "ACCEPT",
                            "review_reason": "Synthetic fixture.",
                            "fact": "One claim.",
                            "quote": history[0]["body"],
                            "excerpt": 1,
                        }
                    ],
                },
            }
            for uid in ("a", "b")
        ]
    }
    cases = [
        {
            "case_id": "case_" + uid,
            "author_id": uid,
            "current_index": 0,
            "history": history,
            "current": {"body": "Same current."},
            "retrieval": {
                task: {"indices": [0, 1, 2]} for task in ("behavior", "disposition")
            },
            "random3": [3, 4, 5],
            "donors": {
                key: {"status": "MATCHED", "history": history}
                for key in ("other_length", "other_condition")
            },
        }
        for uid in ("a", "b")
    ]
    rows = make_material(cases, {r["author_id"]: r for r in review["reviews"]})
    for i, r in enumerate(rows):
        r.update(
            job_id=f"j{i:04d}",
            margin=float(i % 5 - 2),
            conditional_yes=0.5,
            predicted_yes=int(i % 5 > 2),
            prompt_tokens=100,
            yes_no_probability_mass=0.9,
            current_tokens=10,
        )
        r["correct"] = (
            int(r["predicted_yes"] == r["expected_yes"])
            if r["expected_yes"] is not None
            else None
        )
    parent = tmp_path / "parent"
    parent.mkdir()
    pd.DataFrame(rows).to_csv(parent / "readouts.csv", index=False)
    (parent / "run_status.json").write_text(
        json.dumps({"status": "COMPLETE", "cases": len(rows)})
    )
    (parent / "manifest.json").write_text(
        json.dumps(
            {
                "identity": "test",
                "review_sha256": material_hash(review),
                "jobs": [{"job_id": r["job_id"]} for r in rows],
            }
        )
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    features = np.random.default_rng(1).normal(size=(len(rows), 4))
    np.savez(
        parent / "representations.npz",
        block_10_current_mean=features,
        block_16_decision=features,
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summary",
            "--parent-dir",
            str(parent),
            "--review",
            str(review_path),
            "--output-dir",
            str(out),
        ],
    )
    main()
    result = pd.read_csv(out / "context_displacement.csv")
    assert (result[result.arm == "raw_full"].relative_displacement_mean == 0).all()
    assert (result[result.arm == "none"].relative_displacement_mean > 0).all()
    assert json.loads((out / "summary.json").read_text())["status"] == "COMPLETE"


def test_summary_review_cannot_change_a_fact_or_accept_bad_citation():
    import json

    from grain_research.vector_mechanisms.memory import (
        recover_summary,
        validate_summary_review,
    )

    comments = [{"body": "One two three four five six seven eight."}]
    raw = json.dumps(
        {"facts": [{"excerpt": 1, "fact": "A claim.", "quote": comments[0]["body"]}]}
    )
    generated = [{"author_id": "a", "raw": raw}]
    sources = [{"author_id": "a", "history": comments}]
    audit = recover_summary(raw, comments)
    audit["facts"][0]["semantic_review"] = "ACCEPT"
    reviews = [{"author_id": "a", "audit": audit}]
    assert validate_summary_review(generated, reviews, sources)["a"]["audit"] == audit
    audit["facts"][0]["fact"] = "A different claim."
    with pytest.raises(ValueError, match="changed generated"):
        validate_summary_review(generated, reviews, sources)


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


def test_natural_patch_basis_and_row_matched_random():
    from grain_research.vector_mechanisms.natural_patch import (
        current_difference_basis,
        matched_random_rows,
        project_rows,
    )

    delta = np.random.default_rng(72).normal(size=(9, 16))
    q, energy = current_difference_basis(delta, 4)
    projected = project_rows(delta, q)
    assert 0 < energy <= 1
    np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(projected) ** 2 / np.linalg.norm(delta) ** 2, energy
    )
    random = matched_random_rows(delta, projected, 4, "fixed")
    np.testing.assert_allclose(
        np.linalg.norm(random, axis=1), np.linalg.norm(projected, axis=1)
    )
    np.testing.assert_array_equal(
        random, matched_random_rows(delta, projected, 4, "fixed")
    )
    assert np.linalg.matrix_rank(random, tol=1e-8) <= 4
    with pytest.raises(ValueError, match="numerical rank"):
        current_difference_basis(np.zeros((8, 16)), 2)


def test_natural_patch_prompt_preserves_evidence_and_explicit_interval():
    from grain_research.vector_mechanisms.natural_patch import patch_message

    current = "Earlier excerpt: I quoted that phrase. This is my current message :("
    a, ia = patch_message("Older words.", current, "disposition")
    b, ib = patch_message("Older words.", current, "current_guidance")
    assert ia == ib
    assert a[slice(*ia)] == b[slice(*ib)] == current
    assert a.split("Question: ")[0] == b.split("Question: ")[0]
    c, ic = patch_message("A longer historical account.", current, "disposition")
    assert c[slice(*ic)] == current


def test_source_ambiguity_bounds_do_not_select_favorable_codes():
    from scripts.summarize_grain_vector_scope import ambiguity_bounds

    sample = pd.DataFrame({"expected_yes": [0, 1, np.nan], "margin": [-1, 1, 2]})
    bounds = ambiguity_bounds(sample, 0)
    assert bounds["assignments"] == 2
    assert bounds["accuracy_min"] == pytest.approx(2 / 3)
    assert bounds["accuracy_max"] == 1
    assert bounds["raw_margin_auc_min"] == 0.5
    assert bounds["raw_margin_auc_max"] == 1


def test_natural_patch_summary_weights_authors_not_forwards():
    from scripts.summarize_grain_vector_natural_patch import aggregate

    sample = pd.DataFrame(
        {
            "task": ["a"] * 5,
            "author_id": ["u"] * 4 + ["v"],
            "delta": [1] * 4 + [0],
        }
    )
    result = aggregate(sample, ["task"], ["delta"])
    assert result.iloc[0]["delta_mean"] == 0.5
    assert result.iloc[0]["delta_clusters"] == 2


def test_quantized_norm_match_reports_discrete_limits():
    from grain_research.vector_mechanisms.natural_patch import match_quantized_row_norms

    h = np.zeros((2, 2))
    direction = np.array([[0.2, 0.0], [0.1, 0.0]])
    target = np.array([[0.0, 1.0], [1.0, 1.0]])
    value, scale, error = match_quantized_row_norms(h, direction, target, np.round)
    assert error[0] == 0
    assert error[1] > 0.1
    assert np.array_equal(value[:, 1], np.zeros(2))
    assert np.all(scale > 0)
    with pytest.raises(ValueError, match="zero direction"):
        match_quantized_row_norms(h, np.zeros_like(h), target, np.round)


def test_local_margin_derivative_matches_central_difference_and_restores(observer):
    from grain_research.vector_mechanisms.sensitivity import (
        local_margin_gradient,
        local_prediction,
    )
    from grain_research.vector_mechanisms.trace import Patch

    ids = [1, 2, 7]
    flags = [p.requires_grad for p in observer.model.parameters()]
    before, logits = observer.capture(ids, positions=[0, 1, 2])
    state, grad, margin = local_margin_gradient(observer, ids, "block_00", 1, 2)
    np.testing.assert_allclose(state, before["block_00"], atol=1e-6)
    assert margin == pytest.approx(float(logits[1] - logits[2]), abs=1e-6)
    delta = np.random.default_rng(93).normal(size=state.shape)
    delta *= 1e-4 / np.linalg.norm(delta)
    _, plus = observer.capture(ids, patch=Patch("block_00", [0, 1, 2], state + delta))
    _, minus = observer.capture(ids, patch=Patch("block_00", [0, 1, 2], state - delta))
    actual = float((plus[1] - plus[2] - minus[1] + minus[2]) / 2)
    expected = local_prediction(grad, state, state + delta)
    assert actual == pytest.approx(expected, rel=0.02, abs=1e-7)
    assert flags == [p.requires_grad for p in observer.model.parameters()]
    assert not observer.sites["block_00"]._forward_hooks
    assert all(p.grad is None for p in observer.model.parameters())
    _, restored = observer.capture(ids)
    np.testing.assert_array_equal(logits, restored)


def test_derivative_regions_use_complete_disjoint_double_precision_norms():
    from grain_research.vector_mechanisms.sensitivity import region_derivative_stats

    h = np.ones((3, 4), dtype=np.float32)
    g = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    rows = region_derivative_stats(h, g, {"a": [0, 2], "b": [1]})
    assert sum(r["gradient_energy_fraction"] for r in rows) == pytest.approx(
        1, abs=1e-12
    )
    with pytest.raises(ValueError, match="partition"):
        region_derivative_stats(h, g, {"a": [0, 1], "b": [1, 2]})


def test_independent_access_keeps_evidence_and_changes_code_only_after_question():
    from grain_research.vector_mechanisms.independent_access import (
        ACCESS_SYSTEM,
        access_message,
        answer_code_controls,
    )

    a, hi, ci = access_message("Old history.", "Current text :(", "disposition")
    b, hj, cj = access_message(
        "Old history.", "Current text :(", "disposition", swapped=True
    )
    assert hi == hj and ci == cj
    assert a[slice(*hi)] == b[slice(*hi)] == "Old history."
    assert a[slice(*ci)] == b[slice(*ci)] == "Current text :("
    assert a.split("Question:")[0] == b.split("Question:")[0]
    assert "Yes means supported" not in ACCESS_SYSTEM
    controls = answer_code_controls()
    assert len(controls) == 16
    assert sum(c["expected_yes"] for c in controls) == 8
    for a, b in zip(controls[::2], controls[1::2], strict=True):
        assert a["expected_yes"] != b["expected_yes"]


def test_independent_access_norms_and_fixed_direction_are_reproducible():
    from scripts.run_grain_vector_mechanism_independent_access import make_arms

    h = np.zeros((3, 4))
    g = np.arange(12).reshape(3, 4) + 1.0
    arms = make_arms(h, g, 0.1, "synthetic", np.asarray)
    assert len(arms) == 6
    repeat = make_arms(h, g, 0.1, "synthetic", np.asarray)
    for arm, other in zip(arms, repeat, strict=True):
        assert np.linalg.norm(arm[2]) == pytest.approx(0.1, rel=1e-8)
        np.testing.assert_array_equal(arm[2], other[2])
        assert arm[3] < 0.01


def test_independent_access_sign_flip_is_two_sided_and_keeps_zero():
    from scripts.summarize_grain_vector_independent_access import sign_flip_p

    assert sign_flip_p(np.zeros(12)) == 1
    assert sign_flip_p(np.ones(12)) < 0.005
    assert sign_flip_p(-np.ones(12)) == sign_flip_p(np.ones(12))


def test_independent_preflight_accepts_mapping_and_list_tokenizer_results():
    from grain_research.vector_mechanisms.independent_access import unbounded_prompt_ids

    class Tokenizer:
        def __init__(self, value):
            self.value = value

        def apply_chat_template(self, *args, **kwargs):
            return self.value

    for value in ([1, 2, 3], {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}):
        assert unbounded_prompt_ids(Tokenizer(value), "source") == [1, 2, 3]
    with pytest.raises(ValueError, match="flat"):
        unbounded_prompt_ids(Tokenizer(["input_ids"]), "source")


def test_independent_summary_keeps_swapped_polarity_and_all_authors(
    tmp_path, monkeypatch
):
    import json
    import sys

    from grain_research.vector_mechanisms.independent_access import answer_code_controls
    from scripts.summarize_grain_vector_independent_access import main

    parent, output = tmp_path / "parent", tmp_path / "analysis"
    parent.mkdir()
    (parent / "manifest.json").write_text(
        json.dumps({"identity": "synthetic", "eligible_authors": 4})
    )
    (parent / "run_status.json").write_text(
        json.dumps(
            {"status": "COMPLETE", "authors": 4, "all_instrument_controls_exact": True}
        )
    )
    rows = []
    for author in range(4):
        for condition, effect in (
            ("source", 1.0),
            ("transfer", 0.5),
            ("code_swapped", -0.5),
            ("collateral", 0.1),
        ):
            for method in ("candidate", "random_0", "random_1"):
                for sign in (-1, 1):
                    delta = sign * (effect if method == "candidate" else 0.05)
                    rows.append(
                        {
                            "author_id": f"u{author}",
                            "condition": condition,
                            "method": method,
                            "sign": sign,
                            "polarity": -1 if condition == "code_swapped" else 1,
                            "delta": delta,
                            "predicted_delta": delta,
                            "norm_match_error": 0,
                            "gradient_cosine": 1,
                        }
                    )
    pd.DataFrame(rows).to_csv(parent / "interventions.csv", index=False)
    codes = [
        {k: v for k, v in c.items() if k != "user"}
        | {"margin": 2 if c["expected_yes"] else -2, "correct": True}
        for c in answer_code_controls()
    ]
    pd.DataFrame(codes).to_csv(parent / "code_controls.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize", "--parent-dir", str(parent), "--output-dir", str(output)],
    )
    main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["authors"] == 4
    assert summary["code_control"]["limited_code_control_qualified"]
    endpoints = {r["condition"]: r for r in summary["confirmation_endpoints"]}
    assert endpoints["transfer"]["mean"] == pytest.approx(0.45)
    assert endpoints["code_swapped"]["mean"] == pytest.approx(0.55)
    assert all(r["clusters"] == 4 for r in endpoints.values())


def test_capability_cases_and_strict_parse_do_not_extract_explanations():
    from grain_research.vector_mechanisms.capability import (
        capability_cases,
        parse_binary_answer,
    )

    rows = capability_cases()
    assert len(rows) == 216
    assert len({r["case_id"] for r in rows}) == 208
    assert sum(r["repeat"] for r in rows) == 8
    assert parse_binary_answer(" Yes.\n") == 1
    assert parse_binary_answer("no!") == 0
    for text in ("Yes, because", "Not Yes", "No then Yes", "", None):
        assert parse_binary_answer(text) is None


def test_capability_summary_counts_invalid_responses_as_errors():
    from grain_research.vector_mechanisms.capability import capability_cases
    from scripts.summarize_grain_vector_capability import summarize_run

    f = pd.DataFrame(capability_cases())
    f["parsed_yes"] = f.expected_yes.astype(float)
    f["parse_ok"] = True
    f["correct"] = True
    f["content"] = np.where(f.expected_yes == 1, "Yes", "No")
    summary, _ = summarize_run(f)
    assert summary["next_observer_qualified"]
    assert summary["repeat_exact_matches"] == 8
    mask = (f.battery == "semantic_fact") & (f.family == "composition")
    f.loc[mask, "correct"] = False
    f.loc[mask, "parse_ok"] = False
    summary, _ = summarize_run(f)
    assert summary["facts_accuracy"] == 0.75
    assert not summary["next_observer_qualified"]


def test_capability_format_diagnostic_cannot_change_primary_gates():
    from grain_research.vector_mechanisms.capability import (
        capability_cases,
        leading_binary_diagnostic,
    )
    from scripts.summarize_grain_vector_capability import (
        audit_recorded_responses,
        format_diagnostic,
        summarize_run,
    )

    f = pd.DataFrame(capability_cases())
    f["parsed_yes"] = f.expected_yes.astype(float)
    f["parse_ok"] = True
    f["correct"] = True
    f["content"] = np.where(f.expected_yes == 1, "Yes", "No")
    mask = (
        (f.battery == "semantic_fact")
        & (f.family == "composition")
        & (f.expected_yes == 0)
    )
    f.loc[mask, "content"] = "No.\nThis adds an explanation."
    f.loc[mask, ["parse_ok", "correct"]] = False
    f.loc[mask, "parsed_yes"] = np.nan
    audit_recorded_responses(f)
    assert format_diagnostic(f)["leading_binary_accuracy"] == 1
    assert not summarize_run(f)[0]["next_observer_qualified"]
    for text in ("Nobody", "Not Yes", "Yes, but no", "Reason: No."):
        assert leading_binary_diagnostic(text) is None
    f.loc[0, "correct"] = False
    with pytest.raises(ValueError, match="strict recomputation"):
        audit_recorded_responses(f)


def test_external_partitions_reserve_bridge_missing_from_official_axis_members():
    from grain_research.vector_mechanisms.external_sources import (
        exclusive_source_groups,
        source_partitions,
    )

    sets = {
        "big5": {"b", "both", "bridge_extra", "reserved"},
        "bridge": {"both", "bridge_extra"},
    }
    sets.update(
        {a + "_cont": {"m", "both", "reserved"} for a in ("EI", "SN", "TF", "JP")}
    )
    p = source_partitions(sets)
    assert p["big5_only"] == {"b"}
    assert p["bridge_outside_official_mbti"] == {"bridge_extra"}
    assert p["both_nonreport_reserved"] == {"reserved"}
    groups = exclusive_source_groups(sets)
    assert sum(len(v) for v in groups.values()) == len(set().union(*groups.values()))


def test_external_cohort_loads_only_ids_and_separate_source_folds(tmp_path):
    from scripts.run_grain_vector_mechanism_external_cohort import load_source_folds

    (tmp_path / "bridge").mkdir()
    (tmp_path / "mbti_axes").mkdir()
    pd.DataFrame(
        {
            "user_id": ["b", "x"],
            "official_fold": [0, 1],
            "openness": ["not parsed", "not parsed"],
        }
    ).to_csv(tmp_path / "pandora_official_big5_prepared.csv", index=False)
    pd.DataFrame({"user_id": ["x"], "bridge_fold": [4]}).to_csv(
        tmp_path / "bridge/pandora_official_bridge_strict377.csv", index=False
    )
    for i, axis in enumerate(("EI_cont", "SN_cont", "TF_cont", "JP_cont")):
        pd.DataFrame({"user_id": ["m", "x"], "official_fold": [i, 4]}).to_csv(
            tmp_path / "mbti_axes" / f"pandora_official_{axis}_prepared.csv",
            index=False,
        )
    members, folds, meta = load_source_folds(tmp_path)
    assert members["big5"] == {"b", "x"}
    assert [folds[a + "_cont"]["m"] for a in ("EI", "SN", "TF", "JP")] == [0, 1, 2, 3]
    assert meta["big5"]["columns_read"] == ["user_id", "official_fold"]
    pd.DataFrame({"user_id": ["b"], "official_fold": [5]}).to_csv(
        tmp_path / "pandora_official_big5_prepared.csv", index=False
    )
    with pytest.raises(ValueError, match="Invalid official"):
        load_source_folds(tmp_path)


def test_external_material_audit_rejects_cross_author_duplicate_and_wrong_fold():
    import copy

    from grain_research.vector_mechanisms.cohort import ID_SEED
    from grain_research.vector_mechanisms.data_views import stable_key
    from scripts.summarize_grain_vector_external_cohort import validate_materials

    members = {"big5": {"u0"}, "bridge": {"u1"}}
    members.update({a + "_cont": set() for a in ("EI", "SN", "TF", "JP")})
    folds = {k: {u: 0 for u in v} for k, v in members.items()}
    bundles, ledger = [], []
    for i, group in enumerate(("big5_only", "bridge")):
        uid = stable_key(f"u{i}", ID_SEED)[:24]
        units = [
            {
                "author_id": uid,
                "comment_id": f"c{i}_{j}",
                "thread_id": f"t{i}_{j}",
                "body": f"author{i} unit{j} " + "text " * 38,
                "word_count": 40,
                "timestamp": float(j),
                "leakage_flags": {"report_clue": False},
            }
            for j in range(36)
        ]
        bundles.append(
            {
                "author_id": uid,
                "source_group": group,
                "views": [units[j : j + 12] for j in (0, 12, 24)],
                "official_folds": {k: 0 for k, m in members.items() if f"u{i}" in m},
            }
        )
        ledger.append(
            {
                "author_id": uid,
                "source_group": group,
                "eligible": True,
                **{k + "_fold": m.get(f"u{i}", np.nan) for k, m in folds.items()},
            }
        )
    f = pd.DataFrame(ledger)
    _, checks = validate_materials(bundles, f, members, folds)
    assert checks["eligible_authors"] == 2
    bad = copy.deepcopy(bundles)
    bad[1]["views"][0][0]["body"] = bad[0]["views"][0][0]["body"]
    with pytest.raises(ValueError, match="Normalized text shared"):
        validate_materials(bad, f, members, folds)
    f.loc[0, "big5_fold"] = 2
    with pytest.raises(ValueError, match="Source-specific fold"):
        validate_materials(bundles, f, members, folds)


def test_external_mask_is_count_matched_nonoverlapping_and_source_preserving():
    from grain_research.vector_mechanisms.external_material import masked_comment

    text = "I am INFP.\nMy test results were surprising but daily life keeps moving."
    arms, counts = masked_comment(text, "c1")
    assert arms["raw"] == text
    assert counts["cue_words"] == counts["placebo_words"] == 4
    assert counts["cue_placebo_overlap"] == 0
    assert all(len(t.split()) == len(text.split()) for t in arms.values())
    assert all("\n" in t for t in arms.values())
    assert masked_comment(text, "c1") == (arms, counts)
    assert "INFP" not in arms["cue_mask"]
    assert "INFP" in arms["placebo_mask"]
    neutral, c = masked_comment("We took a walk after work.", "c2")
    assert len(set(neutral.values())) == 1 and c["cue_words"] == 0
    with pytest.raises(ValueError, match="Insufficient noncue"):
        masked_comment("INFP ENTP", "c3")


def test_external_named_feature_aggregation_matches_frozen_implementation():
    from grain_research.vector_mechanisms.external_material import raw_author_features
    from suica_core.v7_psychometric import author_features_from_embeddings

    h = np.arange(24).reshape(3, 4, 2).astype(float)
    names = ["std::svd_001", "mean::svd_000", "std::svd_000", "mean::svd_001"]
    frame = pd.DataFrame(
        {"user_id": np.repeat(["z", "a", "b"], 4), "split": "all", "token_count": 40}
    )
    expected = author_features_from_embeddings(frame, h.reshape(-1, 2))[
        names
    ].to_numpy()
    np.testing.assert_array_equal(raw_author_features(h, names), expected)


def test_external_thread_purge_counts_do_not_use_author_or_label_as_proxy():
    from grain_research.vector_mechanisms.external_material import purge_counts

    train = [{"author_id": "a", "views": [[{"thread_id": "x"}, {"thread_id": "y"}]]}]
    test = [{"author_id": "b", "views": [[{"thread_id": "y"}]]}]
    result = purge_counts(train, test)
    assert result["removed_comments"] == 1
    assert result["remaining_min_median_max"] == [1, 1, 1]
    with pytest.raises(ValueError, match="author overlap"):
        purge_counts(train, train)


def test_external_model_signature_ignores_time_not_precision_or_site():
    from scripts.run_grain_vector_mechanism_external_features import model_signature

    a = {"dtype": "bfloat16", "sites": ("block_10",), "unix_time": 1}
    b = {"dtype": "bfloat16", "sites": ["block_10"], "unix_time": 2}
    assert model_signature(a) == model_signature(b)
    assert model_signature(a) != model_signature(b | {"dtype": "float32"})


def test_external_encoder_reuses_equal_texts_but_preserves_all_arm_views(monkeypatch):
    import scripts.run_grain_vector_mechanism_external_features as module

    class Tokenizer:
        def __call__(self, texts, **_):
            return {"input_ids": [[len(t)] for t in texts]}

    calls = []

    def encode(texts, *_args, **_kwargs):
        calls.append(len(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(module, "collect_encoder", encode)
    raw = [f"unit {i}" for i in range(36)]
    arms = {"raw": raw, "cue_mask": raw, "placebo_mask": raw}
    features, metrics = module.encoder_author(
        arms, Tokenizer(), None, "", "cls", "cpu", repeat=True
    )
    assert calls == [36, 36]
    assert metrics["unique_comments"] == 36 and metrics["logical_comments"] == 108
    assert features["raw_comment_vectors"].shape == (3, 12, 2)
    np.testing.assert_array_equal(
        features["raw_view_mean"], features["cue_mask_view_mean"]
    )
    assert len(set(metrics["token_signatures"].values())) == 1
