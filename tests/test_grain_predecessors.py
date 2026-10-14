#!/usr/bin/env python3
"""Synthetic tests for literature adapters, independent of private corpora."""

import numpy as np
import pytest

from grain_research.predecessors import (
    FACETS,
    MeaningExtraction,
    first_person_sentences,
    keyed_proportions,
    match_items,
    mean_pearson,
    parse_ipip_neo,
    reference_percentiles,
    varimax,
)


def test_sentence_offsets_and_first_person():
    text = "  I enjoy music. She disagrees.\nI don't like noise!  "
    rows = first_person_sentences(text)
    assert [r[2] for r in rows] == ["I enjoy music.", "I don't like noise!"]
    assert all(text[lo:hi] == s for lo, hi, s in rows)
    assert first_person_sentences("It is quiet.") == []


def test_varimax_is_rotation_not_new_information():
    x = np.random.default_rng(9).normal(size=(24, 4))
    r, info = varimax(x)
    np.testing.assert_allclose(r.T @ r, np.eye(4), atol=1e-10)
    np.testing.assert_allclose((x @ r) @ (x @ r).T, x @ x.T, atol=1e-10)
    assert info["iterations"] > 0


def test_mem_frozen_vocabulary_batch_independence():
    docs = np.array(
        [
            "music melody garden",
            "music melody ocean",
            "market finance stocks",
            "market finance bank",
            "garden flowers nature",
            "ocean waves nature",
            "books reading library",
            "books library finance",
        ]
        * 3
    )
    model = MeaningExtraction(n_components=3, min_df=0.05).fit(docs)
    vocabulary = model.vectorizer_.vocabulary_.copy()
    single = model.transform(["music melody unknownholdouttoken"])
    many = model.transform(["music melody unknownholdouttoken", "bank finance garden"])
    np.testing.assert_allclose(single[0], many[0])
    assert "unknownholdouttoken" not in model.vectorizer_.vocabulary_
    assert model.vectorizer_.vocabulary_ == vocabulary
    np.testing.assert_allclose(np.cov(model.transform(docs).T), np.eye(3), atol=1e-8)


def test_item_count_one_match_per_sentence_and_missingness():
    items = [
        {"facet": "E1", "trait": "Extraversion", "key": 1},
        {"facet": "E1", "trait": "Extraversion", "key": -1},
    ]
    counts, best, scores, accepted = match_items(
        np.array([[1.0, 0], [0, 1], [-1, -1]]), np.eye(2), items
    )
    assert accepted.tolist() == [True, True, False]
    assert counts[FACETS.index("E1")].tolist() == [1, 1]
    assert counts[len(FACETS)].tolist() == [1, 1]
    ratios = keyed_proportions(counts)
    assert ratios[0] == 0.5 and np.isnan(ratios[1])
    assert scores.shape == best.shape == accepted.shape


def test_item_rejects_zero_and_bad_key():
    with pytest.raises(ValueError, match="Zero vectors"):
        match_items(np.zeros((1, 2)), np.eye(2), [{}, {}])


def test_fixed_reference_no_test_batch_ranking():
    reference = np.array([[0.0, np.nan], [0.5, np.nan], [1.0, np.nan]])
    a = reference_percentiles(reference, np.array([[0.5, 0.2]]))
    b = reference_percentiles(reference, np.array([[0.5, 0.2], [-10.0, 0.3]]))
    assert a[0, 0] == b[0, 0] == 0.5
    assert np.isnan(a[0, 1])


def test_no_silent_trait_omission():
    y = np.arange(20).reshape(10, 2)
    assert mean_pearson(y, y) == pytest.approx(1)
    assert np.isnan(mean_pearson(y, np.ones_like(y)))


def test_ipip_parser_fails_closed_on_wrong_page():
    with pytest.raises(ValueError, match="30 official"):
        parse_ipip_neo("error page")


def test_ipip_parser_keeps_neuroticism_sign():
    sections = []
    for facet in FACETS:
        rows = "".join(
            f"<tr><td>{'+ keyed' if i == 0 else '- keyed' if i == 5 else ''}</td><td>Feel item {i}.</td></tr>"
            for i in range(10)
        )
        sections.append(f"{facet}: EXAMPLE<table>{rows}</table>")
    items = parse_ipip_neo("".join(sections) + "5 NEO Domains")
    assert len(items) == 300
    n = [r for r in items if r["facet"] == "N1"]
    assert n[0]["trait"] == "Neuroticism" and n[0]["key"] == 1 and n[-1]["key"] == -1


def test_nested_preprocessing_never_sees_outer_test_values():
    from scripts.run_grain_predecessor_reference import fit_fold

    rng = np.random.default_rng(71)
    x = rng.normal(size=(45, 10))
    y = x[:, :5] + rng.normal(scale=0.1, size=(45, 5))
    train, test = np.arange(35), np.arange(35, 45)
    feature, head, pred, selection = fit_fold(
        {"grain": x}, y, train, test, "grain_ridge"
    )
    changed = x.copy()
    changed[test] += 10000
    second, head2, _, selection2 = fit_fold(
        {"grain": changed}, y, train, test, "grain_ridge"
    )
    np.testing.assert_allclose(feature.scaler.mean_, second.scaler.mean_)
    np.testing.assert_allclose(head.coef_, head2.coef_)
    assert selection == selection2 and pred.shape == (10, 5)


def test_empty_candidate_set_stays_unavailable():
    items = [{"facet": "E1", "trait": "Extraversion", "key": 1}]
    counts, best, score, accepted = match_items(
        np.empty((0, 2)), np.array([[1.0, 0]]), items
    )
    assert counts.sum() == 0 and len(best) == len(score) == len(accepted) == 0
    assert np.isnan(keyed_proportions(counts)).all()


def test_joint_views_are_fitted_on_training_authors_only():
    """Neither held-out magnitudes nor later batches may alter the reference."""
    from scripts.run_grain_predecessor_integration import lowrank_views

    rng = np.random.default_rng(51)
    sources = [rng.normal(size=(20, 7)), rng.normal(size=(20, 5))]
    train, test = np.arange(15), np.arange(15, 20)
    x, _, models = lowrank_views(sources, train, test, dimensions=3)
    perturbed = [s.copy() for s in sources]
    for s in perturbed:
        s[test] += 10000
    x2, _, models2 = lowrank_views(perturbed, train, test, dimensions=3)
    for one, other, model, model2 in zip(x, x2, models, models2, strict=True):
        np.testing.assert_allclose(one, other)
        np.testing.assert_allclose(model[0].components_, model2[0].components_)
