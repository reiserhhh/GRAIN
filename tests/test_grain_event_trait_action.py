#!/usr/bin/env python3
"""Tests of exact event joins and independent target/action comparisons."""

import numpy as np
import pytest

from grain_research.construction import math_event_trait_action as E


def test_exclude_all_application_authors_without_reordering_source():
    np.testing.assert_array_equal(E.source_exclusion([7, 2, 5, 1], [1, 7], 8), [2, 5])


@pytest.mark.parametrize("source,app", [([1, 1], [2]), ([9], [2]), ([1], [2, 2])])
def test_exclusion_rejects_ambiguous_author_indices(source, app):
    with pytest.raises(ValueError, match="Unique"):
        E.source_exclusion(source, app, 8)


def test_exact_deletion_denominator_and_equal_size_contrast():
    values = np.arange(60, dtype=float).reshape(12, 5) / 60
    s, c = np.arange(4), np.arange(4, 8)
    out = E.deletion_accounts(values, s, c)
    np.testing.assert_allclose(out["identity_error"], 0, atol=1e-15)
    np.testing.assert_allclose(
        out["removal_contrast"], (values[c].sum(0) - values[s].sum(0)) / 8
    )
    np.testing.assert_allclose(out["removal_contrast"], -out["subset_contrast"] / 2)
    assert not np.allclose(out["selected_removal_delta"], values[s].mean(0))


def test_event_permutation_preserves_mean_but_not_identity_selection():
    x = np.arange(36).reshape(12, 3)
    permutation = np.arange(12)[::-1]
    s, c = np.arange(4), np.arange(4, 8)
    ix_s = np.flatnonzero(np.isin(permutation, s))
    ix_c = np.flatnonzero(np.isin(permutation, c))
    a, b = E.deletion_accounts(x, s, c), E.deletion_accounts(x[permutation], ix_s, ix_c)
    np.testing.assert_allclose(a["removal_contrast"], b["removal_contrast"])
    assert not np.array_equal(
        a["removal_contrast"],
        E.deletion_accounts(x[permutation], s, c)["removal_contrast"],
    )


def test_constant_field_cannot_gain_personality_action_by_scaling():
    a = E.deletion_accounts(np.ones((12, 5)) * 0.3, np.arange(4), np.arange(4, 8))
    np.testing.assert_array_equal(a["removal_contrast"], np.zeros(5))
    m = E.alignment(np.zeros(8), np.arange(8), draws=10)
    assert m["pearson"] is None and m["unscaled_cosine"] is None
    assert m["mathematical_zero"] == 8


def test_signed_concordance_is_not_absolute_relevance():
    x = np.array([-2, -1, 1, 2.0])
    m = E.alignment(x, -x, draws=10)
    assert m["pearson"] == pytest.approx(-1)
    assert m["signed_discordant"] == 4 and m["signed_concordant"] == 0
    assert E.pearson(np.abs(x), -x) == pytest.approx(0)


def test_unique_historical_control_restored_without_response():
    rng = np.random.default_rng(19)
    v, poles = rng.normal(size=(12, 7)), rng.normal(size=(2, 7))
    q = E.unit_vectors(v) @ np.diff(E.unit_vectors(poles), axis=0)[0]
    s = np.sort(np.argsort(-np.abs(q), kind="stable")[:4])
    c = np.flatnonzero(~np.isin(np.arange(12), s))[:4]
    contrast = np.abs(q[s]).mean() - np.abs(q[c]).mean()
    actual_s, actual_c, signed = E.recover_controls(v, poles, contrast)
    np.testing.assert_array_equal(actual_s, s)
    np.testing.assert_array_equal(actual_c, c)
    np.testing.assert_array_equal(signed, q)


def test_ambiguous_control_stops_instead_of_choosing_best_native_outcome():
    with pytest.raises(ValueError, match="unique"):
        E.recover_controls(np.ones((12, 3)), np.ones((2, 3)), 0)


@pytest.mark.parametrize("bad", [np.zeros((12, 3)), np.full((12, 3), np.nan)])
def test_invalid_embedding_rejected(bad):
    with pytest.raises(ValueError, match="Finite"):
        E.unit_vectors(bad)


def test_subsets_must_be_disjoint_identity_units():
    with pytest.raises(ValueError, match="disjoint"):
        E.event_variants(np.ones((12, 2)), np.arange(4), np.arange(2, 6))


def test_bootstrap_is_reproducible_and_does_not_reorient_negative_link():
    x, y = np.arange(24), -np.arange(24)
    a, b = E.alignment(x, y, draws=30), E.alignment(x, y, draws=30)
    assert a == b
    assert a["paired_author_bootstrap_ci"] == pytest.approx([-1, -1])
