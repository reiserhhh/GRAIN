#!/usr/bin/env python3
"""Exact synthetic tests for nested path references, not psychological evidence."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from grain_research.construction import math_excursion_order as E
from grain_research.construction import math_path_reference as M
from grain_research.construction.math_path_completion import count_matrix
from scripts.run_grain_path_reference_hm266 import source_accounts

A = (0, 4, 0, 1, 0, 2, 0, 5, 0, 3)
B = (0, 4, 0, 2, 0, 1, 0, 5, 0, 3)


def full_bundle():
    """Provide complete synthetic fields, with no fitted labels or text model."""
    v = np.arange(8**3).reshape(8, 8, 8) / (8**3)
    fields = np.zeros((3, 2, 32, 8, 8, 8))
    fields[:, :, 0] = 1 - v
    fields[:, :, 1] = v
    return {
        "fields": fields.tolist(),
        "supported_triples": np.ones((3, 8, 8, 8), bool).tolist(),
    }


def test_all_short_binary_paths_against_complete_enumeration():
    for length in range(3, 9):
        for path in product(range(2), repeat=length):
            a = M.family_counts(path, states=2)
            words = M.family_words(path, states=2)
            total = sum(
                (E.triple_counts(w, 2) for w in words), start=np.zeros((2, 2, 2), int)
            )
            np.testing.assert_array_equal(total, a["numerator"])
            assert len(words) == a["denominator"]


def test_all_short_ternary_paths_and_boundary_margins():
    for length in range(3, 7):
        for path in product(range(3), repeat=length):
            a = M.family_counts(path)
            words = M.family_words(path)
            total = sum(
                (E.triple_counts(w) for w in words), start=np.zeros((8, 8, 8), int)
            )
            np.testing.assert_array_equal(total, a["numerator"])
            assert M.family_margin_check(path, a)


def test_nested_orbits_partition_the_full_family_and_tower():
    words = M.family_words(A)
    full = M.family_counts(A)
    average = np.zeros((8, 8, 8), object)
    classes = []
    for word in words:
        orbit = E.orbit_counts(word)
        for index, n in np.ndenumerate(orbit["numerator"]):
            average[index] += Fraction(int(n), orbit["denominator"])
        members = set(E.orbit_words(word))
        assert members <= set(words)
        if members not in classes:
            assert all(not members.intersection(c) for c in classes)
            classes.append(members)
    assert set().union(*classes) == set(words)
    np.testing.assert_array_equal(average, full["numerator"])


def test_family_preserves_endpoints_edges_counts_and_occupancy():
    for word in M.family_words(A):
        np.testing.assert_array_equal(count_matrix(word), count_matrix(A))
        np.testing.assert_array_equal(
            np.bincount(word, minlength=8), np.bincount(A, minlength=8)
        )
        assert word[:2] == A[:2] and word[-2:] == A[-2:]
        np.testing.assert_array_equal(
            M.family_counts(word)["numerator"], M.family_counts(A)["numerator"]
        )


def test_full_family_strictly_larger_than_excursion_orbit():
    assert M.family_counts(A)["denominator"] > E.orbit_counts(A)["orbit_size"]
    np.testing.assert_array_equal(
        M.family_counts(A)["numerator"], M.family_counts(B)["numerator"]
    )


def test_fixed_indicator_separates_configuration_and_order():
    values = []
    for path in (A, B):
        fine, coarse = E.orbit_counts(path), M.family_counts(path)
        index = (1, 0, 2)
        s = Fraction(int(fine["observed_counts"][index]), 8)
        a = Fraction(int(fine["numerator"][index]), 8 * fine["denominator"])
        c = Fraction(int(coarse["numerator"][index]), 8 * coarse["denominator"])
        values.append((s, a, c, a - c, s - a))
    assert values[0] == (
        Fraction(1, 8),
        Fraction(1, 16),
        Fraction(1, 24),
        Fraction(1, 48),
        Fraction(1, 16),
    )
    assert values[1] == (
        0,
        Fraction(1, 16),
        Fraction(1, 24),
        Fraction(1, 48),
        Fraction(-1, 16),
    )


def test_source_common_support_and_all_signed_covariance_parts():
    scores = np.zeros((386, 2, 3, 2, 5, 5))
    t = np.linspace(-1, 1, 386)
    for w in range(2):
        c, q, d = t * (w + 1), -np.sin(t), np.cos(t) * (1 - 2 * w)
        for kind, value in enumerate((c + q + d, c + q, c, q, d)):
            scores[:, w, :, :, kind] = value[:, None, None, None]
    scores[0, :, 2, :, 2] = np.nan
    rows, parts, sizes, errors = source_accounts(
        scores, {"bank": np.zeros(386, int), "targets": np.zeros((2, 386, 5))}
    )
    assert len(rows) == 250 and len(parts) == 90
    assert sizes == [385, 385]
    assert {r["joint_complete_authors"] for r in rows} == {385}
    assert max(errors) < 1e-14
    assert any(r["cross_window_covariance"] < 0 for r in parts)


def test_singleton_and_three_event_boundary_case():
    for path in [(0, 0, 0), (0, 1, 2), (0, 1, 2, 3)]:
        a = M.family_counts(path)
        assert a["denominator"] == 1
        np.testing.assert_array_equal(a["numerator"], E.triple_counts(path))


def test_pair_additive_fields_have_zero_both_extra_components():
    a = M.evaluate(full_bundle(), A)
    pair = np.arange(64).reshape(8, 8)
    field = pair[:, :, None] + pair[None]
    count = a["family_numerator"] / a["family_size"]
    orbit = a["orbit_numerator"] / a["orbit_denominator"]
    assert abs(np.sum((orbit - count) * field)) < 1e-12
    assert abs(np.sum((E.triple_counts(A) - orbit) * field)) < 1e-12


def test_nested_scores_reconstruct_all_profiles():
    a = M.evaluate(full_bundle(), A)
    s, fine, coarse, q, d = np.moveaxis(a["profile_readings"], 2, 0)
    np.testing.assert_allclose(s, coarse + q + d, atol=1e-14)
    np.testing.assert_allclose(fine, coarse + q, atol=1e-14)
    np.testing.assert_allclose(a["profile_readings"][:, :, :3].sum(-1), 1)
    np.testing.assert_allclose(a["profile_readings"][:, :, 3:].sum(-1), 0, atol=1e-14)


def test_larger_family_missing_does_not_erase_old_readings():
    a = E.orbit_counts(A)
    c = M.family_counts(A)
    extra = (c["numerator"] > 0) & (a["numerator"] == 0)
    assert extra.any()
    bundle = full_bundle()
    supported = np.array(bundle["supported_triples"])
    supported[:, extra] = False
    bundle["supported_triples"] = supported.tolist()
    v = M.evaluate(bundle, A)["scores"]
    assert np.isfinite(v[:, :, [0, 1, 4]]).all()
    assert np.isnan(v[:, :, [2, 3]]).all()


@pytest.mark.parametrize("path", [[], [0, 1], [0.0, 1.0, 2.0], [0, 8, 1], [0] * 19])
def test_reject_invalid_or_oversize_paths_without_truncation(path):
    with pytest.raises(ValueError):
        M.family_counts(path)


def test_enumeration_bound_does_not_truncate_the_family():
    with pytest.raises(ValueError):
        M.family_words(A, maximum=1)


def test_unknown_homogeneous_kernels_are_uniform_within_family():
    n = np.arange(1, 65).reshape(8, 8)
    p = n / n.sum(1)[:, None]
    values = [np.sum(np.log(p) * count_matrix(w)) for w in M.family_words(A)]
    assert np.ptp(values) == 0


def test_cold_record_identity_and_event_guards():
    def record(author):
        return {
            "author_key": author,
            "events": [
                {"event_id": str(i), "timestamp": i, "cell": x} for i, x in enumerate(A)
            ],
        }

    np.testing.assert_allclose(
        M.score_record(full_bundle(), record("one"))["scores"],
        M.score_record(full_bundle(), record("two"))["scores"],
    )
    bad = record("one")
    bad["events"][1]["event_id"] = "0"
    with pytest.raises(ValueError):
        M.score_record(full_bundle(), bad)
