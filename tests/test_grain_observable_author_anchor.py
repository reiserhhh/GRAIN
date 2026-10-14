#!/usr/bin/env python3
"""Finite observation support, coordinate invariance and source-only scoring."""

import copy
from fractions import Fraction as F

import numpy as np
import pytest

from grain_research.construction import math_observable_author_anchor as M

ALPHABET = np.eye(4)


def test_registered_vector_decoding_and_mark_address():
    assert np.array_equal(M.decode_vectors(ALPHABET, ALPHABET), np.arange(4))
    h = np.array([0, 1, 2, 3])
    endpoint = np.array([0, 1, 2, 3])
    counts = M.observed_table(np.arange(4), ALPHABET[h], ALPHABET[endpoint], ALPHABET)
    for a in range(4):
        q, y = a // 2, (a // 2) ^ (a % 2)
        assert counts[a, q, y] == 1


def test_out_of_alphabet_is_not_silently_personality_scored():
    with pytest.raises(ValueError, match="Out-of-alphabet"):
        M.decode_vectors(np.ones((1, 4)), ALPHABET)


def test_duplicate_source_record_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        M.observed_table([1, 1], ALPHABET[:2], ALPHABET[:2], ALPHABET)


def test_source_conditional_means_and_references():
    t = M.exact_table()
    bundle = M.compile_bundle(t, "randomized_condition")
    profile = M.cold_profile(bundle, ALPHABET[3], ALPHABET)
    witness = M.gauge_witness()
    for q, name in enumerate(("p", "p_squared")):
        target = profile["targets"][q]
        assert target["probability"] == pytest.approx(
            float(witness[name]["conditional_mean"])
        )
        assert target["relative"] == pytest.approx(float(witness[name]["relative"]))
    assert profile["status"] == "RANDOMIZED_CONDITION_PROFILE"


def test_latent_mean_above_average_is_not_monotone_gauge_invariant():
    witness = M.gauge_witness()
    assert witness["p"]["relative"] > 0
    assert witness["p_squared"]["relative"] < 0
    assert witness["midrank_invariant_not_scored"]["conditional_mean"] > F(1, 2)


def test_full_observed_self_selection_alias():
    a = M.selection_alias_table(0, "self_selected_condition")
    b = M.selection_alias_table(1, "self_selected_condition")
    assert a == b
    assert M.selection_alias_table(
        0, "randomized_condition"
    ) != M.selection_alias_table(1, "randomized_condition")
    for world, sign in ((0, -1), (1, 1)):
        table = M.selection_alias_table(world, "randomized_condition")
        profile = M.cold_profile(
            M.compile_bundle(table, "randomized_condition"), ALPHABET[0], ALPHABET
        )
        assert np.sign(profile["targets"][0]["relative"]) == sign


def test_association_retained_not_silently_promoted_to_imposed_condition():
    bundle = M.compile_bundle(
        M.exact_table("self_selected_condition"), "self_selected_condition"
    )
    profile = M.cold_profile(bundle, ALPHABET[3], ALPHABET)
    assert profile["status"] == "ASSOCIATIONAL_CONDITION_PROFILE"
    assert all(x["probability"] is not None for x in profile["targets"])


def test_no_author_signal_null_and_fixed_reference():
    bundle = M.compile_bundle(M.exact_table(null=True), "randomized_condition")
    predictions, _ = M.score_vectors(bundle, ALPHABET, ALPHABET)
    assert np.allclose(predictions, [0.3, 0.6])
    for vector in ALPHABET:
        for target in M.cold_profile(bundle, vector, ALPHABET)["targets"]:
            assert target["relative"] == pytest.approx(0, abs=1e-14)


def test_missing_condition_has_no_fake_zero_or_complete_reference():
    table = np.array(M.exact_table(), float)
    table[3, 1] = 0
    bundle = M.compile_bundle(table, "randomized_condition")
    result = M.cold_profile(bundle, ALPHABET[3], ALPHABET)["targets"][1]
    assert result["status"] == "UNOBSERVED_HISTORY_CONDITION"
    assert result["probability"] is None
    assert result["reference"] is None
    assert result["relative"] is None


def test_cold_input_excludes_hidden_or_teacher_fields():
    bundle = M.compile_bundle(M.exact_table(), "randomized_condition")
    bundle["true_p"] = 0.5
    with pytest.raises(ValueError, match="schema"):
        M.cold_profile(bundle, ALPHABET[3], ALPHABET)


@pytest.mark.parametrize(
    "field", ["history_mass", "reference", "support", "probabilities"]
)
def test_invalid_bundle_readback(field):
    bundle = copy.deepcopy(M.compile_bundle(M.exact_table(), "randomized_condition"))
    if field == "reference":
        bundle[field][0] = 0.9
    elif field == "history_mass":
        bundle[field] = [0.25] * 4
    elif field == "support":
        bundle[field][0][0] = -1
    else:
        bundle[field][0][0] = 1.1
    with pytest.raises(ValueError):
        M.cold_profile(bundle, ALPHABET[3], ALPHABET)


def test_same_cold_score_when_application_batch_changes():
    bundle = M.compile_bundle(M.exact_table(), "randomized_condition")
    alone, _ = M.score_vectors(bundle, ALPHABET[[3]], ALPHABET)
    group, _ = M.score_vectors(bundle, ALPHABET[[0, 2, 3, 1]], ALPHABET)
    assert np.array_equal(alone[0], group[2])


def test_invalid_regime_or_empty_source_not_silently_accepted():
    with pytest.raises(ValueError):
        M.compile_bundle(np.zeros((4, 2, 2)), "randomized_condition")
    with pytest.raises(ValueError):
        M.exact_table("unspecified")
