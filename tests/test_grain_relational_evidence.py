#!/usr/bin/env python3
"""Binding information, unresolved evidence and observation-only reference tests."""

import numpy as np
import pytest

from grain_research.relational_evidence import (
    atom,
    candidate_joint,
    factorize,
    marginalize,
    query_kernel,
    query_measure,
    read_controlled_pair,
    transform_axes,
    validate_joint,
)


def test_identical_marginals_do_not_identify_query_value():
    positive = (atom(0, 1, 0, 0) + atom(0, 1, 1, 1)) / 2
    negative = (atom(0, 1, 0, 1) + atom(0, 1, 1, 0)) / 2
    np.testing.assert_allclose(marginalize(positive), marginalize(negative))
    assert query_measure(positive)["score"] == 1
    assert query_measure(negative)["score"] == -1
    assert query_measure(factorize(positive))["score"] == 0


def test_marginal_projection_has_nontrivial_unidentifiable_query():
    basis = np.eye(16).reshape(16, 2, 2, 2, 2)
    operator = np.array([np.concatenate(marginalize(x)) for x in basis]).T
    assert np.linalg.matrix_rank(operator) == 5
    query = query_kernel().ravel()
    projection = operator.T @ np.linalg.lstsq(operator.T, query, rcond=None)[0]
    assert np.linalg.norm(query - projection) > 1


def test_factorization_preserves_mass_and_every_single_marginal():
    joint = np.random.default_rng(23).dirichlet(np.ones(16)).reshape(2, 2, 2, 2) * 0.7
    separate = factorize(joint)
    np.testing.assert_allclose(marginalize(joint), marginalize(separate))
    assert separate.sum() == pytest.approx(0.7)


def test_zero_evidence_and_unresolved_are_not_zero_personality():
    unknown = query_measure(np.zeros((2, 2, 2, 2)))
    assert unknown["score"] is None and unknown["status"] == "unresolved"
    assert (unknown["lower"], unknown["upper"]) == (-1, 1)
    other = query_measure(atom(1, 0, 1, 1))
    assert other["score"] is None and other["lower"] is None
    assert other["status"] == "no_target_evidence"


def test_unresolved_mass_bounds_enclose_all_signed_completions():
    known = 0.3 * atom(0, 1, 1, 1) + 0.2 * atom(1, 0, 1, 1)
    result = query_measure(known)
    for extra in np.linspace(0, 0.5, 11):
        for positive in np.linspace(0, extra, 11):
            completed = (
                known
                + positive * atom(0, 1, 1, 1)
                + (extra - positive) * atom(0, 1, 1, 0)
            )
            score = query_measure(completed)["score"]
            assert result["lower"] - 1e-12 <= score <= result["upper"] + 1e-12


def test_common_basis_change_and_role_permutation_preserve_contractions():
    rng = np.random.default_rng(17)
    joint = rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2)
    query = query_kernel()
    maps = [np.linalg.qr(rng.normal(size=(2, 2)))[0] for _ in range(4)]
    np.testing.assert_allclose(
        np.sum(joint * query),
        np.sum(transform_axes(joint, maps) * transform_axes(query, maps)),
        atol=1e-14,
    )
    assert query_measure(atom(0, 1, 1, 1), actor=1)["score"] is None
    assert query_measure(atom(0, 1, 1, 1)[::-1], actor=1)["score"] == 1


def test_raw_reference_uses_question_not_yes_alone():
    a = read_controlled_pair(
        "In a social situation, will you accept the offered help tomorrow?", "Yes."
    )
    b = read_controlled_pair(
        "In a social situation, will you refuse the offered help tomorrow?", "Yes."
    )
    assert a["condition"] == b["condition"] == 1
    assert query_measure(a["joint"])["score"] == 1
    assert query_measure(b["joint"])["score"] == -1


@pytest.mark.parametrize(
    "partner,reply",
    [
        ("", "Yes."),
        (
            "In a social situation, will you not accept the offered help tomorrow?",
            "No.",
        ),
        ("In a social situation, will you accept the offered help yesterday?", "Yes."),
        (
            "In a social situation, will you accept the offered help tomorrow?",
            "He said yes.",
        ),
    ],
)
def test_unknown_grammar_and_negative_question_are_not_auto_flipped(partner, reply):
    assert read_controlled_pair(partner, reply)["status"] == "outside_declared_grammar"


def test_candidate_weights_are_bound_to_observed_orientation():
    weights = np.zeros((2, 2, 2))
    weights[0, 1] = [1, 3]
    for orientation in (0, 1):
        j = candidate_joint(weights, orientation)
        assert query_measure(j)["score"] == pytest.approx(0.5)
        assert query_measure(j)["candidate_variance"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "value", [np.ones((2, 2)), np.ones((2, 2, 2, 2)), np.full((2, 2, 2, 2), -0.1)]
)
def test_invalid_mass_rejected(value):
    with pytest.raises(ValueError):
        validate_joint(value)


def test_nli_requests_do_not_read_generator_labels():
    from scripts.run_grain_relational_evidence import public_bank, requests_for_bank

    bank = public_bank()
    original = requests_for_bank(bank)
    for field in (
        "actor",
        "channel",
        "orientation",
        "stance",
        "target_sign",
        "condition",
    ):
        bank[field] = "not-available-to-reader"
    changed = requests_for_bank(bank)
    assert original.equals(changed)
    assert len(original) == 256 and original.audit_id.nunique() == 256


def test_literal_source_and_condition_profile_end_to_end():
    from scripts.run_grain_relational_evidence import (
        public_bank,
        score_profiles,
        simulate,
    )

    bank = public_bank()
    literal = np.array(
        [read_controlled_pair(r.partner, r.reply)["joint"] for r in bank.itertuples()]
    )
    config = {
        "seeds": [100],
        "authors_per_seed": 3,
        "math_cpu_budget": 20,
        "blocks_per_condition": 2,
        "events_per_actor_channel_block": 4,
        "conditions": [-1, 1],
        "new_condition": 0.5,
        "worlds": ["complete_relation", "missing_antecedent", "unbound_relations"],
        "readers": [
            "known_relation_upper",
            "literal_reference",
            "fixed_nli_candidates",
            "reply_only",
        ],
    }
    blocks, events, truth = simulate(config, bank, {"literal": literal, "nli": literal})
    _, scores = score_profiles(blocks, truth, config)
    assert len(events) == 3 * 4 * 16
    unavailable = scores[
        (scores.world == "missing_antecedent")
        & scores.reader.isin(["literal_reference", "fixed_nli_candidates"])
    ]
    assert unavailable.score.isna().all()
    unbound = scores[
        (scores.world == "unbound_relations")
        & scores.reader.isin(["literal_reference", "fixed_nli_candidates"])
    ]
    np.testing.assert_allclose(unbound.score, 0, atol=1e-14)
    a = scores[
        (scores.world == "complete_relation")
        & (scores.reader == "known_relation_upper")
    ]
    b = scores[
        (scores.world == "complete_relation") & (scores.reader == "literal_reference")
    ]
    np.testing.assert_allclose(a.score, b.score)
