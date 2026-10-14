#!/usr/bin/env python3
"""Person qualification and fixed SOURCE comparisons,not assumed human traits."""

import numpy as np
import pytest

from grain_research.construction import math_person_bound_response as M


def test_characters_replay_full_cube():
    x = np.arange(24.0).reshape(3, 2, 2, 2)
    coefficients = M.character_accounts(x)
    assert np.einsum("ns,sc->nc", coefficients, M.CHARACTERS).reshape(
        x.shape
    ) == pytest.approx(x)
    assert M.CHARACTERS @ M.CHARACTERS.T == pytest.approx(np.eye(8) * 8)


def test_generic_sample_and_role_bias_have_no_owner_query_character():
    values = np.array([3 + 2 * t + 7 * d for o, t, d in M.SIGNS]).reshape(1, 2, 2, 2)
    assert M.character_accounts(values)[0, 3] == pytest.approx(0)


def test_ideal_owner_and_first_slot_preferences_are_separate():
    owner = np.array([4 if o == t else 2 for o, t, d in M.SIGNS]).reshape(1, 2, 2, 2)
    first_slot = np.array([o * t * d for o, t, d in M.SIGNS]).reshape(1, 2, 2, 2)
    assert M.character_accounts(owner)[0, 3] == pytest.approx(1)
    assert M.character_accounts(first_slot)[0, 3] == pytest.approx(0)
    assert M.character_accounts(first_slot)[0, 7] == pytest.approx(1)


def test_source_actor_differences_are_graph_balanced():
    w = np.arange(60.0).reshape(2, 6, 5)
    delta = M.actor_contrasts(w, np.roll(np.arange(6), -1))
    assert delta.sum(-2) == pytest.approx(np.zeros((2, 5)))
    assert np.sum(delta[0, :, 0] * np.arange(6)) != 0
    assert np.sum(delta[0, :, 0] * -np.arange(6)) == pytest.approx(
        -np.sum(delta[0, :, 0] * np.arange(6))
    )


def test_material_preserves_every_comment_and_whole_bag():
    def encode(text):
        """Use distinct synthetic markers without a neural tokenizer."""
        return list(text.encode())

    groups = [[[501, 502], [503]], [[601], [602, 603]]]
    rows = []
    for o in (-1, 1):
        for d in (-1, 1):
            tokens, spans = M.compose_tokens(groups, o, d, encode)
            for s in spans:
                assert tokens[s["start"] : s["end"]] == groups[s["actor"]][s["event"]]
            rows.append(tokens)
    M.assert_same_bag(rows)
    rows[0].append(999)
    with pytest.raises(ValueError):
        M.assert_same_bag(rows)


@pytest.mark.parametrize("bad", [np.zeros((1, 2, 1, 2)), np.full((1, 2, 2, 2), np.nan)])
def test_partial_or_failed_observation_not_neutral(bad):
    with pytest.raises(ValueError):
        M.character_accounts(bad)


def test_source_actor_self_or_missing_partner_rejected():
    with pytest.raises(ValueError):
        M.actor_contrasts(np.ones((3, 4)), np.arange(3))
    with pytest.raises(ValueError):
        M.actor_contrasts(np.ones((3, 4)), np.array([1, 1, 0]))


def cold_fixture():
    """Build the registered dyad/event graph and a signed numerical null world."""
    rng = np.random.default_rng(287)
    cases = []
    for u in range(24):
        v = (u + 1) % 24
        for w in (0, 1):
            cases.append(
                {
                    "case_id": f"c-{u}-{w}",
                    "writer_id": f"p-{u}",
                    "partner_id": f"p-{v}",
                    "window": w,
                    "writer_comments": [f"e-{u}-{w}-{i}" for i in (0, 1)],
                    "partner_comments": [f"e-{v}-{w}-{i}" for i in (0, 1)],
                }
            )
    levels = rng.normal(size=(2, 2, 48, 5))
    partner = np.array([2 * ((u + 1) % 24) + w for u in range(24) for w in (0, 1)])
    levels -= levels[:, :, partner]
    components = np.concatenate((levels.sum(1, keepdims=True), levels), axis=1)
    components = np.concatenate((components, components[:1] - components[1:2]), 0)
    bank = [
        {"item": j, "target": j // 10, "pole": 1 if j % 10 < 5 else -1}
        for j in range(50)
    ]
    logits = np.zeros((48, 2, 2, 2, 50, 2))
    return {
        "version": "HM287-v1",
        "cases": cases,
        "source_components": components.tolist(),
        "item_logits": logits.tolist(),
        "item_bank": bank,
    }


def test_cold_generic_response_does_not_create_person_bound_contact():
    record = cold_fixture()
    values = np.asarray(record["item_logits"])
    values[..., 1] = 3.0
    record["item_logits"] = values.tolist()
    result = M.cold(record)
    contact = np.asarray(result["logodds"]["global_common_contacts"])
    assert contact[:, :, 3] == pytest.approx(np.zeros((3, 3, 5)))
    assert result["not_individual_human_trait_predictions"]


def test_cold_owner_relative_contact_retains_source_direction():
    record = cold_fixture()
    source = np.asarray(record["source_components"])
    gamma = source[0, 2, :, 0]
    values = np.asarray(record["item_logits"])
    for o, t, d in np.ndindex(2, 2, 2):
        values[:, o, t, d, :, 1] = ((2 * o - 1) * (2 * t - 1) * gamma)[:, None]
    record["item_logits"] = values.tolist()
    result = M.cold(record)
    contact = np.asarray(result["logodds"]["global_common_contacts"])
    assert contact[0, 2, 3, 0] == pytest.approx(np.mean(gamma**2))
    record["source_components"] = (-source).tolist()
    opposite = M.cold(record)
    assert np.asarray(opposite["logodds"]["global_common_contacts"]) == pytest.approx(
        -contact
    )


def test_cold_event_reuse_cannot_change_true_author():
    record = cold_fixture()
    record["cases"][2]["writer_comments"][0] = record["cases"][0]["writer_comments"][0]
    with pytest.raises(ValueError, match="ledger author"):
        M.cold(record)


def test_cold_rejects_application_labels_and_broken_source_closure():
    record = cold_fixture()
    record["application_labels"] = [0] * 24
    with pytest.raises(ValueError, match="registered numerical"):
        M.cold(record)
    del record["application_labels"]
    record["source_components"][0][0][0][0] += 1
    with pytest.raises(ValueError, match="component identity"):
        M.cold(record)
