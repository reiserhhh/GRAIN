#!/usr/bin/env python3
"""Report units, roles, incidence and finite joint-law counterexamples."""

import numpy as np
import pytest

from grain_research.construction import math_state_exposure as S
from tests.test_grain_math_purpose_accounts import word


def two_conditions():
    """Return synthetic UD words, not an empirical parser validation."""
    text = "I feel uneasy when Tom must check figures and Mary must organize teams."
    specs = [
        ("I", 1, 2, "nsubj", "PRON"),
        ("feel", 2, 0, "root", "VERB"),
        ("uneasy", 3, 2, "xcomp", "ADJ"),
        ("when", 4, 7, "advmod", "ADV"),
        ("Tom", 5, 7, "nsubj", "PROPN"),
        ("must", 6, 7, "aux", "AUX"),
        ("check", 7, 2, "advcl", "VERB"),
        ("figures", 8, 7, "obj", "NOUN"),
        ("and", 9, 12, "cc", "CCONJ"),
        ("Mary", 10, 12, "nsubj", "PROPN"),
        ("must", 11, 12, "aux", "AUX"),
        ("organize", 12, 7, "conj", "VERB"),
        ("teams", 13, 12, "obj", "NOUN"),
    ]
    words = [word(text, *s) for s in specs]
    words[10]["start"] = text.index("must", words[5]["end"])
    words[10]["end"] = words[10]["start"] + 4
    return text, words


def test_one_report_two_other_actor_conditions():
    text, words = two_conditions()
    data = S.observe_reports([words], text)
    assert len(data["reports"]) == 1
    report = data["reports"][0]
    assert report["status"] == "ok"
    assert report["experiencer"] == "self"
    assert len(report["triggers"]) == 2
    assert {t["actor"] for t in report["triggers"]} == {"other"}
    result = S.report_capacity([[1, -1], [-1, 1]], report)
    assert result["report_mass"] == 1
    assert result["hard_coexposure"] == 1
    assert result["same_trigger_coexposure"] == 0
    assert S.report_capacity([[1, -1], [-1, 1]], report, actor_filter=True)[
        "coverage"
    ] == [0, 0]


def test_duplicate_and_permutation_invariant_but_flat_sum_not():
    text, words = two_conditions()
    report = S.observe_reports([words], text)["reports"][0]
    x = [[0.2, -0.2], [0.6, -0.4]]
    base = S.report_capacity(x, report)
    assert (
        S.report_capacity(x[::-1], {**report, "triggers": report["triggers"][::-1]})[
            "coverage"
        ]
        == base["coverage"]
    )
    duplicate = {**report, "triggers": report["triggers"] + report["triggers"]}
    result = S.report_capacity(x + x, duplicate)
    assert result["coverage"] == base["coverage"]
    assert result["flat_sum"] != base["flat_sum"]
    assert result["noisy_or_control"] != base["noisy_or_control"]


def test_other_experiencer_is_unobserved_not_zero_author():
    text, words = two_conditions()
    report = S.observe_reports([words], text)["reports"][0]
    assert (
        S.report_capacity([[1, 1], [1, 1]], {**report, "experiencer": "other"})[
            "coverage"
        ]
        is None
    )


def test_report_sign_not_condition_sign_or_intensity():
    text, words = two_conditions()
    report = S.observe_reports([words], text)["reports"][0]
    result = S.report_capacity([[1, -1], [-1, 1]], {**report, "state_polarity": -1})
    assert result["coverage"] == [1, 1]
    assert result["signed_expression"] == [-1, -1]


def test_missing_trigger_actor_keeps_report_unresolved():
    text, words = two_conditions()
    words[9]["deprel"] = "obl"
    words[11]["deprel"] = "advcl"
    report = S.observe_reports([words], text)["reports"][0]
    assert report["status"] == "unresolved"


def test_incidence_not_identified_by_trigger_marginals():
    w = S.incidence_witness()
    assert w["flat_sum_error"] == 0
    assert w["joint_report_coexposure"] == 0.5
    assert w["distributed_report_coexposure"] == 0


def test_singletons_do_not_identify_joint_law():
    w = S.joint_extension_witness()
    assert w["singleton_error"] == 0
    assert w["pair_alternative"] == w["duplicate_pair_alternative"]
    assert w["pair_alternative"] > w["pair_max"]


def test_invalid_score_shape_rejected():
    text, words = two_conditions()
    report = S.observe_reports([words], text)["reports"][0]
    with pytest.raises(ValueError):
        S.report_capacity([[np.nan, 1], [1, 1]], report)
    with pytest.raises(ValueError):
        S.report_capacity([[1, 1]], report)


def test_revision_replaces_all_conditions_alias_adds_no_mass():
    text, words = two_conditions()
    report = S.observe_reports([words], text)["reports"][0]
    a = {
        "record_id": "a",
        "event_id": "e",
        "author_id": "u",
        "condition": 0,
        "operation": "new",
        "reports": [report],
    }
    alias = {**a, "record_id": "alias", "parent_id": "a", "operation": "alias"}
    assert len(S.replay_reports([a, alias])) == 1
    b = {
        **a,
        "record_id": "b",
        "parent_id": "a",
        "operation": "revise",
        "reports": [{**report, "triggers": report["triggers"][:1]}],
    }
    result = S.replay_reports([a, b])
    assert len(result) == 1 and len(result[0]["triggers"]) == 1
    assert result[0]["report_id"].startswith("b/")
    assert result[0]["triggers"][0]["report_id"] == result[0]["report_id"]


def test_ambiguous_root_conj_retains_two_scopes_without_score_selection():
    from grain_research.construction import math_state_scope as T

    text, words = two_conditions()
    words[11]["head"] = 2
    data = {"words": [words], **S.observe_reports([words], text)}
    alternatives = T.alternatives(data)
    assert len(alternatives[0]["outside"]["triggers"]) == 1
    assert len(alternatives[0]["inside_candidate"]["triggers"]) == 2
    assert not alternatives[0]["unique_scope"]
    assert not alternatives[0]["selection_uses_target_score"]
    assert alternatives[0]["candidate_nodes"][0]["actor"] == "other"
    candidate = alternatives[0]["inside_candidate"]
    assert candidate["status"] == "unresolved_scope"
    assert S.report_capacity([[1, -1], [-1, 1]], candidate)["coverage"] is None


def test_candidate_shared_subject_not_inherited_from_experiencer():
    from grain_research.construction import math_state_scope as T

    text, words = two_conditions()
    words[11]["head"] = 2
    words[9]["deprel"] = "obl"
    data = {"words": [words], **S.observe_reports([words], text)}
    candidate = T.alternatives(data)[0]["candidate_nodes"][0]
    assert candidate["subject_origin"] == "candidate_condition_subject"
    assert candidate["actor"] == "other"
    assert not candidate["experiencer_inherited_as_actor"]


def test_material_addresses_match_literal_raw_words():
    source = [
        {
            "target": target,
            "action_index": ai,
            "object_index": 0,
            "object_type": 1,
            "role": "source",
            "action": f"verb{ai}",
            "object": f"the {target} objects",
            "bits": [1 if ai < 4 else -1, 1, 1, 1],
        }
        for target in S.TARGETS
        for ai in range(8)
    ]
    bank = S.material_bank(source)
    assert len(bank) == 832
    for r in bank:
        for report in r["expected_reports"]:
            assert r["text"][slice(*report["root_span"])] in {"feel", "feels"}
            for t in report["triggers"]:
                assert r["text"][slice(*t["action_span"])] == t["action"]
                assert r["text"][slice(*t["object_span"])] == t["object"]
