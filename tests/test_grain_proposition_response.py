#!/usr/bin/env python3
"""Synthetic unit tests for proposition binding and explicit missingness."""

import pytest

from grain_research.proposition_response import (
    bind_reply,
    response_coordinates,
    response_profile,
)
from scripts.run_grain_proposition_response import public_bank


def frame(actor="respondent", channel="past_action_report", sign=1):
    """Create an explicit test proposition, not a production parser fallback."""
    return {
        "id": "p:0:2",
        "target_relevant": True,
        "question": True,
        "actor": actor,
        "channel": channel,
        "sign": sign,
        "ambiguous_negation": False,
        "source_scoped": False,
        "conditions": [],
    }


@pytest.mark.parametrize("reply,sign", [("Yes.", 1), ("No.", -1)])
@pytest.mark.parametrize("mode", ["past_action_report", "intention"])
def test_bare_binding(reply, sign, mode):
    """The same answer is routed according to its antecedent, not a type guess."""
    result = bind_reply("Question?", reply, [frame(channel=mode)], [])
    coordinates, ok = response_coordinates(result)
    assert ok and coordinates[mode] == sign


def test_other_not_self():
    result = bind_reply("Question?", "Yes.", [frame(actor="other")], [])
    coordinates, ok = response_coordinates(result)
    assert ok and sum(coordinates.values()) == 0


@pytest.mark.parametrize("antecedents", [[], [frame(), frame()], [frame(sign=-1)]])
def test_ambiguous_is_missing(antecedents):
    result = bind_reply("Question?", "Yes.", antecedents, [])
    coordinates, ok = response_coordinates(result)
    assert not ok and all(v is None for v in coordinates.values())


def test_profile_missing_not_opposition():
    profile = response_profile(
        [{"condition": 0, "score": 1}, {"condition": 0, "score": None}], [0, 1]
    )
    assert profile["0"] == {"opportunities": 2, "observed": 1, "score": 1}
    assert profile["1"]["score"] is None


def test_public_crossing():
    bank = public_bank()
    assert len(bank) == 40
    main = [r for r in bank if r["part"] != "challenge"]
    assert (
        len(
            {
                (r["wording"], r["condition"], r["mode"], r["actor"], r["sign"])
                for r in main
            }
        )
        == 32
    )


def test_source_calibration_uses_numpy_sparse_mask():
    """Exercise the failed pandas/SciPy boundary and exclude held vocabulary."""
    import pandas as pd

    from scripts.complete_grain_proposition_response import source_calibrate

    rows = []
    for i, (part, condition, text) in enumerate(
        [
            ("source", 0, "deadline approaches"),
            ("source", 1, "instructions missing"),
            ("held", 0, "novelword deadline"),
            ("held", 1, "novelword instructions"),
        ]
    ):
        rows.append(
            {
                "id": str(i),
                "part": part,
                "condition": condition,
                "condition_text": text,
                "sign": 1,
                "mode": "past_action_report",
                "actor": "respondent",
            }
        )
    table, calibration = source_calibrate(
        pd.DataFrame(rows),
        pd.DataFrame(
            {
                "audit_id": [str(i) for i in range(4)],
                "p_entailment": [0.9, 0.8, 0.7, 0.6],
            }
        ),
    )
    assert table.estimated_condition.tolist() == [0, 1, 0, 1]
    assert "novelword" not in calibration["vocabulary"]
    assert calibration["source_ids"] == ["0", "1"]


def test_missing_value_bounds_exhaustive():
    """Enumerate all bounded binary completions, including unknown assignment."""
    import itertools

    import numpy as np

    from grain_research.proposition_scope_accounting import mean_bounds

    for observed in [[1, -1], [1, 1], [-1, -1], []]:
        for missing in range(4):
            bound = mean_bounds(sum(observed), len(observed), missing)
            if not observed and not missing:
                assert bound["lower"] is None
                continue
            for k in range(missing + 1):
                for suffix in itertools.product([-1, 1], repeat=k):
                    if not observed and not suffix:
                        continue
                    value = float(np.mean(observed + list(suffix)))
                    assert bound["lower"] <= value <= bound["upper"]


def test_speaker_and_action_actor_are_separate():
    from grain_research.proposition_scope_accounting import role_account

    f = {
        "id": "a",
        "actor": "partner",
        "speaker": "respondent",
        "channel": "current_request_candidate",
        "source_scoped": False,
    }
    assert (
        role_account(f, explicit_addressee=False)["proposition_actor_candidate"]
        == "second_person_unresolved"
    )
    assert (
        role_account(f, explicit_addressee=True)["proposition_actor_candidate"]
        == "partner"
    )
    assert role_account(f, explicit_addressee=True)["speech_source"] == "respondent"
