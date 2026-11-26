#!/usr/bin/env python3
"""Check evidence separation and interoperability across carriers."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from grain_core.evidence_panel import (
    Coverage,
    EvidenceEvent,
    EvidencePanel,
    Opportunity,
    run_panel,
)
from grain_core.evidence_text import TextRecord, native_text_evidence
from grain_core.liquid_instrument import fit_atlas, locate
from jsonschema import Draft202012Validator
from threadpoolctl import threadpool_limits


def event(id_, window, actor, context, value, channel="text"):
    return EvidenceEvent(
        id_, window, actor, context, channel, None if value is None else [value]
    )


def test_coverage_gates_absence_but_never_positive_evidence():
    args = {"text": ["coord"]}, [event("e", "now", "a", "essay", None)]
    v = EvidencePanel(*args).participation("now", "a", "forum")
    assert (
        v["state"] == "target_coverage_unknown"
        and v["elsewhere"][0]["context"] == "essay"
    )
    p = EvidencePanel(*args, coverage=[Coverage("now", True, "forum")])
    assert p.participation("now", "a", "forum")["state"] == "observed_elsewhere_only"
    assert (
        p.participation("now", "b", "forum")["state"]
        == "not_observed_at_target_elsewhere_unknown"
    )
    assert p.participation("now", "a", "essay")["state"] == "observed_at_target"
    p = EvidencePanel(*args, coverage=[Coverage("now", True)])
    assert p.participation("now", "b", "forum")["state"] == "not_observed_in_feed"


def test_history_is_explicit_and_future_feed_irrelevant():
    events = [
        event("1", "before", "a", "writing", 0.2),
        event("2", "after", "a", "writing", 0.7),
    ]
    cov = [Coverage("after", True, "forum")]
    q = lambda p: p.participation("after", "a", "forum", history=["before"])
    x = q(EvidencePanel({"text": ["x"]}, events, coverage=cov))
    events.append(event("3", "future", "a", "forum", 1))
    p = EvidencePanel({"text": ["x"]}, events, coverage=cov)
    assert x == q(p)
    assert x["elsewhere"][0]["previously_observed"]
    with pytest.raises(ValueError, match="history"):
        p.participation("after", "a", "forum", history=["after"])


def test_unscored_records_and_other_channels_stay_out_of_mean():
    p = EvidencePanel(
        {"text": ["x"], "action": ["y"]},
        [
            event("1", "before", "a", "forum", 1),
            event("2", "after", "a", "forum", 3),
            event("3", "after", "a", "forum", None),
            event("4", "after", "b", "forum", None),
            event("5", "after", "a", "forum", 60, "action"),
        ],
    )
    c = p.compare("before", "after", "forum", "text", weighting="event")
    assert c["support"][1]["records"] == 3 and c["support"][1]["scored_records"] == 1
    assert (
        c["support"][1]["actors_with_evidence"] == 2
        and c["support"][1]["scored_actors"] == 1
    )
    assert c["decomposition"]["within_incumbent_shift"] == [2]
    assert p.compare("before", "after", "forum", "action")["decomposition"] is None


def test_weighting_and_no_shared_author_are_explicit():
    events = [
        event("a", "t1", "a", "board", 2),
        event("b", "t1", "b", "board", 4),
        event("c", "t2", "c", "board", 1),
        event("d", "t2", "d", "board", 3),
        event("d2", "t2", "d", "board", 3),
    ]
    p = EvidencePanel({"text": ["x"]}, events)
    a = p.compare("t1", "t2", "board", "text")
    b = p.compare("t1", "t2", "board", "text", weighting="event")
    assert (
        a["status"] == "no_stayer_reference"
        and a["decomposition"]["within_incumbent_shift"] is None
    )
    assert a["decomposition"]["total_delta"] == [-1]
    assert b["decomposition"]["total_delta"] == pytest.approx([-2 / 3])


def test_bounds_are_sharp_on_logged_menu_under_missing_outcomes():
    values = [(True, True), (True, False), (True, None), (False, None), (None, None)]
    p = EvidencePanel(
        {},
        [],
        opportunities=[
            Opportunity(str(i), "w", "u", "menu", "take", e, v)
            for i, (e, v) in enumerate(values)
        ],
    )
    c = p.opportunity_card("w", "menu", "take")
    all_allowed = [
        (1 + x + (y is True)) / (3 + (y is not None))
        for x, y in itertools.product((0, 1), (None, False, True))
    ]
    assert (
        c["all_logged_bounds"] == [min(all_allowed), max(all_allowed)] == [0.25, 0.75]
    )
    assert (
        c["known_eligible_bounds"] == [1 / 3, 2 / 3] and c["no"] == c["ineligible"] == 1
    )
    assert (
        p.opportunity_card("w", "menu", "take", actor="v")["all_logged_bounds"] is None
    )


def test_invalid_boundary_inputs_are_rejected():
    row = event("dup", "w", "a", "board", 1)
    with pytest.raises(ValueError, match="duplicate"):
        EvidencePanel({"text": ["reading"]}, [row, row])
    with pytest.raises(ValueError, match="finite vector"):
        EvidencePanel({"text": ["reading"]}, [event("x", "w", "a", "board", np.nan)])
    with pytest.raises(ValueError, match="eligible opportunity"):
        EvidencePanel(
            {},
            [],
            opportunities=[Opportunity("1", "w", "a", "board", "q", None, False)],
        )
    with pytest.raises(ValueError, match="conflicts"):
        EvidencePanel(
            {}, [], coverage=[Coverage("w", True), Coverage("w", False, "board")]
        )


def test_two_readers_of_one_observation_do_not_inflate_participation():
    first = EvidenceEvent("v1", "w", "a", "board", "first", [0.2], "utterance-1")
    second = EvidenceEvent("v2", "w", "a", "board", "second", [0.6], "utterance-1")
    channels = {"first": ["x"], "second": ["y"]}
    p = EvidencePanel(channels, [first, second])
    assert p.participation("w", "a", "board")["target_records"] == 1
    with pytest.raises(ValueError, match="inconsistent address"):
        EvidencePanel(
            channels,
            [
                first,
                EvidenceEvent("v2", "w", "b", "board", "second", [0.6], "utterance-1"),
            ],
        )


def test_native_adapter_matches_frozen_placer_without_imputing_out_of_vocab():
    corpus = [
        f"I checked the route near the {x}."
        for x in ("store", "school", "bridge", "port")
    ] + [f"We agreed to fix the {x}." for x in ("sign", "door", "roof", "window")]
    with threadpool_limits(limits=1):
        atlas = fit_atlas(corpus, 2, "gmm_soft")
        values, covered = locate(atlas, [corpus[0], "zxqvgx"])
        names, rows = native_text_evidence(
            atlas,
            [
                TextRecord("1", "source", "a", "essay", corpus[0]),
                TextRecord("2", "target", "a", "essay", "zxqvgx"),
            ],
        )
    assert covered.tolist() == [True, False]
    assert rows[1].values is None and np.array_equal(
        values[0], np.array(rows[0].values)
    )
    assert tuple(next(iter(names.values()))) == tuple(atlas["labels"])
    p = EvidencePanel(names, rows)
    assert p.participation("target", "a", "essay")["state"] == "observed_at_target"
    assert (
        p.compare("source", "target", "essay", rows[0].channel)["decomposition"] is None
    )


def test_same_actor_context_shift_has_exact_accounting_when_observed():
    rows = [
        event("1", "before", "a", "essay", 4),
        event("2", "before", "a", "forum", 2),
        event("3", "after", "a", "forum", 3),
    ]
    p = EvidencePanel({"text": ["x"]}, rows)
    v = p.context_shift("before", "after", "a", "essay", "forum", "text")
    assert v["status"] == "decomposed"
    assert (v["total"], v["context_at_baseline"], v["within_destination"]) == (
        [-1],
        [-2],
        [1],
    )


def test_offline_demo_is_strict_json_and_reuses_the_cohort_operator():
    root = Path(__file__).parents[1]
    schema = json.loads((root / "schemas/evidence_panel.v1.schema.json").read_text())
    sample = json.loads((root / "examples/evidence_panel_demo.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(sample)
    out = run_panel(sample)["results"]
    json.dumps(out, allow_nan=False)
    assert out["where_a_went"]["state"] == "observed_elsewhere_only"
    assert out["forum_change"]["decomposition"]["identity_max_abs_err"] < 1e-12
    assert out["uncaptured_move"]["within_destination"] is None
    assert out["choice_not_imputed"]["all_logged_bounds"] == [0.25, 0.75]
