#!/usr/bin/env python3
import pytest
from grain_core import response_profile as rp
from grain_core.psychometric_open_stream import (
    calibrate_reader_misclassification as correct,
)


def card(statuses, frame="presented_opportunities", weights=None):
    ref = {
        "unit": "turn",
        "behavior": "adds a fact",
        "reader": "frozen",
        "condition_rule": "request",
        "frame": frame,
        "weights": weights or {"c": 1.0},
    }
    events = [
        {
            "id": str(i),
            "focal": "f",
            "dialogue": "s",
            "condition": "c",
            "status": s,
            "response_id": str(i) if s in rp.REPLIES else None,
        }
        for i, s in enumerate(statuses)
    ]
    return ref, events


CAL = {"c": {"sensitivity": 0.8, "fpr": 0.2}}


def run(statuses, **kwargs):
    r, e = card(statuses, **kwargs)
    c = rp.measure_calibrated(r, e, CAL, calibration_denominator="among_replies")[
        "cards"
    ][0]
    return c, c["reader_correction"]


def test_half_spoke_correct_reply_reader_before_restoring_denominator():
    c, r = run(["support"] * 5 + ["refute"] * 5 + ["no_reply"] * 10)
    assert r["natural_corrected_bounds"] == pytest.approx([0.25, 0.25])
    assert c["natural"] == [0.25, 0.25]
    with pytest.raises(ValueError, match="explicit"):
        correct(c, CAL)
    with pytest.raises(ValueError, match="among|observed replies"):
        correct(c, CAL, calibration_denominator="per_opportunity")


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["no_reply"] * 5, [0.0, 0.0]),
        (["support"] * 5 + ["refute"] * 5, [0.5, 0.5]),
        (
            ["support"] * 5 + ["refute"] * 5 + ["reply_unknown"] * 5 + ["no_reply"] * 5,
            [0.25, 0.5],
        ),
        (["reply_unknown"] * 2, [0.0, 1.0]),
        (["unresolved", "no_reply"], [0.0, 0.5]),
    ],
)
def test_structural_zeros_unknown_mass_and_shared_mix(statuses, expected):
    _, r = run(statuses)
    assert r["natural_corrected_bounds"] == pytest.approx(expected)
    assert r["standardized_corrected_bounds"] == pytest.approx(expected)
    assert r["mix_gap_corrected_bounds"] == [0.0, 0.0]


def test_unsupported_condition_keeps_reference_mass_and_unknown_exposure_excluded():
    _, r = run(["no_reply", "opportunity_unconfirmed"], weights={"c": 0.5, "d": 0.5})
    assert r["standardized_corrected_bounds"] == [0.0, 0.5]
    assert r["mix_gap_corrected_bounds"] == [-0.5, 0.0]
    assert r["unconfirmed_excluded"] == 1


def test_if_everyone_speaks_the_two_frames_agree():
    _, a = run(["support"] * 4 + ["refute"] * 6)
    _, b = run(["support"] * 4 + ["refute"] * 6, frame="observed_replies")
    assert a["natural_corrected_bounds"] == pytest.approx(b["natural_corrected_bounds"])


def test_reply_only_missing_condition_is_not_zero():
    ref, events = card(["unresolved"], frame="observed_replies", weights={"c": .5, "d": .5})
    ref["contrast"] = ["c", "d"]
    row = rp.measure_calibrated(ref, events, CAL, calibration_denominator="among_replies")["cards"][0]
    result = row["reader_correction"]
    assert result["mix_gap_corrected_bounds"] == [-.5, .5]
    assert result["contrast_calibration"]["corrected_bounds"] == [-1., 1.]
    assert result["contrast_calibration"]["dif_decomposition"]["status"] == "unsupported"
