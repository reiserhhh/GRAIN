"""R36 stage-gate ordering."""

import pytest

from grain_research.gptpro_foundation_v3.r36_gates import STAGES, gates


def test_first_non_pass_and_complete():
    rec = dict.fromkeys(STAGES, "PASS")
    assert gates(rec)["complete"]
    rec["admission"] = "FAIL"
    out = gates(rec)
    assert out["first_non_pass"] == "admission" and not out["complete"]


def test_unknown_status_rejected():
    with pytest.raises(ValueError):
        gates({"replicated_regularity": "MAYBE"})
