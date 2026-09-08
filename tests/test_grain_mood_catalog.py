#!/usr/bin/env python3
"""Strict item means and missingness in the separate mood branch."""

import pandas as pd
import pytest

from grain_research.vector_mechanisms.mood_catalog import (
    PHASES,
    audit_mood,
    recompute_mood,
)


def test_adapted_mood_scoring_and_no_imputation():
    rows = [
        {
            "participant_id": "1234567",
            "phase": phase,
            "item_id": f"mood_{i:02d}",
            "source_scale": "PANAS_NA" if i <= 8 else "POMS_SHORT",
            "response": 2.0 if i <= 8 else 4.0,
        }
        for phase in PHASES
        for i in range(1, 14)
    ]
    frame = pd.DataFrame(rows)
    scores, _ = recompute_mood(frame, {"1234567": "P001"})
    assert len(scores) == 12
    assert set(scores[scores.composite == "adapted_na8"].score) == {2.0}
    assert set(scores[scores.composite == "adapted_dejected5"].score) == {4.0}
    frame.loc[0, "response"] = float("nan")
    scores, _ = recompute_mood(frame, {"1234567": "P001"})
    assert scores.score.isna().sum() == 2
    with pytest.raises(ValueError, match="Duplicate"):
        recompute_mood(pd.concat([frame, frame.iloc[:1]]), {"1234567": "P001"})
    with pytest.raises(PermissionError):
        audit_mood("unused")
