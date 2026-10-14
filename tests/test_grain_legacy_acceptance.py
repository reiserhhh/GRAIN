#!/usr/bin/env python3
"""Legacy metric reconciliation must not silently homogenize score rules."""

import numpy as np
import pandas as pd
import pytest

from scripts.reconcile_grain_legacy_acceptance import (
    TRAITS,
    correlation,
    result_tables,
    score_big5,
)


def test_constant_is_unknown_not_zero():
    assert correlation([1, 1, 1], [1, 2, 3]) == (None, 3)


def test_metadata_sidecars_are_not_result_tables(tmp_path):
    for name in ("table.csv", "._table.csv", ".hidden.csv"):
        (tmp_path / name).touch()
    assert [p.name for p in result_tables(tmp_path)] == ["table.csv"]


def test_trait_exclusion_is_local_and_clipping_is_distinct():
    frame = pd.DataFrame(
        {
            f"{prefix}_{trait}": [0.1, 0.4, 0.8, 0.9]
            for trait in TRAITS
            for prefix in ("gt", "pred")
        }
    )
    frame.loc[3, "pred_Openness"] = 8
    raw = score_big5(frame, "raw")
    clean = score_big5(frame, "drop_oob_per_trait")
    clipped = score_big5(frame, "clip_0_1")
    assert clean[0]["n"] == 4 and clean[-1]["n"] == 3
    assert raw[-1]["n"] == clipped[-1]["n"] == 4
    assert not np.isclose(raw[-1]["pearson"], clipped[-1]["pearson"])
    assert np.isclose(clean[-1]["pearson"], 1)


def test_unknown_rule_fails():
    frame = pd.DataFrame(
        {
            f"{prefix}_{trait}": [0, 1, 2]
            for trait in TRAITS
            for prefix in ("gt", "pred")
        }
    )
    with pytest.raises(ValueError, match="Unknown"):
        score_big5(frame, "invented")
