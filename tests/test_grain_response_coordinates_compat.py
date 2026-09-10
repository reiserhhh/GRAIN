#!/usr/bin/env python3
"""Pandas3 runtime compatibility changes categorical dtype, not numeric baselines."""

from io import StringIO

import numpy as np
import pandas as pd


def test_object_string_context_preserves_numeric_values_and_keys():
    text = "cohort,arm,action_r,actions\nnew8,identity,0.123456789012345,32\n"
    with pd.option_context("future.infer_string", True):
        modern = pd.read_csv(StringIO(text))
    with pd.option_context("future.infer_string", False):
        compatible = pd.read_csv(StringIO(text))
        keys = [c for c in compatible if compatible[c].dtype == object]
        assert keys == ["cohort", "arm"]
        np.testing.assert_array_equal(
            compatible[["action_r", "actions"]], modern[["action_r", "actions"]]
        )
        assert compatible[keys].to_dict("records") == modern[keys].to_dict("records")
