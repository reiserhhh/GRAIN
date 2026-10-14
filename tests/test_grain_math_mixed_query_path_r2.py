#!/usr/bin/env python3
"""HM35 single-expression device repair leaves the finite scientific protocol intact."""

from scripts import run_grain_math_mixed_query_path_r2 as R


def test_companion_preserves_budget_and_primary_source():
    scope = R.runtime()
    assert R.digest(R.ORIGINAL) == R.EXPECTED
    assert R.ORIGINAL.read_text().count(R.OLD) == 1
    assert scope["OUT"].name == "mixed_query_path_r2"
    assert scope["SPEC"].name == "grain_math_mixed_query_path_v1.json"


def test_actual_mps_device_label_normalization():
    import torch

    device = torch.device("mps:0")
    assert str(device) != "mps"
    assert device.type == "mps"
