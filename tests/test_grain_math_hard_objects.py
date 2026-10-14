#!/usr/bin/env python3
"""No-class and no-group metrics stay absent, not false numerical evidence."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "hard_probe", SCRIPTS / "probe_grain_math_local_hard_objects.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_one_class_auc_and_absent_controls_are_null():
    bits = np.array(
        [[1, 1, 1, 1, 1], [1, -1, 1, 1, 1], [-1, 1, 1, 1, 1], [-1, -1, 1, 1, 1]]
    )
    score = np.array([1.0, -1.0, 0.0, 0.0])
    result = module.probe_metrics(score, bits[:, :4].astype(float), bits)
    assert result["ownership_auc"] is None
    assert result["actuality_auc"] is None
    assert result["other_polarity_gain"] is None
    assert result["hypothetical_polarity_gain"] is None
    assert result["unit_rmse"] == 0
    assert result["qualification_tp"] == 2
    assert result["qualification_fp"] == 0
