#!/usr/bin/env python3
"""Explicit design inventories exclude AppleDouble and unplanned result files."""

import pytest

from scripts.analyze_grain_native_material_reader import planned_packets


def test_no_filesystem_glob(tmp_path):
    paths = planned_packets(tmp_path, [{"index": 0}, {"index": 3}], ["q1", "q2"])
    assert [p.name for p in paths] == [
        "a00_q1.json",
        "a00_q2.json",
        "a03_q1.json",
        "a03_q2.json",
    ]
    assert all(not p.name.startswith("._") for p in paths)


def test_duplicate_design_rejected(tmp_path):
    with pytest.raises(ValueError):
        planned_packets(tmp_path, [{"index": 0}, {"index": 0}], ["q"])
