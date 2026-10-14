#!/usr/bin/env python3
"""Complete saved-array reader coverage, without model calls or real text."""

import json

import numpy as np
import pytest

from scripts import recover_grain_continuation_routes_hm259 as X
from scripts import run_grain_continuation_routes_hm259 as F


def fixture(tmp_path):
    """Create two artificial uses and every route/context/query block."""
    base, native = tmp_path / "new", tmp_path / "old"
    for directory in (base, native):
        (directory / "captures").mkdir(parents=True)
        cases = [{"unit_id": i, "target_ids": [11, 12]} for i in range(2)]
        (directory / "cases.json").write_text(json.dumps(cases))
    (base / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "parent": str(native),
                    "new_routes": ["closed", "direct", "relay"],
                }
            }
        )
    )
    for unit in range(2):
        for mode in range(2):
            for local, start in enumerate(range(0, 11, 4)):
                shape = (min(4, 11 - start), 2)
                np.savez(
                    native / f"captures/{6 * unit + 3 * mode + local:04d}.npz",
                    unit=unit,
                    mode=mode,
                    start=start,
                    chosen=np.full(shape, 100 + unit),
                    partition=np.zeros(shape),
                )
                for index, route in enumerate(("closed", "direct", "relay")):
                    np.savez(
                        base / F.capture_name(unit, mode, route, start),
                        unit=unit,
                        mode=mode,
                        route=route,
                        start=start,
                        chosen=np.full(shape, 10 * unit + index),
                        partition=np.zeros(shape),
                    )
    return base


def test_all_families_returned_without_container_overwrite(tmp_path):
    """An outer result must remain a list of complete dictionaries."""
    base = fixture(tmp_path)
    cases, families = X.readings(base)
    assert len(cases) == len(families) == 2
    for i, family in enumerate(families):
        assert set(family) == {"closed", "direct", "relay", "native"}
        assert family["native"]["chosen"].shape == (2, 11, 2)
        assert np.all(family["native"]["chosen"] == 100 + i)
        assert np.all(family["relay"]["chosen"] == 10 * i + 2)


def test_mutated_saved_address_rejected(tmp_path):
    """A saved block at the wrong author address cannot quietly contribute."""
    base = fixture(tmp_path)
    path = base / F.capture_name(0, 0, "closed", 0)
    np.savez(
        path,
        unit=1,
        mode=0,
        route="closed",
        start=0,
        chosen=np.zeros((4, 2)),
        partition=np.zeros((4, 2)),
    )
    with pytest.raises(ValueError, match="address"):
        X.readings(base)
