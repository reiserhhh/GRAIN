#!/usr/bin/env python3
"""Status repair must not weaken scientific artifact write protection."""

import numpy as np
import pytest

from scripts.run_grain_diagnostic_law_hm141_io_repair import H, MutableStatusIO


def test_mutable_status_and_exclusive_outputs(tmp_path):
    io = MutableStatusIO(H)
    status = tmp_path / "status.json"
    io.write(status, {"state": "PREPARED"})
    io.write(status, {"state": "RUNNING"})
    assert H.load(status)["state"] == "RUNNING"
    frozen = tmp_path / "metric.json"
    io.write(frozen, {"score": 1})
    with pytest.raises(FileExistsError):
        io.write(frozen, {"score": 2})
    assert H.load(frozen) == {"score": 1}


def test_failed_serialization_preserves_old_checkpoint(tmp_path):
    io = MutableStatusIO(H)
    status = tmp_path / "status.json"
    io.write(status, {"state": "PREPARED"})
    with pytest.raises(ValueError):
        io.write(status, {"bad": np.nan})
    assert H.load(status) == {"state": "PREPARED"}
    assert not list(tmp_path.glob(".status-*"))
