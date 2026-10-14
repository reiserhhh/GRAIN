#!/usr/bin/env python3
"""The recovery adapter changes status only, never immutable science files."""

import pytest

from scripts.run_grain_author_belief_hm281_recovery import E, runtime_writer


def test_scoped_status_replacement_science_stays_exclusive(tmp_path):
    write = runtime_writer(E.R.write, tmp_path)
    write(tmp_path / "runtime.json", {"state": "PREPARED"})
    write(tmp_path / "runtime.json", {"state": "RUNNING"})
    assert E.R.read(tmp_path / "runtime.json")["state"] == "RUNNING"
    write(tmp_path / "science.json", {"x": 1})
    with pytest.raises(FileExistsError):
        write(tmp_path / "science.json", {"x": 2})
