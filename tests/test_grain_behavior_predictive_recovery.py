#!/usr/bin/env python3
"""The metadata recovery retains exact registered data and unchanged algebra."""

from pathlib import PosixPath

from scripts.recover_grain_behavior_predictive_hm261 import RegisteredCapturePath


def test_capture_registration_excludes_metadata_and_unregistered_files(tmp_path):
    """Metadata cannot become a sample; unrelated numeric files cannot expand scope."""
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "._0000.npz").write_bytes(bytes.fromhex("00051607"))
    (captures / "9999.npz").write_bytes(b"not registered")
    registered = list(RegisteredCapturePath(captures).glob("*.npz"))
    assert len(registered) == 672
    assert registered == [captures / f"{i:04d}.npz" for i in range(672)]
    assert all(not p.name.startswith("._") for p in registered)


def test_other_glob_and_path_arithmetic_unchanged(tmp_path):
    """Only the known capture discovery call is adapted, not scientific paths."""
    (tmp_path / "bank.json").write_text("{}")
    path = RegisteredCapturePath(tmp_path)
    assert list(path.glob("*.json")) == [tmp_path / "bank.json"]
    assert isinstance(path / "captures", RegisteredCapturePath)
    assert PosixPath(path) == tmp_path
