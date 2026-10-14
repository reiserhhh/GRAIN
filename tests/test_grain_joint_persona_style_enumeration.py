#!/usr/bin/env python3
"""Filesystem metadata exclusion must not delete genuine prefixed data."""

from scripts.run_grain_joint_persona_style_enumeration_hm150 import ScientificOutputPath


def test_scoped_glob_ignores_only_real_appledouble(tmp_path):
    (tmp_path / "000.json").write_text("[]")
    (tmp_path / "._000.json").write_bytes(b"\x00\x05\x16\x07metadata")
    (tmp_path / "._genuine.json").write_text("[]")
    result = {p.name for p in ScientificOutputPath(tmp_path).glob("*.json")}
    assert result == {"000.json", "._genuine.json"}
    assert (tmp_path / "._000.json").exists()


def test_descendants_keep_scoped_filter(tmp_path):
    nested = tmp_path / "observations"
    nested.mkdir()
    (nested / "000.npz").write_bytes(b"not needed")
    (nested / "._000.npz").write_bytes(b"\x00\x05\x16\x07metadata")
    assert [
        p.name for p in (ScientificOutputPath(tmp_path) / "observations").glob("*.npz")
    ] == ["000.npz"]
