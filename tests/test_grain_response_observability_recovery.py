#!/usr/bin/env python3
"""The status adapter must never overwrite a scientific result or source."""

import pytest

from scripts import recover_grain_response_observability_hm127 as R


def test_only_explicit_status_is_mutable(tmp_path):
    (tmp_path / "01").mkdir()
    writer = R.StatusWriter(R.H, tmp_path)
    for name in ("status.json", "01/capture_status.json"):
        writer.write(tmp_path / name, {"n": 1})
        writer.write(tmp_path / name, {"n": 2})
        assert R.H.load(tmp_path / name) == {"n": 2}
    result = tmp_path / "01/scientific.json"
    writer.write(result, {"n": 1})
    with pytest.raises(FileExistsError):
        writer.write(result, {"n": 2})
    assert R.H.load(result) == {"n": 1}
    assert not list(tmp_path.rglob(".hm127_status_*"))


def test_writer_rejects_nonfinite_before_replacing_status(tmp_path):
    writer = R.StatusWriter(R.H, tmp_path)
    writer.write(tmp_path / "status.json", {"n": 1})
    with pytest.raises(ValueError):
        writer.write(tmp_path / "status.json", {"n": float("nan")})
    assert R.H.load(tmp_path / "status.json") == {"n": 1}


def test_proxy_scope_is_restored_after_failure(tmp_path):
    p, a = R.P.H, R.A.H
    with pytest.raises(RuntimeError), R.mutable_status(tmp_path):
        assert isinstance(R.P.H, R.StatusWriter)
        assert isinstance(R.A.H, R.StatusWriter)
        raise RuntimeError("synthetic failure")
    assert R.P.H is p and R.A.H is a
