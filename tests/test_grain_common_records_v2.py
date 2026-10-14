import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cr2", ROOT / "scripts/build_grain_common_records_v2.py")
cr2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr2)


def test_failing_child_is_reported_as_failure(tmp_path):
    bad = tmp_path / "test_always_fails.py"
    bad.write_text("def test_x():\n    assert False\n")
    ok, text = cr2.run_tests([str(bad)])
    assert not ok and "exit_code=1" in text


def test_passing_child(tmp_path):
    good = tmp_path / "test_always_passes.py"
    good.write_text("def test_x():\n    assert True\n")
    ok, _ = cr2.run_tests([str(good)])
    assert ok


def test_failed_tests_block_packaging(tmp_path):
    pkt = tmp_path / "X" / "p_r1"
    pkt.mkdir(parents=True)
    (pkt / "decision.md").write_text("d")
    bad = tmp_path / "test_bad.py"
    bad.write_text("def test_x():\n    assert False\n")
    spec_ = {"protocols": [], "state": "PASS", "headline": "h", "model_calls": "0", "tests": [str(bad)]}
    with pytest.raises(cr2.PackagingError):
        cr2.build_record("X/p_r1", spec_, "t", root=tmp_path)
    assert not (pkt / "common_record_r1" / "output_manifest.json").exists()


def test_disallowed_state_rejected(tmp_path):
    (tmp_path / "X" / "p_r1").mkdir(parents=True)
    with pytest.raises(cr2.PackagingError):
        cr2.build_record("X/p_r1", {"protocols": [], "state": "DONE", "headline": "", "model_calls": "0", "tests": []}, "t", root=tmp_path)
