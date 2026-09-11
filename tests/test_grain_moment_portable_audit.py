#!/usr/bin/env python3
"""Post-primary portability checks without changing scientific arrays or units."""

import io

import pandas as pd
import pytest

from scripts.audit_grain_joint_moment_nullspace import logical_files


def test_recognized_metadata_is_ignored_without_deleting_it(tmp_path):
    (tmp_path / "result.csv").write_text("value\n1\n")
    metadata = tmp_path / "._result.csv"
    metadata.write_bytes(bytes.fromhex("0005160700020000") + b"metadata")
    assert logical_files(tmp_path) == ({"result.csv"}, 1)
    assert metadata.exists()


@pytest.mark.parametrize(
    "counterpart,header", [(False, "0005160700020000"), (True, "0000000000000000")]
)
def test_arbitrary_or_orphan_sidecars_still_fail(tmp_path, counterpart, header):
    if counterpart:
        (tmp_path / "result.csv").write_text("value\n1\n")
    (tmp_path / "._result.csv").write_bytes(bytes.fromhex(header))
    with pytest.raises(ValueError):
        logical_files(tmp_path)


def test_literal_null_is_a_category_not_missing_data():
    table = pd.DataFrame(
        {"part": ["native", "visible", "null"], "value": [1.0, 0.2, 0.8]}
    )
    loaded = pd.read_csv(io.StringIO(table.to_csv(index=False)), keep_default_na=False)
    pd.testing.assert_frame_equal(table, loaded)
