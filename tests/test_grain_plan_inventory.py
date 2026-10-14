#!/usr/bin/env python3
"""Tests for conservative historical-plan indexing."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "plan_inventory",
    Path(__file__).resolve().parents[1] / "scripts/inventory_grain_research_plans.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_no_executable_or_appledouble_entries(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/PLAN.md").write_text(
        "# Plan\n## Task\n```bash\npending dangerous command\n```\n- [x] Done claimed\n"
    )
    (tmp_path / "docs/._PLAN.md").write_bytes(b"\x00\xff")
    documents, rows = MODULE.inventory({"test": tmp_path})
    assert len(documents) == 1
    assert len(rows) == 2
    assert all(r["review_state"] == "UNREVIEWED" for r in rows)
    assert not any("dangerous" in r["text"] for r in rows)


def test_changed_copy_not_deduplicated(tmp_path):
    (tmp_path / "A_PLAN.md").write_text("# Plan\n## Task A\n")
    (tmp_path / "B_PLAN.md").write_text("# Plan\n## Task B\n")
    documents, _ = MODULE.inventory({"test": tmp_path})
    assert len({d["sha256"] for d in documents}) == 2


def test_report_link_not_completion(tmp_path):
    (tmp_path / "report.md").write_text("# Output\n")
    plan = tmp_path / "PLAN.md"
    plan.write_text("# Plan\n[report](report.md)\n[next](missing.md)\n")
    documents, _ = MODULE.inventory({"test": tmp_path})
    assert documents[0]["state"] == "UNREVIEWED"
    assert [link["exists"] for link in documents[0]["local_links"]] == [True, False]
