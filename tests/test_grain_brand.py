"""Tests for the GRAIN public name and compatibility boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_grain_namespace_loads_existing_core_modules() -> None:
    import grain_core
    import grain_core.v7_release
    import grain_sim

    assert grain_core.__version__ == "1.1.1"
    assert callable(grain_core.v7_release.sha256_file)
    assert callable(grain_sim.conditional_innovation)


def test_active_documentation_uses_grain_brand() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_grain_naming.py"),
            "--strict",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_primary_surfaces_name_grain() -> None:
    assert ROOT.joinpath("README.md").read_text(encoding="utf-8").startswith(
        "# GRAIN"
    )
    citation = ROOT.joinpath("CITATION.cff").read_text(encoding="utf-8")
    assert 'title: "GRAIN:' in citation


def test_canonical_definition_and_language_texture_are_pinned() -> None:
    policy = ROOT.joinpath("docs/GRAIN_NAMING_MIGRATION.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Group-Referenced Analysis of Individual Natural-Language Expression"
        in policy
    )
    assert "language texture" in policy.casefold()


def test_historical_release_identity_is_not_rewritten() -> None:
    expected_headings = {
        "docs/PREREGISTRATION.md": "# SUICA Lockbox Opening #1",
        "docs/RELEASE_NOTES_V020.md": "# SUICA v0.2.0 Release Notes",
        "docs/RELEASE_NOTES_V021.md": "# SUICA v0.2.1 Release Notes",
        "docs/SUICA_COMPLETION_DECLARATION.md": "# SUICA 完成宣言",
    }
    for relative, heading in expected_headings.items():
        assert ROOT.joinpath(relative).read_text(encoding="utf-8").startswith(
            heading
        )
