"""Run a recorded legacy command through a canonical GRAIN filename."""

from __future__ import annotations

import runpy
from pathlib import Path


def run_legacy(filename: str) -> None:
    """Execute a legacy script without changing its frozen implementation."""
    runpy.run_path(str(Path(__file__).with_name(filename)), run_name="__main__")

