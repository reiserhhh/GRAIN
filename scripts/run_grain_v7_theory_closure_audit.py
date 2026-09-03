#!/usr/bin/env python3
"""Run the legacy V7 theory-closure audit in its immutable release tree."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "v0.2.0"


def main() -> int:
    """Materialize the V7 release and forward arguments to its audit."""
    subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{TAG}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    with tempfile.TemporaryDirectory(prefix="grain-v7-closure-") as tmp:
        checkout = Path(tmp) / "release"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(checkout), TAG],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            command = [
                sys.executable,
                str(checkout / "scripts" / "run_suica_v7_theory_closure_audit.py"),
                *sys.argv[1:],
            ]
            return subprocess.run(command, cwd=checkout, check=False).returncode
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(checkout)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
