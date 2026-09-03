#!/usr/bin/env python3
"""Verify the legacy v0.2.1 lockbox in its immutable tagged tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "v0.2.1"


def main() -> int:
    """Materialize the sealed tag and run its original verifier there."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-only", action="store_true")
    args = parser.parse_args()

    subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{TAG}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    with tempfile.TemporaryDirectory(prefix="grain-v021-") as tmp:
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
                str(checkout / "scripts" / "verify_suica_v021_lockbox.py"),
            ]
            if args.content_only:
                command.append("--content-only")
            completed = subprocess.run(command, cwd=checkout, check=False)
            return completed.returncode
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
