#!/usr/bin/env python3
"""Companion entrypoint: preserve HM81's zero-call lock-signature failure."""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_grain_cold_joint_author_hm81 import (
    OUT,
    check,
    collect,
    digest,
    read_json,
    write,
)
from scripts.run_grain_upstream_routes import collector_guard


def main():
    """Supply the missing lock path without changing frozen materials or inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    check(args.out)
    companion = args.out / "02_entrypoint_repair"
    if not companion.exists():
        status = read_json(args.out / "run_status.json")
        if status["calls"] != 0 or status["attempted"] != 0:
            raise ValueError("freeze this entrypoint correction before any call")
        companion.mkdir()
        path = Path(__file__).resolve()
        frozen = companion / path.name
        shutil.copyfile(path, frozen)
        write(
            companion / "manifest.json",
            {
                "live_code": str(path),
                "code_sha256": digest(path),
                "frozen_code_sha256": digest(frozen),
                "primary_entrypoint_failure": "TypeError: collector_guard() missing 1 required positional argument: 'out'",
                "primary_failure_new_calls": 0,
                "change": "collector_guard(out), then unchanged frozen collect(out)",
                "primary_protocol_code_and_materials_changed": False,
            },
        )
    manifest = read_json(companion / "manifest.json")
    if digest(Path(__file__).resolve()) != manifest["code_sha256"]:
        raise ValueError("companion entrypoint drift")
    with collector_guard(args.out):
        collect(args.out)


if __name__ == "__main__":
    main()
