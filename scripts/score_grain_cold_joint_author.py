#!/usr/bin/env python3
"""Authenticated source bundle plus event observations -> cold author profiles."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_cold_joint_author import cold_score


def main():
    """Use only numeric observations; no application raw text or source score tables."""
    parser = argparse.ArgumentParser()
    for name in ("bundle", "input", "output", "sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    allowed = {
        os.path.realpath(getattr(args, n))
        for n in ("bundle", "input", "output", "audit")
    }
    opened = []

    def guard(event, values):
        """Refuse application text, source rows and counterfactual tables at runtime."""
        if event != "open" or isinstance(values[0], int):
            return
        path = os.path.realpath(os.fsdecode(values[0]))
        opened.append(path)
        if path not in allowed and not path.startswith("/dev/"):
            raise PermissionError("cold scorer refused " + path)

    sys.addaudithook(guard)
    data = Path(args.bundle).read_bytes()
    if hashlib.sha256(data).hexdigest() != args.sha256:
        raise ValueError("bundle authentication failed")
    bundle = json.loads(data)
    records = [
        json.loads(line)
        for line in Path(args.input).read_text().splitlines()
        if line.strip()
    ]
    with Path(args.output).open("x") as stream:
        stream.writelines(
            json.dumps(cold_score(bundle, record), allow_nan=False) + "\n"
            for record in records
        )
    with Path(args.audit).open("x") as stream:
        json.dump(
            {
                "opened": sorted(set(opened)),
                "allowed": sorted(allowed),
                "source_or_raw_application_access": False,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
