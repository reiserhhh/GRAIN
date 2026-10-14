#!/usr/bin/env python3
"""Explicit dictionary-to-tuple read adapter for the sealed HM112 analyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_recursive_collector_hm112 as RUN


def current_arrays(raw):
    """Adapt the existing typed raw-unit mapping; never alter primitive values."""
    if not isinstance(raw, dict) or set(raw) != {"k", "v"}:
        raise ValueError("Existing verified raw-unit dictionary required")
    return raw["k"], raw["v"]


def main():
    """Repair packaging only, retaining the original sealed construction protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    RUN.check(out)
    RUN.H.write(
        out / "04/read_adapter.json",
        {
            "reason": "Sealed analyzer destructures raw_unit's dictionary as a tuple; existing interface returns named k/v arrays",
            "discovered": "static interface review before analysis; no failed or repeated neural capture",
            "sealed_original_code_changed": False,
            "numeric_rule_or_selection_changed": False,
            "adapter": str(Path(__file__).relative_to(ROOT)),
            "adapter_sha256": RUN.H.sha(Path(__file__)),
        },
    )
    original = RUN.Q.raw_unit

    def adapted(out, row):
        """Keep all original metadata/hash/shape checks, then return the same arrays."""
        return current_arrays(original(out, row))

    try:
        RUN.Q.raw_unit = adapted
        RUN.analyze(out)
    finally:
        RUN.Q.raw_unit = original


if __name__ == "__main__":
    main()
