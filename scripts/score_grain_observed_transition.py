#!/usr/bin/env python3
"""Read one observed continuation from a frozen bundle, without LLM or labels."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_observed_transition import query


def main():
    """Require only the compiled observation law and two current numeric records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    result = query(
        json.loads(args.bundle.read_text()), json.loads(args.record.read_text())
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
