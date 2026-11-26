#!/usr/bin/env python3
"""Run a versioned evidence panel without models, network access or raw text."""

import argparse
import json
from pathlib import Path

from grain_core.evidence_panel import VERSION, run_panel, run_queries
from grain_core.evidence_stream import read_panel
from jsonschema import Draft202012Validator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path, help="new output file; does not overwrite")
    p.add_argument(
        "--queries", type=Path, help="JSON query array; enables line-delimited input"
    )
    args = p.parse_args()
    schema_path = Path(__file__).parents[1] / "schemas/evidence_panel.v1.schema.json"
    validator = Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    if args.queries:
        queries = json.loads(args.queries.read_text(encoding="utf-8"))
        validator.validate({"schema_version": VERSION, "queries": queries})
        with args.input.open(encoding="utf-8") as source:
            panel = read_panel(source)
        result = {**run_queries(panel, queries), "diagnostics": panel.diagnostics()}
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        validator.validate(payload)
        result = run_panel(payload)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    with args.output.open("x", encoding="utf-8") as f:
        f.write(encoded + "\n")


if __name__ == "__main__":
    main()
