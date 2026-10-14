#!/usr/bin/env python3
"""Score local text JSONL with a frozen JSON reference; no model download or fitting."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.local_measure import explain_document, score_documents


def run(args):
    """Require explicit input/output paths and never overwrite another result."""
    if args.output.exists():
        raise ValueError("Use a new output file")
    bundle = json.loads(args.bundle.read_text())
    with args.input.open() as source, args.output.open("x") as target:
        for line in source:
            row = json.loads(line)
            s = score_documents(bundle, [row["text"]])
            empty = int(s["tokens"][0]) == 0
            result = {
                "id": row["id"],
                "domain": bundle["domain"],
                "method": bundle["method"],
                "factor_ids": bundle["factor_ids"],
                "status": "NO_ALPHABETIC_EVIDENCE" if empty else "SCORED",
                "score": None if empty else s["z"][0].tolist(),
                "percentile": None if empty else s["percentile"][0].tolist(),
                "tokens": int(s["tokens"][0]),
                "dictionary_hits": int(s["dictionary_hits"][0]),
            }
            if args.explain:
                result["explanation"] = explain_document(bundle, row["text"])
            target.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--explain", action="store_true")
    run(p.parse_args())
