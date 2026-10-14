#!/usr/bin/env python3
"""Build a private dialogue-input file from already licensed cached natural sentences.

Reads the natural partition of a frozen proposition-response run (no new corpus
access), keeps the recorded partner turn only when one exists, and writes the
two-party turn format consumed by score_grain_dialogue_ledger_input.py.
The output contains licensed raw text and must stay local (private_ prefix).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = (
    ROOT
    / "results/grain_vector_mechanisms/mechanism_todo_v1/proposition_response_r1/private_inputs.json"
)


def build(records, part="natural"):
    """One dialogue per cached record; missing antecedents stay missing."""
    dialogues = []
    for record in records:
        if record.get("part") != part:
            continue
        turns = []
        if record.get("partner"):
            turns.append(
                {"id": f"{record['id']}:t0", "speaker": "partner", "text": record["partner"]}
            )
        turns.append(
            {
                "id": f"{record['id']}:t{len(turns)}",
                "speaker": "respondent",
                "text": record["reply"],
            }
        )
        dialogues.append({"id": record["id"], "part": part, "turns": turns})
    if not dialogues:
        raise ValueError(f"No records with part={part}")
    return {"dialogues": dialogues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", default="natural")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.name.startswith("private_"):
        raise ValueError("Natural text output must carry the private_ prefix")
    payload = build(json.loads(args.source.read_text()), args.part)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print({"dialogues": len(payload["dialogues"]), "output": str(args.output)})


if __name__ == "__main__":
    main()
