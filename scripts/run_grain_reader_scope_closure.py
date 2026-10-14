#!/usr/bin/env python3
"""Bounded WP1 follow-ups: cached natural replay and authored scope contrasts.

Isolated-question reparses are diagnosis only: they are not automatically
accepted because removing context can change quote and reference ownership.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.dialogue_ledger_generic import read_dialogue_generic
from grain_research.proposition_response import frames
from scripts.score_grain_dialogue_ledger_input import parse_missing

CASES = [
    ("Did you travel?", "Yes, I traveled yesterday.", "retain"),
    ("Did you travel?", "Yes, but I did not enjoy it.", "retain"),
    ("Did you travel?", "No, I did not travel.", "retain"),
    ("Did you travel?", "Yes, I did not travel.", "abstain"),
    ("Did you travel?", "No, I traveled yesterday.", "abstain"),
    ("Did you travel?", "Yes, actually I stayed at home.", "abstain"),
    ("Did you travel?", "Yes, not really.", "abstain"),
    ("Did you travel?", "No, but I enjoyed staying home.", "retain"),
    ("Did you enjoy it?", "Yes, I enjoyed it but never traveled.", "retain"),
    ("Can you swim?", "Yes, I swim every week.", "retain"),
    ("Did your friend travel?", "Yes, she traveled yesterday.", "no_self_score"),
    ("Did you travel and did you swim?", "Yes, it was fun.", "abstain"),
]


def replay(turns, cache, guard):
    """Return an observable operation and whether an author score was emitted."""
    result = read_dialogue_generic(turns, cache, polar_binding=True, scope_guard=guard)
    return result["operations"][-1]["status"], bool(result["snapshots"][-1]["scores"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dialogues = json.loads(
        (args.input_dir / "private_natural_pairs_input.json").read_text()
    )["dialogues"]
    cache = json.loads((args.input_dir / "private_parses.json").read_text())
    no_frame, causes, segments = [], Counter(), {}
    for d in dialogues:
        text = d["turns"][0]["text"]
        ff = frames(text, cache[text], "partner")
        if any(f["question"] for f in ff):
            continue
        no_frame.append(d["id"])
        if not ff:
            causes["no_event_frames"] += 1
        else:
            causes["events_present_but_no_question_frame"] += 1
        if len(cache[text]) > 1:
            causes["multiple_sentences"] += 1
        if re.search(r"\?[\"'\u201d\u2019]", text):
            causes["closing_quote_after_question"] += 1
        slices = [
            m.group().strip()
            for m in re.finditer(r"[^.!?]*\?", text)
            if m.group().strip()
        ]
        segments[d["id"]] = slices
    requested = {t for q, a, _ in CASES for t in (q, a)} | {
        s for ss in segments.values() for s in ss
    }
    new, seconds = parse_missing(sorted(requested), cache, 600)
    cache.update(new)
    recovered = sum(
        any(any(f["question"] for f in frames(s, cache[s], "partner")) for s in ss)
        for ss in segments.values()
    )
    natural = {"unguarded": Counter(), "guarded": Counter()}
    scored = Counter()
    for d in dialogues:
        for name, guard in (("unguarded", False), ("guarded", True)):
            status, has_score = replay(d["turns"], cache, guard)
            natural[name][status] += 1
            scored[name] += has_score
    contrasts = []
    for i, (q, a, expected) in enumerate(CASES):
        turns = [
            {"id": f"c{i}:q", "speaker": "partner", "text": q},
            {"id": f"c{i}:a", "speaker": "respondent", "text": a},
        ]
        status, has_score = replay(turns, cache, True)
        contrasts.append(
            {
                "question": q,
                "answer": a,
                "expected": expected,
                "operation": status,
                "self_score": has_score,
                "pass": has_score == (expected == "retain"),
            }
        )
    summary = {
        "natural_pairs": len(dialogues),
        "no_frame_pairs": len(no_frame),
        "no_frame_causes_overlapping": dict(causes),
        "isolated_question_frame_recovered": recovered,
        "isolation_scope": "Diagnostic only; context/quotation removal not promoted to scoring.",
        "natural_operations": {k: dict(v) for k, v in natural.items()},
        "natural_self_scored": dict(scored),
        "authored_contrasts": contrasts,
        "contrast_pass": sum(c["pass"] for c in contrasts),
        "contrast_total": len(contrasts),
        "new_local_parses": len(new),
        "parse_seconds": seconds,
        "scope": "Coverage and rule contrasts, not human semantic accuracy or personality validity.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "private_new_parses.json").write_text(json.dumps(new))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
