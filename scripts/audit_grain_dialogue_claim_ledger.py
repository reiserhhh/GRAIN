#!/usr/bin/env python3
"""Replay r1 artifacts and add explicitly post-result speech-target accounting."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.dialogue_claim_ledger import read_dialogue
from grain_research.dialogue_speech_scope import scope_speech, scoped_prefixes
from scripts.run_grain_dialogue_claim_ledger import BASE, CALIBRATION, sha, write


def addressed_objects(obj):
    """Walk all nested exact-text addresses without reading any private corpus."""
    if isinstance(obj, dict):
        if {"turn_id", "start", "end", "quote", "sha256"} <= obj.keys():
            yield obj
        for value in obj.values():
            yield from addressed_objects(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from addressed_objects(value)


def independent_prefixes(turns, history):
    """Use direct dictionary replacement and arithmetic, not M02 score helpers."""
    by_turn = {r["record_id"]: r for r in history}
    active, out = {}, []
    for turn in turns:
        record = by_turn.get(turn["id"])
        if record:
            kind = record["kind"]
            if kind in {"retract", "revise"}:
                del active[record["parent_id"]]
            if kind in {"new", "repeat", "revise"}:
                active[record["record_id"]] = record
        scores = {}
        for mode in ["past_action_report", "intention"]:
            for condition in [-1, 1]:
                values = [
                    r["value"]
                    for r in active.values()
                    if r["mode"] == mode and r["condition"] == condition
                ]
                scores[f"{mode}:{condition}"] = (
                    sum(values) / len(values) if values else None
                )
        out.append({"active_ids": list(active), "scores": scores})
    return out


def audit(source):
    """Verify frozen sources, addresses and arithmetic before recording the addendum."""
    out = source / "scope_followup"
    if out.exists():
        raise FileExistsError(out)
    freeze = json.loads((source / "freeze.json").read_text())
    for name, expected in freeze["files"].items():
        assert sha(ROOT / name) == expected, name
    manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in manifest.items():
        assert sha(source / name) == expected, name
    out.mkdir()
    files = [
        Path(__file__).resolve(),
        ROOT / "grain_research/dialogue_speech_scope.py",
        ROOT / "tests/test_grain_dialogue_speech_scope.py",
        source / "manifest.json",
    ]
    write(
        out / "freeze.json",
        {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "post_result": True,
            "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
        },
    )
    bank = json.loads((source / "public_bank.json").read_text())
    parses = json.loads((source / "parses.json").read_text())
    ledgers = json.loads((source / "dialogue_ledgers.json").read_text())
    calibration = json.loads(CALIBRATION.read_text())
    numeric, scoped, events, addresses = [], [], [], set()
    for case in bank:
        result = ledgers[case["id"]]
        assert read_dialogue(case["turns"], parses, calibration) == result
        turns = {t["id"]: t for t in case["turns"]}
        for a in addressed_objects(result):
            t = turns[a["turn_id"]]
            assert 0 <= a["start"] <= a["end"] <= len(t["text"])
            assert t["text"][a["start"] : a["end"]] == a["quote"]
            import hashlib

            assert hashlib.sha256(t["text"].encode()).hexdigest() == a["sha256"]
            assert t["speaker"] == a["speaker"]
            addresses.add((a["turn_id"], a["start"], a["end"]))
        for row, snapshot in zip(
            independent_prefixes(case["turns"], result["history"]),
            result["snapshots"],
            strict=True,
        ):
            assert row["active_ids"] == snapshot["active_ids"]
            for key, expected in row["scores"].items():
                actual = snapshot["scores"][key]["score"]
                assert actual == expected
                numeric.append(
                    {
                        "case": case["id"],
                        "turn": snapshot["turn"],
                        "channel": key,
                        "score": actual,
                        "replay": expected,
                    }
                )
        target_rows = scoped_prefixes(case["turns"], result["speech"])
        for i, row in enumerate(target_rows):
            expected = case["expected_speech"][i] if case["part"] == "main" else None
            scoped.append(
                {
                    "case": case["id"],
                    "part": case["part"],
                    **row,
                    "authored_help_request_count": expected,
                    "matches_authored_help": row["task_assistance_candidates"]
                    == expected
                    if expected is not None
                    else None,
                }
            )
        events.extend({"case": case["id"], **e} for e in scope_speech(result["speech"]))
    baseline = pd.read_csv(source / "ablations.csv")
    old_summary = json.loads((source / "summary.json").read_text())
    generic_speech = next(
        r
        for r in old_summary["checks"]
        if r["part"] == "main" and r["check"] == "speech"
    )
    for arm, values in old_summary["prefix_errors"].items():
        selected = baseline.dropna(subset=[arm, "authored_target"])
        error = float(np.mean(np.abs(selected[arm] - selected.authored_target)))
        assert len(selected) == values["scored_prefixes"]
        assert abs(error - values["mean_absolute_error"]) < 1e-12
    write(out / "speech_events.json", events)
    pd.DataFrame(numeric).to_csv(out / "independent_numeric_replay.csv", index=False)
    pd.DataFrame(scoped).to_csv(out / "scoped_speech_prefixes.csv", index=False)
    main = [r for r in scoped if r["part"] == "main"]
    summary = {
        "source_frozen_files_verified": len(freeze["files"]),
        "source_artifacts_verified": len(manifest),
        "dialogue_replays": len(bank),
        "unique_text_addresses": len(addresses),
        "independent_score_checks": len(numeric),
        "baseline_error_checks": 3,
        "speech_events_retained": len(events),
        "task_assistance_candidates": sum(
            e["task_assistance_candidate"] for e in events
        ),
        "discourse_permission_candidates": sum(
            e["discourse_permission_candidate"] for e in events
        ),
        "main_help_prefix_checks": len(main),
        "main_help_prefix_passed": sum(r["matches_authored_help"] for r in main),
        "original_main_generic_speech_passed": generic_speech["passed"],
        "original_main_generic_speech_total": generic_speech["n"],
        "new_neural_calls": 0,
        "status": "post-result target-scoping account; original mismatched-scope failures retained",
        "independence": "same agent equation replay, not D02 independent construction",
    }
    write(out / "summary.json", summary)
    write(out / "manifest.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=BASE / "dialogue_claim_ledger_r1"
    )
    audit(parser.parse_args().source.resolve())
