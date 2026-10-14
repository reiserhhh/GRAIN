#!/usr/bin/env python3
"""Freeze and run bounded offline dialogue-version and speech ownership tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.dialogue_claim_ledger import answer_operation, read_dialogue
from grain_research.source_event_graph import stanza_words
from scripts.fetch_grain_syntax_reference import MODELS, PACKAGES, PROCESSORS
from scripts.run_grain_source_event_graph import verify_models

BASE = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1"
CALIBRATION = BASE / "proposition_response_r1/completion/source_calibration.json"
CONFIG = ROOT / "configs/grain_dialogue_claim_ledger.json"


def write(path, data):
    """Persist finite JSON without implicit conversion of missing to NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def sha(path):
    """Hash a local source or artifact without loading model tensors."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def public_bank():
    """Create public authored dialogues and evaluation-only operation labels."""
    bank = []
    for condition in [-1, 1]:
        for mode in ["past_action_report", "intention"]:
            for wording in [0, 1]:
                prefix = {
                    -1: ["When the deadline approaches", "When you face a deadline"],
                    1: ["When instructions are missing", "When you lack instructions"],
                }[condition][wording]
                predicate = ["ask someone for help", "request assistance"][wording]
                if mode == "past_action_report":
                    questions = [
                        f"{prefix}, did you {predicate} {t}?"
                        for t in ["yesterday", "today"]
                    ]
                    positive, negative = "Yes, I did.", "No, I did not."
                else:
                    verb = ["intend", "plan"][wording]
                    questions = [
                        f"{prefix}, do you {verb} to {predicate} {t}?"
                        for t in ["today", "tomorrow"]
                    ]
                    positive, negative = "Yes, I do.", "No, I do not."
                repeat = ["To repeat", "I am repeating"][wording]
                repair = ["I need to correct", "Let me correct"][wording]
                texts = [
                    questions[0],
                    positive,
                    f"{repeat} my previous answer: {positive}",
                    f"{repair} my previous answer: {negative}",
                    questions[1],
                    positive,
                    "I withdraw my previous answer.",
                    "Please help me with this task.",
                    'My colleague said, "Please help me with this task."',
                ]
                roles = [
                    "partner",
                    "respondent",
                    "respondent",
                    "respondent",
                    "partner",
                    "respondent",
                    "respondent",
                    "respondent",
                    "respondent",
                ]
                case_id = f"d{len(bank):02d}"
                turns = [
                    {"id": f"{case_id}:t{i}", "speaker": role, "text": text}
                    for i, (text, role) in enumerate(zip(texts, roles, strict=True))
                ]
                bank.append(
                    {
                        "id": case_id,
                        "part": "main",
                        "condition": condition,
                        "mode": mode,
                        "wording": wording,
                        "turns": turns,
                        "expected_operations": [
                            "question_open",
                            "new",
                            "alias",
                            "revise",
                            "question_open",
                            "new",
                            "retract",
                            "no_claim_update",
                            "no_claim_update",
                        ],
                        "expected_scores": [None, 1, 1, -1, -1, 0, -1, -1, -1],
                        "expected_speech": [0, 0, 0, 0, 0, 0, 0, 1, 1],
                    }
                )
    challenges = [
        (
            "negative_question",
            [
                (
                    "partner",
                    "When the deadline approaches, didn't you ask someone for help yesterday?",
                ),
                ("respondent", "Yes."),
            ],
            ["ambiguous_scope", "no_unique_question"],
            0,
        ),
        (
            "no_antecedent",
            [("respondent", "I need to correct my previous answer: No, I did not.")],
            ["no_active_reference"],
            0,
        ),
        (
            "repeat_conflict",
            [
                (
                    "partner",
                    "When the deadline approaches, did you ask someone for help yesterday?",
                ),
                ("respondent", "Yes, I did."),
                ("respondent", "To repeat my previous answer: No, I did not."),
            ],
            ["question_open", "new", "repeat_content_conflict"],
            0,
        ),
        (
            "direct_request",
            [("respondent", "Please help me with this task.")],
            ["no_claim_update"],
            1,
        ),
        (
            "quoted_request",
            [("respondent", 'My colleague said, "Please help me with this task."')],
            ["no_claim_update"],
            0,
        ),
        (
            "complex_ellipsis",
            [("respondent", "I intended to ask for help, but I never did.")],
            ["no_claim_update"],
            0,
        ),
    ]
    for i, (tag, pairs, expected, speech) in enumerate(challenges):
        bank.append(
            {
                "id": f"x{i}",
                "part": "challenge",
                "challenge": tag,
                "turns": [
                    {"id": f"x{i}:t{j}", "speaker": role, "text": text}
                    for j, (role, text) in enumerate(pairs)
                ],
                "expected_operations": expected,
                "expected_final_speech": speech,
            }
        )
    return bank


def ablations(turns, result, score_key):
    """Compare versioned claims with unversioned polarity and last-answer baselines.

    Same question/condition is controlled in the main bank. Baselines deliberately
    do not update old claims on correction or withdrawal; these are not trained
    competitive language models. Auxiliary polarities come from raw text, not labels.
    """
    values, last, rows = [], None, []
    for turn, snapshot in zip(turns, result["snapshots"], strict=True):
        if turn["speaker"] == "respondent":
            _, polarity, _ = answer_operation(turn["text"])
            if polarity is not None:
                values.append(polarity)
                last = polarity
        rows.append(
            {
                "turn": turn["id"],
                "all_polarities": float(np.mean(values)) if values else None,
                "last_polarity": last,
                "active_ledger": snapshot["scores"][score_key]["score"],
                "active_claims": snapshot["scores"][score_key]["active_claims"],
                "active_event_references": snapshot["scores"][score_key][
                    "active_event_references"
                ],
            }
        )
    return rows


def run(out):
    """Run once, preserving frozen inputs and terminal failures without overwrite."""
    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}")
    out.mkdir(parents=True)
    config = json.loads(CONFIG.read_text())
    bank = public_bank()
    assert Counter(r["part"] for r in bank) == {
        "main": config["main_dialogues"],
        "challenge": config["challenge_dialogues"],
    }
    texts = sorted({t["text"] for row in bank for t in row["turns"] if t["text"]})
    if len(texts) > config["max_unique_parse_texts"]:
        raise ValueError("Public material exceeds frozen parse-call budget")
    calibration = json.loads(CALIBRATION.read_text())
    verify_models()
    files = [
        CONFIG,
        CALIBRATION,
        Path(__file__).resolve(),
        ROOT / "grain_research/dialogue_claim_ledger.py",
        ROOT / "grain_research/proposition_response.py",
        ROOT / "grain_research/source_event_graph.py",
        ROOT / "grain_research/evidence_history.py",
        ROOT / "scripts/fetch_grain_syntax_reference.py",
        ROOT / "scripts/run_grain_source_event_graph.py",
        ROOT / "docs/GRAIN_DIALOGUE_CLAIM_LEDGER_PLAN_V1.md",
        ROOT / "tests/test_grain_dialogue_claim_ledger.py",
        MODELS / "grain_inventory.json",
    ]
    write(out / "config.json", config)
    write(out / "public_bank.json", bank)
    files.extend([out / "config.json", out / "public_bank.json"])
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in files}
    write(
        out / "freeze.json",
        {"time_utc": datetime.now(timezone.utc).isoformat(), "files": frozen},
    )
    start = time.monotonic()
    try:
        sys.path.insert(0, str(PACKAGES))
        import stanza
        import torch

        torch.set_num_threads(2)
        pipeline = stanza.Pipeline(
            "en",
            dir=str(MODELS),
            package=None,
            processors=PROCESSORS,
            use_gpu=False,
            download_method=None,
            verbose=False,
        )
        parses = {}
        for text in texts:
            if time.monotonic() - start > config["max_parse_seconds"]:
                raise TimeoutError(
                    "Frozen parse budget exhausted; saved cache retained"
                )
            parses[text] = stanza_words(pipeline(text), text)
            write(out / "parses.json", parses)
        parse_seconds = time.monotonic() - start
        write(
            out / "parse_runtime.json",
            {
                "unique_calls": len(parses),
                "seconds_including_load": parse_seconds,
                "nli_calls": 0,
                "training": False,
            },
        )
        results, checks, baselines = {}, [], []
        for row in bank:
            result = read_dialogue(row["turns"], parses, calibration)
            results[row["id"]] = result
            for i, (actual, expected) in enumerate(
                zip(result["operations"], row["expected_operations"], strict=True)
            ):
                checks.append(
                    {
                        "case": row["id"],
                        "part": row["part"],
                        "turn": actual["turn"],
                        "check": "operation",
                        "expected": expected,
                        "actual": actual["status"],
                        "pass": actual["status"] == expected,
                    }
                )
                if row["part"] == "main":
                    key = f"{row['mode']}:{row['condition']}"
                    score = result["snapshots"][i]["scores"][key]["score"]
                    expected_score = row["expected_scores"][i]
                    checks.append(
                        {
                            "case": row["id"],
                            "part": "main",
                            "turn": actual["turn"],
                            "check": "score",
                            "expected": expected_score,
                            "actual": score,
                            "pass": score == expected_score,
                        }
                    )
                    speech = result["snapshots"][i]["own_direct_requests"]
                    checks.append(
                        {
                            "case": row["id"],
                            "part": "main",
                            "turn": actual["turn"],
                            "check": "speech",
                            "expected": row["expected_speech"][i],
                            "actual": speech,
                            "pass": speech == row["expected_speech"][i],
                        }
                    )
            if row["part"] == "challenge":
                speech = result["snapshots"][-1]["own_direct_requests"]
                checks.append(
                    {
                        "case": row["id"],
                        "part": "challenge",
                        "turn": row["turns"][-1]["id"],
                        "check": "speech",
                        "expected": row["expected_final_speech"],
                        "actual": speech,
                        "pass": speech == row["expected_final_speech"],
                    }
                )
            else:
                baseline = ablations(
                    row["turns"], result, f"{row['mode']}:{row['condition']}"
                )
                for i, b in enumerate(baseline):
                    baselines.append(
                        {
                            "case": row["id"],
                            "mode": row["mode"],
                            "condition": row["condition"],
                            "wording": row["wording"],
                            "prefix": i + 1,
                            "authored_target": row["expected_scores"][i],
                            **b,
                        }
                    )
        write(out / "dialogue_ledgers.json", results)
        table = pd.DataFrame(checks)
        table.to_csv(out / "checks.csv", index=False)
        pd.DataFrame(baselines).to_csv(out / "ablations.csv", index=False)
        errors = {}
        for arm in ["all_polarities", "last_polarity", "active_ledger"]:
            paired = [
                (b[arm], b["authored_target"])
                for b in baselines
                if b[arm] is not None and b["authored_target"] is not None
            ]
            errors[arm] = {
                "scored_prefixes": len(paired),
                "mean_absolute_error": float(np.mean([abs(a - b) for a, b in paired]))
                if paired
                else None,
            }
        summary = {
            "dialogues": len(bank),
            "main_turns": sum(len(r["turns"]) for r in bank if r["part"] == "main"),
            "all_turns": sum(len(r["turns"]) for r in bank),
            "unique_parse_texts": len(parses),
            "parse_seconds_including_load": parse_seconds,
            "checks": [
                {
                    "part": part,
                    "check": check,
                    "n": len(g),
                    "passed": int(g["pass"].sum()),
                }
                for (part, check), g in table.groupby(["part", "check"])
            ],
            "failures": table[~table["pass"]].to_dict("records"),
            "prefix_errors": errors,
            "scope": "authored raw-dialogue version mechanism; no learned open-language grammar or psychological norm",
        }
        write(out / "summary.json", summary)
        for path, expected_hash in frozen.items():
            if sha(ROOT / path) != expected_hash:
                raise ValueError("Frozen source changed during execution: " + path)
        write(
            out / "manifest.json",
            {str(p.relative_to(out)): sha(p) for p in out.iterdir() if p.is_file()},
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        write(
            out / "terminal_error.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "dialogue_claim_ledger_r1"
    )
    run(parser.parse_args().output.resolve())
