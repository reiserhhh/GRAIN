#!/usr/bin/env python3
"""Apply the frozen dialogue claim ledger to NEW raw dialogue text.

Entry for M01/M09: raw two-party turns -> fixed local Stanza parse -> frozen
proposition/condition/version rules -> per-target computability account.
The output says which declared targets are computable from the input, which
turns lack an antecedent or licensed reference, and the finite material range
that unresolved polarity turns could still add. Nothing is fitted; no NLI,
training or population norm is used. Unresolved turns are reported, never
scored as neutral, negative or zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.dialogue_claim_ledger import answer_operation, read_dialogue
from grain_research.dialogue_speech_scope import scope_speech
from grain_research.proposition_response import frames
from grain_research.source_event_graph import stanza_words
from scripts.fetch_grain_syntax_reference import MODELS, PACKAGES, PROCESSORS

BASE = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1"
CALIBRATION = BASE / "proposition_response_r1/completion/source_calibration.json"
CONFIG = ROOT / "configs/grain_dialogue_claim_ledger.json"
TARGETS = [f"{m}:{c}" for m in ("past_action_report", "intention") for c in (-1, 1)]
ENTERED = {"new", "repeat", "alias", "revise", "retract"}
DEPENDENCIES = [
    CONFIG,
    CALIBRATION,
    ROOT / "grain_research/dialogue_claim_ledger.py",
    ROOT / "grain_research/dialogue_speech_scope.py",
    ROOT / "grain_research/proposition_response.py",
    ROOT / "grain_research/source_event_graph.py",
    ROOT / "grain_research/evidence_history.py",
]


def sha(path):
    """Hash a file without loading model tensors."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    """Persist finite JSON; missing values stay null, never NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def load_dialogues(path):
    """Accept the public-bank list format or {"dialogues": [...]}."""
    payload = json.loads(Path(path).read_text())
    rows = payload["dialogues"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must contain a nonempty list of dialogues")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Dialogue IDs must be unique")
    for row in rows:
        for turn in row["turns"]:
            if set(turn) < {"id", "speaker", "text"}:
                raise ValueError("Each turn needs id, speaker and text")
            if turn["speaker"] not in {"partner", "respondent"}:
                raise ValueError("Speakers must be 'partner' or 'respondent'")
    return rows


def load_parse_cache(paths):
    """Merge cached parses: {text: sentences} or [{text, sentences}, ...]."""
    cache = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        items = payload.items() if isinstance(payload, dict) else (
            (row["text"], row["sentences"]) for row in payload
        )
        for text, sentences in items:
            cache.setdefault(text, sentences)
    return cache


def parse_missing(texts, cache, budget_seconds):
    """Run the fixed local syntax model only for texts absent from the cache."""
    from scripts.run_grain_source_event_graph import verify_models

    missing = [t for t in texts if t not in cache]
    if not missing:
        return {}, 0.0
    verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    torch.set_num_threads(2)
    start = time.monotonic()
    pipeline = stanza.Pipeline(
        "en",
        dir=str(MODELS),
        package=None,
        processors=PROCESSORS,
        use_gpu=False,
        download_method=None,
        verbose=False,
    )
    parsed = {}
    for text in missing:
        if time.monotonic() - start > budget_seconds:
            raise TimeoutError("Parse budget exhausted; partial cache retained")
        parsed[text] = stanza_words(pipeline(text), text)
    return parsed, time.monotonic() - start


def bounded_mean_range(total, observed, unread):
    """Finite material mean range when `unread` values in [-1, 1] are missing.

    Inherits the bounded-missing sum identity used in the proposition-response
    audit: [(S-m)/(N+m), (S+m)/(N+m)]. Returns None when nothing is observed
    and nothing is unread; [-1, 1] when only unread material exists.
    """
    if observed < 0 or unread < 0:
        raise ValueError("Counts must be nonnegative")
    if observed == 0 and unread == 0:
        return None
    if observed == 0:
        return [-1.0, 1.0]
    return [(total - unread) / (observed + unread), (total + unread) / (observed + unread)]


def turn_frame_account(turn, parsed):
    """Text-free syntactic candidate counts for one turn; no labels enter."""
    found = frames(turn["text"], parsed, turn["speaker"])
    return {
        "frames": len(found),
        "channels": dict(Counter(f["channel"] for f in found)),
        "actors": dict(Counter(f["actor"] for f in found)),
        "question_frames": sum(f["question"] for f in found),
        "target_relevant_frames": sum(f["target_relevant"] for f in found),
        "condition_clause_frames": sum(bool(f["conditions"]) for f in found),
        "source_scoped_frames": sum(f["source_scoped"] for f in found),
        "ambiguous_negation_frames": sum(f["ambiguous_negation"] for f in found),
    }


def account_dialogue(row, parses, calibration):
    """Run the frozen ledger and derive target computability without new rules."""
    turns = row["turns"]
    result = read_dialogue(turns, parses, calibration)
    speech = scope_speech(result["speech"])
    operations = {op["turn"]: op["status"] for op in result["operations"]}
    final = result["snapshots"][-1]
    turn_rows, unassigned, same_target, reasons = [], 0, Counter(), Counter()
    current = None
    for turn in turns:
        status = operations[turn["id"]]
        kind, polarity, explicit_self = (None, None, None)
        if turn["speaker"] == "partner":
            if status == "question_open":
                question = [
                    r for r in result["history"]
                    if r.get("proposition", {}).get("question_id") == turn["id"]
                ]
                current = (
                    f"{question[0]['mode']}:{question[0]['condition']}"
                    if question
                    else "opened_unanswered"
                )
            else:
                current = None
                reasons[status] += 1
        else:
            kind, polarity, explicit_self = answer_operation(turn["text"])
            if status not in ENTERED:
                reasons[status] += 1
                if polarity is not None:
                    if current in TARGETS and status in {
                        "unresolved_actor",
                        "repeat_content_conflict",
                    }:
                        same_target[current] += 1
                    else:
                        unassigned += 1
        own = [s for s in speech if s["utterance_span"]["turn_id"] == turn["id"]]
        turn_rows.append(
            {
                "dialogue": row["id"],
                "turn": turn["id"],
                "speaker": turn["speaker"],
                "operation": status,
                "answer_kind": kind,
                "polarity": polarity,
                "explicit_self_auxiliary": explicit_self,
                "direct_request_candidates": len(own),
                "task_assistance_candidates": sum(
                    s["task_assistance_candidate"] for s in own
                ),
                "discourse_permission_candidates": sum(
                    s["discourse_permission_candidate"] for s in own
                ),
                "explicit_previous_answer_repair": sum(
                    s["explicit_previous_answer_repair"] for s in own
                ),
                **{
                    "frame_" + k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v)
                    for k, v in turn_frame_account(turn, parses[turn["text"]]).items()
                },
            }
        )
    targets = []
    for target in TARGETS:
        cell = final["scores"][target]
        active = [
            r["value"]
            for r in result["history"]
            if r["record_id"] in final["active_ids"]
            and f"{r['mode']}:{r['condition']}" == target
        ]
        total, observed = float(sum(active)), len(active)
        targets.append(
            {
                "dialogue": row["id"],
                "target": target,
                "computable": cell["score"] is not None,
                "score": cell["score"],
                "active_claims": cell["active_claims"],
                "active_event_references": cell["active_event_references"],
                "unresolved_same_target_polarity_turns": same_target[target],
                "unassigned_polarity_turns": unassigned,
                "known_condition_range": bounded_mean_range(
                    total, observed, same_target[target]
                ),
                "unknown_condition_range": bounded_mean_range(
                    total, observed, same_target[target] + unassigned
                ),
                "missing_antecedent_reasons": json.dumps(dict(reasons), sort_keys=True),
            }
        )
    return result, speech, turn_rows, targets


def replay_check(reference_dir, ledgers):
    """Compare operations/scores with a frozen run; report, never overwrite."""
    reference = json.loads(
        (Path(reference_dir) / "dialogue_ledgers.json").read_text()
    )
    compared, mismatches = 0, []
    for case, mine in ledgers.items():
        if case not in reference:
            continue
        theirs = reference[case]
        for a, b in zip(mine["operations"], theirs["operations"], strict=True):
            compared += 1
            if a != b:
                mismatches.append({"case": case, "turn": a["turn"], "kind": "operation"})
        for a, b in zip(mine["snapshots"], theirs["snapshots"], strict=True):
            for target in TARGETS:
                compared += 1
                if a["scores"][target] != b["scores"][target]:
                    mismatches.append({"case": case, "turn": a["turn"], "kind": target})
            compared += 1
            if a["own_direct_requests"] != b["own_direct_requests"]:
                mismatches.append({"case": case, "turn": a["turn"], "kind": "speech"})
    return {
        "reference": str(reference_dir),
        "dialogues_compared": sum(1 for c in ledgers if c in reference),
        "cells_compared": compared,
        "mismatches": mismatches,
        "exact": not mismatches,
    }


def run(args):
    """Score new dialogues into a fresh directory; keep text out of public tables."""
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}")
    config = json.loads(CONFIG.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    rows = load_dialogues(args.input)
    texts = sorted({t["text"] for row in rows for t in row["turns"] if t["text"]})
    if len(texts) > config["max_unique_parse_texts"]:
        raise ValueError("Input exceeds the frozen unique-parse budget")
    cache = load_parse_cache(args.parses)
    missing = [t for t in texts if t not in cache]
    if missing and not args.allow_parse:
        raise ValueError(
            f"{len(missing)} texts lack cached parses; pass --allow-parse to run "
            "the fixed local syntax model"
        )
    out.mkdir(parents=True)
    parsed, parse_seconds = parse_missing(texts, cache, config["max_parse_seconds"])
    parses = {t: (cache.get(t) if t in cache else parsed[t]) for t in texts}
    prefix = "private_" if args.private else ""
    ledgers, speech_all, turn_rows, target_rows = {}, {}, [], []
    for row in rows:
        result, speech, turns, targets = account_dialogue(row, parses, calibration)
        ledgers[row["id"]] = result
        speech_all[row["id"]] = speech
        turn_rows.extend(turns)
        target_rows.extend(targets)
    write(out / f"{prefix}ledgers.json", ledgers)
    write(out / f"{prefix}speech_events.json", speech_all)
    write(out / f"{prefix}parses.json", parses)
    turn_table = pd.DataFrame(turn_rows)
    target_table = pd.DataFrame(target_rows)
    turn_table.to_csv(out / "turn_accounts.csv", index=False)
    target_table.to_csv(out / "target_computability.csv", index=False)
    channel_hist = Counter()
    for value in turn_table["frame_channels"]:
        channel_hist.update(json.loads(value))
    actor_hist = Counter()
    for value in turn_table["frame_actors"]:
        actor_hist.update(json.loads(value))
    coverage = {
        "dialogues": len(rows),
        "turns": len(turn_table),
        "unique_texts": len(texts),
        "cached_parses_used": len(texts) - len(missing),
        "new_parse_calls": len(missing),
        "parse_seconds": parse_seconds,
        "nli_calls": 0,
        "training": False,
        "targets_total": len(target_table),
        "targets_computable": int(target_table["computable"].sum()),
        "dialogues_with_any_computable_target": int(
            target_table.groupby("dialogue")["computable"].any().sum()
        ),
        "operation_histogram": dict(Counter(turn_table["operation"])),
        "missing_antecedent_reasons": dict(
            sum(
                (Counter(json.loads(v)) for v in target_table.drop_duplicates("dialogue")["missing_antecedent_reasons"]),
                Counter(),
            )
        ),
        "frame_channel_histogram": dict(channel_hist),
        "frame_actor_histogram": dict(actor_hist),
        "target_relevant_frames": int(turn_table["frame_target_relevant_frames"].sum()),
        "condition_clause_frames": int(turn_table["frame_condition_clause_frames"].sum()),
        "direct_request_candidates": int(turn_table["direct_request_candidates"].sum()),
        "task_assistance_candidates": int(turn_table["task_assistance_candidates"].sum()),
        "discourse_permission_candidates": int(
            turn_table["discourse_permission_candidates"].sum()
        ),
        "unassigned_polarity_turns": int(
            target_table.drop_duplicates("dialogue")["unassigned_polarity_turns"].sum()
        ),
        "scope": "frozen explicit-reference rules on declared two-party roles; counts are syntactic/rule candidates, not psychological scores",
    }
    write(out / "coverage.json", coverage)
    if args.replay_check:
        write(out / "replay_check.json", replay_check(args.replay_check, ledgers))
    manifest = {
        "input_sha256": sha(Path(args.input)),
        "dependencies": {str(p.relative_to(ROOT)): sha(p) for p in DEPENDENCIES},
        "parse_caches": [str(Path(p)) for p in args.parses],
        "syntax_model_inventory": sha(MODELS / "grain_inventory.json"),
        "private_outputs": args.private,
        "outputs": {
            p.name: sha(p) for p in out.iterdir() if p.is_file() and not p.name.startswith("._")
        },
    }
    write(out / "manifest.json", manifest)
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parses", type=Path, nargs="*", default=[])
    parser.add_argument("--allow-parse", action="store_true")
    parser.add_argument("--private", action="store_true",
                        help="prefix text-bearing outputs with private_")
    parser.add_argument("--replay-check", type=Path, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
