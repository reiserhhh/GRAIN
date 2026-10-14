#!/usr/bin/env python3
"""WP1: coverage funnel of the dialogue ledgers on natural antecedent pairs.

Samples licensed parent->reply pairs from the local antecedent scan (other-author
parent containing a question mark; bounded lengths; English), parses them with
the fixed local syntax model, and runs (i) the frozen help-request ledger and
(ii) the target-generic ledger. Outputs count-only funnels (public) and private
ledgers (local). No user identifiers leave the private files; no NLI, training
or label use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.dialogue_claim_ledger import read_dialogue
from grain_research.dialogue_ledger_generic import classify_reply, read_dialogue_generic
from scripts.score_grain_dialogue_ledger_input import CALIBRATION, parse_missing, sha

POLAR = r"^\s*(yes|no|yeah|yep|nope|nah)\b"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")


def sample_pairs(pairs, seed, n_random, n_polar, max_words):
    base = pairs[
        pairs.other_author
        & (pairs.lang == "en")
        & pairs.parent_has_question_mark
        & (pairs.parent_word_count <= max_words)
        & (pairs.word_count <= max_words)
        & pairs.body.astype(str).str.strip().ne("")
        & pairs.parent_body.astype(str).str.strip().ne("")
    ].copy()
    base["polar_reply"] = base.body.astype(str).str.match(POLAR, case=False)
    rng = np.random.default_rng(seed)
    polar = base[base.polar_reply]
    rest = base[~base.polar_reply]
    take_polar = polar.iloc[rng.permutation(len(polar))[:n_polar]].assign(stratum="polar_reply")
    take_rest = rest.iloc[rng.permutation(len(rest))[:n_random]].assign(stratum="random_question")
    out = pd.concat([take_polar, take_rest], ignore_index=True)
    return out, {"eligible_pairs": len(base), "eligible_polar": len(polar), "eligible_rest": len(rest)}


def dialogues_from(sample):
    rows = []
    for i, r in enumerate(sample.itertuples()):
        did = f"n{i:03d}"
        rows.append({
            "id": did,
            "stratum": r.stratum,
            "turns": [
                {"id": f"{did}:t0", "speaker": "partner", "text": str(r.parent_body).strip()},
                {"id": f"{did}:t1", "speaker": "respondent", "text": str(r.body).strip()},
            ],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--n-random", type=int, default=48)
    parser.add_argument("--n-polar", type=int, default=48)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--parse-budget-seconds", type=int, default=900)
    parser.add_argument("--reuse-input", type=Path, default=None, help="private_natural_pairs_input.json from a previous run")
    parser.add_argument("--reuse-parses", type=Path, default=None)
    parser.add_argument("--polar-binding", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    pairs = pd.read_csv(args.pairs, dtype={"body": str, "parent_body": str, "lang": str})
    pairs["parent_word_count"] = pd.to_numeric(pairs["parent_word_count"], errors="coerce")
    pairs["word_count"] = pd.to_numeric(pairs["word_count"], errors="coerce")
    if args.reuse_input is not None:
        dialogues = json.loads(args.reuse_input.read_text())["dialogues"]
        sample = pd.DataFrame({"stratum": [d["stratum"] for d in dialogues]})
        pool = {"reused_input": str(args.reuse_input)}
    else:
        sample, pool = sample_pairs(pairs, args.seed, args.n_random, args.n_polar, args.max_words)
        dialogues = dialogues_from(sample)
    write(out / "private_natural_pairs_input.json", {"dialogues": dialogues})
    texts = sorted({t["text"] for d in dialogues for t in d["turns"]})
    start = time.monotonic()
    cache = json.loads(args.reuse_parses.read_text()) if args.reuse_parses is not None else {}
    parsed_new, parse_seconds = parse_missing(texts, cache, args.parse_budget_seconds)
    parsed = {t: cache[t] if t in cache else parsed_new[t] for t in texts}
    write(out / "private_parses.json", parsed)
    calibration = json.loads(CALIBRATION.read_text())
    frozen_ops, generic_ops, question_accounts, reply_classes = Counter(), Counter(), [], Counter()
    frozen_ledgers, generic_ledgers, rows = {}, {}, []
    for d in dialogues:
        f = read_dialogue(d["turns"], parsed, calibration)
        g = read_dialogue_generic(d["turns"], parsed, polar_binding=args.polar_binding)
        frozen_ledgers[d["id"]], generic_ledgers[d["id"]] = f, g
        f_status = [op["status"] for op in f["operations"]]
        g_status = [op["status"] for op in g["operations"]]
        frozen_ops.update(f_status)
        generic_ops.update(g_status)
        acc = g["question_accounts"][0]
        question_accounts.append({"dialogue": d["id"], "stratum": d["stratum"], **acc})
        kind, _, _ = classify_reply(d["turns"][1]["text"])
        reply_classes[kind] += 1
        rows.append({
            "dialogue": d["id"], "stratum": d["stratum"],
            "frozen_question": f_status[0], "frozen_reply": f_status[1],
            "generic_question": g_status[0], "generic_reply": g_status[1],
            "reply_class": kind,
            "question_frames": acc["question_frames"], "eligible_frames": acc["eligible_frames"],
            "condition_clause_frames": acc["condition_clause_frames"],
            "target_relevant_frames": acc["target_relevant_frames"],
            "generic_scored": bool(g["snapshots"][-1]["scores"]),
            "frozen_scored": any(v["score"] is not None for v in f["snapshots"][-1]["scores"].values()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "pair_accounts.csv", index=False)
    write(out / "private_frozen_ledgers.json", frozen_ledgers)
    write(out / "private_generic_ledgers.json", generic_ledgers)

    def funnel(sub):
        n = len(sub)
        return {
            "pairs": int(n),
            "parent_has_question_frame": int((sub.question_frames > 0).sum()),
            "parent_has_unique_bindable_question": int((sub.generic_question == "question_open").sum()),
            "parent_question_reasons": dict(Counter(sub.generic_question)),
            "parent_condition_clause_present": int((sub.condition_clause_frames > 0).sum()),
            "parent_target_relevant_help_request": int((sub.target_relevant_frames > 0).sum()),
            "frozen_question_open": int((sub.frozen_question == "question_open").sum()),
            "frozen_question_reasons": dict(Counter(sub.frozen_question)),
            "reply_classes": dict(Counter(sub.reply_class)),
            "generic_reply_status": dict(Counter(sub.generic_reply)),
            "frozen_reply_status": dict(Counter(sub.frozen_reply)),
            "generic_scored_pairs": int(sub.generic_scored.sum()),
            "frozen_scored_pairs": int(sub.frozen_scored.sum()),
        }

    coverage = {
        "sampling": {**pool, "seed": args.seed, "max_words": args.max_words, "sampled": len(sample), "by_stratum": dict(Counter(sample.stratum))},
        "unique_texts_parsed": len(texts),
        "parse_seconds_including_load": parse_seconds,
        "nli_calls": 0,
        "labels_used": False,
        "polar_binding": args.polar_binding,
        "new_parse_calls": len(parsed_new),
        "overall": funnel(table),
        "by_stratum": {s: funnel(table[table.stratum == s]) for s in sorted(table.stratum.unique())},
        "question_channel_histogram": dict(sum((Counter(a["channels"]) for a in question_accounts), Counter())),
        "question_actor_histogram": dict(sum((Counter(a["actors"]) for a in question_accounts), Counter())),
        "generic_score_keys_channel": dict(Counter(k.split(":")[0] for g in generic_ledgers.values() for k in g["snapshots"][-1]["scores"])),
        "wall_seconds": time.monotonic() - start,
        "scope": "count-only coverage funnel on licensed natural pairs; candidate frames and rule statuses are not human labels; generic ledger is a diagnostic instrument",
    }
    write(out / "coverage.json", coverage)
    write(out / "manifest.json", {
        "pairs_file_sha256": sha(Path(args.pairs)),
        "calibration_sha256": sha(CALIBRATION),
        "modules": {p: sha(ROOT / p) for p in ("grain_research/dialogue_claim_ledger.py", "grain_research/dialogue_ledger_generic.py", "grain_research/proposition_response.py", "grain_research/source_event_graph.py")},
        "outputs": {p.name: sha(p) for p in out.iterdir() if p.is_file() and not p.name.startswith("._")},
        "private_outputs": [p.name for p in out.iterdir() if p.name.startswith("private_")],
    })
    print(json.dumps({k: coverage[k] for k in ("sampling", "unique_texts_parsed", "parse_seconds_including_load", "overall", "by_stratum", "question_channel_histogram", "question_actor_histogram", "generic_score_keys_channel")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
