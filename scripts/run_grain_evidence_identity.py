#!/usr/bin/env python3
"""Run a frozen, local evidence-identity and dialogue-dependency experiment."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_identity import (
    MODES,
    QUERIES,
    evidence_record,
    opportunity_standardize,
    requests,
    simplex_decode,
    target_identifiability,
)
from scripts.prepare_grain_predecessor_reference import digest, write
from scripts.run_grain_entailment_reference import MODEL, REVISION
from scripts.run_grain_evidence_calibration import infer_pairs

BASE = ROOT / "results/grain_vector_mechanisms"
OUT = BASE / "mechanism_todo_v1/evidence_identity_r1"
GRAPH = BASE / "all_plan_closure_v1/source_event_r4"
FROZEN = (
    "grain_research/evidence_identity.py",
    "scripts/run_grain_evidence_identity.py",
    "configs/grain_evidence_identity.json",
    "docs/GRAIN_EVIDENCE_IDENTITY_PLAN_V1.md",
    "grain_research/source_event_graph.py",
    "scripts/run_grain_evidence_calibration.py",
    "scripts/run_grain_entailment_reference.py",
)


def public_bank():
    """Declare stimulus identities; only source identities enter calibration."""
    templates = {
        "intention": [
            "I intend to ask you for help tomorrow.",
            "I plan to request your assistance next week.",
            "My intention is to seek your help when the task starts.",
            "Tomorrow I am going to ask you to help me.",
        ],
        "action_report": [
            "I asked you for help yesterday.",
            "I requested your assistance last week.",
            "Yesterday I sent you a request for help with the task.",
            "I sought your help when the task started yesterday.",
        ],
        "preference": [
            "I prefer asking you for help to working alone.",
            "I like receiving assistance from you.",
            "My preference is to seek your help rather than work alone.",
            "Getting your help is something I enjoy.",
        ],
        "ability": [
            "I am able to ask you for help.",
            "I am capable of requesting your assistance.",
            "I have the ability to seek your help when necessary.",
            "Requesting help from you is something I know how to do.",
        ],
        "experience": [
            "I feel relieved by your help right now.",
            "At this moment I feel grateful for your assistance.",
            "Right now, receiving your help makes me feel reassured.",
            "Your assistance leaves me feeling calm at the moment.",
        ],
        "request": [
            "Please help me with this task.",
            "Could you assist me with this problem, please?",
            "Would you please give me a hand with this task?",
            "Can you help me work this out?",
        ],
    }
    rows = []
    for mode, texts in templates.items():
        for wording, reply in enumerate(texts):
            rows.append(
                {
                    "record_id": f"bank_{mode}_{wording}",
                    "part": "source" if wording < 2 else "held",
                    "expected": mode,
                    "partner": "",
                    "reply": reply,
                    "wording": wording,
                }
            )
    challenges = [
        ("I do not intend to ask you for help.", "intention_negation"),
        ("I did not ask you for help yesterday.", "action_negation"),
        ("I cannot ask you for help.", "ability_negation"),
        ("My colleague intends to ask you for help tomorrow.", "other_intention"),
        ("My colleague asked you for help yesterday.", "other_action"),
        ('My colleague said, "Please help me with this task."', "quoted_request"),
        (
            'The example sentence is "I asked you for help yesterday."',
            "metalinguistic_quote",
        ),
        ("I intended to ask you for help, but I never did.", "unrealized_intention"),
        ("I prefer your help, but I cannot ask for it.", "multiple_channels"),
        ("If I had asked you for help, I might have finished.", "counterfactual"),
        ("I hear that my colleague says I asked you for help.", "nested_source"),
        (
            "I asked you for help yesterday, and I am asking again now: please help.",
            "report_and_current_act",
        ),
    ]
    for i, (reply, expected) in enumerate(challenges):
        rows.append(
            {
                "record_id": f"challenge_{i}",
                "part": "challenge",
                "expected": expected,
                "partner": "",
                "reply": reply,
                "wording": -1,
            }
        )
    dialogue = [
        ("Would you like me to help you?", "Yes, please.", "accept"),
        ("Did you ask someone for help yesterday?", "Yes, please.", "action_report"),
        (
            "Do you intend to ask someone for help tomorrow?",
            "Yes, please.",
            "intention",
        ),
        (
            "Do you prefer receiving help to working alone?",
            "Yes, please.",
            "preference",
        ),
        ("Would you like me to help you?", "Yes.", "accept"),
        ("Did you ask someone for help yesterday?", "Yes.", "action_report"),
        ("Do you intend to ask someone for help tomorrow?", "Yes.", "intention"),
        ("Are you capable of asking someone for help?", "Yes.", "ability"),
        ("What do you need from me?", "Please explain this task to me.", "request"),
        (
            "I think you mean yesterday.",
            "No, I meant tomorrow, not yesterday.",
            "repair",
        ),
        (
            "You should solve this alone.",
            "I disagree. Getting help is better.",
            "disagree",
        ),
        ("Shall I explain the task?", "Yes, please explain it.", "accept"),
        (
            "How are you feeling right now?",
            "I feel relieved by your support.",
            "experience",
        ),
        (
            "What did you do yesterday?",
            "I asked a colleague to help me.",
            "action_report",
        ),
        (
            "What are you planning to do?",
            "I intend to ask a colleague for help.",
            "intention",
        ),
        (
            "I think the task is impossible.",
            "You are right; I now think it is impossible too.",
            "accept_proposition_not_help",
        ),
    ]
    for i, (partner, reply, expected) in enumerate(dialogue):
        rows.append(
            {
                "record_id": f"dialogue_{i}",
                "part": "dialogue",
                "expected": expected,
                "partner": partner,
                "reply": reply,
                "wording": -1,
            }
        )
    return pd.DataFrame(rows)


def natural_bank(limit):
    """Select raw sentences by fixed hashes, not model readings or labels."""
    from scripts.run_grain_multiview_address import load_material

    summary = json.loads((GRAPH / "summary.json").read_text())
    for name in ("private_parses.jsonl", "private_graphs.jsonl"):
        if digest(GRAPH / name) != summary["files"][name]:
            raise ValueError("Changed natural parser journal")
    _, _, raw, _, _ = load_material()
    parses = [
        json.loads(line)
        for line in (GRAPH / "private_parses.jsonl").read_text().splitlines()
    ]
    graphs = {
        r["source_row"]: r
        for r in map(
            json.loads, (GRAPH / "private_graphs.jsonl").read_text().splitlines()
        )
    }
    candidates = []
    for record in parses:
        source = record["source_row"]
        text = raw[source]["body"]
        if hashlib.sha256(text.encode()).hexdigest() != record["raw_sha256"]:
            raise ValueError("Natural original changed")
        for words in record["sentences"]:
            start, end = min(w["start"] for w in words), max(w["end"] for w in words)
            if 30 <= end - start <= 600:
                key = f"{source}:{start}:{end}"
                candidates.append(
                    (
                        hashlib.sha256(f"3407:{key}".encode()).hexdigest(),
                        source,
                        start,
                        end,
                        words[0]["sentence_index"],
                    )
                )
    rows, relations = [], []
    for _, source, start, end, sentence in sorted(candidates)[:limit]:
        rid = f"natural_{source}_{start}"
        rows.append(
            {
                "record_id": rid,
                "part": "natural",
                "expected": "unannotated",
                "partner": "",
                "reply": raw[source]["body"][start:end],
                "wording": -1,
                "source_row": source,
                "source_start": start,
                "source_end": end,
                "raw_sha256": graphs[source]["raw_sha256"],
            }
        )
        relations.append(
            {
                "record_id": rid,
                "events": [
                    e
                    for e in graphs[source]["events"]
                    if e["sentence_index"] == sentence
                ],
            }
        )
    return pd.DataFrame(rows), relations


def prepare(out, config):
    """Freeze code and material before inspecting the first new model output."""
    if out.exists():
        raise FileExistsError("Do not overwrite a frozen run")
    if (config["model"], config["revision"]) != (MODEL, REVISION):
        raise ValueError("Observer revision mismatch")
    out.mkdir(parents=True)
    bank = public_bank()
    bank.to_csv(out / "public_bank.csv", index=False)
    natural, relations = natural_bank(config["natural_limit"])
    natural.to_csv(out / "private_natural_bank.csv", index=False)
    write(out / "private_natural_relations.json", relations)
    write(out / "config.json", config)
    write(
        out / "freeze.json",
        {
            "files": {p: digest(ROOT / p) for p in FROZEN},
            "inputs": {
                p: digest(out / p)
                for p in (
                    "public_bank.csv",
                    "private_natural_bank.csv",
                    "private_natural_relations.json",
                    "config.json",
                )
            },
            "model": MODEL,
            "revision": REVISION,
            "questionnaire_labels_loaded": False,
            "natural_fitted": False,
            "meps_loaded": False,
        },
    )


def verify(out):
    """Reject changed experiment code or frozen inputs."""
    freeze = json.loads((out / "freeze.json").read_text())
    for base, key in ((ROOT, "files"), (out, "inputs")):
        for name, checksum in freeze[key].items():
            if digest(base / name) != checksum:
                raise ValueError("Frozen file changed: " + name)


def observe(out):
    """Run pinned local inference, preserving every query output and address."""
    verify(out)
    if (out / "observations.json").exists():
        raise FileExistsError("Observations already complete")
    bank = pd.concat(
        [
            pd.read_csv(out / "public_bank.csv").fillna(""),
            pd.read_csv(out / "private_natural_bank.csv").fillna(""),
        ],
        ignore_index=True,
    )
    req = []
    for row in bank.itertuples():
        req.extend(requests(row.record_id, row.reply, row.partner))
        if row.part == "dialogue":
            req.extend(
                requests(row.record_id, row.reply, row.partner, include_context=False)
            )
    frame = pd.DataFrame(req)
    frame.to_csv(out / "private_requests.csv", index=False)
    start = time.monotonic()
    predictions = infer_pairs(frame)
    observer = {
        "model": MODEL,
        "revision": REVISION,
        "device": "cpu",
        "threads": 4,
        "dtype": "float32",
        "local_files_only": True,
        "truncation": False,
    }
    seconds = time.monotonic() - start
    merged = frame.merge(predictions, on="audit_id", validate="one_to_one")
    merged.to_csv(out / "private_nli_pairs.csv", index=False)
    records = []
    for row in bank.itertuples():
        f = merged[(merged.record_id == row.record_id) & (merged.view == "context")]
        probabilities = {
            r.query: [r.p_entailment, r.p_neutral, r.p_contradiction]
            for r in f.itertuples()
        }
        records.append(
            evidence_record(row.record_id, row.reply, row.partner, probabilities)
        )
    write(out / "private_evidence.json", records)
    write(
        out / "observations.json",
        {
            "observer": observer,
            "pairs": len(merged),
            "seconds": seconds,
            "records": len(records),
            "files": {
                p: digest(out / p)
                for p in (
                    "private_requests.csv",
                    "private_nli_pairs.csv",
                    "private_evidence.json",
                )
            },
        },
    )


def correlation(a, b):
    """Keep undefined constant correlation distinct from zero."""
    return (
        float(np.corrcoef(a, b)[0, 1])
        if np.std(a) > 1e-12 and np.std(b) > 1e-12
        else None
    )


def analyze(out):
    """Analyze frozen readings; no new neural call or observer adaptation."""
    verify(out)
    observation = json.loads((out / "observations.json").read_text())
    for name, checksum in observation["files"].items():
        if digest(out / name) != checksum:
            raise ValueError("Changed observer output")
    config = json.loads((out / "config.json").read_text())
    public = pd.read_csv(out / "public_bank.csv").fillna("")
    pairs = pd.read_csv(out / "private_nli_pairs.csv")
    readings = pairs.pivot(
        index=["record_id", "view"], columns="query", values="p_entailment"
    )
    readings.to_csv(out / "readings.csv")
    main = public[public.part.isin(["source", "held"])].copy()
    source = main[main.part == "source"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    vectorizer.fit(source.reply)
    tfidf = vectorizer.transform(main.reply).toarray()
    h = np.array(
        [
            readings.loc[(r.record_id, "context"), list(MODES)].to_numpy(float)
            for r in main.itertuples()
        ]
    )
    content = np.array(
        [[readings.loc[(r.record_id, "context"), "content"]] for r in main.itertuples()]
    )
    arrays = {"content_query": content, "word_tfidf": tfidf, "typed_queries": h}
    fixed = {}
    rank = {}
    scores = []
    predicted = []
    catalogs = {}
    target = np.array([-1, 1, 0, 0, 0, 0], float)
    for method, arr in arrays.items():
        b = np.array(
            [arr[(main.part == "source") & (main.expected == m)].mean(0) for m in MODES]
        )
        fixed[method] = b
        rank[method] = target_identifiability(b, target)
        catalogs[method] = {
            "source_means": b.tolist(),
            "columns": list(MODES)
            if method == "typed_queries"
            else ["content"]
            if method == "content_query"
            else vectorizer.get_feature_names_out().tolist(),
        }
    rng = np.random.default_rng(config["seed"])
    truth = rng.dirichlet(np.full(len(MODES), 1.5), config["simulated_authors"])
    for person, pi in enumerate(truth):
        modes = rng.choice(len(MODES), config["utterances_per_author"], p=pi)
        wordings = rng.integers(0, 2, len(modes))
        for part in ("source", "held"):
            indices = [
                main.index[(main.expected == MODES[m]) & (main.part == part)][w]
                for m, w in zip(modes, wordings, strict=True)
            ]
            for method, arr in arrays.items():
                estimate, residual = simplex_decode(arr[indices].mean(0), fixed[method])
                for j, mode in enumerate(MODES):
                    predicted.append(
                        {
                            "person": person,
                            "part": part,
                            "method": method,
                            "mode": mode,
                            "target": float(pi[j]),
                            "observed_fraction": float(np.mean(modes == j)),
                            "estimate": float(estimate[j]),
                            "reconstruction_residual": residual,
                        }
                    )
    prediction = pd.DataFrame(predicted)
    for keys, g in prediction.groupby(["part", "method", "mode"]):
        scores.append(
            dict(
                zip(("part", "method", "mode"), keys, strict=True),
                mse=float(np.mean((g.estimate - g.target) ** 2)),
                r=correlation(g.estimate, g.target),
                observed_fraction_mse=float(
                    np.mean((g.observed_fraction - g.target) ** 2)
                ),
            )
        )
    prediction.to_csv(out / "simulated_composition_predictions.csv", index=False)
    pd.DataFrame(scores).to_csv(out / "composition_metrics.csv", index=False)
    write(out / "source_decoder.json", catalogs)
    write(out / "identifiability.json", rank)
    dialogue = []
    for r in public[public.part == "dialogue"].itertuples():
        for view in ("context", "reply_only"):
            v = readings.loc[(r.record_id, view)]
            allowed = [k for k in QUERIES if k != "content"]
            dialogue.append(
                {
                    "record_id": r.record_id,
                    "expected": r.expected,
                    "view": view,
                    "top_query": v[allowed].idxmax(),
                    "expected_entailment": float(v[r.expected])
                    if r.expected in QUERIES
                    else None,
                    "l2_context_change": float(
                        np.linalg.norm(
                            readings.loc[(r.record_id, "context"), allowed]
                            - readings.loc[(r.record_id, "reply_only"), allowed]
                        )
                    ),
                }
            )
    pd.DataFrame(dialogue).to_csv(out / "dialogue_comparison.csv", index=False)
    contrasts = []
    for w in range(4):
        for a, b in itertools.combinations(MODES, 2):
            ia = main.index[(main.expected == a) & (main.wording == w)][0]
            ib = main.index[(main.expected == b) & (main.wording == w)][0]
            contrasts.append(
                {
                    "wording": w,
                    "a": a,
                    "b": b,
                    "tfidf_cosine": float(tfidf[ia] @ tfidf[ib]),
                    "typed_l2": float(np.linalg.norm(h[ia] - h[ib])),
                    "content_difference": float(abs(content[ia, 0] - content[ib, 0])),
                }
            )
    pd.DataFrame(contrasts).to_csv(out / "identity_contrasts.csv", index=False)
    # Same opportunity-specific response, unequal supplied opportunity mixture.
    rates = np.array([0.8, 0.2])
    pooled = [opportunity_standardize(rates, w) for w in ([0.9, 0.1], [0.1, 0.9])]
    standardized = [opportunity_standardize(rates, [0.5, 0.5])] * 2
    natural_ids = pd.read_csv(out / "private_natural_bank.csv").record_id
    natural = readings.loc[[(rid, "context") for rid in natural_ids]]
    # A supported candidate is a property of the observer, not a verified label.
    write(
        out / "mechanism_summary.json",
        {
            "status": "COMPLETE",
            "simulated_authors": len(truth),
            "independent_utterances_per_author": config["utterances_per_author"],
            "paired_wordings_not_new_people": True,
            "opportunity_counterexample": {
                "same_response_by_opportunity": rates.tolist(),
                "pooled": pooled,
                "fixed_reference": standardized,
            },
            "natural_sentences": len(natural),
            "natural_mean_entailment": {k: float(v) for k, v in natural.mean().items()},
            "natural_gold_labels": False,
            "real_world_action_status": "unobserved",
            "neural_pairs": observation["pairs"],
            "neural_seconds": observation["seconds"],
            "experiment_scope": "single_mode_statement_mixture_not_personality_or_behavior_truth",
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "pairs": observation["pairs"],
                "seconds": observation["seconds"],
                "authors": len(truth),
                "natural": len(natural),
            }
        ),
        flush=True,
    )


def main():
    """Run explicit stages to preserve pre-result inputs and resumable outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "observe", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(
            args.output,
            json.loads((ROOT / "configs/grain_evidence_identity.json").read_text()),
        )
    else:
        {"observe": observe, "analyze": analyze}[args.stage](args.output)


if __name__ == "__main__":
    main()
