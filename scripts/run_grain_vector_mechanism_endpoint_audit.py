#!/usr/bin/env python3
"""Bounded semantic endpoint check after an all-No natural context pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import QUESTIONS, SYSTEM, render_context
from grain_research.vector_mechanisms.controls import SYSTEM as FACT_SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_controls import write_json

CASES = [
    (
        "positive_explicit",
        "I usually initiate social contact. Last weekend I invited three friends to dinner, and yesterday I introduced myself to a new neighbor. Organizing shared activities is something I regularly choose to do.",
        1,
    ),
    (
        "positive_paraphrase",
        "Bringing people together is my normal preference. I asked new colleagues to join me for lunch on Monday and arranged a group walk on Friday. Nobody required me to do these things.",
        1,
    ),
    (
        "negative_explicit",
        "I do not initiate social contact. I decline invitations and never organize shared activities. Yesterday I spent the afternoon alone and decided not to call anyone.",
        0,
    ),
    (
        "other_subject",
        "My colleague usually initiates social contact. She invited three friends to dinner and introduced herself to a new neighbor. I do not share that tendency; I avoid initiating contact or organizing shared activities.",
        0,
    ),
    (
        "wish_not_actual",
        "I wish I were someone who initiated social contact, but I am not. I imagined inviting people to dinner but did not invite anyone. I have not started a conversation or arranged a shared activity.",
        0,
    ),
    (
        "neutral_no_evidence",
        "The package arrived at noon. I checked the printed address, placed the box on a shelf, and read the instructions. The diagram listed four parts and two tools.",
        0,
    ),
]


def main():
    """Use explicit synthetic facts, balanced systems, and polarity before recalibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_vector_contexts_v1.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "run_status.json").exists():
        raise ValueError("Use a new endpoint-audit directory")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    observer = DecoderObserver.load(
        name=config["model"],
        **{
            k: config[k]
            for k in (
                "revision",
                "device",
                "dtype",
                "max_tokens",
                "local_files_only",
                "pad_to_length",
            )
        },
    )
    manifest = {
        "config": config,
        "code_sha256": code_fingerprint(ROOT),
        "synthetic_cases": CASES,
        "systems": {"context": SYSTEM, "fact": FACT_SYSTEM},
        "tasks": QUESTIONS,
        "scope": "Explicit synthetic narrative truth only; not real author labels",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "model_manifest.json", observer.manifest())
    archive_source(ROOT, args.output_dir / "source_snapshot.zip")
    yids, nids = [
        observer.tokenizer.encode(x, add_special_tokens=False) for x in ("Yes", "No")
    ]
    if len(yids) != 1 or len(nids) != 1:
        raise ValueError("Audit needs one-token Yes/No")
    rows = []
    for key, text, expected in CASES:
        for task in QUESTIONS:
            for system_name, system in (("context", SYSTEM), ("fact", FACT_SYSTEM)):
                user, _ = render_context(None, text, task)
                for polarity in ("original", "reverse_answer_code"):
                    prompt = user
                    actual_system = system
                    if polarity != "original":
                        actual_system = system.replace(
                            "Yes means supported by the excerpts; No means not supported, not proof of the opposite.",
                            "",
                        )
                        actual_system += " For this trial reverse the output code: write No when the question is supported by the evidence, and Yes when it is not supported."
                    ids = observer.prompt_ids(actual_system, prompt)
                    _, logits = observer.capture(ids)
                    if not np.isfinite(logits).all():
                        raise ValueError("Nonfinite endpoint")
                    margin = float(logits[yids[0]] - logits[nids[0]])
                    oriented = margin if polarity == "original" else -margin
                    top = np.argsort(logits)[-5:][::-1]
                    rows.append(
                        {
                            "case_id": key,
                            "expected_supported": expected,
                            "task": task,
                            "system": system_name,
                            "polarity": polarity,
                            "actual_system_sha256": material_hash(actual_system),
                            "prompt_token_sha256": material_hash(ids),
                            "raw_margin": margin,
                            "support_oriented_margin": oriented,
                            "correct_default": int((oriented >= 0) == expected),
                            "top_tokens": [
                                observer.tokenizer.decode([int(i)]) for i in top
                            ],
                        }
                    )
    pd.DataFrame(rows).to_csv(args.output_dir / "endpoint_controls.csv", index=False)
    summary = (
        pd.DataFrame(rows)
        .groupby(["task", "system", "polarity"])
        .correct_default.mean()
        .reset_index()
    )
    summary.to_csv(args.output_dir / "endpoint_summary.csv", index=False)
    write_json(
        args.output_dir / "run_status.json",
        {
            "status": "COMPLETE",
            "calls": observer.calls,
            "cases": len(rows),
            "labels_scope": "synthetic_semantic_facts",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
