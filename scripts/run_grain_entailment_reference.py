#!/usr/bin/env python3
"""Standard NLI check on designed evidence pairs, not personality assessment.

This tests an available predecessor tool for the polarity/ownership gap found in
cosine retrieval. It is not a reproduction of supervised Affective-NLI, and does
not read any real participant text or outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_grain_predecessor_reference import BASE, digest, write
from scripts.run_grain_paraphrase_reference import contrasts

MODEL = "cross-encoder/nli-deberta-v3-base"
REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
EXPECTED = {
    "equivalent": "entailment",
    "opposite": "contradiction",
    "other_person": "neutral",
    "single_episode": "neutral",
}


def run(args):
    """Run pinned, unmodified NLI weights; save probabilities and every wrong case."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    rows = contrasts()
    write(
        args.output / "contract.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "device": "cpu",
            "dtype": "float32",
            "pairs": len(rows),
            "fixture_source": "author_designed_semantic_probes_not_natural_labels",
            "expected": EXPECTED,
            "participant_data_loaded": False,
            "code_sha256": digest(Path(__file__)),
            "fixture_code_sha256": digest(
                ROOT / "scripts/run_grain_paraphrase_reference.py"
            ),
            "scope": "Literal text entailment, not a psychological trait-validity judgment",
        },
    )
    path = snapshot_download(
        MODEL,
        revision=REVISION,
        local_files_only=not args.allow_download,
        allow_patterns=["*.json", "*.txt", "spm.model", "model.safetensors"],
    )
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, trust_remote_code=False, use_safetensors=True, dtype=torch.float32
    ).eval()
    mapping = {int(i): str(v).lower() for i, v in model.config.id2label.items()}
    if set(mapping.values()) != {"contradiction", "entailment", "neutral"}:
        raise ValueError("Unknown NLI label direction")
    results, start = [], time.monotonic()
    for i in range(0, len(rows), 8):
        subset = rows[i : i + 8]
        # Text is the premise; the IPIP-style statement is the hypothesis.
        batch = tokenizer(
            [r["sentence"] for r in subset],
            [r["target"] for r in subset],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        if batch["input_ids"].shape[1] > 512:
            raise ValueError("Fixture unexpectedly exceeds model context")
        with torch.inference_mode():
            probability = torch.softmax(model(**batch).logits, dim=-1).numpy()
        for row, p in zip(subset, probability, strict=True):
            predicted, expected = mapping[int(np.argmax(p))], EXPECTED[row["category"]]
            results.append(
                {
                    **row,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                    **{"p_" + mapping[j]: float(p[j]) for j in mapping},
                }
            )
    frame = pd.DataFrame(results)
    frame.to_csv(args.output / "public_nli_probes.csv", index=False)
    frame[~frame.correct].to_csv(args.output / "public_disagreements.csv", index=False)
    pd.crosstab(frame.expected, frame.predicted).to_csv(args.output / "confusion.csv")
    result = {
        "status": "COMPLETE",
        "seconds": time.monotonic() - start,
        "fixture_agreement": float(frame.correct.mean()),
        "agreement_by_category": frame.groupby("category").correct.mean().to_dict(),
        "mean_entailment_by_category": frame.groupby("category")
        .p_entailment.mean()
        .to_dict(),
        "n": len(frame),
        "n_per_category": 10,
        "labels_from": "designed_literal_semantic_cases",
        "fitted_on_fixtures": False,
        "personality_scores_predicted": False,
        "interpretation_boundary": "Single-episode generic statements can be semantically ambiguous; disagreements remain inspectable.",
    }
    write(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    """Download only the public pinned standard model when explicitly enabled."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=BASE / "predecessor_integration_v1/entailment_r1"
    )
    p.add_argument("--allow-download", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
