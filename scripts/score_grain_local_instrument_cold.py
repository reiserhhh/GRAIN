#!/usr/bin/env python3
"""Preload fixed runtime dependencies, then use the frozen guarded cold scorer.

The r1 engine installed its data guard before Transformers' lazy dependency
metadata checks. Preloading model/tokenizer classes requires no inputs or model
forward calls and does not relax the later application-data allowlist.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_grain_local_instrument as engine


def loaded_reader(tokenizer, model, torch):
    """Apply the original reader's batching and logits to preloaded pinned assets."""
    labels = {int(i): str(v).lower() for i, v in model.config.id2label.items()}
    if set(labels.values()) != {"contradiction", "neutral", "entailment"}:
        raise ValueError("Unknown NLI label encoding")

    def infer(frame):
        rows = []
        for start in range(0, len(frame), 8):
            part = frame.iloc[start : start + 8]
            batch = tokenizer(
                part.sentence.tolist(),
                part.item_text.tolist(),
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if batch["input_ids"].shape[1] > 512:
                raise ValueError("Pair exceeds NLI context; no silent truncation")
            with torch.inference_mode():
                probs = torch.softmax(model(**batch).logits, -1).numpy()
            for row, p in zip(part.itertuples(), probs, strict=True):
                rows.append(
                    {
                        "audit_id": row.audit_id,
                        "nli": labels[int(p.argmax())],
                        **{"p_" + labels[i]: float(p[i]) for i in labels},
                    }
                )
        return engine.pd.DataFrame(rows)

    return infer


def run(args):
    """Keep the total cold deadline inclusive of dependency/model preloading."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    start = time.monotonic()
    limit = args.max_seconds

    def timeout(_sig, _frame):
        raise TimeoutError("Cold runtime preload deadline exceeded")

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(limit)
    wrapper_sha = engine.sha(Path(__file__))
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            engine.MODEL, revision=engine.REVISION, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            engine.MODEL,
            revision=engine.REVISION,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
        ).eval()
        torch.set_num_threads(4)
        # Only asset loading is outside the guard; raw text and every forward stay inside.
        engine.infer_pairs = loaded_reader(tokenizer, model, torch)
        elapsed = time.monotonic() - start
        args.max_seconds = int(limit - elapsed)
        if args.max_seconds < 1:
            raise TimeoutError("No cold inference budget left after preload")
        engine.run(args)
        path = args.output / "manifest.json"
        metadata = engine.read(path)
        metadata.update(
            dependency_preload_seconds=elapsed,
            total_wall_seconds=time.monotonic() - start,
            wrapper_sha256=wrapper_sha,
            preload_forward_calls=0,
            offline=True,
            implicit_credentials=False,
            execution_repair="Preload pinned assets and bind the same pair-forward operations; raw inputs/forwards remain under the unchanged r1 guard",
        )
        engine.write(path, metadata)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=160)
    parser.add_argument("--max-seconds", type=int, default=180)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)
