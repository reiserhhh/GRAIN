#!/usr/bin/env python3
"""Bounded no-outcome S9 observer precision/shape comparison on four fixed views."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import (
    candidates,
    native_readout,
    tokenize_prompt,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_real_author_path import MODEL, read, sha, write
from scripts.run_grain_source_remainder import verify_prepared


def main():
    """Compare native full-head paths; never inspect source outcomes or alter S9."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    _, material = verify_prepared(source, "mps")
    out = source / "observer_check"
    if out.exists():
        raise ValueError("Preserve completed observer check")
    out.mkdir()
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    selected = material[:4]
    rows, errors = [], []
    start = time.monotonic()
    calls = 0
    for dtype in ("float32", "bfloat16"):
        observer = None
        try:
            observer = DecoderObserver.load(
                str(MODEL), device="mps", dtype=dtype, max_tokens=3072
            )
            observer.model.requires_grad_(False)
            for pad in (None, 3072):
                observer.pad_to_length = pad
                for author in selected:
                    if time.monotonic() - start > 300:
                        raise TimeoutError("Observer check reached five-minute cap")
                    view = author["views"][0]
                    ids, _, _ = tokenize_prompt(
                        view["texts"], author["query"], observer.tokenizer
                    )
                    if material_hash(ids) != view["prompt_sha256"]:
                        raise ValueError("Observer prompt changed")
                    alphabet = candidates(author["query"], observer.tokenizer)
                    _, logits = observer.capture(ids)
                    calls += 1
                    score = native_readout(logits, alphabet)["conditional_score"]
                    s9 = read(
                        source
                        / "private_packets"
                        / f"author_{author['index']:03d}.json"
                    )["views"][0]["new_teacher"]
                    rows.append(
                        {
                            "author_index": author["index"],
                            "query": author["query"],
                            "dtype": dtype,
                            "padding": pad or 0,
                            "score": score,
                            "old_cuda": view["old_teacher"],
                            "s9_origin": s9,
                            "difference_old_cuda": score - view["old_teacher"],
                            "difference_s9": score - s9,
                        }
                    )
        except (RuntimeError, TypeError, TimeoutError) as exc:
            errors.append(
                {"dtype": dtype, "error": type(exc).__name__, "message": str(exc)}
            )
        finally:
            if observer is not None:
                del observer
            gc.collect()
            torch.mps.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_scores.csv", index=False)
    summary = []
    for (dtype, padding), g in frame.groupby(["dtype", "padding"]):
        summary.append(
            {
                "dtype": dtype,
                "padding": int(padding),
                "views": len(g),
                "rms_difference_old_cuda": float(
                    np.sqrt(np.mean(g.difference_old_cuda**2))
                ),
                "rms_difference_s9": float(np.sqrt(np.mean(g.difference_s9**2))),
                "max_difference_s9": float(np.max(np.abs(g.difference_s9))),
            }
        )
    write(
        out / "summary.json",
        {
            "selection": "first four prepared authors, view0, no outcome/effect selection",
            "source_manifest_sha256": sha(source / "manifest.json"),
            "script_sha256": sha(__file__),
            "calls": calls,
            "seconds": time.monotonic() - start,
            "labels_read": False,
            "rows": summary,
            "errors": errors,
        },
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(
        {"calls": calls, "seconds": time.monotonic() - start, "errors": len(errors)},
        flush=True,
    )


if __name__ == "__main__":
    main()
