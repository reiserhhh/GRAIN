#!/usr/bin/env python3
"""Test causal-prefix invariance across future values and sequence lengths."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import (
    SYSTEM,
    build_controls,
    control_content_anchor,
    control_user_message,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Quantify, rather than hide, hardware/dtype-dependent prefix drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--full-precision-reduction", action="store_true")
    parser.add_argument("--pad-to-length", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/grain_vector_mechanisms_v1.json").read_text())
    torch.set_num_threads(4)
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.full_precision_reduction:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    observer = DecoderObserver.load(
        config["model"],
        revision=config["revision"],
        device=args.device,
        dtype=args.dtype,
        pad_to_length=args.pad_to_length,
    )
    write_json(
        args.output_dir / "model_manifest.json",
        {
            **observer.manifest(),
            "code_sha256": code_fingerprint(ROOT),
            "fp16_reduced_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        },
    )
    archive_source(ROOT, args.output_dir / "source_snapshot.zip")
    start = time.monotonic()
    rows = []
    for row in [r for r in build_controls() if r["variant"] == 0][: args.cases]:
        ids = observer.prompt_ids(SYSTEM, control_user_message(row))
        pos = observer.content_position(
            SYSTEM, control_user_message(row), control_content_anchor(row)
        )
        original, _ = observer.capture(ids, positions=[pos])
        changed_values = ids[: pos + 1] + [
            (v + 17) % observer.model.config.vocab_size for v in ids[pos + 1 :]
        ]
        conditions = {
            "repeat": ids,
            "future_values_same_length": changed_values,
            "prefix_only": ids[: pos + 1],
            "future_longer": ids + [17] * 17,
        }
        for name, other in conditions.items():
            assert ids[: pos + 1] == other[: pos + 1]
            states, _ = observer.capture(other, positions=[pos])
            for site in original:
                before, after = original[site], states[site]
                finite = bool(np.isfinite(before).all() and np.isfinite(after).all())
                rows.append(
                    {
                        "case_id": row["case_id"],
                        "site": site,
                        "condition": name,
                        "original_tokens": len(ids),
                        "other_tokens": len(other),
                        "finite": finite,
                        "max_absolute_difference": float(np.max(np.abs(before - after)))
                        if finite
                        else None,
                        "relative_l2_difference": float(
                            np.linalg.norm(before - after)
                            / max(np.linalg.norm(before), 1e-12)
                        )
                        if finite
                        else None,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "causal_prefix_invariance.csv", index=False)
    summary = frame.groupby("condition").agg(
        max_abs=("max_absolute_difference", "max"),
        max_relative=("relative_l2_difference", "max"),
        finite=("finite", "all"),
    )
    summary.to_csv(args.output_dir / "summary.csv")
    write_json(
        args.output_dir / "run_status.json",
        {
            "status": "COMPLETE",
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
            "cases": args.cases,
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
