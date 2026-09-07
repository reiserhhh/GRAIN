#!/usr/bin/env python3
"""Bounded content-versus-answer-channel follow-up to the complete S1 panel."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import SYSTEM, question, render
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.analyze_grain_multiquery_sensitivity import corr, load_panel, response_rows
from scripts.run_grain_multiquery_sensitivity import (
    differentiable_capture,
    edit_indices,
    native_alphabet,
    token_groups,
)
from scripts.run_grain_real_author_path import BASE, MODEL, capture, sha, write

CONTROL_QUESTION = (
    "Estimate how much the writer's excerpts discuss numerical quantities, measurements, "
    "prices or calculations. Use 0 for very little, 9 for very much, and the digits in "
    "between for intermediate amounts. Return only one digit from 0 to 9."
)


def control_prompt(texts):
    """Change only the question after the same excerpt renderer; no personality cue."""
    user, spans = render(texts, "neuroticism")
    return user.replace(question("neuroticism"), CONTROL_QUESTION), spans


def main():
    """Collect 42F/8B on the first eight author view-zero units, then compare effects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private probe must stay in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    manifest, records = load_panel(args.panel)
    records = [r for r in records if r[0]["panel_index"] < 8 and r[0]["view"] == 0]
    if len(records) != 8:
        raise ValueError("The frozen eight-author subpanel must be complete")
    if sha(MODEL / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Observer changed")
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S1B_NUMERIC_CONTENT_PROBE_V1",
            "panel_sha256": sha(args.panel / "manifest.json"),
            "script_sha256": sha(__file__),
            "question": CONTROL_QUESTION,
            "selection": "first8 S1 authors, view0, same E1/E2; fixed without control-query responses",
            "max_forwards": 42,
            "max_backwards": 8,
            "max_seconds": 600,
            "labels_read": False,
        },
    )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    observer.sites = {"block_16": observer.model.model.layers[16]}
    alphabet = native_alphabet("neuroticism", observer.tokenizer)
    start, rows, arrays, metadata = time.monotonic(), [], {}, []
    try:
        for i, (material, packet, _) in enumerate(records):
            user, spans = control_prompt(material["texts"])
            ids = observer.prompt_ids(SYSTEM, user)
            groups = token_groups(observer, user, spans, ids)
            h, g, attribution, base = differentiable_capture(
                observer, ids, alphabet, groups
            )
            arrays[f"panel_{i}__full"], arrays[f"panel_{i}__gradient"] = h, g
            if i == 0:
                _, repeat = capture(observer, ids, alphabet)
                _, self_patch = capture(
                    observer, ids, alphabet, Patch("block_16", len(ids) - 1, h)
                )
                error = max(
                    abs(s["conditional_score"] - base["conditional_score"])
                    for s in (repeat, self_patch)
                )
                if error > 1e-5 or np.linalg.norm(g) == 0:
                    raise ValueError("Control instrument check failed")
                write(out / "instrument_checks.json", {"repeat_self_error": error})
            masks = edit_indices(12, material["e1"], material["e2"])
            scores = {"full": base}
            for edit in ("e1", "e2"):
                prompt, _ = control_prompt([material["texts"][j] for j in masks[edit]])
                state, natural = capture(
                    observer, observer.prompt_ids(SYSTEM, prompt), alphabet
                )
                donor = state["block_16"][0]
                _, patch = capture(
                    observer, ids, alphabet, Patch("block_16", len(ids) - 1, donor)
                )
                arrays[f"panel_{i}__{edit}"] = donor
                scores[edit], scores["patch_" + edit] = natural, patch
                rows.append(
                    {
                        "author_id": material["author_id"],
                        "view": 0,
                        "edit": edit,
                        "query": "quantitative_content",
                        "natural": natural["conditional_score"]
                        - base["conditional_score"],
                        "patch": patch["conditional_score"] - base["conditional_score"],
                        "linear": float(g @ (donor - h)),
                    }
                )
            metadata.append(
                {
                    "author_id": material["author_id"],
                    "panel_index": material["panel_index"],
                    "scores": scores,
                    "attribution": attribution,
                    "body_token_counts": [len(group) for group in groups],
                }
            )
            write(out / "private_packets.json", metadata)
            np.savez_compressed(out / "private_states.npz", **arrays)
            print(
                f"S1B quantitative-content query: {i + 1}/8, {observer.calls}F",
                flush=True,
            )
            if time.monotonic() - start > 600:
                raise TimeoutError("Bounded content probe exceeded ten minutes")
    finally:
        write(
            out / "cost_ledger.json",
            {
                "forwards": observer.calls,
                "backwards": len(metadata),
                "seconds": time.monotonic() - start,
            },
        )
    if observer.calls != 42:
        raise ValueError("Unexpected call count")
    old, _, _ = response_rows(records)
    combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(out / "private_responses.csv", index=False)
    comparisons = []
    for signal in ("natural", "patch", "linear"):
        table = combined.pivot(
            index=["author_id", "view", "edit"], columns="query", values=signal
        )
        table -= table.groupby(level=["view", "edit"]).transform("mean")
        for left, right in (
            ("neuroticism", "confirmation_seeking"),
            ("neuroticism", "quantitative_content"),
            ("confirmation_seeking", "quantitative_content"),
            ("TF_cont", "quantitative_content"),
        ):
            comparisons.append(
                {
                    "signal": signal,
                    "left": left,
                    "right": right,
                    "authors": 8,
                    "centered_r": corr(table[left], table[right]),
                }
            )
    metrics = pd.DataFrame(comparisons)
    metrics.to_csv(out / "shared_metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "forwards": observer.calls,
            "backwards": 8,
            "seconds": time.monotonic() - start,
            "labels_read": False,
            "metrics": comparisons,
            "states_sha256": sha(out / "private_states.npz"),
        },
    )
    report = "# Same numeric alphabet, different query content\n\n"
    report += "Matched first8 authors/view0/E1/E2; edit-centered descriptive response correlations.\n"
    report += "This is a content probe, not a requirement that every unrelated query have zero covariance.\n\n"
    report += metrics.to_markdown(index=False, floatfmt=".4f") + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
