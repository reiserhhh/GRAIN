#!/usr/bin/env python3
"""Compare complete frozen capability runs without equating generation and margins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.capability import capability_cases
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def summarize_run(frame):
    """Failures to return a binary answer count as errors, not missing cases."""
    original = frame[~frame.repeat].copy()
    if len(original) != 208 or original.case_id.nunique() != 208 or len(frame) != 216:
        raise ValueError("Need the entire frozen battery and repeats")
    if set(original.case_id) != {r["case_id"] for r in capability_cases()}:
        raise ValueError("Unexpected material cases")
    repeats = frame[frame.repeat].merge(
        original, on="case_id", suffixes=("_r", "_o"), validate="one_to_one"
    )
    repeat_match = repeats.content_r.fillna("") == repeats.content_o.fillna("")
    rows = []
    for (battery, family, split), sub in original.groupby(
        ["battery", "family", "split"]
    ):
        ci = grouped_bootstrap(sub.correct, sub.group_id)
        by_class = sub.groupby("expected_yes").correct.mean()
        rows.append(
            {
                "battery": battery,
                "family": family,
                "split": split,
                "n": len(sub),
                **ci,
                "parse_rate": float(sub.parse_ok.mean()),
                "balanced_accuracy": float(by_class.mean())
                if len(by_class) == 2
                else None,
            }
        )
    code = original[original.battery == "answer_code"]
    counts = code.groupby("family").correct.sum()
    code_pairs = code.pivot(
        index=["group_id", "variant"], columns="family", values="parsed_yes"
    )
    opposite = int(
        (
            (code_pairs.normal != code_pairs.swapped) & code_pairs.notna().all(axis=1)
        ).sum()
    )
    facts = original[original.battery == "semantic_fact"]
    family_ba = (
        facts.groupby(["family", "expected_yes"])
        .correct.mean()
        .groupby("family")
        .mean()
    )
    heldout = facts[facts.split == "heldout_scenario"]
    code_qualified = bool(counts.min() >= 6 and opposite >= 6)
    fact_qualified = bool(
        facts.correct.mean() >= 0.80
        and family_ba.min() >= 0.70
        and heldout.correct.mean() >= 0.75
    )
    summary = {
        "unique_cases": 208,
        "repeats": 8,
        "repeat_exact_matches": int(repeat_match.sum()),
        "parse_rate": float(original.parse_ok.mean()),
        "code_normal_correct": int(counts.loc["normal"]),
        "code_swapped_correct": int(counts.loc["swapped"]),
        "code_opposite_pairs": opposite,
        "code_qualified": code_qualified,
        "facts_accuracy": float(facts.correct.mean()),
        "facts_accuracy_ci": grouped_bootstrap(facts.correct, facts.group_id),
        "fact_family_balanced_accuracy": family_ba.to_dict(),
        "heldout_scenario_accuracy": float(heldout.correct.mean()),
        "facts_qualified": fact_qualified,
        "next_observer_qualified": code_qualified and fact_qualified,
        "scope": "Limited known text-evidence capability, not personality accuracy or whitebox equivalence",
    }
    return summary, rows


def main():
    """Keep observers and readout modes separate and preserve every failed condition."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new summary directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    summaries, details, originals = [], [], {}
    for directory in a.parent_dir:
        read = lambda name, directory=directory: json.loads(
            (directory / name).read_text(encoding="utf-8")
        )
        manifest, status = read("manifest.json"), read("run_status.json")
        if status["status"] != "COMPLETE":
            raise ValueError("Partial battery cannot be treated as complete capability")
        frame = pd.read_csv(directory / "responses.csv")
        summary, rows = summarize_run(frame)
        backend = manifest["backend"]
        summaries.append(
            {
                "backend": backend,
                "mode": "generated",
                "parent_identity": manifest["identity"],
                "runtime": status,
                **summary,
            }
        )
        details.extend({"backend": backend, "mode": "generated", **r} for r in rows)
        originals[backend] = frame[~frame.repeat]
        if "restricted_margin" in frame:
            restricted = frame.copy()
            restricted["parsed_yes"] = np.where(
                frame.restricted_margin == 0,
                np.nan,
                (frame.restricted_margin > 0).astype(float),
            )
            restricted["parse_ok"] = restricted.parsed_yes.notna()
            restricted["correct"] = restricted.parse_ok & (
                restricted.parsed_yes == restricted.expected_yes
            )
            rs, rr = summarize_run(restricted)
            rs["repeat_exact_matches"] = None
            summaries.append(
                {"backend": backend, "mode": "restricted_first_margin", **rs}
            )
            details.extend(
                {"backend": backend, "mode": "restricted_first_margin", **r} for r in rr
            )
            agreement = frame[~frame.repeat].copy()
            agreement["restricted_yes"] = restricted.loc[~frame.repeat, "parsed_yes"]
            agreement.to_csv(out / f"{backend}_readout_modes.csv", index=False)
    paired = []
    if "qwen" in originals and "ollama" in originals:
        joined = originals["ollama"].merge(
            originals["qwen"],
            on="case_id",
            suffixes=("_l", "_q"),
            validate="one_to_one",
        )
        for battery, sub in joined.groupby("battery_l"):
            delta = sub.correct_l.astype(float) - sub.correct_q.astype(float)
            paired.append(
                {
                    "battery": battery,
                    **grouped_bootstrap(delta, sub.group_id_l),
                    "comparison": "Llama Q4 Ollama generation minus Qwen BF16 greedy generation; multi-factor contrast",
                }
            )
    pd.DataFrame(details).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_comparison.csv", index=False)
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "observers": summaries,
            "paired": paired,
            "code_sha256": code_fingerprint(ROOT),
        },
    )
    print(json.dumps({"observers": summaries, "paired": paired}), flush=True)


if __name__ == "__main__":
    main()
