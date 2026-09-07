#!/usr/bin/env python3
"""Independent token-offset and scalar audit for the frozen P1H order panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    """Read a local artifact hash without mutating it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(parent, previous, model_dir):
    """Rebuild semantic offsets and restoration metrics without importing the collector."""
    manifest = json.loads((parent / "manifest.json").read_text())
    state = json.loads((parent / "run_status.json").read_text())
    if state["status"] != "COMPLETE" or state["calls"] != 340:
        raise ValueError("Incomplete P1H")
    for name, expected in json.loads(
        (parent / "output_hashes.json").read_text()
    ).items():
        if digest(parent / name) != expected:
            raise ValueError("Output checksum mismatch")
    with zipfile.ZipFile(parent / "source_snapshot.zip") as z:
        for name, expected in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("Source checksum mismatch")
    base = pd.read_csv(parent / "baselines.csv")
    effects = pd.read_csv(parent / "effects.csv")
    saved = pd.read_csv(parent / "transport_summary.csv")
    if len(base) != 48 or base.case_id.nunique() != 48 or len(effects) != 288:
        raise ValueError("Incomplete grid")
    for _, sub in effects.groupby("case_id"):
        if len(sub) != 6 or set(sub.patch) != {"C", "A", "R", "Q", "D", "EQD"}:
            raise ValueError("Missing patch region")
    layouts = json.loads((parent / "token_layouts.json").read_text())
    material = json.loads((parent / "materials.json").read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    original = pd.read_csv(previous / "baselines.csv").set_index("case_id").margin
    lookup = base.set_index("case_id").margin
    replay, layout_count = 0.0, 0
    for row in material["rows"]:
        layout = layouts[row["case_id"]]
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": material["system"]},
                {"role": "user", "content": row["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) != layout["ids"] or text.count(row["user"]) != 1:
            raise ValueError("Token sequence changed")
        origin = text.index(row["user"])
        for key, (a, b) in row["intervals"].items():
            if row["user"][a:b].strip() != row["clauses"][key]:
                raise ValueError("Clause identity changed")
            actual = [
                j
                for j, (left, right) in enumerate(encoded["offset_mapping"])
                if origin + a <= left < right <= origin + b
            ]
            if actual != layout["semantic"][key]:
                raise ValueError("Semantic interval mismatch")
        regions = layout["semantic"]
        if (
            sorted(i for part in regions.values() for i in part)
            != layout["partitions"]["Q"]
        ):
            raise ValueError("Boundary overlap or missing token")
        first, second = ("C", "A") if row["form"] == "question_first" else ("A", "C")
        if max(regions[first]) >= min(regions[second]):
            raise ValueError("Order manipulation failed")
        if row["form"] == "question_first":
            replay = max(
                replay, abs(lookup[row["case_id"]] - original[row["parent_case"]])
            )
        layout_count += 1
    arithmetic = max(
        float(abs(df.margin - df.high_logit + df.low_logit).max())
        for df in (base, effects)
    )
    full = effects[effects.patch == "EQD"]
    full_error = float(abs(full.margin - full.donor_case.map(lookup)).max())
    metric_error = 0.0
    for row in saved.itertuples():
        sub = effects[
            (effects.variant == row.variant)
            & (effects.form == row.form)
            & (effects.patch == row.patch)
        ]
        gap = sub.donor_case.map(lookup).to_numpy() - sub.case_id.map(lookup).to_numpy()
        error = sub.margin.to_numpy() - sub.donor_case.map(lookup).to_numpy()
        expected = np.dot(error, error) / np.dot(gap, gap)
        metric_error = max(metric_error, abs(expected - row.relative_error_energy))
    if max(arithmetic, full_error, metric_error) > 1e-8 or replay > 1e-5:
        raise ValueError("Replay failed")
    return {
        "status": "VERIFIED",
        "parent_identity": manifest["identity"],
        "cases": layout_count,
        "arithmetic_max": arithmetic,
        "full_restore_max": full_error,
        "metric_replay_max": metric_error,
        "previous_native_replay_max": replay,
        "new_neural_calls": 0,
        "interpretation": "token and arithmetic validity, not unique mechanism or personality validity",
    }


def main():
    """Write a distinct receipt and source; never overwrite parent outputs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--previous-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.parent_dir, args.previous_dir, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    result["source_sha256"] = digest(Path(__file__))
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
