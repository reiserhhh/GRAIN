#!/usr/bin/env python3
"""Independently replay P1J provenance, entity-prefix equality and scalar summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Return an artifact digest without loading model weights."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_audit(base, effects, transport, native, pairs):
    """Recalculate published quantities without importing collector arithmetic."""
    if len(base) != 48 or len(effects) != 144:
        raise ValueError("Incomplete table")
    if (
        base.case_id.duplicated().any()
        or effects.duplicated(["case_id", "patch"]).any()
    ):
        raise ValueError("Duplicate case")
    if set(effects.case_id) != set(base.case_id):
        raise ValueError("Different case populations")
    index = base.set_index("case_id")
    arithmetic, metric, restore = 0.0, 0.0, 0.0
    for table in (base, effects):
        arithmetic = max(
            arithmetic,
            float(abs(table.margin - table.high_logit + table.low_logit).max()),
        )
    for case, sub in effects.groupby("case_id"):
        if set(sub.patch) != {"N", "Q", "EQD"}:
            raise ValueError("Missing patch")
        r = index.loc[case]
        for row in sub.itertuples():
            donor = index.loc[row.donor_case]
            source = index.loc[row.state_source_case]
            if (
                any(donor[k] != r[k] for k in ("entity", "criterion", "group_id"))
                or donor.level != -r.level
            ):
                raise ValueError("Invalid paired donor")
            if any(
                source[k] != donor[k] for k in ("entity", "group_id", "level")
            ) or source.criterion != ("event" if row.patch == "N" else donor.criterion):
                raise ValueError("Common event-cache source not used")
            if row.patch == "EQD":
                restore = max(restore, abs(row.margin - donor.margin))
    for row in transport.itertuples():
        sub = effects[
            (effects.criterion == row.criterion) & (effects.patch == row.patch)
        ]
        original = sub.case_id.map(index.margin).to_numpy()
        donor = sub.donor_case.map(index.margin).to_numpy()
        gap, shift = donor - original, sub.margin.to_numpy() - original
        values = {
            "cases": len(sub),
            "relative_error_energy": np.dot(shift - gap, shift - gap)
            / np.dot(gap, gap),
            "gap_rms": np.sqrt(np.mean(gap**2)),
            "shift_rms": np.sqrt(np.mean(shift**2)),
            "toward_donor_fraction": np.mean(shift * gap > 0),
        }
        if row.criterion != "venue":
            values.update(
                intended_behavior_fraction=np.mean(
                    -sub.role_sign.to_numpy() * shift > 0
                ),
                intended_effect_mean=np.mean(-sub.role_sign.to_numpy() * shift),
            )
        for name, value in values.items():
            metric = max(metric, abs(float(value) - getattr(row, name)))
    for row in native.itertuples():
        sub = base[base.criterion == row.criterion]
        gaps = []
        for (setting, entity), pair in sub.groupby(["group_id", "entity"]):
            r = pair.set_index("role_sign").margin
            gap = r[1] - r[-1]
            saved = pairs[
                (pairs.criterion == row.criterion)
                & (pairs.group_id == setting)
                & (pairs.entity == entity)
            ]
            if len(saved) != 1:
                raise ValueError("Missing role pair")
            metric = max(metric, abs(gap - saved.role_aligned_gap.iloc[0]))
            gaps.append(gap)
        values = {
            "pairs": len(gaps),
            "positive_role_pairs": sum(g > 0 for g in gaps),
            "mean_role_gap": np.mean(gaps),
            "expected_preference": np.mean(sub.margin * sub.expected_sign > 0),
        }
        for name, value in values.items():
            metric = max(metric, abs(float(value) - getattr(row, name)))
    if max(arithmetic, metric, restore) > 1e-8:
        raise ValueError("Scalar reconstruction failed")
    return {
        "arithmetic_max": arithmetic,
        "metric_replay_max": metric,
        "full_restore_max": restore,
    }


def audit(result, parent, model_dir):
    """Audit complete entity states with independent tokenizer offsets and no forwards."""
    manifest = json.loads((result / "manifest.json").read_text())
    status = json.loads((result / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] != 196:
        raise ValueError("Incomplete P1J")
    for directory in (result, parent):
        for name, expected in json.loads(
            (directory / "output_hashes.json").read_text()
        ).items():
            if sha(directory / name) != expected:
                raise ValueError("Artifact changed")
    with zipfile.ZipFile(result / "source_snapshot.zip") as z:
        for name, expected in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("Source snapshot changed")
    if (
        manifest["parent_identity"]
        != json.loads((parent / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Parent identity changed")
    data = json.loads((result / "materials.json").read_text())
    parents = {
        r["case_id"]: r
        for r in json.loads((parent / "materials.json").read_text())["rows"]
    }
    layouts = json.loads((result / "token_layouts.json").read_text())
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    by_prefix, prefix_error = {}, 0.0
    for r in data["rows"]:
        p = parents[r["parent_case"]]
        subject = re.search(r"Did (Alex|Blair)\b", p["clauses"]["C"]).group(1)
        role = p["level"] if r["entity"] == subject else -p["level"]
        if (
            r["evidence"] != p["evidence"]
            or r["role_sign"] != role
            or r["expected_sign"] != (1 if r["criterion"] == "venue" else role)
        ):
            raise ValueError("Evidence/orientation changed")
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": data["system"]},
                {"role": "user", "content": r["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        layout = layouts[r["case_id"]]
        if encoded["input_ids"] != layout["ids"]:
            raise ValueError("Token identity changed")
        a, b = r["entity_span"]
        if r["user"][a:b].strip() != r["entity"]:
            raise ValueError("Wrong name interval")
        start = text.index(r["user"])
        indices = [
            j
            for j, (lo, hi) in enumerate(encoded["offset_mapping"])
            if start + a <= lo < hi <= start + b
        ]
        if indices != layout["N"] or len(indices) != 1:
            raise ValueError("Name token layout differs")
        current = (
            layout["ids"][: indices[-1] + 1],
            np.load(result / "activations" / (r["case_id"] + ".npz"))["block_16"][
                indices
            ],
        )
        key = (r["group_id"], r["level"], r["entity"])
        previous = by_prefix.setdefault(key, current)
        if previous[0] != current[0]:
            raise ValueError("Criterion visible before common state")
        prefix_error = max(prefix_error, float(abs(previous[1] - current[1]).max()))
    if prefix_error > 1e-5 or len(by_prefix) != 16:
        raise ValueError("Shared causal state not reproduced")
    tables = [
        pd.read_csv(result / (name + ".csv"))
        for name in (
            "baselines",
            "effects",
            "transport_summary",
            "native_summary",
            "native_role_pairs",
        )
    ]
    check = scalar_audit(*tables)
    return {
        "status": "VERIFIED",
        "result_identity": manifest["identity"],
        "new_neural_calls": 0,
        "cases": 48,
        "prefix_groups": 16,
        "prefix_state_max": prefix_error,
        **check,
        "interpretation": "Reused prefix state has behavioral effects AND irrelevant-venue leakage; not criterion-specific personality evidence.",
    }


def main():
    """Save the replay receipt and its own source separately from the original run."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result-dir", "parent-dir", "model-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    value = audit(args.result_dir, args.parent_dir, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    value["source_sha256"] = sha(Path(__file__))
    (args.output_dir / "summary.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
