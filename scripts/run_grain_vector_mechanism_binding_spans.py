#!/usr/bin/env python3
"""Locate distributed binding use by matched, pre-answer token-span replacement."""

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

from grain_research.vector_mechanisms.binding import binding_user_message
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Compare identity/event/evidence spans; all-prompt replacement is only a control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_status.json").exists():
        raise ValueError("Use a new output directory")
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    config, system = parent["config"], parent["system"]
    if (
        json.loads((args.parent_dir / "run_status.json").read_text())["status"]
        != "COMPLETE"
    ):
        raise ValueError("Parent incomplete")
    cases = parent["cases"]
    by_key = {(r["group_id"], r["a"], r["b"], r["order"], r["task"]): r for r in cases}
    targets = [
        r
        for r in cases
        if r["split"] == "heldout_scenario"
        and r["a"] == 0
        and r["task"] == "narrator_positive"
    ]
    manifest = {
        "parent_identity": parent["identity"],
        "code_sha256": code_fingerprint(ROOT),
        "scope": "exploratory matched donor identity swap; not psychological validity",
        "sites": ["block_10", "block_16"],
        "target_ids": [r["case_id"] for r in targets],
        "random_control": "independent Gaussian direction, same L2 norm at every patched token",
        "max_calls": 1536,
        "max_seconds": 1800,
        "labels": "synthetic_only",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    archive_source(ROOT, out / "source_snapshot.zip")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
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
    tokens = [
        observer.tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
    ]
    if any(len(ids) != 1 for ids in tokens):
        raise ValueError("Need one-token readout")
    yes, no = tokens[0][0], tokens[1][0]
    rows, controls = [], []
    start = time.monotonic()
    for ti, target in enumerate(targets):
        for task in ("narrator_positive", "alias_action"):
            row = by_key[target["group_id"], 0, target["b"], target["order"], task]
            donor = by_key[target["group_id"], 1, target["b"], target["order"], task]
            user, donor_user = binding_user_message(row), binding_user_message(donor)
            ids, donor_ids = (
                observer.prompt_ids(system, user),
                observer.prompt_ids(system, donor_user),
            )
            if len(ids) != len(donor_ids):
                raise ValueError("Matched span patch requires equal sequence lengths")
            positions = list(range(len(ids)))
            before, logits = observer.capture(ids, positions=positions)
            after, donor_logits = observer.capture(donor_ids, positions=positions)
            baseline, natural = (
                float(logits[yes] - logits[no]),
                float(donor_logits[yes] - donor_logits[no]),
            )
            spans = {}
            for name, key in (
                ("identity_record", "identity_record"),
                ("event_record", "event_record"),
                ("all_evidence", "passage"),
            ):
                pos = observer.content_indices(system, user, row[key])
                donor_pos = observer.content_indices(system, donor_user, donor[key])
                if pos != donor_pos:
                    raise ValueError(
                        "Token spans do not align; no arbitrary interpolation"
                    )
                spans[name] = pos
            spans["all_prompt_control"] = positions
            for site in manifest["sites"]:
                for span, pos in spans.items():
                    current, other = (
                        before[site][pos].astype(float),
                        after[site][pos].astype(float),
                    )
                    delta = other - current
                    arms = [("full", other), ("self", current)]
                    for rep in range(2):
                        seed = int(
                            material_hash([row["case_id"], site, span, rep])[:16], 16
                        )
                        random = np.random.default_rng(seed).normal(size=delta.shape)
                        random *= np.linalg.norm(
                            delta, axis=1, keepdims=True
                        ) / np.maximum(
                            np.linalg.norm(random, axis=1, keepdims=True), 1e-12
                        )
                        arms.append((f"random_{rep}", current + random))
                    for arm, values in arms:
                        _, patched = observer.capture(
                            ids, patch=Patch(site, pos, values)
                        )
                        score = float(patched[yes] - patched[no])
                        rows.append(
                            {
                                "case_id": row["case_id"],
                                "group_id": row["group_id"],
                                "task": task,
                                "order": row["order"],
                                "b": row["b"],
                                "site": site,
                                "span": span,
                                "arm": arm,
                                "positions": len(pos),
                                "baseline_margin": baseline,
                                "patched_margin": score,
                                "natural_donor_margin": natural,
                                "delta": score - baseline,
                                "oriented_delta": (score - baseline)
                                * (2 * donor["expected_yes"] - 1)
                                if task == "narrator_positive"
                                else None,
                                "donor_reproduction_error": abs(score - natural),
                                "effect_norm": np.linalg.norm(values - current),
                            }
                        )
            _, restored = observer.capture(ids)
            controls.append(
                {
                    "case_id": row["case_id"],
                    "restore_abs": abs(float(restored[yes] - restored[no]) - baseline),
                }
            )
            if (
                observer.calls > manifest["max_calls"]
                or time.monotonic() - start > manifest["max_seconds"]
            ):
                raise RuntimeError("Span intervention budget exceeded")
        pd.DataFrame(rows).to_csv(out / "span_interventions.csv", index=False)
        print(f"SPANS {ti + 1}/{len(targets)} calls={observer.calls}", flush=True)
    self_max = max(abs(r["delta"]) for r in rows if r["arm"] == "self")
    full_prompt_error = max(
        r["donor_reproduction_error"]
        for r in rows
        if r["arm"] == "full" and r["span"] == "all_prompt_control"
    )
    restore_max = max(r["restore_abs"] for r in controls)
    passed = max(self_max, full_prompt_error, restore_max) == 0
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE" if passed else "FAILED_CONTROL",
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
            "rows": len(rows),
            "self_max_abs": self_max,
            "all_prompt_donor_max_abs": full_prompt_error,
            "restore_max_abs": restore_max,
        },
    )
    if not passed:
        raise RuntimeError("Span intervention controls failed")


if __name__ == "__main__":
    main()
