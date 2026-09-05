#!/usr/bin/env python3
"""Explore identity-binding interventions before the question, with action controls."""

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
from grain_research.vector_mechanisms.interventions import (
    contrast_basis,
    matched_random_effect,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Preserve task wording and event facts while replacing identity-sensitive states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_status.json").exists():
        raise ValueError("Use a new output directory")
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    status = json.loads((args.parent_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["pre_question_invariance_max_abs"] != 0:
        raise ValueError("Binding instrument is not validated")
    config, system = parent["config"], parent["system"]
    cases = parent["cases"]
    by_key = {(r["group_id"], r["a"], r["b"], r["order"], r["task"]): r for r in cases}
    sites = ("block_10", "block_16")
    states = {}
    for row in cases:
        with np.load(args.parent_dir / "cache" / f"{row['case_id']}.npz") as saved:
            states[row["case_id"]] = {
                k: saved[k].astype(float)
                for k in saved.files
                if k.endswith("__evidence_end")
            }
    bases = {}
    discovery = [
        r
        for r in cases
        if r["split"] == "discovery"
        and r["a"] == 0
        and r["task"] == "narrator_positive"
    ]
    for site in sites:
        key = site + "__evidence_end"
        differences = []
        for row in discovery:
            donor = by_key[row["group_id"], 1, row["b"], row["order"], row["task"]]
            differences.append(
                states[donor["case_id"]][key] - states[row["case_id"]][key]
            )
        bases[site] = contrast_basis(np.stack(differences), 2)
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
        "sites": sites,
        "position": "pre_question_evidence_end",
        "rank": 2,
        "target_ids": [r["case_id"] for r in targets],
        "random_replicates": 3,
        "scope": "exploratory binding localization after observing parent; not untouched confirmation",
        "maximum_calls": 1024,
        "maximum_seconds": 1800,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    np.savez_compressed(out / "discovery_bases.npz", **bases)
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
    answer_ids = [
        observer.tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
    ]
    if any(len(v) != 1 for v in answer_ids):
        raise ValueError("Binding patch needs one-token Yes/No")

    def margin(ids, patch=None):
        """Use one shared forward, exactly matching the parent's logit contrast."""
        _, logits = observer.capture(ids, patch=patch)
        return float(logits[answer_ids[0][0]] - logits[answer_ids[1][0]])

    start = time.monotonic()
    rows, controls = [], []
    original = pd.read_csv(args.parent_dir / "predictions.csv").set_index("case_id")
    for ti, target in enumerate(targets):
        for task in ("narrator_positive", "alias_action"):
            row = by_key[target["group_id"], 0, target["b"], target["order"], task]
            donor = by_key[target["group_id"], 1, target["b"], target["order"], task]
            user = binding_user_message(row)
            ids = observer.prompt_ids(system, user)
            position = observer.content_position(system, user, row["passage"])
            if (
                material_hash(ids)
                != original.loc[row["case_id"], "prompt_token_sha256"]
            ):
                raise ValueError("Token sequence changed relative to parent")
            baseline = margin(ids)
            replay = abs(baseline - original.loc[row["case_id"], "margin"])
            for site in sites:
                key = site + "__evidence_end"
                current, other = (
                    states[row["case_id"]][key],
                    states[donor["case_id"]][key],
                )
                delta = other - current
                effect = bases[site] @ (bases[site].T @ delta)
                arms = [
                    ("self", current, 1),
                    ("full", other, 1),
                    ("rank2_half", current + effect, 0.5),
                    ("rank2_full", current + effect, 1),
                ]
                arms += [
                    (
                        f"random_{r}",
                        current
                        + matched_random_effect(
                            delta, effect, 2, f"{row['case_id']}:{site}:{r}"
                        ),
                        1,
                    )
                    for r in range(3)
                ]
                for arm, vector, alpha in arms:
                    score = margin(ids, patch=Patch(site, position, vector, alpha))
                    difference = score - baseline
                    rows.append(
                        {
                            "group_id": row["group_id"],
                            "case_id": row["case_id"],
                            "order": row["order"],
                            "b": row["b"],
                            "task": task,
                            "site": site,
                            "arm": arm,
                            "baseline_margin": baseline,
                            "patched_margin": score,
                            "delta": difference,
                            "donor_truth_oriented_delta": difference
                            * (2 * donor["expected_yes"] - 1)
                            if task == "narrator_positive"
                            else None,
                            "natural_donor_margin_delta": original.loc[
                                donor["case_id"], "margin"
                            ]
                            - baseline,
                            "effect_norm": np.linalg.norm((vector - current) * alpha),
                            "correct_default": int(
                                (score >= 0) == donor["expected_yes"]
                            ),
                        }
                    )
                restored = margin(ids)
                controls.append(
                    {
                        "case_id": row["case_id"],
                        "site": site,
                        "replay_abs": replay,
                        "restore_abs": abs(restored - baseline),
                    }
                )
            if (
                observer.calls > manifest["maximum_calls"]
                or time.monotonic() - start > manifest["maximum_seconds"]
            ):
                raise RuntimeError("Binding intervention budget exceeded")
        pd.DataFrame(rows).to_csv(out / "interventions.csv", index=False)
        if (ti + 1) % 4 == 0:
            print(f"PATCHED {ti + 1}/{len(targets)} calls={observer.calls}", flush=True)
    pd.DataFrame(controls).to_csv(out / "instrument_controls.csv", index=False)
    self_max = max(abs(r["delta"]) for r in rows if r["arm"] == "self")
    replay_max = max(r["replay_abs"] for r in controls)
    restore_max = max(r["restore_abs"] for r in controls)
    passed = max(self_max, replay_max, restore_max) == 0
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE" if passed else "FAILED_INVARIANCE",
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
            "rows": len(rows),
            "self_max_abs": self_max,
            "replay_max_abs": replay_max,
            "restore_max_abs": restore_max,
        },
    )
    if not passed:
        raise RuntimeError("Intervention controls failed")


if __name__ == "__main__":
    main()
