#!/usr/bin/env python3
"""Repair VM04 endpoint random controls with post-BF16 norm matching."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.memory import render_records
from grain_research.vector_mechanisms.natural_patch import (
    current_difference_basis,
    match_quantized_row_norms,
    matched_random_rows,
    patch_message,
    project_rows,
)
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Keep pairs/directions/candidates; change random strength, not outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    parent, status = (
        read(a.parent_dir / "manifest.json"),
        read(a.parent_dir / "run_status.json"),
    )
    if status["status"] != "COMPLETE":
        raise ValueError("Parent must be complete")
    cases = read(a.materials / "cases.json")
    if material_hash(cases) != parent["material_sha256"]:
        raise ValueError("Source mismatch")
    by_case = {c["case_id"]: c for c in cases}
    original = pd.read_csv(a.parent_dir / "interventions.csv")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    manifest = {
        "protocol": "GRAIN_VM04_POSTCAST_RANDOM_NORM_REPAIR_V1",
        "parent_identity": parent["identity"],
        "code_sha256": code_fingerprint(ROOT),
        "rule": "same random hidden subspaces; row scaling matches actual BF16 candidate norm",
        "matching_tolerance": 0.01,
        "max_calls": 1024,
        "max_seconds": 600,
        "selection": "all endpoint random rows; no selection by observed effect",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    torch.manual_seed(731)
    torch.set_num_threads(4)
    obs = parent["observer"]
    observer = DecoderObserver.load(
        obs["model"],
        revision=obs["revision"],
        device="cuda",
        dtype="bfloat16",
        max_tokens=obs["max_tokens"],
        pad_to_length=obs["pad_to_length"],
        local_files_only=True,
    )
    yes, no = [
        observer.tokenizer.encode(x, add_special_tokens=False)[0] for x in ("Yes", "No")
    ]
    score = lambda logits: float(logits[yes] - logits[no])
    quantize = lambda x: (
        torch.as_tensor(x).to(torch.bfloat16).float().numpy().astype(float)
    )
    rows, checks = [], []
    start = time.monotonic()
    with threadpool_limits(limits=4):
        for ji, job in enumerate(parent["jobs"]):
            case = by_case[job["case_id"]]
            user, _ = patch_message(
                render_records(case["history"]), case["current"]["body"], job["task"]
            )
            ids = observer.prompt_ids(SYSTEM, user)
            if material_hash(ids) != job["token_hashes"][0]:
                raise ValueError("Prompt changed after parent run")
            own = np.load(a.parent_dir / "private_cache" / f"j{ji:02d}_a0.npz")
            donor = np.load(a.parent_dir / "private_cache" / f"j{ji:02d}_a1.npz")
            subset = original[
                (original.case_id == job["case_id"])
                & (original.rule == job["rule"])
                & (original.task == job["task"])
                & (original.span == "decision_endpoint")
            ]
            base = float(subset.baseline_margin.iloc[0])
            natural = float(subset.natural_donor_margin.iloc[0])
            _, logits = observer.capture(ids)
            replay = abs(score(logits) - base)
            if replay != 0:
                raise RuntimeError("Parent baseline cannot be replayed exactly")
            for site in parent["sites"]:
                h, other = own[site][-1:].astype(float), donor[site][-1:].astype(float)
                delta = other - h
                q, _ = current_difference_basis(
                    donor[site][:-1].astype(float) - own[site][:-1], 4
                )
                for rank in (2, 4):
                    effect = project_rows(delta, q[:, :rank])
                    target = h + effect
                    _, candidate = observer.capture(
                        ids, patch=Patch(site, [len(ids) - 1], target)
                    )
                    expected = subset[
                        (subset.site == site)
                        & (subset.arm == "candidate")
                        & (subset["rank"] == rank)
                        & (subset.alpha == 1)
                    ]
                    candidate_error = abs(
                        score(candidate) - float(expected.patched_margin.iloc[0])
                    )
                    if candidate_error != 0:
                        raise RuntimeError(
                            "Candidate state reconstruction changed output"
                        )
                    for rep in range(2):
                        random = matched_random_rows(
                            delta,
                            effect,
                            rank,
                            [
                                job["case_id"],
                                job["rule"],
                                site,
                                "decision_endpoint",
                                rank,
                                rep,
                            ],
                        )
                        value, scales, errors = match_quantized_row_norms(
                            h, random, target, quantize
                        )
                        _, patched = observer.capture(
                            ids, patch=Patch(site, [len(ids) - 1], value)
                        )
                        patched_score = score(patched)
                        change = patched_score - base
                        meta = {
                            k: job[k]
                            for k in (
                                "case_id",
                                "author_id",
                                "donor_id",
                                "rule",
                                "task",
                            )
                        }
                        rows.append(
                            meta
                            | {
                                "site": site,
                                "span": "decision_endpoint",
                                "arm": f"random_{rep}",
                                "rank": rank,
                                "alpha": 1.0,
                                "baseline_margin": base,
                                "natural_donor_margin": natural,
                                "patched_margin": patched_score,
                                "delta": change,
                                "oriented_delta": change * np.sign(natural - base),
                                "recovery_ratio": change / (natural - base)
                                if abs(natural - base) >= 1
                                else None,
                                "requested_effect_norm": float(
                                    np.linalg.norm(value - h)
                                ),
                                "actual_effect_norm": float(
                                    np.linalg.norm(quantize(value) - h)
                                ),
                                "cast_error_norm": float(
                                    np.linalg.norm(quantize(value) - value)
                                ),
                                "relative_effect_norm": float(
                                    np.linalg.norm(quantize(value) - h)
                                    / max(np.linalg.norm(h), 1e-12)
                                ),
                                "row_scale": float(scales[0]),
                                "postcast_match_relative_error": float(errors[0]),
                                "candidate_replay_error": candidate_error,
                            }
                        )
            _, restored = observer.capture(ids)
            restore = abs(score(restored) - base)
            checks.append(
                {"job": ji, "baseline_replay_error": replay, "restore_error": restore}
            )
            if restore != 0:
                raise RuntimeError("Restoration failed")
            pd.DataFrame(rows).to_csv(out / "replacement_rows.csv", index=False)
            pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
            print(f"NORM_REPAIR {ji + 1}/48 calls={observer.calls}", flush=True)
            if (
                observer.calls > manifest["max_calls"]
                or time.monotonic() - start > manifest["max_seconds"]
            ):
                write_json(
                    out / "run_status.json",
                    {"status": "PARTIAL_BUDGET", "calls": observer.calls},
                )
                return
    max_error = max(r["postcast_match_relative_error"] for r in rows)
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE"
            if max_error <= manifest["matching_tolerance"]
            else "NEEDS_RETRY_NORM",
            "rows": len(rows),
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
            "postcast_norm_max_relative_error": max_error,
            "candidate_replay_max_error": max(
                r["candidate_replay_error"] for r in rows
            ),
            "baseline_replay_max_error": max(
                r["baseline_replay_error"] for r in checks
            ),
            "restore_max_error": max(r["restore_error"] for r in checks),
            "input_tokens": observer.input_tokens,
            "processed_tokens": observer.processed_tokens,
        },
    )
    print(json.dumps(read(out / "run_status.json")), flush=True)


if __name__ == "__main__":
    main()
