#!/usr/bin/env python3
"""Bounded VM04 token-aligned natural-history activation interventions."""

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
    matched_random_rows,
    patch_message,
    project_rows,
)
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json

SITES = ("block_10", "block_16")


def main():
    """Exchange paired model states, never author identities or psychological labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=1800)
    a = p.parse_args()
    cases = json.loads((a.materials / "cases.json").read_text(encoding="utf-8"))
    if len(cases) != 12 or len({c["author_id"] for c in cases}) != 6:
        raise ValueError("Expected the frozen 12-current, six-discovery-author pilot")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    torch.manual_seed(731)
    torch.set_num_threads(4)
    observer = DecoderObserver.load(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        device="cuda",
        dtype="bfloat16",
        max_tokens=2048,
        pad_to_length=2048,
        local_files_only=True,
    )
    jobs = []
    for c in cases:
        for rule in ("other_condition", "other_length"):
            donor = c["donors"][rule]
            if donor["status"] != "MATCHED":
                raise ValueError("Missing frozen matched donor")
            for task in ("disposition", "current_guidance"):
                prompts = []
                for history in (c["history"], donor["history"]):
                    user, interval = patch_message(
                        render_records(history), c["current"]["body"], task
                    )
                    ids = observer.prompt_ids(SYSTEM, user)
                    positions = observer.content_intervals(SYSTEM, user, [interval])
                    prompts.append({"ids": ids, "positions": positions})
                left, right = prompts
                if [left["ids"][i] for i in left["positions"]] != [
                    right["ids"][i] for i in right["positions"]
                ]:
                    raise ValueError(
                        "Current token identities differ between histories"
                    )
                jobs.append(
                    {
                        "case_id": c["case_id"],
                        "author_id": c["author_id"],
                        "donor_id": donor["donor_id"],
                        "rule": rule,
                        "task": task,
                        "prompts": prompts,
                    }
                )
    maximum = max(len(p["ids"]) for j in jobs for p in j["prompts"])
    # Fix one execution shape before any forward; never adapt it by outcome.
    observer.pad_to_length = 1536 if maximum + 16 <= 1536 else 2048
    manifest = {
        "protocol": "GRAIN_VM04_NATURAL_PATCH_V1",
        "material_sha256": material_hash(cases),
        "code_sha256": code_fingerprint(ROOT),
        "observer": observer.manifest(),
        "system": SYSTEM,
        "sites": list(SITES),
        "ranks": [2, 4],
        "alphas": [0.25, 0.5, 1.0],
        "spans": ["current_token_span", "decision_endpoint"],
        "jobs": [
            {k: v for k, v in j.items() if k != "prompts"}
            | {
                "prompt_lengths": [len(p["ids"]) for p in j["prompts"]],
                "token_hashes": [material_hash(p["ids"]) for p in j["prompts"]],
                "current_token_count": len(j["prompts"][0]["positions"]),
            }
            for j in jobs
        ],
        "max_calls": 4096,
        "max_seconds": a.max_seconds,
        "labels": "none",
        "split": "discovery_only",
        "basis_scope": "per-current and donor pair; not a shared author or personality subspace",
        "random_control": "random hidden subspace; candidate L2 matched per token before BF16",
        "primary": "absolute and donor-oriented Yes-minus-No logit changes",
        "ratio_rule": "Only descriptive, when absolute natural donor contrast >= 1 logit",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    cache = out / "private_cache"
    cache.mkdir()
    candidates = [
        observer.tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
    ]
    if any(len(x) != 1 for x in candidates):
        raise ValueError("Need single-token Yes/No scoring")
    yes, no = candidates[0][0], candidates[1][0]
    margin = lambda logits: float(logits[yes] - logits[no])
    rows, controls, baselines, geometry = [], [], [], []
    prefix_reference, basis_cache = {}, {}
    start = time.monotonic()
    status = "RUNNING"
    write_json(out / "run_status.json", {"status": status, "completed_jobs": 0})
    try:
        with threadpool_limits(limits=4):
            for ji, job in enumerate(jobs):
                if (
                    time.monotonic() - start > a.max_seconds
                    or observer.calls + 55 > 4096
                ):
                    status = "PARTIAL_BUDGET"
                    break
                own, other = job["prompts"]
                states, margins = [], []
                prefix_error = 0.0
                for arm_index, prompt in enumerate((own, other)):
                    pos = prompt["positions"] + [len(prompt["ids"]) - 1]
                    captured, logits = observer.capture(prompt["ids"], positions=pos)
                    retained = {s: captured[s] for s in SITES}
                    states.append(retained)
                    margins.append(margin(logits))
                    for s in SITES:
                        key = (job["case_id"], job["rule"], arm_index, s)
                        prefix = retained[s][:-1]
                        if key in prefix_reference:
                            prefix_error = max(
                                prefix_error,
                                float(np.max(np.abs(prefix - prefix_reference[key]))),
                            )
                        else:
                            prefix_reference[key] = prefix.copy()
                    np.savez_compressed(
                        cache / f"j{ji:02d}_a{arm_index}.npz",
                        **retained,
                        current_positions=np.asarray(prompt["positions"], dtype=int),
                        decision_position=np.asarray([len(prompt["ids"]) - 1]),
                    )
                if prefix_error != 0:
                    raise RuntimeError("Causal evidence-prefix invariance failed")
                base, natural = margins
                meta = {k: v for k, v in job.items() if k != "prompts"}
                baselines.append(
                    meta
                    | {
                        "baseline_margin": base,
                        "natural_donor_margin": natural,
                        "natural_delta": natural - base,
                        "own_tokens": len(own["ids"]),
                        "donor_tokens": len(other["ids"]),
                        "prefix_max_error": prefix_error,
                    }
                )
                for site in SITES:
                    current_delta = (
                        states[1][site][:-1].astype(float) - states[0][site][:-1]
                    )
                    basis_key = (job["case_id"], job["rule"], site)
                    if basis_key not in basis_cache:
                        basis, energy4 = current_difference_basis(current_delta, 4)
                        energy2 = float(
                            np.linalg.norm(project_rows(current_delta, basis[:, :2]))
                            ** 2
                            / np.linalg.norm(current_delta) ** 2
                        )
                        basis_cache[basis_key] = (basis, energy2, energy4)
                        geometry.append(
                            {k: v for k, v in meta.items() if k != "task"}
                            | {
                                "site": site,
                                "rank2_energy": energy2,
                                "rank4_energy": energy4,
                                "current_difference_norm": float(
                                    np.linalg.norm(current_delta)
                                ),
                            }
                        )
                    basis, _, _ = basis_cache[basis_key]
                    for span, selector, positions in (
                        ("current_token_span", slice(None, -1), own["positions"]),
                        ("decision_endpoint", slice(-1, None), [len(own["ids"]) - 1]),
                    ):
                        h, donor = (
                            states[0][site][selector].astype(float),
                            states[1][site][selector].astype(float),
                        )
                        delta = donor - h
                        arms = [
                            ("full", 0, 1.0, donor),
                            ("self", 0, 1.0, h),
                            ("sham", 0, 0.0, h),
                        ]
                        for rank in (2, 4):
                            projected = project_rows(delta, basis[:, :rank])
                            for alpha in (0.25, 0.5, 1.0):
                                arms.append(
                                    ("candidate", rank, alpha, h + alpha * projected)
                                )
                            for rep in range(2):
                                effect = matched_random_rows(
                                    delta,
                                    projected,
                                    rank,
                                    [
                                        job["case_id"],
                                        job["rule"],
                                        site,
                                        span,
                                        rank,
                                        rep,
                                    ],
                                )
                                arms.append((f"random_{rep}", rank, 1.0, h + effect))
                        for arm, rank, alpha, value in arms:
                            actual = (
                                torch.as_tensor(value)
                                .to(torch.bfloat16)
                                .float()
                                .numpy()
                                .astype(float)
                            )
                            _, output = observer.capture(
                                own["ids"], patch=Patch(site, positions, value)
                            )
                            score = margin(output)
                            change = score - base
                            rows.append(
                                meta
                                | {
                                    "site": site,
                                    "span": span,
                                    "arm": arm,
                                    "rank": rank,
                                    "alpha": alpha,
                                    "baseline_margin": base,
                                    "natural_donor_margin": natural,
                                    "patched_margin": score,
                                    "delta": change,
                                    "oriented_delta": change * np.sign(natural - base),
                                    "recovery_ratio": change / (natural - base)
                                    if abs(natural - base) >= 1
                                    else None,
                                    "requested_effect_norm": float(
                                        np.linalg.norm(value - h)
                                    ),
                                    "actual_effect_norm": float(
                                        np.linalg.norm(actual - h)
                                    ),
                                    "cast_error_norm": float(
                                        np.linalg.norm(actual - value)
                                    ),
                                    "relative_effect_norm": float(
                                        np.linalg.norm(actual - h)
                                        / max(np.linalg.norm(h), 1e-12)
                                    ),
                                }
                            )
                _, restored = observer.capture(own["ids"])
                error = abs(margin(restored) - base)
                controls.append(meta | {"restore_abs": error})
                if error != 0 or any(
                    abs(r["delta"]) != 0 for r in rows if r["arm"] in ("self", "sham")
                ):
                    raise RuntimeError("Self/sham/restoration control failed")
                for name, data in (
                    ("interventions", rows),
                    ("controls", controls),
                    ("baselines", baselines),
                    ("geometry", geometry),
                ):
                    pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
                write_json(
                    out / "run_status.json",
                    {
                        "status": "RUNNING",
                        "completed_jobs": ji + 1,
                        "calls": observer.calls,
                        "seconds": time.monotonic() - start,
                    },
                )
                print(
                    f"NATURAL_PATCH {ji + 1}/{len(jobs)} calls={observer.calls}",
                    flush=True,
                )
            else:
                status = "COMPLETE"
    except Exception as exc:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_CONTROL_OR_RUNTIME",
                "completed_jobs": len(controls),
                "calls": observer.calls,
                "error": type(exc).__name__ + ": " + str(exc),
            },
        )
        raise
    write_json(
        out / "run_status.json",
        {
            "status": status,
            "completed_jobs": len(controls),
            "planned_jobs": len(jobs),
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
            "input_tokens": observer.input_tokens,
            "processed_tokens": observer.processed_tokens,
            "self_sham_max_abs": max(
                (abs(r["delta"]) for r in rows if r["arm"] in ("self", "sham")),
                default=0,
            ),
            "restore_max_abs": max((r["restore_abs"] for r in controls), default=0),
            "prefix_max_error": max(
                (r["prefix_max_error"] for r in baselines), default=0
            ),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "external_labels_used": False,
        },
    )
    print(
        json.dumps(json.loads((out / "run_status.json").read_text(encoding="utf-8"))),
        flush=True,
    )


if __name__ == "__main__":
    main()
