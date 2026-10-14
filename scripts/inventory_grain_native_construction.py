#!/usr/bin/env python3
"""Locate reusable native trait scores on exactly the current construction texts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import BIG5, tokenize_prompt
from scripts.analyze_grain_author_discovery import read_native
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import load_texts


def run(out):
    """Verify archived raw prompt IDs and numeric readouts without opening labels."""
    out.mkdir(parents=True, exist_ok=False)
    queue = BASE / "d1_direct_snapshot_47"
    records, hashes, manifest, status = read_native(queue)
    material = json.loads((DEFAULT / "private_material.json").read_text())
    docs = load_texts()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE / "local_model_metadata/qwen_989aa798", trust_remote_code=False
    )
    rows, readouts, checked = [], [], 0
    score_error, mass_error = 0.0, 0.0
    for u, bundle in enumerate(material["bundles"]):
        uid = bundle["author_id"]
        packet = records.get(uid)
        rows.append(
            {
                "author_index": u,
                "author_id": uid,
                "fold": bundle["official_folds"]["big5"],
                "native_available": packet is not None,
            }
        )
        if packet is None:
            continue
        if packet["source_group"] != "big5_only":
            raise ValueError("Wrong native source role")
        for trait in BIG5:
            selected = sorted(
                [
                    r
                    for r in packet["readouts"]
                    if r["arm"] == "raw" and r["task"] == trait
                ],
                key=lambda r: r["view"],
            )
            if [r["view"] for r in selected] != [0, 1, 2]:
                raise ValueError("Incomplete native views")
            for row in selected:
                texts = [c["body"] for c in bundle["views"][row["view"]]]
                ids, _, _ = tokenize_prompt(texts, trait, tokenizer)
                if (
                    material_hash(ids) != row["prompt_sha256"]
                    or len(ids) != row["prompt_tokens"]
                ):
                    raise ValueError("Native prompt differs from current material")
                logits = np.asarray(row["candidate_logits"], float)
                probs = np.exp(logits - logsumexp(logits))
                score = float(probs @ (np.arange(10) / 9))
                mass = float(
                    np.exp(logsumexp(logits) - row["full_vocab_log_normalizer"])
                )
                score_error = max(score_error, abs(score - row["conditional_score"]))
                mass_error = max(mass_error, abs(mass - row["candidate_mass"]))
                checked += 1
                readouts.append(
                    {
                        "author_index": u,
                        "trait": trait,
                        "view": row["view"],
                        "score": score,
                        "candidate_mass": mass,
                        "greedy_first_token_is_candidate": row[
                            "greedy_first_token_is_candidate"
                        ],
                    }
                )
            np.testing.assert_allclose(
                np.mean([r["conditional_score"] for r in selected]),
                packet["view_means"][f"raw__{trait}"],
                atol=1e-12,
            )
    assert score_error < 1e-12 and mass_error < 1e-12
    frame = pd.DataFrame(rows)
    values = pd.DataFrame(readouts)
    frame.to_csv(out / "private_coverage.csv", index=False)
    values.to_csv(out / "private_native_readouts.csv", index=False)
    counts = frame.groupby("fold").agg(
        total=("native_available", "size"), cached=("native_available", "sum")
    )
    counts["missing"] = counts.total - counts.cached
    counts.to_csv(out / "coverage_by_fold.csv")
    write(
        out / "manifest.json",
        {
            "queue_manifest": digest(queue / "manifest.json"),
            "queue_identity": manifest["identity"],
            "completed_shards": len(status["completed"]),
            "material_manifest": digest(DEFAULT / "material_manifest.json"),
            "code": digest(Path(__file__)),
            "native_hashes": hashes,
            "labels_read": False,
            "new_inference": False,
            "new_fits": False,
        },
    )
    result = {
        "status": "COVERAGE_VERIFIED",
        "pandora_authors": len(frame),
        "pandora_cached": int(frame.native_available.sum()),
        "pandora_missing": int((~frame.native_available).sum()),
        "held_cached": int(frame[frame.fold.eq(0)].native_available.sum()),
        "essays_authors": len(docs["essays"]["ids"]),
        "essays_cached_in_this_snapshot": 0,
        "native_prompts_replayed": checked,
        "score_max_error": score_error,
        "candidate_mass_max_error": mass_error,
        "mean_candidate_mass": float(values.candidate_mass.mean()),
        "greedy_first_token_candidate_rate": float(
            values.greedy_first_token_is_candidate.mean()
        ),
        "role": "exact 3x12-comment native cache inventory, not completed all-author Study B",
        "next": "Reuse matching archives; do not revive the old 128-shard queue merely to fill unrelated roles/arms",
    }
    write(out / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=BASE / "all_plan_closure_v1/native_construction_inventory_r1",
    )
    run(p.parse_args().output)
