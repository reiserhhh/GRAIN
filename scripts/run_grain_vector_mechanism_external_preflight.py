#!/usr/bin/env python3
"""VM05 label-blind token/support/control preflight on the complete source cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import (
    COHORT_SEED,
    ID_SEED,
    validate_multicomment_bundle,
)
from grain_research.vector_mechanisms.context import QUESTIONS, SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.data_views import stable_key
from grain_research.vector_mechanisms.external_material import (
    ARMS,
    CUE_PATTERN,
    MASK,
    MASK_SEED,
    chat_ids,
    masked_comment,
    purge_counts,
    raw_author_features,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_baselines import MODELS
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_multicomment_qwen import render_evidence
from suica_core.v7_geometry import GeometryBundle, score_geometry_bundle


def token_lengths(tokenizer, texts):
    """Count actual tokens, including special tokens, without truncation."""
    return [
        len(ids)
        for ids in tokenizer(texts, padding=False, truncation=False)["input_ids"]
    ]


def main():
    """Check all source-author materials before selecting common complete inputs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--qwen-tokenizer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    start = time.monotonic()
    read = lambda n: json.loads((a.parent_dir / n).read_text(encoding="utf-8"))
    inventory, source_manifest = read("data_inventory.json"), read("manifest.json")
    if inventory["status"] != "COMPLETE":
        raise ValueError("Incomplete parent cohort")
    bundles = read("author_views.json")
    if material_hash(bundles) != inventory["material_sha256"]:
        raise ValueError("Parent material changed")
    for b in bundles:
        validate_multicomment_bundle(b, units=12)
    e1 = ROOT / "results/v7_multiview_projection/e1_v72_full_20260715"
    runtime_path = e1 / "artifacts/common_source_comment_representation.joblib"
    geometry_path = (
        ROOT / "results/v7_geometry/g1_corrected_v2_full_20260715/geometry_bundle.json"
    )
    geometry = GeometryBundle.from_dict(
        json.loads(geometry_path.read_text(encoding="utf-8"))
    )
    old_features = e1 / "author_features_native.csv"
    if (
        hashlib.sha256(old_features.read_bytes()).hexdigest()
        != geometry.reference_population["cohort_commitment"]
    ):
        raise ValueError("Frozen geometry training table commitment differs")
    old = pd.read_csv(
        old_features, usecols=["user_id", "split"], dtype={"user_id": str}
    )
    reference_ids = {
        stable_key(u, ID_SEED)[:24]
        for u in old.loc[old.split.eq("discovery"), "user_id"]
    }
    old_ids = {stable_key(u, ID_SEED)[:24] for u in old.user_id}
    if len(reference_ids) != geometry.reference_population["n_authors"]:
        raise ValueError("Frozen reference ID count differs")
    runtime = joblib.load(runtime_path)
    qwen = AutoTokenizer.from_pretrained(
        str(a.qwen_tokenizer), local_files_only=True, trust_remote_code=False
    )
    encoders = {
        name: AutoTokenizer.from_pretrained(
            spec[0], revision=spec[1], local_files_only=True, trust_remote_code=False
        )
        for name, spec in MODELS.items()
    }
    manifest = {
        "protocol": "GRAIN_VM05_SAME_EVIDENCE_PREFLIGHT_V1",
        "design_commit": "643fc86",
        "code_sha256": code_fingerprint(ROOT),
        "source_identity": source_manifest["identity"],
        "source_material_sha256": inventory["material_sha256"],
        "authors": len(bundles),
        "models": {
            k: dict(zip(("model", "revision", "prefix", "pool"), v))
            for k, v in MODELS.items()
        },
        "qwen_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "qwen_tokenizer_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(a.qwen_tokenizer.iterdir())
            if p.is_file() and not p.name.startswith("._")
        },
        "qwen_template_sha256": material_hash(qwen.chat_template),
        "runtime_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        "feature_names": geometry.feature_names,
        "arms": ARMS,
        "mask_regex": CUE_PATTERN.pattern,
        "mask_placeholder": MASK,
        "mask_seed": MASK_SEED,
        "budget": {"encoder_max": 512, "qwen_max_prompt": 3056, "qwen_pad": 3072},
        "labels_read": False,
        "max_seconds": 1800,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    n = len(bundles)
    units = np.empty((n, 36, runtime.svd.n_components), dtype=np.float64)
    rows, comments_rows = [], []
    for i, b in enumerate(bundles):
        cs = [c for v in b["views"] for c in v]
        altered, stats = zip(
            *(masked_comment(c["body"], c["comment_id"]) for c in cs), strict=True
        )
        units[i] = runtime.transform([c["body"] for c in cs])
        for c, s in zip(cs, stats, strict=True):
            if s["cue_words"]:
                comments_rows.append(
                    {
                        "author_id": b["author_id"],
                        "source_group": b["source_group"],
                        "comment_id": c["comment_id"],
                        **s,
                    }
                )
        cached_lengths, cached_qwen = {}, {}
        for arm in ARMS:
            ts = [x[arm] for x in altered]
            cache_key = tuple(ts)
            if cache_key not in cached_lengths:
                cached_lengths[cache_key] = {
                    name: token_lengths(tok, [MODELS[name][2] + t for t in ts])
                    for name, tok in encoders.items()
                }
            lens = cached_lengths[cache_key]
            qcounts = []
            for vi in range(3):
                view = [{"body": ts[j]} for j in range(vi * 12, (vi + 1) * 12)]
                for task in QUESTIONS:
                    user, _, _ = render_evidence(view, task)
                    if user not in cached_qwen:
                        cached_qwen[user] = len(chat_ids(qwen, SYSTEM, user))
                    qcounts.append(cached_qwen[user])
            rows.append(
                {
                    "author_id": b["author_id"],
                    "source_group": b["source_group"],
                    "arm": arm,
                    **{k + "_max_tokens": max(v) for k, v in lens.items()},
                    "qwen_max_tokens": max(qcounts),
                    "qwen_total_prompt_tokens": sum(qcounts),
                    "cue_words": sum(s["cue_words"] for s in stats),
                    "old_reference_member": b["author_id"] in reference_ids,
                    "old_e1_member": b["author_id"] in old_ids,
                    "complete_budget": max(qcounts) <= 3056
                    and all(max(v) <= 512 for v in lens.values()),
                }
            )
        if (i + 1) % 100 == 0:
            progress = {
                "status": "RUNNING",
                "authors": i + 1,
                "planned": n,
                "seconds": time.monotonic() - start,
            }
            write_json(out / "run_status.json", progress)
            print(json.dumps(progress), flush=True)
            if progress["seconds"] > 1800:
                write_json(
                    out / "run_status.json", progress | {"status": "PARTIAL_BUDGET"}
                )
                return
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "budget_by_author.csv", index=False)
    pd.DataFrame(comments_rows).to_csv(out / "private_mask_ledger.csv", index=False)
    keep = frame.groupby("author_id").complete_budget.all()
    common = [b for b in bundles if keep[b["author_id"]]]
    tech = []
    for group in ("big5_only", "mbti_only", "bridge"):
        pool = sorted(
            [b for b in common if b["source_group"] == group],
            key=lambda b: stable_key(b["author_id"], COHORT_SEED),
        )
        if len(pool) < 4:
            raise ValueError("Insufficient technical author support")
        tech.extend(pool[:4])
    write_json(out / "technical_views.json", tech)
    write_json(
        out / "common_material_manifest.json",
        {
            "parent_identity": manifest["identity"],
            "author_ids": [b["author_id"] for b in common],
            "material_sha256": material_hash(common),
            "authors": len(common),
            "source_counts": pd.Series([b["source_group"] for b in common])
            .value_counts()
            .to_dict(),
            "complete_all_arms": True,
            "labels_read": False,
        },
    )
    support = []
    saved = {"grain_units_raw": units}
    for label, ix in [("all36", np.arange(36))] + [
        (f"view{v}", np.arange(v * 12, (v + 1) * 12)) for v in range(3)
    ]:
        h = raw_author_features(units[:, ix], geometry.feature_names)
        scored = score_geometry_bundle(geometry, h, unit_counts=np.full(n, len(ix)))
        saved[label + "_raw_features"] = h
        saved[label + "_landmark_profile"] = scored["landmark_distance_profile"]
        saved[label + "_status"] = np.asarray(scored["status"])
        for i, b in enumerate(bundles):
            support.append(
                {
                    "author_id": b["author_id"],
                    "source_group": b["source_group"],
                    "view": label,
                    "status": scored["status"][i],
                    "radius": scored["reference_radius"][i],
                    "common_budget": bool(keep[b["author_id"]]),
                    "old_reference_member": b["author_id"] in reference_ids,
                }
            )
    saved["author_ids"] = np.asarray([b["author_id"] for b in bundles])
    np.savez_compressed(out / "grain_frozen_features.npz", **saved)
    sf = pd.DataFrame(support)
    sf.to_csv(out / "grain_support_by_author.csv", index=False)
    sf.groupby(["source_group", "view", "status"]).size().reset_index(
        name="authors"
    ).to_csv(out / "support_summary.csv", index=False)
    purges = []
    for source, group in [("big5", "big5_only")] + [
        (x + "_cont", "mbti_only") for x in ("EI", "SN", "TF", "JP")
    ]:
        selected = [b for b in common if b["source_group"] == group]
        for fold in range(5):
            train = [b for b in selected if b["official_folds"][source] != fold]
            test = [b for b in selected if b["official_folds"][source] == fold]
            purges.append(
                {"source": source, "outer_fold": fold, **purge_counts(train, test)}
            )
        purges.append(
            {
                "source": source,
                "outer_fold": "bridge",
                **purge_counts(
                    selected, [b for b in common if b["source_group"] == "bridge"]
                ),
            }
        )
    pd.DataFrame(purges).to_csv(out / "thread_purge_feasibility.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "authors": n,
        "common_authors": len(common),
        "source_counts": pd.Series([b["source_group"] for b in common])
        .value_counts()
        .to_dict(),
        "old_reference_overlap": sum(b["author_id"] in reference_ids for b in common),
        "old_e1_overlap": sum(b["author_id"] in old_ids for b in common),
        "technical_authors": len(tech),
        "seconds": time.monotonic() - start,
        "mask_changed_comments": len(comments_rows),
        "mask_words": sum(r["cue_words"] for r in comments_rows),
        "parent_identity": manifest["identity"],
        "labels_read": False,
        "scope": "Full material preflight; no new trait predictions or modified GRAIN gate",
    }
    write_json(out / "summary.json", summary)
    write_json(out / "run_status.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
