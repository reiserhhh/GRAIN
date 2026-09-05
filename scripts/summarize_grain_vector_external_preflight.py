#!/usr/bin/env python3
"""Independently qualify VM05 material and cross-host inputs before external labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import QUESTIONS, SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import (
    CUE_PATTERN,
    chat_ids,
    pack_features,
    unpack_features,
)
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_baselines import MODELS
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_features import bundle_arms
from scripts.run_grain_vector_mechanism_multicomment_qwen import render_evidence


def main():
    """Verify token identity, exact local cache hashes, finite features, and masks."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--technical-root", type=Path, required=True)
    p.add_argument("--qwen-tokenizer", type=Path, required=True)
    p.add_argument("--e5-model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    common = read(a.preflight_dir / "common_material_manifest.json")
    parent = read(a.parent_dir / "author_views.json")
    selected = [b for b in parent if b["author_id"] in set(common["author_ids"])]
    if material_hash(selected) != common["material_sha256"]:
        raise ValueError("Common material identity changed")
    tech = read(a.preflight_dir / "technical_views.json")
    tokenizers = {
        "qwen": AutoTokenizer.from_pretrained(
            str(a.qwen_tokenizer), local_files_only=True
        ),
        **{
            name: AutoTokenizer.from_pretrained(m, revision=rev, local_files_only=True)
            for name, (m, rev, _prefix, _pool) in MODELS.items()
        },
    }
    technical = []
    for stage, dirname in [
        ("qwen", "vm05_technical_qwen_r1"),
        ("bge_large", "vm05_technical_bge_r1"),
        ("e5_large", "vm05_technical_e5_r1"),
    ]:
        folder = a.technical_root / dirname
        status, manifest = (
            read(folder / "run_status.json"),
            read(folder / "manifest.json"),
        )
        if (
            status["status"] != "COMPLETE"
            or status["authors"] != len(tech)
            or manifest["source_material_sha256"] != material_hash(tech)
            or manifest["common_parent_identity"] != common["parent_identity"]
        ):
            raise ValueError("Incomplete or different technical material")
        if (
            material_hash({k: v for k, v in manifest.items() if k != "identity"})
            != manifest["identity"]
        ):
            raise ValueError("Technical manifest commitment mismatch")
        comparisons, cache_bytes, packed_bytes = 0, 0, 0
        replay_max, prefix_max = 0.0, 0.0
        for b in tech:
            f = folder / "private_cache" / (b["author_id"] + ".npz")
            meta = read(f.with_suffix(".json"))
            if hashlib.sha256(f.read_bytes()).hexdigest() != meta["feature_sha256"]:
                raise ValueError("Transferred feature cache checksum mismatch")
            with np.load(f, allow_pickle=False) as z:
                if any(not np.isfinite(z[k]).all() for k in z.files):
                    raise ValueError("Nonfinite features")
                packed, aliases = pack_features({k: z[k] for k in z.files})
                recovered = unpack_features(packed, aliases)
                if set(recovered) != set(z.files) or any(
                    not np.array_equal(recovered[k], z[k]) for k in z.files
                ):
                    raise ValueError("Lossless storage reconstruction failed")
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **packed)
                packed_bytes += buffer.tell() + len(
                    json.dumps(
                        meta
                        | {
                            "feature_aliases": aliases,
                            "cache_format": "lossless_aliases_v1",
                        }
                    ).encode()
                )
            cache_bytes += f.stat().st_size + f.with_suffix(".json").stat().st_size
            replay_max = max(replay_max, meta["replay_max"])
            arms = bundle_arms(b)
            tok = tokenizers[stage]
            if stage == "qwen":
                prefix_max = max(prefix_max, meta["causal_prefix_max"])
                rows = {(r["arm"], r["view"], r["task"]): r for r in meta["readouts"]}
                if len(rows) != 18:
                    raise ValueError("Missing Qwen readouts")
                for arm, ts in arms.items():
                    for v in range(3):
                        view = [{"body": t} for t in ts[v * 12 : (v + 1) * 12]]
                        for task in QUESTIONS:
                            user, _, _ = render_evidence(view, task)
                            ids = chat_ids(tok, SYSTEM, user)
                            record = rows[arm, v, task]
                            if record["input_token_sha256"] != material_hash(
                                ids
                            ) or record["prompt_tokens"] != len(ids):
                                raise ValueError("Mac/Windows Qwen token mismatch")
                            comparisons += 1
            else:
                prefix = MODELS[stage][2]
                for arm, ts in arms.items():
                    sig = material_hash(
                        tok([prefix + t for t in ts], truncation=False)["input_ids"]
                    )
                    if sig != meta["token_signatures"][arm]:
                        raise ValueError("Mac/Windows encoder token mismatch")
                    comparisons += 1
        if replay_max > (0 if stage == "qwen" else 1e-6) or prefix_max != 0:
            raise ValueError("Observer replay/prefix qualification failed")
        e5_verified = None
        if stage == "e5_large":
            local = manifest["model"]["local_file_hashes"]
            for name, expected in local.items():
                if (
                    hashlib.sha256((a.e5_model_dir / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise ValueError("Copied E5 snapshot differs from Mac original")
            e5_verified = list(local)
        technical.append(
            {
                "stage": stage,
                "authors": len(tech),
                "seconds": status["seconds"],
                "gpu_peak_gib": status["gpu_peak_allocated"] / 2**30,
                "replay_max": replay_max,
                "causal_prefix_max": prefix_max,
                "cross_host_token_comparisons": comparisons,
                "cache_bytes": cache_bytes,
                "linear_full_hours_upper_planning": status["seconds"]
                / len(tech)
                * len(selected)
                / 3600,
                "linear_full_cache_gb": cache_bytes / len(tech) * len(selected) / 1e9,
                "lossless_full_cache_gb": packed_bytes
                / len(tech)
                * len(selected)
                / 1e9,
                "copied_e5_files_verified": e5_verified,
                "labels_read": False,
            }
        )
    cue_rows = []
    for population, bundles in [("parent", parent), ("common", selected)]:
        by_group = {}
        for b in bundles:
            counter = by_group.setdefault(b["source_group"], Counter())
            counter["authors"] += 1
            author_any = False
            for v in b["views"]:
                for c in v:
                    matches = [
                        m.group().lower() for m in CUE_PATTERN.finditer(c["body"])
                    ]
                    marker = sum(x == "type_mention" for x in matches)
                    lexical = len(matches) - marker
                    counter["comments"] += 1
                    counter["cue_comments"] += bool(matches)
                    counter["marker_comments"] += bool(marker)
                    counter["lexical_comments"] += bool(lexical)
                    counter["both_comments"] += bool(marker and lexical)
                    counter["marker_occurrences"] += marker
                    counter["lexical_occurrences"] += lexical
                    author_any |= bool(matches)
            counter["cue_authors"] += author_any
        cue_rows.extend(
            {"population": population, "source_group": g, **c}
            for g, c in by_group.items()
        )
    pd.DataFrame(cue_rows).to_csv(out / "cue_decomposition.csv", index=False)
    budget = pd.read_csv(a.preflight_dir / "budget_by_author.csv")
    budget["in_common"] = budget.author_id.isin(common["author_ids"])
    excludes = []
    for g, frame in budget.groupby("source_group"):
        for field, threshold in [
            ("bge_large_max_tokens", 512),
            ("e5_large_max_tokens", 512),
            ("qwen_max_tokens", 3056),
        ]:
            excludes.append(
                {
                    "source_group": g,
                    "observer": field,
                    "excluded_authors_any_arm": frame.loc[
                        frame[field].gt(threshold), "author_id"
                    ].nunique(),
                }
            )
    pd.DataFrame(excludes).to_csv(out / "budget_exclusion_overlap.csv", index=False)
    support = pd.read_csv(a.preflight_dir / "grain_support_by_author.csv")
    support[support.common_budget].groupby(
        ["source_group", "view", "status"]
    ).size().reset_index(name="authors").to_csv(
        out / "common_grain_support.csv", index=False
    )
    write_json(out / "technical_qualification.json", technical)
    write_json(out / "common_views.json", selected)
    total_cache = sum(r["lossless_full_cache_gb"] for r in technical)
    summary = {
        "status": "COMPLETE",
        "common_authors": len(selected),
        "technical_authors": len(tech),
        "token_comparisons": sum(r["cross_host_token_comparisons"] for r in technical),
        "copied_e5_snapshot_verified": True,
        "all_replay_checks_passed": True,
        "linear_full_gpu_hours_upper_planning": sum(
            r["linear_full_hours_upper_planning"] for r in technical
        ),
        "linear_combined_cache_gb": total_cache,
        "cache_format": "lossless_aliases_v1",
        "lossless_reconstruction_verified": True,
        "cache_10gb_planning_gate": total_cache < 10,
        "parent_identity": common["parent_identity"],
        "material_sha256": common["material_sha256"],
        "labels_read": False,
        "not_personality_outcome": True,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
