#!/usr/bin/env python3
"""Complete outcome-blind common-material support for direct trait readouts."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import validate_multicomment_bundle
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import (
    AXES,
    BIG5,
    SYSTEM,
    candidates,
    question,
    tasks_for_group,
    tokenize_prompt,
)
from grain_research.vector_mechanisms.history_instrument import (
    archive_instrument,
    tokenizer_matches,
)
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CODE = (
    "scripts/prepare_grain_direct_traits.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/cohort.py",
    "grain_research/vector_mechanisms/external_material.py",
    "scripts/run_grain_vector_mechanism_external_features.py",
    "grain_research/vector_mechanisms/controls.py",
)


def run(a):
    """Register every author/task/view/arm or an explicit whole-author budget failure."""
    if not 60 <= a.max_seconds <= 3600:
        raise ValueError("Direct support pass must be bounded")
    started = time.monotonic()
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads(
        (a.preflight_dir / "common_material_manifest.json").read_text(encoding="utf-8")
    )
    if material_hash(source) != common["material_sha256"] or len(
        {b["author_id"] for b in source}
    ) != len(source):
        raise ValueError("Changed original common author material")
    token_parent = json.loads(a.tokenizer_manifest.read_text(encoding="utf-8"))
    if (
        material_hash({k: v for k, v in token_parent.items() if k != "identity"})
        != token_parent["identity"]
        or token_parent["common_material_sha256"] != common["material_sha256"]
        or token_parent["qwen_revision"] != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    ):
        raise ValueError("Require pinned same-common-material tokenizer record")
    tokenizer_matches(a.tokenizer_dir, token_parent)
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    tasks = (*BIG5, *AXES)
    alphabets = {t: candidates(t, tokenizer) for t in tasks}
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private direct-trait outputs belong under results")
    out.mkdir(parents=True, exist_ok=False)
    archive_instrument(ROOT, out / "source_snapshot.zip")
    with zipfile.ZipFile(out / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED) as z:
        z.write(__file__, Path(__file__).relative_to(ROOT).as_posix())
        for name in (
            "docs/GRAIN_VM05_DIRECT_TRAIT_CONTROL_V1.md",
            "tests/test_grain_direct_traits.py",
        ):
            if (ROOT / name).exists():
                z.write(ROOT / name, name)
    fields = {
        "protocol": "VM05_DIRECT_TRAIT_MATERIAL_V1",
        "common_material_sha256": common["material_sha256"],
        "authors": len(source),
        "system": SYSTEM,
        "questions": {t: question(t) for t in tasks},
        "alphabets": alphabets,
        "max_tokens": 3072,
        "margin_tokens": 16,
        "qwen_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "tokenizer_reference_identity": token_parent["identity"],
        "tokenizer_reference_used_for": "Token file identity only, not history-subset selection",
        "tokenizer_sha256": {
            p.name: digest(p)
            for p in sorted(a.tokenizer_dir.iterdir())
            if p.is_file() and not p.name.startswith("._")
        },
        "code_sha256": {name: digest(ROOT / name) for name in CODE},
        "source_snapshot_sha256": digest(out / "source_snapshot.zip"),
        "labels_read": False,
        "bridge_outcomes_read": False,
        "neural_forwards": 0,
        "selection": "Complete3view x3arm x every applicable source task; whole-author budget support",
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    cases, support = [], []
    logical = unique = 0
    try:
        for i, b in enumerate(source):
            if time.monotonic() - started > a.max_seconds:
                raise TimeoutError(
                    "Preserve partial direct-trait material; no completion"
                )
            validate_multicomment_bundle(b, units=12)
            applicable = tasks_for_group(b["source_group"])
            arms = bundle_arms(b)
            rows, cache = [], {}
            supported = True
            for arm, texts in arms.items():
                for view in range(3):
                    excerpts = texts[view * 12 : (view + 1) * 12]
                    for task in applicable:
                        key = (task, tuple(excerpts))
                        if key not in cache:
                            try:
                                ids, _, _ = tokenize_prompt(excerpts, task, tokenizer)
                                cache[key] = {
                                    "supported": True,
                                    "prompt_tokens": len(ids),
                                    "prompt_sha256": material_hash(ids),
                                }
                            except ValueError as e:
                                if not str(e).startswith(
                                    "Direct trait prompt exceeds fixed budget"
                                ):
                                    raise
                                cache[key] = {"supported": False, "reason": str(e)}
                        row = {"arm": arm, "view": view, "task": task, **cache[key]}
                        rows.append(row)
                        supported &= row["supported"]
            logical += len(rows)
            unique += len(cache)
            record = {
                "author_id": b["author_id"],
                "source_group": b["source_group"],
                "supported": supported,
                "tasks": list(applicable),
                "layouts": rows,
            }
            cases.append(record)
            support.append(
                {k: record[k] for k in ("author_id", "source_group", "supported")}
                | {
                    "logical_prompts": len(rows),
                    "unique_prompts": len(cache),
                    "over_budget_prompts": sum(not r["supported"] for r in rows),
                }
            )
            if (i + 1) % 100 == 0:
                write_json(
                    out / "run_status.json",
                    {
                        "status": "RUNNING",
                        "authors": i + 1,
                        "planned": len(source),
                        "seconds": time.monotonic() - started,
                        "neural_forwards": 0,
                    },
                )
        write_json(out / "private_direct_cases.json", cases)
        pd.DataFrame(support).to_csv(out / "private_support.csv", index=False)
        summary = {
            "status": "COMPLETE",
            "parent_identity": fields["identity"],
            "authors": len(cases),
            "logical_prompts": logical,
            "unique_prompts": unique,
            "groups": dict(Counter(c["source_group"] for c in cases)),
            "supported_groups": dict(
                Counter(c["source_group"] for c in cases if c["supported"])
            ),
            "unsupported_authors": sum(not c["supported"] for c in cases),
            "seconds": time.monotonic() - started,
            "neural_forwards": 0,
            "labels_read": False,
            "bridge_outcomes_read": False,
            "outputs_sha256": {
                p.name: digest(p)
                for p in out.iterdir()
                if p.name in ("private_direct_cases.json", "private_support.csv")
            },
        }
        write_json(out / "summary.json", summary)
        write_json(out / "run_status.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException as e:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
                "authors": len(cases),
                "exception_type": type(e).__name__,
                "error": str(e),
                "neural_forwards": 0,
            },
        )
        raise


def main():
    """CPU-only token support; no pretrained weights or outcome tables are loaded."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "views",
        "preflight-dir",
        "tokenizer-dir",
        "tokenizer-manifest",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=1800)
    run(p.parse_args())


if __name__ == "__main__":
    main()
