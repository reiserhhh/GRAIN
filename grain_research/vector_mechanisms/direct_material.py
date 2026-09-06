#!/usr/bin/env python3
"""Verified common-material registry and exact direct-prompt reconstruction."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

from .controls import material_hash
from .data_views import stable_key
from .direct_traits import (
    AXES,
    BIG5,
    SYSTEM,
    question,
    tasks_for_group,
    tokenize_prompt,
)
from .history_instrument import archive_instrument
from .ready_comparison import digest, verified_packet


def archive_direct(root, output):
    """Preserve actual direct/history helper dependencies without private datasets."""
    root = Path(root)
    archive_instrument(root, output)
    with zipfile.ZipFile(output, "a", zipfile.ZIP_DEFLATED) as z:
        saved = set(z.namelist())
        for pattern in (
            "scripts/*grain*direct*traits*.py",
            "tests/test_grain_direct*.py",
            "docs/GRAIN_VM05_DIRECT*.md",
        ):
            for path in sorted(root.glob(pattern)):
                name = path.relative_to(root).as_posix()
                if (
                    path.is_file()
                    and not path.name.startswith("._")
                    and name not in saved
                ):
                    z.write(path, name)
                    saved.add(name)


def load_registration(views, material_dir):
    """Check full source/role/task support independently of a green completion flag."""
    m, s = verified_packet(material_dir)
    source = json.loads(Path(views).read_text(encoding="utf-8"))
    cases = json.loads(
        (Path(material_dir) / "private_direct_cases.json").read_text(encoding="utf-8")
    )
    by_author = {b["author_id"]: b for b in source}
    if (
        m["protocol"] != "VM05_DIRECT_TRAIT_MATERIAL_V1"
        or material_hash(source) != m["common_material_sha256"]
        or len(source) != len(by_author)
        or len(cases) != len(source)
        or m["authors"] != len(source)
        or s["authors"] != len(source)
        or m["system"] != SYSTEM
        or m["questions"] != {t: question(t) for t in (*BIG5, *AXES)}
        or m["labels_read"]
        or m["bridge_outcomes_read"]
        or s["labels_read"]
        or s["bridge_outcomes_read"]
        or m["max_tokens"] != 3072
        or m["margin_tokens"] != 16
        or m["neural_forwards"]
        or s["neural_forwards"]
        or [c["author_id"] for c in cases] != [b["author_id"] for b in source]
    ):
        raise ValueError("Direct material/source/prompt identity mismatch")
    logical = 0
    for c in cases:
        b = by_author[c["author_id"]]
        tasks = tasks_for_group(b["source_group"])
        if c["source_group"] != b["source_group"] or tuple(c["tasks"]) != tasks:
            raise ValueError("Changed direct source role/task set")
        expected = {
            (a, v, t)
            for a in ("raw", "cue_mask", "placebo_mask")
            for v in range(3)
            for t in tasks
        }
        rows = c["layouts"]
        if (
            len(rows) != len(expected)
            or {(r["arm"], r["view"], r["task"]) for r in rows} != expected
            or not isinstance(c["supported"], bool)
            or any(not isinstance(r["supported"], bool) for r in rows)
            or c["supported"] != all(r["supported"] for r in rows)
        ):
            raise ValueError("Incomplete direct support grid")
        logical += len(rows)
    supported = [c for c in cases if c["supported"]]
    if (
        logical != s["logical_prompts"]
        or s["supported_groups"] != dict(Counter(c["source_group"] for c in supported))
        or s["groups"] != dict(Counter(c["source_group"] for c in cases))
        or s["unsupported_authors"] != len(cases) - len(supported)
    ):
        raise ValueError("Direct logical prompt accounting differs")
    return by_author, supported, m


def verify_tokenizer(folder, material):
    """Match the direct material schema without mutating a history certificate."""
    folder = Path(folder).resolve()
    expected = material["tokenizer_sha256"]
    if not expected:
        raise ValueError("Missing direct native tokenizer certificate")
    for name, sha in expected.items():
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file() or digest(path) != sha:
            raise ValueError("Direct tokenizer checksum/path mismatch")


def technical_cases(cases):
    """Fixed2 Big5/4 MBTI source cases, selected without psychological outcomes."""
    chosen = []
    for group, n in (("big5_only", 2), ("mbti_only", 4)):
        eligible = [c for c in cases if c["source_group"] == group]
        if len(eligible) < n:
            raise ValueError("Insufficient direct source technical material")
        chosen.extend(
            sorted(
                eligible,
                key=lambda c: stable_key(c["author_id"], "vm05_direct_technical_731"),
            )[:n]
        )
    return chosen


def verify_prompts(case, arms, tokenizer):
    """Every actual native prompt must reproduce its previously committed input."""
    records = []
    for row in case["layouts"]:
        texts = arms[row["arm"]][row["view"] * 12 : (row["view"] + 1) * 12]
        ids, _, _ = tokenize_prompt(texts, row["task"], tokenizer)
        if (
            not row["supported"]
            or len(ids) != row["prompt_tokens"]
            or material_hash(ids) != row["prompt_sha256"]
        ):
            raise ValueError("Actual direct prompt differs from registered material")
        records.append(
            {
                "arm": row["arm"],
                "view": row["view"],
                "task": row["task"],
                "prompt_tokens": len(ids),
                "prompt_sha256": material_hash(ids),
            }
        )
    return records
