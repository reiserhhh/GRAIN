#!/usr/bin/env python3
"""Strict-registry history reconstruction and fixed-position decoder instruments."""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .context import QUESTIONS
from .controls import material_hash
from .data_views import stable_key
from .external_material import ARMS, masked_comment
from .inventory import research_source_paths
from .ready_comparison import digest, verified_packet
from .source_history import aligned_triplet, capture_aligned, split_view, tokenize_parts


def archive_instrument(root, output):
    """Include history-specific entrypoints omitted by the legacy archive glob."""
    root = Path(root)
    paths = set(research_source_paths(root))
    for pattern in (
        "scripts/*grain*history*.py",
        "tests/test_grain*history*.py",
        "docs/GRAIN_VM05_HISTORY*.md",
    ):
        paths.update(root.glob(pattern))
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            if p.is_file() and not p.name.startswith("._"):
                z.write(p, p.relative_to(root).as_posix())


def registered_cases(views, material_dir, audit_dir):
    """Bind complete material/audit and return the entire strict label-free registry."""
    m, _ = verified_packet(material_dir)
    am, audit = verified_packet(audit_dir)
    source = json.loads(Path(views).read_text())
    if (
        material_hash(source) != m["common_material_sha256"]
        or m.get("history_envelope") != "body_and_separator"
        or am["material_identity"] != m["identity"]
        or audit["material_identity"] != m["identity"]
        or m["labels_read"]
        or am["labels_read"]
        or audit["labels_read"]
        or m["bridge_outcomes_read"]
        or audit["bridge_outcomes_read"]
    ):
        raise ValueError("Require audited separator-envelope material with no outcomes")
    cases = json.loads((Path(material_dir) / "private_history_cases.json").read_text())
    table = pd.read_csv(
        Path(audit_dir) / "private_strict_history_registry.csv",
        dtype={"author_id": str},
    )
    if (
        table.author_id.duplicated().any()
        or set(table.author_id) != {b["author_id"] for b in cases}
        or not table.strict_no_near_copy.isin([True, False]).all()
        or len({b["author_id"] for b in source}) != len(source)
    ):
        raise ValueError("Invalid strict-registry membership")
    allowed = set(table.loc[table.strict_no_near_copy, "author_id"])
    selected = [b for b in cases if b["author_id"] in allowed]
    by_author = {b["author_id"]: b for b in source}
    if len(selected) != len(allowed) or any(
        b["author_id"] not in by_author
        or b["source_group"] != by_author[b["author_id"]]["source_group"]
        for b in selected
    ):
        raise ValueError("Strict source roles differ")
    return by_author, selected, m, am


def technical_cases(views, material_dir, audit_dir):
    """Select the same two Big5/four MBTI cases without using outcome values."""
    source, cases, m, am = registered_cases(views, material_dir, audit_dir)
    selected = []
    for group, count in (("big5_only", 2), ("mbti_only", 4)):
        eligible = [b for b in cases if b["source_group"] == group]
        if len(eligible) < count:
            raise ValueError("Insufficient source-only technical support")
        selected.extend(
            sorted(
                eligible,
                key=lambda b: stable_key(b["author_id"], "history_instrument_731"),
            )[:count]
        )
    return source, selected, m, am


def material_layouts(case, by_author, tokenizer):
    """Reconstruct all54 layouts and require equality with the frozen token hashes."""
    author = by_author[case["author_id"]]
    if case["source_group"] != author["source_group"] or len(case["views"]) != 3:
        raise ValueError("Changed target role or view count")
    layouts = {}
    for saved in case["views"]:
        vi = saved["view"]
        own, current = split_view(author["views"][vi])
        donor_author = by_author[saved["donor_author_id"]]
        donor, _ = split_view(donor_author["views"][saved["donor_view"]])
        if (
            donor_author["source_group"] != "mbti_only"
            or saved["current_id"] != current["comment_id"]
            or saved["own_history_ids"] != [c["comment_id"] for c in own]
            or saved["donor_history_ids"] != [c["comment_id"] for c in donor]
        ):
            raise ValueError("Changed history/current provenance")
        masks = {
            c["comment_id"]: masked_comment(c["body"], c["comment_id"])[0]
            for c in own + donor + [current]
        }
        saved_by_key = {
            (r["text_arm"], r["task"], r["history_arm"]): r for r in saved["layouts"]
        }
        if len(saved_by_key) != 18 or len(saved["layouts"]) != 18:
            raise ValueError("Incomplete saved layout grid")
        for arm in ARMS:
            for task in QUESTIONS:
                parts = [
                    tokenize_parts(
                        tokenizer,
                        [c | {"body": masks[c["comment_id"]][arm]} for c in h],
                        masks[current["comment_id"]][arm],
                        task,
                        enclose_separator=True,
                    )
                    for h in (own, donor)
                ]
                triplet = aligned_triplet(*parts, tokenizer.pad_token_id or 0)
                for history_arm, row in triplet.items():
                    stored = saved_by_key[arm, task, history_arm]
                    if material_hash(row) != stored["layout_sha256"]:
                        raise ValueError(
                            "Full reconstructed layout differs from commitment"
                        )
                    layouts[vi, arm, task, history_arm] = row
    if len(layouts) != 54:
        raise ValueError("Incomplete technical author")
    return layouts


def observation_error(a, b):
    """Largest absolute state/logit discrepancy with exact shape/key checks."""
    if a[0].keys() != b[0].keys() or a[1].shape != b[1].shape:
        raise ValueError("Observer schema mismatch")
    errors = [float(np.max(np.abs(a[1] - b[1])))]
    for key in a[0]:
        if a[0][key].shape != b[0][key].shape:
            raise ValueError("Observer state shape mismatch")
        errors.append(float(np.max(np.abs(a[0][key] - b[0][key]))))
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite instrument discrepancy")
    return max(errors)


def qualify_triplet(observer, rows):
    """Capture one triplet and require replay, masked-key and native equivalence."""
    if set(rows) != {"own", "matched", "invisible"}:
        raise ValueError("Require complete history triplet")
    if (
        len({tuple(r["current_positions"]) for r in rows.values()}) != 1
        or len({r["current_token_sha256"] for r in rows.values()}) != 1
    ):
        raise ValueError("Current tokens/positions not fixed")
    calls = observer.calls
    captures = {k: capture_aligned(observer, r) for k, r in rows.items()}
    zero = captures["invisible"]
    changed = copy.deepcopy(rows["invisible"])
    s, e = changed["slot"]
    vocab = observer.model.config.vocab_size
    changed["ids"][s:e] = [(i + 1) % vocab for i in changed["ids"][s:e]]
    errors = {
        "masked_token_error": observation_error(
            zero, capture_aligned(observer, changed)
        ),
        "replay_error": observation_error(
            captures["own"], capture_aligned(observer, rows["own"])
        ),
    }
    longest = next(k for k in ("own", "matched") if all(rows[k]["mask"]))
    row = rows[longest]
    native = observer.capture(
        row["ids"],
        positions=[row["decision_position"]],
        pools=[("mean", row["current_positions"]), ("std", row["current_positions"])],
    )
    errors["native_error"] = observation_error(captures[longest], native)
    errors["input_embedding_error"] = max(
        float(
            np.max(
                np.abs(
                    captures[k][0]["input_embedding"]
                    - captures["own"][0]["input_embedding"]
                )
            )
        )
        for k in rows
    )
    if any(v != 0 for v in errors.values()):
        raise ValueError(f"Fixed-shape history instrument is not exact: {errors}")
    return captures, errors | {"forwards": observer.calls - calls}


def tokenizer_matches(tokenizer_dir, manifest):
    """Verify locally staged tokenizer files; never fetch a substitute revision."""
    for name, sha in manifest["qwen_tokenizer_files"].items():
        if digest(Path(tokenizer_dir) / name) != sha:
            raise ValueError("Pinned history tokenizer file changed")
