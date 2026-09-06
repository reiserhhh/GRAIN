#!/usr/bin/env python3
"""Independently verify history lineage/matching and flag lexical near copies."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.history_overlap import copy_overlap, shingles
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.run_grain_vector_mechanism_controls import write_json


def independent_view(view, tokenizer):
    """Reconstruct chronological material without producer split/profile helpers."""
    if len(view) != 12 or len({c["author_id"] for c in view}) != 1:
        raise ValueError("Wrong source view")
    ordered = sorted(view, key=lambda c: (c["timestamp"], c["comment_id"]))
    texts = {" ".join(c["body"].lower().split()) for c in ordered}
    if len(texts) != 12 or any(
        not c["thread_id"] or not c["comment_id"] for c in ordered
    ):
        raise ValueError("Missing provenance or duplicate body")
    if (
        len({c["thread_id"] for c in ordered}) != 12
        or len({c["comment_id"] for c in ordered}) != 12
    ):
        raise ValueError("Repeated provenance")
    if (
        not all(math.isfinite(c["timestamp"]) for c in ordered)
        or ordered[-2]["timestamp"] >= ordered[-1]["timestamp"]
    ):
        raise ValueError("Wrong time relation")
    h, current = ordered[:-1], ordered[-1]
    text = "\n\n".join(
        "Earlier excerpt " + str(i + 1) + ":\n" + c["body"] for i, c in enumerate(h)
    )
    return (
        h,
        current,
        {
            "tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            "end": h[-1]["timestamp"],
            "subreddits": {c["subreddit"] for c in h},
        },
    )


def independent_match(profile, current, whole_target, candidates):
    """Recompute all candidate costs and deterministic minimum, without producer code."""
    target_threads = {c["thread_id"] for c in whole_target}
    target_ids = {c["comment_id"] for c in whole_target}
    target_text = {" ".join(c["body"].lower().split()) for c in whole_target}
    eligible = []
    for d in candidates:
        if (
            d["author_id"] == current["author_id"]
            or d["profile"]["end"] >= current["timestamp"]
        ):
            continue
        hist = d["history"]
        if (
            target_threads & {c["thread_id"] for c in hist}
            or target_ids & {c["comment_id"] for c in hist}
            or target_text & {" ".join(c["body"].lower().split()) for c in hist}
        ):
            continue
        p = d["profile"]
        t = current["timestamp"]
        length = abs(math.log(p["tokens"] / profile["tokens"]))
        gap = abs(
            math.log1p((t - p["end"]) / 86400)
            - math.log1p((t - profile["end"]) / 86400)
        )
        j = len(profile["subreddits"] & p["subreddits"]) / max(
            1, len(profile["subreddits"] | p["subreddits"])
        )
        cost = length + 0.15 * gap + 0.5 * (1 - j)
        eligible.append((cost, d["author_id"], d["view"], d, length, gap, j))
    if not eligible:
        raise ValueError("Supported case has no eligible prior reference")
    return min(eligible, key=lambda x: x[:3]), len(eligible)


def main():
    """Audit complete supported cases, not a new estimate using psychological values."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "material-dir",
        "views",
        "preflight-dir",
        "qwen-tokenizer",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=1800)
    a = p.parse_args()
    started = time.monotonic()
    if not 60 <= a.max_seconds <= 1800:
        raise ValueError("Require bounded audit runtime")
    m, s = verified_packet(a.material_dir)
    source = json.loads(a.views.read_text())
    if (
        material_hash(source) != m["common_material_sha256"]
        or m["labels_read"]
        or m["bridge_outcomes_read"]
    ):
        raise ValueError("History source identity/target boundary mismatch")
    if digest(a.preflight_dir / "budget_by_author.csv") != m["budget_sha256"]:
        raise ValueError("Changed reference membership table")
    budget = pd.read_csv(
        a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
    )
    reference = set(budget.loc[budget.old_reference_member, "author_id"]) & {
        b["author_id"] for b in source
    }
    if (
        len(reference) != 99
        or material_hash(sorted(reference)) != m["reference_commitment"]
    ):
        raise ValueError("Wrong history reference dictionary")
    for name, sha in m["qwen_tokenizer_files"].items():
        if digest(a.qwen_tokenizer / name) != sha:
            raise ValueError("Changed tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.qwen_tokenizer), local_files_only=True, trust_remote_code=False
    )
    by_author = {b["author_id"]: b for b in source}
    candidates = []
    rejected = []
    for b in source:
        if b["author_id"] not in reference:
            continue
        if b["source_group"] != "mbti_only":
            raise ValueError("Reference dictionary crosses source role")
        for vi, view in enumerate(b["views"]):
            try:
                h, _current, pf = independent_view(view, tokenizer)
            except ValueError:
                rejected.append((b["author_id"], vi))
                continue
            candidates.append(
                {"author_id": b["author_id"], "view": vi, "history": h, "profile": pf}
            )
    if len(candidates) != m["dictionary_views"]:
        raise ValueError("Dictionary support differs")
    stored_reject = json.loads(
        (a.material_dir / "private_rejected_dictionary_views.json").read_text()
    )
    if sorted(rejected) != sorted((r["author_id"], r["view"]) for r in stored_reject):
        raise ValueError("Rejected dictionary view set differs")
    cases = json.loads((a.material_dir / "private_history_cases.json").read_text())
    support = pd.read_csv(
        a.material_dir / "private_support_by_author.csv", dtype={"author_id": str}
    )
    if (
        len(cases) != s["supported_authors"]
        or support.author_id.duplicated().any()
        or set(support.author_id) != (set(by_author) - reference)
        or {r["author_id"] for r in cases}
        != set(support.loc[support.status.eq("SUPPORTED"), "author_id"])
        or len({r["author_id"] for r in cases}) != len(cases)
    ):
        raise ValueError("Supported author union differs")
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Keep private audit in results")
    out.mkdir(parents=True, exist_ok=False)
    archive_source(ROOT, out / "source_snapshot.zip")
    fields = {
        "protocol": "VM05_HISTORY_MATERIAL_AUDIT_V1",
        "material_identity": m["identity"],
        "script_sha256": digest(Path(__file__)),
        "overlap_code_sha256": digest(
            ROOT / "grain_research/vector_mechanisms/history_overlap.py"
        ),
        "near_copy_rule": "word4gram: min_words>=12, shared>=8, J>=.8 or containment>=.9",
        "scope": "all supported lineage/matching; layout commitment completeness, not full independent tokenizer replay",
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    overlap_cache = {}
    flags = []
    checked = 0
    max_error = 0.0
    reuse = Counter()
    rows = []

    def fingerprint(c):
        if c["comment_id"] not in overlap_cache:
            overlap_cache[c["comment_id"]] = shingles(c["body"])
        return overlap_cache[c["comment_id"]]

    for idx, case in enumerate(cases):
        if time.monotonic() - started > a.max_seconds:
            write_json(
                out / "run_status.json",
                {
                    "status": "PARTIAL_TIME_BOUND",
                    "authors": idx,
                    "planned": len(cases),
                    "seconds": time.monotonic() - started,
                },
            )
            raise TimeoutError("Keep partial audit, do not promote unverified cases")
        b = by_author[case["author_id"]]
        if (
            case["author_id"] in reference
            or case["source_group"] != b["source_group"]
            or case["official_folds"] != b["official_folds"]
        ):
            raise ValueError("Source role/fold changed")
        if len(case["views"]) != 3 or [r["view"] for r in case["views"]] != list(
            range(3)
        ):
            raise ValueError("Incomplete three-view author")
        whole = [c for v in b["views"] for c in v]
        flagged = False
        for row in case["views"]:
            vi = row["view"]
            hist, current, pf = independent_view(b["views"][vi], tokenizer)
            winner, n = independent_match(pf, current, whole, candidates)
            cost, uid, dvi, d, length, gap, j = winner
            if (
                row["current_id"] != current["comment_id"]
                or row["own_history_ids"] != [c["comment_id"] for c in hist]
                or row["donor_author_id"] != uid
                or row["donor_view"] != dvi
                or row["donor_history_ids"] != [c["comment_id"] for c in d["history"]]
            ):
                raise ValueError("Saved current/history/donor differs")
            expected = {
                "cost": cost,
                "length_error": length,
                "time_error": gap,
                "subreddit_jaccard": j,
                "eligible_donors": n,
            }
            err = max(abs(row["match"][k] - v) for k, v in expected.items())
            max_error = max(max_error, err)
            if err > 1e-12:
                raise ValueError("Donor matching arithmetic differs")
            layouts = row["layouts"]
            keys = {(r["text_arm"], r["task"], r["history_arm"]) for r in layouts}
            expected_keys = {
                (a, t, h)
                for a in ("raw", "cue_mask", "placebo_mask")
                for t in ("behavior", "disposition")
                for h in ("own", "matched", "invisible")
            }
            if len(layouts) != 18 or keys != expected_keys:
                raise ValueError("Incomplete context-layout grid")
            for arm in ("raw", "cue_mask", "placebo_mask"):
                selected = [r for r in layouts if r["text_arm"] == arm]
                if (
                    len(
                        {
                            (r["current_token_sha256"], r["current_position_sha256"])
                            for r in selected
                        }
                    )
                    != 1
                ):
                    raise ValueError("Current tokens or position commitment changed")
                if any(
                    r["visible_tokens"] > r["prompt_tokens"]
                    or r["prompt_tokens"] + 16 > 3072
                    for r in selected
                ):
                    raise ValueError("Invalid token budget")
            pairs = [("own_current", current, c) for c in hist] + [
                ("matched_target", x, y) for x in whole for y in d["history"]
            ]
            for kind, x, y in pairs:
                overlap = copy_overlap(fingerprint(x), fingerprint(y))
                if overlap["flagged"]:
                    flagged = True
                    flags.append(
                        {
                            "author_id": b["author_id"],
                            "view": vi,
                            "kind": kind,
                            "first_comment_id": x["comment_id"],
                            "second_comment_id": y["comment_id"],
                            **overlap,
                        }
                    )
            checked += 1
            reuse[uid] += 1
        rows.append(
            {
                "author_id": b["author_id"],
                "source_group": b["source_group"],
                "near_copy_flagged": flagged,
                "strict_no_near_copy": not flagged,
            }
        )
        if (idx + 1) % 100 == 0:
            progress = {
                "status": "RUNNING",
                "authors": idx + 1,
                "planned": len(cases),
                "verified_views": checked,
                "seconds": time.monotonic() - started,
            }
            write_json(out / "run_status.json", progress)
            print(json.dumps(progress), flush=True)
    if dict(reuse) != json.loads(
        (a.material_dir / "private_donor_reuse.json").read_text()
    ):
        raise ValueError("Dictionary reuse differs")
    pd.DataFrame(flags).to_csv(out / "private_near_copy_flags.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_strict_history_registry.csv", index=False)
    frame.groupby(["source_group", "near_copy_flagged"]).size().rename(
        "authors"
    ).reset_index().to_csv(out / "near_copy_summary.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": fields["identity"],
        "material_identity": m["identity"],
        "supported_authors_verified": len(cases),
        "matched_views_verified": checked,
        "dictionary_views": len(candidates),
        "matching_max_absolute_error": max_error,
        "layout_commitment_rows_checked": 18 * checked,
        "full_independent_token_layout_replay": False,
        "unsupported_target_reasons_recomputed": False,
        "near_copy_flagged_authors": int(frame.near_copy_flagged.sum()),
        "near_copy_flagged_pairs": len(flags),
        "strict_supported_by_group": frame.loc[frame.strict_no_near_copy]
        .groupby("source_group")
        .size()
        .to_dict(),
        "seconds": time.monotonic() - started,
        "labels_read": False,
        "model_forwards": 0,
        "bridge_outcomes_read": False,
        "outputs_sha256": {
            f.name: digest(f)
            for f in out.iterdir()
            if f.is_file()
            and f.name
            not in ("manifest.json", "run_status.json", "source_snapshot.zip")
            and not f.name.startswith("._")
        },
    }
    write_json(out / "summary.json", summary)
    write_json(out / "run_status.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
