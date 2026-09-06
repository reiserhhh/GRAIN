#!/usr/bin/env python3
"""Outcome-blind technical diagnosis of rejected native history slot boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.data_views import stable_key
from grain_research.vector_mechanisms.external_material import ARMS, masked_comment
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from grain_research.vector_mechanisms.source_history import (
    aligned_triplet,
    profile,
    select_donor,
    split_view,
    tokenize_parts,
)
from scripts.prepare_grain_source_history import reference_ids
from scripts.run_grain_vector_mechanism_controls import write_json


def probe(target, donors, tokenizer):
    """Compare only boundary bookkeeping; require unchanged native current tokens."""
    whole = [c for v in target["views"] for c in v]
    rows = []
    for vi, view in enumerate(target["views"]):
        own, current = split_view(view)
        d = select_donor(
            profile(own, lambda x: len(tokenizer.encode(x, add_special_tokens=False))),
            current,
            whole,
            donors,
        )["donor"]["history"]
        masked = {
            c["comment_id"]: masked_comment(c["body"], c["comment_id"])[0]
            for c in own + d + [current]
        }
        for arm in ARMS:
            h = [c | {"body": masked[c["comment_id"]][arm]} for c in own]
            other = [c | {"body": masked[c["comment_id"]][arm]} for c in d]
            text = masked[current["comment_id"]][arm]
            for task in ("behavior", "disposition"):
                before = [tokenize_parts(tokenizer, x, text, task) for x in (h, other)]
                after = [
                    tokenize_parts(tokenizer, x, text, task, enclose_separator=True)
                    for x in (h, other)
                ]
                changed = False
                for a, b in zip(before, after):
                    ids1 = [a["suffix"][i] for i in a["current_in_suffix"]]
                    ids2 = [b["suffix"][i] for i in b["current_in_suffix"]]
                    if ids1 != ids2 or a["native_ids"] != b["native_ids"]:
                        changed = True
                if changed:
                    raise ValueError(
                        "Boundary repair changed native text/current tokens"
                    )
                statuses = []
                for pair in (before, after):
                    try:
                        aligned_triplet(*pair, tokenizer.pad_token_id or 0)
                        statuses.append("SUPPORTED")
                    except ValueError as e:
                        statuses.append(str(e))
                rows.append(
                    {
                        "author_id": target["author_id"],
                        "source_group": target["source_group"],
                        "view": vi,
                        "arm": arm,
                        "task": task,
                        "before": statuses[0],
                        "after": statuses[1],
                        "native_ids_changed": changed,
                    }
                )
    return rows


def main():
    """Use64 hashed rejection cases per group; this is not a prediction sample."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "material-dir",
        "views",
        "preflight-dir",
        "qwen-tokenizer",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    m, _ = verified_packet(a.material_dir)
    source = json.loads(a.views.read_text())
    if material_hash(source) != m["common_material_sha256"]:
        raise ValueError("Source changed")
    for name, sha in m["qwen_tokenizer_files"].items():
        if digest(a.qwen_tokenizer / name) != sha:
            raise ValueError("Tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.qwen_tokenizer), local_files_only=True, trust_remote_code=False
    )
    ref = reference_ids(
        source,
        pd.read_csv(a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}),
    )
    donors = []
    for b in source:
        if b["author_id"] not in ref:
            continue
        for vi, view in enumerate(b["views"]):
            try:
                h, _ = split_view(view)
            except ValueError:
                continue
            donors.append(
                {
                    "author_id": b["author_id"],
                    "view": vi,
                    "history": h,
                    "profile": profile(
                        h, lambda s: len(tokenizer.encode(s, add_special_tokens=False))
                    ),
                }
            )
    frame = pd.read_csv(
        a.material_dir / "private_support_by_author.csv", dtype={"author_id": str}
    )
    selected = []
    for group, g in frame.loc[
        frame.reason.eq("Current/prefix/suffix tokens differ between histories")
    ].groupby("source_group"):
        selected += sorted(
            g.author_id,
            key=lambda uid: stable_key(uid, "history_boundary_diagnostic_731"),
        )[:64]
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private output required")
    out.mkdir(parents=True, exist_ok=False)
    archive_source(ROOT, out / "source_snapshot.zip")
    fields = {
        "protocol": "VM05_HISTORY_BOUNDARY_DIAGNOSIS_V1",
        "material_identity": m["identity"],
        "sample_rule": "64 hashed boundary rejections per group",
        "authors": len(selected),
        "selected_commitment": material_hash(selected),
        "source_sha256": digest(Path(__file__)),
        "operator_sha256": digest(
            ROOT / "grain_research/vector_mechanisms/source_history.py"
        ),
        "labels_read": False,
        "model_forwards": 0,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    by_id = {b["author_id"]: b for b in source}
    rows = []
    for uid in selected:
        rows.extend(probe(by_id[uid], donors, tokenizer))
    f = pd.DataFrame(rows)
    f.to_csv(out / "private_boundary_rows.csv", index=False)
    f.groupby(["source_group", "before", "after"]).size().rename(
        "layouts"
    ).reset_index().to_csv(out / "boundary_summary.csv", index=False)
    supported = f.assign(ok=f.after.eq("SUPPORTED")).groupby("author_id").ok.all()
    s = {
        "status": "COMPLETE",
        "parent_identity": fields["identity"],
        "authors": len(selected),
        "layouts": len(f),
        "fully_recovered_authors": int(supported.sum()),
        "before_failed_layouts": int((f.before != "SUPPORTED").sum()),
        "after_failed_layouts": int((f.after != "SUPPORTED").sum()),
        "native_token_changes": int(f.native_ids_changed.sum()),
        "labels_read": False,
        "model_forwards": 0,
        "outputs_sha256": {
            name: digest(out / name)
            for name in ("private_boundary_rows.csv", "boundary_summary.csv")
        },
    }
    write_json(out / "summary.json", s)
    print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
