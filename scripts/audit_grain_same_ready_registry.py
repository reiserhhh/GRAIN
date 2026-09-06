#!/usr/bin/env python3
"""Independently reconstruct same-ready masks and purge memberships without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json


def independent_kfold(n):
    """Reconstruct sklearn's shuffled three-fold membership without its splitter."""
    shuffled = np.random.RandomState(731).permutation(n)
    for part in np.array_split(shuffled, 3):
        valid = np.sort(part)
        mask = np.ones(n, dtype=bool)
        mask[valid] = False
        yield np.flatnonzero(mask), valid


def reconstruct_membership(source, mask, task):
    """Use explicit thread sets and independent split membership, no fitter helpers."""
    group = "big5_only" if task == "big5" else "mbti_only"
    selected = [
        b for i, b in enumerate(source) if mask[i] and b["source_group"] == group
    ]
    bridge = [
        b for i, b in enumerate(source) if mask[i] and b["source_group"] == "bridge"
    ]
    records = []

    def record(train, test, scope, purge):
        held = {c["thread_id"] for b in test for v in b["views"] for c in v}
        clean = (
            [
                b
                for b in train
                if held.isdisjoint({c["thread_id"] for v in b["views"] for c in v})
            ]
            if purge
            else train
        )
        records.append(
            {
                "scope": scope,
                "train_available": len(train),
                "test_authors": len(test),
                "train_retained": len(clean),
                "dense_min20_supported": len(clean) >= 20,
                "train_author_commitment": material_hash(
                    [b["author_id"] for b in clean]
                ),
                "test_author_commitment": material_hash([b["author_id"] for b in test]),
                "labels_read": False,
            }
        )
        return clean

    partitions = [
        (
            f"outer:{f}",
            [b for b in selected if b["official_folds"][task] != f],
            [b for b in selected if b["official_folds"][task] == f],
        )
        for f in range(5)
    ]
    if bridge:
        partitions.append(("bridge_source_fit", selected, bridge))
    for scope, train, test in partitions:
        clean = record(train, test, scope, True)
        for branch, pool in (("retain", train), ("purge", clean)):
            if len(pool) < 3:
                continue
            for j, (ti, vi) in enumerate(independent_kfold(len(pool))):
                record(
                    [pool[i] for i in ti],
                    [pool[i] for i in vi],
                    f"{scope}:{branch}:inner:{j}",
                    branch == "purge",
                )
    return records


def main():
    """Audit actual frozen cohort files, not a manifest assertion of completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry-dir", "profile-dir", "views", "preflight-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private audit output required")
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Fresh audit output required")
    read = lambda p: json.loads(p.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    m, s = read(a.registry_dir / "manifest.json"), read(a.registry_dir / "summary.json")
    if (
        s["status"] != "COMPLETE"
        or s["parent_identity"] != m["identity"]
        or material_hash({k: v for k, v in m.items() if k != "identity"})
        != m["identity"]
        or m["labels_read"]
        or m["bridge_outcomes_read"]
        or m["support_rule_changed"]
    ):
        raise ValueError("Registry is not a sealed label-free result")
    for name, sha in s["outputs_sha256"].items():
        path = (a.registry_dir / name).resolve()
        if not path.is_relative_to(a.registry_dir.resolve()) or digest(path) != sha:
            raise ValueError("Registry file checksum mismatch")
    for name, sha in m["profile_outputs_sha256"].items():
        path = (a.profile_dir / name).resolve()
        if not path.is_relative_to(a.profile_dir.resolve()) or digest(path) != sha:
            raise ValueError("Profile file checksum mismatch")
    source = read(a.views)
    if material_hash(source) != m["material_sha256"]:
        raise ValueError("Different common source")
    budget_path = a.preflight_dir / "budget_by_author.csv"
    if digest(budget_path) != m["budget_sha256"]:
        raise ValueError("Reference membership table changed")
    budget = pd.read_csv(budget_path, dtype={"author_id": str})
    if (
        not budget.old_reference_member.isin([True, False]).all()
        or budget.groupby("author_id").old_reference_member.nunique().max() != 1
    ):
        raise ValueError("Invalid reference membership encoding")
    by_id = budget.groupby("author_id").old_reference_member.first()
    with np.load(
        a.profile_dir / "private_profile_features.npz", allow_pickle=False
    ) as z:
        p = {key: z[key] for key in z.files}
    with np.load(
        a.registry_dir / "private_ready_registry.npz", allow_pickle=False
    ) as z:
        actual = {key: z[key] for key in z.files}
    ids = np.asarray([b["author_id"] for b in source])
    if len(set(ids)) != len(ids) or len(ids) != s["authors"]:
        raise ValueError("Duplicate authors or incorrect total")
    groups = np.asarray([b["source_group"] for b in source])
    ref = np.asarray([bool(by_id[u]) for u in ids])
    if not np.array_equal(p["author_ids"], ids):
        raise ValueError("Profile/source order mismatch")
    raw = np.asarray([x == "GEOMETRY_PROFILE_READY" for x in p["raw_status"]])
    common = np.asarray(
        [
            all(
                p[arm + "_status"][i] == "GEOMETRY_PROFILE_READY"
                for arm in ("raw", "cue_mask", "placebo_mask")
            )
            for i in range(len(ids))
        ]
    )
    masks = {
        "raw_ready": raw,
        "all_arms_ready": common,
        "raw_ready_reference_free": raw & ~ref,
        "all_arms_ready_reference_free": common & ~ref,
    }
    if set(m["cohorts"]) != set(masks) or m["tasks"] != [
        "big5",
        "EI_cont",
        "SN_cont",
        "TF_cont",
        "JP_cont",
    ]:
        raise ValueError("Registry protocol branches changed")
    expected = {
        "author_ids": ids,
        "source_group": groups,
        "reference_member": ref,
        **masks,
    }
    for task in ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont"):
        group = "big5_only" if task == "big5" else "mbti_only"
        expected[task + "_fold"] = np.asarray(
            [
                b["official_folds"][task] if b["source_group"] == group else -1
                for b in source
            ]
        )
    if set(actual) != set(expected) or any(
        actual[k].dtype != v.dtype or not np.array_equal(actual[k], v)
        for k, v in expected.items()
    ):
        raise ValueError("Saved registry differs from independently derived values")
    frame = pd.read_csv(a.registry_dir / "private_membership_feasibility.csv")
    if frame.duplicated(["cohort", "task", "scope"]).any():
        raise ValueError("Duplicate membership scope")
    coverage = pd.read_csv(a.registry_dir / "coverage.csv")
    if len(coverage) != 12:
        raise ValueError("Unexpected coverage branches")
    support = pd.read_csv(a.registry_dir / "training_support.csv")
    if len(support) != 20 or support.duplicated(["cohort", "task"]).any():
        raise ValueError("Unexpected training-support branches")
    checks = 0
    for name, mask in masks.items():
        for group in ("big5_only", "mbti_only", "bridge"):
            sel = mask & (groups == group)
            row = coverage.loc[
                (coverage.cohort == name) & (coverage.source_group == group)
            ]
            if (
                len(row) != 1
                or row.iloc[0].authors != int(sel.sum())
                or row.iloc[0].reference_members != int((sel & ref).sum())
                or row.iloc[0].author_commitment != material_hash(ids[sel].tolist())
            ):
                raise ValueError("Coverage does not match actual eligible authors")
        for task in ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont"):
            rows = reconstruct_membership(source, mask, task)
            reported = support.loc[(support.cohort == name) & (support.task == task)]
            if (
                len(reported) != 1
                or reported.iloc[0].selection_points != len(rows)
                or reported.iloc[0].minimum_train_count
                != min(r["train_retained"] for r in rows)
                or reported.iloc[0].all_min20_supported
                != all(r["dense_min20_supported"] for r in rows)
            ):
                raise ValueError("Training-support summary does not reconstruct")
            observed = frame.loc[
                (frame.cohort == name) & (frame.task == task)
            ].set_index("scope")
            if set(observed.index) != {r["scope"] for r in rows}:
                raise ValueError("Missing or added membership scope")
            for r in rows:
                saved = observed.loc[r["scope"]]
                if any(saved[k] != v for k, v in r.items() if k != "scope"):
                    raise ValueError("Nested author commitment or count differs")
                checks += 1
    if checks != len(frame) or checks != s["membership_records"]:
        raise ValueError("Unexpected membership records")
    if s["under_min20_records"] != int((~frame.dense_min20_supported).sum()):
        raise ValueError("Incorrect thin-cell summary")
    public_coverage = coverage.drop(columns="author_commitment").to_dict(
        orient="records"
    )
    if s["coverage"] != public_coverage:
        raise ValueError("Summary coverage differs from the independently audited rows")
    audit = {
        "status": "COMPLETE",
        "registry_identity": m["identity"],
        "authors": len(ids),
        "array_fields_recomputed": len(expected),
        "membership_records_recomputed": checks,
        "coverage_rows_recomputed": len(coverage),
        "training_support_rows_recomputed": len(support),
        "source_labels_read": False,
        "bridge_outcomes_read": False,
        "model_refits": 0,
        "scope": "Exact eligibility/membership audit, not estimator or class-support acceptance",
    }
    archive_source(ROOT, out / "source_snapshot.zip")
    write_json(out / "summary.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
