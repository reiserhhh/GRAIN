#!/usr/bin/env python3
"""F14-R01 full authorized metadata inventory, actual adapters and joins.

No model calls, text/questionnaire value projection, new source fit or remote I/O.
Mixed CSV files are byte-hashed and parsed with metadata-only column projection.
Only aggregate results leave temporary stores; raw author/event IDs are not saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f14_metadata import (
    MetadataProfile,
    MetadataSpec,
    audit_join,
    inspect_schema,
    normalize_metadata,
    read_projected,
    sha256_file,
)
from grain_research.vector_mechanisms.cohort import ID_SEED
from grain_research.vector_mechanisms.data_views import stable_key

SPECS = {
    "pandora": MetadataSpec(
        "pandora",
        "author",
        ("id",),
        ("subreddit",),
        "created_utc",
        "unix_seconds",
        "lang",
        extras=("parent_id", "link_id"),
        grain="comment",
    ),
    "essays": MetadataSpec(
        "essays",
        "user_id",
        ("text_id",),
        ("source_dataset",),
        extras=("essay_fold", "user_id_source"),
        grain="single_author_document",
    ),
    "x_market": MetadataSpec(
        "x_market",
        "account_id",
        ("post_id",),
        ("symbol", "query_group"),
        "timestamp",
        "iso8601_tz",
        "lang",
        extras=("query_language",),
        grain="collected_post_occurrence",
    ),
    "meps": MetadataSpec(
        "meps",
        "participant_code",
        ("participant_code", "view_kind", "condition", "role", "turn_sequence"),
        ("view_kind", "condition"),
        role="role",
        order="turn_sequence",
        extras=("source_chars", "covered_chars", "windows"),
        grain="speaker_turn",
    ),
}


def cache_author_keys(authors: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Replay the original PANDORA anonymizer, not a guessed ID normalization."""
    result = authors.copy()
    if dataset == "pandora":
        result["author"] = result.author.map(
            lambda value: stable_key(str(value), ID_SEED)[:24]
        )
        if result.author.nunique() != authors.author.nunique():
            raise ValueError("anonymized author collision")
    return result


def write(path: Path, obj) -> None:
    """Save inspectable aggregate/provenance output without NaN values."""
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    """Serialize a fixed-key aggregate table."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def provenance(persona: Path) -> dict:
    """Resolve explicit source roots from registered scripts; no disk-wide scan."""
    data = persona / "data_sets"
    base = ROOT / "results/grain_vector_mechanisms"
    return {
        "pandora": [data / "PANDORA_official/all_comments_since_2015.csv"],
        "essays": [data / "prepared/big5/essays_original_prepared.csv"],
        "x_market": sorted(
            p
            for p in (data / "x_fullmarkettext").rglob("x_posts.csv")
            if not any(v.startswith("._") for v in p.parts)
        ),
        "meps": [base / "vm06_meps_windows_r3/private_turn_integrity.csv"],
    }


def source_chain(persona: Path) -> dict:
    """Record original inputs and the exact transitive locator references."""
    paths = [
        persona / "configs/grain_gptpro_full_coverage_v2.json",
        persona / "docs/GRAIN_GPTPRO_ATOMIC_PROTOCOLS_V3.md",
        ROOT / "results/grain_measurement_discovery_v1/gp05/r1/manifest.json",
        ROOT / "scripts/run_grain_natural_interface_audit.py",
        ROOT / "grain_research/multiview_address.py",
        ROOT
        / "reports/grain_measurement_foundation/GRAIN_ALL_PLAN_LOGIC_FINDINGS_V1.md",
        ROOT / "scripts/scan_grain_natural_antecedents.py",
        ROOT / "scripts/run_grain_local_construction.py",
        ROOT / "scripts/run_grain_local_bilinear.py",
        ROOT / "scripts/run_grain_vector_mechanism_external_cohort.py",
        ROOT / "grain_research/vector_mechanisms/cohort.py",
        ROOT / "grain_research/vector_mechanisms/data_views.py",
        ROOT / "scripts/run_suica_v8_label_free_pilot.py",
        ROOT / "suica_core/v8_realtext.py",
        ROOT / "suica_core/meps_layout.py",
        ROOT / "scripts/audit_grain_meps_material.py",
        ROOT / "scripts/run_grain_interaction_natural.py",
    ]
    return {
        "files": [{"path": str(p), "sha256": sha256_file(p)} for p in paths],
        "routes": {
            "pandora": "V2 F14 -> logic findings -> antecedent scanner -> original comments; GP05 -> local_bilinear -> private author-ID cache",
            "essays": "V2 F14 -> GP05/local_bilinear -> local_construction.ESSAYS",
            "x_market": "F14 X requirement -> existing v8_label_free_pilot.DEFAULT_DATA_ROOT/x_fullmarkettext -> v8_realtext.load_x_panel registered x_posts.csv pattern",
            "meps": "F14 MEPS requirement -> existing interaction_natural data root -> meps_layout decrypted_output; existing audit_grain_meps_material metadata-only inputs",
        },
        "policy": "F14 metadata-only scope; historical text use does not authorize new text reading or fitting",
    }


def placeholder_checks() -> dict:
    """Exercise each actual adapter and negative paths, not literal PASS flags."""
    results = []
    for dataset, spec in SPECS.items():
        row = {c: "SIM_VALUE" for c in spec.columns()}
        row[spec.author] = "SIM_AUTHOR"
        if spec.time:
            row[spec.time] = (
                "1704067200"
                if spec.time_kind == "unix_seconds"
                else "2024-01-01T09:00:00+09:00"
            )
        if spec.role:
            row[spec.role] = "user"
        if spec.order:
            row[spec.order] = "0"
        row["text"] = ""  # A synthetic empty value is not materialized in output.
        normalized = normalize_metadata(pd.DataFrame([row]), spec)
        expected = {"author": "SIM_AUTHOR", "role": "user" if spec.role else "author"}
        ok = all(normalized.iloc[0][k] == v for k, v in expected.items())
        ok = bool(ok and "text" not in normalized and normalized.event.notna().all())
        if spec.time:
            ok = ok and normalized.iloc[0]["time"] == "2024-01-01T00:00:00.000000Z"
        results.append(
            {"check": f"{dataset}_actual_schema_empty_text", "passed": bool(ok)}
        )
    base = pd.DataFrame({"event": ["SIM_EVENT"], "author": ["SIM_AUTHOR"]})
    for name, left, right, kwargs in [
        ("duplicate_key", pd.concat([base, base]), base, {"require": "one_to_one"}),
        ("wrong_author", base, base.assign(author="SIM_OTHER"), {"agree": ("author",)}),
        ("missing_left", base, base.assign(event="SIM_OTHER"), {"complete_left": True}),
        ("null_key", base.assign(event=None), base, {}),
    ]:
        try:
            audit_join(left, right, ["event"], **kwargs)
        except ValueError as exc:
            results.append({"check": name, "passed": True, "reason": str(exc)})
        else:
            results.append({"check": name, "passed": False})
    spec = SPECS["pandora"]
    row = {k: "SIM" for k in spec.columns()}
    row[spec.time] = "1704067200000"
    normalized = normalize_metadata(pd.DataFrame([row]), spec)
    results.append(
        {
            "check": "milliseconds_not_seconds",
            "passed": bool(
                normalized.iloc[0].time_invalid == 1
                and pd.isna(normalized.iloc[0].time)
            ),
        }
    )
    return {
        "scope": "Synthetic placeholders through the real adapter; not human evidence",
        "checks": results,
        "all_passed": all(r["passed"] for r in results),
    }


def verify_meps_metadata() -> dict:
    """Validate the prior metadata manifest, opening no original participant files."""
    base = ROOT / "results/grain_vector_mechanisms"
    roots = [base / "vm06_meps_preflight_r3", base / "vm06_meps_windows_r3"]
    checked = []
    for root in roots:
        summary = json.loads((root / "summary.json").read_text())
        if summary["labels_read"] or summary["model_forwards"]:
            raise ValueError("MEPS metadata source exceeds declared scope")
        for name, expected in summary["outputs_sha256"].items():
            if Path(name).name != name:
                raise ValueError("MEPS metadata path escapes registered directory")
            path = root / name
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError("changed MEPS metadata")
            checked.append({"path": str(path), "sha256": actual})
    return {
        "metadata_hashes": checked,
        "original_text_opened": False,
        "questionnaire_values_read": False,
    }


def meps_joins() -> dict:
    """Test view/turn/window identities without reading text or scale values."""
    base = ROOT / "results/grain_vector_mechanisms"
    view_keys = ["participant_code", "view_kind", "condition"]
    keys = view_keys + ["role", "turn_sequence"]
    views = pd.read_csv(
        base / "vm06_meps_preflight_r3/private_view_metadata.csv",
        dtype={"participant_code": str},
    )
    turns = pd.read_csv(
        base / "vm06_meps_windows_r3/private_turn_integrity.csv",
        dtype={"participant_code": str},
    )
    spans = pd.read_csv(
        base / "vm06_meps_windows_r3/private_window_metadata.csv",
        dtype={"participant_code": str},
    )
    joins = {
        "turns_to_views": audit_join(
            turns, views, view_keys, require="many_to_one", complete_left=True
        ),
        "windows_to_turns": audit_join(
            spans, turns, keys, require="many_to_one", complete_left=True
        ),
    }
    if spans.duplicated(keys + ["window"]).any():
        raise ValueError("duplicate MEPS window")
    check = spans.groupby(keys).agg(
        n=("window", "size"), first=("start", "min"), last=("end", "max")
    )
    check = turns.merge(check, on=keys, validate="one_to_one")
    if not (
        check.n.eq(check.windows).all()
        and check["first"].eq(0).all()
        and check["last"].eq(check.source_chars).all()
        and check.covered_chars.eq(check.source_chars).all()
    ):
        raise ValueError("MEPS turn/window coverage mismatch")
    discontinuities = 0
    for _, group in spans.groupby(keys):
        ordered = group.sort_values("window")
        discontinuities += int(
            (ordered.end.to_numpy()[:-1] != ordered.start.to_numpy()[1:]).sum()
        )
    if discontinuities:
        raise ValueError("MEPS gaps/overlaps in window coverage")
    joins["clock_time_only_rows_reported_in_cache"] = int(
        views.clock_time_only_rows.sum()
    )
    joins["ordinal_order_is_calendar_time"] = False
    return joins


def support_rows(dataset: str, profile: dict) -> list[dict]:
    """Derive finite data support from counts, never promote it to psych validity."""

    def row(target, supported, reason, next_step):
        return {
            "dataset": dataset,
            "target": target,
            "support": "METADATA_SUPPORT" if supported else "MISSING_SUPPORT",
            "reason": reason,
            "next_contract": next_step,
        }

    return [
        row(
            "observed_author_condition_profile",
            profile["authors_multiple_conditions"] > 0,
            f"{profile['authors_multiple_conditions']} authors with multiple recorded conditions; observed selection only",
            "F16-R02-TEXT: same-target reader/unit/error and eligibility",
        ),
        row(
            "calendar_ordered_published_expression",
            profile["authors_multiple_days"] > 0,
            f"{profile['authors_multiple_days']} authors on multiple UTC dates; no fixed experimental occasions inferred",
            "F16-R02-TEXT: deduplicated event-to-vector addresses and irregular-time observation law",
        ),
        row(
            "same_author_condition_across_dates",
            profile["same_author_condition_multiple_days"] > 0,
            f"{profile['same_author_condition_multiple_days']} author-condition cells across dates; not randomized or causal",
            "F04/F05: declared condition support and occasion model",
        ),
        row(
            "complete_opportunity_adjusted_response",
            False,
            profile["opportunity_reason"],
            "F03: observed-collection estimand or explicit bounds; do not invent exposure",
        ),
        row(
            "T1_T2_T3_gaussian_score_directly",
            False,
            "Metadata supplies addresses, not calibrated continuous observations or a state transition law",
            "F16-R02-TEXT: actual text observation mapping before numeric target scorer",
        ),
    ]


def main() -> None:
    """Scan every registered row, save aggregates, and keep protocol gaps visible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona-root", type=Path, default=ROOT.parent / "project persona"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    code = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f14_metadata.py",
        ROOT / "grain_research/vector_mechanisms/cohort.py",
        ROOT / "grain_research/vector_mechanisms/data_views.py",
    ]
    write(
        args.output / "protocol_before_read.json",
        {
            "scope": "F14-R01 authorized metadata; full selected files, no row sampling",
            "fields": {k: v.contract() for k, v in SPECS.items()},
            "code": {str(p): sha256_file(p) for p in code},
            "no_text_values": True,
            "no_questionnaire_values": True,
            "no_training_or_model_calls": True,
            "parent_promotion": False,
        },
    )
    write(args.output / "source_lineage.json", source_chain(args.persona_root))
    write(
        args.output / "metadata_adapter_schema.json",
        {k: v.contract() for k, v in SPECS.items()},
    )
    checks = placeholder_checks()
    if not checks["all_passed"]:
        write(args.output / "adapter_checks.json", checks)
        raise ValueError("actual adapter placeholder checks failed")
    checks["meps_provenance"] = verify_meps_metadata()
    sources = provenance(args.persona_root)
    inventory, schemas, profiles, joins, support = [], [], {}, {}, []
    with tempfile.TemporaryDirectory(prefix="grain-f14-metadata-") as temp:
        for dataset, paths in sources.items():
            if not paths:
                raise ValueError(f"no registered files for {dataset}")
            spec = SPECS[dataset]
            profile = MetadataProfile(Path(temp) / f"{dataset}.sqlite")
            try:
                for file_index, path in enumerate(paths):
                    stat_before = path.stat()
                    schema = inspect_schema(path, spec.columns())
                    schemas.append(schema | {"dataset": dataset})
                    file_rows = 0
                    checksum = sha256_file(path)
                    for frame in read_projected(path, spec):
                        if (
                            dataset == "essays"
                            and not frame.user_id_source.eq("original_author_id").all()
                        ):
                            raise ValueError("Essays original author key changed")
                        normalized = normalize_metadata(frame, spec)
                        profile.add(normalized)
                        file_rows += len(frame)
                    stat_after = path.stat()
                    if (stat_before.st_size, stat_before.st_mtime_ns) != (
                        stat_after.st_size,
                        stat_after.st_mtime_ns,
                    ):
                        raise ValueError("source changed during metadata read")
                    inventory.append(
                        {
                            "dataset": dataset,
                            "source_path": str(path),
                            "hash": checksum,
                            "grain": spec.grain,
                            "author_key": spec.author,
                            "time_key": spec.time or "",
                            "condition_key": "+".join(spec.condition),
                            "opportunity_observed": False,
                            "public_or_private": "local_restricted",
                            "allowed_use": "metadata projection only; no questionnaire/text values or fitting",
                            "target_support": "see natural_support_matrix.csv; no direct Gaussian target",
                            "missing_reason": "no exposure denominator or calibrated observation law",
                            "rows": file_rows,
                            "bytes": stat_after.st_size,
                        }
                    )
                    if file_index % 20 == 0:
                        print(
                            json.dumps(
                                {
                                    "dataset": dataset,
                                    "files_processed": file_index + 1,
                                    "files": len(paths),
                                }
                            ),
                            flush=True,
                        )
                profiles[dataset] = profile.availability_report()
                # Persist completed scans before any downstream join can fail.
                write(args.output / f"{dataset}_availability.json", profiles[dataset])
                write_rows(args.output / "data_source_inventory.csv", inventory)
                write(args.output / "schema_audit.json", schemas)
                if dataset in {"pandora", "essays"}:
                    cached = (
                        ROOT
                        / "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2"
                        / dataset
                        / "private_authors.json"
                    )
                    ids = json.loads(cached.read_text())["ids"]
                    joins[dataset] = audit_join(
                        pd.DataFrame({"author": ids}),
                        cache_author_keys(profile.authors(), dataset),
                        ["author"],
                        require="one_to_one",
                        complete_left=True,
                    )
                    joins[dataset]["cache_manifest_sha256"] = sha256_file(cached)
                    joins[dataset]["author_identity_rule"] = (
                        f"sha256({ID_SEED}::raw_author)[:24]; original cohort rule"
                        if dataset == "pandora"
                        else "identity; original Essays ID"
                    )
                    joins[dataset]["unit_timestamp_mapping"] = (
                        "NOT_ESTABLISHED: author match does not assign original events to cached spans"
                    )
                support.extend(support_rows(dataset, profiles[dataset]))
                write(args.output / f"{dataset}_availability.json", profiles[dataset])
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "rows": profiles[dataset]["rows"],
                            "events": profiles[dataset]["events"],
                            "authors": profiles[dataset]["authors"],
                        }
                    ),
                    flush=True,
                )
            finally:
                profile.close()
    joins["meps"] = meps_joins()
    layout = ROOT / "results/meps_ai_conv_layout_audit/layout_manifest.json"
    known = json.loads(layout.read_text())
    export_root = (
        args.persona_root
        / "data_sets/MEPS+AI_conv_experiment"
        / known["text_export_relative_path"]
    )
    checks["meps_layout"] = {
        "manifest_sha256": sha256_file(layout),
        "export_directory_exists": export_root.is_dir(),
        "participant_directories": sum(
            p.is_dir() and not p.name.startswith(".") for p in export_root.iterdir()
        ),
        "original_files_opened": False,
    }
    checks["joins"] = joins
    checks["integrity_failures"] = [
        f"{d}:{key}"
        for d, p in profiles.items()
        for key in (
            "missing_author_rows",
            "missing_event_rows",
            "event_author_conflicts",
            "event_time_conflicts",
        )
        if p[key]
    ]
    write_rows(args.output / "data_source_inventory.csv", inventory)
    write_rows(args.output / "natural_support_matrix.csv", support)
    write(args.output / "schema_audit.json", schemas)
    write(args.output / "adapter_checks.json", checks)
    summary = {
        "status": "METADATA_CHECKS_COMPLETE_WITH_FINDINGS",
        "datasets": profiles,
        "metadata_integrity_findings": checks["integrity_failures"],
        "input_files": len(inventory),
        "source_bytes": sum(r["bytes"] for r in inventory),
        "seconds": time.monotonic() - start,
        "model_calls": 0,
        "value_projection": "metadata only; mixed CSV bytes hashed, no text/label columns materialized",
        "MEPS_raw_opened": False,
        "identifiers_exported": False,
        "F14_parent_complete": False,
        "remaining": [
            "exact original event-to-cached-vector timestamp linkage",
            "source-fixed observation units/normalizers and actual F16 text law",
            "original dependency/card adoption",
        ],
        "next_decision": "Use measured event/time/role support; resolve duplicate events before chronological scoring; do not infer T2 latent state from timestamps alone",
    }
    write(args.output / "summary.json", summary)
    write(
        args.output / "acceptance.json",
        {
            "metadata_inventory_executed": True,
            "adapter_checks": checks["all_passed"],
            "findings": checks["integrity_failures"],
            "parent_card_promoted": False,
            "natural_psychological_validation": False,
            "remaining": summary["remaining"],
        },
    )
    write(
        args.output / "output_manifest.json",
        {p.name: sha256_file(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "input_files",
                    "seconds",
                    "metadata_integrity_findings",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
