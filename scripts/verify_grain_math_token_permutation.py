#!/usr/bin/env python3
"""Independent zero-forward arithmetic/certificate replay of frozen HM13 endpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import write_json
from scripts.run_grain_math_token_permutation import DEFAULT, verify_source
from scripts.run_grain_upstream_routes import collector_guard

OUTPUT = ROOT / "results/grain_math_response_v1/token_permutation_verification_r1"


def rows(path):
    """Read a serialized output table without using primary evaluation functions."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def categorical_score(native, values):
    """Recalculate the categorical expectation with manually normalized exponentials."""
    logits = np.asarray(native["candidate_logits"], float)
    exp = np.exp(logits - max(logits))
    return float(sum(exp * values) / sum(exp))


def verify(packet, out):
    """Verify exact permutation inputs, old references and every reported effect/interval."""
    if out.exists():
        raise FileExistsError("preserve earlier verification; use a new output path")
    manifest = verify_source(packet)
    spec = manifest["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(packet / name) != expected:
            raise ValueError(f"primary artifact changed: {name}")
    source = ROOT / spec["source_packet"]
    old_manifest = read_json(source / "00/manifest.json")
    original = ROOT / old_manifest["spec"]["source_packet"]
    capture = read_json(packet / "00/capture_addresses.json")
    addresses = {(r["doc_id"], r["query"]): r for r in capture}
    original_design = read_json(source / "00/design.json")
    center_rows = {
        (r["pair_id"], r["query"]): r
        for r in original_design
        if r["arm"] == "centered_only"
    }
    first_rows = [
        r
        for r in original_design
        if r["arm"] == "mean_only" and r["query"] == spec["queries"][0]
    ]
    selected = {r["pair_id"] for r in first_rows if r["tokens"] > 1}
    for role in spec["roles"]:
        for axis in spec["axes"]:
            selected.add(
                min(
                    r["pair_id"]
                    for r in first_rows
                    if r["role"] == role and r["axis"] == axis and r["tokens"] == 1
                )
            )
    design = read_json(packet / "00/design.json")
    if (
        {r["pair_id"] for r in design} != selected
        or len(design) != spec["patch_forwards"]
        or len({(r["pair_id"], r["query"], r["arm"]) for r in design}) != len(design)
    ):
        raise ValueError("independent input-only selection/count failed")
    canonical = {r["doc_id"]: r for r in capture if r["query"] == spec["queries"][0]}
    bodies = {}
    for doc_id, row in canonical.items():
        with np.load(original / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            bodies[doc_id] = {k: arrays[f"L16_{k}"].copy() for k in row["spans"]}
    certificates = {
        (r["pair_id"], r["arm"]): r for r in read_json(packet / "00/geometry.json")
    }
    geometry_error = 0.0
    certificate_count = 0
    for row in design:
        if row["query"] != spec["queries"][0]:
            continue
        h = bodies[row["receiver_doc"]][row["span"]]
        d = bodies[row["donor_doc"]][row["span"]]
        reference = h
        if row["arm"] == "centered_reverse":
            delta = d.astype(np.float64) - h.astype(np.float64)
            # Match HM12's stated parentheses before the final float32 cast.
            reference = (
                h.astype(np.float64)
                + (delta - np.broadcast_to(np.mean(delta, axis=0), delta.shape))
            ).astype(np.float32)
        target = np.take(reference, np.arange(len(reference) - 1, -1, -1), axis=0)
        ref_bytes = sorted(np.ascontiguousarray(v).tobytes() for v in reference)
        target_bytes = sorted(np.ascontiguousarray(v).tobytes() for v in target)
        if ref_bytes != target_bytes:
            raise ValueError("byte-exact multiset failed")
        saved = certificates[row["pair_id"], row["arm"]]
        expected = {
            "reference_state_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            "target_state_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
            "multiset_sha256": hashlib.sha256(b"".join(ref_bytes)).hexdigest(),
        }
        if any(saved[k] != v for k, v in expected.items()):
            raise ValueError("complete matrix certificate failed")
        mean_ref = sum(reference.astype(float)) / len(reference)
        mean_target = sum(target.astype(float)) / len(target)
        cov_ref = sum(
            np.outer(v - mean_ref, v - mean_ref) for v in reference.astype(float)
        ) / len(reference)
        cov_target = sum(
            np.outer(v - mean_target, v - mean_target) for v in target.astype(float)
        ) / len(target)
        geometry_error = max(
            geometry_error,
            float(np.max(np.abs(mean_ref - mean_target))),
            float(np.max(np.abs(cov_ref - cov_target))),
        )
        if bool(saved["is_exact_input_noop"]) != (h.tobytes() == target.tobytes()):
            raise ValueError("independent exact input no-op failed")
        certificate_count += 1
    if geometry_error > spec["moment_replay_tolerance"]:
        raise ValueError("independent moment replay failed")
    values = np.asarray(manifest["alphabet"]["values"], float)
    records = []
    arithmetic_error = 0.0
    grouped = defaultdict(dict)
    for row in design:
        record = read_json(packet / "01/units" / f"unit_{row['unit']:04d}.json")
        if record["address"] != row:
            raise ValueError("full explicit address mismatch")
        old_addr = addresses[row["receiver_doc"], row["query"]]
        base = read_json(original / "01/units" / f"unit_{old_addr['unit']:04d}.json")
        center_addr = center_rows[row["pair_id"], row["query"]]
        center = read_json(source / "01/units" / f"unit_{center_addr['unit']:04d}.json")
        baseline, centered = (v["native"]["conditional_score"] for v in (base, center))
        reference = baseline if row["arm"] == "receiver_reverse" else centered
        if (
            record["reference"] != reference
            or record["baseline"] != baseline
            or record["centered_reference"] != centered
        ):
            raise ValueError("saved actual reference mismatch")
        score = categorical_score(record["native"], values)
        effect = score - reference
        arithmetic_error = max(
            arithmetic_error,
            abs(score - record["native"]["conditional_score"]),
            abs(effect - record["effect"]),
            abs(abs(effect) / 2 - record["set_function_max_error_lower_bound"]),
        )
        certificate = certificates[row["pair_id"], row["arm"]]
        if (
            record["assigned_state_sha256"] != certificate["target_state_sha256"]
            or any(record[k] != v for k, v in certificate.items())
            or record["live_body_error"] != 0
            or record["live_question_error"] != 0
        ):
            raise ValueError("actual hook state mismatch")
        records.append(
            {
                **row,
                "effect": record["native"]["conditional_score"] - reference,
                "hm12_centered_action": centered - baseline,
            }
        )
        grouped[row["pair_id"], row["query"]][row["arm"]] = records[-1]
    contrasts = []
    for arms in grouped.values():
        receiver, centered = (arms[k] for k in spec["arms"])
        contrasts.append(
            {
                **receiver,
                "receiver_placement_effect": receiver["effect"],
                "centered_placement_effect": centered["effect"],
                "placement_interaction": centered["effect"] - receiver["effect"],
            }
        )
    table_error = 0.0
    effect_rows = rows(packet / "02/interventions.csv")
    contrast_rows = rows(packet / "02/placement_contrasts.csv")
    if len(effect_rows) != len(records) or len(contrast_rows) != len(contrasts):
        raise ValueError("serialized output coverage mismatch")
    for saved, recalculated in zip(effect_rows, records, strict=True):
        table_error = max(
            table_error, abs(float(saved["effect"]) - recalculated["effect"])
        )
    for saved, recalculated in zip(contrast_rows, contrasts, strict=True):
        for key in (
            "receiver_placement_effect",
            "centered_placement_effect",
            "placement_interaction",
            "hm12_centered_action",
        ):
            table_error = max(table_error, abs(float(saved[key]) - recalculated[key]))
    checks = (
        ("cell_summary", records, ("role", "axis", "query", "arm", "opportunity")),
        (
            "order_token_summary",
            records,
            ("role", "axis", "query", "arm", "order", "tokens"),
        ),
        ("interaction_summary", contrasts, ("role", "axis", "query", "opportunity")),
    )
    interval_rows = 0
    for name, members, keys in checks:
        for saved in rows(packet / f"02/{name}.csv"):
            cell = [r for r in members if all(str(r[k]) == saved[k] for k in keys)]
            if not cell:
                raise ValueError("empty summary cell")
            metric = saved["metric"]
            families = defaultdict(list)
            for row in cell:
                value = row[metric]
                families[row["family"]].append(
                    abs(value) if saved["statistic"] == "absolute" else value
                )
            x = np.asarray([sum(v) / len(v) for v in families.values()])
            rng = np.random.default_rng(spec["bootstrap_seed"])
            indices = rng.integers(0, len(x), (spec["bootstrap_replicates"], len(x)))
            sample_means = np.sum(np.take(x, indices), axis=1) / len(x)
            lo, hi = np.quantile(sample_means, [0.025, 0.975])
            expected = {
                "mean": sum(x) / len(x),
                "ci_low": lo,
                "ci_high": hi,
                "max_absolute": max(abs(r[metric]) for r in cell),
                "pairs": len(cell),
                "families": len(families),
                "positive_pairs": sum(r[metric] > 0 for r in cell),
                "negative_pairs": sum(r[metric] < 0 for r in cell),
            }
            for key, value in expected.items():
                table_error = max(table_error, abs(float(saved[key]) - value))
            interval_rows += 1
    if max(arithmetic_error, table_error) > 1e-12:
        raise ValueError("independent arithmetic/table replay failed")
    status = read_json(packet / "run_status.json")
    if status["status"] != "COMPLETE_PATCH" or status["patch_calls"] != len(design):
        raise ValueError("actual finite completion mismatch")
    result = {
        "status": "PASS_FINITE_VERIFICATION_NOT_GLOBAL_CLOSURE",
        "verifier_sha256": digest(Path(__file__)),
        "primary_manifest_sha256": digest(packet / "00/manifest.json"),
        "frozen_primary_artifacts_checked": len(ledger),
        "frozen_hm12_artifacts_checked": len(
            read_json(packet / "00/source_hashes.json")
        ),
        "original_hm10b3_artifacts_checked": manifest["original_source_hashes_checked"],
        "actual_units": len(records),
        "geometry_certificates": certificate_count,
        "conditional_interval_rows": interval_rows,
        "full_dimensional_moment_replay_max": geometry_error,
        "categorical_effect_bound_replay_max": arithmetic_error,
        "serialized_table_replay_max": table_error,
        "new_model_calls": 0,
        "independence_scope": "same agent, alternate arithmetic implementation; not third-party replication",
    }
    out.mkdir(parents=True)
    write_json(out / "verification.json", result)
    write_json(
        out / "frozen_primary_artifact_hashes.json",
        {
            str(p.relative_to(packet)): digest(p)
            for p in sorted(packet.rglob("*"))
            if p.is_file()
        },
    )
    print(result, flush=True)


def main():
    """Verify under the primary packet lock without recomputing neural observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with collector_guard(args.packet):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
