#!/usr/bin/env python3
"""Construct source-valued attested-event scores, never fit native answers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_trait_action as E
from grain_research.construction import math_observation_regime_target as M
from grain_research.construction import math_path_edit as P
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_observation_regime_target_hm246.json"
DEFAULT = ROOT / "results/grain_math_response_v1/observation_regime_target_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_OBSERVATION_REGIME_TARGET_PLAN_HM246.md",
    "grain_research/construction/math_observation_regime_target.py",
    "scripts/run_grain_observation_regime_target_hm246.py",
    "tests/test_grain_observation_regime_target.py",
    "grain_research/construction/math_path_edit.py",
    "grain_research/construction/math_path_functional.py",
    "grain_research/construction/math_event_trait_action.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
)
STAGES = ("compiled_seal.json", "cold_seal.json", "analyzed_seal.json")


def prepare(out):
    """Bind current24-event authors to exact old raw coordinates and chronology."""
    cfg = R.read(ROOT / CONFIG)
    parents = {k: ROOT / cfg[k] for k in cfg if k.endswith("_parent")}
    inputs = []
    for key, stage, names in (
        ("source_parent", "prepared_seal.json", ("source_inputs.npz",)),
        (
            "source_parent",
            "compiled_seal.json",
            ("source_tables.npz", "cold_bundle.npz"),
        ),
        ("path_parent", "prepared_seal.json", ("observations.npz",)),
        ("event_parent", "prepared_seal.json", ("observations.npz",)),
        ("source_address_parent", "prepared_seal.json", ("observations.npz",)),
        ("vector_parent", "prepared_seal.json", ("observations.npz",)),
        ("selection_parent", "prepared_seal.json", ("observations.npz",)),
    ):
        parent = parents[key]
        S.parent_seal(parent, stage, names)
        inputs.extend(parent / p for p in (stage,) + names)
    source = S.selected(
        parents["source_parent"] / "source_inputs.npz", ("source_paths", "targets")
    )
    addresses = S.selected(
        parents["source_address_parent"] / "observations.npz",
        ("source_indices", "bank"),
    )
    paths = S.selected(parents["path_parent"] / "observations.npz", ("paths",))
    events = S.selected(
        parents["event_parent"] / "observations.npz",
        ("coordinates", "times", "original_units"),
    )
    vectors = S.selected(
        parents["vector_parent"] / "observations.npz",
        ("panel_indices", "mapping", "panel_vectors", "source_indices"),
    )
    selections = S.selected(
        parents["selection_parent"] / "observations.npz",
        ("author_ids", "selected", "control", "source_indices"),
    )
    with np.load(ROOT / cfg["material"], allow_pickle=False) as z:
        ids = z["author_ids"]
        raw = E.unit_vectors(z["comments"][vectors["panel_indices"], :24])
    inputs.append(ROOT / cfg["material"])
    panel, src = vectors["panel_indices"], addresses["source_indices"]
    if (
        len(panel) != 24
        or len(src) != 386
        or np.intersect1d(panel, src).size
        or not np.array_equal(src, vectors["source_indices"])
        or not np.array_equal(src, selections["source_indices"])
        or not np.array_equal(ids[panel], selections["author_ids"])
        or not np.array_equal(source["source_paths"], paths["paths"][src])
        or source["targets"].shape != (2, 386, 5)
        or not np.array_equal(
            addresses["bank"], np.r_[np.zeros(203, int), np.ones(183, int)]
        )
    ):
        raise ValueError(
            "Different SOURCE/current authors, labels or exact event family"
        )
    positions = np.argsort(events["original_units"], axis=1)[panel, :24]
    coordinates = np.take_along_axis(
        events["coordinates"][panel], positions[:, :, None], 1
    )
    times = np.take_along_axis(events["times"][panel], positions, 1)
    cells = np.take_along_axis(paths["paths"].reshape(624, 36)[panel], positions, 1)
    vector_error = float(
        np.max(abs(raw.reshape(24, 2, 12, 1024) - vectors["panel_vectors"][:, 1]))
    )
    coordinate_error = float(np.max(abs(raw @ vectors["mapping"] - coordinates)))
    if (
        max(vector_error, coordinate_error) > cfg["audit_tolerance"]
        or np.any(np.diff(positions, axis=1) != 1)
        or np.any(np.diff(times, axis=1) <= 0)
    ):
        raise ValueError("Current first/two raw windows are not exact attested events")
    records = []
    for u in range(24):
        author = str(ids[panel[u]])
        observed = [
            {
                "event_id": f"{author}:{i // 12}:{i % 12}",
                "owner": author,
                "position": int(positions[u, i]),
                "timestamp": float(times[u, i]),
                "cell": int(cells[u, i]),
            }
            for i in range(24)
        ]

        def add(name, order, author=author, observed=observed):
            record = {
                "record_id": name,
                "author": author,
                "regime": "raw_pandora_publication",
                "events": [observed[int(i)] for i in order],
            }
            M.canonical_events(record)
            records.append(record)

        add(f"whole/{u}", range(24))
        for v in (0, 1):
            add(f"view/{u}/{v}", range(12 * v, 12 * (v + 1)))
            companion = list(range(12 * (1 - v), 12 * (2 - v)))
            reversed_companion = companion[:4] + companion[4:11][::-1] + companion[11:]
            for arm, key in (("S", "selected"), ("C", "control")):
                removed = selections[key][u, v]
                focal = [12 * v + i for i in range(12) if i not in removed]
                add(f"edit/{u}/{v}/{arm}", focal + companion)
                add(f"display/{u}/{v}/{arm}", focal + reversed_companion)
    if len(records) != cfg["records"]:
        raise ValueError("Changed full finite input support")
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str(p.relative_to(ROOT)): R.digest(p) for p in dict.fromkeys(inputs)
            },
            "cohort_reuse": "historical current24,not fresh encoder/representation holdout",
            "original_scope_complete": False,
        },
    )
    R.archive(out / "source_inputs.npz", **source, bank=addresses["bank"])
    with (out / "cold_inputs.jsonl").open("x") as f:
        for rec in records:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "unique_events": 576,
            "records": len(records),
            "source_authors": 386,
            "source_current_author_overlap": 0,
            "exact_raw_vector_replay_error": vector_error,
            "exact_raw_coordinate_replay_error": coordinate_error,
            "strict_attested_times": True,
            "original_positions": [int(positions.min()), int(positions.max())],
            "cue_raw_vector_rms": float(
                np.sqrt(
                    np.mean(
                        (
                            vectors["panel_vectors"][:, 0]
                            - vectors["panel_vectors"][:, 1]
                        )
                        ** 2
                    )
                )
            ),
            "raw_to_cue_source_calibration_claimed": False,
            "native_answers_application_labels_raw_texts_read": False,
            "triples_are_independent": False,
            "new_model_API_private_training_calls": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("source_inputs.npz", "cold_inputs.jsonl", "input_quality.json"),
    )
    print("PREPARED HM246 current24/576 exact raw events/264records", flush=True)


def compile_source(out):
    """Reuse pooled old gap laws and construct only their observation disintegration."""
    cfg = R.check(out)["config"]
    z = S.selected(out / "source_inputs.npz", ("source_paths", "targets", "bank"))
    parent = ROOT / cfg["source_parent"]
    old = S.selected(parent / "source_tables.npz", ("tables",))["tables"]
    baseline = S.selected(
        parent / "cold_bundle.npz", ("fields_0", "fields_1", "gaps_0")
    )
    start, readings, tables = time.perf_counter(), [], []
    for i, y in enumerate(z["targets"]):
        by_bank, by_table = [], []
        for bank in (-1, 0, 1):
            if bank == -1:
                table = old[i]
            else:
                mask = z["bank"] == bank
                table = P.compile_source(
                    z["source_paths"][mask], y[mask], cfg["maximum_deleted"]
                )["tables"]
            by_table.append(table)
            by_bank.append(M.disintegrate(table))
        tables.append(by_table)
        readings.append(by_bank)
    errors = []
    for i in range(2):
        f, expected = (
            readings[i][0]["field"],
            baseline[f"fields_{i}"].reshape(15, 5, 512),
        )
        if not np.array_equal(np.isfinite(f), np.isfinite(expected)):
            raise ValueError("Existing SOURCE support changed")
        errors.append(
            float(np.max(abs(f[np.isfinite(f)] - expected[np.isfinite(expected)])))
        )
    if max(errors) > cfg["audit_tolerance"]:
        raise ValueError("Old source valuation was replaced rather than reused")
    arrays = {
        name: np.array([[r[name] for r in row] for row in readings])
        for name in (
            "field",
            "composition",
            "succession",
            "reference",
            "bag_field",
            "bag_reference",
            "prior",
        )
    }
    arrays.update(gaps=baseline["gaps_0"], bag_mapping=M.bag_map()[1])
    seconds = time.perf_counter() - start
    if seconds > cfg["max_cpu_seconds"]:
        raise TimeoutError("Finite SOURCE cap exceeded; no automatic retry")
    R.archive(out / "cold_bundle.npz", **arrays)
    R.archive(out / "source_tables.npz", tables=np.asarray(tables))
    R.write(
        out / "compile_receipt.json",
        {
            "seconds": seconds,
            "old_pooled_fields_replay_errors": errors,
            "new_model_calls": 0,
            "native_answer_fit": False,
        },
    )
    R.seal(
        out,
        "compiled_seal.json",
        ("cold_bundle.npz", "source_tables.npz", "compile_receipt.json"),
    )
    print("SOURCE_VALUED", seconds, flush=True)


def cold_runtime(bundle, inputs, output):
    """An actual isolated scoring process sees only aggregate law and event inputs."""
    z = S.selected(bundle, ("field", "composition", "prior", "gaps"))
    with inputs.open() as f, output.open("x") as target:
        for line in f:
            rec, values = json.loads(line), []
            for law in range(2):
                for bank in range(3):
                    value = M.score_record(
                        z["field"][law, bank],
                        z["composition"][law, bank],
                        z["prior"][law, bank, 0],
                        z["gaps"],
                        rec,
                    )
                    values.append({"law": law, "bank": bank, "reading": value})
            target.write(
                json.dumps(
                    {"record_id": rec["record_id"], "values": values}, allow_nan=False
                )
                + "\n"
            )


def cold(out):
    """Run the bounded cold process once without SOURCE rows or model answers."""
    R.check(out, ("compiled_seal.json",))
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / SCIENCE[3]),
            "cold-runtime",
            "--bundle",
            str(out / "cold_bundle.npz"),
            "--inputs",
            str(out / "cold_inputs.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        check=True,
        timeout=120,
    )
    R.write(
        out / "cold_receipt.json",
        {
            "seconds": time.perf_counter() - start,
            "separate_process": True,
            "application_labels_or_native_answers": False,
        },
    )
    R.seal(out, "cold_seal.json", ("cold_output.jsonl", "cold_receipt.json"))
    print("COLD_COMPLETE", flush=True)


def read_cold(out):
    """Index complete and incomplete numeric observations without outcome filtering."""
    rows = [
        json.loads(line)
        for line in (out / "cold_output.jsonl").read_text().splitlines()
    ]
    if len(rows) != 264 or len({r["record_id"] for r in rows}) != 264:
        raise ValueError("Full cold input coverage required")
    return {r["record_id"]: r for r in rows}


def analyze(out):
    """Report all signed target/source-bank accounts, not an accuracy leaderboard."""
    cfg = R.check(out, ("compiled_seal.json", "cold_seal.json"))["config"]
    z = S.selected(
        out / "cold_bundle.npz",
        (
            "field",
            "composition",
            "succession",
            "reference",
            "prior",
            "gaps",
            "bag_mapping",
        ),
    )
    output = read_cold(out)
    scores, actions, source_rows, cross, coverage = [], [], [], [], []
    decomposition_error = 0.0
    for name, row in output.items():
        for variant in row["values"]:
            law, bank, value = variant["law"], variant["bank"], variant["reading"]
            coverage.append(
                {
                    "record_id": name,
                    "law": cfg["laws"][law],
                    "bank": cfg["banks"][bank],
                    "complete": value["complete"],
                    "triples": value["triples"],
                    "unsupported_full_triples": sum(
                        not t["full_supported"] for t in value["ledger"]
                    ),
                    "unsupported_composition_triples": sum(
                        not t["composition_supported"] for t in value["ledger"]
                    ),
                }
            )
            if value["complete"]:
                decomposition_error = max(
                    decomposition_error,
                    float(
                        np.max(
                            abs(
                                np.array(value["full"])
                                - value["composition"]
                                - np.array(value["succession"])
                            )
                        )
                    ),
                )
            for q, target in enumerate(cfg["targets"]):
                scores.append(
                    {
                        "record_id": name,
                        "law": cfg["laws"][law],
                        "bank": cfg["banks"][bank],
                        "target": target,
                        "full": value["full"][q] if value["full"] is not None else None,
                        "composition": value["composition"][q]
                        if value["composition"] is not None
                        else None,
                        "succession": value["succession"][q]
                        if value["succession"] is not None
                        else None,
                    }
                )
    display_error, paired = 0.0, 0
    for u, v in np.ndindex(24, 2):
        for arm in ("S", "C"):
            old, new = output[f"edit/{u}/{v}/{arm}"], output[f"display/{u}/{v}/{arm}"]
            for a, b in zip(old["values"], new["values"], strict=True):
                for key in ("full", "composition", "succession"):
                    av, bv = a["reading"][key], b["reading"][key]
                    if (av is None) != (bv is None):
                        raise ValueError("Display changed support")
                    if av is not None:
                        display_error = max(
                            display_error, float(np.max(abs(np.array(av) - bv)))
                        )
                if a["reading"]["ledger"] != b["reading"]["ledger"]:
                    raise ValueError("Display changed attested event/gap observations")
                paired += 1
        for i in range(6):
            a, b = (output[f"edit/{u}/{v}/{arm}"]["values"][i] for arm in ("S", "C"))
            law, bank = a["law"], a["bank"]
            for q, target in enumerate(cfg["targets"]):
                diff = {
                    key: a["reading"][key][q] - b["reading"][key][q]
                    if a["reading"][key] is not None and b["reading"][key] is not None
                    else None
                    for key in ("full", "composition", "succession")
                }
                actions.append(
                    {
                        "author": u,
                        "view": v,
                        "law": cfg["laws"][law],
                        "bank": cfg["banks"][bank],
                        "target": target,
                        **diff,
                    }
                )
    for law, bank, d, q in np.ndindex(2, 3, 15, 5):
        ref, bag, order = (
            z["reference"][law, bank, d],
            z["composition"][law, bank, d, q],
            z["succession"][law, bank, d, q],
        )
        observed = ref > 0
        centered = bag[observed] - z["prior"][law, bank, d, q]
        source_rows.append(
            {
                "law": cfg["laws"][law],
                "bank": cfg["banks"][bank],
                "d1": int(z["gaps"][d, 0]),
                "d2": int(z["gaps"][d, 1]),
                "target": cfg["targets"][q],
                "supported_ordered_cells": int(observed.sum()),
                "composition_rms": float(np.sqrt(ref[observed] @ centered**2)),
                "succession_rms": float(np.sqrt(ref[observed] @ order[observed] ** 2)),
            }
        )
    for law, d, q, target in np.ndindex(2, 15, 5, 5):
        ref = np.minimum(z["reference"][law, 1, d], z["reference"][law, 2, d])
        common = ref > 0
        w = ref[common] / ref[common].sum()
        a, b = (
            z["succession"][law, 1, d, q, common],
            z["succession"][law, 2, d, target, common],
        )
        cross.append(
            {
                "law": cfg["laws"][law],
                "d1": int(z["gaps"][d, 0]),
                "d2": int(z["gaps"][d, 1]),
                "source_target": cfg["targets"][q],
                "other_target": cfg["targets"][target],
                "signed_order_product": float(w @ (a * b)),
                "common_ordered_cells": int(common.sum()),
            }
        )
    primary = [r for r in coverage if r["law"] == "actual" and r["bank"] == "pooled386"]
    baseline = [
        r
        for r in source_rows
        if r["law"] == "actual" and r["bank"] == "pooled386" and r["d1"] == r["d2"] == 1
    ]
    bank_sign = [
        r
        for r in cross
        if r["d1"] == r["d2"] == 1 and r["source_target"] == r["other_target"]
    ]
    summary = {
        "construction": "source-valued current-event chronology,not native-answer fit",
        "complete_primary_records": sum(r["complete"] for r in primary),
        "total_primary_records": len(primary),
        "complete_primary_whole_authors": sum(
            r["complete"] for r in primary if r["record_id"].startswith("whole/")
        ),
        "display_pairs_including_all_laws_banks": paired,
        "display_score_error": display_error,
        "full_composition_succession_error": decomposition_error,
        "primary_source_adjacent_fields": baseline,
        "cross_bank_adjacent_same_target": bank_sign,
        "new_model_API_private_training_application_truth_calls": 0,
        "author_posterior_or_accuracy_claim": False,
        "original_scope_complete": False,
    }
    for name, rows in (
        ("scores_by_record.csv", scores),
        ("evidence_actions.csv", actions),
        ("source_relation_accounts.csv", source_rows),
        ("cross_bank_accounts.csv", cross),
        ("coverage.csv", coverage),
    ):
        R.table(out / name, rows)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "scores_by_record.csv",
            "evidence_actions.csv",
            "source_relation_accounts.csv",
            "cross_bank_accounts.csv",
            "coverage.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary, allow_nan=False), flush=True)


def verify(out):
    """Replay source conditional laws and literal cold event accounts independently."""
    cfg = R.check(out, STAGES)["config"]
    z = S.selected(
        out / "cold_bundle.npz",
        (
            "field",
            "composition",
            "succession",
            "reference",
            "prior",
            "gaps",
            "bag_mapping",
        ),
    )
    tables = S.selected(out / "source_tables.npz", ("tables",))["tables"]
    output = read_cold(out)
    errors = {
        "conditional_bag_residual": 0.0,
        "literal_source_bag": 0.0,
        "literal_cold_values": 0.0,
    }
    counts = {"source_fields": 0, "cold_target_accounts": 0, "missing_components": 0}
    for law, bank in np.ndindex(2, 3):
        errors["conditional_bag_residual"] = max(
            errors["conditional_bag_residual"],
            M.conditional_residual_error(
                z["reference"][law, bank], z["succession"][law, bank], z["bag_mapping"]
            ),
        )
        t = tables[law, bank].reshape(15, 32, 512)
        for d, group, q in np.ndindex(15, 120, 5):
            support = z["bag_mapping"] == group
            total = t[d, :, support].sum()
            if not total:
                continue
            codes = np.array([bool(g & (1 << q)) for g in range(32)])
            expected = t[d][:, support][codes].sum() / total
            actual = z["composition"][law, bank, d, q, support][0]
            errors["literal_source_bag"] = max(
                errors["literal_source_bag"], abs(float(expected - actual))
            )
            counts["source_fields"] += 1
    for row in output.values():
        for variant in row["values"]:
            law, bank, value = variant["law"], variant["bank"], variant["reading"]
            for key, array in (
                ("full", z["field"]),
                ("composition", z["composition"]),
                ("succession", z["succession"]),
            ):
                if value[key] is None:
                    counts["missing_components"] += 1
                    continue
                sums = []
                for q in range(5):
                    terms = []
                    for unit in value["ledger"]:
                        gap = tuple(unit["original_gap"])
                        j = [tuple(map(int, g)) for g in z["gaps"]].index(gap)
                        terms.append(float(array[law, bank, j, q, unit["cell"]]))
                    val = sum(terms) / len(terms)
                    sums.append(
                        val
                        if key == "succession"
                        else val - float(z["prior"][law, bank, 0, q])
                    )
                errors["literal_cold_values"] = max(
                    errors["literal_cold_values"],
                    float(np.max(abs(np.array(sums) - value[key]))),
                )
                counts["cold_target_accounts"] += 5
    summary = R.read(out / "summary.json")
    if (
        max(errors.values()) > cfg["audit_tolerance"]
        or max(
            summary["display_score_error"], summary["full_composition_succession_error"]
        )
        > cfg["audit_tolerance"]
    ):
        raise ValueError("Source/observation contract arithmetic failed; do not rescue")
    R.write(
        out / "verification.json",
        {
            "status": "PASS",
            "errors": errors,
            "counts": counts,
            "same_agent_alternate_not_blind": True,
            "original_scope_complete": False,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps({"errors": errors, "counts": counts}), flush=True)


def main():
    """Run exactly one requested immutable finite stage, with no automatic retries."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "compile",
            "cold",
            "analyze",
            "verify",
            "check",
            "cold-runtime",
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    for key in ("bundle", "inputs", "output"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    if args.stage == "cold-runtime":
        cold_runtime(args.bundle, args.inputs, args.output)
    elif args.stage == "check":
        R.check(args.out, STAGES + ("verification_seal.json",))
        print("SEALED_PASS", flush=True)
    else:
        {
            "prepare": prepare,
            "compile": compile_source,
            "cold": cold,
            "analyze": analyze,
            "verify": verify,
        }[args.stage](args.out)


if __name__ == "__main__":
    main()
