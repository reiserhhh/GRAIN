#!/usr/bin/env python3
"""Fresh explicit-condition incidence confirmation after preserved HM-09B r1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_grain_math_binding import read_bank
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json
from run_grain_math_state_exposure import (
    check_source,
    coverage_audit,
    evaluate,
    indices,
    parse,
)

from grain_research.construction import math_state_exposure as S
from grain_research.construction import math_state_scope as T

PRIMARY = ROOT / "results/grain_math_response_v1/state_exposure_r1"
DEFAULT = ROOT / "results/grain_math_response_v1/state_exposure_confirmation_r1"


def fresh_bank(primary):
    """Retain the factorial identity contrast but explicitly repeat 'when'."""
    bank = []
    for r in read_bank(primary / "00/documents.jsonl"):
        if r["kind"] != "bundle":
            continue
        text = r["text"]

        def offset(v, text=text):
            return v + 5 * text[:v].count(" and ")

        reports = [
            {
                **q,
                "root_span": [offset(v) for v in q["root_span"]],
                "projection_text": q["projection_text"].replace(" and ", " and when "),
                "triggers": [
                    {
                        **t,
                        "action_span": [offset(v) for v in t["action_span"]],
                        "object_span": [offset(v) for v in t["object_span"]],
                    }
                    for t in q["triggers"]
                ],
            }
            for q in r["expected_reports"]
        ]
        bank.append(
            {
                **r,
                "material_id": "explicit_" + r["material_id"],
                "text": text.replace(" and ", " and when "),
                "expected_reports": reports,
                "followup_is_result_guided_new_cold_design": True,
            }
        )
    if len(bank) != 64:
        raise ValueError("confirmation design changed")
    return bank


def prepare(out):
    """Freeze the new design separately; primary failed materials stay untouched."""
    if (out / "04/summary.json").exists():
        raise ValueError("confirmation frozen")
    for name, digest in json.loads(
        (PRIMARY / "04/artifact_hashes.json").read_text()
    ).items():
        if sha(PRIMARY / name) != digest:
            raise ValueError("primary artifact changed: " + name)
    source = ROOT / "results/grain_math_response_v1/binding_r1"
    check_source(source)
    bank = fresh_bank(PRIMARY)
    paths = [
        Path(__file__),
        ROOT / "scripts/run_grain_math_state_exposure.py",
        ROOT / "grain_research/construction/math_state_exposure.py",
        ROOT / "grain_research/construction/math_state_scope.py",
    ]
    signature = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    protocol = out / "00/protocol_before_run.json"
    if protocol.exists() and json.loads(protocol.read_text())["signature"] != signature:
        raise ValueError("confirmation source changed")
    if not protocol.exists():
        for p in paths:
            dest = out / "00/code" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        (out / "00/documents.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in bank)
        )
        write_json(
            out / "00/reader_registry.json",
            json.loads((source / "00/reader_registry.json").read_text()),
        )
        write_json(
            protocol,
            {
                "signature": signature,
                "primary_manifest": sha(PRIMARY / "04/artifact_hashes.json"),
                "source_manifest": check_source(source),
                "materials": len(bank),
                "frozen_before_calls": True,
                "result_guided_followup": True,
                "changed": "explicit temporal scope, not target weights or thresholds",
            },
        )
    return source, bank


def main():
    """Observe a finite confirmation and expose ambiguous primary alternatives."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    source, bank = prepare(args.output)
    alternatives, scope_rows = [], []
    primary_bank = read_bank(PRIMARY / "00/documents.jsonl")
    for i, row in enumerate(primary_bank):
        data = json.loads((PRIMARY / f"01/parses/{i:04d}.json").read_text())
        rr = T.alternatives(data)
        alternatives.append({"material_id": row["material_id"], "alternatives": rr})
        scope_rows.append(
            {
                "material_id": row["material_id"],
                "kind": row["kind"],
                "ambiguous_report_roots": sum(not r["unique_scope"] for r in rr),
                "candidate_conditions": sum(len(r["candidate_nodes"]) for r in rr),
            }
        )
    write_json(args.output / "01/primary_scope_alternatives.json", alternatives)
    write_csv(args.output / "01/primary_scope_summary.csv", scope_rows)
    parsed = parse(args.output, bank)
    indexed = indices(args.output, source, bank, parsed)
    audit = coverage_audit(bank, indexed)
    write_csv(args.output / "01/coverage_audit.csv", audit)
    for reader in MODELS:
        encode(args.output, reader, args.device)
    _, _, _, contrasts, summaries = evaluate(args.output, source, bank, indexed, audit)
    summary = {
        "packet": "HM-09B-confirmation",
        "materials": len(bank),
        "complete": sum(r["complete"] for r in audit),
        "failures": [r for r in audit if not r["complete"]],
        "score_summary": summaries,
        "primary_ambiguous_report_roots": sum(
            r["ambiguous_report_roots"] for r in scope_rows
        ),
        "primary_candidate_conditions": sum(
            r["candidate_conditions"] for r in scope_rows
        ),
        "encoder_inputs_each_reader": len(
            read_bank(args.output / "00/material_bank.jsonl")
        ),
        "source_fits": 0,
        "pair_summary": [],
    }
    for reader in MODELS:
        for arm in ("observed", "declared"):
            # This matched subset shares the automatic coverage requirement.
            # The alternate audit separately checks all declared-reference rows.
            rr = [
                r
                for r in contrasts
                if r["reader"] == reader and r["arm"] == arm and r["complete_own_pair"]
            ]
            summary["pair_summary"].append(
                {
                    "reader": reader,
                    "arm": arm,
                    "complete_own_pairs": len(rr),
                    "same_word_pairs": sum(r["same_word_bag"] for r in rr),
                    "joint_delta_mean": sum(r["coexposure_delta"] for r in rr) / len(rr)
                    if rr
                    else None,
                    "hard_joint_positive_pairs": sum(
                        r["hard_coexposure_delta"] > 0 for r in rr
                    ),
                    "flat_sum_delta_max": max(
                        (abs(r[t + "_flat_sum_delta"]) for r in rr for t in S.TARGETS),
                        default=0,
                    ),
                }
            )
    check_source(source)
    write_json(args.output / "04/summary.json", summary)
    write_json(
        args.output / "04/artifact_hashes.json",
        {
            str(f.relative_to(args.output)): sha(f)
            for f in sorted(args.output.rglob("*"))
            if f.is_file() and f.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"failures", "score_summary"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
