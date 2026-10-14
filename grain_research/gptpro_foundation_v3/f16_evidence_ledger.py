#!/usr/bin/env python3
"""Versioned numerical evidence ledger for the actual frozen T1/T2/T3 targets.

Qualification here is supplied metadata, not inferred natural-language meaning.
No binary pole probability is converted to a Gaussian response or state.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from . import f16_cutoff as clock
from . import f16_t1_source as t1

VERSION = "F16_numeric_evidence_ledger_v1"
QUALIFICATIONS = {
    "actual": True,
    "negated": True,
    "quoted": False,
    "hypothetical": False,
    "unknown": None,
}


def code_hash():
    """Pin this evidence law separately from unchanged numerical scorer hashes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_bundle(base):
    """Wrap a trusted target without fitting, renormalizing or changing its units."""
    if base["target_id"] == "T1":
        t1.verify_bundle(base)
        scorer = copy.deepcopy(base)
    else:
        scorer = clock.build_adapter(base)
    return {
        "schema_version": VERSION,
        "ledger_code_hash": code_hash(),
        "target_id": base["target_id"],
        "reader_hash": base["reader_hash"],
        "world": "L" if base["target_id"] == "T1" else "core",
        "scorer": scorer,
        "scorer_hash": clock.object_hash(scorer),
        "qualification_origin": "explicit_numeric_evidence_metadata_not_text_inference",
        "history_model": "one_current_numeric_report_per_event;alias_summary_add_no_likelihood",
        "negated_rule": "F08 corrected actual report; the supplied corrected value is not sign-flipped",
    }


def verify_bundle(bundle):
    """Reject changed ledger, reader, observation world or embedded scorer."""
    if (
        bundle.get("schema_version") != VERSION
        or bundle.get("ledger_code_hash") != code_hash()
        or bundle.get("scorer_hash") != clock.object_hash(bundle["scorer"])
    ):
        raise ValueError("ledger/scorer identity mismatch")
    target = bundle["target_id"]
    if target == "T1":
        t1.verify_bundle(bundle["scorer"])
        base = bundle["scorer"]
    elif target in ("T2", "T3"):
        clock.verify_adapter(bundle["scorer"])
        base = bundle["scorer"]["base_bundle"]
    else:
        raise ValueError("unsupported ledger target")
    if (
        base["target_id"] != target
        or bundle["reader_hash"] != base["reader_hash"]
        or bundle["world"] != ("L" if target == "T1" else "core")
    ):
        raise ValueError("target/reader/world mismatch")


def _observation(row, raw, bundle):
    """Validate even excluded numeric payloads; unknown fields are never features."""
    if not isinstance(row, dict) or row.get("author_id") != raw["author_id"]:
        raise ValueError("observation must name the declared measured author")
    if not isinstance(row.get("event_id"), str) or not row["event_id"]:
        raise ValueError("explicit event identity required")
    if type(row.get("y")) not in (int, float) or not math.isfinite(row["y"]):
        raise ValueError("finite numeric response required")
    if bundle["target_id"] == "T1":
        t1.sufficient([row])
        if row["t"] > raw["target_time"]:
            raise ValueError("future observation")
    else:
        clock.validate_input(
            {
                "world": raw["world"],
                "reader_hash": raw["reader_hash"],
                "target_time": raw["target_time"],
                "records": [row],
            },
            bundle["scorer"]["base_bundle"],
        )


def replay(raw, bundle):
    """Replay declared claim versions while preserving event and report clocks."""
    if not isinstance(raw, dict) or set(raw) != {
        "author_id",
        "world",
        "reader_hash",
        "target_time",
        "history",
    }:
        raise ValueError("invalid ledger envelope")
    if not isinstance(raw["author_id"], str) or not raw["author_id"]:
        raise ValueError("explicit author required even for empty evidence")
    if raw["world"] != bundle["world"] or raw["reader_hash"] != bundle["reader_hash"]:
        raise ValueError("wrong ledger world or reader")
    cutoff = raw["target_time"]
    if type(cutoff) is not int or cutoff < 0 or not isinstance(raw["history"], list):
        raise ValueError("nonnegative target clock and history list required")
    seen, active, events, annotations, operations = {}, {}, {}, [], []
    previous = None
    panel_context = None
    for original in raw["history"]:
        if not isinstance(original, dict):
            raise ValueError("each history operation must be an object")  # noqa: TRY004
        record = copy.deepcopy(original)
        common = {"record_id", "op", "recorded_at", "material_ref"}
        kind = record.get("op")
        fields = common | (
            {"observation", "qualification"} if kind in ("new", "revise") else set()
        )
        if kind in ("revise", "alias", "retract"):
            fields.add("parent_id")
        if kind == "summary":
            fields.add("parents")
        if (
            kind not in ("new", "revise", "alias", "retract", "summary")
            or set(record) != fields
        ):
            raise ValueError(
                "unsupported history operation/fields; repeat requires a report-noise model, use alias only for an actual copy"
            )
        if "parent_id" in record and not isinstance(record["parent_id"], str):
            raise ValueError("parent identity must be a string")
        for key in ("record_id", "material_ref"):
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError("record identity and material address required")
        identity = record["record_id"]
        if identity in seen:
            raise ValueError("duplicate record identity")
        arrived = record["recorded_at"]
        if (
            type(arrived) is not int
            or arrived < 0
            or arrived > cutoff
            or (previous is not None and arrived < previous)
        ):
            raise ValueError(
                "report availability must be ordered and not after scoring cutoff"
            )
        previous = arrived
        if kind in ("new", "revise"):
            row = record["observation"]
            _observation(row, raw, bundle)
            current_context = (row["role"], row.get("episode_id"))
            if panel_context is not None and current_context != panel_context:
                raise ValueError(
                    "mixed panel roles or episodes, even in excluded reports"
                )
            panel_context = current_context
            if (
                not isinstance(record["qualification"], str)
                or record["qualification"] not in QUALIFICATIONS
            ):
                raise ValueError("unknown qualification encoding")
            if row["t"] > arrived:
                raise ValueError("report cannot precede its event")
            event = row["event_id"]
            context = {k: v for k, v in row.items() if k != "y"}
            if kind == "new" and event in events:
                raise ValueError(
                    "existing event cannot be counted as independent new evidence"
                )
            if kind == "revise":
                parent = active.get(record["parent_id"])
                if parent is None or parent["observation"]["event_id"] != event:
                    raise ValueError(
                        "revision must replace an active same-event report"
                    )
                if context != events[event]:
                    raise ValueError(
                        "revision cannot silently change event condition/time/identity"
                    )
                del active[record["parent_id"]]
            events[event] = context
            active[identity] = record
        elif kind in ("alias", "retract"):
            if record["parent_id"] not in active:
                raise ValueError("parent must be an active report")
            if kind == "retract":
                del active[record["parent_id"]]
            else:
                annotations.append(record)
        else:
            parents = record["parents"]
            if (
                not isinstance(parents, list)
                or not parents
                or not all(isinstance(p, str) for p in parents)
                or len(parents) != len(set(parents))
                or not set(parents) <= set(active)
            ):
                raise ValueError("summary must cite distinct active reports")
            annotations.append(record)
        seen[identity] = record
        operations.append(
            {
                "record_id": identity,
                "op": kind,
                "recorded_at": arrived,
                "material_ref": record["material_ref"],
            }
        )
    accepted, excluded = [], []
    for identity, record in active.items():
        if QUALIFICATIONS[record["qualification"]] is True:
            accepted.append(record)
        else:
            excluded.append(
                {
                    "record_id": identity,
                    "event_id": record["observation"]["event_id"],
                    "qualification": record["qualification"],
                    "material_ref": record["material_ref"],
                    "reason": "unresolved_not_zero"
                    if record["qualification"] == "unknown"
                    else "not_an_actual_self_observation",
                }
            )
    accepted.sort(key=lambda r: r["observation"]["t"])
    rows = [r["observation"] for r in accepted]
    if bundle["target_id"] == "T1":
        t1.sufficient(rows)
    else:
        clock.validate_input(
            {
                "world": raw["world"],
                "reader_hash": raw["reader_hash"],
                "target_time": cutoff,
                "records": rows,
            },
            bundle["scorer"]["base_bundle"],
        )
    stale = [
        r["record_id"]
        for r in annotations
        if not set(r.get("parents", [r.get("parent_id")])) <= set(active)
    ]
    return {
        "records": rows,
        "accepted_record_ids": [r["record_id"] for r in accepted],
        "accepted_material_refs": [r["material_ref"] for r in accepted],
        "excluded_active": excluded,
        "inactive_report_ids": [
            k
            for k, r in seen.items()
            if r["op"] in ("new", "revise") and k not in active
        ],
        "stale_derived": stale,
        "annotation_ids": [r["record_id"] for r in annotations],
        "operations": operations,
        "reported_events": len(events),
        "active_reports": len(active),
        "eligible_events": len(rows),
        "target_time": cutoff,
        "qualification_origin": bundle["qualification_origin"],
    }


def score_evidence(raw, bundle, *, audit_callback=None):
    """Bind qualified current report versions to the matching frozen target."""
    verify_bundle(bundle)
    ledger = replay(raw, bundle)
    if bundle["target_id"] == "T1":
        result = t1.cold_score(
            {
                "world_contract": bundle["scorer"]["world_contract"],
                "reader_hash": bundle["reader_hash"],
                "records": ledger["records"],
            },
            bundle["scorer"],
        )
    else:
        result = clock.cold_score(
            {
                "world": bundle["world"],
                "reader_hash": bundle["reader_hash"],
                "target_time": raw["target_time"],
                "records": ledger["records"],
            },
            bundle["scorer"],
            audit_callback=audit_callback,
        )
    return {
        "schema_version": VERSION,
        "target_id": bundle["target_id"],
        "ledger": ledger,
        "result": result,
        "measured_self_evidence_present": bool(ledger["records"]),
        "scope": "SIM numeric reports and supplied eligibility; no natural-language reader or new psychological validation",
    }


def main():
    """Cold evidence API with an externally authenticated self-contained bundle."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sha256", required=True)
    args = p.parse_args()
    if hashlib.sha256(args.bundle.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("untrusted ledger bundle")
    bundle = json.loads(args.bundle.read_text())
    with args.input.open() as inputs, args.output.open("x") as outputs:
        for line in inputs:
            outputs.write(
                json.dumps(
                    score_evidence(json.loads(line), bundle),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            outputs.flush()


if __name__ == "__main__":
    main()
