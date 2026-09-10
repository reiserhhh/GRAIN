#!/usr/bin/env python3
"""Bind interpretation packets to immutable local evidence, not clinical truth."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "configs/grain/interpretation_evidence.schema.json"
)


def validate_packet(packet, sources, measurements, candidate_registry=()):
    """Validate local Unicode offsets and exact precomputed measurement bindings.

    sources is a local source_id -> text map; measurements is a trusted map of
    complete measurement records. No text is sent anywhere. Schema compliance
    cannot establish psychological validity or verify narrative reasoning.
    """
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(packet)
    for section in ("evidence", "measurements", "candidates"):
        ids = [item["id"] for item in packet[section]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs in " + section)
    evidence_ids = {item["id"] for item in packet["evidence"]}
    measurement_ids = {item["id"] for item in packet["measurements"]}
    for row in packet["evidence"]:
        text = sources[row["source_id"]]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["source_sha256"]:
            raise ValueError("Changed evidence source")
        if not 0 <= row["start"] < row["end"] <= len(text):
            raise ValueError("Invalid Unicode-character interval")
        if text[row["start"] : row["end"]] != row["quote"]:
            raise ValueError("Quote does not match its source span")
    for row in packet["measurements"]:
        if not math.isfinite(row["value"]):
            raise ValueError("Nonfinite measurement")
        if row != measurements[row["id"]]:
            raise ValueError("Renderer changed a fixed measurement or its binding")
        if not set(row["evidence_ids"]).issubset(evidence_ids):
            raise ValueError("Measurement refers to missing evidence")
    for row in packet["candidates"]:
        if not set(row["evidence_ids"]).issubset(evidence_ids):
            raise ValueError("Candidate refers to missing evidence")
        if not set(row["measurement_ids"]).issubset(measurement_ids):
            raise ValueError("Candidate refers to missing measurements")
        if (
            row["registry_id"] is not None
            and row["registry_id"] not in candidate_registry
        ):
            raise ValueError("Unknown interpretation rubric code")
    return {"valid_bindings": True, "psychological_validity_tested": False}
