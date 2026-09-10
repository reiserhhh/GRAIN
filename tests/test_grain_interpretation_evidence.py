#!/usr/bin/env python3
"""Synthetic-only tests for auditable, non-mutating research interpretation."""

import copy
import hashlib

import pytest
from jsonschema import ValidationError

from grain_research.vector_mechanisms.interpretation_evidence import validate_packet


def example():
    """Create fictional local text and one immutable numerical record."""
    text = "I asked for a quieter room."
    evidence = {
        "id": "e0",
        "source_id": "fictional",
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "start": 0,
        "end": len(text),
        "quote": text,
        "speaker": "speaker0",
        "target": "reported_other",
        "event_kind": "reported_event",
    }
    measurement = {
        "id": "m0",
        "value": 0.2,
        "operator_sha256": "a" * 64,
        "reference": "synthetic-test",
        "query": "test-only",
        "unit": "fictional-comment",
        "kind": "grain_score",
        "evidence_ids": ["e0"],
    }
    packet = {
        "version": "1",
        "research_only": True,
        "mode": "evidence_rendering",
        "episode_id": "episode0",
        "condition": "fictional test",
        "evidence": [evidence],
        "measurements": [copy.deepcopy(measurement)],
        "candidates": [
            {
                "id": "c0",
                "registry_id": None,
                "hypothesis": "The speaker reports making a request.",
                "evidence_ids": ["e0"],
                "measurement_ids": ["m0"],
                "alternative": "This could be an isolated practical preference.",
                "missing_context": ["Other person's response"],
                "support": "tentative",
                "human_status": "not_reviewed",
            }
        ],
    }
    return packet, {"fictional": text}, {"m0": measurement}


def test_bound_packet_does_not_claim_psychological_validity():
    """A valid address and score are accounting checks, not clinical validation."""
    result = validate_packet(*example())
    assert result == {"valid_bindings": True, "psychological_validity_tested": False}


@pytest.mark.parametrize(
    "field,value", [("quote", "invented"), ("start", 2), ("source_sha256", "b" * 64)]
)
def test_reject_changed_evidence(field, value):
    """A renderer cannot invent or relocate a quoted source."""
    packet, sources, measures = example()
    packet["evidence"][0][field] = value
    with pytest.raises(ValueError):
        validate_packet(packet, sources, measures)


@pytest.mark.parametrize(
    "field,value",
    [("value", 0.21), ("reference", "changed"), ("kind", "native_intervention")],
)
def test_reject_changed_fixed_measurement(field, value):
    """Values, referents and functional kinds all belong to the fixed ledger."""
    packet, sources, measures = example()
    packet["measurements"][0][field] = value
    with pytest.raises(ValueError):
        validate_packet(packet, sources, measures)


def test_reject_unbound_candidate():
    """A candidate cannot cite a nonexistent measurement."""
    packet, sources, measures = example()
    packet["candidates"][0]["measurement_ids"] = ["unknown"]
    with pytest.raises(ValueError):
        validate_packet(packet, sources, measures)


def test_registry_and_empty_candidates():
    """No interpretation is allowed, but an unsupplied rubric code is not."""
    packet, sources, measures = example()
    packet["candidates"][0]["registry_id"] = "unsupported-code"
    with pytest.raises(ValueError):
        validate_packet(packet, sources, measures)
    packet["candidates"] = []
    assert validate_packet(packet, sources, measures)["valid_bindings"]


def test_reject_diagnostic_field_and_clinical_mode():
    """The data schema exposes no clinical mode or diagnosis field."""
    packet, sources, measures = example()
    packet["diagnosis"] = "not allowed"
    with pytest.raises(ValidationError):
        validate_packet(packet, sources, measures)
    del packet["diagnosis"]
    packet["research_only"] = False
    with pytest.raises(ValidationError):
        validate_packet(packet, sources, measures)
