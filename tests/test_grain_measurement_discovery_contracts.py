"""Synthetic tests for the GP00 data and target contracts."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contracts", ROOT / "grain_research/measurement_discovery/contracts.py")
contracts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contracts)


def row(event="e1", author="a1"):
    return dict(
        zip(
            contracts.RECORD_FIELDS,
            [author, event, "episode", 0, 1, None, "unknown", 1, 1, "text", "reader", "source", "assigned"],
            strict=True,
        )
    )


def test_record_and_author_split():
    rows = [row("e1", "a1"), row("e2", "a2")]
    assert contracts.validate_records(rows)["events"] == 2
    assert contracts.validate_split(rows, {"a1": "source", "a2": "application"})["source"] == 1


def test_conflicting_event_rejected():
    a, b = row(), row()
    b["observation"] = 0
    with pytest.raises(ValueError):
        contracts.validate_records([a, b])


def test_cross_author_split_rejected():
    with pytest.raises(ValueError):
        contracts.validate_split([row()], {"a1": "unknown"})


def test_target_contract_requires_all_fields():
    values = dict(
        target_id="t1", response_range="0..1", reference_population="source",
        condition_universe="two cells", observation_policy="next only", unit="probability",
        allowed_interpretation="conditional response",
    )
    assert contracts.target_contract(**values)["target_id"] == "t1"
    with pytest.raises(ValueError):
        contracts.target_contract(**{**values, "target_id": ""})
