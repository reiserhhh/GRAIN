#!/usr/bin/env python3
"""Tests of actual source-derived T1 construction and rejection boundaries."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f16_instrument import (
    CONDITIONS,
    READER_HASH,
    build_instrument,
    cold_score,
    raw_t1,
)


@pytest.fixture
def instrument(tmp_path):
    """Create nonconstant source reference without using application truth."""
    model = {"mu": [0.0, 0.0, 0.0, 0.0], "Sigma": np.eye(4).tolist(), "sigma2": 0.25}
    rows = [
        {
            "author_id": str(u),
            "event_id": f"{u}-{j}",
            "world": "L",
            "role": "source",
            "z": z,
            "y": u * z,
        }
        for u in range(3)
        for j, z in enumerate(CONDITIONS)
    ]
    (tmp_path / "source").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "models").write_text(json.dumps({"four_cell": model}))
    manifest = {
        k: {
            "path": str(tmp_path / k),
            "sha256": hashlib.sha256((tmp_path / k).read_bytes()).hexdigest(),
        }
        for k in ("source", "models")
    }
    bundle = build_instrument(
        {"target_id": "T1", "world": "L", "target": "m(+1)-m(-1)"}, manifest
    )
    return bundle, {
        "world_contract": bundle["world_contract"],
        "reader_hash": READER_HASH,
        "records": rows[4:8],
    }


def test_reference_and_support(instrument):
    bundle, raw = instrument
    result = cold_score(raw, bundle)
    assert result["support"] == "observed"
    assert result["calibration_check"] == "NOT_EVALUATED"
    assert result["reference_score"] == pytest.approx(0.0, abs=1e-14)
    assert cold_score(dict(raw, records=[]), bundle)["support"] == "model_borrowed"
    assert (
        cold_score(dict(raw, records=raw["records"][:1]), bundle)["support"]
        == "model_borrowed"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_sigma", 0),
        ("reader_hash", "bad"),
        ("world_contract", "F09"),
        ("scorer_code_hash", "bad"),
        ("formula_version", "unknown"),
    ],
)
def test_reject_bundle(instrument, field, value):
    bundle, raw = instrument
    with pytest.raises(ValueError):
        cold_score(raw, dict(bundle, **{field: value}))


def test_duplicate_and_covariance(instrument):
    bundle, raw = instrument
    with pytest.raises(ValueError):
        cold_score(dict(raw, records=raw["records"] * 2), bundle)
    model = copy.deepcopy(bundle["model"])
    model["Sigma"][0][0] = -1
    with pytest.raises(ValueError):
        raw_t1([], model)


def test_extreme_stranger_does_not_change_reference(instrument):
    bundle, raw = instrument
    before = cold_score(raw, bundle)
    stranger = copy.deepcopy(raw)
    for row in stranger["records"]:
        row["author_id"] = "stranger"
        row["y"] *= 10000
    batch = [cold_score(r, bundle) for r in (stranger, raw)]
    assert before == batch[1] == cold_score(raw, bundle)


def test_cold_cli_requires_trusted_hash(instrument, tmp_path):
    """A separate scorer process rejects altered bundle bytes before scoring."""
    bundle, raw = instrument
    payload = json.dumps(bundle).encode()
    (tmp_path / "bundle.json").write_bytes(payload)
    (tmp_path / "input.jsonl").write_text(json.dumps(raw) + "\n")
    command = [
        sys.executable,
        "-m",
        "grain_research.gptpro_foundation_v3.f16_instrument",
        "--bundle",
        "bundle.json",
        "--sha256",
        "bad",
        "--input",
        "input.jsonl",
        "--output",
        "scores.jsonl",
    ]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    rejected = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert rejected.returncode != 0
    assert "untrusted bundle hash" in rejected.stderr
    assert not (tmp_path / "scores.jsonl").exists()
    # Locate the argument explicitly so additions to the CLI cannot shift the test.
    command[command.index("--sha256") + 1] = hashlib.sha256(payload).hexdigest()
    accepted = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads((tmp_path / "scores.jsonl").read_text()) == cold_score(
        raw, bundle
    )
