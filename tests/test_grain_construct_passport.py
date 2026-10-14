"""Construct passport contracts: completeness, bundle binding, and the claim-ladder rule."""

import json
from pathlib import Path

import pytest

from scripts.build_grain_construct_passports import (
    INDEX,
    OUT,
    ROOT,
    SCHEMA,
    claim_level,
    matrices,
    modules,
)

SCHEMA_OBJ = json.loads(SCHEMA.read_text())


@pytest.mark.parametrize("target", ["T1", "T2", "T3"])
def test_passport_complete_and_bound(target):
    p = json.loads((OUT / f"{target}.passport.json").read_text())
    assert set(p["passport_modules"]) == set(SCHEMA_OBJ["passport_modules"])
    assert len(SCHEMA_OBJ["passport_modules"]) == 17
    assert set(p["construct_contract"]) == set(SCHEMA_OBJ["construct_contract_fields"])
    assert len(SCHEMA_OBJ["construct_contract_fields"]) == 12
    assert set(p["evidence_matrix_P02"]) == set(SCHEMA_OBJ["evidence_matrix_P02"])
    assert set(p["validation_matrix_P03"]) == set(SCHEMA_OBJ["validation_matrix_P03"])
    assert set(p["measurement_tuple"]) == set(SCHEMA_OBJ["measurement_tuple"])
    statuses = set(SCHEMA_OBJ["statuses"])
    for block in ("passport_modules", "construct_contract", "evidence_matrix_P02", "validation_matrix_P03"):
        for v in p[block].values():
            assert v["status"] in statuses
            assert all((ROOT / e).exists() for e in v["evidence"])
    index = json.loads(Path(INDEX).read_text())
    assert p["bound_bundles"]["bundle_sha256"] == sorted(b["sha256"] for b in index if b["target"] == target)


def test_claim_ladder_requires_real_evidence():
    m = modules("T1")
    _, p03 = matrices(m)
    assert claim_level(SCHEMA_OBJ, m, p03)[0] == "BELOW_L0"
    for key in ("representation", "uncertainty"):
        m[key] = dict(m[key], status="EVIDENCED_REAL")
    for key in ("structural_reliability", "person_reliability"):
        p03[key] = dict(p03[key], status="EVIDENCED_REAL")
    assert claim_level(SCHEMA_OBJ, m, p03)[0] == "L1"
    p03["person_reliability"] = dict(p03["person_reliability"], status="EVIDENCED_SIM")
    assert claim_level(SCHEMA_OBJ, m, p03)[0] == "L0"
