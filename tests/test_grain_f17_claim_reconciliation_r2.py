"""Packet 30 A tests: the union supplement must keep every current ledger gap (incl. audited rows) and sentinel clauses."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "results/grain_gptpro_foundation_v3/F17/claim_level_reconciliation_r1/clause_reconciliation.csv"
R2 = ROOT / "results/grain_gptpro_foundation_v3/F17/claim_level_reconciliation_r2/clause_reconciliation_union.csv"
LEDGER = Path("/Volumes/mobile3/projects/project persona/reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv")


def eff():
    return [r for r in csv.DictReader(R2.open()) if r["r2_disposition"] != "REPLACED_BY_SPLIT"]


def test_every_ledger_row_has_current_gap_clause_including_audited_rows():
    have = {r["requirement"] for r in eff() if r["clause_type"] == "current_ledger_gap"}
    assert have == {r["requirement"] for r in csv.DictReader(LEDGER.open())}


def test_no_r1_row_lost():
    r2ids = {r["clause_id"] for r in csv.DictReader(R2.open())}
    assert {r["clause_id"] for r in csv.DictReader(R1.open())} <= r2ids


def test_sentinel_r12_and_r26_clauses_present():
    r12 = " ".join((r["part"] + " " + r["effective_action"]).lower() for r in eff() if r["requirement"] == "R12")
    assert "informative-context" in r12 and "real-token" in r12
    assert any(r["requirement"] == "R12" and r["effective_status"] == "EXECUTABLE_CPU_CACHE" for r in eff())
    r26 = [r for r in eff() if r["requirement"] == "R26" and "ALBA" in (r["part"] + r["effective_action"])]
    assert r26 and all(r["effective_status"] not in ("RESOLVED_FINITE_WITH_AUTHORITY", "HUMAN_VALIDATION") for r in r26)


def test_mixed_rows_split_into_executable_and_human_or_model_parts():
    for req in ("H05", "H06", "R17", "R20", "R13"):
        st = {r["effective_status"] for r in eff() if r["requirement"] == req and r["clause_type"] == "current_ledger_gap"}
        assert "EXECUTABLE_CPU_CACHE" in st and (st & {"HUMAN_VALIDATION", "BLOCKED_MODEL_BUDGET"}), req


def test_ledger_status_counts_preserved():
    c = Counter(r["science_resolution"] for r in csv.DictReader(LEDGER.open()))
    assert c["RESOLVED_SOURCE_DEMANDS_MET"] == 21 and c["HUMAN_PENDING"] == 6 and c["NOT_RESOLVED"] == 23
