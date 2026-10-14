"""Packet 29 claim-level reconciliation tests: coverage, pointers, self-reference, relabeling, status invariance.
These test bookkeeping integrity only; they are never scientific support for any clause."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/claim_level_reconciliation_r1"
PARENT = Path("/Volumes/mobile3/projects/project persona")
LEDGER = PARENT / "reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv"
spec = importlib.util.spec_from_file_location("b", ROOT / "scripts/build_grain_f17_claim_reconciliation.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


def rows():
    return list(csv.DictReader((OUT / "clause_reconciliation.csv").open()))


def test_every_audit_clause_and_every_ledger_row_is_present():
    ids = {r["clause_id"] for r in rows()}
    for a in b.parse_audits():
        assert any(i == a["audit_id"] or (i.startswith(a["audit_id"]) and len(i) == len(a["audit_id"]) + 1 and i[-1].isalpha()) for i in ids), a["audit_id"]
    reqs = {r["requirement"] for r in rows()}
    assert reqs == {r["requirement"] for r in csv.DictReader(LEDGER.open())} and len(reqs) == 50


def test_status_vocabulary_and_root_rules():
    for r in rows():
        assert r["status"] in b.STATUSES
        roots = [x for x in r["evidence_roots"].split(";") if x]
        if r["status"] in b.NEED_ROOT:
            assert roots, r["clause_id"]
        if r["status"] == b.RES:
            assert r["authority"].strip(), r["clause_id"]
        assert b.SELF not in roots and not any("claim_level_reconciliation" in x for x in roots)


def test_pointers_live_and_registered():
    chk = json.loads((OUT / "evidence_check.json").read_text())
    for r in rows():
        for root in [x for x in r["evidence_roots"].split(";") if x]:
            c = chk[root]
            assert c["registered"] and c["kind"] == "CURRENT_REVISION" and c["decision_exists"], root
        assert r["pointer_ok"] in ("True", "", "None")


def test_human_and_optional_labels_are_justified():
    for r in rows():
        if r["status"] == b.HUM:
            assert "human" in r["resource"].lower() and r["reason"].strip(), r["clause_id"]
            assert "not authorized" not in r["reason"].lower(), r["clause_id"]
        if r["status"] == b.OPT:
            assert r["evidence_pointer"].strip() and ("scope" in r["reason"].lower() or "optional" in r["reason"].lower()), r["clause_id"]


def test_no_generic_see_audit_only_action():
    for r in rows():
        if r["status"] not in (b.RES, b.SUP):
            act = r["remaining_action"].strip()
            assert act and "see audit" not in act.lower() and "see AUDIT" not in act, r["clause_id"]


def test_cards_all_accounted_via_crosswalk():
    cards = list(csv.DictReader((OUT / "coverage_100_cards.csv").open()))
    assert len(cards) == 100 and all(c["in_crosswalk"] == "True" for c in cards)


def test_ledger_statuses_unchanged_by_packet29():
    before = {r["requirement"]: r for r in csv.DictReader(open("/Users/huokean/.claude/jobs/582bb33b/tmp/p29/ledger_before.csv"))} \
        if Path("/Users/huokean/.claude/jobs/582bb33b/tmp/p29/ledger_before.csv").exists() else None
    after = {r["requirement"]: r for r in csv.DictReader(LEDGER.open())}
    snap = json.loads((OUT / "ledger_status_snapshot.json").read_text())
    for req, r in after.items():
        for k in ("implementation", "protocol", "science", "science_resolution", "acceptance_decision"):
            assert r[k] == snap[req][k], (req, k)
            if before:
                assert r[k] == before[req][k], (req, k)
    from collections import Counter
    c = Counter(r["science_resolution"] for r in after.values())
    assert c["RESOLVED_SOURCE_DEMANDS_MET"] == 21 and c["HUMAN_PENDING"] == 6
