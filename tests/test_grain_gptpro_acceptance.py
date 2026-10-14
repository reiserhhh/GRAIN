"""Adverse checks for premature research-plan closure."""

import hashlib

from grain_research.gptpro_foundation_v3.acceptance import (
    assess_completion,
    verify_evidence,
)


def crosswalk():
    """Build a complete-shaped fixture without claiming actual research results."""
    return [
        {"requirement": f"{prefix}{i:02d}", "scope": scope, "atomic_tasks": "F00-01"}
        for prefix, count, scope in (
            ("R", 44, "COMPUTATIONAL_QUESTION"),
            ("H", 6, "FUTURE_HUMAN_DESIGN_ONLY"),
        )
        for i in range(1, count + 1)
    ]


def test_empty_progress_cannot_close(tmp_path):
    result = assess_completion({}, crosswalk(), tmp_path)
    assert result["status"] == "OPEN"
    assert len(result["open_requirements"]) == 50


def test_hundred_pass_cards_cannot_close(tmp_path):
    result = assess_completion(
        {"completed_tasks": {str(i): "PASS" for i in range(100)}}, crosswalk(), tmp_path
    )
    assert result["status"] == "OPEN"


def test_missing_crosswalk_cannot_close(tmp_path):
    result = assess_completion({}, [], tmp_path)
    assert result["status"] == "OPEN"
    assert result["errors"]


def test_evidence_hash_and_containment(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"finite_identity_error": 0}')
    entry = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert verify_evidence([entry], tmp_path) == []
    path.write_text("modified")
    assert "SHA256 mismatch" in verify_evidence([entry], tmp_path)[0]
    assert "outside" in verify_evidence([{**entry, "path": "../outside"}], tmp_path)[0]
    assert verify_evidence([], tmp_path)


def test_requirement_review_gate_and_mutations(tmp_path):
    path = tmp_path / "fixture_only.txt"
    path.write_text("Test fixture, not scientific evidence")
    evidence = [
        {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    ]
    rows = crosswalk()
    reviews = {
        row["requirement"]: {
            "science_status": "SUPPORTED"
            if row["requirement"].startswith("R")
            else "DESIGN_READY_HUMAN_PENDING",
            "implementation_status": "PASS",
            "protocol_status": "PASS",
            "reason": "fixture",
            "reviewer": "test",
            "reviewed_at": "test",
            "limitations": "fixture only",
            "reviewed_tasks": ["F00-01"],
            "evidence": evidence,
        }
        for row in rows
    }
    progress = {"requirement_reviews": reviews}
    assert assess_completion(progress, rows, tmp_path)["status"] != "OPEN"
    reviews["R01"]["science_status"] = "INCONCLUSIVE"
    assert assess_completion(progress, rows, tmp_path)["status"] == "OPEN"
    reviews["R01"]["science_status"] = "SUPPORTED"
    reviews["R01"]["reviewed_tasks"] = []
    assert assess_completion(progress, rows, tmp_path)["status"] == "OPEN"
    reviews["R01"]["reviewed_tasks"] = ["F00-01"]
    path.write_text("tampered")
    assert len(assess_completion(progress, rows, tmp_path)["open_requirements"]) == 50
