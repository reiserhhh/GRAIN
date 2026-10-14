"""Fail-closed evidence checks, separate from scientific interpretation.

These checks establish provenance and explicit review, not theorem correctness.
An evaluator still has to inspect the cited calculations and counterexamples.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def verify_evidence(entries: object, root: Path) -> list[str]:
    """Check nonempty, root-contained file references against frozen SHA256s."""
    if not isinstance(entries, list) or not entries:
        return ["missing evidence"]
    errors = []
    root = root.resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("invalid evidence entry")
            continue
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not relative:
            errors.append("missing evidence path")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            errors.append(f"outside evidence root: {relative}")
        elif not path.is_file():
            errors.append(f"missing file: {relative}")
        elif not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"missing SHA256: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"SHA256 mismatch: {relative}")
    return errors


def assess_completion(progress: dict, crosswalk: list[dict], root: Path) -> dict:
    """Require reviewed requirements and human-design preparations, not card counts.

    Every source requirement must have its own decision, reasoning and immutable
    evidence. Human recruitment remains pending independently of design readiness.
    """
    reviews = progress.get("requirement_reviews", {})
    resolved = []
    open_items = []
    global_errors = []
    ids = [row["requirement"] for row in crosswalk]
    if len(ids) != len(set(ids)):
        global_errors.append("duplicate source requirements")
    computational = [
        row for row in crosswalk if row["scope"] == "COMPUTATIONAL_QUESTION"
    ]
    human = [row for row in crosswalk if row["scope"] == "FUTURE_HUMAN_DESIGN_ONLY"]
    if len(computational) != 44 or len(human) != 6 or len(ids) != 50:
        global_errors.append(
            "expected 44 computational and 6 human-design requirements"
        )
    unknown = set(reviews) - set(ids)
    if unknown:
        global_errors.append(f"unknown reviews: {sorted(unknown)}")
    for row in crosswalk:
        rid = row["requirement"]
        review = reviews.get(rid, {})
        errors = []
        if not isinstance(review, dict):
            review = {}
        allowed = (
            {"SUPPORTED", "COUNTEREXAMPLE", "KNOWN_METHOD_SUFFICIENT", "NO_INCREMENT"}
            if row["scope"] == "COMPUTATIONAL_QUESTION"
            else {"DESIGN_READY_HUMAN_PENDING"}
        )
        if review.get("science_status") not in allowed:
            errors.append("unresolved scientific decision or design preparation")
        for key in ("implementation_status", "protocol_status"):
            if review.get(key) != "PASS":
                errors.append(f"{key} not PASS")
        for key in ("reason", "reviewer", "reviewed_at", "limitations"):
            if not isinstance(review.get(key), str) or not review[key].strip():
                errors.append(f"missing {key}")
        required_cards = set(row["atomic_tasks"].split(";"))
        if not required_cards.issubset(set(review.get("reviewed_tasks", []))):
            errors.append("source-to-task coverage incomplete")
        errors.extend(verify_evidence(review.get("evidence"), root))
        if errors:
            open_items.append(
                {"requirement": rid, "scope": row["scope"], "reasons": errors}
            )
        else:
            resolved.append(rid)
    closed = not global_errors and not open_items
    return {
        "status": "COMPUTATIONAL_PLAN_CLOSED_HUMAN_VALIDATION_PENDING"
        if closed
        else "OPEN",
        "evidence_check_only": True,
        "resolved_requirements": resolved,
        "open_requirements": open_items,
        "errors": global_errors,
        "human_validation_completed": False,
        "boundary": "Evidence hashes and explicit reviews do not independently prove scientific correctness.",
    }
