"""Fixed semantic batteries for generated and restricted-candidate readouts."""

from __future__ import annotations

import re

from .controls import SYSTEM, build_controls, control_user_message
from .independent_access import ACCESS_SYSTEM, answer_code_controls


def parse_binary_answer(text):
    """Accept only a complete binary reply, never fish an answer from an essay."""
    match = re.fullmatch(r"\s*(Yes|No)\s*[.!]?\s*", str(text), flags=re.IGNORECASE)
    return None if match is None else int(match.group(1).lower() == "yes")


def leading_binary_diagnostic(text):
    """Read only a leading binary sentence for post-hoc format diagnosis, not gates."""
    match = re.match(r"^\s*(Yes|No)(?:[.!](?:\s|$)|\s*$)", str(text), re.IGNORECASE)
    return None if match is None else int(match.group(1).lower() == "yes")


def capability_cases():
    """Retain both original batteries and eight outcome-independent repeats."""
    rows = []
    for c in answer_code_controls():
        code = "swapped" if c["swapped"] else "normal"
        rows.append(
            {
                "case_id": f"code_{c['scenario']}_{int(c['positive_evidence'])}_{code}",
                "battery": "answer_code",
                "group_id": f"code_{c['scenario']}",
                "family": code,
                "variant": int(c["positive_evidence"]),
                "split": "fixed_control",
                "expected_yes": int(c["expected_yes"]),
                "system": ACCESS_SYSTEM,
                "user": c["user"],
            }
        )
    for c in build_controls():
        rows.append(
            {
                k: c[k]
                for k in (
                    "case_id",
                    "group_id",
                    "family",
                    "variant",
                    "split",
                    "expected_yes",
                )
            }
            | {
                "battery": "semantic_fact",
                "system": SYSTEM,
                "user": control_user_message(c),
            }
        )
    repeats = rows[:4] + [
        r for r in rows if r["group_id"] == "scenario_00" and r["variant"] == 0
    ]
    return [r | {"repeat": False} for r in rows] + [
        r | {"repeat": True} for r in repeats
    ]
