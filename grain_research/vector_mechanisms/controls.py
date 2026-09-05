"""Known semantic facts for instrument validation, not human trait labels."""

from __future__ import annotations

import hashlib
import json

SCENARIOS = (
    ("workshop", "join the discussion", "joined the discussion"),
    ("office", "organize the files", "organized the files"),
    ("trip", "try the new route", "tried the new route"),
    ("meeting", "raise the concern", "raised the concern"),
    ("club", "welcome the newcomer", "welcomed the newcomer"),
    ("kitchen", "check the instructions", "checked the instructions"),
    ("class", "ask a question", "asked a question"),
    ("library", "return the books", "returned the books"),
    ("garden", "help the neighbor", "helped the neighbor"),
    ("studio", "change the design", "changed the design"),
    ("station", "explain the delay", "explained the delay"),
    ("team", "review the plan", "reviewed the plan"),
)

SYSTEM = (
    "Read the passage as evidence. Distinguish the narrator from other people, "
    "actual actions from wishes, and voluntary choices from requirements. "
    "Answer the specific factual question with only Yes or No. "
    "Do not infer a personality type or add facts."
)


def material_hash(value: object) -> str:
    """Hash a canonical, UTF-8 JSON representation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def build_controls() -> list[dict]:
    """Return 192 factorial/paired prompts in 12 disjoint scenario clusters."""
    rows = []
    for index, (scene, action, past) in enumerate(SCENARIOS):
        group = f"scenario_{index:02d}"
        split = "discovery" if index < 8 else "heldout_scenario"
        definitions = {
            "ownership": (
                f"Does the passage report that the narrator actually {past}?",
                [
                    (f"I {past}. My colleague only watched.", 1),
                    (f"My colleague {past}. I only watched.", 0),
                    (f"The person who {past} was me, not my colleague.", 1),
                    (f"The person who {past} was my colleague, not me.", 0),
                ],
            ),
            "actuality": (
                f"Does the passage report that the narrator actually {past}?",
                [
                    (f"I did {action}. It really happened.", 1),
                    (f"I did not {action}. It never happened.", 0),
                    (f"I {past}. This describes an actual event, not a wish.", 1),
                    (
                        f"I wished I could {action}. It was only a wish, not an actual event.",
                        0,
                    ),
                ],
            ),
            "voluntary_choice": (
                f"Does the passage explicitly say that the narrator freely chose to {action}?",
                [
                    (f"I chose to {action}. I was free to refuse.", 1),
                    (f"I was forced to {action}. I was not free to refuse.", 0),
                    (f"I {past} voluntarily. Nobody required it.", 1),
                    (f"I {past} under orders. I had no choice.", 0),
                ],
            ),
            "composition": (
                (
                    f"Does the passage explicitly establish BOTH that the narrator chose to {action} "
                    "and that the narrator carried out this action?"
                ),
                [
                    (
                        (
                            f"I chose to {action}."
                            if a
                            else f"I did not choose to {action}."
                        )
                        + " "
                        + (
                            f"I actually {past}."
                            if b
                            else f"I did not actually {action}."
                        ),
                        int(a and b),
                    )
                    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))
                ],
            ),
        }
        for family, (question, variants) in definitions.items():
            for variant, (passage, expected) in enumerate(variants):
                rows.append(
                    {
                        "case_id": f"{group}__{family}__{variant}",
                        "group_id": group,
                        "family": family,
                        "variant": variant,
                        "split": split,
                        "passage": f"Setting: {scene}. Narrator's account: {passage}",
                        "question": question,
                        "expected_yes": expected,
                        "label_scope": "explicit_semantic_fact_not_personality",
                    }
                )
    return rows


def control_user_message(row: dict) -> str:
    """Use a common terminal readout marker across all control conditions."""
    return f"Passage:\n{control_content_anchor(row)}\n\nQuestion: {row['question']}\nAnswer:"


def control_content_anchor(row: dict) -> str:
    """End evidence with a common pre-question token, avoiding lexical endpoints."""
    return row["passage"] + "\nEvidence ends here."


def validate_controls(rows: list[dict]) -> dict:
    """Check unique cases and scenario-level split separation."""
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate control case IDs")
    groups = {}
    for row in rows:
        if row["expected_yes"] not in (0, 1):
            raise ValueError("Control truth must be binary")
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Scenario crosses discovery and evaluation")
    return {
        "cases": len(rows),
        "independent_scenarios": len(groups),
        "material_sha256": material_hash(rows),
        "label_scope": "explicit_semantic_fact_not_personality",
    }
