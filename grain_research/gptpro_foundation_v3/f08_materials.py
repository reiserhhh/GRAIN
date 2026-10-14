"""Declared synthetic observation materials; no author or hidden-state input."""

from __future__ import annotations

import numpy as np

STATUSES = ("actual", "quoted", "hypothetical", "negated")
FAMILIES = ("plain", "contextual", "retrospective")
ROLES = ("source", "validation", "application")


def render_event(pole, subject_status, family, variant):
    """Render stated behavior, third-party report, hypothetical, or corrected behavior."""
    if pole not in (0, 1) or subject_status not in STATUSES:
        raise ValueError("binary pole and declared status required")
    if family not in FAMILIES or variant not in range(4):
        raise ValueError("declared grammar family and variant required")
    action = ("rejected", "supported")[pole]
    opposite = ("supported", "rejected")[pole]
    bodies = {
        "actual": f"I actually {action} the plan.",
        "quoted": f"My colleague said: 'I actually {action} the plan.' This reports my colleague's response, not mine.",
        "hypothetical": f"If I faced that hypothetical decision, I might have {action} the plan. I have not made that decision.",
        "negated": f"The claim that I {opposite} the plan is incorrect. To correct it: I actually {action} the plan.",
    }
    frames = {
        "plain": (
            "My note reads: {}",
            "Here is my account: {}",
            "I wrote this entry: {}",
            "This is the record: {}",
        ),
        "contextual": (
            "Regarding the committee proposal, {}",
            "In the discussion about the proposal, {}",
            "As for the meeting's proposal, {}",
            "Concerning the group's proposal, {}",
        ),
        "retrospective": (
            "Looking back on the proposal, {}",
            "In recalling the proposal, {}",
            "While reviewing the account later, {}",
            "Reflecting on the recorded discussion, {}",
        ),
    }
    return frames[family][variant].format(bodies[subject_status])


def material_bank():
    """Construct 96 declared materials, with disjoint wrapper families across roles."""
    rows = []
    for family, role in zip(FAMILIES, ROLES, strict=True):
        for pole in (0, 1):
            for status in STATUSES:
                for variant in range(4):
                    rows.append(
                        {
                            "material_id": f"material_{len(rows):03d}",
                            "role": role,
                            "family": family,
                            "pole": pole,
                            "subject_status": status,
                            "qualification": int(status in ("actual", "negated")),
                            "variant": variant,
                            "text": render_event(pole, status, family, variant),
                        }
                    )
    return rows


def read_scores(reader, texts):
    """Preserve raw pole scores before clipping; IDs and truth never enter features."""
    from scipy.sparse import hstack

    features = hstack(
        [reader["word"].transform(texts), reader["char"].transform(texts)]
    ).tocsr()
    raw = reader["pole"].predict(features)
    qualification = reader["qualification"].predict_proba(features)[:, 1]
    return {"pole_raw": raw, "pole": np.clip(raw, 0, 1), "qualification": qualification}
