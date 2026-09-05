"""Frozen cross-current historical-state access and answer-code controls."""

from __future__ import annotations

from .context import SYSTEM
from .data_views import validate_context_bundle
from .memory import render_records
from .natural_patch import patch_message

ACCESS_SYSTEM = SYSTEM.split("Answer the question with exactly")[0] + (
    "Use the answer code stated after the question. Output exactly Yes or No."
)
CODE_NORMAL = (
    "For this question, output Yes if supported by the excerpts; "
    "output No if not supported. Not supported is not proof of the opposite."
)
CODE_SWAPPED = (
    "For this question, use the reversed answer code: output No if supported "
    "by the excerpts; output Yes if not supported. "
    "Not supported is not proof of the opposite."
)


def access_message(history, current, task, *, swapped=False):
    """Place all task and answer-code changes after unchanged evidence tokens."""
    user, interval = patch_message(history, current, task)
    suffix = "\nAnswer:"
    if not user.endswith(suffix):
        raise ValueError("Frozen answer suffix changed")
    code = CODE_SWAPPED if swapped else CODE_NORMAL
    user = user[: -len(suffix)] + "\n" + code + suffix
    start = len(
        "The earlier and current excerpts are presented as writing by one anonymous writer.\n\n"
    )
    history_interval = (start, start + len(history))
    if user[slice(*history_interval)] != history:
        raise ValueError("History interval mismatch")
    return user, history_interval, interval


def confirmation_bundle(view):
    """Select the fixed source/current comments; never choose using outcomes."""
    if view["split"] != "mechanism_holdout" or [len(v) for v in view["views"]] != [
        12,
        12,
        12,
    ]:
        raise ValueError("Need one original 36-comment mechanism-holdout author")
    bundle = {
        "author_id": view["author_id"],
        "history": view["views"][0],
        "current": view["views"][2][:2],
    }
    validate_context_bundle(bundle)
    return bundle


def confirmation_conditions(bundle):
    """Return the four prespecified endpoints, with source always first."""
    history = render_records(bundle["history"])
    conditions = []
    for name, ci, task, swapped in (
        ("source", 0, "disposition", False),
        ("transfer", 1, "disposition", False),
        ("code_swapped", 1, "disposition", True),
        ("collateral", 1, "current_guidance", False),
    ):
        user, hi, current = access_message(
            history, bundle["current"][ci]["body"], task, swapped=swapped
        )
        conditions.append(
            {
                "condition": name,
                "user": user,
                "history_interval": hi,
                "current_interval": current,
                "polarity": -1 if swapped else 1,
            }
        )
    return conditions


def answer_code_controls():
    """Known text-evidence manipulations, not known human personality labels."""
    scenarios = [
        (
            "Every week I invite new colleagues to lunch and start conversations with them.",
            "Every week my colleagues invite me to lunch. I never invite them or start conversations.",
        ),
        (
            "At every club meeting I walk up to unfamiliar members and introduce myself first.",
            "At every club meeting I wait alone. I never approach or introduce myself to members.",
        ),
        (
            "Each month I arrange a neighborhood dinner and personally invite the neighbors.",
            "Each month I attend a dinner only when invited. I never arrange it or invite anyone.",
        ),
        (
            "Whenever a new student arrives, I start a conversation and invite them to our study group.",
            "When a new student arrives, I do not speak or invite them. Others organize the study group.",
        ),
    ]
    rows = []
    for si, pair in enumerate(scenarios):
        for positive, text in zip((True, False), pair, strict=True):
            for swapped in (False, True):
                user, _, _ = access_message(
                    "Earlier excerpt 1:\n" + text,
                    "I checked the schedule and placed my bag next to the chair.",
                    "disposition",
                    swapped=swapped,
                )
                rows.append(
                    {
                        "scenario": si,
                        "positive_evidence": positive,
                        "swapped": swapped,
                        "expected_yes": bool(positive != swapped),
                        "user": user,
                    }
                )
    return rows
