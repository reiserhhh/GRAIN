"""Balanced entity/action binding controls with an explicit relational truth table."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.controls import SCENARIOS

ALIASES = (("Cedar", "Maple"), ("Robin", "Raven"), ("Amber", "Coral"))
SYSTEM = (
    "Read the records as evidence about the narrator and a colleague. "
    "Resolve code names before attributing an action. Actual choices and actions "
    "are different from personality traits. Answer the specific factual question "
    "with only Yes or No, using its ordinary meaning. Do not add facts."
)


def build_binding_controls():
    """Cross narrator identity and event actor; the relational answer is XNOR.

    Every complete quartet has exactly the same token vocabulary. A feature
    map additive across the two records has zero cross-record interaction.
    """
    rows = []
    for index, (scene, action, past) in enumerate(SCENARIOS):
        names = ALIASES[index % len(ALIASES)]
        for a in (0, 1):
            identity = f"I use the code name {names[a]}. My colleague uses the code name {names[1 - a]}."
            for b in (0, 1):
                event = (
                    f"{names[b]} freely chose to {action} and actually {past}. "
                    f"{names[1 - b]} did not choose to {action} and did not {action}."
                )
                for order in ("identity_first", "event_first"):
                    ordered = (
                        (identity, event)
                        if order == "identity_first"
                        else (event, identity)
                    )
                    passage = f"Setting: {scene}.\nRecord 1: {ordered[0]}\nRecord 2: {ordered[1]}\nEvidence ends here."
                    questions = {
                        "narrator_positive": (
                            f"Did the narrator freely choose to {action} and actually carry it out?",
                            int(a == b),
                        ),
                        "narrator_negative": (
                            f"Do the records explicitly say that the narrator did NOT {action}?",
                            int(a != b),
                        ),
                        "alias_action": (
                            f"Did {names[0]} freely choose to {action} and actually carry it out?",
                            int(b == 0),
                        ),
                    }
                    for task, (question, label) in questions.items():
                        rows.append(
                            {
                                "case_id": f"binding_{index:02d}__a{a}b{b}__{order}__{task}",
                                "group_id": f"binding_{index:02d}",
                                "scene": scene,
                                "split": "discovery"
                                if index < 8
                                else "heldout_scenario",
                                "a": a,
                                "b": b,
                                "order": order,
                                "task": task,
                                "identity_record": identity,
                                "event_record": event,
                                "passage": passage,
                                "question": question,
                                "expected_yes": label,
                                "label_scope": "synthetic_relational_fact_not_personality",
                            }
                        )
    return rows


def binding_user_message(row):
    """Separate evidence from the question and use a common response suffix."""
    return f"{row['passage']}\n\nQuestion: {row['question']}\nAnswer:"


def factorial_interaction(quartet):
    """Return the signed second finite difference, not information-theoretic synergy."""
    if set(quartet) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("A complete two-by-two quartet is required")
    return quartet[0, 0] + quartet[1, 1] - quartet[0, 1] - quartet[1, 0]


def relational_grams(identity, event, train):
    """Train-reference linear, tensor-product, and combined positive semidefinite kernels.

    Centering here conditions kernel numerics, not a claim that the removed mean
    is psychological noise. The combined kernel keeps both marginal channels.
    """
    components = []
    for values in (identity, event):
        x = np.asarray(values, dtype=float)
        x = x - x[train].mean(axis=0)
        scale = np.sqrt(np.mean(np.sum(x[train] ** 2, axis=1)))
        x = x / max(scale, 1e-12)
        components.append(x @ x.T)
    a, b = components
    return {
        "additive": a + b,
        "tensor_product": a * b,
        "additive_plus_product": a + b + a * b,
    }
