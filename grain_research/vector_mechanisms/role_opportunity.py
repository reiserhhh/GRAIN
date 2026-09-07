"""Synthetic role-bound evidence panels and condition-sensitive attribution readouts."""

from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

from .analysis import grouped_bootstrap
from .reciprocity import random_direction

SYSTEM = (
    "Read the supplied records carefully. Keep statements about different people "
    "separate. Follow the specified answer convention."
)
DEVELOPMENT = (
    ("d_ceramics", "a ceramics course"),
    ("d_running", "a beginners running session"),
    ("d_cinema", "a film discussion"),
    ("d_science", "a science outreach meeting"),
    ("d_music", "a music appreciation class"),
    ("d_cleanup", "a neighborhood cleanup"),
)
EVALUATION = (
    ("e_photos", "a photography meetup"),
    ("e_theatre", "a theatre rehearsal"),
    ("e_puzzles", "a puzzle club"),
    ("e_sailing", "a sailing lesson"),
    ("e_history", "a local history gathering"),
    ("e_baking", "a baking workshop"),
)
ACTIONS = {
    1: "approached an unfamiliar participant and started a conversation",
    -1: "waited quietly until an unfamiliar participant started a conversation",
}
PROFILES = {
    1: "socially outgoing and extraverted",
    -1: "socially reserved and introverted",
}


def code_answers(question: int, convention: int) -> tuple[str, str]:
    """Map high/low target semantics, not answer-letter preference, to candidates."""
    if question not in (-1, 1) or convention not in (-1, 1):
        raise ValueError("Signs must be +/-1")
    return ("A", "B") if question * convention == 1 else ("B", "A")


def coded_question(target: str, polarity: int, convention: int) -> str:
    """Cross tendency-question direction with an independently reversed answer code."""
    trait = (
        "outgoing rather than reserved"
        if polarity == 1
        else "reserved rather than outgoing"
    )
    mapping = "A for Yes and B for No" if convention == 1 else "A for No and B for Yes"
    return f"Based on this record, does {target} seem {trait}? Return {mapping}."


def build_material() -> list[dict]:
    """Build the complete156-prompt panel without private records or scale labels."""
    rows = []

    def add(group, split, family, level, evidence, question, answers, **extra):
        row = {
            "group_id": group,
            "split": split,
            "family": family,
            "level": level,
            "question_polarity": 1,
            "code": 1,
            "condition": "none",
            "owner": "target",
            "expected_sign": level,
            "evidence": evidence,
            "high_answer": answers[0],
            "low_answer": answers[1],
            "user": evidence + "\n\nQuestion: " + question,
        } | extra
        row["case_id"] = "__".join(
            str(row[k])
            for k in (
                "group_id",
                "family",
                "level",
                "question_polarity",
                "code",
                "condition",
                "owner",
            )
        )
        rows.append(row)

    for group, setting in DEVELOPMENT:
        for source, level in product(("self", "other", "behavior"), (-1, 1)):
            if source == "self":
                evidence = (
                    f"Setting: {setting}.\nYour profile: You are {PROFILES[level]}."
                )
                question = "Speak as this person. What do you do when unfamiliar people are nearby?"
                answers = ("I start a conversation.", "I wait for someone to approach.")
            elif source == "other":
                evidence = (
                    f"Setting: {setting}.\nAlex's profile: Alex is {PROFILES[level]}."
                )
                question = (
                    "Predict Alex's next action when unfamiliar people are nearby."
                )
                answers = (
                    "Alex starts a conversation.",
                    "Alex waits for someone to approach.",
                )
            else:
                evidence = f"Setting: {setting}.\nAlex {ACTIONS[level]}."
                question = "Did Alex initiate the conversation? Answer Yes or No."
                answers = ("Yes", "No")
            add(group, "development", source, level, evidence, question, answers)

    for index, (group, setting) in enumerate(EVALUATION):
        target, companion = ("Alex", "Blair") if index % 2 == 0 else ("Blair", "Alex")
        for level in (-1, 1):
            records = [f"{target} {ACTIONS[level]}.", f"{companion} {ACTIONS[-level]}."]
            if index % 3 == 1:
                records.reverse()
            evidence = f"Setting: {setting}. The room was open.\n" + "\n".join(records)
            for q, c in product((-1, 1), repeat=2):
                add(
                    group,
                    "evaluation",
                    "trait",
                    level,
                    evidence,
                    coded_question(target, q, c),
                    code_answers(q, c),
                    question_polarity=q,
                    code=c,
                    target=target,
                    companion=companion,
                )
            add(
                group,
                "evaluation",
                "venue",
                level,
                evidence,
                "Was the room open? Return A for Yes and B for No.",
                ("A", "B"),
                expected_sign=1,
                target=target,
                companion=companion,
            )
            for owner, name, expected in (
                ("target", target, level),
                ("companion", companion, -level),
            ):
                add(
                    group,
                    "evaluation",
                    "owner",
                    level,
                    evidence,
                    f"Did {name} initiate the conversation rather than wait? Answer Yes or No.",
                    ("Yes", "No"),
                    owner=owner,
                    expected_sign=expected,
                    target=target,
                    companion=companion,
                )
            conditions = {
                "free": "There were no instructions about initiating or waiting. Each person chose freely how to interact.",
                "assigned": "Each person was assigned whether to initiate or wait. Alex's behavior followed that assignment.",
                "neutral": "The session took place indoors during the afternoon. Each person attended the same scheduled event.",
            }
            for condition, clause in conditions.items():
                single = f"Setting: {setting}.\n{clause}\nAlex {ACTIONS[level]}."
                add(
                    group,
                    "evaluation",
                    "opportunity",
                    level,
                    single,
                    "Based on this record, does Alex seem outgoing rather than reserved? Answer Yes or No.",
                    ("Yes", "No"),
                    condition=condition,
                    target="Alex",
                )
    validate_material(rows)
    return rows


def validate_material(rows: list[dict]) -> None:
    """Require full cells, disjoint settings and exact word-bag-preserving role pairs."""
    if len(rows) != 156 or len({r["case_id"] for r in rows}) != 156:
        raise ValueError("Incomplete or duplicate material")
    if {r["group_id"] for r in rows if r["split"] == "development"} & {
        r["group_id"] for r in rows if r["split"] == "evaluation"
    }:
        raise ValueError("Development/evaluation overlap")
    if Counter(r["family"] for r in rows) != {
        "self": 12,
        "other": 12,
        "behavior": 12,
        "trait": 48,
        "venue": 12,
        "owner": 24,
        "opportunity": 36,
    }:
        raise ValueError("Family counts differ")
    for group, _ in EVALUATION:
        group_rows = [
            r for r in rows if r["group_id"] == group and r["family"] == "trait"
        ]
        if {(r["level"], r["question_polarity"], r["code"]) for r in group_rows} != set(
            product((-1, 1), repeat=3)
        ):
            raise ValueError("Missing factorial cells")
        evidence = {r["level"]: r["evidence"] for r in group_rows}
        if Counter(evidence[-1].split()) != Counter(evidence[1].split()):
            raise ValueError("Role swap changed the word bag")
    for row in rows:
        if (
            row["high_answer"] == row["low_answer"]
            or row["user"].count(row["evidence"]) != 1
        ):
            raise ValueError("Invalid evidence or candidates")


def fit_directions(rows: list[dict], states: dict) -> tuple[dict, dict]:
    """Estimate only pre-query developmental endpoints; use identical vectors at both sites."""
    raw = {}
    for family in ("self", "other", "behavior"):
        lookup = {
            (r["group_id"], r["level"]): r["case_id"]
            for r in rows
            if r["split"] == "development" and r["family"] == family
        }
        delta = [
            states[lookup[g, 1]][-2].astype(float)
            - states[lookup[g, -1]][-2].astype(float)
            for g in sorted({g for g, _ in lookup})
        ]
        raw[family] = np.asarray(delta, dtype=float).mean(axis=0)
    norms = {k: float(np.linalg.norm(v)) for k, v in raw.items()}
    if (
        not all(np.isfinite(v).all() for v in raw.values())
        or min(norms.values()) <= 1e-8
    ):
        raise ValueError("Degenerate development direction")
    norm = float(np.median(list(norms.values())))
    directions = {k: v * norm / norms[k] for k, v in raw.items()}
    directions["random"] = random_direction(directions["self"], "P1B:731:shared")
    return directions, {"raw_norms": norms, "shared_norm": norm, "dose": 0.5}


def check_donor_layout(ids, span, donor_ids, donor_span):
    """Reject token-count/position mismatch; never replace query or answer tokens."""
    if len(span) != len(donor_span) or span != donor_span or len(ids) != len(donor_ids):
        raise ValueError("Donor token layout differs")
    outside = set(range(len(ids))) - set(span)
    if any(ids[i] != donor_ids[i] for i in outside):
        raise ValueError("Donor changes outside evidence")


def describe(values) -> dict:
    """Bootstrap distinct authored scenarios, not their factorial repetitions."""
    values = np.asarray(values, dtype=float)
    stats = grouped_bootstrap(values, np.arange(len(values)))
    return {
        "mean": stats["mean"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "positive_fraction": float(np.mean(values > 0)),
        "n_scenarios": len(values),
    }


def summarize(
    baselines: pd.DataFrame, effects: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Keep role correctness, attribution, coding interactions and donor recovery distinct."""
    keys = ["split", "family", "question_polarity", "code", "condition", "owner"]
    native = []
    for key, sub in baselines.groupby(keys):
        pair = sub.pivot(index="group_id", columns="level", values="margin")
        native.append(
            dict(zip(keys, key))
            | describe(pair[1] - pair[-1])
            | {
                "expected_direction_rate": float(
                    (sub.margin * sub.expected_sign > 0).mean()
                ),
                "candidate_mass_min": float(sub.candidate_mass.min()),
            }
        )
    tables = {"native_summary": pd.DataFrame(native)}
    opportunity = baselines[baselines.family == "opportunity"]
    gaps = opportunity.pivot(
        index=["group_id", "condition"], columns="level", values="margin"
    )
    gaps = (gaps[1] - gaps[-1]).unstack("condition")
    tables["opportunity_contrasts"] = pd.DataFrame(
        [
            {"contrast": a + "_minus_" + b} | describe(gaps[a] - gaps[b])
            for a, b in (
                ("free", "assigned"),
                ("neutral", "assigned"),
                ("free", "neutral"),
            )
        ]
    )
    if effects.empty:
        return tables
    additive = effects[effects.kind == "additive"]
    slopes = []
    for key, sub in additive.groupby(["case_id", "source", "position"]):
        if set(sub.sign) != {-1, 1} or len(sub) != 2:
            raise ValueError("Incomplete signed pair")
        vals = sub.set_index("sign").margin
        slopes.append(
            sub.iloc[0].to_dict() | {"slope": float((vals[1] - vals[-1]) / 2)}
        )
    slope = pd.DataFrame(slopes)
    tables["case_slopes"] = slope
    components = []
    for (source, position), sub in slope[slope.family == "trait"].groupby(
        ["source", "position"]
    ):
        for name, cols in {
            "invariant": [],
            "question": ["question_polarity"],
            "code": ["code"],
            "question_code": ["question_polarity", "code"],
            "evidence_gap_half": ["level"],
        }.items():
            weight = (
                np.prod(sub[cols].to_numpy(), axis=1) if cols else np.ones(len(sub))
            )
            per_group = (
                sub.assign(value=sub.slope * weight).groupby("group_id").value.mean()
            )
            components.append(
                {"source": source, "position": position, "component": name}
                | describe(per_group)
            )
    tables["factorial_components"] = pd.DataFrame(components)
    comparison = []
    trait = slope[slope.family == "trait"]
    for position, sub in trait.groupby("position"):
        paired = sub.pivot(
            index=["group_id", "case_id"], columns="source", values="slope"
        )
        for source in ("self", "other"):
            per_group = (paired[source] - paired["random"]).groupby("group_id").mean()
            comparison.append(
                {"position": position, "contrast": source + "_minus_random"}
                | describe(per_group)
            )
        comparison.append(
            {"position": position, "contrast": "other_minus_self"}
            | describe((paired.other - paired.self).groupby("group_id").mean())
        )
    tables["paired_direction_contrasts"] = pd.DataFrame(comparison)
    tables["venue_slopes"] = pd.DataFrame(
        [
            {"source": source, "position": position}
            | describe(sub.groupby("group_id").slope.mean())
            for (source, position), sub in slope[slope.family == "venue"].groupby(
                ["source", "position"]
            )
        ]
    )
    donor = effects[effects.kind == "donor"].copy()
    base = baselines.set_index("case_id")
    donor["donor_baseline"] = donor.donor_case.map(base.margin)
    donor["recovery_error"] = donor.margin - donor.donor_baseline
    tables["donor_recovery"] = donor
    return tables
