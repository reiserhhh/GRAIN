#!/usr/bin/env python3
"""Invented behavioral contrasts for direct-readout semantics, not scale validation."""

from collections.abc import Mapping

import numpy as np

from .direct_traits import AXES, BIG5, SYSTEM, candidates, question, render

# These examples are authored before model outputs, not sampled from participants.
ANCHORS = {
    "extraversion": (
        (
            "At gatherings I stay quiet and wait for others to approach me.",
            "At gatherings I introduce myself and start conversations with many people.",
        ),
        (
            "After a busy week I prefer spending the evening quietly on my own.",
            "After a busy week I seek a lively evening meeting and talking with friends.",
        ),
    ),
    "neuroticism": (
        (
            "When plans go wrong I usually stay calm and soon feel at ease again.",
            "When plans go wrong I become distressed and keep worrying for a long time.",
        ),
        (
            "While waiting for uncertain news I can relax and get on with my day.",
            "While waiting for uncertain news I repeatedly imagine disasters and feel tense.",
        ),
    ),
    "agreeableness": (
        (
            "In disagreements I dismiss other people's needs and insist on getting my way.",
            "In disagreements I listen to other people's needs and seek a fair compromise.",
        ),
        (
            "When a colleague struggles I avoid helping even when it would cost me little.",
            "When a colleague struggles I offer practical help and try to be considerate.",
        ),
    ),
    "conscientiousness": (
        (
            "I leave agreed tasks unfinished and often forget the deadlines I accepted.",
            "I keep track of agreed tasks and reliably finish them before their deadlines.",
        ),
        (
            "I start assignments without planning and quickly abandon the difficult parts.",
            "I plan assignments carefully and persist through the difficult parts.",
        ),
    ),
    "openness": (
        (
            "I prefer familiar ideas and avoid exploring unfamiliar art or perspectives.",
            "I enjoy unfamiliar art and actively explore ideas unlike my usual perspective.",
        ),
        (
            "I choose the same familiar experiences and see little appeal in imagining alternatives.",
            "I seek unusual experiences and enjoy imagining many different possibilities.",
        ),
    ),
    "EI_cont": (
        (
            "I regain energy in quiet solitude and think things through before speaking.",
            "I regain energy by meeting people and often think things through by talking.",
        ),
        (
            "I prefer a quiet conversation with one familiar person to a lively group.",
            "I prefer joining a lively group and readily speak with people I have just met.",
        ),
    ),
    "SN_cont": (
        (
            "I begin with concrete observations and trust practical details I can verify.",
            "I begin with abstract possibilities and look for patterns beyond immediate details.",
        ),
        (
            "In learning a task I want specific examples and a practical step-by-step demonstration.",
            "In learning a task I want an overarching concept and connections to new possibilities.",
        ),
    ),
    "TF_cont": (
        (
            "When deciding I prioritize consistent impersonal criteria and logical trade-offs.",
            "When deciding I prioritize personal values and the effects on relationships.",
        ),
        (
            "To resolve a dilemma I first compare the options against a general logical principle.",
            "To resolve a dilemma I first consider what matters personally to the people involved.",
        ),
    ),
    "JP_cont": (
        (
            "I prefer keeping plans open so I can adapt as new options appear.",
            "I prefer settling plans early so I know what to do and when.",
        ),
        (
            "On a trip I enjoy deciding the day's activities spontaneously as opportunities arise.",
            "On a trip I enjoy deciding the day's schedule in advance and following the plan.",
        ),
    ),
}


def panel():
    """Return all90 frozen cases; the no-evidence prompt is a prior, not a neutral person."""
    rows = []
    for task in (*BIG5, *AXES):
        for variant in ("primary", "counter"):
            for pair, texts in enumerate(ANCHORS[task]):
                for polarity, text in enumerate(texts):
                    rows.append(
                        {
                            "case_id": f"{task}_{variant}_{pair}_{polarity}",
                            "task": task,
                            "variant": variant,
                            "pair": pair,
                            "polarity": polarity,
                            "text": text,
                        }
                    )
            rows.append(
                {
                    "case_id": f"{task}_{variant}_no_evidence",
                    "task": task,
                    "variant": variant,
                    "pair": -1,
                    "polarity": -1,
                    "text": "[No writer excerpts were supplied.]",
                }
            )
    return rows


def counter_question(task):
    """Invert numeric encoding or reverse MBTI answer listing, preserving trait polarity."""
    if task in BIG5:
        return (
            f"Estimate the writer's {task}: {BIG5[task]}. "
            "Use 9 for very low, 0 for very high, and the digits in between for "
            "intermediate tendencies. Return only one digit from 0 to 9."
        )
    if task in AXES:
        negative, positive, low, high = AXES[task]
        return (
            f"Which MBTI preference is more strongly suggested for this writer: "
            f"{positive} ({high}) or {negative} ({low})? "
            f"Return only {positive} or {negative}."
        )
    raise ValueError("Unknown task")


def prepare(case, tokenizer):
    """Keep the native direct template; change only the terminal question for counter cases."""
    if case not in panel():
        raise ValueError("Unregistered semantic case")
    task = case["task"]
    user, _ = render([case["text"]], task)
    alphabet = candidates(task, tokenizer)
    if case["variant"] == "counter":
        suffix = question(task) + "\nAnswer:"
        if not user.endswith(suffix):
            raise ValueError("Frozen direct question suffix changed")
        user = user[: -len(suffix)] + counter_question(task) + "\nAnswer:"
        if task in BIG5:
            alphabet["values"] = (1 - np.asarray(alphabet["values"])).tolist()
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if (
        not ids
        or any(not isinstance(i, int) or i < 0 for i in ids)
        or len(ids) + 16 > 3072
    ):
        raise ValueError("Invalid or oversized native semantic prompt")
    return list(ids), alphabet


def summarize(rows):
    """Describe every fixed contrast and recoded sensitivity, without a population p-value."""
    expected = panel()
    if len(rows) != len(expected) or any(
        any(r.get(k) != c[k] for k in c) for r, c in zip(rows, expected, strict=True)
    ):
        raise ValueError("Semantic grid changed or incomplete")
    if any(
        not np.isfinite(r["conditional_score"]) or not 0 <= r["conditional_score"] <= 1
        for r in rows
    ):
        raise ValueError("Invalid semantic readout")
    index = {
        (r["task"], r["variant"], r["pair"], r["polarity"]): r["conditional_score"]
        for r in rows
    }
    contrasts, tasks = [], []
    for task in (*BIG5, *AXES):
        record = {
            "task": task,
            "counter_type": "numeric_scale_inversion"
            if task in BIG5
            else "answer_listing_reversal",
        }
        for variant in ("primary", "counter"):
            deltas = []
            for pair in range(2):
                delta = index[task, variant, pair, 1] - index[task, variant, pair, 0]
                deltas.append(delta)
                contrasts.append(
                    {
                        "task": task,
                        "variant": variant,
                        "pair": pair,
                        "high_minus_low": delta,
                        "expected_sign": 1,
                        "positive": bool(delta > 1e-6),
                    }
                )
            record[f"{variant}_positive_pairs"] = sum(d > 1e-6 for d in deltas)
            record[f"{variant}_no_evidence_score"] = index[task, variant, -1, -1]
        record["mean_absolute_recoded_counter_change"] = float(
            np.mean(
                [
                    abs(
                        index[task, "primary", pair, pol]
                        - index[task, "counter", pair, pol]
                    )
                    for pair in range(2)
                    for pol in range(2)
                ]
            )
        )
        tasks.append(record)
    return contrasts, tasks
