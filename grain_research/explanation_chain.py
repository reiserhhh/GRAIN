"""Evidence-addressed explanations, without promoting model scores to behavior."""

from __future__ import annotations

import hashlib

import numpy as np


def public_bank():
    """Cross evidence identity, requested support, time budget and wording."""
    phrases = {
        "intention": [
            [
                "I plan to ask you to confirm that my next step is right.",
                "I intend to ask for your reassurance before taking the next step.",
            ],
            [
                "I plan to take the next step without asking you for confirmation.",
                "I intend to proceed on my own without asking for your reassurance.",
            ],
        ],
        "reported_action": [
            [
                "I asked you to confirm that my next step was right.",
                "I asked for your reassurance before taking the next step.",
            ],
            [
                "I took the next step without asking you for confirmation.",
                "I proceeded on my own without asking for your reassurance.",
            ],
        ],
        "current_utterance": [
            [
                "Please confirm that my next step is right.",
                "Please give me your reassurance before I take the next step.",
            ],
            [
                "Please let me take the next step without giving me confirmation.",
                "Please let me proceed on my own without offering your reassurance.",
            ],
        ],
    }
    rows = []
    for identity, poles in phrases.items():
        for pole, wordings in zip([1, -1], poles):
            for wording, phrase in enumerate(wordings):
                for minutes in [5, 45]:
                    text = (
                        "I am reviewing a routine work request with you. "
                        f"The available time for this work is {minutes} minutes. "
                        "There are some notes on the table and a folder containing the instructions. "
                        "The next step concerns where to put the completed form. "
                        + phrase
                        + " The room has a desk and two chairs, and the form has "
                        "a space for the date and a space for the reference number."
                    )
                    rows.append(
                        {
                            "case": f"c{len(rows):02d}",
                            "identity": identity,
                            "pole": pole,
                            "wording": wording,
                            "minutes": minutes,
                            "target_span": phrase,
                            "text": text,
                        }
                    )
    return rows


def decompose_query(g, h, model):
    """Independently expand the frozen second-order potential into token terms."""
    g, h = np.asarray(g, float), np.asarray(h, float)
    if len(g) != len(h) or len(g) < 2 or not np.isfinite(h).all():
        raise ValueError("Aligned finite observations required")
    z = (g - model["g_mean"]) / model["g_scale"]
    x = np.column_stack(
        [
            np.ones(len(g)),
            *z.T,
            *(z[:, i] * z[:, j] for i in range(4) for j in range(i, 4)),
        ]
    )
    between = np.sum((x @ model["beta2"]) * h, axis=1) / len(h)
    residual = h.copy()
    for cell in np.unique(g, axis=0):
        mask = np.all(g == cell, axis=1)
        residual[mask] -= h[mask].mean(axis=0)
    projected = residual @ model["query_basis"]
    within = np.sum((projected @ model["within_matrix"]) * projected, axis=1) / len(h)
    return float(model["offsets"][2]), between, within


def condition_profile(low, high):
    """Return finite level and half-contrast, not a fitted trait or derivative."""
    if not np.isfinite([low, high]).all():
        raise ValueError("Finite endpoints required")
    return {"level": (low + high) / 2, "half_change": (high - low) / 2}


def interpretation_scope(identity):
    """Declare what the controlled stimulus manifests, independently of its score."""
    return {
        "intention": (
            "hypothetical_event",
            "expressed intention; not evidence of enactment",
        ),
        "reported_action": (
            "reported_event",
            "self-reported past action; enactment unobserved",
        ),
        "current_utterance": (
            "observed_utterance",
            "request made in this authored stimulus; not a human action record",
        ),
    }[identity]


def make_packet(row, values, operators):
    """Bind scores to the entire input and candidate meaning to its exact clause."""
    text, key = row["text"], row["case"]
    digest = hashlib.sha256(text.encode()).hexdigest()
    start = text.index(row["target_span"])
    event_kind, meaning = interpretation_scope(row["identity"])
    evidence = [
        {
            "id": "whole",
            "source_id": key,
            "source_sha256": digest,
            "start": 0,
            "end": len(text),
            "quote": text,
            "speaker": "synthetic respondent",
            "target": "work request",
            "event_kind": "observed_utterance",
        },
        {
            "id": "clause",
            "source_id": key,
            "source_sha256": digest,
            "start": start,
            "end": start + len(row["target_span"]),
            "quote": row["target_span"],
            "speaker": "synthetic respondent",
            "target": "confirmation or independent action",
            "event_kind": event_kind,
        },
    ]
    measurements = []
    for name, kind, reference, unit in [
        (
            "grain_f03",
            "grain_score",
            "frozen 29-author calibration; uniform four-slice mean",
            "mean of slice calibration z units; not author percentile",
        ),
        (
            "query",
            "compiled_function",
            "frozen four-source-author query potential",
            "compiled confirmation_seeking potential; not probability",
        ),
    ]:
        measurements.append(
            {
                "id": name,
                "value": float(values[name]),
                "operator_sha256": operators[name],
                "reference": reference,
                "query": name,
                "unit": unit,
                "kind": kind,
                "evidence_ids": ["whole"],
            }
        )
    return {
        "version": "1",
        "research_only": True,
        "mode": "evidence_rendering",
        "episode_id": key,
        "condition": f"available time {row['minutes']} minutes",
        "evidence": evidence,
        "measurements": measurements,
        "candidates": [
            {
                "id": "scope",
                "registry_id": None,
                "hypothesis": meaning,
                "evidence_ids": ["clause"],
                "measurement_ids": ["grain_f03", "query"],
                "alternative": "Shared confirmation vocabulary can drive both scores without identifying enactment.",
                "missing_context": [
                    "Independent human coding",
                    "Actual human behavior",
                    "Natural-language identity recognition outside this stimulus manifest",
                ],
                "support": "tentative",
                "human_status": "not_reviewed",
            }
        ],
    }
