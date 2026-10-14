#!/usr/bin/env python3
"""Finite role binding and query contraction before author-level aggregation.

Axes are actor (respondent/colleague), channel (past report/future intention),
proposition orientation (-/+), and response stance (-/+). They are evidence
coordinates, not psychological factors. Unallocated mass is unresolved; it is
never silently converted to a neutral personal score. Neural candidate weights
are not calibrated probabilities merely because they sum to one.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

SHAPE = (2, 2, 2, 2)
SIGNS = np.array([-1.0, 1.0])
QUESTION = re.compile(
    r"In a (social|practical) situation, (did|will) (you|your colleague) "
    r"(accept|refuse) the offered help (yesterday|tomorrow)\?",
    re.IGNORECASE,
)


def validate_joint(joint):
    """Validate a finite, nonnegative sub-distribution on the named axes."""
    a = np.asarray(joint, float)
    if (
        a.shape != SHAPE
        or not np.isfinite(a).all()
        or np.any(a < 0)
        or a.sum() > 1 + 1e-12
    ):
        raise ValueError("Expected nonnegative 2x2x2x2 mass with total <= 1")
    return a


def atom(actor, channel, orientation, stance):
    """Return one complete binding, using explicit finite coordinate indices."""
    coordinates = (actor, channel, orientation, stance)
    if any(
        not isinstance(i, (int, np.integer)) or i not in (0, 1) for i in coordinates
    ):
        raise ValueError("Four binary coordinate indices required")
    result = np.zeros(SHAPE)
    result[coordinates] = 1
    return result


def query_kernel(actor=0, channel=1):
    """Contrast accepting/refusing help in a specified actor/evidence channel."""
    if actor not in (0, 1) or channel not in (0, 1):
        raise ValueError("Unknown actor or channel")
    result = np.zeros(SHAPE)
    result[actor, channel] = np.outer(SIGNS, SIGNS)
    return result


def query_measure(joint, *, actor=0, channel=1):
    """Keep signed mass, eligible mass and conditional score distinct.

    Bounds allow all unresolved mass to become eligible with either sign. They
    are sensitivity bounds under that mass convention, not confidence intervals.
    Variance describes eligible candidate signs, not sampling error of a trait.
    """
    a = validate_joint(joint)
    kernel = query_kernel(actor, channel)
    mass = float(a[actor, channel].sum())
    signed = float(np.sum(a * kernel))
    unknown = max(0.0, 1 - float(a.sum()))
    denominator = mass + unknown
    score = signed / mass if mass > 0 else None
    return {
        "signed_mass": signed,
        "eligible_mass": mass,
        "unresolved_mass": unknown,
        "score": score,
        "candidate_variance": max(0.0, 1 - score**2) if score is not None else None,
        "lower": (signed - unknown) / denominator if denominator > 0 else None,
        "upper": (signed + unknown) / denominator if denominator > 0 else None,
        "status": "candidate_score"
        if mass > 0
        else "unresolved"
        if unknown > 0
        else "no_target_evidence",
    }


def marginalize(joint):
    """Return all one-axis marginals while preserving their total mass."""
    a = validate_joint(joint)
    return [a.sum(axis=tuple(j for j in range(4) if j != i)) for i in range(4)]


def factorize(joint):
    """Remove dependence among axes without changing any one-axis marginal."""
    a = validate_joint(joint)
    mass = float(a.sum())
    if not mass:
        return a.copy()
    r = marginalize(a)
    return np.einsum("i,j,k,l->ijkl", *r) / mass**3


def transform_axes(value, transforms):
    """Apply common orthogonal basis changes to a tensor or its dual query.

    Transformed coordinates may be negative. They are vector coordinates, not
    probability masses; query contraction is invariant if both objects change.
    """
    result = np.asarray(value, float)
    if result.shape != SHAPE or not np.isfinite(result).all() or len(transforms) != 4:
        raise ValueError("One finite tensor and four orthogonal maps required")
    for axis, transform in enumerate(transforms):
        t = np.asarray(transform, float)
        if (
            t.shape != (2, 2)
            or not np.isfinite(t).all()
            or not np.allclose(t.T @ t, np.eye(2))
        ):
            raise ValueError("Orthogonal 2x2 basis map required")
        result = np.moveaxis(np.tensordot(t, result, axes=(1, axis)), 0, axis)
    return result


def read_controlled_pair(partner, reply):
    """Literal public-grammar reference, not an open-language semantic parser.

    Only complete declared questions and bare Yes/No are supported. True negative
    questions, reported quotations and unfamiliar phrasing remain unresolved.
    The function accepts raw strings only, never generator labels or metadata.
    """
    question = QUESTION.fullmatch(partner.strip())
    answer = re.fullmatch(r"(yes|no)[.!]?", reply.strip(), re.IGNORECASE)
    result = {
        "joint": np.zeros(SHAPE),
        "condition": None,
        "status": "outside_declared_grammar",
        "source_sha256": hashlib.sha256((partner + "\n" + reply).encode()).hexdigest(),
        "partner_span": [0, len(partner)],
        "reply_span": [0, len(reply)],
    }
    if not question or not answer:
        return result
    condition, tense, person, predicate, time_word = (
        v.lower() for v in question.groups()
    )
    if (tense, time_word) not in {("did", "yesterday"), ("will", "tomorrow")}:
        return result
    o, c, p, s = (
        int(person != "you"),
        int(tense == "will"),
        int(predicate == "accept"),
        int(answer.group(1).lower() == "yes"),
    )
    result.update(
        joint=atom(o, c, p, s),
        condition=1.0 if condition == "social" else -1.0,
        status="declared_grammar_reference",
        orientation=p,
    )
    return result


def candidate_joint(entailment, orientation):
    """Bind eight NLI candidate weights to an observed question orientation.

    Input shape is actor x channel x action polarity, with negative first.
    Normalization is a working exhaustive-candidate convention, not calibration.
    The question's orientation comes from observed wording, not hidden truth.
    """
    a = np.asarray(entailment, float)
    if a.shape != (2, 2, 2) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Eight finite nonnegative candidate weights required")
    if orientation not in (0, 1):
        raise ValueError("Observed question orientation required")
    result = np.zeros(SHAPE)
    if not a.sum():
        return result
    a = a / a.sum()
    for o in range(2):
        for c in range(2):
            for target in range(2):
                stance = int(SIGNS[target] * SIGNS[orientation] > 0)
                result[o, c, orientation, stance] = a[o, c, target]
    return result


def symbolic_relations(event, raw_sha256):
    """Adapt an existing syntactic graph without inventing semantic factuality.

    Source, grammatical participant, local polarity and embedding context retain
    their original binding. Word morphology is not declared an event's real time.
    No inferred target score or psychological classification is produced here.
    """
    return {
        "event_id": event["event_id"],
        "source_sha256": raw_sha256,
        "predicate_span": [event["predicate_start"], event["predicate_end"]],
        "word_ids": event["local_word_ids"],
        "subject_reference": event["subject_reference"],
        "source_candidate": event["source_candidate"],
        "source_path": event["source_path"],
        "embedding_path": event["embedding_path"],
        "local_negation": event["local_negation"],
        "modal_cues": event["modal_cues"],
        "morphology": event["morphology"],
        "intention_embedded": any(p["intention"] for p in event["embedding_path"]),
        "condition_links": event["adverbial_clauses"],
        "factuality": "not_inferred",
        "target_orientation": None,
        "semantic_actor": None,
        "event_time": None,
        "personal_score": None,
    }
