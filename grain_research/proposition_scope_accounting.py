#!/usr/bin/env python3
"""Post-result scope correction and bounded evidence-loss accounting.

Do not impute missing evidence as zero. The interval encloses a finite material
mean with bounded missing contributions, not a population confidence interval.
"""

from __future__ import annotations

import math


def mean_bounds(observed_sum, observed_count, missing_count):
    """Bound the complete finite mean when missing outcomes lie in [-1, 1]."""
    if not math.isfinite(observed_sum) or any(
        not isinstance(n, int) or n < 0 for n in [observed_count, missing_count]
    ):
        raise ValueError("Finite sum and nonnegative integer counts required")
    if abs(observed_sum) > observed_count + 1e-9:
        raise ValueError("Observed scores exceed declared unit bounds")
    total = observed_count + missing_count
    if not total:
        return {"lower": None, "upper": None, "status": "no_observations"}
    return {
        "lower": (observed_sum - missing_count) / total,
        "upper": (observed_sum + missing_count) / total,
        "status": "finite_material_bounds",
    }


def role_account(frame, *, explicit_addressee):
    """Keep the speaker of a request distinct from its proposed action actor.

    Standalone second person may be generic. This correction never turns an
    unobserved addressee into a known interlocutor and never changes r1 scores.
    """
    actor = frame["actor"]
    if (
        actor == "partner"
        and frame["speaker"] == "respondent"
        and not explicit_addressee
    ):
        actor = "second_person_unresolved"
    return {
        "event_id": frame["id"],
        "speech_source": frame["speaker"],
        "proposition_actor_candidate": actor,
        "utterance_function_candidate": frame["channel"]
        if frame["channel"] == "current_request_candidate"
        else None,
        "evidence_kind": frame["channel"],
        "source_scoped": frame["source_scoped"],
        "actual_behavior": "not_observed",
    }
