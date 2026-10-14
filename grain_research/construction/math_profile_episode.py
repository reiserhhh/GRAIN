#!/usr/bin/env python3
"""Share a SOURCE anchor profile across episodes, not one expression-mode address."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_author_belief as B
from grain_research.construction import math_conditional_memory as C
from grain_research.construction import math_vector_memory as V

MODELS = ("fresh", "full", "shared_actual_profile", "shared_permuted_profile")


def profile_bank(bundle, assignment):
    """Expose observed reference groups; an absent group is not a neutral person."""
    if type(assignment) is not int or assignment not in (0, 1):
        raise ValueError("Only the two frozen SOURCE anchor assignments")
    codes = np.asarray(bundle["codes"])
    if (
        codes.shape != (2, len(bundle["alpha0"]))
        or codes.dtype.kind not in "iu"
        or np.any((codes < 0) | (codes >= 32))
    ):
        raise ValueError("Aligned finite SOURCE profiles required")
    selected = codes[assignment]
    return {"codes": selected, "counts": np.bincount(selected, minlength=32)}


def episode_logweights(bundle, early_logweights, assignment):
    """Reuse HM211's exact profile-preserving reset as a new EPISODE law."""
    bank = profile_bank(bundle, assignment)
    weights = np.asarray(early_logweights)
    if weights.shape[-1] != len(bank["codes"]):
        raise ValueError("Wrong SOURCE reference address count")
    # The parent prior is uniform over SOURCE addresses, not uniform32 profiles.
    return V.reset_weights(bank, weights)


def cold(bundle, record):
    """Preserve current full anchor profile while allowing a fresh expression mode."""
    episodes = B.validate_record(record, 8)
    if len(episodes) != 2:
        raise ValueError("One early and one late qualified episode required")
    full = C.forward_log_likelihood(bundle, episodes[0], "full")[-1]
    fine = C.posterior_logweights(full)
    projected = [episode_logweights(bundle, fine, g) for g in (0, 1)]
    prior = np.full(len(fine), -np.log(len(fine)))
    weights = [prior, fine, *projected]
    rows = {}
    for k, (name, w) in enumerate(zip(MODELS, weights, strict=True)):
        late = C.observe_late(bundle, episodes[1], w)
        assignment = k - 2 if k >= 2 else None
        rows[name] = {
            "boundary_all_anchor_profiles": B.profile(
                np.exp(w), bundle["codes"]
            ).tolist(),
            "late_logp": late["logp"].tolist(),
            "late_profiles": late["profiles"].tolist(),
            "shared_anchor_assignment": assignment,
            "source_supported_profiles": None
            if assignment is None
            else (profile_bank(bundle, assignment)["counts"] > 0).tolist(),
        }
    return {
        "author_key": record["author_key"],
        "models": rows,
        "source_address_is_expression_mode_not_person_identity": True,
        "no_claim_human_profile_cannot_change": True,
        "not_human_trait_posterior": True,
    }
