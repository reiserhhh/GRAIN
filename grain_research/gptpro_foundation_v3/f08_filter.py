"""Source-transition filtering for qualified, uncertain synthetic evidence."""

import numpy as np


def filter_step(previous, transition, pole_probability, qualification):
    """Return joint old/new states and next pair, without deleting missing times."""
    previous = np.asarray(previous, dtype=float)
    transition = np.asarray(transition, dtype=float)
    if previous.shape != (2, 2) or transition.shape != (2, 2):
        raise ValueError("two binary lag states required")
    if (
        not np.isfinite(previous).all()
        or (previous < 0).any()
        or not np.isclose(previous.sum(), 1)
    ):
        raise ValueError("normalized finite prior required")
    if not np.isfinite(transition).all() or ((transition < 0) | (transition > 1)).any():
        raise ValueError("finite transition probabilities required")
    if not 0 <= pole_probability <= 1 or not 0 <= qualification <= 1:
        raise ValueError("finite pole and qualification probabilities required")
    likelihood = (
        qualification * np.array([1 - pole_probability, pole_probability])
        + (1 - qualification) / 2
    )
    joint = (
        previous[:, :, None]
        * np.stack([1 - transition, transition], axis=-1)
        * likelihood
    )
    normalizer = joint.sum()
    if normalizer <= 0:
        raise ValueError("observation impossible under frozen transition")
    joint /= normalizer
    return {
        "joint": joint,
        "pair": joint.sum(axis=0),
        "evidence_mass": float(qualification),
        "normalizer": float(normalizer),
    }


def history_filter(poles, qualifications, transition, initial_pair):
    """Preserve each time step; all-empty evidence is explicitly model-only."""
    if len(poles) != len(qualifications):
        raise ValueError("unaligned history")
    pair = np.asarray(initial_pair, dtype=float)
    steps = []
    for p, q in zip(poles, qualifications, strict=True):
        step = filter_step(pair, transition, p, q)
        pair = step["pair"]
        steps.append(step)
    evidence = float(np.sum(qualifications))
    return {
        "steps": steps,
        "pair": pair,
        "evidence_mass": evidence,
        "status": "prior_only" if evidence == 0 else "source_transition_filtered",
    }
