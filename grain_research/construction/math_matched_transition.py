#!/usr/bin/env python3
"""Observation-action identities without equating a common destination to an action."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .math_response_function import response_components


def transition_address(messages, start_task, destination_task):
    """Index a task replacement, retaining system and ordered choice protocol.

    The body is an evidence unit, not part of the shared operation address.
    Matching this address is necessary for the declared replacement comparison;
    it does not assert population invariance or isolate semantic task effects.
    """
    if (
        len(messages) != 4
        or [m.get("role") for m in messages] != ["system", "user", "assistant", "user"]
        or not all(isinstance(m.get("content"), str) for m in messages)
        or not isinstance(start_task, str)
        or not isinstance(destination_task, str)
        or not start_task
        or not destination_task
    ):
        raise ValueError("Four addressed messages and two nonempty tasks required")
    payload = {
        "system": messages[0]["content"],
        "start_task": start_task,
        "destination_task": destination_task,
        "ordered_choice_prompt": messages[3]["content"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def action_components(features, start, destination, groups):
    """Construct total/within/between cross-moments of one observed transition."""
    start, destination = np.asarray(start, float), np.asarray(destination, float)
    if start.shape != destination.shape:
        raise ValueError("Start and destination must address the same observations")
    return response_components(features, destination - start, groups)


def support_account(source_addresses, application_addresses):
    """Count exact operation support, without treating missing addresses as zeros."""
    source, application = set(source_addresses), list(application_addresses)
    if not source or not application:
        raise ValueError("Source and application operation addresses required")
    supported = sum(a in source for a in application)
    return {
        "source_unique": len(source),
        "application_unique": len(set(application)),
        "shared_unique": len(source & set(application)),
        "application_units": len(application),
        "supported_application_units": supported,
        "unsupported_application_units": len(application) - supported,
    }
