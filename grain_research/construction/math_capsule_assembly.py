#!/usr/bin/env python3
"""Bind event partitions and separate compiler coupling from reader coupling."""

import numpy as np

BLOCKS = ((0, 1), (2, 3))
MODES = ("first", "second", "sum")


def partition_prompt(decoded, block):
    """Keep exact excerpt strings and original slot labels in a registered split.

    This changes text length, positions and contextual encoding. It is not a
    lossless presentation of already frozen vectors or a noise-removal operation.
    """
    if block not in range(2):
        raise ValueError("Only the two fixed ordered blocks are registered")
    markers = [f"Excerpt {i + 1}:\n" for i in range(4)]
    if any(decoded.count(m) != 1 for m in markers):
        raise ValueError("Unambiguous four-excerpt boundaries required")
    if decoded.count("Evidence ends here.") != 1:
        raise ValueError("Unambiguous evidence boundary required")
    starts = [decoded.index(m) for m in markers]
    end = decoded.index("Evidence ends here.")
    if starts != sorted(starts) or starts[-1] >= end:
        raise ValueError("Ordered original excerpts required")
    stops = starts[1:] + [end]
    pieces = [decoded[a:b] for a, b in zip(starts, stops, strict=True)]
    if decoded[: starts[0]] + "".join(pieces) + decoded[end:] != decoded:
        raise ValueError("Lossless original reconstruction required")
    text = decoded[: starts[0]] + "".join(pieces[i] for i in BLOCKS[block])
    text += decoded[end:]
    return text, pieces


def event_partition(events):
    """Register disjoint evidence identities; six contracts are not six events."""
    if len(events) != 4 or len(set(events)) != 4:
        raise ValueError("Four distinct source event identities required")
    return [[events[i] for i in block] for block in BLOCKS]


def assemble(parts, joint):
    """Export observed joint coupling without fitting it or changing the joint.

    The equality joint = first + second + coupling is an accounting identity,
    not empirical evidence that independent compilation is sufficient.
    """
    p, j = np.asarray(parts, float), np.asarray(joint, float)
    if p.shape != (2, *j.shape) or not np.isfinite(p).all() or not np.isfinite(j).all():
        raise ValueError(
            "Two finite aligned parts and a source-observed joint required"
        )
    summed = p[0] + p[1]
    return summed, j - summed


def action_decomposition(joint, first, second, summed):
    """Separate observed compiler substitution from nonlinear recipient assembly."""
    j, a, b, s = (np.asarray(x, float) for x in (joint, first, second, summed))
    if not (j.shape == a.shape == b.shape == s.shape):
        raise ValueError("Same author, target and observer coordinates required")
    if not all(np.isfinite(x).all() for x in (j, a, b, s)):
        raise ValueError("Finite observed actions required")
    return {"compiler": j - s, "reader": s - a - b, "total": j - a - b}
