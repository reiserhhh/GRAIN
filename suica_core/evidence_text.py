#!/usr/bin/env python3
"""Frozen native GRAIN text adapter for the common observation panel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

from .evidence_panel import EvidenceEvent
from .liquid_instrument import _AtlasLocator, digest


@dataclass(frozen=True, slots=True)
class TextRecord:
    id: str
    window: str
    actor: str
    context: str
    text: str


def native_text_evidence(atlas: dict, records: Iterable[TextRecord]):
    """Return channel definitions and records, with no text in the output."""
    schema, rows = iter_native_text_evidence(atlas, records)
    return schema, list(rows)


def iter_native_text_evidence(
    atlas: dict, records: Iterable[TextRecord], *, batch_size: int = 512
):
    """Yield native readings in bounded batches from a single compiled reference."""
    if type(batch_size) is not int or not 1 <= batch_size <= 512:
        raise ValueError("batch_size must be an integer between 1 and 512")
    channel = f"native-text:{digest(atlas)}"
    schema = {channel: tuple(atlas["labels"])}

    def stream():
        pending = iter(records)
        locator = None
        while rows := list(islice(pending, batch_size)):
            if locator is None:
                locator = _AtlasLocator(atlas)
            vectors, covered = locator.locate([row.text for row in rows])
            for row, v, ok in zip(rows, vectors, covered, strict=True):
                yield EvidenceEvent(
                    row.id,
                    row.window,
                    row.actor,
                    row.context,
                    channel,
                    v.tolist() if ok else None,
                )

    return schema, stream()
