#!/usr/bin/env python3
"""Single-pass interchange for evidence, coverage and opportunity records."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import fields, replace
from itertools import chain
from typing import TYPE_CHECKING, TextIO

from .evidence_panel import Coverage, EvidenceEvent, EvidencePanel, Opportunity, Record

if TYPE_CHECKING:
    from .evidence_text import TextRecord

VERSION = "grain.evidence-stream.v1"
_TYPES = {"event": EvidenceEvent, "coverage": Coverage, "opportunity": Opportunity}
_NAMES = {v: k for k, v in _TYPES.items()}
_FIELDS = {cls: tuple(f.name for f in fields(cls)) for cls in _NAMES}
MAX_LINE = 1024 * 1024


def _objects(source: TextIO, max_line: int) -> Iterator[tuple[int, dict]]:
    if type(max_line) is not int or max_line < 1:
        raise ValueError("max_line must be a positive integer")
    line_no = 0
    while raw := source.readline(max_line + 1):
        line_no += 1
        if len(raw) > max_line:
            raise ValueError(f"line {line_no}: record exceeds the line limit")
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise TypeError(f"line {line_no}: expected a JSON object")
        yield line_no, item


def read_panel(source: TextIO, *, max_line: int = MAX_LINE) -> EvidencePanel:
    """Read from a text stream without retaining decoded corpus rows or raw text."""
    objects = _objects(source, max_line)
    n, head = next(objects, (1, {}))
    if (
        set(head) != {"schema_version", "channels"}
        or head["schema_version"] != VERSION
        or not isinstance(head["channels"], dict)
    ):
        raise ValueError(f"line {n}: expected the {VERSION} header")

    def rows():
        for line, obj in objects:
            typ = obj.pop("type", None)
            if not isinstance(typ, str) or typ not in _TYPES:
                raise ValueError(f"line {line}: unknown record type")
            if (
                typ == "event"
                and (values := obj.get("values")) is not None
                and (
                    not isinstance(values, list)
                    or any(type(x) not in (int, float) for x in values)
                )
            ):
                raise ValueError(
                    f"line {line}: values must be a list of numbers or null"
                )
            try:
                yield _TYPES[typ](**obj)
            except TypeError as exc:
                raise ValueError(f"line {line}: invalid {typ} fields: {exc}") from exc

    return EvidencePanel.from_records(head["channels"], rows())


def write_stream(
    target: TextIO, channels: Mapping[str, Sequence[str]], records: Iterable[Record]
) -> None:
    """Serialize typed records as they arrive; no raw text or whole-corpus object."""

    def save(obj):
        target.write(
            json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )

    save({"schema_version": VERSION, "channels": dict(channels)})
    for item in records:
        typ = _NAMES.get(type(item))
        if typ is None:
            raise TypeError("unrecognized evidence record type")
        save({"type": typ, **{k: getattr(item, k) for k in _FIELDS[type(item)]}})


def namespace_source(
    namespace: str,
    channels: Mapping[str, Sequence[str]],
    records: Iterable[Record],
    *,
    actor_aliases: Mapping[str, str] | None = None,
    shared_channels: Sequence[str] = (),
    shared_measures: Sequence[str] = (),
) -> tuple[dict[str, Sequence[str]], Iterator[Record]]:
    """Keep source-local IDs and coverage separate unless the caller supplies a link."""
    if not namespace or not all(
        c.isascii() and (c.isalnum() or c in "-_.") for c in namespace
    ):
        raise ValueError(
            "namespace must contain only ASCII letters, digits, '-', '_', or '.'"
        )
    aliases = dict(actor_aliases or {})
    if any(not isinstance(v, str) or not v.strip() for v in aliases.values()):
        raise ValueError("actor aliases must be non-empty strings")
    align_channels, align_measures = set(shared_channels), set(shared_measures)
    if not align_channels <= channels.keys():
        raise ValueError("shared channel is not declared in the source")

    def scoped(v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("expected a non-empty source identifier")
        return f"{namespace}:{v}"

    channel_map = {k: k if k in align_channels else scoped(k) for k in channels}
    if len(set(channel_map.values())) != len(channel_map):
        raise ValueError("a shared channel name collides with a scoped channel")
    schema = {channel_map[k]: v for k, v in channels.items()}

    def events():
        for row in records:
            if isinstance(row, Coverage):
                yield replace(
                    row,
                    window=scoped(row.window),
                    context=scoped(row.context) if row.context is not None else None,
                )
                continue
            if not isinstance(row, (EvidenceEvent, Opportunity)):
                raise TypeError("unrecognized evidence record type")
            ids = {
                "id": scoped(row.id),
                "window": scoped(row.window),
                "context": scoped(row.context),
                "actor": f"linked:{aliases[row.actor]}"
                if row.actor in aliases
                else f"source:{scoped(row.actor)}",
            }
            if isinstance(row, EvidenceEvent):
                yield replace(
                    row,
                    **ids,
                    channel=channel_map[row.channel],
                    observation_id=scoped(row.observation_id)
                    if row.observation_id is not None
                    else None,
                )
            else:
                yield replace(
                    row,
                    **ids,
                    measure=row.measure
                    if row.measure in align_measures
                    else scoped(row.measure),
                )

    return schema, events()


def combine_sources(
    *sources: tuple[Mapping[str, Sequence[str]], Iterable[Record]],
) -> EvidencePanel:
    """Build a single panel without silently replacing a shared coordinate declaration."""
    channels = {}
    for schema, _ in sources:
        for name, labels in schema.items():
            value = tuple(labels)
            if name in channels and channels[name] != value:
                raise ValueError(
                    f"incompatible coordinate order for shared channel {name!r}"
                )
            channels[name] = value
    return EvidencePanel.from_records(
        channels, chain.from_iterable(rows for _, rows in sources)
    )


def text_csv(
    source: TextIO,
    *,
    columns: Mapping[str, str],
    constants: Mapping[str, str] | None = None,
) -> Iterator[TextRecord]:
    """Map incoming CSV column names to TextRecord fields; retain no raw corpus list."""
    from .evidence_text import TextRecord

    fixed = dict(constants or {})
    required = {"id", "window", "actor", "context", "text"}
    if set(columns) & fixed.keys() or set(columns) | fixed.keys() != required:
        raise ValueError("provide each of id, window, actor, context and text once")
    if "text" in fixed:
        raise ValueError("text must come from a CSV column")
    reader = csv.DictReader(source, strict=True)
    fields = reader.fieldnames or []
    if len(set(fields)) != len(fields) or not set(columns.values()).issubset(fields):
        raise ValueError("CSV has missing required or duplicate column names")
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"line {reader.line_num}: CSV column count mismatch")
        yield TextRecord(
            **fixed, **{name: row[column] for name, column in columns.items()}
        )
