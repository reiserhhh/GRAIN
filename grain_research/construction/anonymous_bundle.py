"""GRAIN-C4-01 Part A: complete cold bundle for source-learned anonymous coordinates.

Carries every source-fitted transformation needed to score from cached encoder vectors:
encoder spec, L2 normalization, raw-space directions (C3-02), coordinate modes
(composition_inclusive; within_venue = minus source_train venue offset, pooled offset for venues
unseen or rare in source), TOP20 context rule, per-mode moments and source reference. Scores are
anonymous response coordinates; venue is a coarse author-chosen condition; no names.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import declared_target as DT

VERSION = "GRAIN_anonymous_bundle_v1"
MODES = ("composition_inclusive", "within_venue")
MIN_VENUE_EVENTS = 5


def code_hash():
    files = [Path(__file__), Path(DT.__file__)]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, float)).tobytes()).hexdigest()


def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.isfinite(v).all() or (n == 0).any():
        raise ValueError("finite nonzero encoder vectors required")
    return v / n


def venue_offsets(ev_src, venues_src, min_count=MIN_VENUE_EVENTS):
    """ev_src (n, K) raw coordinates and venues (n,) of source_train history events."""
    cnt = Counter(list(venues_src))
    pooled = ev_src.mean(0)
    table = {v: ev_src[np.asarray(venues_src) == v].mean(0).tolist() for v, c in cnt.items() if c >= min_count}
    return table, pooled.tolist()


def apply_mode(ev, venues, table, pooled, mode):
    if mode == "composition_inclusive":
        return ev, np.zeros(len(venues), bool)
    unseen = np.array([v not in table for v in venues])
    sub = np.array([table.get(v, pooled) for v in venues]).reshape(ev.shape)
    return ev - sub, unseen


def build_bundle(directions, families, encoder, rule, table, pooled, fits, reference, provenance):
    d = np.asarray(directions, float)
    return {"version": VERSION, "code_hash": code_hash(), "encoder": encoder, "normalization": "L2 per comment vector",
            "directions": d.tolist(), "directions_sha256": array_hash(d), "families": families,
            "context_rule": rule, "venue_offsets": table, "pooled_offset": pooled,
            "venue_table_sha256": hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest(),
            "min_venue_events": MIN_VENUE_EVENTS, "fits": fits, "reference": reference, "provenance": provenance,
            "unseen_venue_rule": "within_venue uses the pooled source_train offset and flags the event",
            "interpretation": "anonymous source-learned response coordinates of published comments; venue is a coarse, "
                              "author-chosen condition; within_venue is not topic-free; not named, not causal, not psychological"}


def verify_bundle(b):
    if b.get("version") != VERSION or b.get("code_hash") != code_hash():
        raise ValueError("changed anonymous-bundle code or version")
    if array_hash(b["directions"]) != b["directions_sha256"]:
        raise ValueError("directions tampered")
    if hashlib.sha256(json.dumps(b["venue_offsets"], sort_keys=True).encode()).hexdigest() != b["venue_table_sha256"]:
        raise ValueError("venue offsets tampered")
    if set(b["fits"]) != set(MODES):
        raise ValueError("missing coordinate mode")


def evidence(bundle, vectors):
    return normalize(vectors) @ np.asarray(bundle["directions"])


def cold_score(bundle, records):
    verify_bundle(bundle)
    k = len(bundle["families"])
    out = []
    for rec in records:
        if set(rec) != {"author_key", "events"}:
            raise ValueError("record must contain only author_key and events")
        ids = [e["event_id"] for e in rec["events"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event identity")
        for e in rec["events"]:
            if set(e) != {"event_id", "venue", "encoder_id", "vector"}:
                raise ValueError("event must carry id, venue, encoder id and vector only")
            if e["encoder_id"] != bundle["encoder"]["id"]:
                raise ValueError("encoder version does not match the bundle")
        venues = [e["venue"] for e in rec["events"]]
        raw = evidence(bundle, [e["vector"] for e in rec["events"]]) if rec["events"] else np.zeros((0, k))
        c = DT.contexts(venues, bundle["context_rule"])
        res = {"author_key": rec["author_key"]}
        for mode in MODES:
            y, unseen = apply_mode(raw, venues, bundle["venue_offsets"], bundle["pooled_offset"], mode)
            res[mode] = DT.score_author(bundle["fits"][mode], y.reshape(-1, k), c)
            res[mode]["unseen_venue_share"] = float(unseen.mean()) if len(unseen) else None
        out.append(res)
    return out


def save_bundle(b, path):
    data = json.dumps(b, sort_keys=True).encode()
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_bundle(path, sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("untrusted bundle hash")
    return json.loads(data)
