#!/usr/bin/env python3
"""Reset-aware readout of externally observed task episodes in a frozen coordinate system."""

import hashlib

import numpy as np
from scipy.special import softmax

from .liquid_frozen_bank import mix


def freeze_clock(examples, axes, basis, train, *, limit=192):
    """examples: (independent source ID, episode ID, matrix); one path per source."""
    axes = list(axes)
    if not basis or not axes or len(set(axes)) != len(axes) or limit < 1:
        raise ValueError("named basis, axes and positive training limit required")
    choices = {}
    seen = set()
    for person, key, seq in examples:
        if (person, key) in seen:
            raise ValueError("repeated source episode")
        seen.add((person, key))
        x = np.asarray(seq, float)
        if x.ndim != 2 or x.shape[1] != len(axes) or not np.isfinite(x).all():
            raise ValueError("invalid source path")
        if len(x) < 2:
            continue
        h = hashlib.sha256(str((person, key)).encode()).hexdigest()
        if person not in choices or h < choices[person][0]:
            choices[person] = (h, x, key)
    people = sorted(choices, key=lambda u: hashlib.sha256(str(u).encode()).digest())[
        :limit
    ]
    if not people:
        raise ValueError("no independent source paths")
    arrays = [choices[u][1] for u in people]
    values = np.concatenate(arrays)
    sd = values.std(0)
    ref = {
        "version": "task-path-ltc-1",
        "axes": axes,
        "basis": basis,
        "mu": values.mean(0).tolist(),
        "scale": np.where(sd > 1e-8, sd, 1.0).tolist(),
        "n_calibration_people": len(people),
        "calibration_ids_digest": hashlib.sha256(
            repr([(u, choices[u][2]) for u in people]).encode()
        ).hexdigest(),
    }
    ref.update(train([encode(v, ref).tolist() for v in arrays]))
    return ref


def encode(x, ref):
    z = (np.asarray(x, float) - ref["mu"]) / ref["scale"]
    # Bound only the reader input, never the original observed readout.
    return softmax(np.column_stack([2 * np.tanh(z / 2), np.zeros(len(z))]), axis=1)


def read_paths(paths, ref, axes, basis, gate, *, batch_events=12000):
    if (
        ref["version"] != "task-path-ltc-1"
        or list(axes) != ref["axes"]
        or basis != ref["basis"]
        or batch_events < 1
    ):
        raise ValueError("incompatible task coordinates")
    arrays = [np.asarray(p, float) for p in paths]
    if any(
        x.ndim != 2 or not len(x) or x.shape[1] != len(axes) or not np.isfinite(x).all()
        for x in arrays
    ):
        raise ValueError(
            "split missing or unordered records before constructing positive-length paths"
        )
    outputs = [None] * len(arrays)
    err = 0.0

    def flush(ids):
        nonlocal err
        if not ids:
            return
        sizes = [len(arrays[j]) for j in ids]
        g, e = gate(encode(np.concatenate([arrays[j] for j in ids]), ref), sizes, ref)
        if (
            g.shape[0] != sum(sizes)
            or g.ndim != 2
            or not np.isfinite(g).all()
            or np.any((g <= 0) | (g >= 1))
        ):
            raise ValueError("invalid gate batch")
        err = max(err, float(e))
        off = 0
        for j, n in zip(ids, sizes, strict=True):
            x = arrays[j]
            a = g[off : off + n].astype(float)
            k = np.exp(np.log(a[1:]).mean(0))
            outputs[j] = (
                mix(x, np.full((n, 1), 0.5)),
                mix(x, a),
                mix(x, np.broadcast_to(k, a.shape)),
            )
            off += n

    ids = []
    total = 0
    for i, x in enumerate(arrays):
        if len(x) == 1:
            outputs[i] = (x.copy(), x.copy(), x.copy())
            continue
        if ids and total + len(x) > batch_events:
            flush(ids)
            ids = []
            total = 0
        ids.append(i)
        total += len(x)
    flush(ids)
    return outputs, err


def crossing(s, threshold):
    """First passage and return in observed order; threshold is frozen on source."""
    s = np.asarray(s, float)
    if (
        s.ndim != 1
        or not len(s)
        or not np.isfinite(s).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError("cannot assign a crossing for an unavailable response or norm")
    hits = np.flatnonzero(s >= threshold)
    return {
        "ever": bool(len(hits)),
        "end": bool(s[-1] >= threshold),
        "first": int(hits[0]) if len(hits) else -1,
        "steps": len(s),
    }


def pull_axis(stories, units, groups, people, *, minimum_people=5):
    """Direction toward a conditional story mean; no reference is not zero expression."""
    stories = np.asarray(stories, float)
    if (
        stories.ndim != 2
        or not np.isfinite(stories).all()
        or len(stories) != len(units)
        or len(groups) != len(people)
        or len(groups) != len(stories)
        or not len(stories)
        or minimum_people < 2
        or any(
            np.ndim(u) != 2
            or len(u) == 0
            or u.shape[1] != stories.shape[1]
            or not np.isfinite(u).all()
            for u in units
        )
    ):
        raise ValueError("unaligned reference")
    center = stories.mean(0)
    out = {}
    groups = np.asarray(groups)
    for k in np.unique(groups):
        idx = np.flatnonzero(groups == k)
        if len(set(np.asarray(people)[idx])) < minimum_people:
            continue
        delta = stories[idx].mean(0) - center
        size = np.linalg.norm(delta)
        if size < 1e-10:
            continue
        direction = delta / size
        marks = [(units[i].mean(0) - center) @ direction for i in idx]
        out[k] = (center, direction, float(np.median(marks)))
    return out


def observed_episode(record):
    """Do not turn non-presentation, withdrawal or recording loss into a scored silence."""
    flag = record["presented"]
    if not isinstance(flag, bool):
        raise TypeError("presentation must be explicitly true or false")
    events = record["events"]
    if not flag:
        if events:
            raise ValueError("non-presented condition cannot have recorded events")
        return "not_presented", []
    if record.get("withheld", False):
        return "withheld_no_score", []
    if not events:
        return "presented_no_record", []
    ords = [e["index"] for e in events]
    if any(type(j) is not int for j in ords) or sorted(set(ords)) != ords:
        raise ValueError("event indices must be unique increasing integers")
    runs = []
    previous = None
    seen = 0
    for e, j in zip(events, ords, strict=True):
        if e["value"] is None:
            previous = None
            continue
        if previous is None or j != previous + 1:
            runs.append([])
        runs[-1].append(e)
        previous = j
        seen += 1
    if not seen:
        return "presented_no_record", []
    if seen != len(events) or len(runs) > 1:
        return "partially_observed", runs
    return "observed", runs
