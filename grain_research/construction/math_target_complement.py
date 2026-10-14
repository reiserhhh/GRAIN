#!/usr/bin/env python3
"""Target-qualified linear observation complements of a frozen factor map."""

import numpy as np

VERSION = "HM184_target_complement_v1"
SPACES = ("full", "retained", "complement")


def fixed_basis(mapping):
    """Factor an existing independent observation bank without rank selection."""
    a = np.asarray(mapping, float)
    if a.ndim != 2 or a.shape[0] < a.shape[1] or not np.isfinite(a).all():
        raise ValueError("Finite tall observation map required")
    singular = np.linalg.svd(a, compute_uv=False)
    if singular[-1] <= 100 * np.finfo(float).eps * singular[0] * max(a.shape):
        raise ValueError("Frozen observation columns are not independent")
    return np.linalg.qr(a, mode="reduced")


def split_fields(fields, basis):
    """Keep a fixed reader's full, observable and complementary linear values."""
    d, q = np.asarray(fields, float), np.asarray(basis, float)
    if d.ndim < 1 or q.ndim != 2 or d.shape[-1] != len(q):
        raise ValueError("Reader and fixed basis dimensions differ")
    if not np.isfinite(d).all() or not np.isfinite(q).all():
        raise ValueError("Finite reader and basis required")
    retained = (d @ q) @ q.T
    return np.stack((d, retained, d - retained), axis=-2)


def coordinate_fields(fields, basis, triangular):
    """Represent retained readings using the unchanged original factor scores."""
    d, q, r = map(np.asarray, (fields, basis, triangular))
    flat = (d @ q).reshape(-1, q.shape[1])
    return np.linalg.solve(r, flat.T).T.reshape(*d.shape[:-1], q.shape[1])


def cold_record(fields, semantic, record):
    """Score numeric events only; source labels and model answers are forbidden."""
    f, s = np.asarray(fields, float), np.asarray(semantic, float)
    if f.shape != (2, 5, 4, 3, 1024) or s.shape != (3, 1024):
        raise ValueError("Registered complete reader family required")
    if not np.isfinite(f).all() or not np.isfinite(s).all():
        raise ValueError("Nonfinite frozen reader")
    if set(record) != {"frame_key", "events"} or not isinstance(
        record["frame_key"], str
    ):
        raise ValueError("Only addressed numeric observations permitted")
    ids, slots, vectors = [], [], []
    for e in record["events"]:
        if set(e) != {"event_id", "slot", "vector"}:
            raise ValueError("Unexpected evidence fields")
        if not isinstance(e["event_id"], str) or type(e["slot"]) is not int:
            raise ValueError("Event identities and integer slots required")
        if e["slot"] not in range(12):
            raise ValueError("Original12-event slots required")
        ids.append(e["event_id"])
        slots.append(e["slot"])
        vectors.append(e["vector"])
    if len(set(ids)) != len(ids) or len(set(slots)) != len(slots):
        raise ValueError("Repeated evidence event or slot")
    if not ids:
        return {
            "frame_key": record["frame_key"],
            "count": 0,
            "source": None,
            "semantic": None,
        }
    x = np.asarray(vectors, float)
    if x.shape != (len(ids), 1024) or not np.isfinite(x).all():
        raise ValueError("Finite1024-coordinate observations required")
    mean = x.mean(0)
    return {
        "frame_key": record["frame_key"],
        "count": len(ids),
        "source": np.einsum("vqcsd,d->vqcs", f, mean).tolist(),
        "semantic": (s @ mean).tolist(),
    }
