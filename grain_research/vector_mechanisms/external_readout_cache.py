#!/usr/bin/env python3
"""Read declared task/pooling features only from a complete frozen observer queue."""

from __future__ import annotations

import json

import numpy as np

from .controls import material_hash
from .external_material import unpack_features
from .external_readouts import channel, margin_features
from .ready_comparison import require_observer_completion


def cached_readout_matrix(bundles, source, common, vector_dir, arm, site, kind):
    """Return exact requested author order; no imputation, fitting or neural forward."""
    from scripts.run_grain_vector_mechanism_external_queue import verify_completed

    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Changed readout material")
    ids = [b["author_id"] for b in bundles]
    source_ids = [b["author_id"] for b in source]
    if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(source_ids):
        raise ValueError("Readout rows require unique common authors")
    if kind == "logit_margins" and site != "block_10":
        raise ValueError(
            "Logit margins are site-independent; use one block10 identifier"
        )
    queue = require_observer_completion(vector_dir, common["material_sha256"])
    paths = {}
    for i in range(24):
        folder = vector_dir / "qwen" / f"shard_{i:02d}"
        verify_completed(folder, source, common, "qwen", 24, i, queue["code_sha256"])
        m = json.loads((folder / "manifest.json").read_text())
        if set(paths) & set(m["authors"]):
            raise ValueError("Duplicate neural cache author")
        paths.update({u: folder / "private_cache" / u for u in m["authors"]})
    if set(paths) != set(source_ids):
        raise ValueError("Neural cache union differs from common source")
    rows = []
    for uid in ids:
        path = paths[uid]
        meta = json.loads(path.with_suffix(".json").read_text())
        if kind == "logit_margins":
            rows.append(margin_features(meta, arm))
        else:
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                h = unpack_features(
                    {k: z[k] for k in z.files}, meta.get("feature_aliases", {})
                )
            rows.append(channel(h, meta, arm, site, kind))
    return np.stack(rows)
