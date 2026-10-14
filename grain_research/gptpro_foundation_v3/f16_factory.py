#!/usr/bin/env python3
"""Public facade for implemented construction and target-specific cold scorers."""

import hashlib
import json
from pathlib import Path

from . import f16_independent_t3 as t3
from . import f16_instrument as t1
from . import f16_state_instrument as t2
from . import f16_t1_source as t1_source


def build_instrument(brief, source_manifest):
    """Build a complete source-frozen bundle, never normalizing application data.

    T1 retains its existing source/models manifest. T2 additionally requires an
    ordered list of all 200 bootstrap file paths/hashes plus the source seed.
    Large callers may use the checkpointed T2 runner; it calls the same source
    template, individual scorer and norm finalizer used here.
    """
    if brief.get("target_id") == "T1":
        return t1.build_instrument(brief, source_manifest)
    if brief.get("T3", {}).get("target_id") == "T3":
        bundle, _checkpoint = t3.build_instrument(brief, source_manifest)
        t3.verify_bundle(bundle)
        return bundle
    if brief != {"target_id": "T2", "world": "core", "target": "current_state"}:
        raise ValueError(
            "unsupported construction target; T3 independent builder required"
        )
    source = source_manifest["source"]
    path = Path(source["path"])
    if t2.digest(path) != source["sha256"]:
        raise ValueError("source manifest hash mismatch")
    fits = source_manifest["bootstrap_files"]
    if len(fits) != 200:
        raise ValueError("T2 full construction requires all 200 source fits")
    root = Path(fits[0]["path"]).parent
    for b, item in enumerate(fits):
        fit_path = Path(item["path"])
        if (
            fit_path != root / f"fit_{b:03d}.json"
            or t2.digest(fit_path) != item["sha256"]
        ):
            raise ValueError("source bootstrap manifest mismatch")
    template = t2.source_template(brief, path, root, source_manifest["seed"])
    groups = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups.setdefault(row["author_id"], []).append(row)
    results = [
        t2.state_moments(groups[author][:48], template) for author in sorted(groups)
    ]
    return t2.finalize_reference(template, results)


def cold_score(raw, bundle):
    """Dispatch only declared implemented targets; never reinterpret a world."""
    if bundle.get("target_id") == "T1":
        if bundle.get("formula_version") == "T1_four_cell_source_mixture_v1":
            return t1_source.cold_score(raw, bundle)
        return t1.cold_score(raw, bundle)
    if bundle.get("target_id") == "T2":
        return t2.cold_score(raw, bundle)
    if bundle.get("target_id") == "T3":
        return t3.cold_score(raw, bundle)
    raise ValueError("unsupported scoring target")


def load_bundle(path, expected_sha256):
    """Deserialize JSON only after verification against an external trust record."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("untrusted bundle hash")
    return json.loads(data)
