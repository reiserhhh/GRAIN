#!/usr/bin/env python3
"""Independently reconstruct all cached-readout qualification matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import unpack_features
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.run_grain_vector_mechanism_controls import write_json


def reconstruct(h, meta):
    """Use raw second moments and direct per-task arithmetic, not producer helpers."""
    arms = ("raw", "cue_mask", "placebo_mask")
    sites = ("block_04", "block_10", "block_16", "final_norm")
    tasks = ("behavior", "disposition")
    rows = {(r["arm"], r["view"], r["task"]): r for r in meta["readouts"]}
    wanted = {(a, v, t) for a in arms for v in range(3) for t in tasks}
    if (
        len(meta["readouts"]) != 18
        or set(rows) != wanted
        or meta["replay_max"] != 0
        or meta["causal_prefix_max"] != 0
    ):
        raise ValueError("Unqualified or incomplete original observations")
    output = {}
    for a in arms:
        counts = np.asarray(
            [rows[a, v, "behavior"]["body_tokens"] for v in range(3)], dtype=float
        )
        if (
            not np.isfinite(counts).all()
            or np.any(counts <= 0)
            or any(
                rows[a, v, "disposition"]["body_tokens"] != counts[v] for v in range(3)
            )
        ):
            raise ValueError("Invalid independent body-count reconstruction")
        for site in sites:
            m = np.asarray(h[f"{a}_{site}_body_mean"], dtype=float)
            s = np.asarray(h[f"{a}_{site}_body_std"], dtype=float)
            if (
                m.ndim != 2
                or m.shape[0] != 3
                or s.shape != m.shape
                or not np.isfinite(m).all()
                or not np.isfinite(s).all()
                or np.any(s < 0)
            ):
                raise ValueError("Malformed cached moments")
            second = s**2 + m**2
            for name, weights in (
                ("body_token", counts / counts.sum()),
                ("body_equal_view", np.full(3, 1 / 3)),
            ):
                mu = np.sum(weights[:, None] * m, axis=0)
                var = np.sum(weights[:, None] * second, axis=0) - mu**2
                if np.any(var < -1e-9):
                    raise ValueError("Negative reconstructed total variance")
                output[f"{a}__{site}__{name}"] = np.r_[mu, np.sqrt(np.maximum(var, 0))]
            output[f"{a}__{site}__body_variance_parts"] = np.r_[
                m.mean(0), np.sqrt((s**2).mean(0)), m.std(0)
            ]
            b = np.asarray(h[f"{a}_{site}_decision_behavior"], dtype=float)
            d = np.asarray(h[f"{a}_{site}_decision_disposition"], dtype=float)
            for name, value in (
                ("decision_behavior", b),
                ("decision_disposition", d),
                ("decision_task_delta", d - b),
            ):
                if value.shape != m.shape or not np.isfinite(value).all():
                    raise ValueError("Wrong task/view dimension")
                mu = sum(value) / 3
                std = np.sqrt(sum((row - mu) ** 2 for row in value) / 3)
                output[f"{a}__{site}__{name}"] = np.r_[mu, std]
        margins = []
        for task in tasks:
            values = np.asarray([rows[a, v, task]["margin"] for v in range(3)])
            margins.extend([float(values.mean()), float(values.std())])
        output[f"{a}__logit_margins"] = np.asarray(margins)
    return output


def main():
    """Read only the12 saved technical caches; reject altered identities or outputs."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("qualification-dir", "technical-dir", "technical-views", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private audit output required")
    out.mkdir(parents=True, exist_ok=False)
    m, _ = verified_packet(a.qualification_dir)
    parent = json.loads((a.technical_dir / "manifest.json").read_text())
    source = json.loads(a.technical_views.read_text())
    status = json.loads((a.technical_dir / "run_status.json").read_text())
    if (
        m["protocol"] != "VM05_CACHED_READOUT_QUALIFICATION_V1"
        or m["labels_read"]
        or m["technical_parent_identity"] != parent["identity"]
        or material_hash({k: v for k, v in parent.items() if k != "identity"})
        != parent["identity"]
        or parent["technical"] is not True
        or parent["stage"] != "qwen"
        or parent["labels_read"]
        or status["status"] != "COMPLETE"
        or status["parent_identity"] != parent["identity"]
        or parent["source_material_sha256"] != material_hash(source)
        or digest(a.technical_views) != m["technical_views_sha256"]
        or len(source) != 12
    ):
        raise ValueError("Different technical evidence or observer parent")
    ids = [b["author_id"] for b in source]
    if len(set(ids)) != 12 or parent["authors"] != ids:
        raise ValueError("Technical author identity differs")
    with np.load(a.qualification_dir / "private_readouts.npz", allow_pickle=False) as z:
        saved = {k: z[k] for k in z.files}
    if saved.pop("author_ids").tolist() != ids:
        raise ValueError("Derived array authors differ")
    inputs = json.loads(
        (a.qualification_dir / "private_input_commitments.json").read_text()
    )
    if set(inputs) != set(ids):
        raise ValueError("Missing source commitments")
    expected = {}
    for uid in ids:
        path = a.technical_dir / "private_cache" / uid
        meta = json.loads(path.with_suffix(".json").read_text())
        if (
            meta["author_id"] != uid
            or digest(path.with_suffix(".json")) != inputs[uid]["json"]
            or digest(path.with_suffix(".npz")) != inputs[uid]["npz"]
            or inputs[uid]["npz"] != meta["feature_sha256"]
        ):
            raise ValueError("Changed technical input")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            h = unpack_features(
                {k: z[k] for k in z.files}, meta.get("feature_aliases", {})
            )
        for k, v in reconstruct(h, meta).items():
            expected.setdefault(k, []).append(v)
    if set(saved) != set(expected) or len(saved) != 75:
        raise ValueError("Missing or unexpected derived channel")
    records = []
    for k, values in expected.items():
        wanted = np.stack(values)
        if wanted.shape != saved[k].shape or not np.isfinite(saved[k]).all():
            raise ValueError("Changed derived shape/value")
        error = float(np.max(abs(saved[k] - wanted)))
        if error > 1e-9:
            raise ValueError("Independent readout reconstruction differs")
        records.append(
            {
                "channel": k,
                "authors": 12,
                "dimensions": wanted.shape[1],
                "max_absolute_error": error,
            }
        )
    pd.DataFrame(records).to_csv(out / "reconstruction_by_channel.csv", index=False)
    archive_source(ROOT, out / "source_snapshot.zip")
    summary = {
        "status": "COMPLETE",
        "qualification_identity": m["identity"],
        "authors": 12,
        "matrices_reconstructed": len(records),
        "author_channel_rows": 12 * len(records),
        "maximum_absolute_error": max(r["max_absolute_error"] for r in records),
        "labels_read": False,
        "neural_forwards": 0,
        "model_refits": 0,
        "scope": "Independent cached algebra replay, not personality or full-source acceptance",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
