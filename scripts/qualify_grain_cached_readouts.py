#!/usr/bin/env python3
"""Qualify declared readout transforms on the existing12 no-label technical cases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import pooled_moments
from grain_research.vector_mechanisms.external_material import unpack_features
from grain_research.vector_mechanisms.external_readouts import (
    ARMS,
    KINDS,
    SITES,
    author_channels,
    readout_table,
)
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.run_grain_vector_mechanism_controls import write_json


def technical_source(folder, views):
    """Verify exact technical material and pinned pre-answer observer provenance."""
    m = json.loads((folder / "manifest.json").read_text())
    s = json.loads((folder / "run_status.json").read_text())
    source = json.loads(views.read_text())
    if (
        m["protocol"] != "GRAIN_VM05_FEATURES_V1"
        or m["stage"] != "qwen"
        or m["technical"] is not True
        or m["labels_read"]
        or material_hash({k: v for k, v in m.items() if k != "identity"})
        != m["identity"]
        or s["status"] != "COMPLETE"
        or s["parent_identity"] != m["identity"]
        or len(source) != 12
        or s["authors"] != 12
        or m["source_material_sha256"] != material_hash(source)
        or m["authors"] != [b["author_id"] for b in source]
        or len(set(m["authors"])) != 12
        or any(
            sum(b["source_group"] == g for b in source) != 4
            for g in ("big5_only", "mbti_only", "bridge")
        )
    ):
        raise ValueError("Need the exact completed12-author technical packet")
    model = m["model"]
    if (
        model["model"] != "Qwen/Qwen2.5-1.5B-Instruct"
        or model["revision"] != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        or tuple(model["selected_sites"]) != SITES
    ):
        raise ValueError("Wrong cached Qwen model or sites")
    paths = {
        p.stem
        for p in (folder / "private_cache").glob("*.npz")
        if not p.name.startswith("._")
    }
    if paths != set(m["authors"]):
        raise ValueError("Incomplete technical author cache")
    return source, m


def main():
    """Save derived matrices and aggregate algebra diagnostics without any outcome."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("technical-dir", "technical-views", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private technical results only")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source, parent = technical_source(a.technical_dir, a.technical_views)
    fields = {
        "protocol": "VM05_CACHED_READOUT_QUALIFICATION_V1",
        "technical_parent_identity": parent["identity"],
        "technical_views_sha256": digest(a.technical_views),
        "model": parent["model"],
        "sites": SITES,
        "kinds": KINDS,
        "arms": ARMS,
        "scoring_code": {
            name: digest(ROOT / name)
            for name in (
                "grain_research/vector_mechanisms/external_readouts.py",
                "grain_research/vector_mechanisms/external_material.py",
                "grain_research/vector_mechanisms/external_heads.py",
                "scripts/qualify_grain_cached_readouts.py",
            )
        },
        "labels_read": False,
        "neural_forwards": 0,
        "model_refits": 0,
        "scope": "Original12 technical authors only; no full-source comparison",
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    archive_source(ROOT, out / "source_snapshot.zip")
    arrays, diagnostics, inputs = {}, [], {}
    try:
        for b in source:
            uid = b["author_id"]
            path = a.technical_dir / "private_cache" / uid
            meta = json.loads(path.with_suffix(".json").read_text())
            if (
                meta["author_id"] != uid
                or digest(path.with_suffix(".npz")) != meta["feature_sha256"]
            ):
                raise ValueError("Changed technical author/cache")
            inputs[uid] = {
                "npz": meta["feature_sha256"],
                "json": digest(path.with_suffix(".json")),
            }
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                h = unpack_features(
                    {k: z[k] for k in z.files}, meta.get("feature_aliases", {})
                )
            channels = author_channels(h, meta)
            for key, value in channels.items():
                arrays.setdefault(key, []).append(value)
            table = readout_table(meta)
            for arm in ARMS:
                counts = [table[arm, v, "behavior"]["body_tokens"] for v in range(3)]
                for site in SITES:
                    expected = pooled_moments(
                        h[f"{arm}_{site}_body_mean"],
                        h[f"{arm}_{site}_body_std"],
                        counts,
                    )
                    actual = channels[f"{arm}__{site}__body_token"]
                    error = float(np.max(abs(actual - expected)))
                    parts = channels[f"{arm}__{site}__body_variance_parts"]
                    equal = channels[f"{arm}__{site}__body_equal_view"]
                    n = len(parts) // 3
                    variance_error = float(
                        np.max(
                            abs(
                                parts[n : 2 * n] ** 2
                                + parts[2 * n :] ** 2
                                - equal[n:] ** 2
                            )
                        )
                    )
                    if error > 1e-9 or variance_error > 1e-9:
                        raise ValueError("Moment/readout reconstruction mismatch")
                    diagnostics.append(
                        {
                            "arm": arm,
                            "site": site,
                            "token_pool_max_error": error,
                            "variance_identity_max_error": variance_error,
                        }
                    )
            if time.monotonic() - started > 600:
                raise TimeoutError("Technical offline transformation exceeded10min")
        packed = {k: np.stack(v) for k, v in arrays.items()}
        np.savez_compressed(
            out / "private_readouts.npz",
            author_ids=np.asarray([b["author_id"] for b in source]),
            **packed,
        )
        write_json(out / "private_input_commitments.json", inputs)
        frame = pd.DataFrame(diagnostics)
        frame.groupby(["arm", "site"]).agg(
            authors=("token_pool_max_error", "size"),
            token_pool_max_error=("token_pool_max_error", "max"),
            variance_identity_max_error=("variance_identity_max_error", "max"),
        ).to_csv(out / "numerical_diagnostics.csv")
        summary = {
            "status": "COMPLETE",
            "parent_identity": fields["identity"],
            "authors": 12,
            "derived_feature_matrices": len(packed),
            "author_channel_rows": sum(len(v) for v in packed.values()),
            "token_pool_max_error": float(frame.token_pool_max_error.max()),
            "variance_identity_max_error": float(
                frame.variance_identity_max_error.max()
            ),
            "labels_read": False,
            "neural_forwards": 0,
            "model_refits": 0,
            "seconds": time.monotonic() - started,
            "outputs_sha256": {
                name: digest(out / name)
                for name in (
                    "private_readouts.npz",
                    "private_input_commitments.json",
                    "numerical_diagnostics.csv",
                )
            },
            "open_requirements": [
                "independent reconstruction",
                "full-source and same-ready refits",
                "true same-current history effects",
            ],
        }
        write_json(out / "summary.json", summary)
        write_json(out / "run_status.json", summary)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "outputs_sha256"}),
            flush=True,
        )
    except Exception as error:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
                "exception_type": type(error).__name__,
                "labels_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
