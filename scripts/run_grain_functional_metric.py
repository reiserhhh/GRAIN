#!/usr/bin/env python3
"""S22: zero-call response-function geometry after the frozen S21 experiment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cross_query_functional import (
    QUERIES,
    cosine_matrix,
    effect_quality,
)
from grain_research.vector_mechanisms.functional_metric import functional_metric
from scripts.run_grain_cross_query_functional import (
    TRANSFER,
    freeze_readers,
    load_packet,
    packet_path,
    reference_arrays,
)
from scripts.run_grain_cross_query_functional import (
    verify as verify_s21,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write


def main():
    """Freeze the source, compute both retained channels and report all queries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=BASE / "cross_query_functional_s21_r1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "functional_metric_s22_r1"
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private outputs must remain in ignored results")
    _m, _s20, _s19, _s12, material = verify_s21(source, args.device)
    old_summary = read(source / "summary.json")
    if old_summary["status"] != "COMPLETE":
        raise ValueError("Complete frozen S21 evaluation required")
    refs = freeze_readers(source, material)
    _packets, _g, d_reference = reference_arrays(source, material)
    effects, d_transfer = [], []
    packet_hashes = {}
    for path in sorted((source / "private_packets").glob("*.json")):
        packet_hashes[path.name] = sha(path)
        packet_hashes[path.with_suffix(".npz").name] = sha(path.with_suffix(".npz"))
    for index in TRANSFER:
        es, ds = [], []
        for query in QUERIES:
            p, a = load_packet(
                packet_path(source, "transfer", index, query),
                material[index],
                query,
                "transfer",
            )
            es.append(p["effect"])
            ds.append(a["d"])
        effects.append(es)
        d_transfer.append(ds)
    effects, d_transfer = np.asarray(effects), np.asarray(d_transfer)
    out.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/functional_metric.py",
        ROOT / "docs/GRAIN_FUNCTIONAL_METRIC_S22_PLAN.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S22_FUNCTIONAL_METRIC_V1",
            "source": str(source),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_summary_sha256": sha(source / "summary.json"),
            "readers_freeze_sha256": sha(source / "readers_frozen.json"),
            "source_packet_hashes": packet_hashes,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "anchor": "quantitative_content",
            "author_weight": "equal",
            "recipient_query_weight": "equal",
            "reference_authors": 12,
            "transfer_authors": 12,
            "new_neural_calls": 0,
            "labels_read": False,
            "max_seconds": 120,
        },
    )
    start = time.monotonic()
    try:
        with threadpool_limits(limits=2):
            result = functional_metric(
                refs, d_reference, QUERIES.index("quantitative_content")
            )
        private, rows = [], []
        maximum = 0.0
        for qi, query in enumerate(QUERIES):
            total = d_transfer[:, qi] @ refs[qi]
            shared = d_transfer[:, qi] @ result["shared"][qi]
            specific = d_transfer[:, qi] @ result["specific"][qi]
            maximum = max(maximum, float(np.max(np.abs(total - shared - specific))))
            for component, scores in (
                ("total", total),
                ("shared", shared),
                ("specific", specific),
            ):
                rows.append(
                    {
                        "query": query,
                        "component": component,
                        **effect_quality(effects[:, qi], scores),
                        "response_rms": float(np.sqrt(np.mean(scores**2))),
                        "reference_anchor_coefficient": float(
                            result["coefficients"][qi]
                        ),
                        "transfer_shared_specific_covariance": float(
                            np.mean(
                                (shared - shared.mean()) * (specific - specific.mean())
                            )
                        ),
                    }
                )
            for ui, index in enumerate(TRANSFER):
                private.append(
                    {
                        "author_id": material[index]["author_id"],
                        "query": query,
                        "effect": effects[ui, qi],
                        "total": total[ui],
                        "shared": shared[ui],
                        "specific": specific[ui],
                    }
                )
        if maximum > 1e-12:
            raise ValueError("Functional component sum failed")
        np.savez_compressed(
            out / "private_functional_components.npz", **result, references=refs
        )
        pd.DataFrame(private).to_csv(
            out / "private_transfer_components.csv", index=False
        )
        pd.DataFrame(rows).to_csv(out / "component_metrics.csv", index=False)
        eigenvalues = result["eigenvalues"]
        summary = {
            "status": "COMPLETE",
            "queries": QUERIES,
            "new_neural_calls": 0,
            "labels_read": False,
            "seconds": time.monotonic() - start,
            "native_cosine": cosine_matrix(refs).tolist(),
            "functional_correlation": result["normalized"].tolist(),
            "functional_gram": result["gram"].tolist(),
            "functional_spectrum": (eigenvalues / eigenvalues.sum()).tolist(),
            "s21_native_gradient_first_energy": old_summary[
                "sensitivity_spectrum_energy"
            ][0],
            "coefficients": result["coefficients"].tolist(),
            "reference_anchor_covariance": result[
                "reference_anchor_covariance"
            ].tolist(),
            "max_additive_error": maximum,
            "metrics": rows,
        }
        if summary["seconds"] > 120:
            summary["resource_budget_exceeded"] = True
        write(out / "summary.json", summary)
        print(
            {
                k: summary[k]
                for k in (
                    "functional_correlation",
                    "functional_spectrum",
                    "coefficients",
                    "reference_anchor_covariance",
                    "max_additive_error",
                    "seconds",
                )
            },
            flush=True,
        )
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    except Exception as exc:
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
