#!/usr/bin/env python3
"""HM40S source-fixed prefix sensitivity; preserve all frozen primary artifacts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_cancellation_transport import (
    parts,
    signed_summary,
)
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_cancellation_transport import arrays, coverage, paired
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_target_active_support import load
from scripts.verify_grain_math_cancellation_transport import alternate, scalar_moments
from scripts.verify_grain_math_crossed_field import table

PARENT = ROOT / "results/grain_math_response_v1/cancellation_transport_r1"
VERIFY = (
    ROOT
    / "results/grain_math_response_v1/cancellation_transport_verification_r1/summary.json"
)
OUT = ROOT / "results/grain_math_response_v1/compensation_resolution_sensitivity_r1"
CODE = (
    "scripts/run_grain_math_compensation_resolution.py",
    "tests/test_grain_math_compensation_resolution.py",
    "docs/GRAIN_COMPENSATION_RESOLUTION_SENSITIVITY_V1.md",
)


def prefix_view(metadata, bank):
    """Only a source-determined contiguous numerical-resolution prefix is permitted."""
    if metadata["role"] != "source" or metadata["application_nodes_loaded"] != 0:
        raise ValueError("source-only numerical resolution required")
    flags = bank["resolved_against_proxy"][: metadata["rank"]].astype(bool)
    rank = int(flags.sum())
    if rank != metadata["resolved_against_proxy"] or not np.array_equal(
        flags, np.arange(len(flags)) < rank
    ):
        raise ValueError("a source-fixed contiguous prefix is required")
    if (
        bank["basis"].shape != bank["random_basis"].shape
        or bank["basis"].shape[1] != metadata["rank"]
    ):
        raise ValueError("same-rank frozen controls required")
    return bank["basis"][:, :rank], bank["random_basis"][:, :rank], rank


def main():
    """Freeze prefixes first, then perform one bounded sensitivity and alternate replay."""
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError("preserve existing companion")
    m = check(PARENT)
    status = read_json(PARENT / "run_status.json")
    verification = read_json(VERIFY)
    if (
        status["status"] != "EVALUATED"
        or verification["status"] != "VERIFIED_FINITE_HM40_NOT_GLOBAL"
    ):
        raise ValueError("verified frozen HM40 required")
    previous = verification["combined_analysis_seconds"]
    if previous >= 300:
        raise TimeoutError("original CPU budget exhausted")
    ledger = read_json(PARENT / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(PARENT / name) != sha:
            raise ValueError("changed primary artifact " + name)
    files = {name: digest(ROOT / name) for name in CODE}
    files[str(VERIFY.relative_to(ROOT))] = digest(VERIFY)
    source = {}
    for query in m["spec"]["queries"]:
        file = PARENT / "01/models" / f"{query}.json"
        meta, bank = load(file)
        _, _, rank = prefix_view(meta, bank)
        source[query] = {
            "rank": rank,
            "full_rank": meta["rank"],
            "model_sha256": digest(file),
            "array_sha256": digest(file.with_suffix(".npz")),
        }
    OUT.mkdir(parents=True)
    write_json(
        OUT / "source_barrier.json",
        {
            "frozen_at": now(),
            "source_views": source,
            "application_nodes_loaded": 0,
            "new_model_calls": 0,
            "files": files,
            "primary_manifest_sha256": digest(PARENT / "00/manifest.json"),
            "result_after": True,
            "proxy_is_certified_error_bound": False,
        },
    )
    rows = []
    primary = {
        (int(r["path"]), r["arm"]): r for r in table(PARENT / "02/observations.csv")
    }
    maximum = 0.0
    numeric = 0

    def close(a, b):
        nonlocal maximum, numeric
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError("resolution flags differ")
            return
        x, y = np.asarray(a, float), np.asarray(b, float)
        if x.shape != y.shape or not np.isfinite(x + y).all():
            raise ValueError("finite aligned comparison required")
        err = float(np.max(abs(x - y))) if x.size else 0.0
        maximum = max(maximum, err)
        numeric += x.size
        if err > 1e-8:
            raise ValueError("physical prefix replay failed " + str(err))

    try:
        for query in m["spec"]["queries"]:
            meta, bank = load(PARENT / "01/models" / f"{query}.json")
            _, old = load(
                ROOT / m["spec"]["basis_parent"] / "01/models" / f"{query}_V.json"
            )
            u = old["basis"]
            w, wr, rank = prefix_view(meta, bank)
            for path in read_json(PARENT / "00/design.json"):
                if path["query"] != query:
                    continue
                data = arrays(
                    ROOT / m["spec"]["parent"], path, path["role"], source_barrier=True
                )
                v, l, d, r = (
                    data[k].reshape(2, -1)
                    for k in ("parent_total", "link", "decoder", "roundoff")
                )
                dp = data["delta_p"].ravel()
                base = parts(v, l, d, r, dp, u, bank["basis"], bank["random_basis"])
                calc = parts(v, l, d, r, dp, u, w, wr)
                alt = alternate(v, l, d, r, dp, u, w, wr)
                for key in calc:
                    for value, other in zip(calc[key], alt[key], strict=True):
                        close(value, other)
                for ai, arm in enumerate(m["spec"]["arms"]):
                    original = primary[path["path"], arm]
                    for key in base:
                        close(
                            base[key][ai],
                            None if original[key] == "" else float(original[key]),
                        )
                    head = {
                        k: path[k]
                        for k in ("query", "role", "wording", "path", "H", "B", "Q")
                    }
                    head.update(
                        cell=4 * path["H"] + 2 * path["B"] + path["Q"],
                        arm=arm,
                        prefix_rank=rank,
                    )
                    rows.append(
                        {
                            **head,
                            **{
                                k: float(val[ai]) if val[ai] is not None else None
                                for k, val in calc.items()
                            },
                        }
                    )
                if previous + time.monotonic() - started > 300:
                    raise TimeoutError("original combined300-second budget reached")
        pairs = paired(rows, m["spec"]["wordings"])
        moments, covers = [], []
        for query in m["spec"]["queries"]:
            for role in ("source", "application"):
                for arm in m["spec"]["arms"]:
                    head = {
                        "query": query,
                        "role": role,
                        "arm": arm,
                        "prefix_rank": source[query]["rank"],
                    }
                    selected = [
                        r
                        for r in rows
                        if all(r[k] == head[k] for k in ("query", "role", "arm"))
                    ]
                    covers.append(coverage(selected, head, 1e-8))
                    for measure in ("wording", "pair_mean", "wording_contrast"):
                        picked = (
                            selected
                            if measure == "wording"
                            else [
                                r
                                for r in pairs
                                if r["measure"] == measure
                                and all(
                                    r[k] == head[k] for k in ("query", "role", "arm")
                                )
                            ]
                        )
                        value = signed_summary(picked)
                        checkvalue = scalar_moments(picked)
                        for key in value:
                            close(value[key], checkvalue[key])
                        moments.append({**head, "measure": measure, **value})
        write_csv(OUT / "observations.csv", rows)
        write_csv(OUT / "paired_actions.csv", pairs)
        write_csv(OUT / "action_metrics.csv", moments)
        write_csv(OUT / "coverage.csv", covers)
        elapsed = time.monotonic() - started
        write_json(
            OUT / "summary.json",
            {
                "status": "VERIFIED_RESULT_AFTER_HM40_SOURCE_RESOLUTION_SENSITIVITY",
                "source_views": source,
                "observations": len(rows),
                "paired_actions": len(pairs),
                "metric_groups": len(moments),
                "coverage_groups": len(covers),
                "maximum_alternate_error": maximum,
                "numeric_values_checked": int(numeric),
                "analysis_seconds": elapsed,
                "combined_analysis_seconds": previous + elapsed,
                "new_model_calls": 0,
                "new_fits": 0,
                "result_after": True,
            },
        )
        hashes = {p.name: digest(p) for p in OUT.iterdir() if p.is_file()}
        write_json(OUT / "artifact_hashes.json", hashes)
        print(read_json(OUT / "summary.json"), flush=True)
    except Exception as exc:
        write_json(
            OUT / "failure.json",
            {
                "status": "FAILED_PRESERVED",
                "error": repr(exc),
                "elapsed_seconds": time.monotonic() - started,
                "primary_unchanged": True,
            },
        )
        raise


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
