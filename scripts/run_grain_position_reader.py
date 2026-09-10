#!/usr/bin/env python3
"""NEXT31: cached token-reader compression, transfer and source-only positions."""

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
from grain_research.vector_mechanisms.native_material_reader import direction
from grain_research.vector_mechanisms.position_reader import (
    contractions,
    errors,
    fit_position,
    hidden_basis,
    predict_position,
    project_field,
)
from scripts.run_grain_native_material_reader import QUERIES, cached_unit, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write

SOURCE = BASE / "logic_native_material_reader_v1_r2"
PLAN = ROOT / "docs/GRAIN_POSITION_SPECIFIC_READER_PLAN_V1.md"
RANKS = (1, 2, 4, 8)


def prepare(out):
    """Bind numeric inputs/derivatives and archive the original author roles."""
    verify(SOURCE)
    if read(SOURCE / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Native packets incomplete")
    design = read(SOURCE / "private_design.json")
    if len(design) != 4 or len({r["index"] for r in design}) != 4:
        raise ValueError("Four distinct old authors required")
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/position_reader.py",
        ROOT / "grain_research/vector_mechanisms/native_material_reader.py",
    ]
    hashes = {
        str((SOURCE / n).relative_to(ROOT)): sha(SOURCE / n)
        for n in ("manifest.json", "private_design.json", "private_coordinates.npz")
    }
    for r in design:
        path = cached_unit(r["index"], QUERIES[0]).with_suffix(".npz")
        hashes[str(path.relative_to(ROOT))] = sha(path)
        for q in QUERIES:
            p = SOURCE / "private_packets" / f"a{r['index']:02d}_{q}.json"
            meta = read(p)
            if (
                meta["index"] != r["index"]
                or meta["query"] != q
                or meta["role"] != r["role"]
                or sha(p.with_suffix(".npz")) != meta["arrays_sha256"]
            ):
                raise ValueError("Native gradient identity/hash mismatch")
            hashes[str(p.relative_to(ROOT))] = sha(p)
            hashes[str(p.with_suffix(".npz").relative_to(ROOT))] = meta["arrays_sha256"]
    out.mkdir(exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_POSITION_READER_V1",
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "inputs": hashes,
            "source_authors": 4,
            "queries": list(QUERIES),
            "ranks": list(RANKS),
            "position_degrees": [0, 3],
            "density": "T*J; source token mass1/T; target J=K/T",
            "position": "selected body ordinal midpoint in [-1,1]",
            "new_neural_calls": 0,
            "human_labels_read": False,
            "max_seconds": 240,
            "seed": 2501,
            "n_iter": 5,
            "threads": 4,
        },
    )
    write(
        out / "private_membership.json",
        {
            "authors": design,
            "folds": [
                {
                    "held": r["index"],
                    "source": [s["index"] for s in design if s["index"] != r["index"]],
                }
                for r in design
            ],
        },
    )


def verified(out):
    """Verify old numerical source material and the frozen construction code."""
    m = read(out / "manifest.json")
    for key in ("code_hashes", "inputs"):
        for p, digest in m[key].items():
            if sha(ROOT / p) != digest:
                raise ValueError("Frozen reader dependency changed: " + Path(p).name)
    return m


def construct(name):
    """Execute query-complete numerical units; preserve them at the resource boundary."""
    out = BASE / name
    if not out.exists():
        prepare(out)
    m = verified(out)
    design = read(out / "private_membership.json")["authors"]
    indices = [r["index"] for r in design]
    with np.load(SOURCE / "private_coordinates.npz") as z:
        coordinates = dict(z)
    dirs = {}
    replay = 0.0
    for i in indices:
        with np.load(cached_unit(i, QUERIES[0]).with_suffix(".npz")) as z:
            h = z["full_state"][coordinates[f"a{i}__positions"]].astype(float)
        g = coordinates[f"a{i}__grain"]
        dirs[i] = np.stack([direction(h, g, k)[0] for k in range(4)])
    start = time.monotonic()
    finished = []
    files = {}
    with threadpool_limits(limits=4):
        for qi, q in enumerate(QUERIES):
            dst = out / q
            if (dst / "freeze.json").exists():
                f = read(dst / "freeze.json")
                for p, digest in f["files"].items():
                    if sha(dst / p) != digest:
                        raise ValueError("Completed numerical unit changed")
                    files[q + "/" + p] = digest
                finished.append(q)
                continue
            if dst.exists():
                raise ValueError("Partial unsealed unit exists; inspect before retry")
            if (
                qi
                and (time.monotonic() - start) / len(finished) * len(QUERIES)
                > m["max_seconds"]
            ):
                write(
                    out / "run_status.json",
                    {
                        "status": "PAUSED_RESOURCE_BOUNDARY",
                        "complete_queries": finished,
                        "seconds": time.monotonic() - start,
                        "new_neural_calls": 0,
                    },
                )
                return
            dst.mkdir()
            fields = {}
            metadata = {}
            for i in indices:
                p = SOURCE / "private_packets" / f"a{i:02d}_{q}.json"
                metadata[i] = read(p)
                with np.load(p.with_suffix(".npz")) as z:
                    fields[i] = z["body_gradient"].astype(float)
                actual = contractions(fields[i], dirs[i])
                replay = max(
                    replay,
                    float(
                        np.max(
                            abs(
                                actual
                                - np.array([r["total"] for r in metadata[i]["rows"]])
                            )
                        )
                    ),
                )
            if replay > 1e-10:
                raise ValueError("Old GRAIN tangent contraction changed")
            assets = {}
            geometry = []
            actions = []
            factor_records = []

            def record(
                i,
                method,
                rank,
                access,
                pred,
                *,
                current_fields=fields,
                query=q,
                field_rows=geometry,
                action_rows=actions,
            ):
                """Record the current query immediately with explicitly bound data."""
                result = errors(current_fields[i], pred, dirs[i])
                row = {
                    "query": query,
                    "index": i,
                    "method": method,
                    "rank": rank,
                    "access": access,
                    "tokens": len(current_fields[i]),
                }
                field_rows.append(
                    row
                    | {
                        k: result[k]
                        for k in (
                            "field_error_ratio",
                            "field_actual_energy",
                            "field_error_energy",
                        )
                    }
                )
                action_rows.extend(
                    row
                    | {
                        "factor": f"whole:F{k + 1:02d}",
                        "actual": result["actual"][k],
                        "predicted": result["predicted"][k],
                        "error_bound": result["error_bound"][k],
                    }
                    for k in range(4)
                )

            for slot, i in enumerate(indices):
                j = fields[i]
                record(
                    i,
                    "uniform",
                    1,
                    "current_gradient_diagnostic",
                    np.repeat(j.mean(0)[None], len(j), 0),
                )
                own = hidden_basis([j], seed=2501 + 100 * qi + slot)
                assets[f"a{i}__own_basis"] = own["basis"]
                assets[f"a{i}__own_singular"] = own["singular_values"]
                for k in RANKS:
                    record(
                        i,
                        "local_svd",
                        k,
                        "current_gradient_diagnostic",
                        project_field(j, own["basis"], k),
                    )
                source = [fields[s] for s in indices if s != i]
                shared = hidden_basis(source, seed=3501 + 100 * qi + slot)
                assets[f"a{i}__source_basis"] = shared["basis"]
                assets[f"a{i}__source_singular"] = shared["singular_values"]
                factor_records.append(
                    {
                        "held": i,
                        "source_authors": 3,
                        "own_weighted_energy": float(own["weighted_energy"]),
                        "source_weighted_energy": float(shared["weighted_energy"]),
                    }
                )
                for k in RANKS:
                    record(
                        i,
                        "shared_svd",
                        k,
                        "held_coefficients_oracle",
                        project_field(j, shared["basis"], k),
                    )
                for degree in (0, 3):
                    model = fit_position(source, degree)
                    for key, v in model.items():
                        assets[f"a{i}__degree{degree}__{key}"] = v
                    predicted = predict_position(len(j), model)
                    record(
                        i,
                        "source_position",
                        degree,
                        "source_only_length_input",
                        predicted,
                    )
            np.savez_compressed(dst / "private_factors.npz", **assets)
            pd.DataFrame(geometry).to_csv(dst / "private_fields.csv", index=False)
            pd.DataFrame(actions).to_csv(dst / "private_actions.csv", index=False)
            write(dst / "private_fit_records.json", factor_records)
            names = (
                "private_factors.npz",
                "private_fields.csv",
                "private_actions.csv",
                "private_fit_records.json",
            )
            f = {p: sha(dst / p) for p in names}
            write(
                dst / "freeze.json",
                {
                    "files": f,
                    "new_neural_calls": 0,
                    "source_basis_fits": 4,
                    "position_fits": 8,
                    "within_field_factorizations": 4,
                },
            )
            files.update({q + "/" + p: v for p, v in f.items()})
            finished.append(q)
            write(
                out / "run_status.json",
                {
                    "status": "RUNNING",
                    "complete_queries": finished,
                    "seconds": time.monotonic() - start,
                    "new_neural_calls": 0,
                },
            )
            print(
                {
                    "query": q,
                    "status": "UNIT_COMPLETE",
                    "seconds": round(time.monotonic() - start, 3),
                },
                flush=True,
            )
    write(
        out / "construction_summary.json",
        {
            "status": "COMPLETE",
            "queries": len(finished),
            "authors": 4,
            "gradient_fields": 12,
            "source_basis_fits": 12,
            "position_fits": 24,
            "within_field_factorizations": 12,
            "old_contraction_replay_max": replay,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "position_input": "length and source coefficients only",
            "membership": "LOO exploratory on the old4; original2/2 roles archived, not new held population",
        },
    )
    write(
        out / "prediction_freeze.json",
        {
            "manifest_sha256": sha(out / "manifest.json"),
            "files": files
            | {
                p: sha(out / p)
                for p in ("private_membership.json", "construction_summary.json")
            },
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "complete_queries": finished,
            "seconds": time.monotonic() - start,
            "new_neural_calls": 0,
        },
    )
    print(
        {
            "status": "COMPLETE",
            "queries": 3,
            "source_basis_fits": 12,
            "position_fits": 24,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="logic_position_reader_v1")
    construct(p.parse_args().name)
