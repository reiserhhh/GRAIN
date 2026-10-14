#!/usr/bin/env python3
"""HM37 source-only finite conditional reading and cold prediction before targets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_conditional_active_reading as C
from grain_research.construction.math_mixed_query_path import quadrature
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_target_active_support as SUPPORT
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard
from scripts.verify_grain_math_crossed_field import table

SPEC = ROOT / "configs/grain_math_conditional_active_reading_v1.json"
PLAN = ROOT / "docs/GRAIN_CONDITIONAL_ACTIVE_READING_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/conditional_active_reading_r1"
CODE = (
    "grain_research/construction/math_conditional_active_reading.py",
    "scripts/run_grain_math_conditional_active_reading.py",
    "scripts/verify_grain_math_conditional_active_reading.py",
    "tests/test_grain_math_conditional_active_reading.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


class Budget:
    """Charge validation, construction, evaluation and checkpoints to one bound."""

    def __init__(self, out, spec, started):
        self.out, self.spec, self.started = out, spec, started
        self.previous = read_json(out / "run_status.json")["analysis_seconds"]

    def checkpoint(self, status):
        """Save elapsed time, then enforce the fixed300-second ceiling."""
        status["analysis_seconds"] = self.previous + time.monotonic() - self.started
        write_json(self.out / "run_status.json", status)
        if status["analysis_seconds"] > self.spec["maximum_analysis_seconds"]:
            raise TimeoutError("fixed HM37 analysis budget reached")


def prepare(out):
    """Freeze all ancestry and design without fitting or reading application arrays."""
    started = time.monotonic()
    if out.exists():
        raise FileExistsError("preserve prior packet; no overwrite or automatic replay")
    spec = read_json(SPEC)
    if spec["laws"] != list(C.LAWS) or spec["nodes"] != list(C.NODES):
        raise ValueError("fixed laws and nodes required")
    files = {}
    for key, verification_key, prefix in (
        ("parent", "parent_verification", "HM35"),
        ("carrier_parent", "carrier_verification", "HM36"),
    ):
        parent = ROOT / spec[key]
        files.update(check(parent)["files"])
        for name, sha in read_json(parent / "02/artifact_hashes.json").items():
            if digest(parent / name) != sha:
                raise ValueError("frozen ancestor changed " + name)
            files[str((parent / name).relative_to(ROOT))] = sha
        verification = ROOT / spec[verification_key] / "summary.json"
        if read_json(verification)["status"] != f"VERIFIED_FINITE_{prefix}_NOT_GLOBAL":
            raise ValueError("verified ancestry required")
        files[str(verification.relative_to(ROOT))] = digest(verification)
    files.update({name: digest(ROOT / name) for name in CODE})
    design = read_json(ROOT / spec["parent"] / "00/design.json")
    if len(design) != 96 or sum(r["role"] == "source" for r in design) != 48:
        raise ValueError("unchanged96path design required")
    for sub in ("00/code", "01/models", "01/predictions", "02/observations"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_json(out / "00/design.json", design)
    for name in CODE:
        destination = out / "00/code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "spec": spec,
            "files": files,
            "inputs": {"00/design.json": digest(out / "00/design.json")},
            "frozen_at": now(),
            "new_model_calls": 0,
        },
    )
    elapsed = time.monotonic() - started + spec["reserved_preparation_failure_seconds"]
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "analysis_seconds": elapsed,
            "source_laws": 0,
            "source_nodes": 0,
            "application_target_nodes": 0,
            "prediction_paths": 0,
            "new_model_calls": 0,
        },
    )
    if elapsed > spec["maximum_analysis_seconds"]:
        raise TimeoutError("preparation exhausted fixed budget")
    print({"status": "PREPARED", "seconds": elapsed}, flush=True)


def path_node(parent, path, node):
    """Bind role, full address and original node before any numerical input access."""
    file = parent / "01/nodes" / f"path_{path['path']:03d}_node_{node}.json"
    metadata = array_unit(file)
    if (
        metadata["address"] != path
        or metadata["node"] != node
        or metadata["t"] != node / 2
    ):
        raise ValueError("node address changed")
    return file


def carrier(out, query, spec):
    """Read only the frozen source coframe, preserving its actual token coordinates."""
    file = ROOT / spec["carrier_parent"] / "01/models" / f"{query}_V.json"
    meta, arrays = SUPPORT.load(file)
    raw = read_json(file.with_name(f"{query}_raw.json"))
    return meta, arrays["basis"], raw


def source(out):
    """Build thirty declared source arithmetic laws with complete character spectra."""
    started = time.monotonic()
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    budget = Budget(out, spec, started)
    if status["status"] != "PREPARED":
        raise ValueError("untouched source stage required")
    status["status"] = "SOURCE_RUNNING"
    budget.checkpoint(status)
    parent = ROOT / spec["parent"]
    design = read_json(out / "00/design.json")
    provenance, effects = [], []
    for query in spec["queries"]:
        _, u, raw = carrier(out, query, spec)
        rank = u.shape[1]
        coefficients = np.empty((2, 8, 3, 2, rank))
        norms = np.empty((2, 8, 3, 2))
        directions = np.empty((2, 8, 3, 2, 4, 1536))
        visited = set()
        for path in design:
            if path["query"] != query or path["role"] != "source":
                continue
            if path["left"]["early_query_token_ids"] != raw["token_ids"]:
                raise ValueError("source query coordinates differ from frozen carrier")
            wi = spec["source_wordings"].index(path["wording"])
            cell = C.CELLS.index((path["H"], path["B"], path["Q"]))
            for node in range(3):
                address = wi, cell, node
                if address in visited:
                    raise ValueError("duplicate source cell")
                visited.add(address)
                file = path_node(parent, path, node)
                with np.load(file.with_suffix(".npz"), allow_pickle=False) as bank:
                    v = bank["p_covectors"].astype(float).reshape(2, -1)
                    h = bank["delta_h"].astype(float)
                physical = np.linalg.norm(h.reshape(2, -1), axis=1)
                if not (physical > 0).all():
                    raise ValueError("zero physical H action")
                coefficients[address] = (v @ u) / physical[:, None]
                norms[address] = physical
                directions[address] = h / physical[:, None, None]
                status["source_nodes"] += 1
                provenance.append(
                    {
                        "query": query,
                        "path": path["path"],
                        "node": node,
                        "role": "source",
                        "wording": path["wording"],
                        "cell": cell,
                        "file": str(file.relative_to(ROOT)),
                        "sha256": digest(file),
                    }
                )
        if len(visited) != 48:
            raise ValueError("all48 source cells/nodes required per target")
        fitted = C.source_laws(coefficients, ["source"] * 2)
        error = float(
            np.max(abs(C.decode_characters(fitted["characters"]) - fitted["cells"]))
        )
        if error > spec["absolute_tolerance"]:
            raise ValueError("source character inversion failed")
        reference = directions[0, 0, 0]
        SUPPORT.save(
            out / "01/models" / f"{query}.json",
            {
                "query": query,
                "rank": rank,
                "source_wordings": spec["source_wordings"],
                "cells": list(C.CELLS),
                "laws": list(C.LAWS),
                "norm_unit": "per actual physical deltaH norm",
                "source_frozen_at": now(),
                "inversion_error": error,
                "carrier_sha256": digest(
                    ROOT / spec["carrier_parent"] / "01/models" / f"{query}_V.json"
                ),
                "token_ids": raw["token_ids"],
                "p_shape": raw["p_shape"],
            },
            {
                **fitted,
                "source_coefficients": coefficients,
                "norms": norms,
                "h_directions": directions,
                "reference_h_directions": reference,
            },
        )
        status["source_laws"] += len(C.LAWS) * len(spec["arms"])
        for character, bits in enumerate(C.CELLS):
            for node in range(3):
                for ai, arm in enumerate(spec["arms"]):
                    value = fitted["characters"][character, node, ai]
                    effects.append(
                        {
                            "query": query,
                            "arm": arm,
                            "character": character,
                            "H": bits[0],
                            "B": bits[1],
                            "Q": bits[2],
                            "node": node,
                            "coefficient_squared_norm": float(value @ value),
                            "inversion_error": error,
                        }
                    )
        budget.checkpoint(status)
        print(
            {
                "source_query": query,
                "source_laws": status["source_laws"],
                "seconds": status["analysis_seconds"],
            },
            flush=True,
        )
    if (status["source_laws"], status["source_nodes"]) != (30, 144):
        raise ValueError("complete source construction required")
    write_csv(out / "01/source_provenance.csv", provenance)
    write_csv(out / "01/source_character_energy.csv", effects)
    files = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "01").rglob("*")
        if p.is_file() and not p.name.startswith("._")
    }
    write_json(
        out / "01/source_barrier.json",
        {
            "files": files,
            "frozen_at": now(),
            "source_nodes": 144,
            "source_laws": 30,
            "application_target_nodes": status["application_target_nodes"],
        },
    )
    status["status"] = "SOURCE_CHECKPOINTED"
    budget.checkpoint(status)


def input_arrays(parent, path, node):
    """Load cold inputs only; no derivative target or gradient is returned/read."""
    file = path_node(parent, path, node)
    with np.load(file.with_suffix(".npz"), allow_pickle=False) as bank:
        return bank["delta_p"].astype(float), bank["delta_h"].astype(float)


def evaluate(out):
    """Commit each path prediction before loading its numerical derivative targets."""
    started = time.monotonic()
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    budget = Budget(out, spec, started)
    if status["status"] != "SOURCE_CHECKPOINTED":
        raise ValueError("unique complete source checkpoint required")
    barrier_file = out / "01/source_barrier.json"
    for name, sha in read_json(barrier_file)["files"].items():
        if digest(out / name) != sha:
            raise ValueError("source law changed")
    status.update(
        status="APPLICATION_RUNNING",
        application_started_at=now(),
        source_sha256=digest(barrier_file),
    )
    budget.checkpoint(status)
    parent = ROOT / spec["parent"]
    design = read_json(out / "00/design.json")
    original = {
        (int(r["path"]), r["arm"]): r for r in table(parent / "02/path_comparisons.csv")
    }
    node_rows, integral_rows, coefficient_rows = [], [], []
    for query in spec["queries"]:
        _, u, _ = carrier(out, query, spec)
        model_meta, model = SUPPORT.load(out / "01/models" / f"{query}.json")
        for path in design:
            if path["query"] != query:
                continue
            if path["left"]["early_query_token_ids"] != model_meta["token_ids"]:
                raise ValueError("cold query coordinates differ from source law")
            cell = C.CELLS.index((path["H"], path["B"], path["Q"]))
            predictions, coefficients, input_z, norms, directional_error = (
                [],
                [],
                [],
                [],
                [],
            )
            for node, t in enumerate(C.NODES):
                delta, dh = input_arrays(parent, path, node)
                physical = np.linalg.norm(dh.reshape(2, -1), axis=1)
                z = u.T @ delta.ravel()
                coef, pred = C.predict(model["laws"], cell, t, physical, z)
                predictions.append(pred)
                coefficients.append(coef)
                input_z.append(z)
                norms.append(physical)
                directional_error.append(
                    np.linalg.norm(
                        (
                            dh / physical[:, None, None]
                            - model["reference_h_directions"]
                        ).reshape(2, -1),
                        axis=1,
                    )
                )
            arrays = {
                "predicted": np.stack(predictions),
                "coefficients": np.stack(coefficients),
                "z": np.stack(input_z),
                "h_norms": np.stack(norms),
                "h_direction_error": np.stack(directional_error),
            }
            file = out / "01/predictions" / f"path_{path['path']:03d}.json"
            SUPPORT.save(
                file,
                {
                    "address": path,
                    "cell": cell,
                    "predicted_at": now(),
                    "source_sha256": digest(barrier_file),
                    "target_read_before_prediction": False,
                },
                arrays,
            )
            status["prediction_paths"] += 1
            actual, supported, actual_coefficients = [], [], []
            for node in range(3):
                target_file = path_node(parent, path, node)
                with np.load(
                    target_file.with_suffix(".npz"), allow_pickle=False
                ) as bank:
                    v = bank["p_covectors"].astype(float).reshape(2, -1)
                    delta = bank["delta_p"].astype(float).ravel()
                a = v @ delta
                coef = v @ u
                s = coef @ arrays["z"][node]
                actual.append(a)
                supported.append(s)
                actual_coefficients.append(coef)
                if path["role"] == "application":
                    status["application_target_nodes"] += 1
                for ai, arm in enumerate(spec["arms"]):
                    for li, law in enumerate(C.LAWS):
                        p = arrays["predicted"][node, li, ai]
                        err = coef[ai] - arrays["coefficients"][node, li, ai]
                        row = {
                            "path": path["path"],
                            "role": path["role"],
                            "query": query,
                            "arm": arm,
                            "law": law,
                            "node": node,
                            "actual": float(a[ai]),
                            "supported": float(s[ai]),
                            "predicted": float(p),
                            "carrier_error": float(a[ai] - s[ai]),
                            "reading_error": float(s[ai] - p),
                            "identity_error": float(
                                abs((a[ai] - p) - (a[ai] - s[ai]) - (s[ai] - p))
                            ),
                            "coefficient_error_squared_norm": float(err @ err),
                            "coefficient_actual_squared_norm": float(
                                coef[ai] @ coef[ai]
                            ),
                            "H_direction_unit_difference": float(
                                arrays["h_direction_error"][node, ai]
                            ),
                        }
                        node_rows.append(row)
            actual, supported = np.stack(actual), np.stack(supported)
            SUPPORT.save(
                out / "02/observations" / file.name,
                {
                    "address": path,
                    "observed_at": now(),
                    "prediction_sha256": digest(file),
                },
                {
                    "actual": actual,
                    "supported": supported,
                    "coefficients": np.stack(actual_coefficients),
                },
            )
            for rule, prediction in quadrature(arrays["predicted"]).items():
                a, s = quadrature(actual)[rule], quadrature(supported)[rule]
                for ai, arm in enumerate(spec["arms"]):
                    reference = original[path["path"], arm]
                    for li, law in enumerate(C.LAWS):
                        integral_rows.append(
                            {
                                "path": path["path"],
                                "role": path["role"],
                                "query": query,
                                "arm": arm,
                                "law": law,
                                "rule": rule,
                                "actual": float(a[ai]),
                                "supported": float(s[ai]),
                                "predicted": float(prediction[li, ai]),
                                "endpoint_gradient_halfcontrast": float(
                                    reference["actual_gradient_halfcontrast"]
                                ),
                                "endpoint_finite_halfcontrast": float(
                                    reference["actual_finite_halfcontrast"]
                                ),
                                "hm34_predicted_finite_contrast": float(
                                    reference["hm34_predicted"]
                                ),
                                "carrier_error": float(a[ai] - s[ai]),
                                "reading_error": float(s[ai] - prediction[li, ai]),
                            }
                        )
            budget.checkpoint(status)
        print(
            {
                "evaluated_query": query,
                "node_rows": len(node_rows),
                "seconds": status["analysis_seconds"],
            },
            flush=True,
        )
    if (status["application_target_nodes"], status["prediction_paths"]) != (144, 96):
        raise ValueError("full cold evaluation required")
    metrics = []
    for role in ("source", "application"):
        for query in spec["queries"]:
            for arm in spec["arms"]:
                for law in C.LAWS:
                    selected = [
                        r
                        for r in node_rows
                        if (r["role"], r["query"], r["arm"], r["law"])
                        == (role, query, arm, law)
                    ]
                    coefficient_rows.append(
                        {
                            "role": role,
                            "query": query,
                            "arm": arm,
                            "law": law,
                            "nodes": len(selected),
                            "coefficient_error_rms": float(
                                np.sqrt(
                                    np.mean(
                                        [
                                            r["coefficient_error_squared_norm"]
                                            for r in selected
                                        ]
                                    )
                                )
                            ),
                            "coefficient_actual_rms": float(
                                np.sqrt(
                                    np.mean(
                                        [
                                            r["coefficient_actual_squared_norm"]
                                            for r in selected
                                        ]
                                    )
                                )
                            ),
                            "coefficient_relative_error": float(
                                np.sqrt(
                                    sum(
                                        r["coefficient_error_squared_norm"]
                                        for r in selected
                                    )
                                    / sum(
                                        r["coefficient_actual_squared_norm"]
                                        for r in selected
                                    )
                                )
                            ),
                        }
                    )
                    for scale, selected_rows in (
                        ("nodes", selected),
                        *(
                            (
                                rule,
                                [
                                    r
                                    for r in integral_rows
                                    if (
                                        r["role"],
                                        r["query"],
                                        r["arm"],
                                        r["law"],
                                        r["rule"],
                                    )
                                    == (role, query, arm, law, rule)
                                ],
                            )
                            for rule in ("simpson", "trapezoid")
                        ),
                    ):
                        metrics.append(
                            {
                                "role": role,
                                "query": query,
                                "arm": arm,
                                "law": law,
                                "scale": scale,
                                **C.error_metrics(
                                    [
                                        [
                                            r[k]
                                            for k in (
                                                "actual",
                                                "supported",
                                                "predicted",
                                            )
                                        ]
                                        for r in selected_rows
                                    ]
                                ),
                            }
                        )
    write_csv(out / "02/nodes.csv", node_rows)
    write_csv(out / "02/integrals.csv", integral_rows)
    write_csv(out / "02/metrics.csv", metrics)
    write_csv(out / "02/coefficient_metrics.csv", coefficient_rows)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM37_EVALUATED_NOT_GLOBAL",
            "node_rows": len(node_rows),
            "integral_rows": len(integral_rows),
            "metrics": len(metrics),
            "coefficient_groups": len(coefficient_rows),
            "new_model_calls": 0,
            "prediction_before_target": True,
            "source_sha256": digest(barrier_file),
            "scope": spec["scope"],
        },
    )
    status["status"] = "EVALUATED"
    budget.checkpoint(status)
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for sub in ("01", "02")
            for p in (out / sub).rglob("*")
            if p.is_file() and not p.name.startswith("._")
        },
    )
    budget.checkpoint(status)
    print({"status": "EVALUATED", "seconds": status["analysis_seconds"]}, flush=True)


def main():
    """Execute an explicit bounded stage with failures retained and no automatic retry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "source", "evaluate", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        try:
            if args.stage in ("prepare", "all"):
                prepare(args.out)
            if args.stage in ("source", "all"):
                source(args.out)
            if args.stage in ("evaluate", "all"):
                evaluate(args.out)
        except Exception as error:
            if args.out.exists():
                write_json(
                    args.out / "failure.json",
                    {
                        "at": now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "new_model_calls": 0,
                    },
                )
            raise


if __name__ == "__main__":
    main()
