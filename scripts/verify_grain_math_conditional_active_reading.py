#!/usr/bin/env python3
"""Independent cell averages, character sums and direct-dot replay for HM37."""

from __future__ import annotations

import argparse
import itertools
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_target_active_support as SUPPORT
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_crossed_field import table

OUT = ROOT / "results/grain_math_response_v1/conditional_active_reading_r1"
DEST = (
    ROOT / "results/grain_math_response_v1/conditional_active_reading_verification_r1"
)
CELLS = tuple(itertools.product(range(2), repeat=3))
LAWS = ("global", "node", "fiber", "full", "mismatched")


def independent_metrics(rows):
    """Compute scalar error moments without importing the production calculator."""
    actual = np.array([float(r["actual"]) for r in rows])
    predicted = np.array([float(r["predicted"]) for r in rows])
    supported = np.array([float(r["supported"]) for r in rows])
    e, c, b = actual - predicted, actual - supported, supported - predicted

    def rms(x):
        return math.sqrt(math.fsum(float(v) ** 2 for v in x) / len(x))

    cross = 2 * math.fsum(float(x * y) for x, y in zip(c, b, strict=True)) / len(c)
    return {
        "count": len(rows),
        "actual_rms": rms(actual),
        "predicted_rms": rms(predicted),
        "rmse": rms(e),
        "relative_rmse": rms(e) / rms(actual) if rms(actual) else None,
        "carrier_error_rms": rms(c),
        "reading_error_rms": rms(b),
        "error_cross_mean": cross,
        "error_identity": abs(rms(e) ** 2 - rms(c) ** 2 - rms(b) ** 2 - cross),
        "sign_agreement": sum(
            x * y > 0 for x, y in zip(actual, predicted, strict=True)
        ),
        "actual_nonzero": sum(x != 0 for x in actual),
    }


def verify(out, destination):
    """Rebuild every source law and scored row within the original combined budget."""
    started = time.monotonic()
    if destination.exists():
        raise FileExistsError("preserve prior verification")
    destination.mkdir(parents=True)
    status = read_json(out / "run_status.json")
    bound = read_json(out / "00/manifest.json")["spec"]["maximum_analysis_seconds"]
    remaining = bound - status["analysis_seconds"]
    if status["status"] != "EVALUATED" or remaining <= 0:
        raise ValueError("evaluated packet and remaining combined budget required")

    def stop(_signal, _frame):
        raise TimeoutError("combined HM37 analysis/verification budget reached")

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(max(1, math.floor(remaining)))
    maximum, comparisons = 0.0, 0

    def close(a, b):
        nonlocal maximum, comparisons
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite or wrong-shaped replay")
        error = float(np.max(abs(a - b))) if a.size else 0.0
        maximum = max(maximum, error)
        comparisons += a.size
        if error > 1e-8:
            raise ValueError(f"original absolute tolerance failed: {error}")

    def row_close(actual, expected):
        for key, value in expected.items():
            if value is None:
                if actual[key] not in (None, ""):
                    raise ValueError("null value not preserved")
            elif isinstance(value, str):
                if actual[key] != value:
                    raise ValueError("wrong row address " + key)
            else:
                close(float(actual[key]), value)

    def half(values, rule):
        weights = (
            (1 / 12, 1 / 3, 1 / 12) if rule == "simpson" else (1 / 8, 1 / 4, 1 / 8)
        )
        return sum(w * x for w, x in zip(weights, values, strict=True))

    try:
        manifest = check(out)
        spec = manifest["spec"]
        if tuple(spec["laws"]) != LAWS or spec["mismatch_xor"] != [0, 1, 1]:
            raise ValueError("fixed law family changed")
        artifact_hashes = read_json(out / "02/artifact_hashes.json")
        for name, sha in artifact_hashes.items():
            if digest(out / name) != sha:
                raise ValueError("artifact changed " + name)
        barrier = read_json(out / "01/source_barrier.json")
        for name, sha in barrier["files"].items():
            if digest(out / name) != sha:
                raise ValueError("source changed " + name)
        if (
            barrier["source_nodes"],
            barrier["source_laws"],
            barrier["application_target_nodes"],
        ) != (144, 30, 0):
            raise ValueError("source/application isolation failed")
        if status["application_started_at"] <= barrier["frozen_at"]:
            raise ValueError("application preceded source barrier")
        design = read_json(out / "00/design.json")
        parent = ROOT / spec["parent"]
        originals = {
            (int(r["path"]), r["arm"]): r
            for r in table(parent / "02/path_comparisons.csv")
        }
        recorded_nodes = {
            (int(r["path"]), int(r["node"]), r["arm"], r["law"]): r
            for r in table(out / "02/nodes.csv")
        }
        recorded_integrals = {
            (int(r["path"]), r["rule"], r["arm"], r["law"]): r
            for r in table(out / "02/integrals.csv")
        }
        provenance = {
            (r["query"], int(r["path"]), int(r["node"])): r
            for r in table(out / "01/source_provenance.csv")
        }
        energies = {
            (r["query"], r["arm"], int(r["character"]), int(r["node"])): r
            for r in table(out / "01/source_character_energy.csv")
        }
        if tuple(
            map(len, (recorded_nodes, recorded_integrals, provenance, energies))
        ) != (2880, 1920, 144, 144):
            raise ValueError("missing or duplicate scored addresses")
        rebuilt_nodes, rebuilt_integrals = [], []
        pred_count, source_count, app_count = 0, 0, 0
        for query in spec["queries"]:
            _, bank = SUPPORT.load(
                ROOT / spec["carrier_parent"] / "01/models" / f"{query}_V.json"
            )
            u = bank["basis"]
            metadata, model = SUPPORT.load(out / "01/models" / f"{query}.json")
            if not metadata["source_frozen_at"] < barrier["frozen_at"]:
                raise ValueError("source model was not frozen before barrier")
            if metadata["query"] != query or metadata["rank"] != u.shape[1]:
                raise ValueError("source carrier address changed")
            raw = np.empty((2, 8, 3, 2, u.shape[1]))
            norms = np.empty((2, 8, 3, 2))
            directions = np.empty((2, 8, 3, 2, 4, 1536))
            for path in design:
                if path["query"] != query or path["role"] != "source":
                    continue
                wi, ci = (
                    spec["source_wordings"].index(path["wording"]),
                    CELLS.index((path["H"], path["B"], path["Q"])),
                )
                for node in range(3):
                    file = (
                        parent
                        / "01/nodes"
                        / f"path_{path['path']:03d}_node_{node}.json"
                    )
                    meta = array_unit(file)
                    if meta["address"] != path or meta["t"] != node / 2:
                        raise ValueError("source node address mismatch")
                    row_close(
                        provenance[query, path["path"], node],
                        {
                            "role": "source",
                            "cell": ci,
                            "wording": path["wording"],
                            "file": str(file.relative_to(ROOT)),
                            "sha256": digest(file),
                        },
                    )
                    with np.load(
                        file.with_suffix(".npz"), allow_pickle=False
                    ) as arrays:
                        dh = arrays["delta_h"].astype(float)
                        v = arrays["p_covectors"].astype(float).reshape(2, -1)
                    norm = np.sqrt(np.sum(dh * dh, axis=(1, 2)))
                    norms[wi, ci, node] = norm
                    directions[wi, ci, node] = dh / norm[:, None, None]
                    raw[wi, ci, node] = (
                        np.einsum("ad,dr->ar", v, u, optimize=True) / norm[:, None]
                    )
                    source_count += 1
            cells = (raw[0] + raw[1]) / 2
            characters = np.zeros_like(cells)
            for si, s in enumerate(CELLS):
                for ci, c in enumerate(CELLS):
                    sign = math.prod((-1) ** (x * y) for x, y in zip(s, c, strict=True))
                    characters[si] += sign * cells[ci] / 8
            full = np.zeros_like(cells)
            for ci, c in enumerate(CELLS):
                for si, s in enumerate(CELLS):
                    full[ci] += (
                        math.prod((-1) ** (x * y) for x, y in zip(s, c, strict=True))
                        * characters[si]
                    )
            laws = np.empty((5, *cells.shape))
            for ci in range(8):
                for node in range(3):
                    laws[0, ci, node] = (
                        sum(cells[c, n] for c in range(8) for n in range(3)) / 24
                    )
                    laws[1, ci, node] = sum(cells[c, node] for c in range(8)) / 8
                    laws[2, ci, node] = sum(cells[ci, n] for n in range(3)) / 3
                    laws[3, ci, node], laws[4, ci, node] = (
                        full[ci, node],
                        full[ci ^ 3, node],
                    )
            for key, value in {
                "source_coefficients": raw,
                "cells": cells,
                "characters": characters,
                "laws": laws,
                "norms": norms,
                "h_directions": directions,
                "reference_h_directions": directions[0, 0, 0],
                "wording_residual": raw - cells,
            }.items():
                close(model[key], value)
            close(np.sum(cells * cells) / 8, np.sum(characters * characters))
            for si in range(8):
                for node in range(3):
                    for ai, arm in enumerate(spec["arms"]):
                        row_close(
                            energies[query, arm, si, node],
                            {
                                "coefficient_squared_norm": float(
                                    np.sum(characters[si, node, ai] ** 2)
                                )
                            },
                        )
            for path in design:
                if path["query"] != query:
                    continue
                ci = CELLS.index((path["H"], path["B"], path["Q"]))
                pred_file = out / "01/predictions" / f"path_{path['path']:03d}.json"
                pm, predicted = SUPPORT.load(pred_file)
                om, observed = SUPPORT.load(out / "02/observations" / pred_file.name)
                if not (barrier["frozen_at"] < pm["predicted_at"] < om["observed_at"]):
                    raise ValueError("prediction/target ordering failed")
                if (
                    pm["target_read_before_prediction"]
                    or pm["source_sha256"] != digest(out / "01/source_barrier.json")
                    or om["prediction_sha256"] != digest(pred_file)
                ):
                    raise ValueError("prediction commitment changed")
                if pm["address"] != path or om["address"] != path or pm["cell"] != ci:
                    raise ValueError("prediction address changed")
                pred_count += 1
                actions, supported_actions, full_actions = [], [], []
                for node in range(3):
                    file = (
                        parent
                        / "01/nodes"
                        / f"path_{path['path']:03d}_node_{node}.json"
                    )
                    with np.load(
                        file.with_suffix(".npz"), allow_pickle=False
                    ) as arrays:
                        delta = arrays["delta_p"].astype(float).ravel()
                        dh = arrays["delta_h"].astype(float)
                        v = arrays["p_covectors"].astype(float).reshape(2, -1)
                    norm = np.sqrt(np.sum(dh * dh, axis=(1, 2)))
                    z = delta @ u
                    theta = laws[:, ci, node] * norm[None, :, None]
                    action = np.array(
                        [
                            [
                                math.fsum(
                                    float(x * y) for x, y in zip(coef, z, strict=True)
                                )
                                for coef in law
                            ]
                            for law in theta
                        ]
                    )
                    actual_coef = v @ u
                    actual, supported = v @ delta, actual_coef @ z
                    direction_error = np.sqrt(
                        np.sum(
                            (dh / norm[:, None, None] - directions[0, 0, 0]) ** 2,
                            axis=(1, 2),
                        )
                    )
                    for key, value in {
                        "coefficients": theta,
                        "predicted": action,
                        "z": z,
                        "h_norms": norm,
                        "h_direction_error": direction_error,
                    }.items():
                        close(predicted[key][node], value)
                    close(observed["coefficients"][node], actual_coef)
                    close(observed["actual"][node], actual)
                    close(observed["supported"][node], supported)
                    if node == 0 and ci == 0:
                        close((u @ theta[3].T).T @ delta, action[3])
                        close((u @ actual_coef.T).T @ delta, supported)
                    actions.append(action)
                    supported_actions.append(supported)
                    full_actions.append(actual)
                    app_count += int(path["role"] == "application")
                    for ai, arm in enumerate(spec["arms"]):
                        for li, law in enumerate(LAWS):
                            error = actual_coef[ai] - theta[li, ai]
                            row = {
                                "path": path["path"],
                                "role": path["role"],
                                "query": query,
                                "arm": arm,
                                "law": law,
                                "node": node,
                                "actual": actual[ai],
                                "supported": supported[ai],
                                "predicted": action[li, ai],
                                "carrier_error": actual[ai] - supported[ai],
                                "reading_error": supported[ai] - action[li, ai],
                                "identity_error": 0,
                                "coefficient_error_squared_norm": np.sum(error * error),
                                "coefficient_actual_squared_norm": np.sum(
                                    actual_coef[ai] ** 2
                                ),
                                "H_direction_unit_difference": direction_error[ai],
                            }
                            row_close(recorded_nodes[path["path"], node, arm, law], row)
                            rebuilt_nodes.append(row)
                for rule in ("simpson", "trapezoid"):
                    prediction = half(actions, rule)
                    actual, supported = (
                        half(full_actions, rule),
                        half(supported_actions, rule),
                    )
                    for ai, arm in enumerate(spec["arms"]):
                        reference = originals[path["path"], arm]
                        for li, law in enumerate(LAWS):
                            row = {
                                "path": path["path"],
                                "role": path["role"],
                                "query": query,
                                "arm": arm,
                                "law": law,
                                "rule": rule,
                                "actual": actual[ai],
                                "supported": supported[ai],
                                "predicted": prediction[li, ai],
                                "carrier_error": actual[ai] - supported[ai],
                                "reading_error": supported[ai] - prediction[li, ai],
                                "endpoint_gradient_halfcontrast": float(
                                    reference["actual_gradient_halfcontrast"]
                                ),
                                "endpoint_finite_halfcontrast": float(
                                    reference["actual_finite_halfcontrast"]
                                ),
                                "hm34_predicted_finite_contrast": float(
                                    reference["hm34_predicted"]
                                ),
                            }
                            row_close(
                                recorded_integrals[path["path"], rule, arm, law], row
                            )
                            rebuilt_integrals.append(row)
        metrics = table(out / "02/metrics.csv")
        coefficient_metrics = table(out / "02/coefficient_metrics.csv")
        if (
            len(metrics),
            len(coefficient_metrics),
            pred_count,
            source_count,
            app_count,
        ) != (180, 60, 96, 144, 144):
            raise ValueError("finite packet not fully replayed")
        for metric in metrics:
            fields = ("role", "query", "arm", "law")
            key = tuple(metric[k] for k in fields)
            rows = (
                rebuilt_nodes
                if metric["scale"] == "nodes"
                else [r for r in rebuilt_integrals if r["rule"] == metric["scale"]]
            )
            selected = [r for r in rows if tuple(r[k] for k in fields) == key]
            row_close(metric, independent_metrics(selected))
        for metric in coefficient_metrics:
            fields = ("role", "query", "arm", "law")
            selected = [
                r for r in rebuilt_nodes if all(r[k] == metric[k] for k in fields)
            ]
            error_ms = math.fsum(
                float(r["coefficient_error_squared_norm"]) for r in selected
            ) / len(selected)
            actual_ms = math.fsum(
                float(r["coefficient_actual_squared_norm"]) for r in selected
            ) / len(selected)
            row_close(
                metric,
                {
                    "nodes": len(selected),
                    "coefficient_error_rms": math.sqrt(error_ms),
                    "coefficient_actual_rms": math.sqrt(actual_ms),
                    "coefficient_relative_error": math.sqrt(error_ms / actual_ms),
                },
            )
        seconds = time.monotonic() - started
        if status["analysis_seconds"] + seconds > bound:
            raise TimeoutError("combined budget exceeded")
        write_json(
            destination / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM37_NOT_GLOBAL",
                "verified_at": now(),
                "packet": str(out.relative_to(ROOT)),
                "manifest_sha256": digest(out / "00/manifest.json"),
                "source_sha256": digest(out / "01/source_barrier.json"),
                "artifact_ledger_sha256": digest(out / "02/artifact_hashes.json"),
                "artifact_hashes_checked": len(artifact_hashes),
                "numerical_values_checked": comparisons,
                "maximum_absolute_error": maximum,
                "prediction_paths_checked": pred_count,
                "source_nodes_checked": source_count,
                "application_nodes_checked": app_count,
                "verification_seconds": seconds,
                "combined_analysis_seconds": status["analysis_seconds"] + seconds,
                "new_model_calls": 0,
                "scope": spec["scope"],
            },
        )
        print(read_json(destination / "summary.json"), flush=True)
    except Exception as error:
        write_json(
            destination / "failure.json",
            {
                "at": now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "verification_seconds": time.monotonic() - started,
                "new_model_calls": 0,
            },
        )
        raise
    finally:
        signal.alarm(0)


def main():
    """Verify one explicit frozen run without overwrite or automatic retry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--destination", type=Path, default=DEST)
    args = parser.parse_args()
    verify(args.out, args.destination)


if __name__ == "__main__":
    main()
