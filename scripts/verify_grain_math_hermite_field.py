#!/usr/bin/env python3
"""Alternate HM26 Gaussian-operator arithmetic without the production kernel."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_paths import node_path
from scripts.run_grain_math_hermite_field import OUT, check
from scripts.run_grain_math_response import write_json

DEFAULT = ROOT / "results/grain_math_response_v1/hermite_field_verification_r1"


def direct_input(parent, address, node, representation):
    """Rebuild actual moments by explicit token-weight sums, not runner helpers."""
    names = tuple(address["capture"]["spans"])
    with np.load(ROOT / address["reference_body_npz"]) as p:
        states = {name: p["L16_" + name].astype(float) for name in names}
    with np.load(ROOT / address["reference_npz"]) as p:
        q = p["L16_Q"].astype(float).ravel()
    with np.load(
        node_path(parent, address["hm24_unit"], node).with_suffix(".npz")
    ) as p:
        states[address["span"]], delta = (
            p["actual_property_state"].copy(),
            p["path_delta"].copy(),
        )
    slots = [(n, 0) for n in names] + [
        (n, k) for n in names if n.endswith("property") for k in (1, 2, 3)
    ]
    if representation == "full24_Q_layout":
        slots = [(n, k) for n in names for k in range(4)]
    features, directions = [], []
    for name, degree in slots:
        count = len(states[name])
        position = np.zeros(1) if count == 1 else np.arange(count) * 2 / (count - 1) - 1
        weight = position**degree / count
        features.append(np.sum(states[name] * weight[:, None], axis=0))
        directions.append(
            np.sum(delta * weight[:, None], axis=0)
            if name == address["span"]
            else np.zeros_like(features[-1])
        )
    layout = (
        [len(states[n]) for n in names]
        + [address["capture"]["spans"][n]["positions"][0] for n in names]
        + [len(address["capture"]["input_ids"])]
    )
    return np.r_[np.concatenate(features), q, layout], np.r_[
        np.concatenate(directions), np.zeros(len(q) + len(layout))
    ]


def operator_gram(x, v, kinds, y, w, other_kinds, length2):
    """Evaluate kernels with explicit coordinate differences, one row at a time."""
    gram = np.empty((len(x), len(y)))
    for i, (position, direction, kind) in enumerate(zip(x, v, kinds, strict=True)):
        displacement = position[None, :] - y
        base = np.exp(-np.sum(displacement**2, axis=1) / (2 * length2))
        a = np.sum(displacement * direction, axis=1) / length2
        b = np.sum(displacement * w, axis=1) / length2
        if kind:
            factor = np.where(
                other_kinds == 1, np.sum(direction * w, axis=1) / length2 - a * b, -a
            )
        else:
            factor = np.where(other_kinds == 1, b, 1)
        gram[i] = base * factor
    return gram


def verify(packet, out):
    """Check every source fit/cold value/derivative/action and scalar summary."""
    m = check(packet)
    if out.exists():
        raise FileExistsError("preserve prior verification")
    out.mkdir(parents=True)
    (out / "verify.py").write_bytes(Path(__file__).read_bytes())
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(packet / name) != sha:
            raise ValueError("HM26 output changed " + name)
    spec, design = m["spec"], read_json(packet / "00/design.json")
    parent = ROOT / spec["parent"]
    read_table = lambda filename: list(
        csv.DictReader((packet / "02" / filename).open())
    )
    node_table = read_table("node_predictions.csv")
    action_table = read_table("actions.csv")
    maxima = {
        k: 0.0
        for k in (
            "input",
            "gram",
            "coefficient",
            "value",
            "derivative",
            "action",
            "summary",
        )
    }

    def compare(name, actual, expected, tolerance=2e-9):
        error = float(np.max(abs(np.asarray(actual) - np.asarray(expected)), initial=0))
        maxima[name] = max(maxima[name], error)
        if error > tolerance:
            raise ValueError(f"{name} error {error} > {tolerance}")

    predicted_nodes, predicted_actions, models = {}, {}, 0
    for filename, metadata in read_json(packet / "01/model_manifest.json").items():
        with np.load(packet / "01/models" / filename) as p:
            model = {k: p[k].copy() for k in p.files}
        query, rep, method = (
            metadata["query"],
            metadata["representation"],
            metadata["method"],
        )
        lookup = {r["hm24_unit"]: r for r in design if r["query"] == query}
        source_x, source_v, values, slopes = [], [], [], []
        for address in metadata["source_addresses"]:
            row = lookup[address["unit"]]
            if row["role"] != "source" or row["family"] != address["family"]:
                raise ValueError("non-source model data")
            x, v = direct_input(parent, row, address["node"], rep)
            source_x.append(x)
            source_v.append(v)
            record = read_json(node_path(parent, address["unit"], address["node"]))
            values.append(record["native"]["score"])
            slopes.append(record["native_derivative"])
        source_x, source_v = np.array(source_x), np.array(source_v)
        origin = source_x.mean(axis=0)
        unit = np.sqrt(np.mean(np.sum((source_x - origin) ** 2, axis=1)))
        z, v = (source_x - origin) / unit, source_v / unit
        distances = [
            float(np.sum((z[i] - z[j]) ** 2))
            for i in range(len(z))
            for j in range(i + 1, len(z))
        ]
        length2 = np.median([d for d in distances if d > 100 * np.finfo(float).eps])
        compare("input", origin, model["origin"])
        compare("input", unit, model["input_unit"])
        compare("input", length2, model["length2"])
        prior = np.mean(values)
        if method == "hermite":
            z, v = np.vstack((z, z)), np.vstack((v, v))
            kinds = np.r_[np.zeros(len(values), int), np.ones(len(values), int)]
            target = np.r_[values - prior, slopes]
        else:
            kinds, target = np.zeros(len(values), int), values - prior
        compare("input", z, model["positions"])
        compare("input", v, model["directions"])
        gram = operator_gram(z, v, kinds, z, v, kinds, length2)
        compare("gram", gram, model["gram"])
        coefficients = np.linalg.solve(
            gram + spec["alpha"] * np.diag(np.diag(gram)), target
        )
        compare("coefficient", coefficients, model["coefficients"], 2e-8)
        for row in lookup.values():
            xs, vs = zip(
                *(direct_input(parent, row, k, rep) for k in range(9)), strict=True
            )
            x, direction = (np.array(xs) - origin) / unit, np.array(vs) / unit
            zeros = np.zeros(len(x), int)
            prediction = (
                operator_gram(x, direction, zeros, z, v, kinds, length2) @ coefficients
                + prior
            )
            derivative = (
                operator_gram(x, direction, zeros + 1, z, v, kinds, length2)
                @ coefficients
            )
            key = (query, rep, method, row["hm24_unit"])
            predicted_nodes[key] = (prediction, derivative)
            predicted_actions[key] = float(prediction[-1] - prediction[0])
        models += 1
    for r in node_table:
        key = (r["query"], r["representation"], r["method"], int(r["unit"]))
        prediction, derivative = predicted_nodes[key]
        compare("value", prediction[int(r["node"])], float(r["prediction"]))
        compare(
            "derivative", derivative[int(r["node"])], float(r["predicted_derivative"])
        )
        row = next(a for a in design if a["hm24_unit"] == int(r["unit"]))
        actual = read_json(node_path(parent, row["hm24_unit"], int(r["node"])))
        if actual["address"] != row or r["role"] != row["role"]:
            raise ValueError("node address mismatch")
        compare("value", float(r["native"]), actual["native"]["score"], 0)
        compare(
            "derivative", float(r["native_derivative"]), actual["native_derivative"], 0
        )
    for r in action_table:
        key = (r["query"], r["representation"], r["method"], int(r["unit"]))
        compare("action", predicted_actions[key], float(r["predicted_action"]))
    for row in read_table("summary.csv"):
        selected = lambda r, row=row: all(
            r[k] == row[k] for k in ("query", "representation", "method", "role")
        )
        ns, acts = (
            list(filter(selected, node_table)),
            list(filter(selected, action_table)),
        )
        native = np.array([float(r["native_action"]) for r in acts])
        predicted = np.array(
            [
                predicted_actions[
                    (r["query"], r["representation"], r["method"], int(r["unit"]))
                ]
                for r in acts
            ]
        )
        rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
        values = {
            "families": len(acts),
            "nodes": len(ns),
            "native_action_rms": rms(native),
            "predicted_action_rms": rms(predicted),
            "action_rmse": rms(native - predicted),
            "action_r": np.corrcoef(native, predicted)[0, 1],
            "sign_agreements": sum(native * predicted > 0),
            "value_rmse": rms(
                [
                    predicted_nodes[
                        (r["query"], r["representation"], r["method"], int(r["unit"]))
                    ][0][int(r["node"])]
                    - float(r["native"])
                    for r in ns
                ]
            ),
            "derivative_rmse": rms(
                [
                    predicted_nodes[
                        (r["query"], r["representation"], r["method"], int(r["unit"]))
                    ][1][int(r["node"])]
                    - float(r["native_derivative"])
                    for r in ns
                ]
            ),
        }
        for name, value in values.items():
            compare("summary", value, float(row[name]))
    summary = {
        "status": "VERIFIED_FINITE_HM26_NOT_GLOBAL",
        "counts": {
            "models": models,
            "nodes": len(node_table),
            "actions": len(action_table),
            "summaries": 36,
        },
        "max_errors": maxima,
        "new_model_calls": 0,
        "alternate_same_agent_not_human_review": True,
        "results_or_tolerances_changed": False,
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "artifact_hashes.json",
        {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    )
    print(summary, flush=True)


def main():
    """Run one fresh finite verifier without touching primary artifacts."""
    p = argparse.ArgumentParser()
    p.add_argument("--packet", type=Path, default=OUT)
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    try:
        verify(a.packet, a.out)
    except BaseException as exc:
        if a.out.exists():
            write_json(a.out / "failure.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
