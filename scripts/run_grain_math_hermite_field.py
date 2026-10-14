#!/usr/bin/env python3
"""HM26 bounded source Hermite compilation, then cold family evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_hermite_field as H
from grain_research.construction.math_decoder_fibers import observation_matrix
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_paths import check as parent_check
from scripts.run_grain_math_decoder_paths import node_path
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_hermite_field_v1.json"
PLAN = ROOT / "docs/GRAIN_HERMITE_ACTION_FIELD_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/hermite_field_r1"
CODE = (
    "grain_research/construction/math_hermite_field.py",
    "scripts/run_grain_math_hermite_field.py",
    "scripts/verify_grain_math_hermite_field.py",
    "tests/test_grain_math_hermite_field.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out):
    """Validate the frozen design, code and inherited native observations."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("frozen input changed " + name)
    if digest(out / "00/design.json") != m["design_sha256"]:
        raise ValueError("design changed")
    return m


def prepare(out):
    """Freeze all methods/representations before source fit or held diagnostics."""
    if out.exists():
        raise FileExistsError("preserve this packet; use a fresh output")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    m = parent_check(parent)
    v = read_json(ROOT / spec["parent_verification"] / "summary.json")
    if v["status"] != "VERIFIED_FINITE_HM25_NOT_GLOBAL":
        raise ValueError("completed HM25 verification required")
    design = read_json(parent / "00/design.json")
    if len(design) != 48 or {r["query"] for r in design} != set(spec["queries"]):
        raise ValueError("original48 units/queries required")
    for query in spec["queries"]:
        roles = {
            role: {
                r["family"] for r in design if r["query"] == query and r["role"] == role
            }
            for role in ("source", "validation", "application")
        }
        if tuple(map(len, roles.values())) != (8, 4, 4) or any(
            roles[a] & roles[b]
            for a, b in (
                ("source", "validation"),
                ("source", "application"),
                ("validation", "application"),
            )
        ):
            raise ValueError("family split changed")
    files = dict(m["files"])
    for name, sha in read_json(parent / "02/artifact_hashes.json").items():
        if digest(parent / name) != sha:
            raise ValueError("HM25 artifact changed " + name)
        files[str((parent / name).relative_to(ROOT))] = sha
    files.update({name: digest(ROOT / name) for name in CODE})
    (out / "00/code").mkdir(parents=True)
    (out / "01/models").mkdir(parents=True)
    write_json(out / "00/design.json", design)
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {"files": files, "spec": spec, "design_sha256": digest(out / "00/design.json")},
    )
    write_json(
        out / "run_status.json", {"status": "PREPARED", "fits": 0, "model_calls": 0}
    )
    print({"status": "PREPARED", "maximum_fits": 12}, flush=True)


def input_path(parent, address, node, representation):
    """Construct numerical input/direction without consuming its native score."""
    names = tuple(address["capture"]["spans"])
    with np.load(ROOT / address["reference_body_npz"]) as p:
        states = {name: p["L16_" + name].astype(float) for name in names}
    with np.load(ROOT / address["reference_npz"]) as p:
        q = p["L16_Q"].astype(float)
    with np.load(
        node_path(parent, address["hm24_unit"], node).with_suffix(".npz")
    ) as p:
        states[address["span"]] = p["actual_property_state"].copy()
        delta = p["path_delta"].copy()
    chunks, directions = [], []
    # Preserve the actual old12 observation layout rather than a renamed full map.
    slots = [(name, 0) for name in names]
    slots += [
        (name, k) for name in names if name.endswith("property") for k in (1, 2, 3)
    ]
    if representation == "full24_Q_layout":
        slots = [(name, k) for name in names for k in range(4)]
    elif representation != "old12_Q_layout":
        raise ValueError("undeclared representation")
    for name, k in slots:
        if len(states[name]) > 4:
            raise ValueError("full24 finite injectivity scope exceeded")
        o = observation_matrix(len(states[name]), 3)
        chunks.append(o[k] @ states[name])
        directions.append(
            o[k] @ delta if name == address["span"] else np.zeros(states[name].shape[1])
        )
    layout = np.array(
        [len(states[n]) for n in names]
        + [address["capture"]["spans"][n]["positions"][0] for n in names]
        + [len(address["capture"]["input_ids"])],
        float,
    )
    return np.r_[np.concatenate(chunks), q.ravel(), layout], np.r_[
        np.concatenate(directions), np.zeros(q.size + len(layout))
    ]


def fit(out):
    """Fit all12 source models and freeze them before reading any held outputs."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "PREPARED":
        raise ValueError("fit is one bounded stage; no blind replay")
    spec, design, start = m["spec"], read_json(out / "00/design.json"), time.monotonic()
    parent, models, fits = ROOT / spec["parent"], {}, 0
    for query in spec["queries"]:
        source = [r for r in design if r["query"] == query and r["role"] == "source"]
        for representation in spec["representations"]:
            inputs, directions, values, derivatives, addresses = [], [], [], [], []
            for r in source:
                for node in range(spec["nodes"]):
                    x, v = input_path(parent, r, node, representation)
                    record = read_json(node_path(parent, r["hm24_unit"], node))
                    if record["address"] != r:
                        raise ValueError("source node address mismatch")
                    inputs.append(x)
                    directions.append(v)
                    values.append(record["native"]["score"])
                    derivatives.append(record["native_derivative"])
                    addresses.append(
                        {"unit": r["hm24_unit"], "family": r["family"], "node": node}
                    )
            z, v, origin, unit, length2 = H.source_geometry(inputs, directions)
            for method in spec["methods"]:
                model = H.fit_potential(
                    z,
                    v,
                    values,
                    derivatives,
                    length2,
                    spec["alpha"],
                    method == "hermite",
                )
                key = query + "__" + representation + "__" + method
                path = out / "01/models" / (key + ".npz")
                np.savez_compressed(path, **model, origin=origin, input_unit=unit)
                metadata = {
                    "query": query,
                    "representation": representation,
                    "method": method,
                    "source_addresses": addresses,
                    "length2": length2,
                    "input_unit": unit,
                    "system_residual": model["system_residual"],
                    "arrays_sha256": digest(path),
                }
                write_json(path.with_suffix(".json"), metadata)
                models[path.name] = metadata
                fits += 1
                write_json(
                    out / "run_status.json",
                    {"status": "FITTING_SOURCE", "fits": fits, "model_calls": 0},
                )
    write_json(out / "01/model_manifest.json", models)
    write_json(
        out / "run_status.json",
        {
            "status": "SOURCE_MODELS_FROZEN",
            "fits": fits,
            "model_calls": 0,
            "stage_seconds": time.monotonic() - start,
        },
    )
    print(read_json(out / "run_status.json"), flush=True)


def rms(v):
    """Return physical-unit RMS without small denominator normalization."""
    return float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))


def summaries(nodes, actions):
    """Report every declared condition/role; no held winner selection."""
    result = []
    keys = sorted(
        {(r["query"], r["representation"], r["method"], r["role"]) for r in nodes}
    )
    for query, rep, method, role in keys:
        n = [
            r
            for r in nodes
            if (r["query"], r["representation"], r["method"], r["role"])
            == (query, rep, method, role)
        ]
        a = [
            r
            for r in actions
            if (r["query"], r["representation"], r["method"], r["role"])
            == (query, rep, method, role)
        ]
        native = np.array([r["native_action"] for r in a])
        prediction = np.array([r["predicted_action"] for r in a])
        result.append(
            {
                "query": query,
                "representation": rep,
                "method": method,
                "role": role,
                "families": len(a),
                "nodes": len(n),
                "native_action_rms": rms(native),
                "predicted_action_rms": rms(prediction),
                "action_rmse": rms(native - prediction),
                "action_r": float(np.corrcoef(native, prediction)[0, 1])
                if native.std() and prediction.std()
                else None,
                "sign_agreements": int(sum(native * prediction > 0)),
                "value_rmse": rms([r["prediction"] - r["native"] for r in n]),
                "derivative_rmse": rms(
                    [r["predicted_derivative"] - r["native_derivative"] for r in n]
                ),
            }
        )
    return result


def evaluate(out):
    """Load held reference paths only after all source-model snapshots are frozen."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "SOURCE_MODELS_FROZEN":
        raise ValueError("source models must be frozen before cold evaluation")
    spec, design = m["spec"], read_json(out / "00/design.json")
    parent, nodes, actions = ROOT / spec["parent"], [], []
    for filename, metadata in read_json(out / "01/model_manifest.json").items():
        model_path = out / "01/models" / filename
        if digest(model_path) != metadata["arrays_sha256"]:
            raise ValueError("source model changed")
        with np.load(model_path) as p:
            model = {k: p[k].copy() for k in p.files}
        for r in [r for r in design if r["query"] == metadata["query"]]:
            xs, vs = zip(
                *(
                    input_path(parent, r, k, metadata["representation"])
                    for k in range(spec["nodes"])
                ),
                strict=True,
            )
            x = (np.array(xs) - model["origin"]) / model["input_unit"]
            v = np.array(vs) / model["input_unit"]
            prediction, derivative = H.predict(model, x), H.predict(model, x, v)
            records = [
                read_json(node_path(parent, r["hm24_unit"], k))
                for k in range(spec["nodes"])
            ]
            common = {
                "unit": r["hm24_unit"],
                "family": r["family"],
                "role": r["role"],
                "query": r["query"],
                "representation": metadata["representation"],
                "method": metadata["method"],
            }
            for k, record in enumerate(records):
                if record["address"] != r:
                    raise ValueError("evaluation node address mismatch")
                nodes.append(
                    {
                        **common,
                        "node": k,
                        "native": record["native"]["score"],
                        "prediction": float(prediction[k]),
                        "native_derivative": record["native_derivative"],
                        "predicted_derivative": float(derivative[k]),
                    }
                )
            actions.append(
                {
                    **common,
                    "native_action": records[-1]["native"]["score"]
                    - records[0]["native"]["score"],
                    "predicted_action": float(prediction[-1] - prediction[0]),
                }
            )
    (out / "02").mkdir()
    write_csv(out / "02/node_predictions.csv", nodes)
    write_csv(out / "02/actions.csv", actions)
    summary = summaries(nodes, actions)
    write_csv(out / "02/summary.csv", summary)
    write_json(
        out / "02/summary.json",
        {
            "status": "EVALUATED_FINITE_HM26_NOT_GLOBAL",
            "fits": 12,
            "nodes": len(nodes),
            "actions": len(actions),
            "summary_rows": len(summary),
            "new_model_calls": 0,
            "held_gradients_used_in_fit": False,
            "scope": "compiled upper-reader potentials; lower-model states required; controlled paths not people",
        },
    )
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in out.rglob("*")
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    print(read_json(out / "02/summary.json"), flush=True)
    print([r for r in summary if r["role"] == "application"], flush=True)


def main():
    """Run one finite stage; do not overwrite a completed or failed stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit", "evaluate", "check"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    collector_guard(ROOT)
    if args.stage == "check":
        check(args.out)
        print("HM26_FROZEN_FILES_VALID", flush=True)
    else:
        {"prepare": prepare, "fit": fit, "evaluate": evaluate}[args.stage](args.out)


if __name__ == "__main__":
    main()
