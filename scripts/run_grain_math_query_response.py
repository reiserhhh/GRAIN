#!/usr/bin/env python3
"""HM34 compile three source-only P-response laws without new model calls."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_crossed_field as C
from grain_research.construction import math_position_support as P
from grain_research.construction import math_query_context as U
from grain_research.construction import math_query_response as R
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_query_context as OLD
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.verify_grain_math_crossed_field import table

SPEC = ROOT / "configs/grain_math_query_response_v1.json"
PLAN = ROOT / "docs/GRAIN_EARLY_QUERY_RESPONSE_OPERATOR_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/early_query_response_operator_r1"
CODE = (
    "grain_research/construction/math_query_response.py",
    "scripts/run_grain_math_query_response.py",
    "scripts/verify_grain_math_query_response.py",
    "tests/test_grain_math_query_response.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def prepare(out):
    """Freeze all input hashes and protocols; no held numeric readout is loaded."""
    if out.exists():
        raise FileExistsError("preserve prior packet; resume explicitly")
    spec = read_json(SPEC)
    parent, verification = ROOT / spec["parent"], ROOT / spec["parent_verification"]
    pm = check(parent)
    if (
        read_json(verification / "summary.json")["status"]
        != "VERIFIED_FINITE_HM33_NOT_GLOBAL"
    ):
        raise ValueError("verified HM33 required")
    files = dict(pm["files"])
    for base, relative in ((parent, "02/artifact_hashes.json"),):
        ledger = read_json(base / relative)
        for name, sha in ledger.items():
            path = base / name
            if digest(path) != sha:
                raise ValueError("frozen predecessor changed " + str(path))
            files[str(path.relative_to(ROOT))] = sha
        files[str((base / relative).relative_to(ROOT))] = digest(base / relative)
    files[str((verification / "summary.json").relative_to(ROOT))] = digest(
        verification / "summary.json"
    )
    files.update({name: digest(ROOT / name) for name in CODE})
    rows = read_json(parent / "00/design.json")
    if len(rows) != 768 or any(r["pair"] not in spec["roles"][r["role"]] for r in rows):
        raise ValueError("original complete design/roles required")
    for name in ("00/code", "01/models", "02"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write_json(out / "00/design.json", rows)
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "inputs": {"00/design.json": digest(out / "00/design.json")},
            "spec": spec,
            "frozen_before_outputs": now(),
            "new_model_calls": 0,
            "neural_training": False,
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "source_fits": 0, "new_model_calls": 0},
    )
    print({"status": "PREPARED", "cubes": 48, "maximum_source_fits": 3}, flush=True)


def group(parent, rows):
    """Read one matched P pair and its16 gradients with exact token binding."""
    pm = read_json(parent / "00/manifest.json")
    by = {
        (r["donor_side"], r["receiver_side"], r["early_query_side"], r["query_side"]): r
        for r in rows
    }
    if set(by) != set(U.BITS) or len(rows) != 16:
        raise ValueError("complete sixteen addressed corners required")
    left, right = by[0, 0, 0, 0], by[0, 0, 1, 0]
    tokens = R.paired_tokens(left, right)
    p0 = OLD.state(parent, left["early_query_donor_unit"])["p"].astype(float)
    p1 = OLD.state(parent, right["early_query_donor_unit"])["p"].astype(float)
    if p0.shape != p1.shape or p0.shape[0] != len(tokens):
        raise ValueError("paired P coordinates changed")
    gradient, hstates = [], {}
    for bits in U.BITS:
        row = by[bits]
        if tuple(row["early_query_token_ids"]) != tokens:
            raise ValueError("P sequence varies within group")
        path, _ = OLD.paths(parent, row, pm)
        array_unit(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as bank:
            g, h = bank["gradient"].astype(float), bank["h"].copy()
        gradient.append(g)
        side = row["donor_side"]
        if side in hstates and not np.array_equal(hstates[side], h):
            raise ValueError("H donor changed across P/B/Q")
        hstates[side] = h
    cube = np.stack(gradient).reshape(2, 2, 2, 2, *gradient[0].shape)
    common, fixed = R.paired_gradient(cube)
    return {
        **{k: left[k] for k in ("pair", "wording", "query", "role")},
        "d": ((p0 - p1) / 2).ravel(),
        "g": common.ravel(),
        "fixed": fixed,
        "h": hstates,
        "by": by,
        "tokens": tokens,
        "p_shape": p0.shape,
    }


def selected(design, query, pairs):
    """Select semantic groups without reading their numeric states."""
    return [
        [r for r in design if (r["pair"], r["wording"], r["query"]) == (pair, w, query)]
        for pair in pairs
        for w in range(4)
    ]


def load_model(out, query):
    """Load hash-checked source coefficients with no application recalibration."""
    path = out / "01/models" / f"{query}.json"
    meta = read_json(path)
    if digest(path.with_suffix(".npz")) != meta["array_sha256"]:
        raise ValueError("source operator changed")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        return {**meta, **{k: arrays[k].copy() for k in arrays.files}}


def fit_source(out):
    """Fit exactly three source operators and freeze before any held readouts."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "PREPARED":
        raise ValueError("source fit requires untouched prepared packet")
    spec, design = m["spec"], read_json(out / "00/design.json")
    parent = ROOT / spec["parent"]
    hashes = {}
    for q in spec["queries"]:
        records = [group(parent, rows) for rows in selected(design, q, (0, 1))]
        if (
            len({r["tokens"] for r in records}) != 1
            or len({r["p_shape"] for r in records}) != 1
        ):
            raise ValueError("one fixed query coordinate space required")
        d, g = R.source_columns(records)
        model = R.fit(d, g)
        path = out / "01/models" / f"{q}.json"
        arrays = {k: model[k] for k in ("D", "G", "gram_inverse", "eigenvalues")}
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        write_json(
            path,
            {
                "query": q,
                "source_keys": [[r["pair"], r["wording"]] for r in records],
                "query_token_ids": records[0]["tokens"],
                "p_shape": records[0]["p_shape"],
                "gradient_shape": list(records[0]["fixed"].shape[3:]),
                **{k: v for k, v in model.items() if k not in arrays},
                "input_metric": "identity in original physical token-state coordinates",
                "array_sha256": digest(path.with_suffix(".npz")),
                "fitted_at": now(),
                "held_numeric_readouts_used": False,
            },
        )
        for p in (path, path.with_suffix(".npz")):
            hashes[str(p.relative_to(out))] = digest(p)
    write_json(
        out / "01/source_barrier.json",
        {
            "status": "THREE_SOURCE_OPERATORS_FROZEN",
            "files": hashes,
            "source_fits": 3,
            "new_model_calls": 0,
            "frozen_at": now(),
            "scope": "source spans only; arbitrary N(I-PiD) extension remains unidentified",
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "SOURCE_FROZEN", "source_fits": 3, "new_model_calls": 0},
    )
    print({"status": "SOURCE_FROZEN", "fits": 3}, flush=True)


def evaluate(out):
    """Cold-score all cubes and native contrasts without fitting any held law."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "SOURCE_FROZEN":
        raise ValueError("three source operators must be frozen first")
    barrier_path = out / "01/source_barrier.json"
    barrier = read_json(barrier_path)
    for name, sha in barrier["files"].items():
        if digest(out / name) != sha:
            raise ValueError("frozen source model changed")
    spec, design = m["spec"], read_json(out / "00/design.json")
    parent = ROOT / spec["parent"]
    pm = read_json(parent / "00/manifest.json")
    with np.load(parent / "00/directions.npz", allow_pickle=False) as bank:
        directions = {q: C.directions(bank[q + "_b"]) for q in spec["queries"]}
    original = {(int(r["unit"]), r["arm"]): r for r in table(parent / "02/actions.csv")}
    saved = np.load(parent / "02/cube_arrays.npz", allow_pickle=False)
    coefficients, fixed_rows, actions, arrays = [], [], [], {}
    for q in spec["queries"]:
        model = load_model(out, q)
        for rows in selected(design, q, range(4)):
            r = group(parent, rows)
            if (
                list(r["tokens"]) != model["query_token_ids"]
                or list(r["p_shape"]) != model["p_shape"]
            ):
                raise ValueError("cold query coordinates changed")
            result = R.predict(model, r["d"])
            predicted = result["predicted"].reshape(model["gradient_shape"])
            common = r["g"].reshape(model["gradient_shape"])
            key = f"p{r['pair']}_w{r['wording']}_{q}"
            old = saved[key + "_gradient_none_coefficients"][2]
            identity = float(np.max(abs(old - common)))
            if identity > 1e-10:
                raise ValueError(
                    "independent pure P averaging disagrees with character"
                )
            address = {k: r[k] for k in ("pair", "wording", "query", "role")}
            dnorm = float(np.linalg.norm(r["d"]))
            coefficients.append(
                {
                    **address,
                    "source_rank": model["rank"],
                    "input_norm": dnorm,
                    "supported_norm": float(np.linalg.norm(result["supported"])),
                    "unsupported_norm": float(np.linalg.norm(result["unsupported"])),
                    "unsupported_ratio": float(np.linalg.norm(result["unsupported"]))
                    / dnorm
                    if dnorm
                    else None,
                    "character_identity_error": identity,
                    **R.statistics(common, predicted),
                    **{
                        a + suffix: float(
                            np.sum(
                                v * directions[q][a] / np.linalg.norm(directions[q][a])
                            )
                        )
                        for a in spec["arms"]
                        for suffix, v in (
                            ("_actual", common),
                            ("_predicted", predicted),
                        )
                    },
                }
            )
            arrays.update(
                {
                    key + "_" + k: v
                    for k, v in {
                        "input": r["d"],
                        "common": common,
                        "fixed": r["fixed"],
                        "predicted": predicted,
                        "weights": result["weights"],
                        "unsupported": result["unsupported"],
                    }.items()
                }
            )
            for hside, bside, qside in itertools.product(range(2), repeat=3):
                fixed = r["fixed"][hside, bside, qside]
                fixed_rows.append(
                    {
                        **address,
                        "H": hside,
                        "B": bside,
                        "Q": qside,
                        **R.statistics(fixed, predicted),
                    }
                )
                for arm in spec["arms"]:
                    units = [
                        r["by"][hside, bside, pside, qside]["unit"] for pside in (0, 1)
                    ]
                    native = (
                        float(original[units[0], arm]["odd_action"])
                        - float(original[units[1], arm]["odd_action"])
                    ) / 2
                    physical = P.physical_patches(
                        r["h"][hside],
                        directions[q][arm],
                        pm["spec"]["radius_relative_span_norm"],
                    )
                    delta = physical["half_delta"]
                    actions.append(
                        {
                            **address,
                            "H": hside,
                            "B": bside,
                            "Q": qside,
                            "arm": arm,
                            "left_unit": units[0],
                            "right_unit": units[1],
                            "physical_delta_norm": float(np.linalg.norm(delta)),
                            **R.error_parts(predicted, common, fixed, delta, native),
                        }
                    )
    saved.close()
    metrics = []
    for role in spec["roles"]:
        for q in spec["queries"]:
            chosen = [r for r in coefficients if (r["role"], r["query"]) == (role, q)]
            for arm in spec["arms"]:
                a = [
                    r
                    for r in actions
                    if (r["role"], r["query"], r["arm"]) == (role, q, arm)
                ]
                y = np.array([r["actual"] for r in a])
                metric = {
                    "role": role,
                    "query": q,
                    "arm": arm,
                    "cubes": len(chosen),
                    "unsupported_ratio_mean": float(
                        np.mean([r["unsupported_ratio"] for r in chosen])
                    ),
                    **{
                        method + "_" + k: v
                        for method in ("predicted", "common_local", "fixed_local")
                        for k, v in R.statistics(y, [r[method] for r in a]).items()
                    },
                    **{
                        "coefficient_" + k: v
                        for k, v in R.statistics(
                            [r[arm + "_actual"] for r in chosen],
                            [r[arm + "_predicted"] for r in chosen],
                        ).items()
                    },
                }
                parts = np.array(
                    [
                        [
                            r[k]
                            for k in (
                                "operator_error",
                                "interaction_error",
                                "finite_error",
                            )
                        ]
                        for r in a
                    ]
                )
                for i, name in enumerate(("operator", "interaction", "finite")):
                    metric[name + "_mse"] = float(np.mean(parts[:, i] ** 2))
                    for j in range(i + 1, 3):
                        metric[f"cross_{i}_{j}"] = float(
                            2 * np.mean(parts[:, i] * parts[:, j])
                        )
                metric["error_mse_identity"] = abs(
                    float(np.mean((np.array([r["predicted"] for r in a]) - y) ** 2))
                    - sum(
                        metric[k]
                        for k in metric
                        if k.endswith("_mse") or k.startswith("cross_")
                    )
                )
                metrics.append(metric)
    for name, data in (
        ("coefficients", coefficients),
        ("fixed_responses", fixed_rows),
        ("actions", actions),
        ("metrics", metrics),
    ):
        write_csv(out / f"02/{name}.csv", data)
    np.savez_compressed(out / "02/response_arrays.npz", **arrays)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM34_EXECUTED_NOT_GLOBAL",
            "counts": {
                "source_fits": 3,
                "coefficient_rows": len(coefficients),
                "fixed_gradient_rows": len(fixed_rows),
                "finite_action_contrasts": len(actions),
                "metric_rows": len(metrics),
                "new_model_calls": 0,
            },
            "source_barrier_sha256": digest(barrier_path),
            "evaluated_at": now(),
            "scope": spec["scope"],
            "failures_retained": True,
        },
    )
    ledger = {
        str(p.relative_to(out)): digest(p)
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name not in ("artifact_hashes.json", "run_status.json")
    }
    write_json(out / "02/artifact_hashes.json", ledger)
    write_json(
        out / "run_status.json",
        {"status": "EVALUATED", "source_fits": 3, "new_model_calls": 0},
    )
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Run one resumable frozen CPU stage, never launch a model or old queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit_source", "evaluate", "check"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "check":
        print({"frozen_files": len(check(args.out)["files"])}, flush=True)
    else:
        {"prepare": prepare, "fit_source": fit_source, "evaluate": evaluate}[
            args.stage
        ](args.out)


if __name__ == "__main__":
    main()
