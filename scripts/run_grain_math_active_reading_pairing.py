#!/usr/bin/env python3
"""HM38 bounded actual-reader pairing diagnostic, preserving all HM37 errors."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_active_reading_pairing as P
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_target_active_support as SUPPORT
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_active_reading_pairing_v1.json"
PLAN = ROOT / "docs/GRAIN_ACTIVE_READING_PAIRING_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/active_reading_pairing_r1"
CODE = (
    "grain_research/construction/math_active_reading_pairing.py",
    "scripts/run_grain_math_active_reading_pairing.py",
    "scripts/verify_grain_math_active_reading_pairing.py",
    "tests/test_grain_math_active_reading_pairing.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def prepare(out):
    """Freeze the diagnostic before reading paired derivative/action arrays."""
    started = time.monotonic()
    if out.exists():
        raise FileExistsError("preserve existing diagnostic; no overwrite")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    files = check(parent)["files"]
    for name, sha in read_json(parent / "02/artifact_hashes.json").items():
        if digest(parent / name) != sha:
            raise ValueError("frozen parent changed " + name)
        files[str((parent / name).relative_to(ROOT))] = sha
    vf = ROOT / spec["verification"]
    if read_json(vf)["status"] != "VERIFIED_FINITE_HM37_NOT_GLOBAL":
        raise ValueError("verified parent required")
    files[str(vf.relative_to(ROOT))] = digest(vf)
    files.update({name: digest(ROOT / name) for name in CODE})
    design = read_json(parent / "00/design.json")
    if len(design) != 96 or {p["wording"] for p in design} != {0, 2}:
        raise ValueError("unchanged two-wording96path bank required")
    for sub in ("00/code", "01/arrays", "02"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_json(out / "00/design.json", design)
    for name in CODE:
        destination = out / "00/code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "inputs": {"00/design.json": digest(out / "00/design.json")},
            "spec": spec,
            "frozen_at": now(),
            "new_fits": 0,
            "new_model_calls": 0,
        },
    )
    elapsed = time.monotonic() - started
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "analysis_seconds": elapsed,
            "paired_groups": 0,
            "new_fits": 0,
            "new_model_calls": 0,
        },
    )
    if elapsed > spec["maximum_analysis_seconds"]:
        raise TimeoutError("preparation exhausted diagnostic budget")
    print({"status": "PREPARED", "seconds": elapsed}, flush=True)


def checkpoint(out, status, previous, started, spec):
    """Charge ancestry validation, numerical work and artifact writing together."""
    status["analysis_seconds"] = previous + time.monotonic() - started
    write_json(out / "run_status.json", status)
    if status["analysis_seconds"] > spec["maximum_analysis_seconds"]:
        raise TimeoutError("fixed120-second diagnostic budget reached")


def half(rows, rule, keys):
    """Integrate scalar node terms, never products of integrated marginal vectors."""
    weights = (1 / 12, 1 / 3, 1 / 12) if rule == "simpson" else (1 / 8, 1 / 4, 1 / 8)
    return {k: sum(w * r[k] for w, r in zip(weights, rows, strict=True)) for k in keys}


def analyze(out):
    """Decompose all source/application pairs without fitting or choosing cells."""
    started = time.monotonic()
    manifest = check(out)
    spec = manifest["spec"]
    parent = ROOT / spec["parent"]
    status = read_json(out / "run_status.json")
    previous = status["analysis_seconds"]
    if status["status"] != "PREPARED":
        raise ValueError("untouched prepared packet required")
    status["status"] = "ANALYZING"
    checkpoint(out, status, previous, started, spec)
    design = read_json(out / "00/design.json")
    groups = {}
    for path in design:
        key = path["query"], path["role"], path["H"], path["B"], path["Q"]
        groups.setdefault(key, []).append(path)
    if len(groups) != 48:
        raise ValueError("all48addressed pairs required")
    models = {
        q: SUPPORT.load(parent / "01/models" / f"{q}.json") for q in spec["queries"]
    }
    paired_nodes, paired_integrals, wording_nodes, wording_integrals = [], [], [], []
    provenance = []
    maximum = 0.0
    order = sorted(
        groups,
        key=lambda k: (spec["queries"].index(k[0]), k[1] == "application", k[2:]),
    )
    for key in order:
        query, role, h, b, q = key
        paths = sorted(groups[key], key=lambda p: p["wording"])
        if [p["wording"] for p in paths] != spec["wordings"] or len(
            {p["pair"] for p in paths}
        ) != 1:
            raise ValueError("one ordered two-wording pair required")
        cell = h * 4 + b * 2 + q
        model_meta, model = models[query]
        rank = model_meta["rank"]
        arrays = {
            "theta": np.empty((3, 2, 2, rank)),
            "eta": np.empty((3, 2, 2, rank)),
            "reference": model["laws"][3, cell],
        }
        for name in ("full", "supported", "predicted", "norms", "direction_error"):
            arrays[name] = np.empty((3, 2, 2))
        for wi, path in enumerate(paths):
            pf = parent / "01/predictions" / f"path_{path['path']:03d}.json"
            of = parent / "02/observations" / pf.name
            pm, pred = SUPPORT.load(pf)
            om, obs = SUPPORT.load(of)
            if (
                pm["address"] != path
                or om["address"] != path
                or om["prediction_sha256"] != digest(pf)
            ):
                raise ValueError("parent prediction/observation address changed")
            if path["left"]["early_query_token_ids"] != model_meta["token_ids"]:
                raise ValueError("separate ordered query coordinates required")
            for node in range(3):
                theta, eta = P.physical_pair(
                    obs["coefficients"][node], pred["z"][node], pred["h_norms"][node]
                )
                arrays["theta"][node, :, wi], arrays["eta"][node, :, wi] = theta, eta
            arrays["full"][..., wi], arrays["supported"][..., wi] = (
                obs["actual"],
                obs["supported"],
            )
            arrays["predicted"][..., wi] = pred["predicted"][:, 3]
            arrays["norms"][..., wi], arrays["direction_error"][..., wi] = (
                pred["h_norms"],
                pred["h_direction_error"],
            )
            provenance.append(
                {
                    "query": query,
                    "role": role,
                    "cell": cell,
                    "path": path["path"],
                    "wording": path["wording"],
                    "prediction_file": str(pf.relative_to(ROOT)),
                    "prediction_sha256": digest(pf),
                    "observation_file": str(of.relative_to(ROOT)),
                    "observation_sha256": digest(of),
                }
            )
        group = f"{query}_{role}_cell_{cell}"
        address = {
            "group": group,
            "query": query,
            "role": role,
            "cell": cell,
            "H": h,
            "B": b,
            "Q": q,
            "pair": paths[0]["pair"],
        }
        SUPPORT.save(
            out / "01/arrays" / f"{group}.json",
            {
                "address": address,
                "wordings": spec["wordings"],
                "paths": [p["path"] for p in paths],
                "parent": spec["parent"],
                "new_fits": 0,
                "new_model_calls": 0,
            },
            arrays,
        )
        for ai, arm in enumerate(spec["arms"]):
            pr, wr = [], [[], []]
            for node in range(3):
                pair, words = P.decompose(
                    arrays["theta"][node, ai],
                    arrays["eta"][node, ai],
                    arrays["full"][node, ai],
                    arrays["reference"][node, ai],
                )
                maximum = max(
                    maximum,
                    pair["pairing_identity_error"],
                    pair["error_identity_error"],
                )
                pair_row = {**address, "arm": arm, "node": node, "t": node / 2, **pair}
                paired_nodes.append(pair_row)
                pr.append(pair_row)
                for wi, word in enumerate(words):
                    err = max(
                        abs(word["supported"] - arrays["supported"][node, ai, wi]),
                        abs(word["predicted"] - arrays["predicted"][node, ai, wi]),
                        word["error_identity_error"],
                        word["supported_identity_error"],
                    )
                    maximum = max(maximum, err)
                    row = {
                        **address,
                        "arm": arm,
                        "node": node,
                        "t": node / 2,
                        "wording": spec["wordings"][wi],
                        "path": paths[wi]["path"],
                        "H_norm": float(arrays["norms"][node, ai, wi]),
                        "H_direction_difference": float(
                            arrays["direction_error"][node, ai, wi]
                        ),
                        **word,
                    }
                    wording_nodes.append(row)
                    wr[wi].append(row)
            pair_keys = (
                "actual",
                "supported",
                "predicted",
                "common",
                *P.PAIR_COMPONENTS,
                "error",
                "supported_contrast",
                "reading_contrast",
                "coefficient_contrast",
                "input_contrast",
            )
            word_keys = (
                "actual",
                "supported",
                "predicted",
                *P.WORD_COMPONENTS,
                "error",
            )
            for rule in spec["rules"]:
                values = half(pr, rule, pair_keys)
                values["pairing_identity_error"] = abs(
                    values["supported"] - values["common"] - values["coupling"]
                )
                values["error_identity_error"] = abs(
                    values["error"] - sum(values[k] for k in P.PAIR_COMPONENTS)
                )
                paired_integrals.append({**address, "arm": arm, "rule": rule, **values})
                for wi in range(2):
                    values = half(wr[wi], rule, word_keys)
                    values["error_identity_error"] = abs(
                        values["error"] - sum(values[k] for k in P.WORD_COMPONENTS)
                    )
                    wording_integrals.append(
                        {
                            **address,
                            "arm": arm,
                            "wording": spec["wordings"][wi],
                            "path": paths[wi]["path"],
                            "rule": rule,
                            **values,
                        }
                    )
        status["paired_groups"] += 1
        checkpoint(out, status, previous, started, spec)
    if maximum > spec["absolute_tolerance"]:
        raise ValueError("pairing or parent error replay exceeded original tolerance")
    metrics = []
    for view, nodes, integrals, components in (
        ("pair", paired_nodes, paired_integrals, P.PAIR_COMPONENTS),
        ("wording", wording_nodes, wording_integrals, P.WORD_COMPONENTS),
    ):
        for role in ("source", "application"):
            for query in spec["queries"]:
                for arm in spec["arms"]:
                    for scale in ("nodes", *spec["rules"]):
                        rows = (
                            nodes
                            if scale == "nodes"
                            else [r for r in integrals if r["rule"] == scale]
                        )
                        selected = [
                            r
                            for r in rows
                            if (r["role"], r["query"], r["arm"]) == (role, query, arm)
                        ]
                        metrics.append(
                            {
                                "view": view,
                                "role": role,
                                "query": query,
                                "arm": arm,
                                "scale": scale,
                                **P.component_metrics(selected, components),
                            }
                        )
    for name, rows, expected in (
        ("paired_nodes", paired_nodes, 288),
        ("paired_integrals", paired_integrals, 192),
        ("wording_nodes", wording_nodes, 576),
        ("wording_integrals", wording_integrals, 384),
        ("metrics", metrics, 72),
    ):
        if len(rows) != expected:
            raise ValueError("incomplete fixed output " + name)
        write_csv(out / "02" / f"{name}.csv", rows)
    write_csv(out / "01/provenance.csv", provenance)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM38_EVALUATED_NOT_GLOBAL",
            "groups": status["paired_groups"],
            "counts": {
                "paired_nodes": 288,
                "paired_integrals": 192,
                "wording_nodes": 576,
                "wording_integrals": 384,
                "metrics": 72,
            },
            "maximum_formula_error": maximum,
            "new_fits": 0,
            "new_model_calls": 0,
            "scope": spec["scope"],
        },
    )
    status["status"] = "EVALUATED"
    checkpoint(out, status, previous, started, spec)
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for sub in ("01", "02")
            for p in (out / sub).rglob("*")
            if p.is_file() and not p.name.startswith("._")
        },
    )
    checkpoint(out, status, previous, started, spec)
    print(
        {
            "status": "EVALUATED",
            "seconds": status["analysis_seconds"],
            "groups": status["paired_groups"],
            "maximum_formula_error": maximum,
        },
        flush=True,
    )


def main():
    """Run one explicit finite stage; preserve failures and reject automatic replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        try:
            if args.stage in ("prepare", "all"):
                prepare(args.out)
            if args.stage in ("analyze", "all"):
                analyze(args.out)
        except Exception as error:
            if args.out.exists():
                write_json(
                    args.out / "failure.json",
                    {
                        "at": now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "new_fits": 0,
                        "new_model_calls": 0,
                    },
                )
            raise


if __name__ == "__main__":
    main()
