#!/usr/bin/env python3
"""Observe one frozen relation operation through several fixed endpoints."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_endpoint_action as M
from grain_research.construction import math_qualified_target_field as Q
from grain_research.construction import math_relation_entry_channels as A
from scripts import run_grain_relation_entry_channels_hm158 as G

C, R = G.R, G.R.R
DEFAULT = ROOT / "results/grain_math_response_v1/endpoint_actions_r1"
CONFIG = "configs/grain_endpoint_action_hm161.json"
CODE = (
    CONFIG,
    "grain_research/construction/math_endpoint_action.py",
    "scripts/run_grain_endpoint_action_hm161.py",
    "scripts/verify_grain_endpoint_action_hm161.py",
    "tests/test_grain_endpoint_action.py",
    "docs/GRAIN_ENDPOINT_ACTION_PLAN_HM161.md",
    "scripts/run_grain_relation_entry_channels_hm158.py",
    "grain_research/construction/math_relation_entry_channels.py",
)


def write_csv(path, rows):
    """Write a complete nonempty result table once, without overwriting old work."""
    if not rows:
        raise ValueError("A missing result table is not an empty successful run")
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check(out):
    """Verify fixed design, inherited data and source-only scientific code."""
    m = R.read(out / "manifest.json")
    C.check(ROOT / m["parent"])
    for name, checksum in m["hashes"].items():
        if C.H.digest(ROOT / name) != checksum:
            raise ValueError("Scientific dependency changed: " + name)
    for name, checksum in m["inputs"].items():
        if C.H.digest(out / name) != checksum:
            raise ValueError("Registered endpoint input changed: " + name)
    return m


def prepare(out):
    """Freeze all endpoint text and operations before observing their outputs."""
    from transformers import AutoTokenizer

    config = R.read(ROOT / CONFIG)
    parent = ROOT / config["parent"]
    old = C.check(parent)
    G.authenticate_results(parent)
    tokenizer = AutoTokenizer.from_pretrained(str(R.J.MODEL), local_files_only=True)
    rows = R.read(parent / "rows.json")
    for row in rows:
        row["role"] = "historical_public_fictional_record_new_endpoint_suffix"
    definitions, texts = M.questions()
    queries = {
        "definitions": definitions,
        "questions": texts,
        "suffixes": [
            tokenizer.encode(
                "\nQuestion: " + q + "\n<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False,
            )
            for q in texts
        ],
        "answers": {
            s: tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
        },
    }
    if any(len(ids) != 1 for ids in queries["answers"].values()):
        raise ValueError("Registered native answers must be single tokens")
    maximum = max(
        len(row["input_ids"]) + len(s) for row in rows for s in queries["suffixes"]
    )
    if maximum > 768 or len(rows) != 16 or len(definitions) != 16:
        raise ValueError("Complete fixed support required; no truncation")
    old_queries = R.read(parent / "queries.json")
    for i, d in enumerate(definitions):
        if d["family"] == "diagnostic":
            j = next(
                j
                for j, q in enumerate(old_queries["definitions"])
                if q == {**d, "family": "named"}
            )
            if queries["suffixes"][i] != old_queries["suffixes"][j]:
                raise ValueError("Diagnostic replay endpoint changed")
    design = [
        {"batch": i, "row": row["id"], "mode": "natural"} for i, row in enumerate(rows)
    ]
    for row in config["self_rows"]:
        design.append({"batch": len(design), "row": row, "mode": "self"})
    for axis, row in itertools.product(("e", "n"), rows):
        design.append(
            {"batch": len(design), "row": row["id"], "mode": axis + "_relation"}
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    for name, value in (("rows", rows), ("queries", queries), ("design", design)):
        C.P.write_json(out / f"{name}.json", value)
    dependencies = list(CODE)
    dependencies += [
        str((parent / name).relative_to(ROOT))
        for name in (
            "manifest.json",
            "capture_seal.json",
            "output_seal.json",
            "source_operations.npz",
            "rows.json",
            "queries.json",
        )
    ]
    C.P.write_json(
        out / "manifest.json",
        {
            **config,
            "hashes": {name: C.H.digest(ROOT / name) for name in dependencies},
            "inputs": {
                name: C.H.digest(out / name)
                for name in ("rows.json", "queries.json", "design.json")
            },
            "model_weights_sha256": old["model_weights_sha256"],
            "maximum_tokens": maximum,
            "batches": len(design),
            "inquiry_sequences": len(design) * len(definitions),
        },
    )
    R.status(out, "PREPARED", batches_done=0, pending=None)
    print({"prepared_batches": len(design), "new_endpoint_inquiries": 800})


def parent_natural(parent):
    """Read authenticated old natural captures, not future-choice outputs."""
    G.authenticate_results(parent)
    return {
        d["row"]: R.arrays(parent / "captures" / f"{d['batch']:03d}.npz")
        for d in R.read(parent / "design.json")
        if d["mode"] == "natural"
    }


def collect(out):
    """Run50 finite batches once; do not refit, replay or renew a failed run."""
    m, start = check(out), time.monotonic()
    if R.read(out / "collection_status.json")["status"] != "PREPARED":
        raise RuntimeError("Started endpoint collections cannot be replayed")
    parent = ROOT / m["parent"]
    old = parent_natural(parent)
    rows, queries, design = (
        R.read(out / f"{name}.json") for name in ("rows", "queries", "design")
    )
    source = R.arrays(parent / "source_operations.npz")
    original_queries = R.read(parent / "queries.json")
    diagnostic_pairs = [
        (
            i,
            next(
                j
                for j, d in enumerate(original_queries["definitions"])
                if d == {**query, "family": "named"}
            ),
        )
        for i, query in enumerate(queries["definitions"])
        if query["family"] == "diagnostic"
    ]
    natural, done, pending = {}, 0, None
    try:
        R.status(out, "LOADING", pid=os.getpid(), batches_done=0)
        observer = R.J.offline_model(m)
        C.P.write_json(out / "model_manifest.json", observer.manifest())
        for entry in design:
            if time.monotonic() - start > m["collection_seconds_cap"]:
                raise TimeoutError("Finite420second endpoint cap; no renewal")
            pending, row = entry, rows[entry["row"]]
            mode = entry["mode"]
            R.status(out, "RUNNING", pid=os.getpid(), batches_done=done, pending=entry)
            if mode == "natural":
                donors, slots = None, []
            else:
                donors, slots = G.arguments(
                    row,
                    "self" if mode == "self" else mode[0] + "_joint_control",
                    natural[row["id"]],
                    source,
                )
            result = R.native_batch(observer, row, queries, donors)
            if float(result["cache_spread"]) > 1e-4:
                raise ValueError("Different suffixes changed causal historical memory")
            if mode == "natural":
                natural[row["id"]] = result
                prefix_reference = old[row["id"]]["prefix_states"]
                error = max(
                    float(
                        np.max(
                            abs(
                                result["selected_logits"][i]
                                - old[row["id"]]["selected_logits"][j]
                            )
                        )
                    )
                    for i, j in diagnostic_pairs
                )
                result["diagnostic_replay_error"] = np.asarray(error)
                if error > 1e-4:
                    raise ValueError("Unchanged diagnostic failed replay tolerance")
            else:
                prefix_reference = natural[row["id"]]["prefix_states"]
                result["donor_key"], result["donor_value"] = donors
                result["patched_slots"] = np.asarray(slots)
                if mode == "self":
                    error = float(
                        np.max(
                            abs(
                                result["selected_logits"]
                                - natural[row["id"]]["selected_logits"]
                            )
                        )
                    )
                    result["self_error"] = np.asarray(error)
                    if error > 1e-4:
                        raise ValueError("Self intervention changed endpoint")
            prefix_error = float(
                np.max(abs(result["prefix_states"] - prefix_reference))
            )
            if prefix_error > 1e-4:
                raise ValueError("Suffix-only operation changed raw prefix")
            result["prefix_error"] = np.asarray(prefix_error)
            np.savez_compressed(
                out / "captures" / f"{entry['batch']:03d}.npz", **result
            )
            done += 1
            R.status(
                out,
                "RUNNING",
                pid=os.getpid(),
                batches_done=done,
                pending=entry,
                seconds=time.monotonic() - start,
            )
            print({"batch": done, "total": 50}, flush=True)
        C.P.write_json(
            out / "capture_seal.json",
            {
                str(p.relative_to(out)): C.H.digest(p)
                for p in sorted((out / "captures").glob("*.npz"))
            },
        )
        R.status(
            out,
            "COLLECTED",
            pid=os.getpid(),
            batches_done=done,
            pending=None,
            seconds=time.monotonic() - start,
            model_batches=observer.calls,
            inquiry_sequences=done * len(queries["definitions"]),
        )
    except Exception as exc:
        R.status(
            out,
            "FAILED",
            pid=os.getpid(),
            batches_done=done,
            pending=pending,
            seconds=time.monotonic() - start,
            error=repr(exc),
        )
        raise


def analyze(out):
    """Keep all target/condition/character actions and distinguish policy mass."""
    m, start = check(out), time.monotonic()
    if (
        R.read(out / "collection_status.json")["status"] != "COLLECTED"
        or (out / "summary.json").exists()
    ):
        raise RuntimeError("Complete unanalysed collection required")
    for name, checksum in R.read(out / "capture_seal.json").items():
        if C.H.digest(out / name) != checksum:
            raise ValueError("Actual endpoint capture changed")
    rows, queries, design = (
        R.read(out / f"{name}.json") for name in ("rows", "queries", "design")
    )
    indexed = {(r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows}
    scores, endpoints = {}, []
    replay, self_error, prefix_error = 0.0, 0.0, 0.0
    for entry in design:
        raw = R.arrays(out / "captures" / f"{entry['batch']:03d}.npz")
        scores[entry["row"], entry["mode"]] = M.endpoint_scores(
            raw["selected_logits"], raw["normalizer"]
        )
        replay = max(replay, float(raw.get("diagnostic_replay_error", 0)))
        self_error = max(self_error, float(raw.get("self_error", 0)))
        prefix_error = max(prefix_error, float(raw["prefix_error"]))
        for i, query in enumerate(queries["definitions"]):
            endpoints.append(
                {
                    **entry,
                    **query,
                    **{
                        k: rows[entry["row"]][k]
                        for k in ("condition", "order", "e", "n")
                    },
                    "yes_logit": float(raw["selected_logits"][i, 0]),
                    "no_logit": float(raw["selected_logits"][i, 1]),
                    "normalizer": float(raw["normalizer"][i]),
                    **{
                        name: float(value[i])
                        for name, value in scores[entry["row"], entry["mode"]].items()
                    },
                }
            )
    table, fields = [], {}
    for condition, order, axis, target, family, measure in itertools.product(
        range(2), range(2), ("e", "n"), M.AXES, M.FAMILIES, M.MEASURES
    ):
        if time.monotonic() - start > m["analysis_seconds_cap"]:
            raise TimeoutError("Finite analysis cap; no renewal")
        cubes = {k: np.zeros((2, 2, 2)) for k in ("base", "source", "natural")}
        for ei, ni, pi in itertools.product(range(2), repeat=3):
            e, n, person = 2 * ei - 1, 2 * ni - 1, 2 * pi - 1
            row_id = indexed[condition, order, e, n]
            partner = M.natural_partner(rows[row_id], axis, indexed)
            query = {"axis": target, "family": family, "person": person}
            qi = queries["definitions"].index(query)
            baseline = scores[row_id, "natural"][measure][qi]
            cubes["base"][ei, ni, pi] = baseline
            cubes["source"][ei, ni, pi] = (
                scores[row_id, axis + "_relation"][measure][qi] - baseline
            )
            cubes["natural"][ei, ni, pi] = (
                scores[partner, "natural"][measure][qi] - baseline
            )
        for component, cube in cubes.items():
            coeff = Q.characters(cube)
            fields[condition, order, axis, target, family, measure, component] = coeff
            for mask, value in enumerate(coeff):
                table.append(
                    {
                        "condition": condition,
                        "order": order,
                        "axis": axis,
                        "target": target,
                        "family": family,
                        "measure": measure,
                        "component": component,
                        "mask": mask,
                        "coefficient": float(value),
                    }
                )
    transport, contrasts, cross = [], [], []
    for axis, target, family, measure in itertools.product(
        ("e", "n"), M.AXES, M.FAMILIES, M.MEASURES
    ):
        mask = 5 if target == "extraversion" else 6

        def qualified(
            component,
            family=family,
            axis=axis,
            target=target,
            measure=measure,
            mask=mask,
        ):
            """Retain the registered actor-qualified character in each condition."""
            return np.array(
                [
                    fields[c, o, axis, target, family, measure, component][mask]
                    for c, o in itertools.product(range(2), repeat=2)
                ]
            )

        source_action, natural_action = qualified("source"), qualified("natural")
        transport.append(
            {
                "axis": axis,
                "target": target,
                "family": family,
                "measure": measure,
                "mask": mask,
                **A.action_description(source_action, natural_action),
            }
        )
        if family == "prospective":
            contrasts.append(
                {
                    "axis": axis,
                    "target": target,
                    "measure": measure,
                    **M.policy_contrast(
                        source_action,
                        qualified("source", "forced_yes"),
                        qualified("source", "forced_no"),
                    ),
                }
            )
            cross.append(
                {
                    "axis": axis,
                    "target": target,
                    "measure": measure,
                    **A.action_description(
                        source_action, qualified("source", "diagnostic")
                    ),
                }
            )
    mass = [r for r in endpoints if r["mode"] == "natural"]
    summary = {
        "packet": "HM161",
        "status": "FINITE_ENDPOINT_ACTION_COMPLETE_NOT_GLOBAL",
        "collection": R.read(out / "collection_status.json"),
        "source_answer_fitting": False,
        "diagnostic_replay_error": replay,
        "self_error": self_error,
        "prefix_error": prefix_error,
        "natural_endpoint_summary": [
            {
                "target": target,
                "family": family,
                "restricted_yes_mean": float(
                    np.mean(
                        [
                            r["restricted_yes"]
                            for r in mass
                            if r["axis"] == target and r["family"] == family
                        ]
                    )
                ),
                "answer_mass_min": min(
                    r["answer_mass"]
                    for r in mass
                    if r["axis"] == target and r["family"] == family
                ),
                "answer_mass_mean": float(
                    np.mean(
                        [
                            r["answer_mass"]
                            for r in mass
                            if r["axis"] == target and r["family"] == family
                        ]
                    )
                ),
            }
            for target, family in itertools.product(M.AXES, M.FAMILIES)
        ],
        "transport": transport,
        "policy_contrasts": contrasts,
        "cross_endpoint_action": cross,
        "analysis_seconds": time.monotonic() - start,
        "original_scope_complete": False,
        "interpretation": "Output-policy endpoints on historical fictional evidence; source operations frozen, new suffixes fixed prospectively. Neither observed human behavior nor a newly identified psychological trait.",
    }
    for name, values in (
        ("endpoints.csv", endpoints),
        ("characters.csv", table),
        ("transport.csv", transport),
        ("policy_contrasts.csv", contrasts),
        ("cross_endpoint_action.csv", cross),
    ):
        write_csv(out / name, values)
    C.P.write_json(out / "summary.json", summary)
    C.P.write_json(
        out / "output_seal.json",
        {
            p.name: C.H.digest(p)
            for p in sorted(out.iterdir())
            if p.suffix == ".csv" or p.name == "summary.json"
        },
    )
    print(
        {
            "status": summary["status"],
            "matched_transport": [
                r
                for r in transport
                if r["measure"] == "restricted_yes"
                and ((r["axis"] == "e") == (r["target"] == "extraversion"))
            ],
        },
        flush=True,
    )


def main():
    """Expose preparation, one finite collection and immutable analysis separately."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](
        args.output.resolve()
    )


if __name__ == "__main__":
    main()
