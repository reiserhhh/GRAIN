#!/usr/bin/env python3
"""Locate qualified evidence carriers; do not fit prediction or patch weights."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_carrier_closure as M
from scripts import run_grain_qualified_target_field_hm155 as R

H = R.H
DEFAULT = ROOT / "results/grain_math_response_v1/target_carrier_closure_r1"
MODES = (
    "e_tail",
    "n_tail",
    "e_other",
    "n_other",
    "e_closure",
    "n_closure",
    "both_tail_real",
    "both_tail_additive",
)
CODE = (
    "grain_research/construction/math_target_carrier_closure.py",
    "scripts/run_grain_target_carrier_closure_hm156.py",
    "scripts/verify_grain_target_carrier_closure_hm156.py",
    "tests/test_grain_target_carrier_closure.py",
    "docs/GRAIN_TARGET_CARRIER_CLOSURE_PLAN_HM156.md",
)


def write_json(path, value):
    """Use the inherited strict writer with explicit finite native conversion."""
    H.write_json(path, M.json_native(value))


def array_digest(a):
    """Authenticate actual memory arguments including shape and dtype."""
    x = np.ascontiguousarray(a)
    h = hashlib.sha256(str((x.shape, x.dtype.str)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def authenticate_parent(parent):
    """Check original scientific seals, including its preserved failed summary."""
    manifest = R.check(parent)
    for name in ("capture_seal.json", "output_seal.json"):
        for path, checksum in R.read(parent / name).items():
            if H.digest(parent / path) != checksum:
                raise ValueError("Parent result changed: " + path)
    if R.read(parent / "collection_status.json")["status"] != "COLLECTED":
        raise ValueError("Parent observation support is incomplete")
    return manifest


def check(out):
    """Verify all pre-outcome dependencies and immutable prospective inputs."""
    manifest = R.read(out / "manifest.json")
    for name, checksum in manifest["hashes"].items():
        if H.digest(ROOT / name) != checksum:
            raise ValueError("Sealed code/source changed: " + name)
    for name, checksum in manifest["inputs"].items():
        if H.digest(out / name) != checksum:
            raise ValueError("Prepared input changed: " + name)
    return manifest


def parent_natural(parent):
    """Read complete frozen natural memories without rerunning the old model."""
    return {
        e["row"]: R.arrays(parent / "captures" / f"{e['batch']:03d}.npz")
        for e in R.read(parent / "design.json")
        if e["mode"] == "natural"
    }


def prepare(out, parent):
    """Freeze partitions and all260 native interventions before model calls."""
    manifest = authenticate_parent(parent)
    rows, queries = (R.read(parent / f"{name}.json") for name in ("rows", "queries"))
    if len(rows) != 32 or len(queries["definitions"]) != 12:
        raise ValueError("No outcome-dependent support reduction")
    indexed = {(r["bank"], r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    scopes = {str(r["id"]): M.carrier_scopes(r) for r in rows}
    for row in rows:
        for e, n in itertools.product((-1, 1), repeat=2):
            donor = indexed[row["bank"], row["condition"], row["order"], e, n]
            if len(row["input_ids"]) != len(donor["input_ids"]):
                raise ValueError("Native carrier widths differ")
            if Counter(row["input_ids"]) != Counter(donor["input_ids"]):
                raise ValueError("Natural assignment changed complete token bag")
            if scopes[str(row["id"])] != scopes[str(donor["id"])]:
                raise ValueError("Actor assignment changed carrier addresses")
            tail = scopes[str(row["id"])]["tail"]
            if any(
                row["input_ids"][row["history"][s]]
                != donor["input_ids"][donor["history"][s]]
                for s in tail
            ):
                raise ValueError("Ending must be literally identical")
    design = [
        {"batch": i, "row": row, "mode": "self"} for i, row in enumerate((0, 3, 16, 31))
    ]
    for mode in MODES:
        for row in rows:
            design.append({"batch": len(design), "row": row["id"], "mode": mode})
    if len(design) != 260:
        raise ValueError("Finite intervention count changed")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("captures", "accounts"):
        (out / name).mkdir()
    for name, value in (
        ("rows", rows),
        ("queries", queries),
        ("design", design),
        ("scopes", scopes),
    ):
        write_json(out / f"{name}.json", value)
    dependencies = dict(manifest["hashes"])
    for path in CODE:
        dependencies[path] = H.digest(ROOT / path)
    for path in (
        "manifest.json",
        "capture_seal.json",
        "output_seal.json",
        "rows.json",
        "queries.json",
        "design.json",
        "serialization_recovery_r1/summary.json",
    ):
        relative = str((parent / path).relative_to(ROOT))
        dependencies[relative] = H.digest(parent / path)
    write_json(
        out / "manifest.json",
        {
            "packet": "HM156",
            "parent": str(parent.relative_to(ROOT)),
            "hashes": dependencies,
            "inputs": {
                f"{name}.json": H.digest(out / f"{name}.json")
                for name in ("rows", "queries", "design", "scopes")
            },
            "model_weights_sha256": manifest["model_weights_sha256"],
            "batches": 260,
            "inquiry_sequences": 3120,
            "collection_seconds_cap": 900,
            "analysis_seconds_cap": 300,
            "audit_seconds_cap": 120,
            "prefix_atol": 1e-4,
            "selected_atol": 1e-4,
            "selected_rtol": 1e-5,
            "original_scope_complete": False,
            "additive_dtype": "compute float64 then cast once to native float32",
            "estimand": "fixed-reference suffix reader response under historical memory interventions",
        },
    )
    R.status(out, "PREPARED", batches_done=0, pending=None)
    print({"prepared": 32, "batches": 260, "inquiries": 3120}, flush=True)


def donor_arguments(row, mode, natural, indexed):
    """Construct actual arguments; additions are explicitly off-natural interventions."""
    if mode not in (*MODES, "self"):
        raise ValueError("Unknown prospective carrier operation")
    addr = (row["bank"], row["condition"], row["order"])
    base = natural[row["id"]]
    e = natural[indexed[*addr, -row["e"], row["n"]]["id"]]
    n = natural[indexed[*addr, row["e"], -row["n"]]["id"]]
    both = natural[indexed[*addr, -row["e"], -row["n"]]["id"]]
    scopes = M.carrier_scopes(row)
    slots = (
        list(range(len(row["history"])))
        if mode == "self"
        else scopes[mode]
        if mode in scopes
        else scopes["tail"]
    )
    source = (
        base
        if mode == "self"
        else e
        if mode.startswith("e_")
        else n
        if mode.startswith("n_")
        else both
    )
    result, rounding = [], 0.0
    for name in ("history_key", "history_value"):
        if mode == "both_tail_additive":
            ideal = M.additive_memory(base[name], e[name], n[name])
            source_memory = ideal.astype(base[name].dtype)
            rounding = max(
                rounding,
                float(
                    np.max(
                        abs(
                            source_memory[:, :, slots].astype(float)
                            - ideal[:, :, slots]
                        )
                    )
                ),
            )
        else:
            source_memory = source[name]
        result.append(M.replace_slots(base[name], source_memory, slots))
    return tuple(result), slots, rounding


def collect(out):
    """Run the finite prospective support once, retaining partials on failure."""
    manifest = check(out)
    if R.read(out / "collection_status.json")["status"] != "PREPARED":
        raise RuntimeError("No automatic replay of a started experiment")
    start, done, pending = time.monotonic(), 0, None
    parent = ROOT / manifest["parent"]
    authenticate_parent(parent)
    rows, queries, design = (
        R.read(out / f"{name}.json") for name in ("rows", "queries", "design")
    )
    natural = parent_natural(parent)
    indexed = {(r["bank"], r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    try:
        R.status(out, "LOADING", pid=os.getpid(), batches_done=0, pending=None)
        observer = R.J.offline_model(manifest)
        write_json(out / "model_manifest.json", observer.manifest())
        for entry in design:
            if time.monotonic() - start > manifest["collection_seconds_cap"]:
                raise TimeoutError("Finite900second collection cap; no renewal")
            pending, row = entry, rows[entry["row"]]
            R.status(
                out,
                "RUNNING",
                pid=os.getpid(),
                batches_done=done,
                pending=pending,
                seconds=time.monotonic() - start,
            )
            begin = time.monotonic()
            donors, slots, rounding = donor_arguments(
                row, entry["mode"], natural, indexed
            )
            result = R.native_batch(observer, row, queries, donors)
            base = natural[row["id"]]
            error = float(np.max(abs(result["prefix_states"] - base["prefix_states"])))
            if (
                error > manifest["prefix_atol"]
                or float(result["cache_spread"]) > manifest["prefix_atol"]
            ):
                raise ValueError("Intervention changed original prefix")
            if entry["mode"] == "self" and not np.allclose(
                result["selected_logits"], base["selected_logits"], atol=1e-4, rtol=1e-5
            ):
                raise ValueError("Own-donor native control failed")
            result.update(
                prefix_error=np.asarray(error),
                patched_slots=np.asarray(slots),
                donor_key_slice=donors[0][:, :, slots],
                donor_value_slice=donors[1][:, :, slots],
                rounding_error=np.asarray(rounding),
            )
            stem = out / "captures" / f"{entry['batch']:03d}"
            H.write_npz(stem.with_suffix(".npz"), **result)
            write_json(
                stem.with_suffix(".json"),
                {
                    **entry,
                    "seconds": time.monotonic() - begin,
                    "sha256": H.digest(stem.with_suffix(".npz")),
                    "donor_key_sha256": array_digest(donors[0]),
                    "donor_value_sha256": array_digest(donors[1]),
                },
            )
            done, pending = done + 1, None
            if done % 16 == 0:
                print(
                    {"HM156_batches": done, "seconds": time.monotonic() - start},
                    flush=True,
                )
        if time.monotonic() - start > manifest["collection_seconds_cap"]:
            raise TimeoutError("Finite budget crossed on final batch")
        write_json(
            out / "capture_seal.json",
            {
                str(p.relative_to(out)): H.digest(p)
                for p in sorted((out / "captures").glob("*"))
                if not p.name.startswith("._")
            },
        )
        R.status(
            out,
            "COLLECTED",
            batches_done=done,
            pending=None,
            seconds=time.monotonic() - start,
            model_batches=observer.calls,
            inquiry_sequences=12 * done,
        )
    except BaseException as exc:
        R.status(
            out,
            "FAILED_OR_BUDGET_STOP",
            pid=os.getpid(),
            batches_done=done,
            pending=pending,
            seconds=time.monotonic() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def observations(out):
    """Read actual scalar cubes including frozen parent natural endpoints."""
    manifest = check(out)
    for path, checksum in R.read(out / "capture_seal.json").items():
        if H.digest(out / path) != checksum:
            raise ValueError("Capture changed: " + path)
    rows, design = R.read(out / "rows.json"), R.read(out / "design.json")
    natural = parent_natural(ROOT / manifest["parent"])
    captured = {
        (e["mode"], e["row"]): R.arrays(out / "captures" / f"{e['batch']:03d}.npz")
        for e in design
    }
    cubes = {}
    for group in R.GROUPS:
        for mode in ("natural", *MODES):
            cube = np.empty((2, 2, 2, 2, 3))
            for row in rows:
                if (row["bank"], row["condition"], row["order"]) != group:
                    continue
                a = (
                    natural[row["id"]]
                    if mode == "natural"
                    else captured[mode, row["id"]]
                )
                logits = a["selected_logits"].astype(float)
                cube[(row["e"] + 1) // 2, (row["n"] + 1) // 2] = (
                    (logits[:, 0] - logits[:, 1]).reshape(2, 3, 2).transpose(2, 0, 1)
                )
            cubes[group, mode] = cube
    return cubes, captured, natural


def analyze(out):
    """Account complete qualified characters, tail composition and native controls."""
    start, manifest = time.monotonic(), check(out)
    if (
        R.read(out / "collection_status.json")["status"] != "COLLECTED"
        or (out / "summary.json").exists()
    ):
        raise RuntimeError("One full-support analysis only; never overwrite")
    cubes, captured, natural = observations(out)
    rows = R.read(out / "rows.json")
    actions, coefficients, interactions, latent = [], [], [], []
    max_identity = 0.0
    for gi, group in enumerate(R.GROUPS):
        if time.monotonic() - start > manifest["analysis_seconds_cap"]:
            raise TimeoutError("Finite300second analysis cap")
        account = {
            mode + "_logodds": cube for (g, mode), cube in cubes.items() if g == group
        }
        for mode in ("natural", *MODES):
            c = R.M.characters(cubes[group, mode])
            account[mode + "_characters"] = c
            for mask, axis, family in itertools.product(range(8), range(2), range(3)):
                coefficients.append(
                    {
                        "group": gi,
                        "bank": group[0],
                        "condition": group[1],
                        "order": group[2],
                        "mode": mode,
                        "axis": R.M.AXES[axis],
                        "family": R.M.FAMILIES[family],
                        "mask": mask,
                        "logodds": c[mask, axis, family],
                    }
                )
        for axis in range(2):
            base = cubes[group, "natural"][..., axis, :]
            for mode in MODES:
                reference = (
                    base[::-1]
                    if mode.startswith("e_")
                    else base[:, ::-1]
                    if mode.startswith("n_")
                    else base[::-1, ::-1]
                )
                actions.append(
                    {
                        "group": gi,
                        "bank": group[0],
                        "condition": group[1],
                        "order": group[2],
                        "axis": R.M.AXES[axis],
                        "mode": mode,
                        **R.M.action_account(
                            cubes[group, mode][..., axis, :] - base, reference - base
                        ),
                    }
                )
            parts = M.interaction_account(
                base,
                *(
                    cubes[group, mode][..., axis, :]
                    for mode in (
                        "e_tail",
                        "n_tail",
                        "both_tail_real",
                        "both_tail_additive",
                    )
                ),
            )
            err = float(
                np.max(abs(parts["total"] - parts["encoding"] - parts["reader"]))
            )
            max_identity = max(max_identity, err)
            interactions.append(
                {
                    "group": gi,
                    "bank": group[0],
                    "condition": group[1],
                    "order": group[2],
                    "axis": R.M.AXES[axis],
                    **{
                        name + "_rms": float(np.sqrt(np.mean(v * v)))
                        for name, v in parts.items()
                    },
                    "identity_error": err,
                    "encoding_reader_inner": float(
                        np.mean(parts["encoding"] * parts["reader"])
                    ),
                }
            )
            for name, v in parts.items():
                account[f"{name}_{axis}"] = v
        grouped = [
            row
            for row in rows
            if (row["bank"], row["condition"], row["order"]) == group
        ]
        for name in ("history_key", "history_value"):
            tail = M.carrier_scopes(grouped[0])["tail"]
            x = np.empty((2, 2, *natural[grouped[0]["id"]][name][:, :, tail].shape))
            for row in grouped:
                x[(row["e"] + 1) // 2, (row["n"] + 1) // 2] = natural[row["id"]][name][
                    :, :, tail
                ]
            c = M.memory_characters(x)
            account[name + "_tail_characters"] = c
            for mask in range(4):
                latent.append(
                    {
                        "group": gi,
                        "bank": group[0],
                        "condition": group[1],
                        "order": group[2],
                        "carrier": name,
                        "mask": mask,
                        "rms": float(np.sqrt(np.mean(c[mask] ** 2))),
                    }
                )
        H.write_npz(out / "accounts" / f"{gi:02d}.npz", **account)
    for name, table in (
        ("characters", coefficients),
        ("actions", actions),
        ("interactions", interactions),
        ("latent_characters", latent),
    ):
        H.write_csv(out / f"{name}.csv", M.json_native(table))
    mode_ranges = {}
    for mode in MODES:
        selected = [a for a in actions if a["mode"] == mode]
        mode_ranges[mode] = {
            metric: [
                min(a[metric] for a in selected if a[metric] is not None),
                max(a[metric] for a in selected if a[metric] is not None),
            ]
            for metric in ("rms", "signed_projection", "relative_remainder")
        }
    natural_c = [
        c
        for c in coefficients
        if c["mode"] == "natural"
        and c["mask"] == (5 if c["axis"] == R.M.AXES[0] else 6)
    ]
    qualification = {}
    for mode in MODES:
        own = [
            c
            for c in coefficients
            if c["mode"] == mode and c["mask"] == (5 if c["axis"] == R.M.AXES[0] else 6)
        ]
        qualification[mode] = {
            "positive": sum(c["logodds"] > 0 for c in own),
            "negative": sum(c["logodds"] < 0 for c in own),
            "total": len(own),
            "mean": np.mean([c["logodds"] for c in own]),
        }
    write_json(
        out / "summary.json",
        {
            "status": "FINITE_CARRIER_ACCOUNT_COMPLETE_NOT_GLOBAL",
            "batches": 260,
            "inquiry_sequences": 3120,
            "collection": R.read(out / "collection_status.json"),
            "mode_ranges": mode_ranges,
            "qualification": qualification,
            "natural_qualified_positive": sum(c["logodds"] > 0 for c in natural_c),
            "interactions": interactions,
            "maximum_interaction_identity_error": max_identity,
            "max_prefix_error": max(
                float(a["prefix_error"]) for a in captured.values()
            ),
            "self_selected_error": max(
                float(
                    np.max(abs(a["selected_logits"] - natural[row]["selected_logits"]))
                )
                for (mode, row), a in captured.items()
                if mode == "self"
            ),
            "maximum_additive_rounding_error": max(
                float(a["rounding_error"]) for a in captured.values()
            ),
            "seconds": time.monotonic() - start,
            "original_scope_complete": False,
            "scope": "fictional actor-qualified evidence, not human trait accuracy or a model-free scorer",
        },
    )
    names = [
        "manifest.json",
        "capture_seal.json",
        "summary.json",
        "characters.csv",
        "actions.csv",
        "interactions.csv",
        "latent_characters.csv",
    ]
    names += [
        str(p.relative_to(out))
        for p in sorted((out / "accounts").glob("*.npz"))
        if not p.name.startswith("._")
    ]
    write_json(out / "output_seal.json", {name: H.digest(out / name) for name in names})
    print(
        {
            "analyzed": len(actions),
            "characters": len(coefficients),
            "interactions": len(interactions),
            "identity_error": max_identity,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--parent", type=Path, default=R.DEFAULT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out, args.parent)
    else:
        {"collect": collect, "analyze": analyze}[args.stage](args.out)
