#!/usr/bin/env python3
"""HM30 acquire a fully crossed wording/context field with source-only fits."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_crossed_field as C
from grain_research.construction import math_position_support as P
from grain_research.construction import math_source_field as S
from grain_research.construction import math_span_binding as B
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_position_support import (
    array_unit,
    baseline_path,
    capture,
    check,
    load_observer,
    now,
    patch_path,
    patched,
    recover_in_flight,
    save_array_unit,
)
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_narrow_functions import reference_view
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/crossed_wording_context_r1"
SPEC = ROOT / "configs/grain_math_crossed_field_v1.json"
PLAN = ROOT / "docs/GRAIN_CROSSED_WORDING_CONTEXT_FIELD_PLAN_V1.md"
CODE = (
    "grain_research/construction/math_crossed_field.py",
    "scripts/run_grain_math_crossed_field.py",
    "scripts/verify_grain_math_crossed_field.py",
    "tests/test_grain_math_crossed_field.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def prepare(out):
    """Freeze32 new cells and all exact token addresses before native outputs."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("preserve previous packet; resume existing stages")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    inherited = check(parent)
    verification = ROOT / spec["parent_verification"] / "summary.json"
    if read_json(verification)["status"] != "VERIFIED_FINITE_HM29_NOT_GLOBAL":
        raise ValueError("verified HM29 required")
    files = dict(inherited["files"])
    for name, sha in read_json(parent / "02/artifact_hashes.json").items():
        if digest(parent / name) != sha:
            raise ValueError("HM29 frozen output changed " + name)
        files[str((parent / name).relative_to(ROOT))] = sha
    files[str(verification.relative_to(ROOT))] = digest(verification)
    files.update({name: digest(ROOT / name) for name in CODE})
    definitions = read_json(parent / "00/queries.json")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    prior_texts = {d["text"] for d in read_json(parent / "00/documents.json")}
    docs, design = [], []
    for context in range(spec["contexts"]):
        role = (
            "source" if context < 4 else "validation" if context < 6 else "application"
        )
        for wording in range(spec["wordings"]):
            text = C.material(context, wording)
            if text in prior_texts:
                raise ValueError(
                    "duplicate HM29 text; explicit inherited design needed"
                )
            docs.append(
                {"context": context, "wording": wording, "role": role, "text": text}
            )
            previous = None
            for query in spec["queries"]:
                rendered, bodies, encoded, mask = reference_view(
                    tokenizer, [text], definitions[query]["question"]
                )
                qstart = int(np.flatnonzero(mask)[0])
                spans = {
                    name: B.token_span(rendered, bodies[0][0], raw, encoded, qstart)
                    for name, raw in B.raw_spans(text).items()
                }
                if len(spans["A_property"]["positions"]) != 4:
                    raise ValueError("declared wording is not exactly four tokens")
                current = encoded["input_ids"][:qstart], spans
                if previous is not None and current != previous:
                    raise ValueError("query-dependent prefix or property address")
                previous = current
                design.append(
                    {
                        "unit": len(design),
                        "context": context,
                        "wording": wording,
                        "role": role,
                        "query": query,
                        "input_ids": encoded["input_ids"],
                        "question_start": qstart,
                        "spans": spans,
                    }
                )
    if len(design) != spec["units"] or list(C.ARMS) != spec["arms"]:
        raise ValueError("crossed design changed")
    for name in ("00/code", "01/baselines", "01/patches", "01/source", "02"):
        (out / name).mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("design", design),
        ("documents", docs),
        ("queries", definitions),
    ):
        write_json(out / f"00/{name}.json", value)
    with np.load(parent / "00/directions.npz", allow_pickle=False) as bank:
        np.savez_compressed(
            out / "00/directions.npz",
            **{q + "_b": bank[q + "_b"] for q in spec["queries"]},
        )
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "inputs": {
                name: digest(out / name)
                for name in (
                    "00/design.json",
                    "00/documents.json",
                    "00/queries.json",
                    "00/directions.npz",
                )
            },
            "spec": spec,
            "alphabet": inherited["alphabet"],
            "frozen_before_outputs": now(),
            "new_cells": len(docs),
            "same_wordings_across_context_splits": True,
            "no_duplicate_parent_texts": True,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "backward_calls": 0,
            "in_flight": None,
            "model_seconds": 0.0,
        },
    )
    print(
        {"status": "PREPARED", "cells": len(docs), "references": len(design)},
        flush=True,
    )


def fields(out):
    """Load only frozen source fields and verify each source input dependency."""
    source = read_json(out / "01/source_manifest.json")
    result = {}
    for key, entry in source.items():
        for name, sha in {
            entry["file"]: entry["sha256"],
            **entry["source_hashes"],
        }.items():
            if digest(out / name) != sha:
                raise ValueError("source field or gradient changed " + name)
        with np.load(out / entry["file"], allow_pickle=False) as data:
            result[key] = data["covector"].copy()
    return result


def calls(out, source):
    """Collect exactly two arms, saving each call within one cumulative time budget."""
    m = check(out)
    spec = m["spec"]
    status = recover_in_flight(out, read_json(out / "run_status.json"))
    spath = out / "01/source_manifest.json"
    if source == spath.exists():
        raise ValueError("source calls precede fit; held calls require frozen fields")
    source_sha = None if source else digest(spath)
    if not source:
        fields(out)
    all_rows = read_json(out / "00/design.json")
    rows = [r for r in all_rows if (r["role"] == "source") == source]
    completed = sum(baseline_path(out, r["unit"]).exists() for r in all_rows)
    endpoints = sum(
        patch_path(out, r["unit"], a, s).exists()
        for r in all_rows
        for a in spec["arms"]
        for s in ("plus", "minus")
    )
    if (status["forward_calls"], status["backward_calls"]) != (
        completed + endpoints,
        completed,
    ):
        raise ValueError("call ledger differs from committed artifacts")
    if all(
        baseline_path(out, r["unit"]).exists()
        and all(
            patch_path(out, r["unit"], a, s).exists()
            for a in spec["arms"]
            for s in ("plus", "minus")
        )
        for r in rows
    ):
        print("ALREADY_ACQUIRED", flush=True)
        return
    if status["model_seconds"] >= spec["maximum_model_seconds"]:
        raise ValueError("cumulative model-time boundary reached")
    started, elapsed = time.monotonic(), status["model_seconds"]
    observer = load_observer()
    if any(p.requires_grad for p in observer.model.parameters()):
        raise ValueError("neural weights must remain fixed")
    write_json(
        out / ("01/source_observer.json" if source else "01/held_observer.json"),
        observer.manifest(),
    )
    with np.load(out / "00/directions.npz", allow_pickle=False) as bank:
        directions = {q: C.directions(bank[q + "_b"]) for q in spec["queries"]}

    def begin(path, backwards):
        nonlocal status
        seconds = elapsed + time.monotonic() - started
        if (
            status["forward_calls"] >= spec["maximum_forwards"]
            or status["backward_calls"] + backwards > spec["maximum_backwards"]
            or seconds >= spec["maximum_model_seconds"]
        ):
            write_json(
                out / "run_status.json",
                {**status, "status": "RESOURCE_BOUNDARY", "model_seconds": seconds},
            )
            return False
        status = {
            **status,
            "status": "SOURCE_CALLS" if source else "HELD_CALLS",
            "model_seconds": seconds,
            "in_flight": {
                "file": str(path.relative_to(out)),
                "backwards": backwards,
                "started_at": now(),
            },
        }
        write_json(out / "run_status.json", status)
        return True

    def finish(backwards):
        nonlocal status
        status = {
            **status,
            "in_flight": None,
            "forward_calls": status["forward_calls"] + 1,
            "backward_calls": status["backward_calls"] + backwards,
            "model_seconds": elapsed + time.monotonic() - started,
        }
        write_json(out / "run_status.json", status)

    try:
        for row in rows:
            path = baseline_path(out, row["unit"])
            if not path.exists():
                if not begin(path, 1):
                    return
                arrays, native, grad_score = capture(observer, row, m)
                save_array_unit(
                    path,
                    {
                        "address": row,
                        "native": native,
                        "gradient_readout": grad_score,
                        "acquired_at": now(),
                        "source_sha256": source_sha,
                    },
                    arrays,
                )
                finish(1)
            meta = array_unit(path)
            if meta["address"] != row or meta["source_sha256"] != source_sha:
                raise ValueError("reference address/source changed")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                h = data["h"].copy()
            for arm in spec["arms"]:
                physical = P.physical_patches(
                    h, directions[row["query"]][arm], spec["radius_relative_span_norm"]
                )
                for sign in ("plus", "minus"):
                    path = patch_path(out, row["unit"], arm, sign)
                    address = {"unit": row["unit"], "arm": arm, "sign": sign}
                    if path.exists():
                        end = read_json(path)
                        if (
                            end["address"] != address
                            or end["source_sha256"] != source_sha
                        ):
                            raise ValueError("endpoint address/source changed")
                        continue
                    if not begin(path, 0):
                        return
                    native, audit = patched(observer, row, m, h, physical[sign])
                    write_json(
                        path,
                        {
                            "address": address,
                            "native": native,
                            "audit": audit,
                            "baseline_sha256": digest(baseline_path(out, row["unit"])),
                            "source_sha256": source_sha,
                            "acquired_at": now(),
                        },
                    )
                    finish(0)
            print(
                {
                    "unit": row["unit"],
                    "context": row["context"],
                    "wording": row["wording"],
                    "role": row["role"],
                    "forwards": status["forward_calls"],
                    "backwards": status["backward_calls"],
                },
                flush=True,
            )
        write_json(
            out / "run_status.json",
            {**status, "status": "SOURCE_ACQUIRED" if source else "HELD_ACQUIRED"},
        )
    except Exception as exc:
        write_json(
            out / "run_status.json", {**status, "status": "ERROR", "error": repr(exc)}
        )
        raise


def freeze_source(out):
    """Freeze15 equal-source-context fields before any held native observation."""
    m = check(out)
    rows = read_json(out / "00/design.json")
    if (out / "01/source_manifest.json").exists() or any(
        baseline_path(out, r["unit"]).exists() for r in rows if r["role"] != "source"
    ):
        raise ValueError("source fit must precede held observations")
    if read_json(out / "run_status.json")["status"] != "SOURCE_ACQUIRED":
        raise ValueError("complete source endpoints required")
    source = {}
    for q in m["spec"]["queries"]:
        for w in (-1, 0, 1, 2, 3):
            selected = [
                r
                for r in rows
                if r["role"] == "source"
                and r["query"] == q
                and (w == -1 or r["wording"] == w)
            ]
            gradients, hashes = [], {}
            for r in selected:
                p = baseline_path(out, r["unit"])
                array_unit(p)
                hashes.update(
                    {
                        str(f.relative_to(out)): digest(f)
                        for f in (p, p.with_suffix(".npz"))
                    }
                )
                if not all(
                    patch_path(out, r["unit"], a, s).exists()
                    for a in m["spec"]["arms"]
                    for s in ("plus", "minus")
                ):
                    raise ValueError("source endpoint control missing")
                with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
                    gradients.append(data["gradient"].astype(float))
            field = S.common_covector(gradients)
            if len(gradients) != (16 if w == -1 else 4) or field["rank"] != 4:
                raise ValueError("complete source crossing required")
            key = q + ("_global" if w == -1 else f"_wording{w}")
            path = out / "01/source" / (key + ".npz")
            np.savez_compressed(path, **field)
            source[key] = {
                "file": str(path.relative_to(out)),
                "sha256": digest(path),
                "source_hashes": hashes,
                "source_units": [r["unit"] for r in selected],
                "rank": field["rank"],
                "frozen_at": now(),
            }
    write_json(out / "01/source_manifest.json", source)
    write_json(
        out / "run_status.json",
        {**read_json(out / "run_status.json"), "status": "SOURCE_FROZEN"},
    )
    print({"status": "SOURCE_FROZEN", "fields": len(source)}, flush=True)


def evaluate(out):
    """Analyze source-frozen finite actions and balanced field/action projections."""
    m = check(out)
    spec = m["spec"]
    if read_json(out / "run_status.json")["status"] != "HELD_ACQUIRED":
        raise ValueError("complete actual acquisition required")
    fitted = fields(out)
    actions, endpoints, gradients = [], [], {}
    design = read_json(out / "00/design.json")
    with np.load(out / "00/directions.npz", allow_pickle=False) as bank:
        for r in design:
            p = baseline_path(out, r["unit"])
            base = array_unit(p)
            with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
                h, g = data["h"].copy(), data["gradient"].astype(float)
            gradients[r["unit"]] = g
            for arm, direction in C.directions(bank[r["query"] + "_b"]).items():
                physical = P.physical_patches(
                    h, direction, spec["radius_relative_span_norm"]
                )
                ends = {
                    s: read_json(patch_path(out, r["unit"], arm, s))
                    for s in ("plus", "minus")
                }
                effect = P.contrast(
                    base["native"]["conditional_score"],
                    ends["plus"]["native"]["conditional_score"],
                    ends["minus"]["native"]["conditional_score"],
                )
                moments = S.moment_map(4) @ physical["half_delta"]
                actions.append(
                    {k: r[k] for k in ("unit", "context", "wording", "query", "role")}
                    | {
                        "arm": arm,
                        **effect,
                        "physical_delta_norm": float(
                            np.linalg.norm(physical["half_delta"])
                        ),
                        "global_source": S.contract(
                            fitted[r["query"] + "_global"], moments
                        ),
                        "wording_source": S.contract(
                            fitted[r["query"] + f"_wording{r['wording']}"], moments
                        ),
                        "local_derivative": float(np.sum(g * physical["half_delta"])),
                    }
                )
                for sign, end in ends.items():
                    if any(end["audit"].values()) or end["baseline_sha256"] != digest(
                        p
                    ):
                        raise ValueError("endpoint reference/complement audit failed")
                    endpoints.append(
                        {"unit": r["unit"], "arm": arm, "sign": sign, **end["native"]}
                    )
    metrics, field_decomp, action_decomp, covariance, arrays = [], [], [], [], {}
    for role in ("source", "validation", "application"):
        contexts = sorted({r["context"] for r in design if r["role"] == role})
        for q in spec["queries"]:
            reference = {
                (r["context"], r["wording"]): r
                for r in design
                if r["role"] == role and r["query"] == q
            }
            g = np.array(
                [
                    [gradients[reference[c, w]["unit"]] for w in range(4)]
                    for c in contexts
                ]
            )
            lam = np.einsum("ij,cpjd->cpid", np.linalg.inv(S.moment_map(4).T), g)
            for name, x in (("gradient", g), ("covector", lam)):
                key = role + "_" + q + "_" + name
                arrays[key] = x
                parts, summary = C.decompose(x)
                arrays.update({key + "_" + k: v for k, v in parts.items()})
                field_decomp.append(
                    {
                        "role": role,
                        "query": q,
                        "space": name,
                        "contexts": len(contexts),
                        "wordings": 4,
                        **summary,
                    }
                )
            for arm in spec["arms"]:
                items = [
                    r
                    for r in actions
                    if (r["role"], r["query"], r["arm"]) == (role, q, arm)
                ]
                metrics.append(
                    {"role": role, "query": q, "arm": arm, **C.summarize(items)}
                )
                lookup = {(r["context"], r["wording"]): r for r in items}
                for units in ("raw", "unit_displacement"):
                    x = np.array(
                        [
                            [
                                lookup[c, w]["odd_action"]
                                / (
                                    1
                                    if units == "raw"
                                    else lookup[c, w]["physical_delta_norm"]
                                )
                                for w in range(4)
                            ]
                            for c in contexts
                        ]
                    )
                    arrays[role + "_" + q + "_" + arm + "_" + units] = x
                    _, summary = C.decompose(x)
                    action_decomp.append(
                        {
                            "role": role,
                            "query": q,
                            "arm": arm,
                            "units": units,
                            **summary,
                        }
                    )
        for q, z in itertools.combinations(spec["queries"], 2):
            for arm in spec["arms"]:
                for units in ("raw", "unit_displacement"):
                    covariance.append(
                        {
                            "role": role,
                            "left": q,
                            "right": z,
                            "arm": arm,
                            "units": units,
                            **C.covariance_parts(
                                arrays[role + "_" + q + "_" + arm + "_" + units],
                                arrays[role + "_" + z + "_" + arm + "_" + units],
                            ),
                        }
                    )
    for name, rows in (
        ("actions", actions),
        ("endpoints", endpoints),
        ("metrics", metrics),
        ("field_decomposition", field_decomp),
        ("action_decomposition", action_decomp),
        ("paired_query_covariance", covariance),
    ):
        write_csv(out / f"02/{name}.csv", rows)
    np.savez_compressed(out / "02/crossed_arrays.npz", **arrays)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM30_ACQUIRED_NOT_GLOBAL",
            "counts": {
                "references": len(design),
                "actions": len(actions),
                "endpoints": len(endpoints),
                "metrics": len(metrics),
                "source_fields": len(fitted),
                "field_decompositions": len(field_decomp),
                "action_decompositions": len(action_decomp),
                "paired_covariances": len(covariance),
            },
            "calls": read_json(out / "run_status.json"),
            "scope": spec["scope"],
        },
    )
    paths = [
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.endswith(".tmp")
        and p.name not in ("run_status.json", "artifact_hashes.json")
    ]
    write_json(
        out / "02/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    write_json(
        out / "run_status.json",
        {
            **read_json(out / "run_status.json"),
            "status": "FINITE_HM30_ACQUIRED_NOT_GLOBAL",
        },
    )
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Dispatch only one explicit finite stage, protected against duplicate workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "source_calls",
            "freeze_source",
            "held_calls",
            "evaluate",
            "check",
        ),
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        if args.stage in ("source_calls", "held_calls"):
            calls(args.out, args.stage == "source_calls")
        elif args.stage == "check":
            print({"frozen_files": len(check(args.out)["files"])})
        else:
            globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
