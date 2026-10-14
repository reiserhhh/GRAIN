#!/usr/bin/env python3
"""HM47: frozen outside-appraisal interventions, not a prediction-accuracy search."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_construct_routing as M
from grain_research.construction import math_evaluative_routing as E
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import CODE as PARENT_CODE
from scripts.run_grain_math_construct_routing import check as parent_check
from scripts.run_grain_math_construct_routing import observe, save, write
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_evaluative_routing_v1.json"
OUT = ROOT / "results/grain_math_response_v1/evaluative_routing_r2"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_evaluative_routing_v1.json",
            "docs/GRAIN_EVALUATIVE_CONTEXT_ROUTING_PLAN_V1.md",
            "grain_research/construction/math_evaluative_routing.py",
            "scripts/run_grain_math_evaluative_routing.py",
            "scripts/verify_grain_math_evaluative_routing.py",
            "tests/test_grain_math_evaluative_routing.py",
            "scripts/run_grain_math_response.py",
            "scripts/run_grain_upstream_routes.py",
            *PARENT_CODE,
        )
    )
)


def prepare(out):
    """Verify exact cached prompts, freeze all new inputs and record their provenance."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve existing packets; preparation is not a reset")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    previous = parent_check(parent)
    verification = read_json(parent / "verification.json")
    if verification["status"] != "VERIFIED_FINITE_HM46_NOT_GLOBAL":
        raise ValueError("Require actually verified HM46")
    old_rows = read_json(parent / "00/design.json")
    lookup = {(r["doc_id"], r["query"]): r for r in old_rows}
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    alphabet = previous["alphabet"]
    if [tokenizer.encode(x, add_special_tokens=False) for x in alphabet["labels"]] != [
        [i] for i in alphabet["token_ids"]
    ] or spec["layer"] != previous["spec"]["layer"]:
        raise ValueError("Changed alphabet or carrier site")
    rows, parent_hashes = [], {}
    for doc in E.documents(previous["spec"], spec):
        for query in previous["spec"]["targets"]:
            rendered, encoded, offset = M.render(doc["body"], query, tokenizer)
            start, stop = doc["property_interval"]
            positions = M.token_positions(
                encoded["offset_mapping"], offset + start, offset + stop
            )
            body_positions = M.token_positions(
                encoded["offset_mapping"], offset, offset + len(doc["body"])
            )
            old = lookup[doc["hm46_doc_id"], query]
            if (
                encoded["input_ids"][: positions[-1] + 1]
                != old["input_ids"][: old["property_positions"][-1] + 1]
            ):
                raise ValueError(
                    "Future appraisal changed the causal property token prefix"
                )
            row = {
                **doc,
                "unit": len(rows),
                "query": query,
                "input_ids": list(encoded["input_ids"]),
                "rendered": rendered,
                "property_positions": positions,
                "property_tokens": [encoded["input_ids"][i] for i in positions],
                "body_endpoint": body_positions[-1],
                "question_start": M.token_positions(
                    encoded["offset_mapping"],
                    rendered.index("\nQuestion: ") + 1,
                    len(rendered),
                )[0],
                "reused": doc["evaluation"] == "none",
                "hm46_unit": old["unit"],
            }
            if row["reused"]:
                for field in (
                    "input_ids",
                    "rendered",
                    "property_positions",
                    "property_tokens",
                    "body_endpoint",
                    "question_start",
                ):
                    if row[field] != old[field]:
                        raise ValueError("Not an identical cached HM46 input: " + field)
                meta_path = parent / f"01/units/unit_{old['unit']:04d}.json"
                meta = read_json(meta_path)
                if digest(meta_path.with_suffix(".npz")) != meta["array_sha256"]:
                    raise ValueError("Changed cached carrier arrays")
                for path in (meta_path, meta_path.with_suffix(".npz")):
                    parent_hashes[str(path.relative_to(parent))] = digest(path)
            rows.append(row)
    reused = sum(r["reused"] for r in rows)
    if (
        len(rows) != spec["expected_scores"]
        or reused != spec["expected_reused"]
        or len(rows) - reused != spec["max_fresh_forwards"]
    ):
        raise ValueError("Full planned score/provenance scope required")
    for directory in ("00/code", "01/units", "03"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    write(out / "00/design.json", rows)
    code = {}
    for name in CODE:
        target = out / "00/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(target)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM47",
            "spec": spec,
            "targets": previous["spec"]["targets"],
            "alphabet": alphabet,
            "code": code,
            "design_sha256": digest(out / "00/design.json"),
            "parent_manifest_sha256": digest(parent / "00/manifest.json"),
            "parent_verification_sha256": digest(parent / "verification.json"),
            "parent_design_sha256": digest(parent / "00/design.json"),
            "parent_hashes": parent_hashes,
            "parent_observer": read_json(parent / "01/observer.json"),
            "model_config_sha256": digest(MODEL / "config.json"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_scores": len(rows),
            "reused_scores": reused,
            "fresh_scores": len(rows) - reused,
            "no_training_or_api": True,
            "human_personality_truth": False,
            "parent_weights_historically_not_separately_hashed": True,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "attempted_forward_calls": 0,
            "completed_units": 0,
            "reused_units": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        f"PREPARED:{len(rows)} scores/{reused} reused/{len(rows) - reused} fresh",
        flush=True,
    )


def check(out):
    """Check this packet and its exact reused ancestry, not unrelated historical files."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Changed frozen code " + name)
    if (
        digest(out / "00/design.json") != manifest["design_sha256"]
        or digest(MODEL / "config.json") != manifest["model_config_sha256"]
    ):
        raise ValueError("Changed material or local observer config")
    parent = ROOT / manifest["spec"]["parent"]
    for file, key in (
        ("00/manifest.json", "parent_manifest_sha256"),
        ("00/design.json", "parent_design_sha256"),
        ("verification.json", "parent_verification_sha256"),
    ):
        if digest(parent / file) != manifest[key]:
            raise ValueError("Changed parent identity " + file)
    for file, sha in manifest["parent_hashes"].items():
        if digest(parent / file) != sha:
            raise ValueError("Changed reused parent unit " + file)
    return manifest


def load_arrays(path):
    """Read a hash-bound completed carrier capture without a model call."""
    meta = read_json(path)
    if digest(path.with_suffix(".npz")) != meta["array_sha256"]:
        raise ValueError("Changed actual capture")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as values:
        arrays = {k: values[k].copy() for k in values.files}
    return meta, arrays


def collect(out):
    """Seal source before application; record fresh/reused units and uncertain calls."""
    manifest = check(out)
    spec = manifest["spec"]
    parent = ROOT / spec["parent"]
    status = read_json(out / "run_status.json")
    if status["in_flight"] is not None:
        raise RuntimeError("Inspect uncertain call; do not automatically repeat it")
    if status["status"] == "COLLECTED":
        return
    started, previous = time.monotonic(), status["model_seconds"]
    observer = load_observer()
    actual_observer = observer.manifest()
    for field in (
        "model",
        "dtype",
        "device",
        "torch",
        "transformers",
        "attention",
        "use_cache",
        "training",
        "template_sha256",
    ):
        if actual_observer[field] != manifest["parent_observer"][field]:
            raise ValueError("Reused scores have different observer setting: " + field)
    write(out / "01/observer.json", actual_observer)
    try:
        rows = read_json(out / "00/design.json")
        for role in ("source", "application"):
            if role == "application":
                barrier = read_json(out / "01/source_barrier.json")
                for name, sha in barrier["files"].items():
                    if digest(out / name) != sha:
                        raise ValueError("Changed sealed source")
            for row in rows:
                if row["role"] != role:
                    continue
                path = out / f"01/units/unit_{row['unit']:04d}.json"
                if path.exists():
                    load_arrays(path)
                    continue
                if row["reused"]:
                    meta, arrays = load_arrays(
                        parent / f"01/units/unit_{row['hm46_unit']:04d}.json"
                    )
                    native = meta["native"]
                    status["reused_units"] += 1
                else:
                    if (
                        status["attempted_forward_calls"] >= spec["max_fresh_forwards"]
                        or previous + time.monotonic() - started
                        >= spec["max_model_seconds"]
                    ):
                        raise TimeoutError("Finite model budget reached")
                    status.update(
                        status="COLLECTING",
                        in_flight=f"native/{row['unit']}",
                        model_seconds=previous + time.monotonic() - started,
                    )
                    status["attempted_forward_calls"] += 1
                    write(out / "run_status.json", status)
                    native, arrays = observe(
                        observer, row, manifest["alphabet"], spec["layer"]
                    )
                    status["forward_calls"] += 1
                save(
                    path,
                    {
                        "unit": row["unit"],
                        "role": role,
                        "reused": row["reused"],
                        "hm46_unit": row["hm46_unit"],
                        "native": native,
                    },
                    arrays,
                )
                status.update(
                    in_flight=None,
                    completed_units=status["completed_units"] + 1,
                    model_seconds=previous + time.monotonic() - started,
                )
                write(out / "run_status.json", status)
                if status["completed_units"] % 108 == 0:
                    print(
                        f"{role}: {status['completed_units']} scores/{status['forward_calls']} fresh",
                        flush=True,
                    )
            if role == "source" and not (out / "01/source_barrier.json").exists():
                paths = [
                    out / f"01/units/unit_{r['unit']:04d}.json"
                    for r in rows
                    if r["role"] == role
                ]
                files = {
                    str(p.relative_to(out)): digest(p)
                    for path in paths
                    for p in (path, path.with_suffix(".npz"))
                }
                if len(paths) != 432:
                    raise ValueError("Source432 scores required before application")
                write(
                    out / "01/source_barrier.json",
                    {
                        "files": files,
                        "sealed_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "source_scores": len(paths),
                    },
                )
        status.update(
            status="COLLECTED", model_seconds=previous + time.monotonic() - started
        )
        if (
            status["completed_units"] != spec["expected_scores"]
            or status["forward_calls"] != spec["max_fresh_forwards"]
            or status["reused_units"] != spec["expected_reused"]
        ):
            raise ValueError("Actual acquisition coverage differs from plan")
        write(out / "run_status.json", status)
        print(json.dumps(status), flush=True)
    except BaseException as exc:
        status.update(
            status="STOPPED_WITH_CHECKPOINT",
            error=type(exc).__name__ + ": " + str(exc),
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def analyze(out):
    """Retain all factorial contrasts, full matrices and available carrier diagnostics."""
    manifest = check(out)
    status = read_json(out / "run_status.json")
    if status["status"] != "COLLECTED" or status["in_flight"] is not None:
        raise ValueError("Need actual full finite acquisition")
    rows = read_json(out / "00/design.json")
    targets = manifest["targets"]
    scores, by_document, tables = {}, {}, {}
    native_rows, coefficient_rows, effect_rows, matrix_rows, carrier_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for row in rows:
        meta, arrays = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        score = meta["native"]["conditional_score"]
        scores[row["unit"]] = score
        by_document[row["hm46_doc_id"], row["query"], row["evaluation"]] = (row, arrays)
        key = (
            row["role"],
            row["wording"],
            row["other"],
            row["imagined"],
            row["content"],
            row["query"],
        )
        table = tables.setdefault(key, np.full((2, 3), np.nan))
        index = (row["polarity"], E.EVALUATIONS.index(row["evaluation"]))
        if not np.isnan(table[index]):
            raise ValueError("Duplicate actual cell")
        table[index] = score
        native_rows.append(
            {
                k: row[k]
                for k in (
                    "unit",
                    "doc_id",
                    "role",
                    "wording",
                    "content",
                    "query",
                    "polarity",
                    "other",
                    "imagined",
                    "evaluation",
                    "reused",
                    "hm46_unit",
                )
            }
            | {"score": score, "candidate_mass": meta["native"]["candidate_mass"]}
        )
    matrices = {}
    for key, table in tables.items():
        role, wording, other, imagined, content, query = key
        base = dict(
            zip(
                ("role", "wording", "other", "imagined", "content", "query"),
                key,
                strict=True,
            )
        )
        coeff, detail = E.decompose(table)
        for p, e in itertools.product(range(2), range(3)):
            coefficient_rows.append(
                {
                    **base,
                    "behaviour_component": p,
                    "appraisal_component": e,
                    "value": float(coeff[p, e]),
                    "reconstruction_error": detail["reconstruction_max_error"],
                    "parseval_error": detail["parseval_error"],
                }
            )
        for level, value in zip(
            E.EVALUATIONS, detail["behaviour_contrast"], strict=True
        ):
            effect_rows.append(
                {**base, "effect": "behaviour", "level": level, "value": float(value)}
            )
            group = (role, wording, other, imagined, level)
            matrices.setdefault(group, np.full((6, 6), np.nan))[
                targets.index(content), targets.index(query)
            ] = value
        for name, field in (
            ("evaluation", "evaluation_contrast_by_behaviour"),
            ("presence", "appraisal_presence_by_behaviour"),
        ):
            for polarity, value in enumerate(detail[field]):
                effect_rows.append(
                    {
                        **base,
                        "effect": name,
                        "level": str(polarity),
                        "value": float(value),
                    }
                )
        effect_rows.append(
            {
                **base,
                "effect": "interaction",
                "level": "difference_of_behaviour_contrasts",
                "value": float(detail["interaction"]),
            }
        )
    group_summary, spectral_rows = [], []
    v = np.array([1, -1, 1, 1, 1, 0], float) / np.sqrt(5)
    for role, wording, other, imagined in sorted({key[:4] for key in matrices}):
        key = (role, wording, other, imagined)
        baseline, positive, negative = (matrices[(*key, e)] for e in E.EVALUATIONS)
        evaluation, presence = np.full((6, 6), np.nan), np.full((6, 6), np.nan)
        for c, q in itertools.product(range(6), repeat=2):
            table = tables[(*key, targets[c], targets[q])]
            evaluation[c, q] = np.mean(table[:, 1] - table[:, 2])
            presence[c, q] = np.mean((table[:, 1] + table[:, 2]) / 2 - table[:, 0])
        reference_rms = float(np.sqrt(np.mean(baseline * baseline)))
        eval_energy = float(np.sum(evaluation * evaluation))
        for e in E.EVALUATIONS:
            m = matrices[(*key, e)]
            spectral_rows.append(
                {
                    "role": role,
                    "wording": wording,
                    "other": other,
                    "imagined": imagined,
                    "evaluation": e,
                    **M.spectrum(m),
                    **E.matrix_comparison(baseline, m),
                }
            )
            for c, q in itertools.product(range(6), repeat=2):
                matrix_rows.append(
                    {
                        "role": role,
                        "wording": wording,
                        "other": other,
                        "imagined": imagined,
                        "evaluation": e,
                        "content": targets[c],
                        "query": targets[q],
                        "behaviour_delta": float(m[c, q]),
                    }
                )
        group_summary.append(
            {
                "role": role,
                "wording": wording,
                "other": other,
                "imagined": imagined,
                "baseline_behaviour_rms": reference_rms,
                "evaluation_rms": float(np.sqrt(np.mean(evaluation * evaluation))),
                "evaluation_to_behaviour_rms": float(
                    np.linalg.norm(evaluation) / np.linalg.norm(baseline)
                )
                if reference_rms
                else None,
                "appraisal_presence_rms": float(np.sqrt(np.mean(presence * presence))),
                "interaction_rms": float(np.sqrt(np.mean((positive - negative) ** 2))),
                "evaluation_query_axis_squared_share": float(
                    np.sum((evaluation @ v) ** 2) / eval_energy
                )
                if eval_energy
                else None,
                "positive": E.matrix_comparison(baseline, positive),
                "negative": E.matrix_comparison(baseline, negative),
                "evaluation_matrix": evaluation.tolist(),
                "presence_matrix": presence.tolist(),
                "interaction_matrix": (positive - negative).tolist(),
            }
        )
    for key, (row, arrays) in by_document.items():
        if row["evaluation"] == "none":
            continue
        _, reference = by_document[key[0], key[1], "none"]
        for carrier in ("property", "endpoint", "query_end"):
            carrier_rows.append(
                {
                    "unit": row["unit"],
                    "role": row["role"],
                    "content": row["content"],
                    "query": row["query"],
                    "evaluation": row["evaluation"],
                    "carrier": carrier,
                    **E.array_change(reference[carrier], arrays[carrier]),
                }
            )
    for name, values in (
        ("readouts.csv", native_rows),
        ("coefficients.csv", coefficient_rows),
        ("effects.csv", effect_rows),
        ("response_matrices.csv", matrix_rows),
        ("carrier_changes.csv", carrier_rows),
    ):
        write_csv(out / "03" / name, values)
    write(out / "03/group_summary.json", group_summary)
    write(out / "03/spectra.json", spectral_rows)
    summary = {
        "status": "FINITE_HM47_ACQUIRED_AND_ANALYZED_NOT_GLOBAL",
        "run_status": status,
        "scores": len(native_rows),
        "effect_rows": len(effect_rows),
        "coefficient_rows": len(coefficient_rows),
        "matrix_cells": len(matrix_rows),
        "carrier_comparisons": len(carrier_rows),
        "minimum_candidate_mass": min(r["candidate_mass"] for r in native_rows),
        "median_candidate_mass": float(
            np.median([r["candidate_mass"] for r in native_rows])
        ),
        "maximum_property_numerical_difference_reported": max(
            r["maximum_absolute_change"]
            for r in carrier_rows
            if r["carrier"] == "property"
        ),
        "maximum_design_reconstruction_error": max(
            r["reconstruction_error"] for r in coefficient_rows
        ),
        "maximum_design_parseval_error": max(
            r["parseval_error"] for r in coefficient_rows
        ),
        "new_fits": 0,
        "human_personality_accuracy": False,
        "parent_evidence_reused_not_independent_replication": True,
        "group_summary": [
            {k: v for k, v in row.items() if not k.endswith("_matrix")}
            for row in group_summary
        ],
    }
    write(out / "03/summary.json", summary)
    write(
        out / "03/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for folder in ("00", "01", "03")
            for p in (out / folder).rglob("*")
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Run explicit finite stages; never launch a precision search or another worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        for stage, function in (
            ("prepare", prepare),
            ("collect", collect),
            ("analyze", analyze),
        ):
            if args.stage in (stage, "all"):
                function(args.out)


if __name__ == "__main__":
    main()
