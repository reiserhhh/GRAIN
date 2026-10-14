#!/usr/bin/env python3
"""HM87 body-only cold construction and finite query/carrier discrimination."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_cold_evidence as C
from grain_research.construction import math_query_evidence as M
from grain_research.construction.math_passive_event_composition import action_line
from grain_research.vector_mechanisms.navigation import (
    MODEL,
    digest,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.prefix import capture_prefix
from scripts.run_grain_author_contrast_hm82 import table, write
from scripts.run_grain_math_construct_routing import save
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_query_evidence_hm86 import check_hashes, seal, tokenize, unit
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_cold_evidence_kernel_hm87.json"
OUT = ROOT / "results/grain_math_response_v1/cold_evidence_kernel_r1"
READOUTS = ("conditional", "event_uniform", "pairing", "uniform_body")
CODE = (
    "configs/grain_cold_evidence_kernel_hm87.json",
    "docs/GRAIN_COLD_EVIDENCE_KERNEL_PLAN_HM87.md",
    "grain_research/construction/math_cold_evidence.py",
    "scripts/run_grain_cold_evidence_hm87.py",
    "scripts/score_grain_cold_evidence_hm87.py",
    "scripts/verify_grain_cold_evidence_hm87.py",
    "tests/test_grain_math_cold_evidence.py",
    "grain_research/vector_mechanisms/prefix.py",
    "scripts/run_grain_query_evidence_hm86.py",
    "grain_research/construction/math_query_evidence.py",
)


def prepare(out):
    """Compile sealed source parameters and all new inputs before new captures."""
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    if out.exists():
        raise FileExistsError("Preserve completed and partial packets")
    spec = read_json(SPEC)
    source = ROOT / spec["source"]
    check_hashes(ROOT, read_json(source / "00/manifest.json")["code"])
    check_hashes(source, read_json(source / "01/raw_seal.json"))
    if (
        read_json(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM86_NOT_GLOBAL"
    ):
        raise ValueError("Unverified source")
    old = read_json(source / "00/rows.json")
    material = read_json(ROOT / spec["source_material"])
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    if config.rope_parameters["rope_type"] != "default":
        raise ValueError("Only frozen default RoPE supported")
    frequencies, factor = Qwen2RotaryEmbedding.compute_default_rope_parameters(
        config, torch.device("cpu")
    )
    if factor != 1:
        raise ValueError("Unexpected rotary scale")
    with np.load(source / "01/calibration.npz", allow_pickle=False) as data:
        raw_queries, directions = data["raw_queries"].copy(), data["directions"].copy()
    gaps = []
    for target in spec["targets"]:
        ref = next(
            r for r in old if r["stage"] == "reference" and r["target"] == target
        )
        gaps.append(
            len(ref["input_ids"]) - 1 - max(i for g in ref["groups"] for i in g)
        )
    refs = [r for r in old if r["stage"] == "reference"]
    scaling = float(unit(source, refs[0]["unit"])["scaling"])
    bundle = {
        "raw_queries": raw_queries.tolist(),
        "directions": directions.tolist(),
        "frequencies": frequencies.numpy().tolist(),
        "scaling": scaling,
        "gaps": gaps,
        "targets": spec["targets"],
        "source_sealed_before_application": True,
    }
    rows = []

    def append(meta, tokens):
        rows.append({"unit": len(rows), **meta, **tokens})

    for row in [
        r
        for r in old
        if r["stage"] == "application"
        and r["context"] == 0
        and r["query"] == spec["targets"][0]
    ]:
        length = max(i for g in row["groups"] for i in g) + 1
        append(
            {"stage": "control", "old_unit": row["unit"], "event_order": [0, 1, 2]},
            {
                "body": row["body"],
                "input_ids": row["input_ids"][:length],
                "groups": row["groups"],
            },
        )
    for context, focal in enumerate(spec["targets"]):
        companion = spec["targets"][(context + 1) % 6]
        for order in (0, 1):
            permutation = [0, 1, 2] if order == 0 else [2, 1, 0]
            for corner, (x, y, g, h) in enumerate(M.points().astype(int).tolist()):
                lines = [
                    action_line("K1", "A", spec["fresh_behaviours"][focal][int(x > 0)]),
                    action_line(
                        "K2",
                        "A" if g > 0 else "B",
                        spec["fresh_behaviours"][companion][int(y > 0)],
                    ),
                    action_line("K3", "A" if h > 0 else "B", "opened a plain notebook"),
                ]
                body = "\n".join(lines[i] for i in permutation)
                meta = {
                    "context": context,
                    "order": order,
                    "corner": corner,
                    "coordinates": [x, y, g, h],
                    "event_order": permutation,
                    "focal": focal,
                    "companion": companion,
                }
                append(
                    {"stage": "body", **meta},
                    tokenize(body, focal, material["system"], tokenizer, True),
                )
    body_rows = [r for r in rows if r["stage"] == "body" and r["order"] == 0]
    for row in body_rows:
        for qi, query in enumerate(spec["targets"]):
            tokens = tokenize(row["body"], query, material["system"], tokenizer)
            endpoint = max(i for group in tokens["groups"] for i in group)
            if len(tokens["input_ids"]) - 1 - endpoint != gaps[qi]:
                raise ValueError("Query-position gap drift; do not tune a replacement")
            if tokens["input_ids"][: endpoint + 1] != row["input_ids"]:
                raise ValueError("Body-only input differs from native causal prefix")
            append(
                {"stage": "native", "body_unit": row["unit"], "query_index": qi}, tokens
            )
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete fixed design")
    (out / "00").mkdir(parents=True)
    write(out / "00/bundle.json", bundle)
    write(out / "00/rows.json", rows)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: digest(ROOT / n) for n in CODE},
            "source": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    source / "01/raw_seal.json",
                    source / "01/calibration.json",
                    source / "01/calibration.npz",
                    source / "verification.json",
                    MODEL / "config.json",
                ]
            },
            "model": read_json(source / "00/manifest.json")["model"],
            "inputs": seal(out, out / "00"),
            "readouts": READOUTS,
            "no_source_fit_or_answer_target": True,
        },
    )
    print(
        "HM87 prepared",
        len(rows),
        "captures; fixed source numerical bundle",
        flush=True,
    )


def collect(out):
    """Run the finite prefix collector with resumable units and no scoring head."""
    manifest, rows = (
        read_json(out / "00/manifest.json"),
        read_json(out / "00/rows.json"),
    )
    check_hashes(ROOT, manifest["code"] | manifest["source"])
    check_hashes(out, manifest["inputs"])
    check_hashes(MODEL, manifest["model"])
    observer, status_path = load_observer(), out / "01/status.json"
    status = (
        read_json(status_path)
        if status_path.exists()
        else {"completed": 0, "model_seconds": 0.0}
    )
    with collector_guard(out):
        try:
            for row in rows:
                path = out / f"01/units/unit_{row['unit']:04d}.json"
                if path.exists():
                    unit(out, row["unit"])
                    continue
                if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                    raise TimeoutError("Finite scientific packet budget reached")
                begin = time.monotonic()
                raw, trace = capture_prefix(observer, row["input_ids"])
                elapsed = time.monotonic() - begin
                raw.pop("boundary")
                save(
                    path,
                    {
                        "unit": row["unit"],
                        "trace": trace,
                        "seconds": elapsed,
                        "stage": row["stage"],
                    },
                    raw,
                )
                status = {
                    "state": "ACTIVE",
                    "completed": row["unit"] + 1,
                    "model_seconds": status["model_seconds"] + elapsed,
                    "last_unit": row["unit"],
                    "new_calls_this_invocation": observer.calls,
                }
                write_json(status_path, status)
                if status["completed"] % 64 == 0:
                    print(
                        "HM87",
                        status["completed"],
                        "/",
                        len(rows),
                        round(status["model_seconds"], 2),
                        "s",
                        flush=True,
                    )
            status["state"] = "CAPTURES_COMPLETE"
            write_json(status_path, status)
            write(out / "01/raw_seal.json", seal(out, out / "01/units"))
        except BaseException as exc:
            write_json(
                status_path,
                {**status, "state": "STOPPED_CHECKPOINT", "error": repr(exc)},
            )
            raise


def carrier(raw, row):
    """Export body KV and canonical event IDs only, retaining original rotary positions."""
    indices = sorted(i for group in row["groups"] for i in group)
    mapping = {
        index: event
        for group, event in zip(row["groups"], row["event_order"], strict=True)
        for index in group
    }
    return (
        raw["k"][indices],
        raw["v"][indices],
        np.asarray([mapping[i] for i in indices]),
        indices[-1],
    )


def analyze(out):
    """Freeze cold outputs before joining design truth or native-query comparisons."""
    if (out / "02").exists():
        raise FileExistsError("Preserve analysis and failed attempts")
    manifest, rows = (
        read_json(out / "00/manifest.json"),
        read_json(out / "00/rows.json"),
    )
    check_hashes(ROOT, manifest["code"] | manifest["source"])
    check_hashes(out, read_json(out / "01/raw_seal.json"))
    bundle = read_json(out / "00/bundle.json")
    body_rows = [r for r in rows if r["stage"] == "body"]
    carriers = [carrier(unit(out, r["unit"]), r) for r in body_rows]
    size = max(len(k) for k, _, _, _ in carriers)
    keys = np.zeros((len(carriers), size, *carriers[0][0].shape[1:]), np.float32)
    values, ids = np.zeros_like(keys), np.full((len(carriers), size), -1, int)
    lengths, endpoints = [], []
    for i, (k, v, event_ids, endpoint) in enumerate(carriers):
        keys[i, : len(k)], values[i, : len(k)], ids[i, : len(k)] = k, v, event_ids
        lengths.append(len(k))
        endpoints.append(endpoint)
    folder = out / "02/cold"
    folder.mkdir(parents=True)
    write(folder / "bundle.json", bundle)
    with (folder / "carriers.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            keys=keys,
            values=values,
            event_ids=ids,
            lengths=lengths,
            endpoints=endpoints,
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_cold_evidence_hm87.py"),
            str(folder),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, flush=True)
    write(folder / "seal.json", seal(out, folder))
    with np.load(folder / "observations.npz", allow_pickle=False) as data:
        flat = data["vectors"].copy()
    cold = np.zeros((6, 2, 16, 6, 4, 12, 128))
    for row, vector in zip(body_rows, flat, strict=True):
        cold[row["context"], row["order"], row["corner"]] = vector
    native = np.zeros((6, 16, 6, 4, 12, 128))
    raw_queries = np.zeros((6, 16, 6, 12, 128))
    causal_error, control_error, phase_error = 0.0, 0.0, 0.0
    source = ROOT / manifest["spec"]["source"]
    for row in rows:
        raw = unit(out, row["unit"])
        if row["stage"] == "control":
            old = unit(source, row["old_unit"])
            ix = np.concatenate(row["groups"])
            control_error = max(
                control_error,
                *(float(np.max(np.abs(raw[n][ix] - old[n][ix]))) for n in ("k", "v")),
            )
        if row["stage"] != "native":
            continue
        body = rows[row["body_unit"]]
        body_raw = unit(out, body["unit"])
        ix = sorted(i for group in row["groups"] for i in group)
        causal_error = max(
            causal_error,
            *(float(np.max(np.abs(raw[n][ix] - body_raw[n][ix]))) for n in ("k", "v")),
        )
        k, v, event_ids, endpoint = carrier(body_raw, body)
        qi, context, corner = row["query_index"], body["context"], body["corner"]
        q = C.rotate(
            raw["raw_query"],
            endpoint + bundle["gaps"][qi],
            bundle["frequencies"],
            bundle["scaling"],
        )
        phase_error = max(phase_error, float(np.max(np.abs(q - raw["q"]))))
        native[context, corner, qi] = C.reading(k, v, q, event_ids)
        raw_queries[context, corner, qi] = raw["raw_query"]
    dirs = np.asarray(bundle["directions"])
    projections = np.einsum("coaqrhd,thd->coaqrt", cold, dirs)
    native_projections = np.einsum("caqrhd,thd->caqrt", native, dirs)
    conditions, accounts, actions = [], [], []
    points = M.points().astype(int)
    plus, minus = M.evidence_laws(manifest["spec"]["epsilon"])
    for view, field in [("cold", projections), ("native", native_projections[:, None])]:
        for context in range(6):
            target = (context + 1) % 6
            for order in range(field.shape[1]):
                for readout, name in enumerate(READOUTS):
                    score = field[context, order, :, :, readout, target]
                    own = score[(points[:, 2] == 1) & (points[:, 1] == 1)].mean(
                        0
                    ) - score[(points[:, 2] == 1) & (points[:, 1] == -1)].mean(0)
                    other = score[(points[:, 2] == -1) & (points[:, 1] == 1)].mean(
                        0
                    ) - score[(points[:, 2] == -1) & (points[:, 1] == -1)].mean(0)
                    conditions.append(
                        {
                            "view": view,
                            "order": order,
                            "target": bundle["targets"][target],
                            "readout": name,
                            "matched_own_action": float(own[target]),
                            "matched_other_action": float(other[target]),
                            "target_selectivity": float(
                                own[target] - np.delete(own, target).mean()
                            ),
                        }
                    )
                    for qi, query in enumerate(bundle["targets"]):
                        contrast = (plus - minus) @ score[:, qi]
                        equal_counts = points[:, 3] == -points[:, 2]
                        eq_weights = (plus - minus)[equal_counts] * 2
                        accounts.append(
                            {
                                "view": view,
                                "order": order,
                                "target": bundle["targets"][target],
                                "query": query,
                                "readout": name,
                                "yg_contrast": float(contrast),
                                "equal_name_count_contrast": float(
                                    eq_weights @ score[equal_counts, qi]
                                ),
                            }
                        )
    body_map = {(r["context"], r["corner"]): r for r in body_rows if r["order"] == 0}
    polarization_error = 0.0
    for context in range(6):
        for low_index, point in enumerate(points):
            if point[1] != -1:
                continue
            high_point = point.copy()
            high_point[1] = 1
            hi = int(np.flatnonzero((points == high_point).all(1))[0])
            high_carrier = carrier(
                unit(out, body_map[context, hi]["unit"]), body_map[context, hi]
            )
            low_carrier = carrier(
                unit(out, body_map[context, low_index]["unit"]),
                body_map[context, low_index],
            )
            for qi, query in enumerate(bundle["targets"]):

                def reader(data, query_index=qi):
                    k, v, event_ids, endpoint = data
                    return lambda raw_q: C.reading(
                        k,
                        v,
                        C.rotate(
                            raw_q,
                            endpoint + bundle["gaps"][query_index],
                            bundle["frequencies"],
                            bundle["scaling"],
                        ),
                        event_ids,
                    )

                b, q, full = C.polarized_action(
                    reader(high_carrier),
                    reader(low_carrier),
                    raw_queries[context, hi, qi],
                    raw_queries[context, low_index, qi],
                )
                polarization_error = max(
                    polarization_error, float(np.max(np.abs(b + q - full)))
                )
                for ri, name in enumerate(READOUTS):
                    for ti, target in enumerate(bundle["targets"]):
                        actions.append(
                            {
                                "context": context,
                                "low_corner": low_index,
                                "query": query,
                                "direction": target,
                                "actor_A": bool(point[2] == 1),
                                "readout": name,
                                "body": float(np.sum(b[ri] * dirs[ti])),
                                "query_action": float(np.sum(q[ri] * dirs[ti])),
                                "full": float(np.sum(full[ri] * dirs[ti])),
                            }
                        )
    if (
        max(causal_error, control_error, phase_error)
        > manifest["spec"]["prefix_tolerance"]
    ):
        raise ValueError(
            "Causal prefix/rotary implementation discrepancy; retain artifacts"
        )
    table(out / "02/target_selection.csv", conditions)
    table(out / "02/author_composition.csv", accounts)
    table(out / "02/polarized_actions.csv", actions)
    with (out / "02/observations.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            cold=cold,
            native=native,
            projections=projections,
            native_projections=native_projections,
            raw_queries=raw_queries,
        )
    summary = {
        "status": "ANALYZED_PENDING_HM87_VERIFICATION",
        "calls": len(rows),
        "fresh_bodies": 192,
        "native_bodies_forward_order": 96,
        "cold_neural_calls": 0,
        "model_seconds": read_json(out / "01/status.json")["model_seconds"],
        "body_KV_native_error": causal_error,
        "old_control_KV_error": control_error,
        "virtual_query_rotation_error": phase_error,
        "polarization_error": polarization_error,
        "all_target_selection": conditions,
        "original_scope_complete": False,
        "same_agent_arithmetic_not_independent_rebuilder": True,
        "source_parameters_supervised_not_new_factor_discovery": True,
    }
    write(out / "02/summary.json", summary)
    write(out / "02/output_seal.json", seal(out, out / "02"))
    print("HM87 analysis complete; all views retained", flush=True)


def main():
    """Select one finite phase; no automatic parameter or material search."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    for phase in (
        ("prepare", "collect", "analyze") if args.phase == "all" else (args.phase,)
    ):
        globals()[phase](args.out)


if __name__ == "__main__":
    main()
