#!/usr/bin/env python3
"""HM71: source-fixed event relations tested on unseen wording and ordering."""

from __future__ import annotations

import argparse
import itertools
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_cold_relational as M
from grain_research.construction import math_ownership_composition as O
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_common_observables import (
    OUT as CHARTS,
)
from scripts.run_grain_math_common_observables import (
    SOURCES,
    load_sources,
    write,
    write_csv,
)
from scripts.run_grain_math_common_observables import (
    bound_check as chart_check,
)
from scripts.run_grain_math_construct_routing import save
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_ownership_composition import CODE as OLD_CODE
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_cold_relational_v1.json"
OUT = ROOT / "results/grain_math_response_v1/cold_relational_measurement_r1"
MAX_CALLS, MAX_SECONDS = 2880, 1200
IDENTITY = (
    "unit",
    "doc_id",
    "form",
    "pair_id",
    "focal",
    "companion",
    "corner",
    "query",
)
CODE = tuple(
    dict.fromkeys(
        (
            "docs/GRAIN_COLD_RELATIONAL_MEASUREMENT_PLAN_V1.md",
            "configs/grain_math_cold_relational_v1.json",
            "grain_research/construction/math_cold_relational.py",
            "grain_research/construction/math_common_observables.py",
            "scripts/run_grain_math_common_observables.py",
            "scripts/run_grain_math_cold_relational.py",
            "tests/test_grain_math_cold_relational.py",
            *OLD_CODE,
        )
    )
)


def load_chart(stage, pair_id):
    """Load immutable source-only coefficients; no cold fitting is available."""
    with np.load(CHARTS / f"01/{stage}_pair{pair_id:02d}.npz", allow_pickle=False) as a:
        return {name: a[name] for name in a.files}


def prepare(out):
    """Seal all new bodies, fixed event correspondences and source maps beforehand."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve the existing packet; never overwrite")
    chart_check(CHARTS)
    old = read_json(SOURCES["HM68"] / "00/manifest.json")
    material = read_json(ROOT / "configs/grain_math_passive_event_composition_v1.json")
    spec = read_json(SPEC)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    if [
        tokenizer.encode(s, add_special_tokens=False) for s in old["alphabet"]["labels"]
    ] != [[t] for t in old["alphabet"]["token_ids"]]:
        raise ValueError("Observer candidate alphabet drift")
    docs, rows, prefix, twins, links = M.documents(spec), [], {}, {}, []
    for doc in docs:
        for query in O.TARGETS:
            text, ids, part, _, _ = render(
                doc["body"], query, "A", 1, material["system"], tokenizer
            )
            body_prefix = ids[: part["body"][-1] + 1]
            if doc["doc_id"] in prefix and prefix[doc["doc_id"]] != body_prefix:
                raise ValueError("Question-dependent evidence prefix")
            prefix[doc["doc_id"]] = body_prefix
            row = {
                **doc,
                "unit": len(rows),
                "query": query,
                "input_ids": ids,
                "partition": part,
                "rendered": text,
            }
            rows.append(row)
            twins.setdefault((doc["pair_id"], doc["corner"], query), []).append(row)
        for event in doc["events"]:
            links.append(
                {
                    "form": doc["form"],
                    "pair_id": doc["pair_id"],
                    "corner": doc["corner"],
                    **event,
                    "source_action": O.ANCHOR
                    if event["target"] == "anchor"
                    else material["behaviours"][event["target"]][0][
                        int(event["polarity"] > 0)
                    ],
                    "new_action": spec["anchor"]
                    if event["target"] == "anchor"
                    else spec["behaviours"][event["target"]][
                        int(event["polarity"] > 0)
                    ],
                    "equivalence": "DESIGN_DECLARED_NOT_INDEPENDENTLY_CODED",
                }
            )
    proofs = []
    for key, (a, b) in twins.items():
        if a["events"] != b["events"] or Counter(a["body"].splitlines()) != Counter(
            b["body"].splitlines()
        ):
            raise ValueError("Undeclared semantic edit in reordering control")
        proofs.append(
            {
                "pair_id": key[0],
                "corner": key[1],
                "query": key[2],
                "units": [a["unit"], b["unit"]],
                "same_event_lines": True,
                "same_actual_token_bag": Counter(a["input_ids"])
                == Counter(b["input_ids"]),
            }
        )
    if len(rows) != MAX_CALLS or len({d["body"] for d in docs}) != 480:
        raise ValueError("Incomplete or duplicated material")
    (out / "00/code").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/reordering_proofs.json", proofs)
    write_csv(out / "00/event_correspondences.csv", links)
    code = {}
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dst)
        code[name] = digest(dst)
    inputs = [
        CHARTS / "manifest.json",
        CHARTS / "output_seal.json",
        CHARTS / "verification.json",
        SOURCES["HM69"] / "01/profiles.npz",
        SOURCES["HM69"] / "01/rows.json",
        SOURCES["HM68"] / "03/stages_and_worlds.npz",
        SOURCES["HM68"] / "01/observer.json",
    ]
    inputs.extend(
        CHARTS / f"01/{stage}_pair{i:02d}.npz"
        for stage, i in itertools.product(O.STAGES, range(30))
    )
    write(
        out / "00/manifest.json",
        {
            "packet": "HM71",
            "code": code,
            "assets": {
                str(p.relative_to(out)): digest(p)
                for p in (
                    out / "00/rows.json",
                    out / "00/reordering_proofs.json",
                    out / "00/event_correspondences.csv",
                )
            },
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            "alphabet": old["alphabet"],
            "model_config_sha256": digest(MODEL / "config.json"),
            "tokenizer_files": {
                p.name: digest(p) for p in MODEL.glob("*token*") if p.is_file()
            },
            "max_forwards": MAX_CALLS,
            "max_collector_wall_seconds": MAX_SECONDS,
            "layer": 16,
            "targets": O.TARGETS,
            "forms": M.FORMS,
            "pairs": M.PAIRS,
            "cold_source_family": 0,
            "new_bodies": 480,
            "cold_fits": 0,
            "model_weights_not_separately_hashed": True,
            "human_truth": False,
            "event_equivalence_is_design_declared": True,
            "original_scope_complete": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "attempted_forward_calls": 0,
            "completed_units": 0,
            "collector_wall_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        "HM71 prepared:480 new bodies/2880 fixed readings; no cold calibration",
        flush=True,
    )


def check(out):
    """Verify actual source/code/material identities, not a completion label."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift:" + name)
    for name, sha in manifest["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen material drift:" + name)
    for name, sha in manifest["sources"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source drift:" + name)
    for name, sha in manifest["tokenizer_files"].items():
        if digest(MODEL / name) != sha:
            raise ValueError("Frozen tokenizer drift")
    if digest(MODEL / "config.json") != manifest["model_config_sha256"]:
        raise ValueError("Frozen model config drift")
    return manifest


def collect(out):
    """Acquire each frozen reading once, retaining uncertain/failed attempts."""
    manifest, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("Stopped/uncertain acquisition requires review, not retry")
    actual_files = list((out / "01/units").glob("unit_*.json"))
    if len(actual_files) != status["completed_units"]:
        raise ValueError("Capture/status mismatch")
    start, previous = time.monotonic(), status["collector_wall_seconds"]
    try:
        observer = load_observer()
        actual, old = (
            observer.manifest(),
            read_json(SOURCES["HM68"] / "01/observer.json"),
        )
        for key in (
            "model",
            "torch",
            "transformers",
            "dtype",
            "device",
            "hidden_size",
            "layers",
            "max_tokens",
            "attention",
            "training",
            "template_sha256",
        ):
            if actual[key] != old[key]:
                raise ValueError("Observer drift:" + key)
        write(out / "01/observer.json", actual)
        for row in read_json(out / "00/rows.json"):
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                meta, _ = load_arrays(path)
                if any(meta[k] != row[k] for k in IDENTITY):
                    raise ValueError("Resume identity mismatch")
                continue
            elapsed = previous + time.monotonic() - start
            if status["attempted_forward_calls"] >= MAX_CALLS or elapsed >= MAX_SECONDS:
                raise TimeoutError("Fixed HM71 collection bound")
            status.update(
                status="COLLECTING",
                in_flight=row["unit"],
                collector_wall_seconds=elapsed,
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, arrays = observe_full(
                observer, row, manifest["alphabet"], manifest["layer"]
            )
            save(
                path,
                {k: row[k] for k in IDENTITY} | {"native": native},
                {
                    "body_end": arrays["full"][row["partition"]["body"][-1]],
                    "query_end": arrays["full"][-1],
                },
            )
            status["forward_calls"] += 1
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                collector_wall_seconds=previous + time.monotonic() - start,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 240 == 0:
                print(
                    f"HM71 ACTUAL:{status['completed_units']}/2880;"
                    f"{status['collector_wall_seconds']:.1f}s",
                    flush=True,
                )
        if any(
            status[k] != MAX_CALLS
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        ):
            raise ValueError("Incomplete collection")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
        paths = [out / "run_status.json", out / "01/observer.json"]
        paths.extend(sorted((out / "01/units").glob("unit_*.*")))
        write(
            out / "01/raw_seal.json",
            {str(p.relative_to(out)): digest(p) for p in paths},
        )
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            collector_wall_seconds=previous + time.monotonic() - start,
        )
        write(out / "run_status.json", status)
        raise


def raw_tables(out):
    """Reconstruct every cold factorial table from actual captures with prefix checks."""
    for name, sha in read_json(out / "01/raw_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Raw capture drift:" + name)
    groups, seen, body_prefix = {}, set(), {}
    prefix_error = 0.0
    for row in read_json(out / "00/rows.json"):
        meta, arrays = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        if any(meta[k] != row[k] for k in IDENTITY):
            raise ValueError("Raw unit identity mismatch")
        key = (row["form"], row["pair_id"])
        group = groups.setdefault(
            key, {"logits": np.zeros((8, 6, 10)), "body_endpoint": np.zeros((8, 1536))}
        )
        q, cell = O.TARGETS.index(row["query"]), row["corner"]
        if (key, cell, q) in seen:
            raise ValueError("Duplicate cold event/query")
        seen.add((key, cell, q))
        z = np.asarray(meta["native"]["candidate_logits"], float)
        if abs(O.decode(z)["score"] - meta["native"]["conditional_score"]) > 1e-10:
            raise ValueError("Raw candidate decoder mismatch")
        group["logits"][cell, q] = z
        h = arrays["body_end"].astype(float)
        if (key, cell) in body_prefix:
            prefix_error = max(
                prefix_error, float(np.max(abs(h - body_prefix[key, cell])))
            )
        else:
            body_prefix[key, cell] = h
            group["body_endpoint"][cell] = h
    if len(seen) != MAX_CALLS or len(groups) != 60 or prefix_error > 1e-5:
        raise ValueError("Incomplete cold support/question-dependent body")
    for group in groups.values():
        logits = group.pop("logits")
        centered = logits - logits.mean(-1, keepdims=True)
        _, control = O.link_controls(centered)
        group.update(
            centered_logits=centered,
            **O.decode(logits),
            **{k + "_link_control": v for k, v in control.items()},
        )
    return groups, prefix_error


def norm(value):
    """Store actual magnitudes rather than labeling tiny differences as discovery."""
    return float(np.linalg.norm(value))


def analyze(out):
    """Test frozen measurement transport without adjusting it to cold responses."""
    check(out)
    if (
        read_json(out / "run_status.json")["status"] != "COLLECTED"
        or (out / "03").exists()
    ):
        raise ValueError("Complete acquisition and fresh analysis destination required")
    (out / "03").mkdir()
    tables, prefix = raw_tables(out)
    sources, profiles = load_sources()
    arrays, rows, fibers, chart_rows = {}, [], [], []
    for form, pair_id, stage in itertools.product(M.FORMS, range(30), O.STAGES):
        chart = load_chart(stage, pair_id)
        old_z = chart["basis"]
        f = tables[form, pair_id][stage].reshape(8, -1)
        new_z = M.cold_chart(f, chart["calibration0"])
        key = f"{form}_{stage}_pair{pair_id:02d}"
        arrays[key + "__native"] = f
        arrays[key + "__z"] = new_z
        arrays[key + "__kernel"] = M.polynomial_kernel(new_z)
        chart_rows.append(
            {
                "form": form,
                "stage": stage,
                "pair_id": pair_id,
                "focal": M.PAIRS[pair_id][0],
                "companion": M.PAIRS[pair_id][1],
                "calibration_coefficient_norm": norm(chart["calibration0"]),
                "event_distortion_max": float(
                    np.max(np.linalg.norm(new_z - old_z, axis=1))
                ),
                "event_distortion_RMS": float(np.sqrt(np.mean((new_z - old_z) ** 2))),
                "source_event_radius": float(np.max(np.linalg.norm(old_z, axis=1))),
            }
        )
        for case_id, p in enumerate(profiles[pair_id]):
            old_delta, delta = M.contrast(p, old_z), M.contrast(p, new_z)
            wrong = M.contrast(p, np.roll(new_z, 1, axis=0))
            certificate = M.perturbation_certificate(p, old_z, new_z)
            d = p[1] - p[0]
            native = d @ f
            old_native = d @ sources[stage][pair_id, 0]
            other_native = d @ sources[stage][pair_id, 1]
            reference_bank = 1 if case_id == 6 else 0
            reference_native = d @ sources[stage][pair_id, reference_bank]
            lifted = M.moments(p, new_z)["lifted"]
            reconstructed = lifted @ chart["inverse"]
            arrays[f"{key}__case{case_id}"] = np.concatenate(
                (
                    native,
                    delta["means"],
                    delta["second"].ravel(),
                    delta["independent_second"].ravel(),
                    wrong["means"],
                    wrong["second"].ravel(),
                    reconstructed.ravel(),
                )
            )
            row = {
                "form": form,
                "stage": stage,
                "pair_id": pair_id,
                "case_id": case_id,
                "case": (
                    "choice_f0",
                    "choice_f1",
                    "expression_f0",
                    "expression_f1",
                    "law_only",
                    "mixed",
                    "same_scores_f0",
                    "same_scores_f1",
                )[case_id],
                "source_mean_gap": norm(old_delta["means"]),
                "cold_mean_gap": norm(delta["means"]),
                "source_second_gap": norm(old_delta["second"]),
                "cold_second_gap": norm(delta["second"]),
                "cold_independent_second_gap": norm(delta["independent_second"]),
                "source_joint_distance": norm(M.feature_delta(old_delta)),
                "cold_joint_distance": norm(M.feature_delta(delta)),
                "joint_direction_cosine": M.cosine(
                    M.feature_delta(old_delta), M.feature_delta(delta)
                ),
                "second_direction_cosine": M.cosine(
                    old_delta["second"], delta["second"]
                ),
                "wrong_identity_joint_cosine": M.cosine(
                    M.feature_delta(old_delta), M.feature_delta(wrong)
                ),
                "cold_native_gap": norm(native),
                "source_native_gap": norm(old_native),
                "other_source_native_gap": norm(other_native),
                "native_direction_cosine": M.cosine(old_native, native),
                "other_source_native_cosine": M.cosine(other_native, native),
                "native_reference_bank": reference_bank,
                "transport_reference_native_cosine": M.cosine(reference_native, native),
                "old_inverse_profile_max_error": float(np.max(abs(reconstructed - p))),
                "old_inverse_probability_min": float(reconstructed.min()),
                "old_inverse_sum_max_error": float(
                    np.max(abs(reconstructed.sum(1) - 1))
                ),
                "kernel_identity_error": abs(
                    float(d @ arrays[key + "__kernel"] @ d)
                    - norm(M.feature_delta(delta)) ** 2
                ),
                **certificate,
            }
            rows.append(row)
            if stage == "score" and case_id in (6, 7):
                fibers.append(row)
    write_csv(out / "03/profile_contrasts.csv", rows)
    write_csv(out / "03/score_fibers.csv", fibers)
    write_csv(out / "03/cold_charts.csv", chart_rows)
    np.savez_compressed(out / "03/observations_and_moments.npz", **arrays)
    summaries = []
    for form, stage in itertools.product(M.FORMS, O.STAGES):
        selected = [
            r
            for r in rows
            if r["form"] == form and r["stage"] == stage and r["case_id"] in (6, 7)
        ]

        def avg(name, selected=selected):
            values = [r[name] for r in selected if r[name] is not None]
            return float(np.mean(values)) if values else None

        summaries.append(
            {
                "form": form,
                "stage": stage,
                "fixed_score_fibers": len(selected),
                "source_mean_max": max(r["source_mean_gap"] for r in selected),
                "cold_mean_RMS_norm": float(
                    np.sqrt(np.mean([r["cold_mean_gap"] ** 2 for r in selected]))
                ),
                "cold_second_RMS_norm": float(
                    np.sqrt(np.mean([r["cold_second_gap"] ** 2 for r in selected]))
                ),
                "cold_second_min": min(r["cold_second_gap"] for r in selected),
                "cold_native_RMS_norm": float(
                    np.sqrt(np.mean([r["cold_native_gap"] ** 2 for r in selected]))
                ),
                "cold_native_min": min(r["cold_native_gap"] for r in selected),
                "joint_direction_mean": avg("joint_direction_cosine"),
                "second_direction_mean": avg("second_direction_cosine"),
                "wrong_identity_joint_mean": avg("wrong_identity_joint_cosine"),
                "other_source_native_cosine_mean": avg("other_source_native_cosine"),
                "transport_reference_native_cosine_mean": avg(
                    "transport_reference_native_cosine"
                ),
                "old_inverse_error_max": max(
                    r["old_inverse_profile_max_error"] for r in selected
                ),
                "certificate_positive": sum(
                    r["certified_separation_lower"] > 0 for r in selected
                ),
                "mean_bound_violation_max": max(
                    r["mean_discrepancy"] - r["mean_bound"] for r in selected
                ),
                "second_bound_violation_max": max(
                    r["second_discrepancy"] - r["second_bound"] for r in selected
                ),
            }
        )
    write(
        out / "03/summary.json",
        {
            "packet": "HM71",
            "actual_new_forwards": MAX_CALLS,
            "new_bodies": 480,
            "profile_rows": len(rows),
            "score_fiber_rows": len(fibers),
            "cold_chart_rows": len(chart_rows),
            "new_fits_API_private_labels_training": [0, 0, 0, 0, 0],
            "prefix_error": prefix,
            "stages": summaries,
            "original_scope_complete": False,
            "collector_wall_seconds": read_json(out / "run_status.json")[
                "collector_wall_seconds"
            ],
            "interpretation": "Cold source-map distortion is retained, not refitted; designed profiles are not people",
        },
    )
    paths = sorted(
        p for p in (out / "03").iterdir() if p.is_file() and not p.name.startswith("._")
    )
    write(
        out / "03/output_seal.json", {str(p.relative_to(out)): digest(p) for p in paths}
    )
    write(
        out / "checkpoint.json",
        {
            "status": "ANALYZED_HM71",
            "active_processes": [],
            "no_fidelity_repair": True,
            "original_scope_complete": False,
        },
    )
    print("HM71 analyzed:2880 profiles/120 score fibers/360 cold charts", flush=True)


def verify(out):
    """Replay actual readouts and event sums with alternate scalar arithmetic."""
    import csv

    check(out)
    for name, sha in read_json(out / "03/output_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Scientific output changed:" + name)
    tables, prefix = raw_tables(out)
    sources, profiles = load_sources()
    checks, worst = 0, 0.0

    def compare(a, b, tolerance=2e-8):
        nonlocal checks, worst
        a, b = np.asarray(a, float), np.asarray(b, float)
        error = float(np.max(abs(a - b))) if a.size else 0.0
        if (
            a.shape != b.shape
            or not np.isfinite(a).all()
            or error > tolerance * max(1.0, float(np.max(abs(b))))
        ):
            raise ValueError(f"Alternate arithmetic mismatch:{error}")
        checks += a.size
        worst = max(worst, error)

    for row in read_json(out / "00/rows.json"):
        meta, _ = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        z = meta["native"]["candidate_logits"]
        weights = [math.exp(v - max(z)) for v in z]
        score = math.fsum(i * p for i, p in enumerate(weights)) / (
            9 * math.fsum(weights)
        )
        compare(score, meta["native"]["conditional_score"], 1e-12)
    with (out / "03/profile_contrasts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(out / "03/observations_and_moments.npz", allow_pickle=False) as data:
        for form, stage, pair_id in itertools.product(M.FORMS, O.STAGES, range(30)):
            key, chart = f"{form}_{stage}_pair{pair_id:02d}", load_chart(stage, pair_id)
            f = tables[form, pair_id][stage].reshape(8, -1)
            compare(f, data[key + "__native"])
            a = np.column_stack((np.ones(8), f))
            c = chart["calibration0"]
            z = np.array(
                [
                    [
                        math.fsum(
                            float(a[e, k]) * float(c[k, j]) for k in range(len(c))
                        )
                        for j in range(c.shape[1])
                    ]
                    for e in range(8)
                ]
            )
            compare(z, data[key + "__z"])
            kernel = np.array(
                [
                    [
                        math.fsum(z[e] * z[v]) + math.fsum(z[e] * z[v]) ** 2
                        for v in range(8)
                    ]
                    for e in range(8)
                ]
            )
            compare(kernel, data[key + "__kernel"])
            for case_id, p in enumerate(profiles[pair_id]):
                d = p[1] - p[0]
                mean = np.array(
                    [
                        math.fsum(d[e] * z[e, j] for e in range(8))
                        for j in range(z.shape[1])
                    ]
                )
                second = np.array(
                    [
                        [
                            math.fsum(d[e] * z[e, i] * z[e, j] for e in range(8))
                            for j in range(z.shape[1])
                        ]
                        for i in range(z.shape[1])
                    ]
                )
                individual_mean = np.array(
                    [
                        [
                            math.fsum(p[u, e] * z[e, j] for e in range(8))
                            for j in range(z.shape[1])
                        ]
                        for u in range(2)
                    ]
                )
                independent = np.outer(
                    individual_mean[1], individual_mean[1]
                ) - np.outer(individual_mean[0], individual_mean[0])
                wrong_z = np.roll(z, 1, axis=0)
                wrong_mean = np.array(
                    [
                        math.fsum(d[e] * wrong_z[e, j] for e in range(8))
                        for j in range(z.shape[1])
                    ]
                )
                wrong_second = np.array(
                    [
                        [
                            math.fsum(
                                d[e] * wrong_z[e, i] * wrong_z[e, j] for e in range(8)
                            )
                            for j in range(z.shape[1])
                        ]
                        for i in range(z.shape[1])
                    ]
                )
                products = [
                    z[:, i] * z[:, j]
                    for i in range(z.shape[1])
                    for j in range(i, z.shape[1])
                ]
                h = np.column_stack((np.ones(8), z, *products))
                reconstructed = (p @ h) @ chart["inverse"]
                native = d @ f
                saved_vector = np.concatenate(
                    (
                        native,
                        mean,
                        second.ravel(),
                        independent.ravel(),
                        wrong_mean,
                        wrong_second.ravel(),
                        reconstructed.ravel(),
                    )
                )
                compare(saved_vector, data[f"{key}__case{case_id}"])
                chosen = next(
                    r
                    for r in rows
                    if r["form"] == form
                    and r["stage"] == stage
                    and int(r["pair_id"]) == pair_id
                    and int(r["case_id"]) == case_id
                )
                compare(np.linalg.norm(mean), float(chosen["cold_mean_gap"]))
                compare(np.linalg.norm(second), float(chosen["cold_second_gap"]))
                compare(
                    np.linalg.norm(independent),
                    float(chosen["cold_independent_second_gap"]),
                )
                compare(np.linalg.norm(native), float(chosen["cold_native_gap"]))
                compare(
                    np.max(abs(reconstructed - p)),
                    float(chosen["old_inverse_profile_max_error"]),
                )
                compare(
                    reconstructed.min(), float(chosen["old_inverse_probability_min"])
                )
                compare(
                    np.max(abs(reconstructed.sum(1) - 1)),
                    float(chosen["old_inverse_sum_max_error"]),
                )
                source_delta = M.feature_delta(M.contrast(p, chart["basis"]))
                for name, value in (
                    (
                        "joint_direction_cosine",
                        M.cosine(source_delta, np.concatenate((mean, second.ravel()))),
                    ),
                    (
                        "second_direction_cosine",
                        M.cosine(source_delta[len(mean) :], second.ravel()),
                    ),
                    (
                        "wrong_identity_joint_cosine",
                        M.cosine(
                            source_delta,
                            np.concatenate((wrong_mean, wrong_second.ravel())),
                        ),
                    ),
                    (
                        "native_direction_cosine",
                        M.cosine(d @ sources[stage][pair_id, 0], native),
                    ),
                    (
                        "other_source_native_cosine",
                        M.cosine(d @ sources[stage][pair_id, 1], native),
                    ),
                    (
                        "transport_reference_native_cosine",
                        M.cosine(
                            d @ sources[stage][pair_id, 1 if case_id == 6 else 0],
                            native,
                        ),
                    ),
                ):
                    if value is None:
                        if chosen[name] != "":
                            raise ValueError("Undefined zero direction was scored")
                    else:
                        compare(value, float(chosen[name]))
                kernel_distance = math.fsum(
                    d[i] * kernel[i, j] * d[j] for i in range(8) for j in range(8)
                )
                compare(kernel_distance, float(chosen["cold_joint_distance"]) ** 2)
                certificate = M.perturbation_certificate(p, chart["basis"], z)
                for name, value in certificate.items():
                    compare(value, float(chosen[name]))
                if (
                    certificate["mean_discrepancy"] > certificate["mean_bound"] + 1e-8
                    or certificate["second_discrepancy"]
                    > certificate["second_bound"] + 1e-8
                ):
                    raise ValueError("Perturbation certificate violated")
    summary = read_json(out / "03/summary.json")
    if (len(rows), summary["score_fiber_rows"], summary["cold_chart_rows"]) != (
        2880,
        120,
        360,
    ):
        raise ValueError("Incomplete scientific coverage")
    for item in summary["stages"]:
        selected = [
            r
            for r in rows
            if r["form"] == item["form"]
            and r["stage"] == item["stage"]
            and int(r["case_id"]) in (6, 7)
        ]
        compare(
            max(float(r["source_mean_gap"]) for r in selected), item["source_mean_max"]
        )
        for field, source in (
            ("cold_mean_RMS_norm", "cold_mean_gap"),
            ("cold_second_RMS_norm", "cold_second_gap"),
            ("cold_native_RMS_norm", "cold_native_gap"),
        ):
            compare(
                math.sqrt(
                    math.fsum(float(r[source]) ** 2 for r in selected) / len(selected)
                ),
                item[field],
            )
        for field, source in (
            ("joint_direction_mean", "joint_direction_cosine"),
            ("second_direction_mean", "second_direction_cosine"),
            ("wrong_identity_joint_mean", "wrong_identity_joint_cosine"),
            (
                "transport_reference_native_cosine_mean",
                "transport_reference_native_cosine",
            ),
        ):
            selected_values = [float(r[source]) for r in selected if r[source]]
            compare(math.fsum(selected_values) / len(selected_values), item[field])
    write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM71_NOT_GLOBAL",
            "actual_calls": read_json(out / "run_status.json")["forward_calls"],
            "alternate_scalar_checks": checks,
            "max_arithmetic_error": worst,
            "prefix_error": prefix,
            "same_agent_alternate_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    write(
        out / "checkpoint.json",
        {
            "status": "VERIFIED_FINITE_HM71_NOT_GLOBAL",
            "active_processes": [],
            "original_scope_complete": False,
        },
    )
    print(f"HM71 verified:{checks} alternate scalars;maxerror={worst:.3g}", flush=True)


def main():
    """Expose separate sealed acquisition, analysis and numerical verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.action == "collect":
        with collector_guard(args.out):
            collect(args.out)
    else:
        {"prepare": prepare, "analyze": analyze, "verify": verify}[args.action](
            args.out
        )


if __name__ == "__main__":
    main()
