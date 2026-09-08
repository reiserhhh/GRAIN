#!/usr/bin/env python3
"""Frozen reference matrix readout on prior source authors, with prefix-only inference."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.level_response import (
    affine_parts,
    correlation,
    paired_quality,
)
from grain_research.vector_mechanisms.prefix import capture_prefix
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_level_response import FULL, question_mask, verify
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION, covariance

LEVEL = BASE / "complete_level_response_v1"
SOURCE = BASE / "source_remainder_s9_r1"
PLAN = ROOT / "docs/GRAIN_PREFIX_SOURCE_CONNECTION_PLAN_V1.md"


def prepare(out):
    """Lock prior source membership and scorer before collecting or joining outcomes."""
    lm = verify(LEVEL)
    if read(LEVEL / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete level connection required")
    sm = read(SOURCE / "manifest.json")
    material = SOURCE / "private_material.json"
    if sha(material) != sm["material_sha256"]:
        raise ValueError("S9 source material changed")
    rows = read(material)
    if len(rows) != 277 or len({r["author_id"] for r in rows}) != 277:
        raise ValueError("Complete frozen S9 source panel required")
    previous = {r["author_id"] for r in read(FULL / "private_inputs.json")}
    if previous & {r["author_id"] for r in rows}:
        raise ValueError("Source authors overlap construction references")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    (out / "private_pilot").mkdir()
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/level_response.py",
        ROOT / "grain_research/vector_mechanisms/prefix.py",
        ROOT / "scripts/run_grain_level_response.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_PREFIX_SOURCE_CONNECTION_V1",
            "authors": 277,
            "query_counts": sm["query_counts"],
            "views": 831,
            "pilot_prefixes": 6,
            "max_prefix_calls": 837,
            "max_seconds": 1200,
            "model_sha256": lm["model_sha256"],
            "source_material_sha256": sha(material),
            "source_manifest_sha256": sha(SOURCE / "manifest.json"),
            "reference_sha256": lm["reference_sha256"],
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "outcomes_used_for_scoring": False,
            "source_outcome_sha256": sha(CONNECTION / "private_predictions.csv"),
            "reference_overlap": 0,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed_authors": 0},
    )
    print({"status": "PREPARED", "authors": 277, "max_prefixes": 837})


def inputs(out):
    """Bind resumes to scorer, model, material and already-observed source outcomes."""
    m = read(out / "manifest.json")
    for p, digest in (
        (SOURCE / "private_material.json", m["source_material_sha256"]),
        (LEVEL / "reference.npz", m["reference_sha256"]),
        (CONNECTION / "private_predictions.csv", m["source_outcome_sha256"]),
    ):
        if sha(p) != digest:
            raise ValueError("Frozen input changed")
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        reference = dict(a)
    return m, reference, read(SOURCE / "private_material.json")


def run(out):
    """Validate early exit on six saved cells, then collect at most 831 source prefixes."""
    m, reference, material = inputs(out)
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete, no calls")
        return
    for p, digest in m["code_hashes"].items():
        if sha(ROOT / p) != digest:
            raise ValueError("Frozen implementation changed")
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Checkpoint changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    current, old = observer.manifest(), read(LEVEL / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if current[key] != old[key]:
            raise ValueError(f"Observer changed: {key}")
    write(out / "observer.json", current)
    start, completed = time.monotonic(), 0

    def status(state):
        """Write finite cost and outcome-blind liveness markers."""
        v = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_authors": completed,
            "planned_authors": 277,
            "suffix_calls": 0,
            "gradient_calls": 0,
            "labels_read": False,
        }
        write(out / "run_status.json", v)
        print(v, flush=True)

    try:
        for d in read(LEVEL / "private_design.json")[:6]:
            stem = f"unit_{d['index']:02d}_{d['query']}"
            destination = out / "private_pilot" / f"{stem}.json"
            if destination.exists():
                if read(destination)["status"] != "PASS":
                    raise ValueError("Previous pilot failed; do not bypass")
                continue
            cache, trace = capture_prefix(observer, d["full_ids"])
            with np.load(
                LEVEL / "private_packets" / f"{stem}.npz", allow_pickle=False
            ) as a:
                error = float(np.max(np.abs(cache["boundary"] - a["full_state"])))
            write(
                destination,
                {
                    "status": "PASS"
                    if error <= 2e-5 and not trace["forbidden_calls"]
                    else "FAIL",
                    "max_state_error": error,
                    "trace": trace,
                    "source_sha256": sha(LEVEL / "private_packets" / f"{stem}.npz"),
                },
            )
            if error > 2e-5 or trace["forbidden_calls"]:
                raise ValueError("Prefix replay differs from complete observer")
        for author in material:
            path = out / "private_packets" / f"author_{author['index']:03d}.json"
            if path.exists():
                r = read(path)
                if (
                    r["author_id"] != author["author_id"]
                    or sha(path.with_suffix(".npz")) != r["arrays_sha256"]
                ):
                    raise ValueError("Cached source unit changed")
                completed += 1
                continue
            if (
                prior["calls"] + observer.calls + 3 > m["max_prefix_calls"]
                or prior["seconds"] + time.monotonic() - start > m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            q = author["query"]
            get = lambda key, query=q: reference[f"{query}__{key}"]
            zs, hs, views = [], [], []
            for i, view in enumerate(author["views"]):
                _, _, encoded, mask = question_mask(
                    observer.tokenizer, view["texts"], q
                )
                ids = encoded["input_ids"]
                if material_hash(ids) != view["prompt_sha256"]:
                    raise ValueError("Source native prompt differs")
                if not np.array_equal(
                    np.asarray(ids)[mask],
                    read(LEVEL / "manifest.json")["question_tokens"][q],
                ):
                    raise ValueError("Query tokens differ from frozen matrix")
                cache, trace = capture_prefix(observer, ids)
                z, h = cache["boundary"][mask], cache["boundary"][-1]
                parts = affine_parts(z, h, get("Z"), get("h"), get("G"), get("g"))
                views.append(
                    {
                        "view": i,
                        **{
                            name: float(get("b")) + value
                            for name, value in parts.items()
                        },
                        "trace": trace,
                    }
                )
                zs.append(z)
                hs.append(h)
            np.savez_compressed(
                path.with_suffix(".npz"), P=np.asarray(zs), Q=np.asarray(hs)
            )
            write(
                path,
                {
                    "author_id": author["author_id"],
                    "query": q,
                    "index": author["index"],
                    "views": views,
                    "arrays_sha256": sha(path.with_suffix(".npz")),
                },
            )
            completed += 1
            if completed % 16 == 0:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(out):
    """Freeze all author predictions, then join exposed source outcomes without fitting."""
    _, reference, material = inputs(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Collect all fixed source scores before outcome join")
    rows, per_view = [], []
    for a in material:
        path = out / "private_packets" / f"author_{a['index']:03d}.json"
        r = read(path)
        if (
            r["author_id"] != a["author_id"]
            or sha(path.with_suffix(".npz")) != r["arrays_sha256"]
        ):
            raise ValueError("Source score archive changed")
        old = read(SOURCE / "private_packets" / path.name)
        if old["author_id"] != a["author_id"]:
            raise ValueError("S9 teacher join differs")
        value = {
            "author_id": a["author_id"],
            "query": a["query"],
            "student": a["student"],
            "old_teacher": float(np.mean([v["old_teacher"] for v in a["views"]])),
            "native": float(np.mean([v["new_teacher"] for v in old["views"]])),
        }
        value.update(
            {
                key: float(np.mean([v[key] for v in r["views"]]))
                for key in ("P", "Q", "PQ")
            }
        )
        rows.append(value)
        for v in r["views"]:
            per_view.append(
                {
                    "author_id": a["author_id"],
                    "query": a["query"],
                    "view": v["view"],
                    "PQ": v["PQ"],
                }
            )
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "private_scores_before_outcomes.csv", index=False)
    write(
        out / "scores_frozen.json",
        {
            "sha256": sha(out / "private_scores_before_outcomes.csv"),
            "outcomes_used": False,
        },
    )
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    joined = scores.merge(labels, on=["author_id", "query"], validate="one_to_one")
    if len(joined) != 277:
        raise ValueError("Missing source labels")
    metrics, fidelity, accounting, intervals, stability = [], [], [], [], []
    for q, group in joined.groupby("query"):
        y, native, pred = [group[k].to_numpy() for k in ("y_true", "native", "PQ")]
        for method in ("native", "old_teacher", "student", "P", "Q", "PQ"):
            metrics.append(
                {
                    "query": q,
                    "method": method,
                    **scalar_metrics(y, group[method].to_numpy(), q == "TF_cont"),
                }
            )
        for method in ("P", "Q", "PQ"):
            fidelity.append(
                {
                    "query": q,
                    "method": method,
                    **paired_quality(
                        native, group[method].to_numpy(), float(reference[f"{q}__b"])
                    ),
                }
            )
        for name, values in (
            ("native", native),
            ("PQ", pred),
            ("native_minus_PQ", native - pred),
            ("PQ_minus_student", pred - group.student.to_numpy()),
            ("student", group.student.to_numpy()),
        ):
            accounting.append(
                {
                    "query": q,
                    "component": name,
                    "covariance_y": covariance(y, values),
                    "r_y": correlation(y, values),
                    "sd": float(np.std(values)),
                }
            )
        if not np.isclose(
            covariance(y, native),
            covariance(y, pred) + covariance(y, native - pred),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("Source covariance ledger differs")
        rng = np.random.default_rng(20260908)
        metric = "auc" if q == "TF_cont" else "pearson_r"
        samples, differences = [], []
        for _ in range(1000):
            indices = rng.integers(len(y), size=len(y))
            a = scalar_metrics(y[indices], pred[indices], q == "TF_cont").get(metric)
            b = scalar_metrics(y[indices], native[indices], q == "TF_cont").get(metric)
            if a is not None and b is not None:
                samples.append(a)
                differences.append(a - b)
        intervals.append(
            {
                "query": q,
                "metric": metric,
                "PQ_ci": np.quantile(samples, [0.025, 0.975]).tolist()
                if samples
                else None,
                "PQ_minus_native_ci": np.quantile(differences, [0.025, 0.975]).tolist()
                if samples
                else None,
            }
        )
        views = (
            pd.DataFrame(per_view)
            .query("query == @q")
            .pivot(index="author_id", columns="view", values="PQ")
        )
        for a, b in ((0, 1), (0, 2), (1, 2)):
            stability.append(
                {
                    "query": q,
                    "views": f"{a}-{b}",
                    "r": correlation(views[a].to_numpy(), views[b].to_numpy()),
                }
            )
    joined.to_csv(out / "private_outcome_connection.csv", index=False)
    for name, value in (
        ("metrics", metrics),
        ("teacher_fidelity", fidelity),
        ("covariance_account", accounting),
        ("paired_intervals", intervals),
        ("view_stability", stability),
    ):
        pd.DataFrame(value).to_csv(out / f"{name}.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "scope": "exposed source-only no-fit connection; not a new benchmark",
            "cost": read(out / "run_status.json"),
            "metrics": metrics,
            "teacher_fidelity": fidelity,
            "covariance_account": accounting,
            "paired_intervals": intervals,
            "view_stability": stability,
            "predictions_sha256": sha(out / "private_scores_before_outcomes.csv"),
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


def main():
    """Separate preparation, collection and evaluation while preserving private outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "prefix_source_connection_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private output must remain under ignored results")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.phase](out)


if __name__ == "__main__":
    main()
