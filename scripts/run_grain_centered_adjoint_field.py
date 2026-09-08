#!/usr/bin/env python3
"""Construct conditional token adjoints and transport a fixed reference field."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.centered_adjoint import (
    body_partition,
    field_account,
    fit_reference_field,
    integrate_nodes,
    read_field,
)
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.level_response import paired_quality
from grain_research.vector_mechanisms.query_functional import (
    path_state,
    region_gradient,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_conditional_baseline import SOURCE
from scripts.run_grain_level_response import question_mask
from scripts.run_grain_mean_null_path import mean_null_direction
from scripts.run_grain_multiquery_sensitivity import native_alphabet
from scripts.run_grain_prefix_source_connection import LEVEL
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import covariance

PARENT = BASE / "conditional_baseline_construction_v1"
CELLS = BASE / "configuration_query_factorial_v1"
PLAN = ROOT / "docs/GRAIN_CENTERED_ADJOINT_FIELD_PLAN_V1.md"


def prepare(out):
    """Freeze hash-selected references/checks and every transfer author's token layout."""
    inherited = read(PARENT / "manifest.json")
    if read(CELLS / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Completed M15 factorial required")
    parent_design = PARENT / "private_design.json"
    if sha(parent_design) != inherited["design_sha256"]:
        raise ValueError("M12 author selection changed")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    material, design, sources = read(parent_design), [], {}
    for query in ("TF_cont", "neuroticism"):
        rows = sorted(
            [r for r in material if r["query"] == query],
            key=lambda r: hashlib.sha256(
                f"GRAIN_CONDITIONAL_BASELINE_V1:{r['author_id']}".encode()
            ).hexdigest(),
        )
        if len(rows) != 24:
            raise ValueError("Frozen24 authors/query required")
        for order, row in enumerate(rows):
            path = SOURCE / "private_packets" / f"author_{row['index']:03d}.json"
            cell = CELLS / "private_packets" / path.name
            for p in (path, path.with_suffix(".npz")):
                if sha(p) != inherited["sources"][str(p.relative_to(ROOT))]:
                    raise ValueError("Cached native state or anchor changed")
                sources[str(p.relative_to(ROOT))] = sha(p)
            sources[str(cell.relative_to(ROOT))] = sha(cell)
            views = []
            for view, v in enumerate(row["views"]):
                _, spans, encoded, p_mask = question_mask(tokenizer, v["texts"], query)
                groups = [
                    [
                        j
                        for j, (lo, hi) in enumerate(encoded["offset_mapping"])
                        if hi > lo >= left and hi <= right
                    ]
                    for left, right in spans
                ]
                if p_mask[np.concatenate(groups)].any():
                    raise ValueError("E body overlaps query region")
                views.append(
                    {
                        "view": view,
                        "groups": groups,
                        "p_mask": p_mask.tolist(),
                        "alphabet": native_alphabet(query, tokenizer),
                    }
                )
            design.append(
                {
                    "index": row["index"],
                    "author_id": row["author_id"],
                    "query": query,
                    "role": "reference" if order < 4 else "evaluation",
                    "gradient_check": order < 8,
                    "hash_order": order,
                    "views": views,
                }
            )
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("private_nodes", "private_packets", "private_fields"):
        (out / directory).mkdir()
    write(out / "private_design.json", design)
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/centered_adjoint.py",
        ROOT / "grain_research/vector_mechanisms/query_functional.py",
        ROOT / "scripts/run_grain_mean_null_path.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CENTERED_ADJOINT_FIELD_V1",
            "design_sha256": sha(out / "private_design.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "sources": sources,
            "model_sha256": inherited["model_sha256"],
            "reference_sha256": inherited["reference_sha256"],
            "source_outcome_sha256": sha(
                SOURCE / "private_anchor_source_connection.csv"
            ),
            "max_pairs": 80,
            "max_seconds": 900,
            "rank": 64,
            "alpha": 1.0,
            "seed": 20260908,
            "fit_target": "integrated conditional model adjoints, not human labels",
            "references_per_query": 4,
            "gradient_units": 16,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "units": 0},
    )
    print(
        {
            "status": "PREPARED",
            "gradient_units": 16,
            "max_pairs": 80,
            "reference_authors": 8,
            "transfer_authors": 40,
        }
    )


def verify(out):
    """Validate construction artifacts without joining human outcomes."""
    m = read(out / "manifest.json")
    for path, digest in (
        (out / "private_design.json", m["design_sha256"]),
        (LEVEL / "reference.npz", m["reference_sha256"]),
        *[(ROOT / p, h) for p, h in m["code_hashes"].items()],
    ):
        if sha(path) != digest:
            raise ValueError("Frozen field construction changed")
    return m


def collect(out):
    """Integrate the entire E-body adjoint, preserving nodes at resource boundaries."""
    m, prior = verify(out), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Adjoint collection complete")
        return
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Model changed")
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        refs = dict(a)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now, old = observer.manifest(), read(SOURCE / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if now[key] != old[key]:
            raise ValueError("Observer changed")
    write(out / "observer.json", now)
    device, start, completed = (
        next(observer.model.parameters()).device,
        time.monotonic(),
        0,
    )

    def status(value):
        """Record actual forward/backward pairs and completed immutable units."""
        r = {
            "status": value,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "units": completed,
            "call_unit": "suffix forward/backward pair",
            "prefix_calls": 0,
            "parameter_training": False,
            "human_labels_read": False,
        }
        write(out / "run_status.json", r)
        return r

    try:
        for row in read(out / "private_design.json"):
            if not row["gradient_check"]:
                continue
            dst = out / "private_packets" / f"author_{row['index']:03d}.json"
            if dst.exists():
                if sha(dst.with_suffix(".npz")) != read(dst)["arrays_sha256"]:
                    raise ValueError("Completed adjoint packet changed")
                completed += 1
                continue
            source = SOURCE / "private_packets" / dst.with_suffix(".npz").name
            cells = CELLS / "private_packets" / dst.name
            for path in (source, cells):
                if sha(path) != m["sources"][str(path.relative_to(ROOT))]:
                    raise ValueError("Source state or target cells changed")
            query, view = row["query"], row["views"][0]
            with np.load(source, allow_pickle=False) as a:
                native = a["view_0"]
            mask, labels, body, centered = body_partition(native, view["groups"])
            p_mask = np.asarray(view["p_mask"], bool)
            state = torch.tensor(native[None], device=device)
            state[:, p_mask] = torch.as_tensor(
                refs[f"{query}__Z"], device=device, dtype=state.dtype
            )
            state[:, -1] = torch.as_tensor(
                refs[f"{query}__h"], device=device, dtype=state.dtype
            )
            origin = (
                state
                + mean_null_direction(
                    state[0], view["groups"], list(range(len(view["groups"])))
                )[None]
            )
            z0, z1 = origin[:, mask], state[:, mask]
            displacement = z1[0].cpu().numpy().astype(float) - z0[
                0
            ].cpu().numpy().astype(float)
            expected = read(cells)["views"][0]
            actual = expected["f10"] - expected["f00"]
            kwargs = cached_suffix_kwargs(observer.model, origin)
            gradients, scores = {}, {}
            nodes = [0.0, 0.5, 1.0]
            while nodes:
                t = nodes.pop(0)
                node = (
                    out
                    / "private_nodes"
                    / f"author_{row['index']:03d}_a{int(t * 100):03d}.json"
                )
                if node.exists():
                    saved = read(node)
                    if sha(node.with_suffix(".npz")) != saved["arrays_sha256"]:
                        raise ValueError("Saved adjoint node changed")
                    with np.load(node.with_suffix(".npz"), allow_pickle=False) as a:
                        gradient = a["gradient"]
                    score = saved["score"]
                else:
                    if (
                        prior["calls"] + observer.calls >= m["max_pairs"]
                        or prior["seconds"] + time.monotonic() - start
                        >= m["max_seconds"]
                    ):
                        print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                        return
                    score, gradient = region_gradient(
                        observer,
                        origin,
                        path_state(z0, z1, t),
                        mask,
                        kwargs,
                        view["alphabet"],
                    )
                    np.savez_compressed(node.with_suffix(".npz"), gradient=gradient)
                    write(
                        node,
                        {
                            "score": score,
                            "fraction": t,
                            "arrays_sha256": sha(node.with_suffix(".npz")),
                        },
                    )
                    status("RUNNING")
                gradients[t], scores[t] = gradient, score
                if set(gradients) == {0.0, 0.5, 1.0}:
                    preliminary = integrate_nodes(gradients)
                    gap = abs(float(np.sum(preliminary * displacement)) - actual)
                    if gap > max(1e-5, 0.05 * abs(actual)):
                        nodes.extend([0.25, 0.75])
            endpoint_error = max(
                abs(scores[0.0] - expected["f00"]), abs(scores[1.0] - expected["f10"])
            )
            if endpoint_error > 2e-6:
                raise ValueError("Adjoint endpoints differ from actual M15 cells")
            integrated = integrate_nodes(gradients)
            account = field_account(displacement, integrated, labels)
            gap = account["action"] - actual
            np.savez_compressed(
                dst.with_suffix(".npz"),
                integrated=integrated,
                displacement=centered,
                path_displacement=displacement,
                body=body,
                labels=labels,
            )
            write(
                dst,
                {
                    "author_id": row["author_id"],
                    "query": query,
                    "role": row["role"],
                    "arrays_sha256": sha(dst.with_suffix(".npz")),
                    "actual": actual,
                    "nodes": sorted(scores),
                    "endpoint_error": endpoint_error,
                    "gap": gap,
                    "numeric_integral_pass": abs(gap) <= max(1e-5, 0.05 * abs(actual)),
                    **account,
                    "parameter_gradients": any(
                        p.grad is not None for p in observer.model.parameters()
                    ),
                },
            )
            completed += 1
            print(status("RUNNING"), flush=True)
        print(status("COMPLETE"), flush=True)
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def fit(out):
    """Learn only the frozen-reference field and read all remaining cached source views."""
    m = verify(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete gradient bank required")
    design, outputs, diagnostics = read(out / "private_design.json"), [], []
    for query in ("TF_cont", "neuroticism"):
        packets = []
        for row in design:
            if row["query"] != query or not row["gradient_check"]:
                continue
            path = out / "private_packets" / f"author_{row['index']:03d}.json"
            packet = read(path)
            if (
                packet["author_id"] != row["author_id"]
                or sha(path.with_suffix(".npz")) != packet["arrays_sha256"]
            ):
                raise ValueError("Adjoint identity or matrices changed")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
                packets.append({**packet, **dict(a)})
            diagnostics.append(
                {
                    k: v
                    for k, v in packet.items()
                    if k not in ("author_id", "arrays_sha256")
                }
            )
        if len(packets) != 8 or sum(p["role"] == "reference" for p in packets) != 4:
            raise ValueError("Exact four-reference/four-check field required")
        model = fit_reference_field(
            packets, rank=m["rank"], alpha=m["alpha"], seed=m["seed"]
        )
        path = out / "private_fields" / f"{query}.npz"
        np.savez_compressed(path, **model)
        write(
            path.with_suffix(".json"),
            {
                "arrays_sha256": sha(path),
                "reference_authors": 4,
                "reference_view": 0,
                "target": "T times centered integrated adjoint",
                "human_labels_used": False,
            },
        )
        for row in (r for r in design if r["query"] == query):
            native_path = SOURCE / "private_packets" / f"author_{row['index']:03d}.npz"
            cell_path = (
                CELLS / "private_packets" / native_path.with_suffix(".json").name
            )
            for p in (native_path, cell_path):
                if sha(p) != m["sources"][str(p.relative_to(ROOT))]:
                    raise ValueError("Scoring states or target cells changed")
            cells, views = read(cell_path)["views"], []
            with np.load(native_path, allow_pickle=False) as a:
                for view in row["views"]:
                    i = view["view"]
                    _, labels, _, d = body_partition(a[f"view_{i}"], view["groups"])
                    prediction = read_field(d, labels, model)
                    views.append(
                        {
                            "author_id": row["author_id"],
                            "query": query,
                            "role": row["role"],
                            "view": i,
                            "actual": cells[i]["f10"] - cells[i]["f00"],
                            **prediction,
                        }
                    )
            outputs.extend(views)
    frame = pd.DataFrame(outputs)
    frame.to_csv(out / "private_predictions_before_outcomes.csv", index=False)
    write(
        out / "predictions_frozen.json",
        {
            "sha256": sha(out / "private_predictions_before_outcomes.csv"),
            "fit_target": "model adjoint field",
            "human_labels_used": False,
            "estimator_selection": "none",
        },
    )
    metrics = []
    means = frame.groupby(["author_id", "query", "role"], as_index=False).mean(
        numeric_only=True
    )
    for (query, role), g in means.groupby(["query", "role"]):
        for method in ("rank64_field", "isotropic_field", "constant_field"):
            metrics.append(
                {
                    "query": query,
                    "role": role,
                    "method": method,
                    **paired_quality(
                        g.actual.to_numpy(), g[method].to_numpy(), np.zeros(len(g))
                    ),
                }
            )
    pd.DataFrame(metrics).to_csv(out / "transport_metrics.csv", index=False)
    write(out / "gradient_diagnostics.json", diagnostics)
    write(
        out / "fit_summary.json",
        {
            "metrics": metrics,
            "numeric_integral_passes": sum(
                r["numeric_integral_pass"] for r in diagnostics
            ),
            "gradient_units": len(diagnostics),
            "new_neural_calls_for_fitting_or_transport": 0,
        },
    )
    print(
        pd.DataFrame(
            [
                {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                for r in metrics
            ]
        ).to_string(index=False)
    )


def evaluate(out):
    """Connect already-exposed outcomes after freezing, separating actual from estimated E."""
    m = verify(out)
    path = out / "private_predictions_before_outcomes.csv"
    if sha(path) != read(out / "predictions_frozen.json")["sha256"]:
        raise ValueError("Frozen field predictions changed")
    source_path = SOURCE / "private_anchor_source_connection.csv"
    if sha(source_path) != m["source_outcome_sha256"]:
        raise ValueError("Authorized exposed source linkage changed")
    scores = (
        pd.read_csv(path, dtype={"author_id": str})
        .groupby(["author_id", "query", "role"], as_index=False)
        .mean(numeric_only=True)
    )
    source = pd.read_csv(source_path, dtype={"author_id": str})
    scores = scores.merge(
        source[
            ["author_id", "query", "y_true", "native", "PQ", "configuration_response"]
        ],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if len(scores) != 48:
        raise ValueError("Complete source-only link required")
    rows = []
    for (query, role), g in scores.groupby(["query", "role"]):
        for method in ("actual", "rank64_field", "isotropic_field", "native", "PQ"):
            r = scalar_metrics(g.y_true.to_numpy(), g[method].to_numpy())
            r.update(
                query=query,
                role=role,
                method=method,
                cov_y=covariance(g.y_true.to_numpy(), g[method].to_numpy()),
            )
            rows.append(r)
    scores.to_csv(out / "private_source_connection.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "source_metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "scope": "4 reference/20 evaluation authors per query, exposed source panel; derivative-supervised field, not human-label fitting",
            "cost": read(out / "run_status.json"),
            "fit": read(out / "fit_summary.json"),
            "source_metrics": rows,
        },
    )
    print(
        pd.DataFrame(rows)[
            ["query", "role", "method", "authors", "pearson_r", "cov_y"]
        ].to_string(index=False)
    )


def main():
    """Execute bounded construction, fitting and then source linkage in separate phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "fit", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "centered_adjoint_field_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private output must remain under ignored results")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "collect": collect, "fit": fit, "evaluate": evaluate}[
        args.phase
    ](out)


if __name__ == "__main__":
    main()
