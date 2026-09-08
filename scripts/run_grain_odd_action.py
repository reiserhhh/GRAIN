#!/usr/bin/env python3
"""Construct and transport bidirectional odd actions from cached real-author states."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from grain_research.vector_mechanisms.centered_adjoint import (
    body_partition,
    center_groups,
)
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.level_response import paired_quality
from grain_research.vector_mechanisms.odd_action import (
    FAMILIES,
    bidirectional_account,
    fit_odd_fields,
    odd_token_readouts,
)
from grain_research.vector_mechanisms.query_functional import (
    path_state,
    region_gradient,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_centered_adjoint_field import SOURCE, verify
from scripts.run_grain_configuration_parity import anchor_coordinates
from scripts.run_grain_mean_null_path import mean_null_direction
from scripts.run_grain_prefix_source_connection import LEVEL
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import covariance

FIELD = BASE / "centered_adjoint_field_v1"
PARITY = BASE / "configuration_parity_v1_r2"
PLAN = ROOT / "docs/GRAIN_ODD_ACTION_CONSTRUCTION_PLAN_V1.md"
METHODS = tuple(
    f"{family}_{kind}" for family in FAMILIES for kind in ("field", "action")
)


def prepare(out):
    """Freeze the same reference/check authors, observer, and reusable positive nodes."""
    inherited = verify(FIELD)
    if read(PARITY / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete real parity endpoints required")
    design = read(FIELD / "private_design.json")
    inputs = [
        FIELD / "private_design.json",
        FIELD / "manifest.json",
        PARITY / "manifest.json",
    ]
    reusable = 0
    for row in design:
        inputs.append(PARITY / "private_packets" / f"author_{row['index']:03d}.json")
        if row["gradient_check"]:
            for t in (0, 25, 50, 75, 100):
                p = FIELD / "private_nodes" / f"author_{row['index']:03d}_a{t:03d}.json"
                if p.exists():
                    inputs.extend([p, p.with_suffix(".npz")])
                    reusable += 1
    for query in ("TF_cont", "neuroticism"):
        inputs.append(FIELD / "private_fields" / f"{query}.npz")
    if sum(r["gradient_check"] for r in design) != 16 or reusable != 74:
        raise ValueError("Frozen16-unit,74-node positive bank required")
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("private_nodes", "private_packets", "private_fields"):
        (out / directory).mkdir()
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/odd_action.py",
        ROOT / "grain_research/vector_mechanisms/query_functional.py",
        ROOT / "scripts/run_grain_configuration_parity.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_ODD_ACTION_V1",
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "reference_sha256": inherited["reference_sha256"],
            "model_sha256": inherited["model_sha256"],
            "source_outcome_sha256": inherited["source_outcome_sha256"],
            "max_pairs": 70,
            "max_seconds": 600,
            "field_alpha": 1.0,
            "action_alpha": 0.01,
            "reused_positive_nodes": reusable,
            "human_labels_in_fit": False,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "units": 0},
    )
    print(
        {
            "status": "PREPARED",
            "units": 16,
            "cached_positive_nodes": reusable,
            "max_new_pairs": 70,
        }
    )


def verify_plan(out):
    """Check frozen code and evidence, excluding outcome contents until evaluation."""
    m = read(out / "manifest.json")
    for path, digest in {**m["code_hashes"], **m["inputs"]}.items():
        if sha(ROOT / path) != digest:
            raise ValueError("Frozen bidirectional construction or input changed")
    if sha(LEVEL / "reference.npz") != m["reference_sha256"]:
        raise ValueError("Reference coordinates changed")
    return m


def collect(out):
    """Complete the negative path, with cached positive-node reuse and adaptive finite cost."""
    m, inherited = verify_plan(out), verify(FIELD)
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Bidirectional collection already complete")
        return
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Observer weights changed")
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        refs = dict(a)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now, old = observer.manifest(), read(FIELD / "observer.json")
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
            raise ValueError("Observer execution changed")
    write(out / "observer.json", now)
    start, completed = time.monotonic(), 0
    device = next(observer.model.parameters()).device

    def status(value):
        """Persist actual new forward/backward cost and completed units."""
        record = {
            "status": value,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "units": completed,
            "call_unit": "suffix forward/backward pair",
            "prefix_calls": 0,
            "model_training": False,
            "human_labels_read": False,
        }
        write(out / "run_status.json", record)
        return record

    try:
        for row in read(FIELD / "private_design.json"):
            if not row["gradient_check"]:
                continue
            dst = out / "private_packets" / f"author_{row['index']:03d}.json"
            if dst.exists():
                if sha(dst.with_suffix(".npz")) != read(dst)["arrays_sha256"]:
                    raise ValueError("Completed odd packet changed")
                completed += 1
                continue
            source = SOURCE / "private_packets" / dst.with_suffix(".npz").name
            if sha(source) != inherited["sources"][str(source.relative_to(ROOT))]:
                raise ValueError("Cached state changed")
            expected = read(PARITY / "private_packets" / dst.name)["views"][0]
            query, view = row["query"], row["views"][0]
            with np.load(source, allow_pickle=False) as a:
                h = a["view_0"]
            mask, labels, body, d = body_partition(h, view["groups"])
            state = anchor_coordinates(
                torch.tensor(h[None], device=device),
                np.asarray(view["p_mask"], bool),
                refs[f"{query}__Z"],
                refs[f"{query}__h"],
            )
            origin = (
                state
                + mean_null_direction(
                    state[0], view["groups"], list(range(len(view["groups"])))
                )[None]
            )
            z0, zp = origin[:, mask], state[:, mask]
            zn = 2 * z0 - zp
            path_d = zp[0].cpu().numpy().astype(float) - z0[0].cpu().numpy().astype(
                float
            )
            mirror_rounding = float(
                np.max(
                    np.abs(
                        (
                            zn[0].cpu().numpy().astype(float)
                            - z0[0].cpu().numpy().astype(float)
                        )
                        + path_d
                    )
                )
            )
            kwargs = cached_suffix_kwargs(observer.model, origin)
            gradients, scores = {"p": {}, "n": {}}, {"p": {}, "n": {}}
            nodes = [0.0, 0.5, 1.0]
            while nodes:
                t = nodes.pop(0)
                for sign in ("p", "n"):
                    old_node = (
                        FIELD
                        / "private_nodes"
                        / f"author_{row['index']:03d}_a{int(t * 100):03d}.json"
                    )
                    use_old = (sign == "p" or t == 0) and old_node.exists()
                    node = (
                        old_node
                        if use_old
                        else out
                        / "private_nodes"
                        / f"author_{row['index']:03d}_{sign}{int(t * 100):03d}.json"
                    )
                    if node.exists():
                        meta = read(node)
                        if sha(node.with_suffix(".npz")) != meta["arrays_sha256"]:
                            raise ValueError("Gradient node checksum changed")
                        with np.load(node.with_suffix(".npz"), allow_pickle=False) as a:
                            gradient = a["gradient"]
                        score = meta["score"]
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
                            path_state(z0, zp if sign == "p" else zn, t),
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
                                "sign": sign,
                                "arrays_sha256": sha(node.with_suffix(".npz")),
                            },
                        )
                        status("RUNNING")
                    gradients[sign][t], scores[sign][t] = gradient, score
                if sorted(gradients["p"]) == [0.0, 0.5, 1.0]:
                    account = bidirectional_account(
                        path_d, gradients["p"], gradients["n"], labels
                    )
                    if any(
                        abs(account[f"{part}_action"] - expected[part])
                        > max(1e-5, 0.05 * abs(expected[part]))
                        for part in ("odd", "even")
                    ):
                        nodes.extend([0.25, 0.75])
            account = bidirectional_account(
                path_d, gradients["p"], gradients["n"], labels
            )
            endpoint_error = max(
                abs(scores["p"][0] - expected["f_origin"]),
                abs(scores["p"][1] - expected["f_positive"]),
                abs(scores["n"][1] - expected["f_negative"]),
            )
            if endpoint_error > 2e-6:
                raise ValueError("Bidirectional endpoints differ from M19")
            arrays = {k: account[k] for k in ("integrated_even", "integrated_odd")}
            np.savez_compressed(
                dst.with_suffix(".npz"),
                displacement=d,
                path_displacement=path_d,
                body=body,
                labels=labels,
                **arrays,
            )
            meta = {
                "author_id": row["author_id"],
                "query": query,
                "role": row["role"],
                "arrays_sha256": sha(dst.with_suffix(".npz")),
                "nodes": sorted(gradients["p"]),
                "actual_odd": expected["odd"],
                "actual_even": expected["even"],
                "endpoint_error": endpoint_error,
                "mirror_rounding": mirror_rounding,
                **{k: v for k, v in account.items() if k not in arrays},
            }
            for part in ("odd", "even"):
                meta[f"{part}_gap"] = account[f"{part}_action"] - expected[part]
                meta[f"{part}_numeric_pass"] = abs(meta[f"{part}_gap"]) <= max(
                    1e-5, 0.05 * abs(expected[part])
                )
            write(dst, meta)
            completed += 1
            print(status("RUNNING"), flush=True)
        print(status("COMPLETE"), flush=True)
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def fit(out):
    """Learn odd functions on four references and transport them to all cached materials."""
    m, inherited = verify_plan(out), verify(FIELD)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete paired gradient bank required")
    design = read(FIELD / "private_design.json")
    results, diagnostics, integration = [], [], []
    parity_error = 0.0
    start = time.monotonic()
    for query in ("TF_cont", "neuroticism"):
        packets = []
        for row in design:
            if row["query"] != query or not row["gradient_check"]:
                continue
            p = out / "private_packets" / f"author_{row['index']:03d}.json"
            meta = read(p)
            if (
                meta["author_id"] != row["author_id"]
                or meta["role"] != row["role"]
                or sha(p.with_suffix(".npz")) != meta["arrays_sha256"]
            ):
                raise ValueError("Gradient reference/check identity changed")
            integration.append(
                {
                    k: v
                    for k, v in meta.items()
                    if k not in ("author_id", "arrays_sha256")
                }
            )
            with np.load(p.with_suffix(".npz"), allow_pickle=False) as a:
                packets.append({**meta, **dict(a)})
        if len(packets) != 8 or sum(p["role"] == "reference" for p in packets) != 4:
            raise ValueError("Exact four reference/four check bank required")
        with np.load(
            FIELD / "private_fields" / f"{query}.npz", allow_pickle=False
        ) as a:
            model = fit_odd_fields(
                packets,
                a["projection"],
                a["scale"],
                m["field_alpha"],
                m["action_alpha"],
            )
        np.savez_compressed(out / "private_fields" / f"{query}.npz", **model)
        for p in packets:
            d = center_groups(p["displacement"], p["labels"])
            target = np.sum(
                d * (len(d) * center_groups(p["integrated_even"], p["labels"])), axis=1
            )
            for method, values in odd_token_readouts(d, p["labels"], model).items():
                diagnostics.append(
                    {
                        "query": query,
                        "role": p["role"],
                        "method": method,
                        "token_action_mse": float(np.mean((values - target) ** 2)),
                        "integral_error": float(values.mean() - target.mean()),
                        "endpoint_error": float(values.mean() - p["actual_odd"]),
                    }
                )
        for row in (r for r in design if r["query"] == query):
            p = SOURCE / "private_packets" / f"author_{row['index']:03d}.npz"
            if sha(p) != inherited["sources"][str(p.relative_to(ROOT))]:
                raise ValueError("Scoring state changed")
            endpoints = read(PARITY / "private_packets" / p.with_suffix(".json").name)[
                "views"
            ]
            with np.load(p, allow_pickle=False) as a:
                for view in row["views"]:
                    i = view["view"]
                    _, labels, _, d = body_partition(a[f"view_{i}"], view["groups"])
                    positive = odd_token_readouts(d, labels, model)
                    negative = odd_token_readouts(-d, labels, model)
                    parity_error = max(
                        parity_error,
                        max(
                            float(np.max(np.abs(positive[k] + negative[k])))
                            for k in positive
                        ),
                    )
                    results.append(
                        {
                            "author_id": row["author_id"],
                            "query": query,
                            "role": row["role"],
                            "view": i,
                            "actual_odd": endpoints[i]["odd"],
                            "actual_even": endpoints[i]["even"],
                            **{k: float(v.mean()) for k, v in positive.items()},
                        }
                    )
    if len(results) != 144 or parity_error > 1e-9:
        raise ValueError("Complete score matrix and reflection oddness required")
    frame = pd.DataFrame(results)
    frame.to_csv(out / "private_predictions_before_outcomes.csv", index=False)
    write(
        out / "predictions_frozen.json",
        {
            "sha256": sha(out / "private_predictions_before_outcomes.csv"),
            "human_labels_used": False,
        },
    )
    pd.DataFrame(diagnostics).to_csv(out / "token_diagnostics.csv", index=False)
    write(out / "integration_diagnostics.json", integration)
    means = frame.groupby(["author_id", "query", "role"], as_index=False).mean(
        numeric_only=True
    )
    metrics = []
    for (query, role), g in means.groupby(["query", "role"]):
        for method in METHODS:
            metrics.append(
                {
                    "query": query,
                    "role": role,
                    "method": method,
                    **paired_quality(
                        g.actual_odd.to_numpy(), g[method].to_numpy(), np.zeros(len(g))
                    ),
                }
            )
    pd.DataFrame(metrics).to_csv(out / "transport_metrics.csv", index=False)
    write(
        out / "fit_summary.json",
        {
            "transport": metrics,
            "parity_error_max": parity_error,
            "numeric_odd_passes": sum(p["odd_numeric_pass"] for p in integration),
            "numeric_even_passes": sum(p["even_numeric_pass"] for p in integration),
            "gradient_units": len(integration),
            "neural_calls": 0,
            "seconds": time.monotonic() - start,
        },
    )
    print(
        pd.DataFrame(metrics)[["query", "role", "method", "r", "mse_ratio"]].to_string(
            index=False
        )
    )


def evaluate(out):
    """Connect frozen odd functions and their exact source-covariance remainder."""
    m = verify_plan(out)
    path = out / "private_predictions_before_outcomes.csv"
    source = SOURCE / "private_anchor_source_connection.csv"
    if (
        sha(path) != read(out / "predictions_frozen.json")["sha256"]
        or sha(source) != m["source_outcome_sha256"]
    ):
        raise ValueError("Frozen predictions or source outcomes changed")
    frame = (
        pd.read_csv(path, dtype={"author_id": str})
        .groupby(["author_id", "query", "role"], as_index=False)
        .mean(numeric_only=True)
    )
    target = pd.read_csv(source, dtype={"author_id": str})
    frame = frame.merge(
        target[["author_id", "query", "y_true"]],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if len(frame) != 48:
        raise ValueError("Same48-author exposed source linkage required")
    rows = []
    for (query, role), g in frame.groupby(["query", "role"]):
        actual_cov = covariance(g.y_true.to_numpy(), g.actual_odd.to_numpy())
        for method in ("actual_odd", "actual_even", *METHODS):
            cov = covariance(g.y_true.to_numpy(), g[method].to_numpy())
            remainder = covariance(
                g.y_true.to_numpy(), (g.actual_odd - g[method]).to_numpy()
            )
            if abs(actual_cov - cov - remainder) > 1e-12:
                raise ValueError("Source odd covariance account failed")
            rows.append(
                {
                    "query": query,
                    "role": role,
                    "method": method,
                    **scalar_metrics(g.y_true.to_numpy(), g[method].to_numpy()),
                    "cov_y": cov,
                    "odd_remainder_cov_y": remainder,
                }
            )
    frame.to_csv(out / "private_source_connection.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "source_metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "cost": read(out / "run_status.json"),
            "fit": read(out / "fit_summary.json"),
            "source_metrics": rows,
            "scope": "same exposed48-author panel, reference4/query; model-derivative fitting only",
        },
    )
    print(
        pd.DataFrame(rows)[["query", "role", "method", "pearson_r", "cov_y"]].to_string(
            index=False
        )
    )


def main():
    """Execute bounded preparation, paired gradients, odd field fitting and source linkage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "fit", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "odd_action_construction_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private results must remain ignored")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    with threadpool_limits(limits=2):
        {"prepare": prepare, "collect": collect, "fit": fit, "evaluate": evaluate}[
            args.phase
        ](out)


if __name__ == "__main__":
    main()
