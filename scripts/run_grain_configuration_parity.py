#!/usr/bin/env python3
"""Measure one mirrored endpoint per cached source material to locate odd information."""

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

from grain_research.vector_mechanisms.configuration_parity import (
    paired_even_error,
    parity_components,
)
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_centered_adjoint_field import CELLS, SOURCE, verify
from scripts.run_grain_mean_null_path import mean_null_direction
from scripts.run_grain_prefix_source_connection import LEVEL
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import covariance

FIELD = BASE / "centered_adjoint_field_v1"
PLAN = ROOT / "docs/GRAIN_CONFIGURATION_PARITY_PLAN_V1.md"


def prepare(out):
    """Freeze the existing48-author panel and this single sign-reflection contrast."""
    parent = verify(FIELD)
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/configuration_parity.py",
        ROOT / "scripts/run_grain_mean_null_path.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONFIGURATION_PARITY_V1",
            "field_manifest_sha256": sha(FIELD / "manifest.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "model_sha256": parent["model_sha256"],
            "reference_sha256": parent["reference_sha256"],
            "source_outcome_sha256": parent["source_outcome_sha256"],
            "design_sha256": sha(FIELD / "private_design.json"),
            "max_calls": 148,
            "max_seconds": 600,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "authors": 0},
    )
    print(
        {
            "status": "PREPARED",
            "authors": 48,
            "new_endpoints": 144,
            "max_suffix_calls": 148,
        }
    )


def verify_plan(out):
    """Verify immutable implementation, source membership and fixed reference coordinates."""
    m = read(out / "manifest.json")
    for p, digest in [
        (FIELD / "manifest.json", m["field_manifest_sha256"]),
        (FIELD / "private_design.json", m["design_sha256"]),
        (LEVEL / "reference.npz", m["reference_sha256"]),
        *[(ROOT / p, d) for p, d in m["code_hashes"].items()],
    ]:
        if sha(p) != digest:
            raise ValueError("Frozen parity construction changed")
    return m


def run(out):
    """Replay two old endpoints per query and collect mirrored body states only."""
    m, parent = verify_plan(out), verify(FIELD)
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Parity collection already complete")
        return
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Observer model changed")
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        refs = dict(a)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now, before = observer.manifest(), read(FIELD / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if now[key] != before[key]:
            raise ValueError("Observer execution changed")
    write(out / "observer.json", now)
    start, count, seen = time.monotonic(), 0, set()
    device = next(observer.model.parameters()).device

    def status(value):
        """Persist completed author count and actual bounded suffix cost."""
        result = {
            "status": value,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "authors": count,
            "prefix_calls": 0,
            "gradient_calls": 0,
            "model_training": False,
        }
        write(out / "run_status.json", result)
        return result

    try:
        for row in read(FIELD / "private_design.json"):
            query = row["query"]
            dst = out / "private_packets" / f"author_{row['index']:03d}.json"
            if dst.exists():
                count += 1
                seen.add(query)
                continue
            if (
                prior["calls"] + observer.calls + 3 + 2 * int(query not in seen)
                > m["max_calls"]
                or prior["seconds"] + time.monotonic() - start > m["max_seconds"]
            ):
                print(status("PAUSED_RESOURCE_BOUNDARY"))
                return
            source = SOURCE / "private_packets" / dst.with_suffix(".npz").name
            cell = CELLS / "private_packets" / dst.name
            for p in (source, cell):
                if sha(p) != parent["sources"][str(p.relative_to(ROOT))]:
                    raise ValueError("Native state or factorial endpoints changed")
            cells, views = read(cell)["views"], []
            with np.load(source, allow_pickle=False) as a:
                for view in row["views"]:
                    i, groups = view["view"], view["groups"]
                    native = torch.tensor(a[f"view_{i}"][None], device=device)
                    pmask = np.asarray(view["p_mask"], bool)
                    native[:, pmask] = torch.as_tensor(
                        refs[f"{query}__Z"], device=device
                    )
                    native[:, -1] = torch.as_tensor(refs[f"{query}__h"], device=device)
                    collapsed = (
                        native
                        + mean_null_direction(
                            native[0], groups, list(range(len(groups)))
                        )[None]
                    )
                    body = np.concatenate(groups)
                    mirrored = native.clone()
                    mirrored[:, body] = 2 * collapsed[:, body] - native[:, body]
                    unaffected = np.ones(native.shape[1], bool)
                    unaffected[body] = False
                    if not torch.equal(mirrored[:, unaffected], native[:, unaffected]):
                        raise ValueError("Reflection changed query or non-body tokens")
                    mean_error = max(
                        float(
                            torch.max(
                                torch.abs(mirrored[0, g].mean(0) - native[0, g].mean(0))
                            ).cpu()
                        )
                        for g in groups
                    )
                    if mean_error > 3e-5:
                        raise ValueError("Reflection failed comment mean preservation")
                    kwargs = cached_suffix_kwargs(observer.model, native)
                    replay_error = None
                    if query not in seen:
                        fp = replay_suffix(observer, native, kwargs, view["alphabet"])[
                            "conditional_score"
                        ]
                        f0 = replay_suffix(
                            observer, collapsed, kwargs, view["alphabet"]
                        )["conditional_score"]
                        replay_error = max(
                            abs(fp - cells[i]["f10"]), abs(f0 - cells[i]["f00"])
                        )
                        if replay_error > 2e-6:
                            raise ValueError("Old fixed-query endpoints failed replay")
                        seen.add(query)
                    fm = replay_suffix(observer, mirrored, kwargs, view["alphabet"])[
                        "conditional_score"
                    ]
                    values = parity_components(cells[i]["f10"], fm, cells[i]["f00"])
                    views.append(
                        {
                            "view": i,
                            "f_positive": cells[i]["f10"],
                            "f_negative": fm,
                            "f_origin": cells[i]["f00"],
                            "mean_error": mean_error,
                            "replay_error": replay_error,
                            **{k: float(v) for k, v in values.items()},
                        }
                    )
            write(
                dst,
                {
                    "author_id": row["author_id"],
                    "query": query,
                    "role": row["role"],
                    "gradient_check": row["gradient_check"],
                    "views": views,
                },
            )
            count += 1
            print(status("RUNNING"), flush=True)
        print(status("COMPLETE"), flush=True)
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def analyze(out):
    """Freeze all contrasts, compare linear response, and only then join exposed outcomes."""
    m = verify_plan(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete parity endpoints required")
    rows, linear = [], []
    for design in read(FIELD / "private_design.json"):
        p = out / "private_packets" / f"author_{design['index']:03d}.json"
        packet = read(p)
        if (
            packet["author_id"] != design["author_id"]
            or packet["query"] != design["query"]
        ):
            raise ValueError("Parity author linkage changed")
        for v in packet["views"]:
            rows.append(
                {
                    "author_id": packet["author_id"],
                    "query": packet["query"],
                    "role": packet["role"],
                    **v,
                }
            )
        if design["gradient_check"]:
            node = FIELD / "private_nodes" / f"author_{design['index']:03d}_a000.json"
            gradient_file = node.with_suffix(".npz")
            if sha(gradient_file) != read(node)["arrays_sha256"]:
                raise ValueError("Cached origin gradient changed")
            with np.load(gradient_file, allow_pickle=False) as a:
                gradient = a["gradient"]
            with np.load(
                FIELD / "private_packets" / p.with_suffix(".npz").name,
                allow_pickle=False,
            ) as a:
                slope = float(np.sum(a["path_displacement"] * gradient))
            v = packet["views"][0]
            linear.append(
                {
                    "query": packet["query"],
                    "role": packet["role"],
                    "plus": v["plus"],
                    "odd": v["odd"],
                    "even": v["even"],
                    "linear": slope,
                    "beyond_linear": v["plus"] - slope,
                    "odd_beyond_linear": v["odd"] - slope,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 144 or np.max(np.abs(frame.plus - frame.odd - frame.even)) > 1e-12:
        raise ValueError("Complete exact parity identity required")
    frame.to_csv(out / "private_parity_before_outcomes.csv", index=False)
    pd.DataFrame(linear).to_csv(out / "linear_diagnostics.csv", index=False)
    write(
        out / "parity_frozen.json",
        {
            "sha256": sha(out / "private_parity_before_outcomes.csv"),
            "human_labels_used": False,
        },
    )
    means = frame.groupby(["author_id", "query", "role"], as_index=False).mean(
        numeric_only=True
    )
    prior = (
        pd.read_csv(
            BASE / "adjoint_action_space_v1/private_predictions_before_outcomes.csv",
            dtype={"author_id": str},
        )
        .groupby(["author_id", "query"], as_index=False)
        .mean(numeric_only=True)
    )
    means = means.merge(
        prior[["author_id", "query", "action_refit"]],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    transport = []
    for (query, role), g in means.groupby(["query", "role"]):
        transport.append(
            {
                "query": query,
                "role": role,
                "authors": len(g),
                "odd_rms": float(np.sqrt(np.mean(g.odd**2))),
                "even_rms": float(np.sqrt(np.mean(g.even**2))),
                "plus_rms": float(np.sqrt(np.mean(g.plus**2))),
                "odd_even_cov": covariance(g.odd.to_numpy(), g.even.to_numpy()),
                "reader_odd_r": float(np.corrcoef(g.action_refit, g.odd)[0, 1]),
                "reader_even_r": float(np.corrcoef(g.action_refit, g.even)[0, 1]),
                **paired_even_error(g.plus, g.minus, g.action_refit),
            }
        )
    pd.DataFrame(transport).to_csv(out / "mechanism_metrics.csv", index=False)
    source_path = SOURCE / "private_anchor_source_connection.csv"
    if sha(source_path) != m["source_outcome_sha256"]:
        raise ValueError("Source outcomes changed")
    source = pd.read_csv(source_path, dtype={"author_id": str})
    joined = means.merge(
        source[["author_id", "query", "y_true"]],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if len(joined) != 48:
        raise ValueError("Exact source48 required")
    results = []
    for query, all_query in joined.groupby("query"):
        for role in ("all", "reference", "evaluation"):
            g = all_query if role == "all" else all_query[all_query.role == role]
            covs = {}
            for method in ("plus", "odd", "even", "action_refit"):
                covs[method] = covariance(g.y_true.to_numpy(), g[method].to_numpy())
                results.append(
                    {
                        "query": query,
                        "role": role,
                        "method": method,
                        **scalar_metrics(g.y_true.to_numpy(), g[method].to_numpy()),
                        "cov_y": covs[method],
                    }
                )
            if abs(covs["plus"] - covs["odd"] - covs["even"]) > 1e-12:
                raise ValueError("Source covariance parity account failed")
    pd.DataFrame(results).to_csv(out / "source_metrics.csv", index=False)
    joined.to_csv(out / "private_source_connection.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "cost": read(out / "run_status.json"),
            "mechanism": transport,
            "source": results,
            "scope": "same exposed48-author panel; fixed internal sign-reflection, not natural opposite text",
        },
    )
    print(
        pd.DataFrame(results)[
            ["query", "role", "method", "pearson_r", "cov_y"]
        ].to_string(index=False)
    )


def main():
    """Prepare/run/analyze the bounded mirrored-endpoint mechanism contrast."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "analyze"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "configuration_parity_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private results must remain ignored")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "run": run, "analyze": analyze}[args.phase](out)


if __name__ == "__main__":
    main()
