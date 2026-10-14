#!/usr/bin/env python3
"""Collect and replay600 fixed behavioral probes of25 source-referenced functions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.narrow_evidence import (
    build_worlds,
    paired_response_tables,
    reader_links,
    response_quality,
)
from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    digest,
    load_functions,
    read_json,
    write_json,
)
from scripts.run_grain_narrow_functions import reference_view
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_upstream_routes import collector_guard

REFERENCE = BASE / "narrow_functions_m30b_v1"
PLAN = ROOT / "docs/GRAIN_NARROW_EVIDENCE_ACTION_PLAN_V1.md"
SPEC = ROOT / "configs/grain_narrow_evidence_v1.json"
DEFAULT = BASE / "all_plan_closure_v1/narrow_actions_r1"
CODE = (
    "scripts/run_grain_narrow_actions.py",
    "grain_research/vector_mechanisms/narrow_evidence.py",
    "scripts/run_grain_narrow_functions.py",
    "scripts/run_grain_navigation_baseline.py",
    "grain_research/vector_mechanisms/functional_profiles.py",
    "grain_research/vector_mechanisms/full_state_edit.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/navigation.py",
)


def verify(out):
    """Check frozen inputs, reference coordinates and implementation before use."""
    manifest = read_json(out / "manifest.json")
    for name, expected in manifest["inputs"].items():
        if digest(out / name) != expected:
            raise ValueError(f"Input changed: {name}")
    for name, expected in manifest["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source changed: {name}")
    if (
        digest(PLAN) != manifest["plan_sha256"]
        or {name: digest(ROOT / name) for name in CODE} != manifest["code_sha256"]
    ):
        raise ValueError("Frozen plan or implementation changed")
    return manifest


def prepare(out):
    """Freeze authored worlds and exact old question rows before model loading."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Use a fresh output packet")
    definitions = read_json(REFERENCE / "queries.json")
    spec = read_json(SPEC)
    old = read_json(REFERENCE / "manifest.json")
    freeze = read_json(REFERENCE / "reference_manifest.json")
    if digest(REFERENCE / "reference.npz") != freeze["sha256"]:
        raise ValueError("Old function reference changed")
    if digest(MODEL / "model.safetensors") != old["model_sha256"]:
        raise ValueError("Old model weights changed")
    worlds, links = build_worlds(definitions, spec), reader_links(definitions, spec)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design = []
    for world in worlds:
        for link in links:
            if world["target"] != link["target"]:
                continue
            query = link["query"]
            _, _, encoded, mask = reference_view(
                tokenizer, world["texts"], definitions[query]["question"]
            )
            question_ids = np.asarray(encoded["input_ids"])[mask].tolist()
            if question_ids != old["question_tokens"][query]:
                raise ValueError("Old query coordinates differ")
            design.append(
                {
                    "index": len(design),
                    "world_id": world["world_id"],
                    **link,
                    "input_ids": encoded["input_ids"],
                    "mask": mask.tolist(),
                    "alphabet": old["alphabet"],
                }
            )
    if len(worlds) != 200 or len(design) != 600 or len(definitions) != 25:
        raise ValueError("Unexpected frozen packet coverage")
    out.mkdir(parents=True)
    (out / "private_packets").mkdir()
    write_json(out / "private_worlds.json", worlds)
    write_json(out / "private_design.json", design)
    write_json(out / "reader_links.json", links)
    sources = [
        SPEC,
        REFERENCE / "queries.json",
        REFERENCE / "manifest.json",
        REFERENCE / "reference.npz",
        REFERENCE / "reference_manifest.json",
    ]
    write_json(
        out / "manifest.json",
        {
            "protocol": spec["protocol"],
            "queries": list(definitions),
            "worlds": len(worlds),
            "units": len(design),
            "links": len(links),
            "model_sha256": old["model_sha256"],
            "question_tokens": old["question_tokens"],
            "inputs": {
                n: digest(out / n)
                for n in (
                    "private_worlds.json",
                    "private_design.json",
                    "reader_links.json",
                )
            },
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "plan_sha256": digest(PLAN),
            "code_sha256": {n: digest(ROOT / n) for n in CODE},
            "human_labels_accessed": False,
            "meps_accessed": False,
            "neural_training": False,
            "material": "authored synthetic retrospective posts",
            "reference": "unchanged two-PANDORA-author M30B functions",
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "units": 0},
    )
    print(
        {
            "status": "PREPARED",
            "units": len(design),
            "max_tokens": max(len(d["input_ids"]) for d in design),
        },
        flush=True,
    )


def collect(out, seconds):
    """Resume only this bounded packet; atomic markers preserve complete units."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = verify(out)
    previous = read_json(out / "run_status.json")
    if previous["status"] == "COMPLETE":
        print(previous, flush=True)
        return
    if digest(MODEL / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Model changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    write_json(out / "observer.json", observer.manifest())
    start, complete = time.monotonic(), 0

    def status(label):
        result = {
            "status": label,
            "units": complete,
            "calls": previous["calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
        }
        write_json(out / "run_status.json", result)
        return result

    try:
        for d in read_json(out / "private_design.json"):
            path = out / "private_packets" / f"unit_{d['index']:04d}.json"
            identity = {k: d[k] for k in ("index", "world_id", "query")}
            if path.exists():
                metadata, _ = checked_arrays(path)
                if any(metadata[k] != v for k, v in identity.items()):
                    raise ValueError("Resume identity mismatch")
            else:
                if time.monotonic() - start >= seconds:
                    print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                    return
                state, _, native = capture_boundary(
                    observer, d["input_ids"], d["alphabet"]
                )
                mask = np.asarray(d["mask"], bool)
                save_unit(
                    path,
                    identity | {"native": native},
                    P=state[0, mask].cpu().numpy(),
                    Q=state[0, -1].cpu().numpy(),
                )
            complete += 1
            if complete % 50 == 0:
                print(status("CAPTURING"), flush=True)
        print(status("COMPLETE"), flush=True)
    except Exception as exc:
        status("ERROR")
        write_json(
            out / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise


def evaluate(out):
    """Recompute all scores from committed vectors, then paired functional actions."""
    manifest = verify(out)
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete the packet before evaluating")
    functions = load_functions(REFERENCE / "reference.npz", queries=manifest["queries"])
    worlds = {w["world_id"]: w for w in read_json(out / "private_worlds.json")}
    rows = []
    point_error = 0.0
    with np.load(REFERENCE / "reference.npz", allow_pickle=False) as coefficients:
        for d in read_json(out / "private_design.json"):
            meta, z = checked_arrays(
                out / "private_packets" / f"unit_{d['index']:04d}.json"
            )
            if any(meta[k] != d[k] for k in ("index", "world_id", "query")):
                raise ValueError("Scoring identity mismatch")
            w = worlds[d["world_id"]]
            base = {
                k: d[k] for k in ("index", "world_id", "query", "reader_role", "target")
            }
            base |= {k: w[k] for k in ("family", "pair", "polarity", "owner")}
            native = meta["native"]
            rows.append(
                base
                | {
                    "readout": "native",
                    "score": native["conditional_score"],
                    "candidate_mass": native["candidate_mass"],
                    "mean": np.nan,
                    "configuration": np.nan,
                    "final": np.nan,
                }
            )
            parts = functions[d["query"]].parts(z["P"], z["Q"])
            query = d["query"]
            c = {k: coefficients[f"{query}__{k}"] for k in ("Z", "h", "G", "g", "b")}
            direct = float(
                c["b"]
                + np.sum(c["G"] * (z["P"].astype(float) - c["Z"]))
                + c["g"] @ (z["Q"].astype(float) - c["h"])
            )
            point_error = max(point_error, abs(parts["score"] - direct))
            rows.append(base | {"readout": "affine", **parts, "candidate_mass": np.nan})
    scores = pd.DataFrame(rows)
    responses, attribution = paired_response_tables(scores)
    query_rows, scope_rows = [], []
    for identity, group in responses.groupby(
        ["target", "query", "reader_role", "owner"], sort=True
    ):
        query_rows.append(
            dict(
                zip(("target", "query", "reader_role", "owner"), identity, strict=True)
            )
            | response_quality(group.native, group.affine)
        )
    for identity, group in responses.groupby(["reader_role", "owner"], sort=True):
        scope_rows.append(
            dict(zip(("reader_role", "owner"), identity, strict=True))
            | response_quality(group.native, group.affine)
        )
    outputs = {
        "scores.csv": scores,
        "polarity_responses.csv": responses,
        "attribution_contrasts.csv": attribution,
        "query_summary.csv": pd.DataFrame(query_rows),
        "scope_summary.csv": pd.DataFrame(scope_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    component_error = float(
        np.max(
            np.abs(
                responses.affine
                - responses[["mean", "configuration", "final"]].sum(axis=1)
            )
        )
    )
    ownership_error = float(
        np.max(
            np.abs(
                attribution.affine
                - attribution[["mean", "configuration", "final"]].sum(axis=1)
            )
        )
    )
    audit = {
        "status": "PASS"
        if max(point_error, component_error, ownership_error) < 1e-10
        else "FAIL",
        "point_error_max": point_error,
        "response_sum_error_max": component_error,
        "ownership_sum_error_max": ownership_error,
        "units": len(scores) // 2,
        "scores": len(scores),
        "polarity_pairs": len(responses),
        "attribution_pairs": len(attribution),
        "unique_worlds": len(worlds),
    }
    if audit["status"] != "PASS" or audit["units"] != manifest["units"]:
        raise ValueError("Full-packet arithmetic failed")
    write_json(out / "audit.json", audit)
    target = responses[
        (responses.reader_role == "target") & (responses.owner == "self")
    ]
    own = attribution[attribution.reader_role == "target"]
    means = target.groupby("target")[["native", "affine"]].mean()
    summary = {
        "status": "CONTROLLED_PACKET_COMPLETE",
        "audit": audit,
        "target_self": response_quality(target.native, target.affine),
        "target_positive_mean_queries": {
            r: int((means[r] > 0).sum()) for r in ("native", "affine")
        },
        "target_attribution": response_quality(own.native, own.affine),
        "all_scopes": scope_rows,
        "native_candidate_mass_min": float(
            scores[scores.readout == "native"].candidate_mass.min()
        ),
        "native_candidate_mass_median": float(
            scores[scores.readout == "native"].candidate_mass.median()
        ),
        "outputs": {name: digest(out / name) for name in outputs},
        "scope": "authored directional tests; not natural-person criterion validity or full-prefix independence",
    }
    write_json(out / "summary.json", summary)
    print(
        {
            "status": summary["status"],
            "target_self": summary["target_self"],
            "audit": audit,
        },
        flush=True,
    )


def main():
    """Expose separate freeze, bounded local collection and numerical replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1200 or not args.out.resolve().is_relative_to(
        (BASE / "all_plan_closure_v1").resolve()
    ):
        parser.error("Use a bounded local all-plan packet")
    with collector_guard(args.out):
        if args.mode == "prepare":
            prepare(args.out)
        elif args.mode == "collect":
            collect(args.out, args.seconds)
        else:
            evaluate(args.out)


if __name__ == "__main__":
    main()
