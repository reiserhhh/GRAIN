#!/usr/bin/env python3
"""NAV-02: bounded query-site attribution ranking followed by actual donor patches."""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    read_json,
    write_json,
)
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit


@contextmanager
def collector_guard(out):
    """Reject simultaneous writers to the same restartable upstream packet."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.parent / (out.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def plan_pairs(reference):
    """Fix construction/held-scene edit partners before any upstream result."""
    m = read_json(reference / "manifest.json")
    design = {
        (d["world_id"], d["query"]): d for d in read_json(reference / "design.json")
    }
    pairs = []
    for q in QUERIES:
        for split, scenes in (
            ("construction", [w.split("_")[0] for w in m["synthetic_references"][q]]),
            ("held_scene", ["s09", "s10", "s11"]),
        ):
            for scene in scenes:
                origin = f"{scene}_self_voluntary_r0"
                for axis, donor in (
                    ("actor", f"{scene}_other_voluntary_r0"),
                    ("opportunity", f"{scene}_self_assigned_r0"),
                ):
                    pairs.append(
                        {
                            "index": len(pairs),
                            "query": q,
                            "split": split,
                            "scene_id": scene,
                            "axis": axis,
                            "origin": design[(origin, q)]["index"],
                            "donor": design[(donor, q)]["index"],
                        }
                    )
    return pairs, {d["index"]: d for d in design.values()}


def run(reference, out, seconds=1200):
    """Never select candidate routes on held scenes or external human correlations."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver
    from grain_research.vector_mechanisms.upstream_routes import (
        assemble_patch,
        capture_differentiable_prefix,
        rank_route_candidates,
    )

    parent = read_json(reference / "manifest.json")
    ref_path = reference / "real_reference.npz"
    if digest(ref_path) != parent["real_reference_sha256"]:
        raise ValueError("Frozen reader changed")
    pairs, design = plan_pairs(reference)
    codes = (
        "scripts/run_grain_upstream_routes.py",
        "grain_research/vector_mechanisms/upstream_routes.py",
    )
    if not out.exists():
        out.mkdir(parents=True)
        (out / "cache").mkdir()
        (out / "patches").mkdir()
        write_json(
            out / "manifest.json",
            {
                "protocol": "NAV02_QUERY_SITE_V1",
                "pairs": pairs,
                "candidate_reference": "R_real",
                "max_candidate_layer": 12,
                "candidates_per_query": 8,
                "max_per_layer": 2,
                "native_checked_candidates": 2,
                "code_sha256": {p: digest(ROOT / p) for p in codes},
                "reference_sha256": digest(ref_path),
                "parent_manifest_sha256": digest(reference / "manifest.json"),
                "human_labels_used": False,
                "scope": "attention-head writes and MLP writes at aligned query slots; not all evidence-token routes",
            },
        )
        write_json(
            out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
        )
    m, prior = read_json(out / "manifest.json"), read_json(out / "run_status.json")
    if m["parent_manifest_sha256"] != digest(reference / "manifest.json") or m[
        "reference_sha256"
    ] != digest(ref_path):
        raise ValueError("Parent packet changed")
    for p, sha in m["code_sha256"].items():
        if digest(ROOT / p) != sha:
            raise ValueError("Frozen upstream code changed")
    if prior["status"] == "COMPLETE":
        print(prior, flush=True)
        return
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    if digest(MODEL / "model.safetensors") != parent["model_sha256"]:
        raise ValueError("Model changed")
    functions, start = load_functions(ref_path), time.monotonic()
    head_dim = observer.model.model.layers[0].self_attn.head_dim
    write_json(out / "observer.json", observer.manifest())

    def progress(state):
        """Save actual compute cost at restartable boundaries."""
        record = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
        }
        write_json(out / "run_status.json", record)
        return record

    def check_time():
        if time.monotonic() - start >= seconds:
            raise TimeoutError("bounded upstream packet")

    def state(index, need_grad=False):
        """Persist differentiable-run derivatives, then reuse only numeric caches."""
        d = design[index]
        path = out / "cache" / f"state_{index:04d}_{int(need_grad)}.json"
        if path.exists():
            meta, z = checked_arrays(path)
            if meta["index"] != index:
                raise ValueError("Upstream cache identity mismatch")
            return {
                "score": meta["score"],
                "sites": {k[6:]: v for k, v in z.items() if k.startswith("sites_")},
                "gradients": {k[5:]: v for k, v in z.items() if k.startswith("grad_")},
            }
        check_time()
        result = capture_differentiable_prefix(
            observer,
            d["input_ids"],
            d["mask"],
            functions[d["query"]],
            gradients=need_grad,
        )
        arrays = {"P": result["P"], "Q": result["Q"]}
        arrays.update({"sites_" + k: v for k, v in result["sites"].items()})
        arrays.update({"grad_" + k: v for k, v in result["gradients"].items()})
        if need_grad:
            arrays["input_gradient"] = result["input_gradient"]
        save_unit(
            path,
            {"index": index, "score": result["score"], "trace": result["trace"]},
            **arrays,
        )
        return result

    try:
        ranked_path = out / "selected_routes.json"
        if not ranked_path.exists():
            ranked = []
            for pair in pairs:
                if pair["split"] != "construction":
                    continue
                r, d = state(pair["origin"], True), state(pair["donor"])
                ranked.extend(
                    {"pair": pair["index"], "query": pair["query"], **c}
                    for c in rank_route_candidates(r, d, head_dim)
                )
                print(progress("RANKING_CONSTRUCTION"), flush=True)
            ranking = pd.DataFrame(ranked)
            ranking.to_csv(out / "edge_candidates.csv", index=False)
            ranking["absolute"] = abs(ranking.effect_estimate)
            average = ranking.groupby(
                ["query", "site", "head", "layer", "kind"], as_index=False
            ).absolute.mean()
            selections = {}
            for q in QUERIES:
                counts, selected = {}, []
                candidates = average[
                    (average["query"] == q)
                    & (average.layer <= m["max_candidate_layer"])
                ].sort_values(
                    ["absolute", "site", "head"], ascending=[False, True, True]
                )
                for row in candidates.to_dict("records"):
                    if counts.get(row["layer"], 0) >= m["max_per_layer"]:
                        continue
                    selected.append(
                        {k: row[k] for k in ("site", "head", "layer", "kind")}
                    )
                    counts[row["layer"]] = counts.get(row["layer"], 0) + 1
                    if len(selected) == 8:
                        break
                selections[q] = selected
            write_json(ranked_path, selections)
            write_json(
                out / "route_freeze.json",
                {
                    "sha256": digest(ranked_path),
                    "held_scenes_used_for_selection": False,
                },
            )
        elif digest(ranked_path) != read_json(out / "route_freeze.json")["sha256"]:
            raise ValueError("Selected routes changed")
        selections, rows = read_json(ranked_path), []
        for pair in pairs:
            r, d = state(pair["origin"]), state(pair["donor"])
            source = design[pair["origin"]]
            native = read_json(
                reference / "packets" / f"unit_{pair['origin']:04d}.json"
            )["native"]["conditional_score"]
            native_donor = read_json(
                reference / "packets" / f"unit_{pair['donor']:04d}.json"
            )["native"]["conditional_score"]
            selected = selections[pair["query"]]
            groups = [(f"single_{i}", [s]) for i, s in enumerate(selected)] + [
                ("top2", selected[:2]),
                ("all8", selected),
            ]
            for gi, (name, group) in enumerate(groups):
                path = out / "patches" / f"pair_{pair['index']:02d}_{name}.json"
                if path.exists():
                    rows.append(read_json(path))
                    continue
                check_time()
                alphabet = source["alphabet"] if gi < 2 or name == "all8" else None
                result = capture_differentiable_prefix(
                    observer,
                    source["input_ids"],
                    source["mask"],
                    functions[pair["query"]],
                    gradients=False,
                    patches=assemble_patch(r, d, group, head_dim),
                    alphabet=alphabet,
                )
                row = {
                    **pair,
                    "patch": name,
                    "routes": group,
                    "function_effect": result["score"] - r["score"],
                    "function_natural_delta": d["score"] - r["score"],
                    "native_effect": result["native"]["conditional_score"] - native
                    if result["native"]
                    else None,
                    "native_natural_delta": native_donor - native,
                }
                write_json(path, row)
                rows.append(row)
            print(progress("PATCHING_ROUTES"), flush=True)
        pd.DataFrame(rows).drop(columns="routes").to_csv(
            out / "patched_effects.csv", index=False
        )
        write_json(
            out / "summary.json",
            {
                "status": "COMPLETE_UPSTREAM_LOCALIZATION",
                "pairs": len(pairs),
                "construction_pairs": 12,
                "held_pairs": 18,
                "patches": len(rows),
                "independent_state_constructor": False,
                "query_slot_route_candidates_per_query": 8,
                "reference": "R_real",
            },
        )
        print(progress("COMPLETE"), flush=True)
    except TimeoutError:
        print(progress("PAUSED_RESOURCE_BOUNDARY"), flush=True)
    except Exception:
        progress("ERROR")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "navigation_v1/NAV-02/query_sites_r1"
    )
    parser.add_argument("--seconds", type=float, default=1200)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1200 or not args.output_dir.resolve().is_relative_to(
        (BASE / "navigation_v1").resolve()
    ):
        parser.error("Bounded local navigation output required")
    with collector_guard(args.output_dir.resolve()):
        run(args.reference_dir.resolve(), args.output_dir.resolve(), args.seconds)
