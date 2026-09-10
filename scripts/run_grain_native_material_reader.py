#!/usr/bin/env python3
"""NEXT30: cached causal-prefix audit and bounded native GRAIN tangent experiment."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.native_material_reader import (
    allocate_slices,
    common_prefix_length,
    cosine,
    direction,
    tangent_parts,
)
from scripts.run_grain_full_state_edit_connection import SOURCE, token_view
from scripts.run_grain_multiquery_sensitivity import QUERIES, conditional_torch
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_within_comment_channels import FROZEN, lexical_slices
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

OLD = BASE / "full_state_edit_connection_v1"
PLAN = ROOT / "docs/GRAIN_NATIVE_MATERIAL_READER_PLAN_V1.md"
HELPER = ROOT / "grain_research/vector_mechanisms/native_material_reader.py"
BUNDLE = FROZEN / "operator_bundles/whole_factor_bundle.json"
REPRESENTATION = FROZEN / "artifacts/common_source_comment_representation.joblib"
FC = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
EPS = 0.001


def verify(out):
    """Verify native inputs, frozen coordinates and current implementation hashes."""
    manifest = read(out / "manifest.json")
    for name, digest in manifest["hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Changed frozen dependency: " + name)
    for name, digest in manifest["prepared_files"].items():
        if sha(out / name) != digest:
            raise ValueError("Changed prepared coordinates: " + name)
    return manifest


def cached_unit(index, query):
    """Locate one previously exposed native state without inspecting labels."""
    return OLD / "private_packets" / f"unit_{index:02d}_{query}.json"


def prepare(out):
    """Bind old common materials; score lexical slices with no new fitting."""
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    started = time.monotonic()
    old_manifest = read(OLD / "manifest.json")
    if sha(OLD / "private_inputs.json") != old_manifest["input_sha256"]:
        raise ValueError("Old native design changed")
    if (
        sha(SOURCE / "private_material.json")
        != read(SOURCE / "manifest.json")["material_sha256"]
    ):
        raise ValueError("Old source material changed")
    if read(OLD / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete old native units required")
    authors = read(OLD / "private_inputs.json")
    if (
        len(authors) != 4
        or sorted(a["role"] for a in authors) != ["evaluation"] * 2 + ["reference"] * 2
    ):
        raise ValueError("Unchanged four-author reference/evaluation panel required")
    material = [r for r in read(SOURCE / "private_material.json") if r["view"] == 0]
    if len({r["author_id"] for r in material}) != len(material):
        raise ValueError("Ambiguous source author/view")
    material = {r["author_id"]: r for r in material}
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    audit, design, raw, meta = [], [], [], []
    positions, allocation_inputs, native_operators = {}, {}, {}
    files = [
        PLAN,
        HELPER,
        Path(__file__),
        BUNDLE,
        REPRESENTATION,
        OLD / "manifest.json",
        OLD / "private_inputs.json",
        OLD / "observer.json",
        SOURCE / "manifest.json",
        SOURCE / "private_material.json",
        MODEL / "config.json",
        MODEL / "tokenizer.json",
        MODEL / "tokenizer_config.json",
    ]
    full_states = {}
    for a in authors:
        index, r = a["index"], material[a["author_id"]]
        kept = [j for j in range(len(r["texts"])) if j not in r["e1"]]
        loaded = {}
        for q in QUERIES:
            path = cached_unit(index, q)
            p = read(path)
            if (
                p["author_id"] != a["author_id"]
                or sha(path.with_suffix(".npz")) != p["arrays_sha256"]
            ):
                raise ValueError("Native state identity/hash changed")
            files.extend([path, path.with_suffix(".npz")])
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                loaded[q] = {arm: z[arm + "_state"] for arm in ("full", "edited")}
        for arm, texts in (
            ("full", r["texts"]),
            ("edited", [r["texts"][j] for j in kept]),
        ):
            encoded, spans = {}, {}
            for q in QUERIES:
                _, spans[q], encoded[q] = token_view(tokenizer, texts, q)
                if encoded[q]["input_ids"] != a["queries"][q][arm + "_ids"]:
                    raise ValueError("Material/template differs from native cache")
                if loaded[q][arm].shape != (len(encoded[q]["input_ids"]), 1536):
                    raise ValueError("Native state shape changed")
            prefix = common_prefix_length([encoded[q]["input_ids"] for q in QUERIES])
            body_end = (
                max(
                    i
                    for i, (_, hi) in enumerate(encoded[QUERIES[0]]["offset_mapping"])
                    if hi <= spans[QUERIES[0]][-1][1]
                )
                + 1
            )
            if prefix < body_end or any(spans[q] != spans[QUERIES[0]] for q in QUERIES):
                raise ValueError("Query changes the material prefix")
            for qa, qb in itertools.combinations(QUERIES, 2):
                ha, hb = loaded[qa][arm], loaded[qb][arm]
                delta = ha[:prefix].astype(float) - hb[:prefix]
                audit.append(
                    {
                        "index": index,
                        "role": a["role"],
                        "arm": arm,
                        "query_a": qa,
                        "query_b": qb,
                        "common_prefix_tokens": prefix,
                        "body_end_token": body_end,
                        "prefix_max": float(abs(delta).max()),
                        "prefix_rms": float(np.sqrt(np.mean(delta**2))),
                        "final_state_rms": float(
                            np.sqrt(np.mean((ha[-1].astype(float) - hb[-1]) ** 2))
                        ),
                    }
                )
                if abs(delta).max() > 2e-5:
                    pd.DataFrame(audit).to_csv(
                        out / "private_prefix_audit.csv", index=False
                    )
                    raise ValueError("Prefix discrepancy exceeds declared bound")
            if arm == "full":
                full_states[index] = loaded[QUERIES[0]][arm]
                allocation_inputs[index] = (encoded[QUERIES[0]]["offset_mapping"], [])
                eligible = 0
                for j, text in enumerate(texts):
                    parts = lexical_slices(text)
                    if not parts:
                        continue
                    eligible += 1
                    lo = spans[QUERIES[0]][j][0]
                    for s, part in enumerate(parts):
                        key = f"a{index:02d}_c{j:02d}_s{s}"
                        raw.append({"author": key, "body": part})
                        meta.append(
                            {"user_id": key, "index": index, "comment": j, "slice": s}
                        )
                        allocation_inputs[index][1].append((lo, lo + len(part)))
                        lo += len(part)
                if eligible < 2:
                    raise ValueError(
                        "Fewer than two eligible comments; no dropping authors"
                    )
                design.append(
                    {
                        "index": index,
                        "author_id": a["author_id"],
                        "role": a["role"],
                        "eligible_comments": eligible,
                        "input_comments": len(texts),
                    }
                )
    frame = canonicalize_comments(
        pd.DataFrame(raw), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    bundle = read_factor_bundle(BUNDLE)
    _, _, scored = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=joblib.load(REPRESENTATION),
        bundle=bundle,
    )
    if len(scored) != len(meta) or not scored.score_status.eq("SCORED").all():
        raise ValueError("Every eligible slice must score")
    scored = pd.DataFrame(meta).merge(
        scored[["user_id", *FC]], on="user_id", validate="one_to_one"
    )
    arrays = {}
    for a in design:
        index = a["index"]
        g = (
            scored[scored["index"] == index]
            .sort_values(["comment", "slice"])[FC]
            .to_numpy(float)
        )
        offsets, spans = allocation_inputs[index]
        pos, token_g, weights = allocate_slices(offsets, spans, g)
        h = full_states[index][pos].astype(float)
        hc, gc = h - h.mean(0), token_g - token_g.mean(0)
        arrays[f"a{index}__positions"], arrays[f"a{index}__grain"] = pos, token_g
        native_operators[f"a{index}__operator"] = gc.T @ hc / len(h)
        a.update(
            selected_tokens=len(pos),
            allocated_slices=len(g),
            allocation_error=float(abs(weights.sum(1) - 1).max()),
        )
        positions[index] = pos
    np.savez_compressed(out / "private_coordinates.npz", **arrays)
    np.savez_compressed(out / "private_native_operators.npz", **native_operators)
    pd.DataFrame(audit).to_csv(out / "private_prefix_audit.csv", index=False)
    scored.to_csv(out / "private_slice_scores.csv", index=False)
    write(out / "private_design.json", design)
    files.extend(
        ROOT / p
        for p in [
            "grain_research/vector_mechanisms/direct_traits.py",
            "grain_research/vector_mechanisms/full_state_edit.py",
            "grain_research/vector_mechanisms/context_carrier.py",
            "scripts/run_grain_full_state_edit_connection.py",
            "scripts/run_grain_within_comment_channels.py",
            "scripts/run_grain_multiquery_sensitivity.py",
            "suica_core/v7_operator_controls.py",
            "suica_core/v7_psychometric.py",
        ]
    )
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_NATIVE_MATERIAL_READER_V1",
            "authors": 4,
            "queries": list(QUERIES),
            "units": 12,
            "max_calls": 111,
            "max_seconds": 900,
            "epsilon": EPS,
            "block": 16,
            "labels_read": False,
            "fits": 0,
            "model_sha256": old_manifest["model_sha256"],
            "hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "prepared_files": {
                n: sha(out / n)
                for n in [
                    "private_coordinates.npz",
                    "private_native_operators.npz",
                    "private_prefix_audit.csv",
                    "private_slice_scores.csv",
                    "private_design.json",
                ]
            },
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "units": 0},
    )
    write(
        out / "construction_summary.json",
        {
            "authors": 4,
            "queries": 3,
            "material_arms": 8,
            "prefix_pairs": len(audit),
            "prefix_max": max(r["prefix_max"] for r in audit),
            "selected_tokens": sum(a["selected_tokens"] for a in design),
            "slices": len(meta),
            "eligible_comments": sum(a["eligible_comments"] for a in design),
            "input_comments": sum(a["input_comments"] for a in design),
            "seconds": time.monotonic() - started,
            "new_neural_calls": 0,
            "fits": 0,
            "labels_read": False,
        },
    )
    print(read(out / "construction_summary.json"), flush=True)


def suffix_gradient(observer, state, kwargs, alphabet):
    """Differentiate the fixed suffix from its leaf state; no parameter gradients."""
    import torch

    observer.calls += 1
    leaf = state.detach().requires_grad_(True)
    with torch.enable_grad():
        h = leaf
        for layer in observer.model.model.layers[17:]:
            h = layer(h, **kwargs)
            h = h[0] if isinstance(h, tuple) else h
        logits = observer.model.lm_head(observer.model.model.norm(h[:, -1:]))[0, -1]
        score = conditional_torch(logits, alphabet)
        (grad,) = torch.autograd.grad(score, leaf)
    return float(score.detach().cpu()), grad[0].detach().float().cpu().numpy()


def collect(out):
    """Run exactly the declared local responses, resumably within total budget."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = verify(out)
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete; no calls")
        return
    if sha(MODEL / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Checkpoint changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now, old = observer.manifest(), read(OLD / "observer.json")
    for k in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if now[k] != old[k]:
            raise ValueError("Native observer changed: " + k)
    write(out / "observer.json", now)
    with np.load(out / "private_coordinates.npz", allow_pickle=False) as z:
        coordinates = dict(z)
    original = {a["index"]: a for a in read(OLD / "private_inputs.json")}
    start, completed = time.monotonic(), 0

    def status(state):
        """Persist cumulative actual calls including partially completed units."""
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "units": completed,
            "labels_read": False,
            "fits": 0,
        }
        write(out / "run_status.json", value)
        print(value, flush=True)

    try:
        for a in read(out / "private_design.json"):
            index = a["index"]
            pos, g = (
                coordinates[f"a{index}__positions"],
                coordinates[f"a{index}__grain"],
            )
            with np.load(
                cached_unit(index, QUERIES[0]).with_suffix(".npz"), allow_pickle=False
            ) as z:
                common_h = z["full_state"][pos].astype(float)
            for q in QUERIES:
                dst = out / "private_packets" / f"a{index:02d}_{q}.json"
                if dst.exists():
                    if sha(dst.with_suffix(".npz")) != read(dst)["arrays_sha256"]:
                        raise ValueError("Completed gradient changed")
                    completed += 1
                    continue
                budget = 10 if index == 0 else 9
                if (
                    prior["calls"] + observer.calls + budget > manifest["max_calls"]
                    or prior["seconds"] + time.monotonic() - start
                    > manifest["max_seconds"]
                ):
                    status("PAUSED_RESOURCE_BOUNDARY")
                    return
                path = cached_unit(index, q)
                with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                    full = z["full_state"]
                h = torch.tensor(full[None], device="mps")
                kw = cached_suffix_kwargs(observer.model, h)
                alphabet = original[index]["queries"][q]["alphabet"]
                score, jacobian = suffix_gradient(observer, h, kw, alphabet)
                replay = abs(score - read(path)["full"]["conditional_score"])
                if replay > 2e-6 or not np.isfinite(jacobian).all():
                    raise ValueError("Native score/gradient replay failed")
                repeat = None
                if index == 0:
                    score2, j2 = suffix_gradient(observer, h, kw, alphabet)
                    repeat = {
                        "score_error": abs(score2 - score),
                        "gradient_max": float(abs(jacobian - j2).max()),
                    }
                    del j2
                parts = tangent_parts(common_h, g, jacobian[pos])
                rows = []
                for k in range(4):
                    d, scale = direction(common_h, g, k)
                    values = []
                    for sign in (1, -1):
                        changed = h.clone()
                        changed[:, pos] += torch.tensor(
                            sign * EPS * d[None], device=h.device, dtype=h.dtype
                        )
                        values.append(
                            replay_suffix(observer, changed, kw, alphabet)[
                                "conditional_score"
                            ]
                        )
                        del changed
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite finite intervention")
                    observed = (values[0] - values[1]) / (2 * EPS)
                    rows.append(
                        {
                            "factor": f"whole:F{k + 1:02d}",
                            "total": float(parts["total"][k]),
                            "common": float(parts["common"][k]),
                            "localized": float(parts["local"][k]),
                            "scale": scale,
                            "finite_derivative": observed,
                            "positive_score": values[0],
                            "negative_score": values[1],
                            "finite_delta_error": abs(
                                EPS * (observed - parts["total"][k])
                            ),
                        }
                    )
                np.savez_compressed(
                    dst.with_suffix(".npz"),
                    body_gradient=jacobian[pos],
                    final_gradient=jacobian[-1],
                    reader=parts["reader"],
                )
                write(
                    dst,
                    {
                        "index": index,
                        "query": q,
                        "role": a["role"],
                        "score": score,
                        "score_replay_error": replay,
                        "gradient_replay": repeat,
                        "rows": rows,
                        "identity_error": parts["identity_error"],
                        "arrays_sha256": sha(dst.with_suffix(".npz")),
                    },
                )
                completed += 1
                del h, kw, jacobian
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def analyze(out, name):
    """Recompute exact decompositions; publish no identities or raw token fields."""
    verify(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete packet required")
    dest = out / name
    dest.mkdir(exist_ok=False)
    rows, pairs, jacobians = [], [], {}
    contraction_replay_max = 0.0
    with np.load(out / "private_coordinates.npz", allow_pickle=False) as z:
        coordinates = dict(z)
    for a in read(out / "private_design.json"):
        index = a["index"]
        pos, g = coordinates[f"a{index}__positions"], coordinates[f"a{index}__grain"]
        with np.load(
            cached_unit(index, QUERIES[0]).with_suffix(".npz"), allow_pickle=False
        ) as z:
            h = z["full_state"][pos].astype(float)
        for q in QUERIES:
            dst = out / "private_packets" / f"a{a['index']:02d}_{q}.json"
            p = read(dst)
            if sha(dst.with_suffix(".npz")) != p["arrays_sha256"]:
                raise ValueError("Gradient bytes changed")
            with np.load(dst.with_suffix(".npz"), allow_pickle=False) as z:
                jacobians[(a["index"], q)] = z["body_gradient"]
            for k, row in enumerate(p["rows"]):
                d, _ = direction(h, g, k)
                direct = float(np.sum(d * jacobians[(index, q)]))
                contraction_replay_max = max(
                    contraction_replay_max, abs(direct - row["total"])
                )
                rows.append({"index": a["index"], "role": a["role"], "query": q, **row})
        for qa, qb in itertools.combinations(QUERIES, 2):
            ja, jb = (jacobians[(a["index"], q)] for q in (qa, qb))
            pairs.append(
                {
                    "index": a["index"],
                    "query_a": qa,
                    "query_b": qb,
                    "gradient_cosine": cosine(ja, jb),
                    "mean_reader_cosine": cosine(ja.mean(0), jb.mean(0)),
                }
            )
    if contraction_replay_max > 1e-10:
        raise ValueError("Native tangent does not replay saved gradient contraction")
    frame, pairframe = pd.DataFrame(rows), pd.DataFrame(pairs)
    aggregate = []
    for q, g in frame.groupby("query", sort=True):
        norm = np.linalg.norm(g.total)
        aggregate.append(
            {
                "query": q,
                "authors": g["index"].nunique(),
                "directions": len(g),
                "total_rms": float(np.sqrt(np.mean(g.total**2))),
                "common_rms": float(np.sqrt(np.mean(g.common**2))),
                "localized_rms": float(np.sqrt(np.mean(g.localized**2))),
                "common_error_over_total": float(
                    np.linalg.norm(g.total - g.common) / norm
                )
                if norm
                else None,
                "finite_delta_error_max": float(g.finite_delta_error.max()),
                "finite_derivative_relative_l2_error": float(
                    np.linalg.norm(g.total - g.finite_derivative) / norm
                )
                if norm
                else None,
                "sign_agreement": float(
                    np.mean(np.sign(g.total) == np.sign(g.finite_derivative))
                ),
            }
        )
    frame.to_csv(dest / "private_tangents.csv", index=False)
    pairframe.to_csv(dest / "private_query_pairs.csv", index=False)
    pd.DataFrame(aggregate).to_csv(dest / "reader_summary.csv", index=False)
    pairframe.groupby(["query_a", "query_b"])[
        ["gradient_cosine", "mean_reader_cosine"]
    ].agg(["mean", "min", "max"]).to_csv(dest / "query_pair_summary.csv")
    frame.groupby(["query", "factor"])[
        ["total", "common", "localized", "finite_derivative"]
    ].agg(["mean", "std"]).to_csv(dest / "factor_summary.csv")
    checks = [read(p) for p in sorted((out / "private_packets").glob("*.json"))]
    summary = {
        "status": "COMPLETE",
        "units": len(checks),
        "directions": len(frame),
        "contraction_replay_max": contraction_replay_max,
        "score_replay_max": max(p["score_replay_error"] for p in checks),
        "identity_error_max": max(p["identity_error"] for p in checks),
        "gradient_replay_max": max(
            p["gradient_replay"]["gradient_max"] for p in checks if p["gradient_replay"]
        ),
        "finite_delta_error_max": float(frame.finite_delta_error.max()),
        "fits": 0,
        "labels_read": False,
        "calls": read(out / "run_status.json")["calls"],
        "files": {
            n: sha(dest / n)
            for n in [
                "private_tangents.csv",
                "private_query_pairs.csv",
                "reader_summary.csv",
                "query_pair_summary.csv",
                "factor_summary.csv",
            ]
        },
    }
    write(dest / "summary.json", summary)
    print(summary, flush=True)


def main():
    """Keep preparation, native calls and numerical analysis explicitly separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_native_material_reader_v1"
    )
    parser.add_argument("--name", default="analysis_r1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.name.replace("_", "").isalnum()
    ):
        raise ValueError("Use ignored results and simple analysis name")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    with threadpool_limits(limits=2):
        if args.phase == "analyze":
            analyze(out, args.name)
        else:
            {"prepare": prepare, "collect": collect}[args.phase](out)


if __name__ == "__main__":
    main()
