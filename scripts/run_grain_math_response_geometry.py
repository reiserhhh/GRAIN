#!/usr/bin/env python3
"""HM-09A: fixed conditional operators, source geometry and cold relation action."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_joint_context import inputs, parse
from run_grain_math_local_roles import vectors
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json

from grain_research.construction import math_binding as B
from grain_research.construction import math_context_operator as C
from grain_research.construction import math_response_geometry as G

SPEC = ROOT / "configs/grain_math_response_geometry_v1.json"
DEFAULT = ROOT / "results/grain_math_response_v1/response_geometry_r1"
TARGETS = ("coordination_action", "verification_action")


def checked_manifest(folder):
    """Check every frozen dependency before and after executing this packet."""
    manifest = folder / "04/artifact_hashes.json"
    for name, digest in json.loads(manifest.read_text()).items():
        if sha(folder / name) != digest:
            raise ValueError("frozen dependency changed: " + name)
    return sha(manifest)


def prepare(out):
    """Freeze source law, materials and code before new parsings or embeddings."""
    if (out / "04/summary.json").exists():
        raise ValueError("completed packet frozen; choose a new output")
    spec = json.loads(SPEC.read_text())
    source, cached = ROOT / spec["source_packet"], ROOT / spec["cached_packet"]
    dependencies = {
        str(p.relative_to(ROOT)): checked_manifest(p) for p in (source, cached)
    }
    rows = G.cold_bank(spec, read_bank(source / "00/singles.jsonl"))
    if len(rows) != spec["new_materials"]:
        raise ValueError("cold material count differs")
    files = [
        Path(__file__),
        SPEC,
        ROOT / "grain_research/construction/math_response_geometry.py",
        ROOT / "grain_research/construction/math_context_operator.py",
        ROOT / "scripts/run_grain_math_joint_context.py",
        ROOT / "grain_research/construction/math_joint_context.py",
        ROOT / "grain_research/construction/math_observation_dag.py",
        ROOT / "grain_research/construction/math_local_roles.py",
        ROOT / "grain_research/source_event_graph.py",
        ROOT / "scripts/run_grain_math_response.py",
        ROOT / "scripts/run_grain_math_binding.py",
    ]
    signature = {str(p.relative_to(ROOT)): sha(p) for p in files}
    content = "".join(json.dumps(r) + "\n" for r in rows)
    path = out / "00/protocol_before_run.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if (
            prior["signature"] != signature
            or prior["dependencies"] != dependencies
            or (out / "00/documents.jsonl").read_text() != content
        ):
            raise ValueError("frozen protocol changed")
    else:
        for p in files:
            dest = out / "00/code" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        (out / "00/documents.jsonl").write_text(content)
        write_json(
            out / "00/reader_registry.json",
            json.loads((source / "00/reader_registry.json").read_text()),
        )
        write_json(
            path,
            {
                "spec": spec,
                "signature": signature,
                "dependencies": dependencies,
                "materials": len(rows),
                "cached_analysis_is_result_after": True,
                "cold_materials_frozen_before_calls": True,
                "source_fits": 0,
                "pending": [
                    "HM-09B multi-condition exposure",
                    "source uncertainty",
                    "real author external linkage",
                ],
            },
        )
    return spec, source, cached, rows


def models(source, reader):
    """Load both original models without refitting or pooling embedding spaces."""
    return [load_model(source / f"02/{reader}_{t}.npz")["auto_tensor"] for t in TARGETS]


def source_geometry(out, source):
    """Declare source-only empirical tangent law and freeze response conversion."""
    dest = out / "01A/source_geometry.json"
    if dest.exists():
        return json.loads(dest.read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_bank(source / "00/singles.jsonl") if r["role"] == "source"]
    actions, objects = (
        sorted({r["action"] for r in rows}),
        sorted({r["object"] for r in rows}),
    )
    lookup = {
        r["text"]: i for i, r in enumerate(read_bank(source / "00/material_bank.jsonl"))
    }
    summary, score_rows = {}, []
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        a = old[[lookup[f"Action: {s}."] for s in actions]]
        o = old[[lookup[f"Object: {s}."] for s in objects]]
        pair_indices = list(combinations(range(len(objects)), 2))
        delta = np.array([o[j] - o[i] for i, j in pair_indices])
        support, spectrum, tolerance = G.row_basis(o - o.mean(axis=0))
        gradients = [G.gradient(C.compile_tensor(m), a) for m in models(source, reader)]
        fields = [j @ delta.T for j in gradients]
        forward = G.source_transport(fields[0], fields[1])
        reverse = G.source_transport(fields[1], fields[0])
        summary[reader] = {
            "actions": len(actions),
            "objects": len(objects),
            "reference_deltas": len(delta),
            "support_rank": len(support),
            "support_tolerance": tolerance,
            "euclidean_principal_cosines": G.principal_cosines(*gradients).tolist(),
            "empirical_principal_cosines": G.principal_cosines(*fields).tolist(),
            "euclidean_field_overlap": G.overlap(*gradients),
            "empirical_field_overlap": G.overlap(*fields),
            "coordination_to_verification": forward,
            "verification_to_coordination": reverse,
        }
        np.savez_compressed(
            out / f"01A/{reader}_geometry.npz",
            support=support,
            support_spectrum=spectrum,
            actions=a,
            objects=o,
            source_deltas=delta,
            gradient_coordination=gradients[0],
            gradient_verification=gradients[1],
            reference_coordination=fields[0],
            reference_verification=fields[1],
        )
        for ai, action in enumerate(actions):
            for oi, (i, j) in enumerate(pair_indices):
                score_rows.append(
                    {
                        "reader": reader,
                        "action": action,
                        "before": objects[i],
                        "after": objects[j],
                        "coordination_response": float(fields[0][ai, oi]),
                        "verification_response": float(fields[1][ai, oi]),
                    }
                )
    write_csv(out / "01A/source_response_fields.csv", score_rows)
    write_json(dest, summary)
    return summary


def response_rows(reader, model_list, action, before, after, meta, support):
    """Evaluate both targets on exactly the same addressed inputs and displacement."""
    delta = after - before
    supported, unobserved = G.support_parts(delta, support)
    gradients = [G.gradient(C.compile_tensor(m), action) for m in model_list]
    shifts = [np.sum(j * delta, axis=1) for j in gradients]
    full = [
        B.score_kernel(m, action, after) - B.score_kernel(m, action, before)
        for m in model_list
    ]
    error = max(
        float(np.max(np.abs(x - y))) if len(x) else 0
        for x, y in zip(shifts, full, strict=True)
    )
    if error > 1e-9:
        raise ValueError("gradient does not reproduce actual kernel difference")
    result = []
    for i, info in enumerate(meta):
        row = {
            "reader": reader,
            **info,
            "delta_norm": float(np.linalg.norm(delta[i])),
            "source_supported_norm": float(np.linalg.norm(supported[i])),
            "source_unobserved_norm": float(np.linalg.norm(unobserved[i])),
            "coordination_response": float(shifts[0][i]),
            "verification_response": float(shifts[1][i]),
            "shared_response": float((shifts[0][i] + shifts[1][i]) / 2),
            "target_contrast": float((shifts[1][i] - shifts[0][i]) / 2),
        }
        for t, j in zip(TARGETS, gradients, strict=True):
            row[t + "_supported"] = float(j[i] @ supported[i])
            row[t + "_unobserved"] = float(j[i] @ unobserved[i])
        result.append(row)
    return result, error


def summarize_responses(rows, geometry):
    """Group actual action by declared material domain, without a rank/alpha sweep."""
    groups = defaultdict(list)
    for r in rows:
        if r["delta_norm"] > 1e-12:
            groups[r["reader"], r["domain"], r["view"]].append(r)
    summary = []
    for (reader, domain, view), rr in groups.items():
        x, y = [
            np.array([r[k] for r in rr])
            for k in ("coordination_response", "verification_response")
        ]
        summary.append(
            {
                "reader": reader,
                "domain": domain,
                "view": view,
                **G.overlap(x, y),
                "transport_coordination_to_verification": G.transport_metrics(
                    x, y, geometry[reader]["coordination_to_verification"]["beta"]
                ),
                "transport_verification_to_coordination": G.transport_metrics(
                    y, x, geometry[reader]["verification_to_coordination"]["beta"]
                ),
                "source_unobserved_delta_rms": float(
                    np.sqrt(np.mean([r["source_unobserved_norm"] ** 2 for r in rr]))
                ),
                **{
                    t + "_unobserved_response_rms": float(
                        np.sqrt(np.mean([r[t + "_unobserved"] ** 2 for r in rr]))
                    )
                    for t in TARGETS
                },
            }
        )
    return summary


def cached_geometry(out, source, cached, geometry):
    """Result-after HM-08 diagnostics; no new vectorization or cold-claim upgrade."""
    dest = out / "01A/cached_summary.json"
    if dest.exists():
        return json.loads(dest.read_text())
    bank = read_bank(cached / "00/documents.jsonl")
    indices = json.loads((cached / "01/observation_indices.json").read_text())
    rows, errors, random_rows = [], {}, []
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(cached / f"01B/embeddings/{reader}.npy").astype(float)
        support = np.load(out / f"01A/{reader}_geometry.npz")["support"]
        mm = models(source, reader)
        events = [(i, e) for i, idx in enumerate(indices) for e in idx["observed"]]
        action = vectors([e["action_ref"] for _, e in events], old, new)
        before = vectors([e["object_ref"] for _, e in events], old, new)
        for view, field in (("legal_joint", "legal_ref"), ("naive_joint", "naive_ref")):
            after = vectors([e[field] for _, e in events], old, new)
            meta = [
                {
                    "material_id": bank[i]["material_id"],
                    "node_id": e["node_id"],
                    "domain": bank[i]["target"],
                    "variant": bank[i]["variant"],
                    "view": view,
                }
                for i, e in events
            ]
            got, error = response_rows(reader, mm, action, before, after, meta, support)
            rows.extend(got)
            errors[reader + ":" + view] = error
            if view == "legal_joint":
                mask = np.linalg.norm(after - before, axis=1) > 1e-12
                aa, norm = action[mask], np.linalg.norm((after - before)[mask], axis=1)
                gradients = [G.gradient(C.compile_tensor(m), aa) for m in mm]
                rng = np.random.default_rng(9109)
                for draw in range(16):
                    random = rng.normal(size=aa.shape)
                    random *= (norm / np.linalg.norm(random, axis=1))[:, None]
                    shifts = [np.sum(j * random, axis=1) for j in gradients]
                    random_rows.append(
                        {
                            "reader": reader,
                            "draw": draw,
                            "law": "isotropic norm-matched; not natural language",
                            **G.overlap(*shifts),
                        }
                    )
    summary = {
        "result_after": True,
        "rows": len(rows),
        "max_error": max(errors.values()),
        "responses": summarize_responses(rows, geometry),
        "new_encodings": 0,
    }
    write_csv(out / "01A/cached_responses.csv", rows)
    write_csv(out / "01A/isotropic_orientation_control.csv", random_rows)
    write_json(dest, summary)
    return summary


def cold_evaluate(out, source, bank, indexed, geometry):
    """Test new relation inputs with both fixed target functions and own routing."""
    events, complete, lookup = [], [], {}
    for i, row in enumerate(bank):
        ee = indexed[i]["observed"]
        valid = len(ee) == 1 and ee[0]["status"] == "ok"
        complete.append(valid)
        if valid:
            events.append((i, ee[0]))
            lookup[
                (
                    row["target"],
                    row["family"],
                    row["action_target"],
                    row["action_index"],
                    row["variant"],
                    row["mode"],
                    row["own"],
                )
            ] = (i, ee[0])
    score_rows, responses, errors, gate_rows = [], [], {}, []
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        mm = models(source, reader)
        support = np.load(out / f"01A/{reader}_geometry.npz")["support"]
        actions = vectors([e["action_ref"] for _, e in events], old, new)
        for view, key in (("strict", "object_ref"), ("legal_joint", "legal_ref")):
            obj = vectors([e[key] for _, e in events], old, new)
            scores = [B.score_kernel(m, actions, obj) for m in mm]
            for j, (i, e) in enumerate(events):
                row = bank[i]
                own = e["actor"] == "self" and (
                    e["mode"] == "actual" or e["conceiver"] == "self"
                )
                for ti, target in enumerate(TARGETS):
                    expected = (
                        row["action_target"] == target
                        and row["target"] == target
                        and row["action_type"] == 1
                        and row["variant"] != "decoy"
                    )
                    k, shared = (
                        float(scores[ti][j]),
                        float((scores[0][j] + scores[1][j]) / 2),
                    )
                    score_rows.append(
                        {
                            "reader": reader,
                            "material_id": row["material_id"],
                            "target": target,
                            "view": view,
                            "domain": row["target"],
                            "variant": row["variant"],
                            "mode": row["mode"],
                            "own": row["own"],
                            "observed_actor": e["actor"],
                            "observed_conceiver": e["conceiver"],
                            "observed_mode": e["mode"],
                            "k": k,
                            "shared_k": shared,
                            "expected_relation": expected,
                            "target_gate": k > 0,
                            "blind_shared_gate": shared > 0,
                            "expected_own_relation": expected and row["own"] == 1,
                            "qualified_target_gate": k > 0 and own,
                            "qualified_shared_gate": shared > 0 and own,
                        }
                    )
            pairs, meta = [], []
            for key_tuple, (i, e0) in lookup.items():
                domain, family, at, ai, variant, mode, own = key_tuple
                if variant != "bare":
                    continue
                for after_variant in ("joined", "decoy"):
                    dest = lookup.get(
                        (domain, family, at, ai, after_variant, mode, own)
                    )
                    if dest is None:
                        continue
                    ni, e1 = dest
                    if e0["action_ref"] != e1["action_ref"]:
                        raise ValueError(
                            "observation changed action; object-only transport not applicable"
                        )
                    pairs.append((e0, e1))
                    meta.append(
                        {
                            "material_id": bank[i]["material_id"],
                            "after_id": bank[ni]["material_id"],
                            "domain": domain,
                            "variant": after_variant,
                            "view": view + ":bare_to_" + after_variant,
                            "mode": mode,
                            "own": own,
                        }
                    )
            aa = vectors([p[0]["action_ref"] for p in pairs], old, new)
            before = vectors(
                [
                    p[0]["object_ref" if view == "strict" else "legal_ref"]
                    for p in pairs
                ],
                old,
                new,
            )
            after = vectors(
                [
                    p[1]["object_ref" if view == "strict" else "legal_ref"]
                    for p in pairs
                ],
                old,
                new,
            )
            got, error = response_rows(reader, mm, aa, before, after, meta, support)
            responses.extend(got)
            errors[reader + ":" + view] = error
    groups = defaultdict(list)
    for r in score_rows:
        groups[r["reader"], r["target"], r["view"], r["domain"], r["variant"]].append(r)
    for key, rows in groups.items():
        gate_rows.append(
            {
                **dict(
                    zip(
                        ("reader", "target", "view", "domain", "variant"),
                        key,
                        strict=True,
                    )
                ),
                "n": len(rows),
                **{
                    name + "_errors": sum(r[name] != r[truth] for r in rows)
                    for name, truth in (
                        ("target_gate", "expected_relation"),
                        ("blind_shared_gate", "expected_relation"),
                        ("qualified_target_gate", "expected_own_relation"),
                        ("qualified_shared_gate", "expected_own_relation"),
                    )
                },
                "other_actor_qualified": sum(
                    r["own"] == -1 and r["qualified_target_gate"] for r in rows
                ),
            }
        )
    summary = {
        "materials": len(bank),
        "complete_documents": sum(complete),
        "unresolved_documents": [
            r["material_id"] for r, ok in zip(bank, complete, strict=True) if not ok
        ],
        "event_scores": len(score_rows),
        "response_rows": len(responses),
        "max_gradient_error": max(errors.values()),
        "responses": summarize_responses(responses, geometry),
        "gates": gate_rows,
        "zero_target_refits": True,
        "multi_condition_exposure_tested": False,
    }
    write_csv(out / "02/cold_readouts.csv", score_rows)
    write_csv(out / "02/cold_response_decomposition.csv", responses)
    write_csv(out / "03/gate_controls.csv", gate_rows)
    write_json(out / "03/cold_summary.json", summary)
    return summary


def main():
    """Execute finite stages with resumable caches and an immutable completion."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument(
        "--stage", choices=("prepare", "geometry", "cold", "all"), default="all"
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    start = time.monotonic()
    spec, source, cached, bank = prepare(args.out)
    if args.stage == "prepare":
        print("HM-09A protocol frozen; no calls", flush=True)
        return
    geometry = source_geometry(args.out, source)
    cached_summary = cached_geometry(args.out, source, cached, geometry)
    if args.stage == "geometry":
        print(
            json.dumps({"geometry": geometry, "cached_rows": cached_summary["rows"]}),
            flush=True,
        )
        return
    parsed = parse(args.out, bank)
    indexed = inputs(args.out, source, bank, parsed)
    for reader in MODELS:
        encode(args.out, reader, args.device)
    summary = cold_evaluate(args.out, source, bank, indexed, geometry)
    protocol = json.loads((args.out / "00/protocol_before_run.json").read_text())
    for relative, digest in protocol["signature"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError("code changed during execution")
    for relative, digest in protocol["dependencies"].items():
        if checked_manifest(ROOT / relative) != digest:
            raise ValueError("dependency manifest changed")
    final = {
        "packet": "HM-09A",
        "elapsed_seconds": time.monotonic() - start,
        "source_geometry": geometry,
        "cache_response_rows": cached_summary["rows"],
        "cold": summary,
        "source_target_fits": 0,
        "source_unit_conversions": 4,
        "new_neural_training": 0,
        "api_calls": 0,
        "private_reads": 0,
        "pending": spec["pending"],
    }
    write_json(args.out / "04/summary.json", final)
    write_json(
        args.out / "04/artifact_hashes.json",
        {
            str(p.relative_to(args.out)): sha(p)
            for p in sorted(args.out.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                "packet": "HM-09A",
                "complete": summary["complete_documents"],
                "responses": summary["response_rows"],
                "elapsed_seconds": final["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
