#!/usr/bin/env python3
"""S21: bounded matched-format functional query sharing on fixed source authors."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.cross_query_functional import (
    QUERIES,
    QUESTIONS,
    cosine_matrix,
    cross_readers,
    effect_quality,
    matched_prompt,
)
from grain_research.vector_mechanisms.direct_traits import candidates
from grain_research.vector_mechanisms.head_gradient import (
    projection_gradient,
    simpson_gradient,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_attention_pairing import capture_head, patch_head_score
from scripts.run_grain_functional_transfer import verify as verify_s20
from scripts.run_grain_multiquery_sensitivity import conditional_torch, token_groups
from scripts.run_grain_query_representation import pairing_vector
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

REFERENCE = tuple(range(12))
TRANSFER = tuple(range(24, 36))


def packet_path(out, role, index, query):
    """Keep complete author/query units in private, deterministic paths."""
    return out / "private_packets" / f"{role}_{index:03d}_{query}.json"


def save_packet(path, packet, arrays):
    """Write complete arrays before their hash-bearing metadata."""
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    write(path, {**packet, "arrays_sha256": sha(path.with_suffix(".npz"))})


def load_packet(path, author, query, role):
    """Reject partial, changed or wrong-author packets before resumption/analysis."""
    p = read(path)
    if (
        p["author_id"] != author["author_id"]
        or p["query"] != query
        or p["role"] != role
        or sha(path.with_suffix(".npz")) != p["arrays_sha256"]
    ):
        raise ValueError("Complete immutable author/query packet required")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if arrays["d"].shape != (1536,) or not all(
        np.isfinite(v).all() for v in arrays.values()
    ):
        raise ValueError("Finite native head arrays required")
    return p, arrays


def inherited(source, device):
    """Reuse the authoritative S19/S20 panel without any questionnaire join."""
    m, _s14, s12, material, _scores = verify_s20(source, device)
    if read(source / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete S20 source required")
    return Path(m["source"]), s12, material


def cache_neuroticism(out, source, s19, s12, material, role, index):
    """Reuse exact N gradients/effects instead of re-running the same model."""
    author, query = material[index], "neuroticism"
    path = packet_path(out, role, index, query)
    if path.exists():
        return load_packet(path, author, query, role)
    if role == "reference":
        original = s19 / "private_packets" / f"author_{index:03d}.json"
        old = read(original)
        if (
            old["author_id"] != author["author_id"]
            or sha(original.with_suffix(".npz")) != old["arrays_sha256"]
        ):
            raise ValueError("Cached N gradient mismatch")
        with np.load(original.with_suffix(".npz"), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        native, removed = old["scores"]["0.0"], old["scores"]["1.0"]
        detail = {
            "nodes": old["nodes"],
            "final_gap": old["final_gap"],
            "endpoint_error": old["endpoint_error"],
        }
    else:
        original = source / "private_packets" / f"author_{index:03d}.json"
        old = read(original)
        features = s12 / "private_packets" / f"author_{index:03d}.json"
        if (
            old["author_id"] != author["author_id"]
            or read(features)["author_id"] != author["author_id"]
        ):
            raise ValueError("Cached N transfer mismatch")
        with np.load(features.with_suffix(".npz"), allow_pickle=False) as z:
            arrays = {"d": z["kv_pairing"][0]}
        native = read(features)["views"][0]["score"]
        removed = native - old["effect"]
        detail = {"nodes": 0}
    save_packet(
        path,
        {
            "author_id": author["author_id"],
            "index": index,
            "role": role,
            "query": query,
            "effect": old["effect"],
            "native": native,
            "removed": removed,
            "reused": True,
            "source_packet_sha256": sha(original),
            **detail,
        },
        arrays,
    )
    return load_packet(path, author, query, role)


def prepare(args):
    """Freeze all question wording, author membership and prompt identities."""
    source, out = args.source.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private experiment outputs must stay in ignored results")
    s19, s12, material = inherited(source, args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    prompts, maximum = {}, 0
    for index in (*REFERENCE, *TRANSFER):
        author = material[index]
        payload = author["views"][0]
        native_prefix = None
        for query in QUERIES:
            ids, user, spans = matched_prompt(payload["texts"], query, tokenizer)
            groups = token_groups(
                SimpleNamespace(tokenizer=tokenizer), user, spans, ids
            )
            prefix = material_hash(ids[: max(groups[-1]) + 1])
            if native_prefix is not None and prefix != native_prefix:
                raise ValueError("Matched queries changed body token prefix")
            native_prefix = prefix
            digest = material_hash(ids)
            if query == "neuroticism" and digest != payload["prompt_sha256"]:
                raise ValueError("Cached N prompt mismatch")
            prompts[f"{index}:{query}"] = digest
            maximum = max(maximum, len(ids))
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/cross_query_functional.py",
        ROOT / "docs/GRAIN_CROSS_QUERY_FUNCTIONAL_S21_PLAN.md",
        ROOT / "scripts/run_grain_numeric_query_probe.py",
        ROOT / "scripts/run_grain_multiquery_sensitivity.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S21_CROSS_QUERY_FUNCTIONAL_V1",
            "source": str(source),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_summary_sha256": sha(source / "summary.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "questions": QUESTIONS,
            "queries": QUERIES,
            "device": args.device,
            "prompt_hashes": prompts,
            "max_prompt_tokens": maximum,
            "reference": REFERENCE,
            "transfer": TRANSFER,
            "membership_sha256": material_hash(
                [material[i]["author_id"] for i in (*REFERENCE, *TRANSFER)]
            ),
            "max_forwards": 288,
            "max_backwards": 180,
            "max_seconds": 600,
            "questionnaire_access": False,
            "score_values": (np.arange(10) / 9).tolist(),
        },
    )
    for index in REFERENCE:
        cache_neuroticism(out, source, s19, s12, material, "reference", index)
    print(
        {
            "status": "PREPARED",
            "reference": 12,
            "transfer": 12,
            "queries": 4,
            "max_tokens": maximum,
        },
        flush=True,
    )


def verify(out, device):
    """Revalidate the complete source chain and frozen local specification."""
    m = read(out / "manifest.json")
    source = Path(m["source"])
    if (
        m["device"] != device
        or sha(source / "manifest.json") != m["source_manifest_sha256"]
        or sha(source / "summary.json") != m["source_summary_sha256"]
        or any(sha(ROOT / p) != h for p, h in m["code_hashes"].items())
        or m["questions"] != QUESTIONS
        or m["reference"] != list(REFERENCE)
        or m["transfer"] != list(TRANSFER)
    ):
        raise ValueError("Frozen S21 protocol/source changed")
    s19, s12, material = inherited(source, device)
    if (
        material_hash([material[i]["author_id"] for i in (*REFERENCE, *TRANSFER)])
        != m["membership_sha256"]
    ):
        raise ValueError("Frozen S21 author membership changed")
    return m, source, s19, s12, material


def reference_arrays(out, material, count=12):
    """Load complete reference authors equally across every fixed query."""
    packets, gradients, displacements = [], [], []
    for index in REFERENCE[:count]:
        ps, gs, ds = [], [], []
        for query in QUERIES:
            p, a = load_packet(
                packet_path(out, "reference", index, query),
                material[index],
                query,
                "reference",
            )
            ps.append(p)
            gs.append(a["integrated"])
            ds.append(a["d"])
        packets.append(ps)
        gradients.append(gs)
        displacements.append(ds)
    return packets, np.asarray(gradients), np.asarray(displacements)


def freeze_readers(out, material):
    """Bind all four mean covectors before new transfer responses are inspected."""
    _ps, g, d = reference_arrays(out, material)
    references, _predictions = cross_readers(g, d)
    hashes = {
        p.name: sha(p)
        for p in sorted((out / "private_packets").glob("reference_*.json"))
    }
    path = out / "readers_frozen.json"
    array_path = out / "private_readers.npz"
    if path.exists():
        old = read(path)
        if old["packet_hashes"] != hashes or old["readers_sha256"] != sha(array_path):
            raise ValueError("Frozen functional reader changed")
        with np.load(array_path, allow_pickle=False) as z:
            if not np.array_equal(z["references"], references):
                raise ValueError("Frozen covectors changed")
    else:
        np.savez_compressed(array_path, references=references)
        write(
            path,
            {
                "packet_hashes": hashes,
                "readers_sha256": sha(array_path),
                "transfer_used": False,
            },
        )
    return references


def collect(
    observer, alphabet, material, index, query, role, expected_prompt, count_backward
):
    """Measure one native query and its declared head-removal path."""
    ids, user, spans = matched_prompt(
        material[index]["views"][0]["texts"], query, observer.tokenizer
    )
    if material_hash(ids) != expected_prompt:
        raise ValueError("Frozen prompt changed at collection")
    groups = token_groups(observer, user, spans, ids)
    native, cache = capture_head(observer, ids, alphabet)
    d, head_error = pairing_vector(cache, groups)
    h = cache["native_head"].reshape(-1)
    arrays = {"h": h, "d": d, "q": cache["q"].reshape(-1)}
    detail = {"head_error": head_error, "nodes": 0}
    if role == "reference":
        device = next(observer.model.parameters()).device
        tokens = torch.tensor([ids], device=device)
        scores, gradients = {}, {}

        def forward():
            observer.calls += 1
            logits = observer.model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            ).logits[0, -1]
            return conditional_torch(logits, alphabet)

        def node(t):
            donor = torch.tensor(h - t * d, dtype=torch.float32, device=device)
            scores[t], gradients[t] = projection_gradient(
                observer.model.model.layers[17].self_attn.o_proj, donor, forward
            )
            count_backward()

        for t in (0.0, 0.5, 1.0):
            node(t)
        removed = scores[1.0]
        effect = native - removed
        first = simpson_gradient(gradients)
        first_gap = float(first @ d - effect)
        if abs(first_gap) > max(1e-5, 0.05 * abs(effect)):
            for t in (0.25, 0.75):
                node(t)
        integrated = simpson_gradient(gradients)
        endpoint_error = abs(native - scores[0.0])
        if endpoint_error > 2e-5 or any(
            p.grad is not None or p.requires_grad for p in observer.model.parameters()
        ):
            raise ValueError("Native replay or frozen-parameter check failed")
        times = sorted(gradients)
        arrays.update(
            gradients=np.asarray([gradients[t] for t in times]),
            times=np.asarray(times),
            integrated=integrated,
            three_node_integral=first,
        )
        detail.update(
            nodes=len(times),
            three_node_gap=first_gap,
            final_gap=float(integrated @ d - effect),
            endpoint_error=endpoint_error,
            scores={str(t): scores[t] for t in times},
        )
    else:
        removed = patch_head_score(observer, ids, alphabet, h - d)
        effect = native - removed
    return {
        "author_id": material[index]["author_id"],
        "index": index,
        "role": role,
        "query": query,
        "native": native,
        "removed": removed,
        "effect": effect,
        "prompt_sha256": expected_prompt,
        "reused": False,
        **detail,
    }, arrays


def run(args):
    """Run a resumable reference pilot, freeze readers, then collect transfer12."""
    out = args.output_dir
    m, source, s19, s12, material = verify(out, args.device)
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"calls": 0, "backwards": 0, "seconds": 0}
    )
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    alphabet = candidates("neuroticism", observer.tokenizer)
    backwards, start = 0, time.monotonic()

    def count_backward():
        nonlocal backwards
        backwards += 1

    def status(name):
        state = {
            "status": name,
            "calls": previous["calls"] + observer.calls,
            "backwards": previous["backwards"] + backwards,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "reference_units": len(
                list((out / "private_packets").glob("reference_*.json"))
            ),
            "transfer_units": len(
                list((out / "private_packets").glob("transfer_*.json"))
            ),
            "labels_read": False,
        }
        write(out / "run_status.json", state)
        print(state, flush=True)

    try:
        for role, indices in (("reference", REFERENCE), ("transfer", TRANSFER)):
            if role == "transfer":
                freeze_readers(out, material)
            for ordinal, index in enumerate(indices):
                if role == "reference" and ordinal >= args.max_reference_authors:
                    status("PAUSED_PILOT_BOUNDARY")
                    return
                cache_neuroticism(out, source, s19, s12, material, role, index)
                for query in QUERIES[1:]:
                    path = packet_path(out, role, index, query)
                    if path.exists():
                        load_packet(path, material[index], query, role)
                        continue
                    nf, nb = (6, 5) if role == "reference" else (2, 0)
                    if (
                        previous["seconds"] + time.monotonic() - start
                        >= m["max_seconds"]
                        or previous["calls"] + observer.calls + nf > m["max_forwards"]
                        or previous["backwards"] + backwards + nb > m["max_backwards"]
                    ):
                        status("PAUSED_RESOURCE_BOUNDARY")
                        return
                    packet, arrays = collect(
                        observer,
                        alphabet,
                        material,
                        index,
                        query,
                        role,
                        m["prompt_hashes"][f"{index}:{query}"],
                        count_backward,
                    )
                    save_packet(path, packet, arrays)
                status("RUNNING")
        status("COMPLETE")
    except Exception as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def inspect(args):
    """Assess the fixed four-reference numerical/resource pilot without outcomes."""
    out = args.output_dir
    _m, _source, _s19, _s12, material = verify(out, args.device)
    ps, g, _d = reference_arrays(out, material, 4)
    means = g.mean(axis=0)
    summary = {
        "status": "PILOT",
        "authors": 4,
        "queries": QUERIES,
        "labels_read": False,
        "reference_cosines": cosine_matrix(means).tolist(),
        "mean_gradient_norms": np.linalg.norm(means, axis=1).tolist(),
        "max_completeness_gap": max(
            abs(p["final_gap"]) for author in ps for p in author
        ),
        "max_endpoint_error": max(p["endpoint_error"] for author in ps for p in author),
        "five_node_units": sum(p["nodes"] == 5 for author in ps for p in author),
        "cost": read(out / "run_status.json"),
    }
    path = out / "pilot_004.json"
    if path.exists():
        raise ValueError("Preserve recorded pilot")
    write(path, summary)
    print(summary, flush=True)


def intervals(effect, prediction, reference):
    """Retain constant conditions instead of fabricating correlation intervals."""
    if any(np.ptp(x) == 0 for x in (effect, prediction, reference)):
        return {
            k: None
            for k in (
                "student_lower",
                "student_upper",
                "delta",
                "delta_lower",
                "delta_upper",
            )
        }
    return paired_intervals(effect, prediction, reference, False)


def evaluate(args):
    """Report the full crossed response matrix without reading human outcomes."""
    out = args.output_dir
    _m, _source, _s19, _s12, material = verify(out, args.device)
    state = read(out / "run_status.json")
    if (
        state["status"] != "COMPLETE"
        or state["reference_units"] != 48
        or state["transfer_units"] != 48
    ):
        raise ValueError("Complete12+12 authors and four queries required")
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed evaluation")
    refs = freeze_readers(out, material)
    reference_packets, g, _rd = reference_arrays(out, material)
    effects, vectors = [], []
    for index in TRANSFER:
        es, ds = [], []
        for query in QUERIES:
            p, a = load_packet(
                packet_path(out, "transfer", index, query),
                material[index],
                query,
                "transfer",
            )
            es.append(p["effect"])
            ds.append(a["d"])
        effects.append(es)
        vectors.append(ds)
    effects, vectors = np.asarray(effects), np.asarray(vectors)
    _refs, predictions = cross_readers(g, vectors)
    rows, private = [], []
    for qi, query in enumerate(QUERIES):
        own = predictions[:, qi, qi]
        for ri, reader in enumerate(QUERIES):
            predicted = predictions[:, qi, ri]
            rows.append(
                {
                    "query": query,
                    "reader": reader,
                    **effect_quality(effects[:, qi], predicted),
                    **intervals(effects[:, qi], predicted, own),
                }
            )
        shared = predictions[:, qi].mean(axis=1)
        rows.append(
            {
                "query": query,
                "reader": "all_query_mean",
                **effect_quality(effects[:, qi], shared),
                **intervals(effects[:, qi], shared, own),
            }
        )
        for ui, index in enumerate(TRANSFER):
            private.append(
                {
                    "author_id": material[index]["author_id"],
                    "query": query,
                    "effect": effects[ui, qi],
                    **{
                        f"reader_{q}": predictions[ui, qi, j]
                        for j, q in enumerate(QUERIES)
                    },
                }
            )
    with threadpool_limits(limits=2):
        s = np.linalg.svd(g.reshape(-1, g.shape[-1]), compute_uv=False)
        sq = [np.linalg.svd(g[:, qi], compute_uv=False) for qi in range(4)]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "cross_reader_metrics.csv", index=False)
    pd.DataFrame(private).to_csv(out / "private_transfer_predictions.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "queries": QUERIES,
        "reference_authors": 12,
        "transfer_authors": 12,
        "labels_read": False,
        "source_cost": state,
        "reader_metrics": rows,
        "reference_mean_cosines": cosine_matrix(refs).tolist(),
        "mean_gradient_norms": np.linalg.norm(refs, axis=1).tolist(),
        "within_query_gradient_norm_cv": (
            np.linalg.norm(g, axis=-1).std(axis=0)
            / np.linalg.norm(g, axis=-1).mean(axis=0)
        ).tolist(),
        "sensitivity_spectrum_energy": (s**2 / np.sum(s**2)).tolist(),
        "within_query_first_energy": [float(a[0] ** 2 / np.sum(a**2)) for a in sq],
        "transfer_mean_pairing_cosines": np.mean(
            [cosine_matrix(v) for v in vectors], axis=0
        ).tolist(),
        "transfer_effect_rms": np.sqrt(np.mean(effects**2, axis=0)).tolist(),
        "transfer_effect_correlations": np.corrcoef(effects.T).tolist(),
        "max_reference_completeness_gap": max(
            abs(p["final_gap"]) for a in reference_packets for p in a
        ),
        "max_reference_endpoint_error": max(
            p["endpoint_error"] for a in reference_packets for p in a
        ),
        "five_node_reference_units": sum(
            p["nodes"] == 5 for a in reference_packets for p in a
        ),
    }
    write(out / "summary.json", summary)
    print(
        frame[
            ["query", "reader", "r", "mse_over_effect_energy", "sign_agreement"]
        ].to_string(index=False),
        flush=True,
    )
    print(
        {
            k: summary[k]
            for k in (
                "reference_mean_cosines",
                "mean_gradient_norms",
                "within_query_first_energy",
                "transfer_mean_pairing_cosines",
                "transfer_effect_rms",
                "transfer_effect_correlations",
                "max_reference_completeness_gap",
                "five_node_reference_units",
            )
        },
        flush=True,
    )


def main():
    """Expose immutable prepare, resumable run, pilot inspect and full evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "inspect", "evaluate"])
    parser.add_argument(
        "--source", type=Path, default=BASE / "functional_transfer_s20_r1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "cross_query_functional_s21_r1"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-reference-authors", type=int, default=4)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "inspect": inspect, "evaluate": evaluate}[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
