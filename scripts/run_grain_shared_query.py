#!/usr/bin/env python3
"""S14: author-out common query versus contextual query on frozen source KV readers."""

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
from grain_research.vector_mechanisms.direct_traits import tokenize_prompt
from grain_research.vector_mechanisms.prefix import (
    capture_prefix,
    prefix_pairing,
    rotate_query,
)
from grain_research.vector_mechanisms.query_decomposition import (
    fold_query_references,
    mix_raw_query,
    pairing_query_expansion,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_query_representation import target_quality
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_query_representation import validate_packet, verify
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION, covariance

ALPHAS = (0.0, 0.5, 1.0)


def prepare(args):
    """Freeze same-source membership, hidden queries and existing author-out readers."""
    source, out = args.collection.resolve(), args.output_dir
    _manifest, material = verify(source, args.device)
    oof_path = source / "analysis/private_oof_before_outcomes.csv"
    if sha(oof_path) != read(source / "analysis/summary.json")["oof_sha256"]:
        raise ValueError("S12 source predictions changed")
    frame = pd.read_csv(oof_path, dtype={"author_id": str})
    frame = frame[
        (frame.representation == "kv_pairing")
        & (frame.method == "pca64")
        & (frame.target == "configuration_response")
    ].set_index("author_id")
    names = [a["author_id"] for a in material]
    frame = frame.loc[names]
    states, packet_hashes = [], {}
    for i, author in enumerate(material):
        path = source / "private_packets" / f"author_{i:03d}.json"
        validate_packet(path, author)
        packet_hashes[path.name] = sha(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            states.append(z["query_state"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    frame.to_csv(out / "private_inputs.csv")
    np.savez_compressed(out / "private_query_states.npz", states=np.array(states))
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S14_SHARED_QUERY_V2",
            "collection": str(source),
            "collection_manifest_sha256": sha(source / "manifest.json"),
            "source_packet_hashes": packet_hashes,
            "private_inputs_sha256": sha(out / "private_inputs.csv"),
            "query_states_sha256": sha(out / "private_query_states.npz"),
            "oof_sha256": sha(oof_path),
            "authors": 213,
            "views": 639,
            "alphas": ALPHAS,
            "max_seconds": 2100,
            "max_prefix_calls": 639,
            "device": args.device,
            "dtype": "float32",
            "model_hash": sha(MODEL / "model.safetensors"),
            "code_hashes": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    Path(__file__),
                    ROOT / "grain_research/vector_mechanisms/prefix.py",
                    ROOT / "grain_research/vector_mechanisms/query_decomposition.py",
                )
            },
            "reader_hashes": {
                p.name: sha(p)
                for p in (source / "analysis/private_models").glob(
                    "kv_pairing_pca64_configuration_response_fold*.npz"
                )
            },
            "questionnaire_fit": False,
        },
    )
    print(
        {"status": "PREPARED", "authors": len(frame), "prefix_calls": 639}, flush=True
    )


def verify_inputs(out, device):
    """Refuse a resumed collection with changed source/model/readout inputs."""
    m = read(out / "manifest.json")
    source = Path(m["collection"])
    if (
        m["device"] != device
        or sha(source / "manifest.json") != m["collection_manifest_sha256"]
        or any(
            sha(source / "private_packets" / p) != digest
            for p, digest in m["source_packet_hashes"].items()
        )
        or sha(out / "private_inputs.csv") != m["private_inputs_sha256"]
        or sha(out / "private_query_states.npz") != m["query_states_sha256"]
        or sha(MODEL / "model.safetensors") != m["model_hash"]
        or sha(source / "analysis/private_oof_before_outcomes.csv") != m["oof_sha256"]
        or any(sha(ROOT / p) != digest for p, digest in m["code_hashes"].items())
        or any(
            sha(source / "analysis/private_models" / p) != digest
            for p, digest in m["reader_hashes"].items()
        )
    ):
        raise ValueError("Frozen shared-query inputs changed")
    _manifest, material = verify(source, device)
    return m, material


def reference_queries(observer, states, folds, layer_index=17):
    """Use fixed input normalization/Q projection, then train-fold-only query means."""
    layer = observer.model.model.layers[layer_index]
    device = next(observer.model.parameters()).device
    q = torch.as_tensor(
        states.reshape(-1, 1, states.shape[-1]), device=device, dtype=torch.float32
    )
    with torch.no_grad():
        raw = (
            layer.self_attn.q_proj(layer.input_layernorm(q))[:, 0].float().cpu().numpy()
        )
    raw = raw.reshape(len(states), 3, -1, layer.self_attn.head_dim)
    return fold_query_references(raw, folds)


def run(args):
    """Collect all three fixed doses from each unchanged KV prefix, with no outcome read."""
    out = args.output_dir
    manifest, material = verify_inputs(out, args.device)
    source = Path(manifest["collection"])
    inputs = pd.read_csv(out / "private_inputs.csv", dtype={"author_id": str})
    if list(inputs.author_id) != [a["author_id"] for a in material]:
        raise ValueError("Source order mismatch")
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"calls": 0, "seconds": 0}
    )
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    with np.load(out / "private_query_states.npz", allow_pickle=False) as q:
        references = reference_queries(observer, q["states"], inputs.fold.to_numpy())
    ref_path = out / "private_references.npz"
    if ref_path.exists():
        with np.load(ref_path, allow_pickle=False) as saved:
            if any(
                not np.allclose(saved[str(k)], v, atol=1e-6, rtol=1e-6)
                for k, v in references.items()
            ):
                raise ValueError("Frozen reference queries changed")
    else:
        np.savez_compressed(ref_path, **{str(k): v for k, v in references.items()})
        write(
            out / "references_frozen.json",
            {"sha256": sha(ref_path), "labels_read": False},
        )
    if sha(ref_path) != read(out / "references_frozen.json")["sha256"]:
        raise ValueError("Reference artifact hash mismatch")
    start = time.monotonic()

    def status(name):
        state = {
            "status": name,
            "calls": previous["calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "authors": len(list((out / "private_packets").glob("author_*.json"))),
            "labels_read": False,
        }
        write(out / "run_status.json", state)
        print(state, flush=True)

    try:
        for i, author in enumerate(material):
            path = out / "private_packets" / f"author_{i:03d}.json"
            if path.exists():
                packet = read(path)
                if (
                    packet["author_id"] != author["author_id"]
                    or len(packet["views"]) != 3
                ):
                    raise ValueError("Resumed author mismatch")
                continue
            if (
                i >= args.max_authors
                or previous["seconds"] + time.monotonic() - start >= 2100
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            fold = int(inputs.iloc[i].fold)
            model_path = (
                source
                / "analysis/private_models"
                / f"kv_pairing_pca64_configuration_response_fold{fold}.npz"
            )
            with np.load(model_path, allow_pickle=False) as model:
                coef, intercept = model["coef"], float(model["intercept"])
            with np.load(
                source / "private_packets" / f"author_{i:03d}.npz", allow_pickle=False
            ) as z:
                native_features = z["kv_pairing"]
            rows = []
            for view, payload in enumerate(author["views"]):
                ids, user, spans = tokenize_prompt(
                    payload["texts"], "neuroticism", observer.tokenizer
                )
                if material_hash(ids) != payload["prompt_sha256"]:
                    raise ValueError("Frozen source prompt changed")
                groups = token_groups(observer, user, spans, ids)
                cache, trace = capture_prefix(observer, ids)
                native, reference = (
                    cache["raw_query"],
                    references[fold].astype(np.float32),
                )
                cosine = np.sum(native * reference, axis=1) / (
                    np.linalg.norm(native, axis=1) * np.linalg.norm(reference, axis=1)
                )
                relative = np.linalg.norm(native - reference, axis=1) / np.linalg.norm(
                    native, axis=1
                )
                predictions, vector_error = {}, None
                reference_rotated = rotate_query(
                    reference,
                    cache["last_cos"],
                    cache["last_sin"],
                    cache["scaling"],
                )
                for alpha in ALPHAS:
                    mixed = mix_raw_query(reference, native, alpha)
                    query = (
                        cache["q"]
                        if alpha == 1
                        else rotate_query(
                            mixed,
                            cache["last_cos"],
                            cache["last_sin"],
                            cache["scaling"],
                        )
                    )
                    features = prefix_pairing(cache, groups, queries=query)
                    predictions[str(alpha)] = float(features @ coef + intercept)
                    if alpha == 1:
                        vector_error = float(
                            np.max(np.abs(features - native_features[view]))
                        )
                if vector_error > 1e-5 or trace["forbidden_calls"]:
                    raise ValueError("Native replay or prefix boundary failed")
                expansion = pairing_query_expansion(
                    cache, groups, reference_rotated, coef
                )
                increment_error = abs(
                    expansion["exact_increment"]
                    - (predictions["1.0"] - predictions["0.0"])
                )
                if increment_error > 1e-8:
                    raise ValueError("Centered-value derivative readout disagrees")
                rows.append(
                    {
                        "view": view,
                        "scores": predictions,
                        "native_vector_error": vector_error,
                        "query_cosines": cosine.tolist(),
                        "query_relative_changes": relative.tolist(),
                        "query_expansion": expansion,
                        "expansion_identity_error": increment_error,
                    }
                )
            native_mean = float(np.mean([v["scores"]["1.0"] for v in rows]))
            score_error = abs(native_mean - inputs.iloc[i].prediction)
            if score_error > 1e-6:
                raise ValueError("Native affine source replay failed")
            write(
                path,
                {
                    "author_id": author["author_id"],
                    "fold": fold,
                    "views": rows,
                    "native_score_error": float(score_error),
                },
            )
            if (i + 1) % 16 == 0:
                status("RUNNING")
            if previous["calls"] + observer.calls > 639:
                raise RuntimeError("Prefix call budget exceeded")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(args):
    """Freeze score components before connecting only the exposed source N outcomes."""
    out = args.output_dir
    _manifest, material = verify_inputs(out, args.device)
    state = read(out / "run_status.json")
    if state["status"] != "COMPLETE" or state["calls"] != 639:
        raise ValueError("Complete source run required")
    inputs = pd.read_csv(
        out / "private_inputs.csv", dtype={"author_id": str}
    ).set_index("author_id")
    rows, cosines, changes, packets = [], [], [], []
    for i, author in enumerate(material):
        packet = read(out / "private_packets" / f"author_{i:03d}.json")
        if packet["author_id"] != author["author_id"]:
            raise ValueError("Author mismatch at outcome connection")
        row = {
            "author_id": author["author_id"],
            "teacher": float(inputs.loc[author["author_id"], "teacher"]),
            "null": float(inputs.loc[author["author_id"], "null_prediction"]),
        }
        for alpha in ALPHAS:
            row[str(alpha)] = float(
                np.mean([v["scores"][str(alpha)] for v in packet["views"]])
            )
        row["query_increment"] = row["1.0"] - row["0.0"]
        row["first_order"] = float(
            np.mean([v["query_expansion"]["first"] for v in packet["views"]])
        )
        row["second_derivative"] = float(
            np.mean([v["query_expansion"]["second"] for v in packet["views"]])
        )
        row["second_order"] = row["first_order"] + 0.5 * row["second_derivative"]
        rows.append(row)
        packets.append(packet)
        for v in packet["views"]:
            cosines.extend(v["query_cosines"])
            changes.extend(v["query_relative_changes"])
    frame = pd.DataFrame(rows)
    frozen = out / "private_scores_before_outcomes.csv"
    if frozen.exists() or (out / "summary.json").exists():
        raise ValueError("Preserve frozen completed analysis")
    frame.to_csv(frozen, index=False)
    write(out / "scores_frozen.json", {"sha256": sha(frozen), "labels_used": False})
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(labels, on="author_id", validate="one_to_one")
    if not np.isfinite(frame.y_true).all():
        raise ValueError("Source label mismatch")
    metrics, pairs, moments, taylor = [], [], [], []
    ix = np.random.default_rng(734).integers(len(frame), size=(1000, len(frame)))
    for measure in ("0.0", "0.5", "1.0", "query_increment"):
        y, pred = frame.y_true.to_numpy(), frame[measure].to_numpy()
        external = scalar_metrics(y, pred)
        row = {
            "measure": measure,
            "source_r": external["pearson_r"],
            "source_lower": external["r_lower"],
            "source_upper": external["r_upper"],
            "source_p": external["pearson_p"],
        }
        if measure != "query_increment":
            row.update(target_quality(frame.teacher, pred, frame["null"]))
            pairs.extend(
                [
                    {
                        "measure": measure,
                        "comparison": "target_vs_native",
                        **paired_intervals(frame.teacher, pred, frame["1.0"], False),
                    },
                    {
                        "measure": measure,
                        "comparison": "source_vs_native",
                        **paired_intervals(y, pred, frame["1.0"], False),
                    },
                ]
            )
        metrics.append(row)
        boot = [covariance(y[j], pred[j]) for j in ix]
        moments.append(
            {
                "measure": measure,
                "covariance": covariance(y, pred),
                "lower": float(np.quantile(boot, 0.025)),
                "upper": float(np.quantile(boot, 0.975)),
            }
        )
    identity = abs(
        covariance(frame.y_true, frame["1.0"])
        - covariance(frame.y_true, frame["0.0"])
        - covariance(frame.y_true, frame.query_increment)
    )
    for alpha in (0.5, 1.0):
        truth = frame[str(alpha)] - frame["0.0"]
        for order in (1, 2):
            pred = alpha * frame.first_order
            if order == 2:
                pred = pred + 0.5 * alpha**2 * frame.second_derivative
            detail = scalar_metrics(truth, pred)
            mse = float(np.mean((truth - pred) ** 2))
            zero = float(np.mean(truth**2))
            taylor.append(
                {
                    "alpha": alpha,
                    "order": order,
                    "increment_r": detail["pearson_r"],
                    "mse": mse,
                    "mse_over_zero": mse / zero if zero else None,
                    "source_r": scalar_metrics(frame.y_true, pred)["pearson_r"],
                }
            )
    for name, records in (
        ("metrics", metrics),
        ("paired_intervals", pairs),
        ("covariance_split", moments),
        ("query_taylor", taylor),
    ):
        pd.DataFrame(records).to_csv(out / f"{name}.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "manifest_sha256": sha(out / "manifest.json"),
        "scores_sha256": sha(frozen),
        "prefix_calls": 639,
        "full_calls": 0,
        "seconds": state["seconds"],
        "source_questionnaire_fit": False,
        "max_native_reader_error": max(p["native_score_error"] for p in packets),
        "max_native_vector_error": max(
            v["native_vector_error"] for p in packets for v in p["views"]
        ),
        "median_query_cosine": float(np.median(cosines)),
        "query_cosine_10_90": [float(v) for v in np.quantile(cosines, [0.1, 0.9])],
        "median_relative_query_change": float(np.median(changes)),
        "covariance_identity_error": identity,
        "max_expansion_identity_error": max(
            v["expansion_identity_error"] for p in packets for v in p["views"]
        ),
        "metrics": metrics,
        "paired_intervals": pairs,
        "covariance_split": moments,
        "query_taylor": taylor,
        "scope": "frozen native reader under author-out prototype query intervention; source discovery",
    }
    write(out / "summary.json", summary)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(pd.DataFrame(moments).to_string(index=False), flush=True)
    print(pd.DataFrame(taylor).to_string(index=False), flush=True)


def main():
    """Prepare, collect, or analyze one frozen shared-query experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "evaluate"))
    parser.add_argument(
        "--collection", type=Path, default=BASE / "query_representation_s12_r1"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-authors", type=int, default=213)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(BASE):
        raise ValueError("Private source outputs must remain ignored")
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
