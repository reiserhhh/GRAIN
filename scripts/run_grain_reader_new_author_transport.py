#!/usr/bin/env python3
"""NEXT37: frozen readers on eight source-disjoint authors, no new fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from grain_research.vector_mechanisms.full_state_edit import (
    capture_boundary,
    replay_suffix,
)
from grain_research.vector_mechanisms.native_material_reader import (
    allocate_slices,
    common_prefix_length,
    direction,
)
from grain_research.vector_mechanisms.reader_ensemble import pack_models
from scripts.analyze_grain_query_direction_transfer import (
    verify_predictions as verify_transfer,
)
from scripts.analyze_grain_query_residual_input import (
    verify_predictions as verify_residual,
)
from scripts.run_grain_full_state_edit_connection import SOURCE, token_view
from scripts.run_grain_multiquery_sensitivity import QUERIES, native_alphabet
from scripts.run_grain_native_material_reader import (
    BUNDLE,
    EPS,
    FC,
    OLD,
    REPRESENTATION,
    suffix_gradient,
)
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_within_comment_channels import lexical_slices
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

PLAN = ROOT / "docs/GRAIN_READER_NEW_AUTHOR_TRANSPORT_PLAN_V1.md"
INVENTORY = BASE / "logic_reader_transport_inventory_v1"
RESIDUAL = BASE / "logic_query_residual_input_v1_r2"
TRANSFER = BASE / "logic_query_direction_transfer_v1"
ARMS = (*QUERIES, "variance")


def verify_hashes(base, hashes):
    """Fail closed on changed declared files, without loading their contents."""
    for name, digest in hashes.items():
        if sha(base / name) != digest:
            raise ValueError("Changed frozen dependency: " + name)


def verify(out):
    """Check preparation and fixed source assets before either stage."""
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    verify_hashes(out, m["prepared"])
    return m


def verify_prediction_freeze(out):
    """Verify numeric predictions before accessing evaluation gradients."""
    verify(out)
    f = read(out / "prediction_freeze.json")
    if sha(out / "manifest.json") != f["manifest_sha256"]:
        raise ValueError("Prediction manifest changed")
    verify_hashes(out, f["files"])
    return f


def prepare(out):
    """Bind frozen authors, tokenize and score slices with existing GRAIN assets."""
    from transformers import AutoTokenizer

    inv = read(INVENTORY / "manifest.json")
    verify_hashes(ROOT, inv["inputs"])
    verify_hashes(INVENTORY, inv["files"])
    verify_residual(RESIDUAL)
    verify_transfer(TRANSFER)
    members = read(INVENTORY / "private_selection.json")["authors"]
    old_ids = {
        a["author_id"] for a in read(RESIDUAL / "private_membership.json")["authors"]
    }
    if len(members) != 8 or old_ids & {a["author_id"] for a in members}:
        raise ValueError("Eight source-disjoint authors required")
    material = {
        r["author_id"]: r
        for r in read(SOURCE / "private_material.json")
        if r["view"] == 0
    }
    out.mkdir(exist_ok=False)
    for name in (
        "private_prefix",
        "private_packets",
        "private_models",
        "private_inputs",
    ):
        (out / name).mkdir()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design, raw, meta, alloc = [], [], [], {}
    for member in members:
        i = member["index"]
        texts = material[member["author_id"]]["texts"]
        digest = hashlib.sha256(
            json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != member["text_list_sha256"] or len(texts) != 12:
            raise ValueError("Frozen twelve-comment material changed")
        encoded, spans, queries = {}, {}, {}
        for q in QUERIES:
            _, spans[q], encoded[q] = token_view(tokenizer, texts, q)
            queries[q] = {
                "ids": encoded[q]["input_ids"],
                "alphabet": native_alphabet(q, tokenizer),
            }
        prefix = common_prefix_length([queries[q]["ids"] for q in QUERIES])
        body_end = (
            max(
                j
                for j, (_, hi) in enumerate(encoded[QUERIES[0]]["offset_mapping"])
                if hi <= spans[QUERIES[0]][-1][1]
            )
            + 1
        )
        if prefix < body_end or any(spans[q] != spans[QUERIES[0]] for q in QUERIES):
            raise ValueError("Query changes common material token prefix")
        slice_spans, eligible = [], 0
        for j, text in enumerate(texts):
            parts = lexical_slices(text)
            if not parts:
                continue
            eligible += 1
            lo = spans[QUERIES[0]][j][0]
            for s, part in enumerate(parts):
                key = f"a{i:02d}_c{j:02d}_s{s}"
                raw.append({"author": key, "body": part})
                meta.append({"user_id": key, "index": i, "comment": j, "slice": s})
                slice_spans.append((lo, lo + len(part)))
                lo += len(part)
        if eligible < 2:
            raise ValueError("Insufficient slices; no author replacement permitted")
        alloc[i] = (encoded[QUERIES[0]]["offset_mapping"], slice_spans)
        design.append(
            dict(
                member,
                eligible_comments=eligible,
                common_prefix_tokens=prefix,
                queries=queries,
            )
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
        raise ValueError("All predeclared slices must score")
    scored = pd.DataFrame(meta).merge(
        scored[["user_id", *FC]], on="user_id", validate="one_to_one"
    )
    arrays = {}
    for a in design:
        i = a["index"]
        g = (
            scored[scored["index"] == i]
            .sort_values(["comment", "slice"])[FC]
            .to_numpy(float)
        )
        positions, token_g, weights = allocate_slices(*alloc[i], g)
        arrays[f"a{i}__positions"], arrays[f"a{i}__grain"] = positions, token_g
        a.update(
            selected_tokens=len(positions),
            slices=len(g),
            allocation_error=float(abs(weights.sum(1) - 1).max()),
        )
    np.savez_compressed(out / "private_coordinates.npz", **arrays)
    write(out / "private_design.json", design)
    files = [
        PLAN,
        Path(__file__),
        INVENTORY / "manifest.json",
        INVENTORY / "private_selection.json",
        BUNDLE,
        REPRESENTATION,
        OLD / "observer.json",
        OLD / "manifest.json",
        SOURCE / "private_material.json",
        MODEL / "config.json",
        MODEL / "tokenizer.json",
        MODEL / "tokenizer_config.json",
        RESIDUAL / "prediction_freeze.json",
        TRANSFER / "prediction_freeze.json",
    ]
    files += [
        ROOT / p
        for p in (
            "grain_research/vector_mechanisms/reader_ensemble.py",
            "grain_research/vector_mechanisms/native_material_reader.py",
            "grain_research/vector_mechanisms/query_residual_input.py",
            "grain_research/vector_mechanisms/content_coefficients.py",
            "grain_research/vector_mechanisms/full_state_edit.py",
            "grain_research/vector_mechanisms/context_carrier.py",
            "grain_research/vector_mechanisms/direct_traits.py",
            "scripts/replay_grain_reader_ensemble.py",
            "scripts/replay_grain_content_coefficients.py",
            "scripts/run_grain_native_material_reader.py",
            "scripts/run_grain_full_state_edit_connection.py",
            "scripts/run_grain_multiquery_sensitivity.py",
            "scripts/run_grain_within_comment_channels.py",
            "suica_core/v7_operator_controls.py",
            "suica_core/v7_psychometric.py",
            "suica_core/v7_observations.py",
        )
    ]
    for q in QUERIES:
        for arm in ARMS:
            models = []
            for s in range(4):
                path = (
                    RESIDUAL
                    / q
                    / f"private_model_{s}_residual_{'variance' if arm == 'variance' else 'query'}.npz"
                    if arm in (q, "variance")
                    else TRANSFER / q / f"private_model_{s}_from_{arm}.npz"
                )
                files.append(path)
                with np.load(path, allow_pickle=False) as z:
                    models.append(dict(z))
            np.savez_compressed(
                out / "private_models" / f"{q}__{arm}.npz", **pack_models(models)
            )
    prepared = [
        out / "private_coordinates.npz",
        out / "private_design.json",
        *sorted((out / "private_models").glob("*.npz")),
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_READER_NEW_AUTHOR_TRANSPORT_V1",
            "authors": 8,
            "queries": list(QUERIES),
            "arms": list(ARMS),
            "source_components": 4,
            "fits": 0,
            "human_labels_read": False,
            "max_calls": 54,
            "max_seconds": 600,
            "max_backwards": 24,
            "model_sha256": read(OLD / "manifest.json")["model_sha256"],
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "backwards": 0, "seconds": 0},
    )
    summary = {
        "authors": 8,
        "comments": 96,
        "slices": sum(a["slices"] for a in design),
        "selected_tokens": sum(a["selected_tokens"] for a in design),
        "fits": 0,
        "calls": 0,
    }
    write(out / "construction_summary.json", summary)
    print(summary, flush=True)


def prefix_path(out, i, q):
    """Address a per-query complete block16 state, not a last-token surrogate."""
    return out / "private_prefix" / f"a{i:02d}_{q}.json"


def packet_arrays(path):
    """Load only hash-verified numerical packets."""
    m = read(path)
    if sha(path.with_suffix(".npz")) != m["arrays_sha256"]:
        raise ValueError("Packet arrays changed")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        return m, dict(z)


def numeric_predict(out):
    """Freeze all96 predictions in a separate input/model-only process."""
    verify(out)
    if (out / "prediction_freeze.json").exists():
        verify_prediction_freeze(out)
        return
    jobs, paths = [], []
    with np.load(out / "private_coordinates.npz", allow_pickle=False) as z:
        coords = dict(z)
    for a in read(out / "private_design.json"):
        i = a["index"]
        for q in QUERIES:
            p = prefix_path(out, i, q)
            packet_arrays(p)
            paths += [p, p.with_suffix(".npz")]
        _, z = packet_arrays(prefix_path(out, i, QUERIES[0]))
        pos, g = coords[f"a{i}__positions"], coords[f"a{i}__grain"]
        inp = out / "private_inputs" / f"a{i:02d}.npz"
        np.savez_compressed(
            inp, hidden=z["state"][pos].astype(float), grain=g, positions=pos
        )
        paths.append(inp)
        for q in QUERIES:
            for arm in ARMS:
                jobs.append(
                    {
                        "key": f"{q}__a{i}__{arm}",
                        "model": str(out / "private_models" / f"{q}__{arm}.npz"),
                        "inputs": str(inp),
                    }
                )
    write(out / "private_jobs.json", jobs)
    command = [
        sys.executable,
        str(ROOT / "scripts/replay_grain_reader_ensemble.py"),
        "--jobs",
        str(out / "private_jobs.json"),
        "--output",
        str(out / "private_predictions.npz"),
        "--forbid",
        str(out / "private_design.json"),
        "--forbid",
        str(OLD / "private_inputs.json"),
        "--forbid",
        str(out / "private_packets/absent_gradient.npz"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    runtime = json.loads(result.stdout)
    if runtime != {"jobs": 96, "denial_probes": 3, "extra_denials": 0}:
        raise ValueError("Restricted runtime protocol changed")
    paths += [out / "private_jobs.json", out / "private_predictions.npz"]
    write(
        out / "prediction_freeze.json",
        {
            "manifest_sha256": sha(out / "manifest.json"),
            "runtime": runtime,
            "before_new_gradient_access": not any((out / "private_packets").iterdir()),
            "files": {str(p.relative_to(out)): sha(p) for p in paths},
        },
    )
    print({"status": "PREDICTIONS_FROZEN", **runtime}, flush=True)


def native_stage(out, stage):
    """Capture prefix states or evaluate frozen predictions with cumulative bounds."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    m = verify(out)
    if stage == "collect":
        verify_prediction_freeze(out)
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE" or (
        stage == "prefix" and prior["status"] == "PREFIX_COMPLETE"
    ):
        print("Requested native stage already complete; zero calls")
        return
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Checkpoint changed")
    torch.set_num_threads(2)
    loading = time.monotonic()
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    load_seconds = time.monotonic() - loading
    now, old = observer.manifest(), read(OLD / "observer.json")
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
            raise ValueError("Observer changed: " + key)
    write(out / "observer.json", now)
    with np.load(out / "private_coordinates.npz", allow_pickle=False) as z:
        coords = dict(z)
    start, backwards = time.monotonic(), 0

    def status(state):
        """Persist even partially spent neural work instead of resetting on retries."""
        value = {
            "status": state,
            "stage": stage,
            "calls": prior["calls"] + observer.calls,
            "backwards": prior["backwards"] + backwards,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "load_seconds": prior.get("load_seconds", 0) + load_seconds,
            "prefix_units": len(list((out / "private_prefix").glob("*.json"))),
            "gradient_units": len(list((out / "private_packets").glob("*.json"))),
            "fits": 0,
            "human_labels_read": False,
        }
        write(out / "run_status.json", value)
        print(value, flush=True)
        return value

    def room(calls):
        """Check before each atomic unit using persisted plus current costs."""
        return (
            prior["calls"] + observer.calls + calls <= m["max_calls"]
            and prior["seconds"] + time.monotonic() - start < m["max_seconds"]
        )

    try:
        for a in read(out / "private_design.json"):
            i = a["index"]
            pos, g = coords[f"a{i}__positions"], coords[f"a{i}__grain"]
            if stage == "collect":
                _, common = packet_arrays(prefix_path(out, i, QUERIES[0]))
                common = common["state"]
                hbody = common[pos].astype(float)
            for q in QUERIES:
                dst = (
                    prefix_path(out, i, q)
                    if stage == "prefix"
                    else out / "private_packets" / f"a{i:02d}_{q}.json"
                )
                if dst.exists():
                    packet_arrays(dst)
                    continue
                budget = 1 if stage == "prefix" or i != 0 else 3
                if not room(budget):
                    status("PAUSED_RESOURCE_BOUNDARY")
                    return
                alphabet = a["queries"][q]["alphabet"]
                if stage == "prefix":
                    state, kw, score = capture_boundary(
                        observer, a["queries"][q]["ids"], alphabet
                    )
                    np.savez_compressed(
                        dst.with_suffix(".npz"), state=state[0].float().cpu().numpy()
                    )
                    record = {"index": i, "query": q, "full": score}
                    del state, kw
                else:
                    original, z = packet_arrays(prefix_path(out, i, q))
                    full = z["state"]
                    prefix = a["common_prefix_tokens"]
                    delta = full[:prefix].astype(float) - common[:prefix]
                    full[:prefix] = common[:prefix]
                    state = torch.tensor(full[None], device="mps")
                    kw = cached_suffix_kwargs(observer.model, state)
                    backwards += 1
                    score, jacobian = suffix_gradient(observer, state, kw, alphabet)
                    replay = abs(score - original["full"]["conditional_score"])
                    if replay > 2e-6 or not np.isfinite(jacobian).all():
                        raise ValueError(
                            "Canonicalized score replay/finite gradient failed"
                        )
                    fd = None
                    if i == 0:
                        d, _ = direction(hbody, g, 0)
                        values = []
                        for sign in (1, -1):
                            changed = state.clone()
                            changed[:, pos] += torch.tensor(
                                sign * EPS * d[None],
                                device=state.device,
                                dtype=state.dtype,
                            )
                            values.append(
                                replay_suffix(observer, changed, kw, alphabet)[
                                    "conditional_score"
                                ]
                            )
                            del changed
                        analytic = float(np.sum(d * jacobian[pos]))
                        observed = (values[0] - values[1]) / (2 * EPS)
                        fd = {
                            "factor": "whole:F01",
                            "epsilon": EPS,
                            "analytic": analytic,
                            "observed": observed,
                            "finite_delta_error": abs(EPS * (observed - analytic)),
                        }
                    np.savez_compressed(
                        dst.with_suffix(".npz"), body_gradient=jacobian[pos]
                    )
                    record = {
                        "index": i,
                        "query": q,
                        "score": score,
                        "score_replay_error": replay,
                        "prefix_max": float(abs(delta).max()),
                        "prefix_rms": float(np.sqrt(np.mean(delta**2))),
                        "finite_difference": fd,
                    }
                    del state, kw, jacobian
                write(dst, record | {"arrays_sha256": sha(dst.with_suffix(".npz"))})
                status("RUNNING")
            if i == 0:
                elapsed = time.monotonic() - start
                # The gradient phase's first author includes all six FD calls.
                forecast = prior["seconds"] + elapsed * 8
                write(
                    out / f"{stage}_forecast.json",
                    {
                        "first_author_seconds": elapsed,
                        "conservative_total_seconds": forecast,
                        "cap_seconds": m["max_seconds"],
                        "decision": "CONTINUE"
                        if forecast <= m["max_seconds"]
                        else "PAUSE",
                    },
                )
                if forecast > m["max_seconds"]:
                    status("PAUSED_FORECAST_BOUNDARY")
                    return
        status("PREFIX_COMPLETE" if stage == "prefix" else "COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def main():
    """Expose individually resumable stages; never silently overwrite a packet."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "prefix", "predict", "collect"))
    p.add_argument("--name", default="logic_reader_new_author_transport_v1")
    a = p.parse_args()
    out = BASE / a.name
    with threadpool_limits(limits=4):
        if a.stage == "prepare":
            prepare(out)
        elif a.stage == "predict":
            numeric_predict(out)
        else:
            native_stage(out, a.stage)


if __name__ == "__main__":
    main()
