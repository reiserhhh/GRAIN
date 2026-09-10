#!/usr/bin/env python3
"""NEXT41: source-fixed final readout of complete H16 forward transport."""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.complete_transport import (
    body_direction,
    fixed_means,
    forward_response,
    post_attention,
    tail_contrast,
)
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_position_reader import action_metrics
from scripts.run_grain_fixed_seed_attention import context
from scripts.run_grain_fixed_seed_attention import verify as verify_native
from scripts.run_grain_reader_new_author_transport import packet_arrays, verify_hashes
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

NATIVE = BASE / "logic_fixed_seed_attention_v1"
PLAN = ROOT / "docs/GRAIN_COMPLETE_TRANSPORT_JVP_PLAN_V1.md"


def verify(out):
    """Verify the immutable design, exported forward inputs and current code."""
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    verify_hashes(out, m["prepared"])
    return m


def prepare(out):
    """Export exact canonical H16 states without new-author target fields."""
    native = verify_native(NATIVE)
    if read(NATIVE / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete NEXT39 inputs required")
    out.mkdir(exist_ok=False)
    for folder in (
        "private_inputs",
        "private_sources",
        "private_models",
        "private_predictions",
    ):
        (out / folder).mkdir()
    jobs, files = (
        [],
        [PLAN, Path(__file__), NATIVE / "manifest.json", NATIVE / "private_jobs.json"],
    )
    for name in (
        "complete_transport",
        "context_carrier",
        "native_material_reader",
        "source_cotangent",
        "trace",
    ):
        files.append(ROOT / "grain_research/vector_mechanisms" / (name + ".py"))
    for job in read(NATIVE / "private_jobs.json"):
        full, _, grain, pos = context(job)
        dst = out / "private_inputs" / (job["key"] + ".npz")
        np.savez_compressed(dst, state=full, grain=grain, positions=pos)
        jobs.append({k: job[k] for k in ("key", "cohort", "index", "query")})
        for name in ("prefix", "common"):
            p = ROOT / job[name]
            files.extend([p, p.with_suffix(".npz")])
        files.append(ROOT / job["input"])
    if len(jobs) != 24:
        raise ValueError("Fixed24 units required")
    write(out / "private_jobs.json", jobs)
    prepared = [out / "private_jobs.json"] + sorted(
        (out / "private_inputs").glob("*.npz")
    )
    prepared = [p for p in prepared if not p.name.startswith("._")]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_COMPLETE_TRANSPORT_JVP_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "alphabet": native["alphabet"],
            "c": native["c"],
            "model_sha256": native["model_sha256"],
            "max_seconds": 300,
            "max_source_calls": 8,
            "max_jvps": 96,
            "max_fd_calls": 4,
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print({"status": "PREPARED", "units": 24, "targets_exported": False}, flush=True)


def load_observer(m):
    """Load the same frozen MPS observer; device/kernel changes are not automatic."""
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Frozen model changed")
    torch.set_num_threads(2)
    start = time.monotonic()
    ob = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    ob.model.requires_grad_(False)
    old, new = read(NATIVE / "observer.json"), ob.manifest()
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if old[key] != new[key]:
            raise ValueError("Observer differs: " + key)
    return ob, time.monotonic() - start


def inputs(out, job):
    """Read only the declared forward input schema."""
    with np.load(
        out / "private_inputs" / (job["key"] + ".npz"), allow_pickle=False
    ) as z:
        if set(z.files) != {"state", "grain", "positions"}:
            raise ValueError("Unexpected forward input fields")
        state = torch.tensor(z["state"][None], device="mps")
        return state, z["grain"], z["positions"]


def sources(out):
    """Capture only source t, verify its exact layer relation, then freeze means."""
    m = verify(out)
    if (out / "source_freeze.json").exists():
        print("Sources already frozen; zero new calls")
        return
    ob, loading = load_observer(m)
    jobs = read(out / "private_jobs.json")
    prior = (
        read(out / "source_status.json")
        if (out / "source_status.json").exists()
        else {"seconds": 0, "completed": []}
    )
    started, done = time.monotonic(), list(prior["completed"])
    records, values = [], {}
    ids = torch.tensor(m["alphabet"]["token_ids"], device="mps")
    contrast = torch.tensor(m["c"], device="mps", dtype=torch.float32)
    for job in [j for j in jobs if j["cohort"] == "source4"]:
        dst = out / "private_sources" / (job["key"] + ".json")
        if job["key"] not in done:
            if prior["seconds"] + time.monotonic() - started >= m["max_seconds"]:
                raise TimeoutError(
                    "Source resource cap reached; completed units preserved"
                )
            state, _, _ = inputs(out, job)
            kw = cached_suffix_kwargs(ob.model, state)
            with torch.no_grad():
                r = post_attention(ob.model, state, kw)[0]
            leaf = r.detach().requires_grad_(True)
            with torch.enable_grad():
                score = tail_contrast(ob.model, leaf, ids, contrast)
                t = torch.autograd.grad(score, leaf)[0]
            g = t @ ob.model.model.layers[-1].self_attn.o_proj.weight
            ref, arr = packet_arrays(
                NATIVE / "private_packets" / (job["key"] + ".json")
            )
            err = g.detach().cpu().numpy().ravel().astype(float) - arr["g"].ravel()
            rec = {
                "key": job["key"],
                "score_error": abs(float(score.detach().cpu()) - ref["contrast"]),
                "g_relative_l2": float(np.linalg.norm(err) / np.linalg.norm(arr["g"])),
                "g_absolute": float(abs(err).max()),
            }
            np.savez_compressed(dst.with_suffix(".npz"), t=t.detach().cpu().numpy()[0])
            rec["arrays_sha256"] = sha(dst.with_suffix(".npz"))
            rec["pass"] = rec["score_error"] <= 2e-6 and (
                rec["g_relative_l2"] <= 1e-4 or rec["g_absolute"] <= 1e-7
            )
            write(dst, rec)
            done.append(job["key"])
            write(
                out / "source_status.json",
                {
                    "status": "RUNNING" if rec["pass"] else "PARITY_ERROR",
                    "completed": done,
                    "seconds": prior["seconds"] + time.monotonic() - started,
                    "source_forwards": len(done),
                    "source_backwards": len(done),
                    "load_seconds": loading,
                },
            )
            print({"source_units": len(done), **rec}, flush=True)
            if not rec["pass"]:
                raise ValueError("Source capture parity failed; no tolerance changes")
        rec, arr = packet_arrays(dst)
        if not rec["pass"]:
            raise ValueError(
                "Prior source parity failure requires an explicit amendment"
            )
        records.append(rec)
        values[job["index"], job["query"]] = arr["t"]
    for job in jobs:
        means = fixed_means(
            values,
            job["query"],
            held=job["index"] if job["cohort"] == "source4" else None,
        )
        np.savez_compressed(out / "private_models" / (job["key"] + ".npz"), **means)
    assets = list((out / "private_models").glob("*.npz")) + list(
        (out / "private_sources").glob("*")
    )
    assets = [p for p in assets if p.is_file() and not p.name.startswith("._")]
    status = read(out / "source_status.json") | {"status": "COMPLETE"}
    write(out / "source_status.json", status)
    write(
        out / "source_freeze.json",
        {
            "manifest_sha256": sha(out / "manifest.json"),
            "source_records": records,
            "status": status,
            "files": {str(p.relative_to(out)): sha(p) for p in sorted(assets)},
        },
    )
    print(
        {"status": "SOURCES_FROZEN", "source_calls": 8, "seconds": status["seconds"]},
        flush=True,
    )


def install_guard(out, jobs):
    """Deny undeclared data/target reads while allowing trusted library code loads."""
    reads = {
        str((out / folder / (j["key"] + ".npz")).resolve())
        for j in jobs
        for folder in ("private_inputs", "private_models")
    }
    outputs = [
        out / "prediction_status.json",
        out / "prediction_freeze.json",
        out / "resource_forecast.json",
    ]
    outputs += [
        out / "private_predictions" / (j["key"] + ext)
        for j in jobs
        for ext in (".npz", ".json")
    ]
    writes = {str(p.resolve()) for p in outputs}
    reads |= writes
    code_roots = [
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        ROOT / "grain_research",
        ROOT / "scripts",
    ]
    denied = []

    def guard(event, args):
        if event != "open":
            return
        path, _, flags = args
        if isinstance(path, int):
            raise PermissionError("Undeclared descriptor")
        path = Path(os.path.realpath(os.fsdecode(path)))
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        code = path.suffix in (".py", ".pyc", ".so", ".dylib") and any(
            path.is_relative_to(r) for r in code_roots
        )
        if (writing and str(path) not in writes) or (
            not writing and str(path) not in reads and not code
        ):
            denied.append(str(path))
            raise PermissionError("Undeclared target/data path denied")

    sys.dont_write_bytecode = True
    b = io.BytesIO()
    np.savez_compressed(b, x=np.zeros(1))
    b.seek(0)
    with np.load(b, allow_pickle=False) as z:
        _ = z["x"]
    # Prime forward AD internals without a native model or a target.
    torch.func.jvp(
        lambda x: x.square(),
        (torch.ones(2, device="mps"),),
        (torch.ones(2, device="mps"),),
    )
    sys.addaudithook(guard)
    for name in ("new8_a00_neuroticism.npz", "new8_a00_neuroticism.json"):
        try:
            with open(NATIVE / "private_packets" / name, "rb"):
                pass
        except PermissionError:
            continue
        raise AssertionError("Target denial failed")
    return denied


def predict(out):
    """Run full forward-mode transport with no current-loss evaluation/backward."""
    m = verify(out)
    source = read(out / "source_freeze.json")
    verify_hashes(out, source["files"])
    if source["manifest_sha256"] != sha(out / "manifest.json"):
        raise ValueError("Sources from a different construction")
    if (out / "prediction_freeze.json").exists():
        print("Already frozen; zero new JVPs")
        return
    jobs = read(out / "private_jobs.json")
    prior = (
        read(out / "prediction_status.json")
        if (out / "prediction_status.json").exists()
        else {"seconds": 0, "completed": [], "jvps": 0, "fd_calls": 0}
    )
    if prior.get("status") == "ERROR":
        raise ValueError("Inspect recorded error before an explicit correction/resume")
    ob, loading = load_observer(m)
    manifest_hash = sha(out / "manifest.json")
    before = not (out / "analysis").exists()
    torch.set_grad_enabled(False)
    denied = install_guard(out, jobs)
    start, done = time.monotonic(), list(prior["completed"])
    jvps, fds = prior["jvps"], prior["fd_calls"]

    def status(kind):
        value = {
            "status": kind,
            "completed": done.copy(),
            "jvps": jvps,
            "fd_calls": fds,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "source_seconds": source["status"]["seconds"],
            "load_seconds": loading,
            "denial_probes": 2,
            "extra_denials": len(denied) - 2,
            "new_author_backwards": 0,
            "new_author_scores": 0,
        }
        write(out / "prediction_status.json", value)
        print(
            {k: v for k, v in value.items() if k != "completed"}
            | {"completed_units": len(done)},
            flush=True,
        )
        return value

    try:
        for job in jobs:
            if job["key"] in done:
                continue
            spent = (
                source["status"]["seconds"]
                + prior["seconds"]
                + time.monotonic()
                - start
            )
            if spent >= m["max_seconds"] or jvps + 4 > m["max_jvps"]:
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            dst = out / "private_predictions" / (job["key"] + ".npz")
            if dst.exists():
                raise ValueError("Unregistered existing packet; inspect before resume")
            state, grain, pos = inputs(out, job)
            kw = cached_suffix_kwargs(ob.model, state)
            with np.load(
                out / "private_models" / (job["key"] + ".npz"), allow_pickle=False
            ) as z:
                means = dict(z)
            arrays, responses, direct, attention, finite = {}, [], [], [], None
            for factor in range(4):
                delta = body_direction(state, grain, pos, factor)
                jvps += 1
                _primal, dot = forward_response(ob.model, state, kw, delta)
                responses.append(dot[0][0].cpu().numpy())
                direct.append(dot[1][0].cpu().numpy())
                attention.append(dot[2][0].cpu().numpy())
                if job["cohort"] == "new8" and job["index"] == 0 and factor == 0:
                    pm = []
                    for sign in (1, -1):
                        fds += 1
                        pm.append(
                            post_attention(ob.model, state + sign * 0.001 * delta, kw)[
                                0
                            ][0]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    fd = (pm[0].astype(float) - pm[1]) / 0.002
                    err = fd - responses[-1]
                    finite = {
                        "epsilon": 0.001,
                        "factor": "whole:F01",
                        "relative_l2": float(
                            np.linalg.norm(err)
                            / max(np.linalg.norm(responses[-1]), 1e-30)
                        ),
                        "absolute": float(abs(err).max()),
                    }
                    finite["pass"] = (
                        finite["relative_l2"] <= 0.05 or finite["absolute"] <= 1e-4
                    )
            arrays.update(
                response=np.asarray(responses),
                direct=np.asarray(direct),
                attention=np.asarray(attention),
            )
            for mode in ("own", "pooled"):
                for channel in ("response", "direct", "attention"):
                    arrays[mode + "_" + channel] = arrays[channel].astype(
                        float
                    ) @ means[mode + "_t"].astype(float)
            closure = float(
                abs(arrays["response"] - arrays["direct"] - arrays["attention"]).max()
            )
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite forward responses")
            np.savez_compressed(dst, **arrays)
            write(
                dst.with_suffix(".json"),
                {
                    "key": job["key"],
                    "arrays_sha256": sha(dst),
                    "finite_difference": finite,
                    "response_addition_max_error": closure,
                },
            )
            done.append(job["key"])
            status("RUNNING")
            if finite is not None and not finite["pass"]:
                raise ValueError(
                    "Fixed finite-difference check failed; packet retained"
                )
            if len(done) == 2 and not prior["completed"]:
                forecast = (time.monotonic() - start) * len(jobs) / 2 + source[
                    "status"
                ]["seconds"]
                write(
                    out / "resource_forecast.json",
                    {"forecast_seconds": forecast, "native_cap": m["max_seconds"]},
                )
                if forecast > m["max_seconds"]:
                    status("PAUSED_FORECAST")
                    return
        final = status("COMPLETE")
        if (
            final["extra_denials"]
            or fds != 4
            or jvps != 96
            or final["seconds"] + source["status"]["seconds"] > m["max_seconds"]
        ):
            raise ValueError("Runtime boundary differs from frozen plan")
        files = [
            out / "private_predictions" / (j["key"] + ext)
            for j in jobs
            for ext in (".npz", ".json")
        ]
        write(
            out / "prediction_freeze.json",
            {
                "manifest_sha256": manifest_hash,
                "before_evaluation": before,
                "runtime": final,
                "files": {str(p.relative_to(out)): sha(p) for p in files},
            },
        )
        print("PREDICTIONS_FROZEN", flush=True)
    except BaseException:
        status("ERROR")
        raise


def evaluate(out, name):
    """Open existing H16 target gradients only after forward predictions freeze."""
    verify(out)
    frozen = read(out / "prediction_freeze.json")
    if not frozen["before_evaluation"] or frozen["manifest_sha256"] != sha(
        out / "manifest.json"
    ):
        raise ValueError("Prediction freeze must precede evaluation")
    verify_hashes(out, frozen["files"])
    dest = out / name
    dest.mkdir(exist_ok=False)
    rows, checks, native_hashes = [], [], {}
    for job in read(out / "private_jobs.json"):
        ref = NATIVE / "private_packets" / (job["key"] + ".json")
        _, native = packet_arrays(ref)
        native_hashes[str(ref.relative_to(ROOT))] = sha(ref)
        native_hashes[str(ref.with_suffix(".npz").relative_to(ROOT))] = sha(
            ref.with_suffix(".npz")
        )
        state, grain, pos = inputs(out, job)
        actual = np.asarray(
            [
                float(
                    np.sum(
                        body_direction(state, grain, pos, f)[0, pos]
                        .cpu()
                        .numpy()
                        .astype(float)
                        * native["body_gradient"]
                    )
                )
                for f in range(4)
            ]
        )
        rec, arr = packet_arrays(out / "private_predictions" / (job["key"] + ".json"))
        with np.load(
            out / "private_models" / (job["key"] + ".npz"), allow_pickle=False
        ) as z:
            means = dict(z)
        replay_error = 0.0
        for mode in ("own", "pooled"):
            for channel in ("response", "direct", "attention"):
                reconstructed = arr[channel].astype(float) @ means[mode + "_t"].astype(
                    float
                )
                replay_error = max(
                    replay_error,
                    float(abs(reconstructed - arr[mode + "_" + channel]).max()),
                )
                for f in range(4):
                    rows.append(
                        job
                        | {
                            "mode": mode,
                            "channel": channel,
                            "factor": f"whole:F{f + 1:02d}",
                            "actual": actual[f],
                            "predicted": reconstructed[f],
                            "direct": arr[mode + "_direct"][f],
                            "attention": arr[mode + "_attention"][f],
                        }
                    )
        checks.append(
            job
            | {
                "replay_error": replay_error,
                "addition_error": rec["response_addition_max_error"],
                "finite_difference": rec["finite_difference"],
            }
        )
    actions = pd.DataFrame(rows)
    comparison, factors = [], []
    keys = ["cohort", "query", "mode", "channel"]
    for key, a in actions.groupby(keys, sort=True):
        meta = dict(zip(keys, key, strict=True))
        centered = a[["actual", "predicted"]] - a.groupby("factor")[
            ["actual", "predicted"]
        ].transform("mean")
        comparison.append(
            meta
            | {
                "authors": a["index"].nunique(),
                **action_metrics(a.actual, a.predicted),
                "factor_centered_action_r": action_metrics(
                    centered.actual, centered.predicted
                )["action_r"],
            }
        )
        for factor, b in a.groupby("factor", sort=True):
            factors.append(
                meta | {"factor": factor, **action_metrics(b.actual, b.predicted)}
            )
    actions.to_csv(dest / "private_actions.csv", index=False)
    pd.DataFrame(comparison).to_csv(dest / "comparison.csv", index=False)
    pd.DataFrame(factors).to_csv(dest / "factor_actions.csv", index=False)
    write(dest / "private_checks.json", checks)
    summary = {
        "units": 24,
        "action_conditions": len(actions),
        "replay_max_error": max(r["replay_error"] for r in checks),
        "addition_max_error": max(r["addition_error"] for r in checks),
        "finite_differences": [
            r["finite_difference"] for r in checks if r["finite_difference"] is not None
        ],
        "source_forwards": 8,
        "source_backwards": 8,
        "runtime": frozen["runtime"],
        "fits": 0,
        "human_labels_read": False,
        "files": {
            p.name: sha(p)
            for p in sorted(dest.glob("*.csv"))
            if not p.name.startswith("._")
        },
        "native_target_hashes": native_hashes,
    }
    if summary["replay_max_error"] > 1e-12:
        raise ValueError("Saved response contraction replay differs")
    write(dest / "summary.json", summary)
    print(
        {
            k: v
            for k, v in summary.items()
            if k not in ("files", "native_target_hashes", "runtime")
        },
        flush=True,
    )


def main():
    """Keep source capture, target-denied prediction and evaluation in separate processes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "sources", "predict", "evaluate"))
    p.add_argument("--name", default="logic_complete_transport_jvp_v1")
    p.add_argument("--analysis-name", default="analysis")
    a = p.parse_args()
    if any(Path(n).name != n for n in (a.name, a.analysis_name)):
        raise ValueError("Simple output names required")
    out = BASE / a.name
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(out)
        elif a.stage == "sources":
            sources(out)
        elif a.stage == "predict":
            predict(out)
        else:
            evaluate(out, a.analysis_name)


if __name__ == "__main__":
    main()
