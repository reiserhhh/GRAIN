#!/usr/bin/env python3
"""NEXT63: source-only native query-last replacement, fixed BODY and GRAIN directions."""

from __future__ import annotations

import argparse
import inspect
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.context_reading_interaction import (
    account,
    native_body_gradient,
    other_source_last,
    replace_last,
)
from grain_research.vector_mechanisms.hermite_token_potential import (
    predict as additive_predict,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_hermite_token_potential as prior

old = prior.old
OUT = old.BASE / "logic_context_reading_interaction_v1"
NATIVE = old.BASE / "logic_fixed_seed_attention_v1"
PLAN = ROOT / "docs/GRAIN_CONTEXT_READING_INTERACTION_PLAN_V1.md"


def verify(out):
    """Verify primary/source hashes only; no application/new8 data read."""
    m = old.read(out / "manifest.json")
    old.verify_hashes(ROOT, m["inputs"])
    old.verify_hashes(out, {"private_jobs.json": m["jobs_sha256"]})
    return m


def prepare(out):
    """Bind source4, native contrast and exact unchanged H16 inputs before intervention."""
    start = time.process_time()
    m = prior.manifest(prior.OUT)
    jobs = old.read(prior.OUT / "private_jobs.json")
    src = [j for j in jobs if j["cohort"] == "source4"]
    if len(src) != 4 or {j["index"] for j in src} != set(range(4)):
        raise ValueError("Four original paired-query sources required")
    source_paths = [
        p
        for j in src
        for k in j["query_keys"]
        for p in (old.input_path(k), old.gradient_path(k), old.target_path(j, k))
    ]
    old.check_paths(m, source_paths)
    model_freeze = old.frozen(prior.OUT, "models")
    models = [
        prior.model_path(prior.OUT, old.ref(j), q) for j in src for q in old.QUERIES
    ]
    old.verify_hashes(
        prior.OUT / "models", {p.name: model_freeze["files"][p.name] for p in models}
    )
    native = old.read(NATIVE / "manifest.json")
    from transformers.models.qwen2 import modeling_qwen2

    if (
        old.sha(Path(inspect.getfile(modeling_qwen2)))
        != native["qwen_implementation_sha256"]
    ):
        raise ValueError("Original suffix implementation changed")
    files = (
        source_paths
        + models
        + [
            PLAN,
            Path(__file__),
            NATIVE / "manifest.json",
            NATIVE / "observer.json",
            ROOT / "grain_research/vector_mechanisms/context_reading_interaction.py",
            ROOT / "grain_research/vector_mechanisms/hermite_token_potential.py",
            ROOT / "grain_research/vector_mechanisms/token_potential.py",
            ROOT / "grain_research/vector_mechanisms/context_carrier.py",
            ROOT / "grain_research/vector_mechanisms/selection_endpoint.py",
            ROOT / "grain_research/vector_mechanisms/stage_file_guard.py",
            ROOT / "scripts/run_grain_complete_transport_jvp.py",
            ROOT / "scripts/run_grain_token_potential_extraction.py",
            ROOT / "tests/test_grain_context_reading_interaction.py",
        ]
    )
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", src)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONTEXT_READING_INTERACTION_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "inputs": {str(p.relative_to(ROOT)): old.sha(p) for p in files},
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "model_sha256": native["model_sha256"],
            "alphabet": native["alphabet"],
            "c": native["c"],
            "qwen_implementation_sha256": native["qwen_implementation_sha256"],
            "max_calls": 10,
            "planned_calls": 9,
            "max_native_seconds": 120,
            "max_loading_seconds": 300,
            "replay_score_atol": 1e-4,
            "replay_gradient_relative_tolerance": 3e-4,
            "prepare_cpu_seconds": time.process_time() - start,
            "human_labels_read": False,
            "application_target_reads": 0,
        },
    )
    print({"status": "PREPARED", "source_contexts": 8, "planned_calls": 9}, flush=True)


def capture(out):
    """One native unchanged replay plus eight single-state interventions, no new fitting."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from scripts.run_grain_complete_transport_jvp import load_observer

    m = verify(out)
    jobs = old.read(out / "private_jobs.json")
    dst = out / "capture"
    if (dst / "status.json").exists() and old.read(dst / "status.json")[
        "status"
    ] == "COMPLETE":
        print("Already complete; zero new neural calls")
        return
    dst.mkdir(exist_ok=False)
    stats = {
        "status": "LOADING",
        "calls": 0,
        "native_seconds": 0.0,
        "loading_seconds": 0.0,
        "completed": [],
        "fits": 0,
        "application_target_reads": 0,
        "human_labels_read": False,
    }
    status_path = dst / "status.json"
    old.write(status_path, stats)

    def alarm(_sig, _frame):
        raise TimeoutError("Frozen observer load wall cap exceeded")

    original_handler = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(m["max_loading_seconds"])
    try:
        ob, loading = load_observer(m)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
    stats["loading_seconds"] = loading
    torch.set_grad_enabled(False)
    ids = torch.tensor(m["alphabet"]["token_ids"], device="mps")
    contrast = torch.tensor(m["c"], device="mps", dtype=torch.float32)
    reads = [ROOT / p for p in m["inputs"]]
    outputs = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    outputs += [dst / "unchanged_replay.npz", status_path]
    all_jobs = old.read(prior.OUT / "private_jobs.json")
    probes = old.forbidden() + [
        p
        for j in all_jobs
        if j["cohort"] == "new8"
        for k in j["query_keys"]
        for p in (old.input_path(k), old.gradient_path(k), old.target_path(j, k))
    ]
    denied = install(
        reads, outputs, probes, [ROOT / "scripts", ROOT / "grain_research"]
    )
    bank = {
        (j["index"], q): old.pure_input(k)[2]
        for j in jobs
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True)
    }
    paths, replay_checks = [], {}

    def status(kind):
        stats.update(
            status=kind,
            files={p.name: old.sha(p) for p in paths},
            replay_checks=replay_checks.copy(),
            denial_probes=len(probes),
            extra_denials=len(denied) - len(probes),
        )
        old.write(status_path, stats)
        print(
            {
                k: v
                for k, v in stats.items()
                if k not in ("files", "completed", "replay_checks")
            }
            | {"completed_contexts": len(stats["completed"])},
            flush=True,
        )

    def timed(x, kwargs, pos):
        if (
            stats["calls"] >= m["max_calls"]
            or stats["native_seconds"] >= m["max_native_seconds"]
        ):
            raise TimeoutError("Frozen native cap reached")
        stats["calls"] += 1
        status("RUNNING")
        torch.mps.synchronize()
        start = time.monotonic()
        try:
            return native_body_gradient(ob.model, x, kwargs, ids, contrast, pos)
        finally:
            torch.mps.synchronize()
            stats["native_seconds"] += time.monotonic() - start

    try:
        for j in jobs:
            for q, k in zip(old.QUERIES, j["query_keys"], strict=True):
                data, _, last = old.pure_input(k)
                state = torch.tensor(data["state"][None], device="mps")
                kw = cached_suffix_kwargs(ob.model, state)
                for a, n in zip(kw["position_embeddings"], ("cos", "sin"), strict=True):
                    np.testing.assert_array_equal(a.cpu().numpy()[0], data[n])
                np.testing.assert_array_equal(
                    kw["attention_mask"][0, 0, -1].cpu().numpy(), data["mask"]
                )
                pos = data["positions"]
                if not paths:
                    score, g = timed(state, kw, pos)
                    target = old.selected(old.target_path(j, k), ("score",))["score"]
                    native = old.selected(old.gradient_path(k), ("body_gradient",))[
                        "body_gradient"
                    ].astype(float)
                    ds = old.body_directions(data)[:, pos]
                    dg = g.astype(float) - native
                    replay_checks.update(
                        score_abs_error=abs(float(score - target)),
                        gradient_relative_error=float(
                            np.linalg.norm(dg) / np.linalg.norm(native)
                        ),
                        gradient_delta_rms=float(np.sqrt(np.mean(dg * dg))),
                        action_abs_error=float(
                            np.max(abs(np.einsum("th,fth->f", dg, ds)))
                        ),
                    )
                    p = dst / "unchanged_replay.npz"
                    np.savez_compressed(p, score=np.array(score), body_gradient=g)
                    paths.append(p)
                    if (
                        replay_checks["score_abs_error"] > m["replay_score_atol"]
                        or replay_checks["gradient_relative_error"]
                        > m["replay_gradient_relative_tolerance"]
                    ):
                        raise ValueError(
                            "Unchanged native replay mismatch; do not change targets"
                        )
                last_ref, members = other_source_last(bank, j["index"], q, old.QUERIES)
                changed = replace_last(data["state"], last_ref, pos)
                score, g = timed(torch.tensor(changed[None], device="mps"), kw, pos)
                p = dst / (k + ".npz")
                np.savez_compressed(
                    p,
                    score=np.array(score),
                    body_gradient=g,
                    replacement_last=changed[-1],
                    members=np.array(members),
                    last_displacement=np.array(
                        np.linalg.norm(changed[-1].astype(float) - last)
                    ),
                )
                paths.append(p)
                stats["completed"].append(k)
                if len(stats["completed"]) == 2:
                    stats["projected_native_seconds"] = (
                        stats["native_seconds"] * 9 / stats["calls"]
                    )
                    if stats["projected_native_seconds"] > m["max_native_seconds"]:
                        raise TimeoutError(
                            "First pair exceeds projected cost allowance"
                        )
                status("RUNNING")
        if (
            stats["calls"] != 9
            or stats["native_seconds"] > m["max_native_seconds"]
            or len(denied) != len(probes)
        ):
            raise ValueError("Completed call/cost/access contract differs")
        status("COMPLETE")
    except BaseException:
        status("FAILED_OR_BOUNDED_STOP")
        raise


def analyze(out, folder):
    """Use frozen source interventions to account changes and the additive zero witness."""
    start = time.process_time()
    m = verify(out)
    status = old.read(out / "capture/status.json")
    if status["status"] != "COMPLETE":
        raise ValueError("Complete bounded source capture required")
    old.verify_hashes(out / "capture", status["files"])
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    rows, actions = [], []
    for j in jobs:
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True):
            data, h, last = old.pure_input(k)
            packet = old.arrays(out / "capture" / (k + ".npz"))
            g = old.selected(old.gradient_path(k), ("body_gradient",))[
                "body_gradient"
            ].astype(float)
            target = old.selected(old.target_path(j, k), ("score", "tau"))
            ds = old.body_directions(data)[:, data["positions"]]
            model = old.arrays(prior.model_path(prior.OUT, old.ref(j), q))
            expected = [i for i in range(4) if i != j["index"]]
            np.testing.assert_array_equal(packet["members"], expected)
            validate_members(model, expected)
            record, act = account(g, packet["body_gradient"], ds, model["basis"])
            np.testing.assert_allclose(act["tau"], target["tau"], atol=1e-5, rtol=3e-4)
            s0, f0 = additive_predict(h, last, model)
            s1, f1 = additive_predict(h, packet["replacement_last"], model)
            np.testing.assert_array_equal(f0, f1)
            delta_add = hidden_gradient(f1 - f0, model)
            record.update(
                additive_field_change=float(np.linalg.norm(delta_add)),
                additive_action_change=float(
                    np.max(abs(np.einsum("th,fth->f", delta_add, ds)))
                ),
                additive_score_change=float(s1 - s0),
                native_score_change=float(packet["score"] - target["score"]),
                last_displacement=float(packet["last_displacement"]),
                last_relative_displacement=float(packet["last_displacement"])
                / float(np.linalg.norm(last)),
                field_change_above_replay=bool(
                    record["field_change_ratio"]
                    > 10
                    * max(status["replay_checks"]["gradient_relative_error"], 1e-12)
                ),
            )
            meta = {"author_key": j["key"], "query_name": q}
            rows.append(meta | record)
            for fi in range(4):
                actions.append(
                    meta
                    | {"factor": f"whole:F{fi + 1:02d}"}
                    | {n: float(a[fi]) for n, a in act.items()}
                )
    sd, ad = pd.DataFrame(rows), pd.DataFrame(actions)
    tables = {"private_contexts": sd, "private_actions": ad}
    public = []
    for q, g in sd.groupby("query_name", sort=True):
        row = {"query_name": q, "authors": len(g)}
        for n in (
            "field_change_ratio",
            "inside_field_fraction",
            "last_displacement",
            "last_relative_displacement",
            "native_score_change",
            "additive_score_change",
        ):
            row[n + "_mean"] = float(g[n].mean())
            row[n + "_min"] = float(g[n].min())
            row[n + "_max"] = float(g[n].max())
        for n in (
            "additive_field_change",
            "additive_action_change",
            "energy_relative_error",
            "action_closure_error",
            "finite_action_error",
        ):
            row[n + "_max"] = float(g[n].max())
        row["field_changes_above_replay"] = int(g.field_change_above_replay.sum())
        public.append(row)
    tables["context_summary"] = pd.DataFrame(public)
    for keys, name in (
        (["query_name"], "action_summary"),
        (["query_name", "factor"], "factor_summary"),
    ):
        result = []
        for key, g in ad.groupby(keys, sort=True):
            delta, inside, outside = (
                g[n].to_numpy()
                for n in ("delta_tau", "inside_delta_tau", "outside_delta_tau")
            )
            energy = float(np.mean(delta * delta))
            row = dict(
                zip(keys, key if isinstance(key, tuple) else (key,), strict=True)
            )
            row.update(
                authors=g.author_key.nunique(),
                actions=len(g),
                native_delta_rms=float(np.sqrt(energy)),
                original_action_rms=float(np.sqrt(np.mean(g.tau**2))),
                changed_action_rms=float(np.sqrt(np.mean(g.changed_tau**2))),
                inside_delta_rms=float(np.sqrt(np.mean(inside * inside))),
                outside_delta_rms=float(np.sqrt(np.mean(outside * outside))),
                inside_signed_projection=float(np.mean(inside * delta) / energy)
                if energy
                else 0.0,
                outside_signed_projection=float(np.mean(outside * delta) / energy)
                if energy
                else 0.0,
                inside_outside_covariance=float(
                    np.mean((inside - inside.mean()) * (outside - outside.mean()))
                ),
                actions_above_replay=int(
                    np.sum(
                        abs(delta)
                        > 10 * max(status["replay_checks"]["action_abs_error"], 1e-12)
                    )
                ),
            )
            result.append(row)
        tables[name] = pd.DataFrame(result)
    for name, table in tables.items():
        table.to_csv(dst / (name + ".csv"), index=False)
    old.write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": time.process_time() - start,
            "native_calls": status["calls"],
            "native_seconds": status["native_seconds"],
            "prepare_cpu_seconds": m["prepare_cpu_seconds"],
            "fits": 0,
            "application_target_reads": 0,
            "replay_checks": status["replay_checks"],
            "files": {(n + ".csv"): old.sha(dst / (n + ".csv")) for n in tables},
        },
    )
    print(
        {"status": "ANALYZED", "folder": folder, "source_contexts": len(sd)}, flush=True
    )


def audit(out):
    """Verify frozen sources, five-table replay and independent finite response identities."""
    start = time.process_time()
    verify(out)
    status = old.read(out / "capture/status.json")
    old.verify_hashes(out / "capture", status["files"])
    summaries = [old.read(out / n / "summary.json") for n in ("analysis", "replay")]
    for n, s in zip(("analysis", "replay"), summaries, strict=True):
        old.verify_hashes(out / n, s["files"])
    if summaries[0]["files"] != summaries[1]["files"]:
        raise ValueError("Finite statistics replay differs")
    a = pd.read_csv(out / "analysis/private_actions.csv")
    c = pd.read_csv(out / "analysis/private_contexts.csv")
    error = float(
        np.max(abs(a.changed_tau - a.tau - a.inside_delta_tau - a.outside_delta_tau))
    )
    if (
        error > 1e-8
        or c.additive_field_change.max() != 0
        or c.additive_action_change.max() != 0
        or len(a) != 32
        or len(c) != 8
        or status["calls"] != 9
        or status["extra_denials"]
        or status["native_seconds"] > 120
        or status["loading_seconds"] > 300
    ):
        raise ValueError("Fixed intervention/accounting boundary differs")
    old.write(
        out / "audit.json",
        {
            "status": "PASS",
            "native_calls": status["calls"],
            "native_seconds": status["native_seconds"],
            "source_contexts": 8,
            "actions": 32,
            "logical_csv_replay": 5,
            "new_fits": 0,
            "action_identity_error": error,
            "additive_zero_exact": True,
            "cpu_seconds": time.process_time() - start,
            "application_target_reads": 0,
            "human_labels_read": False,
        },
    )
    print(
        {"status": "PASS", "native_calls": 9, "action_identity_error": error},
        flush=True,
    )


def main():
    """Run one explicit source experiment stage with bounded compute."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", required=True, choices=("prepare", "capture", "analyze", "audit")
    )
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--folder", choices=("analysis", "replay"), default="analysis")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage == "analyze":
            analyze(a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
