#!/usr/bin/env python3
"""NEXT67: bounded native cross-BODY intervention with a frozen local-sum comparator."""

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
from grain_research.vector_mechanisms import full_context_geometry as scalar
from grain_research.vector_mechanisms.context_reading_interaction import (
    native_body_gradient,
)
from grain_research.vector_mechanisms.cross_token_reading import (
    intervention,
    region_account,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_context_reading_interaction as native_prior
from scripts import run_grain_paired_response_objective as prior

old = native_prior.old
OUT = old.BASE / "logic_cross_token_reading_v1"
PLAN = ROOT / "docs/GRAIN_CROSS_TOKEN_READING_PLAN_V1.md"


def verify(out):
    """Check only source-side inputs, code and job hashes."""
    m = old.read(out / "manifest.json")
    old.verify_hashes(ROOT, m["inputs"])
    old.verify_hashes(out, {"private_jobs.json": m["jobs_sha256"]})
    return m


def prepare(out):
    """Bind pre-result code and existing source-only native/model evidence."""
    start = time.process_time()
    native = native_prior.verify(native_prior.OUT)
    jobs = old.read(native_prior.OUT / "private_jobs.json")
    if len(jobs) != 4 or {j["index"] for j in jobs} != set(range(4)):
        raise ValueError("Exactly four original source authors required")
    if old.read(prior.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("NEXT66 must have passed")
    freeze = old.frozen(prior.OUT, "models")
    models = [prior.prior.model_path(prior.OUT, q) for q in old.QUERIES]
    old.verify_hashes(
        prior.OUT / "models", {p.name: freeze["files"][p.name] for p in models}
    )
    from transformers.models.qwen2 import modeling_qwen2

    if (
        old.sha(Path(inspect.getfile(modeling_qwen2)))
        != native["qwen_implementation_sha256"]
    ):
        raise ValueError("Frozen Qwen suffix implementation changed")
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/cross_token_reading.py",
        ROOT / "tests/test_grain_cross_token_reading.py",
        ROOT / "grain_research/vector_mechanisms/full_context_geometry.py",
        ROOT / "grain_research/vector_mechanisms/context_modulated_potential.py",
        ROOT / "grain_research/vector_mechanisms/position_reader.py",
    ]
    files = [ROOT / p for p in native["inputs"]] + code + models
    files += [
        native_prior.OUT / "audit.json",
        prior.OUT / "audit.json",
        prior.OUT / "models/freeze.json",
    ]
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CROSS_TOKEN_READING_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "inputs": {str(p.relative_to(ROOT)): old.sha(p) for p in files},
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            **{
                k: native[k]
                for k in (
                    "model_sha256",
                    "alphabet",
                    "c",
                    "qwen_implementation_sha256",
                    "replay_score_atol",
                    "replay_gradient_relative_tolerance",
                )
            },
            "max_calls": 17,
            "max_native_seconds": 120,
            "max_loading_seconds": 300,
            "prepare_cpu_seconds": time.process_time() - start,
            "human_labels_read": False,
            "application_inputs_or_targets_read": False,
        },
    )
    print({"status": "PREPARED", "source_contexts": 8, "max_calls": 17}, flush=True)


def capture(out):
    """Capture 1 replay and 16 interventions; preserve every completed native call."""
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
        "human_labels_read": False,
    }
    sp = dst / "status.json"
    old.write(sp, stats)

    def alarm(_sig, _frame):
        raise TimeoutError("Frozen loading cap reached")

    handler = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(m["max_loading_seconds"])
    try:
        ob, loading = load_observer(m)
    except BaseException as exc:
        stats.update(status="LOAD_FAILED", error_type=type(exc).__name__)
        old.write(sp, stats)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
    stats["loading_seconds"] = loading
    torch.set_grad_enabled(False)
    ids = torch.tensor(m["alphabet"]["token_ids"], device="mps")
    contrast = torch.tensor(m["c"], device="mps", dtype=torch.float32)
    reads = [ROOT / p for p in m["inputs"]]
    outputs = [
        dst / (k + suffix + ".npz")
        for j in jobs
        for k in j["query_keys"]
        for suffix in ("_design", "_plus", "_minus")
    ]
    outputs += [dst / "unchanged_replay.npz", sp]
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
    paths, replay = [], {}

    def status(kind):
        stats.update(
            status=kind,
            files={p.name: old.sha(p) for p in paths},
            replay_checks=replay.copy(),
            denial_probes=len(probes),
            extra_denials=len(denied) - len(probes),
        )
        old.write(sp, stats)
        print(
            {
                k: stats[k]
                for k in ("status", "calls", "native_seconds", "loading_seconds")
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
                data, _, _ = old.pure_input(k)
                pos = data["positions"]
                ds = old.body_directions(data)[:, pos]
                plus, minus, meta = intervention(data["state"], pos, ds)
                p = dst / (k + "_design.npz")
                np.savez_compressed(p, **meta)
                paths.append(p)
                state = torch.tensor(data["state"][None], device="mps")
                kwargs = cached_suffix_kwargs(ob.model, state)
                for a, n in zip(
                    kwargs["position_embeddings"], ("cos", "sin"), strict=True
                ):
                    np.testing.assert_array_equal(a.cpu().numpy()[0], data[n])
                np.testing.assert_array_equal(
                    kwargs["attention_mask"][0, 0, -1].cpu().numpy(), data["mask"]
                )
                if stats["calls"] == 0:
                    score, g = timed(state, kwargs, pos)
                    native = old.selected(old.gradient_path(k), ("body_gradient",))[
                        "body_gradient"
                    ].astype(float)
                    target = float(
                        old.selected(old.target_path(j, k), ("score",))["score"]
                    )
                    dg = g.astype(float) - native
                    cut = int(meta["cut"])
                    replay.update(
                        score_abs_error=abs(score - target),
                        gradient_relative_error=float(
                            np.linalg.norm(dg) / np.linalg.norm(native)
                        ),
                        B_relative_error=float(
                            np.linalg.norm(dg[cut:]) / np.linalg.norm(native[cut:])
                        ),
                        B_action_abs_error=float(
                            np.max(abs(np.einsum("th,fth->f", dg[cut:], ds[:, cut:])))
                        ),
                    )
                    p = dst / "unchanged_replay.npz"
                    np.savez_compressed(p, score=np.array(score), body_gradient=g)
                    paths.append(p)
                    if (
                        replay["score_abs_error"] > m["replay_score_atol"]
                        or replay["gradient_relative_error"]
                        > m["replay_gradient_relative_tolerance"]
                    ):
                        raise ValueError("Unchanged native replay mismatch")
                for name, x in (("plus", plus), ("minus", minus)):
                    score, g = timed(torch.tensor(x[None], device="mps"), kwargs, pos)
                    p = dst / (k + "_" + name + ".npz")
                    np.savez_compressed(p, score=np.array(score), body_gradient=g)
                    paths.append(p)
                    status("RUNNING")
                stats["completed"].append(k)
                if len(stats["completed"]) == 2:
                    stats["projected_native_seconds"] = (
                        stats["native_seconds"] * 17 / stats["calls"]
                    )
                    if stats["projected_native_seconds"] > m["max_native_seconds"] - 20:
                        raise TimeoutError("Projected native cost exhausts reserve")
        if (
            stats["calls"] != 17
            or stats["native_seconds"] > 120
            or len(denied) != len(probes)
        ):
            raise ValueError("Call/cost/access contract differs")
        status("COMPLETE")
    except BaseException as exc:
        stats["error_type"] = type(exc).__name__
        status("FAILED_OR_BOUNDED_STOP")
        raise


def analyze(out, folder):
    """Account all eight source cells and all fixed factors, without predictive fitting."""
    start = time.process_time()
    verify(out)
    status = old.read(out / "capture/status.json")
    if status["status"] != "COMPLETE":
        raise ValueError("Complete capture required; preserve partial runs separately")
    old.verify_hashes(out / "capture", status["files"])
    dst = out / folder
    dst.mkdir(exist_ok=False)
    jobs = old.read(out / "private_jobs.json")
    rows, actions = [], []
    for j in jobs:
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True):
            data, _, last = old.pure_input(k)
            pos = data["positions"]
            ds = old.body_directions(data)[:, pos]
            plus, minus, meta = intervention(data["state"], pos, ds)
            saved = old.arrays(out / "capture" / (k + "_design.npz"))
            for n in meta:
                np.testing.assert_array_equal(meta[n], saved[n])
            cut = int(meta["cut"])
            pk = [
                old.arrays(out / "capture" / (k + "_" + n + ".npz"))
                for n in ("plus", "minus")
            ]
            g = old.selected(old.gradient_path(k), ("body_gradient",))[
                "body_gradient"
            ].astype(float)
            record, acts = region_account(
                g, pk[0]["body_gradient"], pk[1]["body_gradient"], ds, cut
            )
            model = old.arrays(prior.prior.model_path(prior.OUT, q))
            validate_members(model, [0, 1, 2, 3])
            controls = [
                scalar.predict(x[pos].astype(float), last, model) for x in (plus, minus)
            ]
            dc = hidden_gradient((controls[0][1] - controls[1][1]) / 2, model)
            zero = float(np.linalg.norm(dc[cut:]))
            if zero > 1e-12:
                raise ValueError("Local-sum untouched-region zero violated")
            common = {
                "author_key": j["key"],
                "query_name": q,
                "native_score_half_change": float(
                    (pk[0]["score"] - pk[1]["score"]) / 2
                ),
                "local_scalar_score_half_change": (controls[0][0] - controls[1][0]) / 2,
                "local_B_gradient_change": zero,
                "B_change_above_replay": record["B"]["relative_change"]
                > 10 * max(status["replay_checks"]["B_relative_error"], 1e-12),
                **{
                    n: float(meta[n])
                    for n in (
                        "epsilon",
                        "realized_plus_norm",
                        "realized_minus_norm",
                        "midpoint_error",
                    )
                },
                "A_tokens": cut,
                "B_tokens": len(pos) - cut,
            }
            for region in ("A", "B"):
                rows.append(common | {"region": region} | record[region])
                for fi in range(4):
                    actions.append(
                        {
                            "author_key": j["key"],
                            "query_name": q,
                            "region": region,
                            "factor": f"whole:F{fi + 1:02d}",
                        }
                        | {n: float(v[fi]) for n, v in acts[region].items()}
                    )
    rd, ad = pd.DataFrame(rows), pd.DataFrame(actions)
    tables = {"private_regions": rd, "private_actions": ad}
    pub = []
    for (q, region), group in rd.groupby(["query_name", "region"], sort=True):
        row = {
            "query_name": q,
            "region": region,
            "authors": len(group),
            "B_changes_above_replay": int(group.B_change_above_replay.sum()),
            "local_B_gradient_change_max": float(group.local_B_gradient_change.max()),
        }
        for n in (
            "relative_change",
            "baseline_norm",
            "delta_norm",
            "native_score_half_change",
            "local_scalar_score_half_change",
            "epsilon",
            "realized_plus_norm",
            "realized_minus_norm",
            "midpoint_error",
        ):
            for stat in ("mean", "min", "max"):
                row[n + "_" + stat] = float(getattr(group[n], stat)())
        pub.append(row)
    tables["region_summary"] = pd.DataFrame(pub)
    for keys, name in (
        (["query_name", "region"], "action_summary"),
        (["query_name", "region", "factor"], "factor_summary"),
    ):
        records = []
        for key, group in ad.groupby(keys, sort=True):
            drms = float(np.sqrt(np.mean(group.delta**2)))
            brms = float(np.sqrt(np.mean(group.original**2)))
            records.append(
                dict(zip(keys, key, strict=True))
                | {
                    "authors": group.author_key.nunique(),
                    "actions": len(group),
                    "delta_rms": drms,
                    "baseline_rms": brms,
                    "relative_rms": drms / brms if brms else None,
                    "delta_mean": float(group.delta.mean()),
                    "actions_above_replay": int(
                        (
                            abs(group.delta)
                            > 10
                            * max(status["replay_checks"]["B_action_abs_error"], 1e-12)
                        ).sum()
                    ),
                }
            )
        tables[name] = pd.DataFrame(records)
    for name, table in tables.items():
        table.to_csv(dst / (name + ".csv"), index=False)
    old.write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": time.process_time() - start,
            "files": {n + ".csv": old.sha(dst / (n + ".csv")) for n in tables},
        },
    )
    print(
        {
            "status": "ANALYZED",
            "folder": folder,
            "contexts": 8,
            "regions": len(rd),
            "actions": len(ad),
        },
        flush=True,
    )


def audit(out):
    """Verify exact arithmetic replay, fixed-region preservation and zero comparator."""
    start = time.process_time()
    verify(out)
    status = old.read(out / "capture/status.json")
    old.verify_hashes(out / "capture", status["files"])
    summaries = [old.read(out / f / "summary.json") for f in ("analysis", "replay")]
    for f, s in zip(("analysis", "replay"), summaries, strict=True):
        old.verify_hashes(out / f, s["files"])
    if summaries[0]["files"] != summaries[1]["files"]:
        raise ValueError("Analysis replay differs")
    a = pd.read_csv(out / "analysis/private_actions.csv")
    r = pd.read_csv(out / "analysis/private_regions.csv")
    error = float(np.max(abs(a.delta - (a.plus - a.minus) / 2)))
    if (
        status["status"] != "COMPLETE"
        or status["calls"] != 17
        or len(status["completed"]) != 8
        or len(status["files"]) != 25
        or len(a) != 64
        or len(r) != 16
        or error > 1e-8
        or r.local_B_gradient_change.max() > 1e-12
        or status["extra_denials"]
        or status["native_seconds"] > 120
        or status["loading_seconds"] > 300
    ):
        raise ValueError("Frozen experimental contract differs")
    result = {
        "status": "PASS",
        "native_calls": 17,
        "native_seconds": status["native_seconds"],
        "loading_seconds": status["loading_seconds"],
        "logical_csv_replay": 5,
        "source_contexts": 8,
        "actions": 64,
        "new_fits": 0,
        "local_B_gradient_change_max": float(r.local_B_gradient_change.max()),
        "finite_action_identity_error": error,
        "replay_checks": status["replay_checks"],
        "cpu_seconds": time.process_time() - start,
        "human_labels_read": False,
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run an explicit bounded stage without restarting any earlier queue."""
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
