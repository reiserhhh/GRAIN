#!/usr/bin/env python3
"""NEXT49: bounded full-sequence transport and exact attention22 Q/K/V channels."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.compiled_query_transfer import MODES, QUERIES
from grain_research.vector_mechanisms.query_counteraction import vector_check
from grain_research.vector_mechanisms.upstream_attention_channels import (
    CHANNELS,
    decompose,
)
from scripts.run_grain_compiled_query_transfer import (
    BASE,
    arrays,
    read,
    sha,
    verify,
    verify_hashes,
    write,
)
from scripts.run_grain_query_counteraction import guard

SOURCE = BASE / "logic_query_transport_formation_v1"
NATIVE = BASE / "logic_layer_accumulation_v1"
PLAN = ROOT / "docs/GRAIN_UPSTREAM_ATTENTION_CHANNELS_PLAN_V1.md"


def prepare(out):
    """Pin existing forward states and shared readers without exporting target updates."""
    verify(SOURCE)
    nm = read(NATIVE / "manifest.json")
    if nm["stages"][10] != "block22_attention":
        raise ValueError("Attention22 target address differs")
    jobs = read(SOURCE / "private_jobs.json")
    expected = {(c, i) for c, n in (("source4", 4), ("new8", 8)) for i in range(n)}
    if len(jobs) != 12 or {(j["cohort"], j["index"]) for j in jobs} != expected:
        raise ValueError("Exact12 author pairs required")
    out.mkdir(exist_ok=False)
    for name in ("private_inputs", "private_readers", "private_capture"):
        (out / name).mkdir()
    lineage = [SOURCE / "manifest.json", NATIVE / "manifest.json"]
    prepared = []
    for j in jobs:
        if j["query_keys"] != [j["key"] + "_" + q for q in QUERIES]:
            raise ValueError("Query order differs")
        src = SOURCE / "private_inputs" / (j["key"] + ".npz")
        a = arrays(src)
        dst = out / "private_readers" / src.name
        np.savez(dst, shared=a["shared"])
        lineage.append(src)
        prepared.append(dst)
        for key in j["query_keys"]:
            src = NATIVE / "private_inputs" / (key + ".npz")
            if sha(src) != nm["prepared"][str(src.relative_to(NATIVE))]:
                raise ValueError("Forward source hash differs")
            dst = out / "private_inputs" / src.name
            shutil.copyfile(src, dst)
            lineage.append(src)
            prepared.append(dst)
    write(out / "private_jobs.json", jobs)
    prepared.append(out / "private_jobs.json")
    code = [
        PLAN,
        Path(__file__),
        ROOT / "tests/test_grain_upstream_attention_channels.py",
    ]
    code += [
        ROOT / "grain_research/vector_mechanisms" / (n + ".py")
        for n in (
            "upstream_attention_channels",
            "fixed_seed_attention",
            "complete_transport",
            "context_carrier",
            "native_material_reader",
            "query_counteraction",
            "trace",
            "stage_file_guard",
        )
    ]
    code += [
        ROOT / "scripts" / (n + ".py")
        for n in (
            "run_grain_complete_transport_jvp",
            "run_grain_compiled_query_transfer",
            "run_grain_query_counteraction",
        )
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_UPSTREAM_ATTENTION_CHANNELS_V1",
            "block": 22,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "model_sha256": nm["model_sha256"],
            "max_native_seconds": 120,
            "max_jvps": 96,
            "numeric_max_seconds": 60,
            "fits": 0,
            "human_labels_read": False,
            "target_fields_copied": False,
            "inputs": {
                str(p.relative_to(ROOT)): sha(p) for p in sorted(set(code + lineage))
            },
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print({"status": "PREPARED", "pairs": 12, "block": 22}, flush=True)


def capture(out):
    """Capture full pre-attention states/tangents with a strict native-call boundary."""
    import torch

    from grain_research.vector_mechanisms.complete_transport import body_direction
    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from grain_research.vector_mechanisms.upstream_attention_channels import (
        export_weights,
        pre_attention,
    )
    from scripts.run_grain_complete_transport_jvp import inputs, load_observer

    m = verify(out)
    if (out / "capture_status.json").exists():
        raise ValueError("Inspect the existing capture; no implicit restart")
    jobs = read(out / "private_jobs.json")
    ob, loading = load_observer(m)
    torch.set_grad_enabled(False)
    torch.func.jvp(
        lambda x: x.square(),
        (torch.ones(2, device="mps"),),
        (torch.ones(2, device="mps"),),
    )
    wp = out / "private_weights.npz"
    np.savez(wp, **export_weights(ob.model))
    files = [
        out / "private_capture" / (k + ".npz") for j in jobs for k in j["query_keys"]
    ]
    denied = guard(
        [out / "private_inputs" / p.name for p in files],
        files + [wp, out / "capture_status.json", out / "capture_freeze.json"],
    )
    calls, native_seconds, done, checks = 0, 0.0, [], []
    start = time.monotonic()

    def status(kind):
        """Separate native compute, loading and total wall time, retaining partial units."""
        r = {
            "status": kind,
            "jvps": calls,
            "native_seconds": native_seconds,
            "wall_seconds": time.monotonic() - start,
            "loading_seconds": loading,
            "completed": done.copy(),
            "primal_checks": checks.copy(),
            "extra_denials": len(denied) - 4,
            "scores": 0,
            "backwards": 0,
            "fits": 0,
        }
        write(out / "capture_status.json", r)
        print(
            {k: v for k, v in r.items() if k not in ("completed", "primal_checks")},
            flush=True,
        )
        return r

    def forbidden(*_):
        raise AssertionError("Selected attention or downstream operation called")

    handles = [
        a.register_forward_pre_hook(forbidden)
        for a in (*ob.model.model.layers[22:], ob.model.lm_head)
    ]
    result = None
    try:
        for j in jobs:
            for key in j["query_keys"]:
                if calls >= m["max_jvps"] or native_seconds >= m["max_native_seconds"]:
                    raise TimeoutError("Declared native boundary reached")
                state, grain, pos = inputs(out, {"key": key})
                kw = cached_suffix_kwargs(ob.model, state)
                ds, xs = [], []
                for f in range(4):
                    if (
                        calls >= m["max_jvps"]
                        or native_seconds >= m["max_native_seconds"]
                    ):
                        break
                    delta = body_direction(state, grain, pos, f)
                    torch.mps.synchronize()
                    t = time.monotonic()
                    x, d = torch.func.jvp(
                        lambda h, kw=kw: pre_attention(ob.model, h, kw), (state,), (delta,)
                    )
                    torch.mps.synchronize()
                    native_seconds += time.monotonic() - t
                    calls += 1
                    xs.append(x.detach().cpu().numpy()[0])
                    ds.append(d.detach().cpu().numpy()[0])
                cos, sin = [
                    a.detach().cpu().numpy()[0] for a in kw["position_embeddings"]
                ]
                mask = kw["attention_mask"][0, 0, -1].detach().cpu().numpy()
                if not ds or not all(np.isfinite(a).all() for a in xs + ds):
                    raise ValueError("Missing/nonfinite boundary")
                error = max(float(abs(x - xs[0]).max()) for x in xs)
                checks.append({"key": key, "directions": len(ds), "primal_max": error})
                if error > 1e-5:
                    raise ValueError("Repeated boundary primal differs")
                np.savez(
                    out / "private_capture" / (key + ".npz"),
                    x=xs[0],
                    d=np.stack(ds),
                    cos=cos,
                    sin=sin,
                    mask=mask,
                )
                if len(ds) != 4:
                    raise TimeoutError("Partial query directions preserved")
                done.append(key)
                status("RUNNING")
        result = status(
            "COMPLETE"
            if native_seconds <= m["max_native_seconds"]
            else "PAUSED_RESOURCE_BOUNDARY"
        )
    except TimeoutError:
        result = status("PAUSED_RESOURCE_BOUNDARY")
    except Exception:
        status("ERROR")
        raise
    finally:
        for h in handles:
            h.remove()
    paths = [out / "private_capture" / (k + ".npz") for k in done] + [wp]
    write(
        out / "capture_freeze.json",
        {
            "runtime": result,
            "block": 22,
            "files": {str(p.relative_to(out)): sha(p) for p in paths},
        },
    )


def completed_pairs(out):
    """Select complete pairs by resource accounting, never by result values."""
    cap = read(out / "capture_freeze.json")
    if (
        cap["runtime"]["status"] not in ("COMPLETE", "PAUSED_RESOURCE_BOUNDARY")
        or cap["block"] != 22
    ):
        raise ValueError("Usable block22 capture required")
    verify_hashes(out, cap["files"])
    done = set(cap["runtime"]["completed"])
    return [j for j in read(out / "private_jobs.json") if set(j["query_keys"]) <= done]


def predict(out, name):
    """Compute fixed full Q/K/V actions before reading native increment targets."""
    m = verify(out)
    jobs = completed_pairs(out)
    if not jobs:
        raise ValueError("No complete author pair within resource budget")
    dst = out / name
    dst.mkdir(exist_ok=False)
    wp = out / "private_weights.npz"
    w = arrays(wp)
    if int(w["block"]) != 22:
        raise ValueError("Wrong attention matrices")
    reads = [
        out / "private_capture" / (k + ".npz") for j in jobs for k in j["query_keys"]
    ]
    reads += [out / "private_readers" / (j["key"] + ".npz") for j in jobs]
    files = [dst / (j["key"] + ".npz") for j in jobs]
    denied = guard(reads, files + [dst / "prediction_freeze.json"])
    start = time.monotonic()
    for j, path in zip(jobs, files, strict=True):
        if time.monotonic() - start > m["numeric_max_seconds"]:
            raise TimeoutError("Numeric cap reached; do not silently repeat")
        both = []
        for key in j["query_keys"]:
            a = arrays(out / "private_capture" / (key + ".npz"))
            both.append(decompose(a["x"], a["d"], a["cos"], a["sin"], a["mask"], w))
        channel = np.stack([p["channels"] for p in both])
        reader = arrays(out / "private_readers" / path.name)["shared"]
        scalar = np.einsum("qcfh,mh->qmcf", channel, reader)
        np.savez_compressed(
            path,
            channels=channel,
            head_channels=np.stack([p["head_channels"] for p in both]),
            actions=scalar,
            contrast=scalar[0] - scalar[1],
        )
    seconds = time.monotonic() - start
    if seconds > m["numeric_max_seconds"]:
        raise TimeoutError("Numeric cap exceeded")
    freeze = {
        "status": "COMPLETE",
        "seconds": seconds,
        "pairs": len(jobs),
        "target_reads": 0,
        "extra_denials": len(denied) - 4,
        "native_calls": 0,
        "fits": 0,
        "files": {p.name: sha(p) for p in files},
    }
    write(dst / "prediction_freeze.json", freeze)
    print({k: v for k, v in freeze.items() if k != "files"}, flush=True)


def evaluate(out, name, predictions):
    """Check full vectors, then summarize signed channels and their cancellation."""
    verify(out)
    jobs = completed_pairs(out)
    freeze = read(out / predictions / "prediction_freeze.json")
    if freeze["status"] != "COMPLETE" or freeze["pairs"] != len(jobs):
        raise ValueError("Frozen predictions differ")
    verify_hashes(out / predictions, freeze["files"])
    nf = read(NATIVE / "prediction_freeze.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    checks, rows, action_rows = [], [], []
    for j in jobs:
        p = arrays(out / predictions / (j["key"] + ".npz"))
        link = arrays(SOURCE / "private_inputs" / (j["key"] + ".npz"))
        reader = arrays(out / "private_readers" / (j["key"] + ".npz"))["shared"]
        targets = []
        for q, key in enumerate(j["query_keys"]):
            path = NATIVE / "private_predictions" / (key + ".npz")
            if sha(path) != nf["files"][str(path.relative_to(NATIVE))]:
                raise ValueError("Native target changed")
            native = arrays(path)["updates"][:, 10].astype(float)
            check = vector_check(native, p["channels"][q].sum(0))
            checks.append({"key": key, **check})
            if not check["pass"]:
                write(dst / "parity_failure.json", checks)
                raise ValueError("Full-vector attention22 parity failed")
            targets.append(native)
            for m, mode in enumerate(MODES):
                for f in range(4):
                    action_rows.append(
                        {
                            "key": j["key"],
                            "cohort": j["cohort"],
                            "mode": mode,
                            "query": QUERIES[q],
                            "factor": f"whole:F{f + 1:02d}",
                            "native": float(native[f] @ reader[m]),
                            **{
                                ch: float(p["actions"][q, m, c, f])
                                for c, ch in enumerate(CHANNELS)
                            },
                        }
                    )
        for m, mode in enumerate(MODES):
            for f in range(4):
                rows.append(
                    {
                        "key": j["key"],
                        "cohort": j["cohort"],
                        "mode": mode,
                        "factor": f"whole:F{f + 1:02d}",
                        "native": float((targets[0][f] - targets[1][f]) @ reader[m]),
                        "final_T": float(link["transport"][m, f]),
                        **{
                            ch: float(p["contrast"][m, c, f])
                            for c, ch in enumerate(CHANNELS)
                        },
                    }
                )
    df = pd.DataFrame(rows)
    df["sum"] = df[list(CHANNELS)].sum(axis=1)
    adf = pd.DataFrame(action_rows)
    adf["sum"] = adf[list(CHANNELS)].sum(axis=1)
    df.to_csv(dst / "private_contrasts.csv", index=False)
    adf.to_csv(dst / "private_actions.csv", index=False)
    files = ["private_contrasts.csv", "private_actions.csv"]
    for frame, group, name_ in (
        (df, ["cohort", "mode"], "channel_summary.csv"),
        (df, ["cohort", "mode", "factor"], "factor_channels.csv"),
        (adf, ["cohort", "mode", "query"], "query_actions.csv"),
    ):
        result = []
        for key, g in frame.groupby(group, sort=True):
            native = g.native.to_numpy()
            for channel in (*CHANNELS, "sum"):
                x = g[channel].to_numpy()
                row = dict(zip(group, key, strict=True)) | {
                    "channel": channel,
                    "authors": g.key.nunique(),
                    "actions": len(g),
                    "rms": float(np.sqrt(np.mean(x * x))),
                    "native_rms": float(np.sqrt(np.mean(native * native))),
                    "native_projection": float(
                        np.mean(x * native) / np.mean(native * native)
                    ),
                    "omission_mse_over_native": float(
                        np.mean((g["sum"].to_numpy() - x - native) ** 2)
                        / np.mean(native * native)
                    ),
                    "sum_error": float(abs(g["sum"] - g.native).max()),
                }
                if "final_T" in g:
                    t = g.final_T.to_numpy()
                    row["final_T_projection"] = float(np.mean(x * t) / np.mean(t * t))
                result.append(row)
        pd.DataFrame(result).to_csv(dst / name_, index=False)
        files.append(name_)
    covariance = []
    for (cohort, mode), g in df.groupby(["cohort", "mode"], sort=True):
        for i, a in enumerate(CHANNELS):
            for b in CHANNELS[i:]:
                covariance.append(
                    {
                        "cohort": cohort,
                        "mode": mode,
                        "channel_a": a,
                        "channel_b": b,
                        "second_moment": float(np.mean(g[a] * g[b])),
                        "covariance": float(
                            np.mean((g[a] - g[a].mean()) * (g[b] - g[b].mean()))
                        ),
                        "sum_energy_term": float(
                            (1 if a == b else 2) * np.mean(g[a] * g[b])
                        ),
                    }
                )
    pd.DataFrame(covariance).to_csv(dst / "covariance.csv", index=False)
    files.append("covariance.csv")
    write(dst / "private_checks.json", checks)
    files.append("private_checks.json")
    r = {
        "status": "COMPLETE",
        "pairs": len(jobs),
        "actions": len(adf),
        "contrasts": len(df),
        "vector_relative_max": max(c["relative_max"] for c in checks),
        "vector_absolute_max": max(c["absolute_max"] for c in checks),
        "all_vector_checks_pass": all(c["pass"] for c in checks),
        "fits": 0,
        "human_labels_read": False,
        "files": {n: sha(dst / n) for n in files},
    }
    write(dst / "summary.json", r)
    print({k: v for k, v in r.items() if k != "files"}, flush=True)


def main():
    """Execute one stage in a dedicated process with bounded resource use."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "capture", "predict", "evaluate"))
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_upstream_attention_channels_v1"
    )
    p.add_argument("--name", default="predictions")
    p.add_argument("--predictions", default="predictions")
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.stage in ("prepare", "capture"):
            globals()[a.stage](a.output)
        elif a.stage == "predict":
            predict(a.output, a.name)
        else:
            evaluate(a.output, a.name, a.predictions)


if __name__ == "__main__":
    main()
