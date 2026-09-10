#!/usr/bin/env python3
"""NEXT46: bounded MLP24 input/operator exchange on fixed real-author materials."""

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
from grain_research.vector_mechanisms.compiled_query_transfer import (
    MODES,
    contrast_accounting,
)
from grain_research.vector_mechanisms.query_counteraction import (
    decompose,
    exchange,
    vector_check,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_compiled_query_transfer import (
    BASE,
    arrays,
    read,
    sha,
    verify,
    verify_hashes,
    write,
)

SOURCE = BASE / "logic_query_transport_formation_v1"
NATIVE = BASE / "logic_layer_accumulation_v1"
PLAN = ROOT / "docs/GRAIN_QUERY_COUNTERACTION_PLAN_V1.md"


def prepare(out):
    """Freeze only prefix directions, forward inputs and unchanged shared readers."""
    old = verify(SOURCE)
    nm = read(NATIVE / "manifest.json")
    if nm["stages"][14:16] != ["block24_attention", "block24_mlp"]:
        raise ValueError("Native stage addresses differ")
    jobs = read(SOURCE / "private_jobs.json")
    if len(jobs) != 12 or len({j["key"] for j in jobs}) != 12:
        raise ValueError("Exact12 paired units required")
    out.mkdir(exist_ok=False)
    for folder in (
        "private_inputs",
        "private_prefix",
        "private_capture",
        "private_predictions",
    ):
        (out / folder).mkdir()
    prepared, lineage = [], [SOURCE / "manifest.json", NATIVE / "manifest.json"]
    for j in jobs:
        a = arrays(SOURCE / "private_inputs" / (j["key"] + ".npz"))
        d = a["initial"] + a["updates"][:, :, :15].sum(axis=2)
        dst = out / "private_prefix" / (j["key"] + ".npz")
        np.savez_compressed(dst, d=d, shared=a["shared"])
        prepared.append(dst)
        lineage.append(SOURCE / "private_inputs" / dst.name)
        for key in j["query_keys"]:
            src = NATIVE / "private_inputs" / (key + ".npz")
            if sha(src) != nm["prepared"][str(src.relative_to(NATIVE))]:
                raise ValueError("Native input changed")
            dst = out / "private_inputs" / src.name
            shutil.copy2(src, dst)
            prepared.append(dst)
            lineage.append(src)
    write(out / "private_jobs.json", jobs)
    prepared.append(out / "private_jobs.json")
    code = [PLAN, Path(__file__), ROOT / "tests/test_grain_query_counteraction.py"]
    code += [
        ROOT / "grain_research/vector_mechanisms" / (n + ".py")
        for n in (
            "query_counteraction",
            "gated_update",
            "complete_transport",
            "context_carrier",
            "compiled_query_transfer",
            "stage_file_guard",
            "trace",
        )
    ]
    code += [
        ROOT / "scripts" / (n + ".py")
        for n in (
            "run_grain_complete_transport_jvp",
            "run_grain_compiled_query_transfer",
        )
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_COUNTERACTION_V1",
            "block": 24,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "model_sha256": nm["model_sha256"],
            "max_seconds": 120,
            "numeric_max_seconds": 60,
            "max_prefix_calls": 28,
            "primary_observed_before": "NEXT45 MLP24 signed counteraction",
            "native_target_fields_copied": False,
            "fits": 0,
            "human_labels_read": False,
            "source_manifest_commit": old["commit"],
            "inputs": {
                str(p.relative_to(ROOT)): sha(p) for p in sorted(set(code + lineage))
            },
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print({"status": "PREPARED", "pairs": 12, "block": 24}, flush=True)


def guard(reads, writes):
    """Deny original native targets and earlier full-gradient packets."""
    probes = [
        BASE / folder / area / ("new8_a00_neuroticism" + ext)
        for folder, area in (
            ("logic_layer_accumulation_v1", "private_predictions"),
            ("logic_fixed_seed_attention_v1", "private_packets"),
        )
        for ext in (".npz", ".json")
    ]
    return install(reads, writes, probes, [ROOT / "grain_research", ROOT / "scripts"])


def capture(out):
    """24 primal prefixes and four preselected F01 JVP checks, with no MLP24 call."""
    import torch

    from grain_research.vector_mechanisms.complete_transport import body_direction
    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from grain_research.vector_mechanisms.gated_update import (
        capture_jvp,
        export_weights,
        pre_mlp,
    )
    from scripts.run_grain_complete_transport_jvp import inputs, load_observer

    m = verify(out)
    if (out / "capture_status.json").exists():
        raise ValueError("Existing capture; inspect status rather than silently repeat")
    jobs = read(out / "private_jobs.json")
    ob, loading = load_observer(m)
    torch.set_grad_enabled(False)
    torch.func.jvp(
        lambda x: x.square(),
        (torch.ones(2, device="mps"),),
        (torch.ones(2, device="mps"),),
    )
    wp = out / "private_weights.npz"
    start = time.monotonic()
    weights = export_weights(ob.model, block=24)
    if any(weights[k].dtype != np.float32 for k in ("gate", "up", "down", "gamma")):
        raise ValueError("Pinned float32 weights required")
    np.savez(wp, **weights)
    weights_hash = sha(wp)
    files = [
        out / "private_capture" / (key + ".npz")
        for j in jobs
        for key in j["query_keys"]
    ]
    reads = [out / "private_inputs" / p.name for p in files]
    reads += [out / "private_prefix" / (j["key"] + ".npz") for j in jobs]
    denied = guard(
        reads, files + [out / "capture_status.json", out / "capture_freeze.json"]
    )
    calls, checks, done = 0, [], []

    def status(kind):
        """Persist terminal/partial resource accounting without raw materials."""
        result = {
            "status": kind,
            "prefix_calls": calls,
            "jvp_checks": len(checks),
            "seconds": time.monotonic() - start,
            "loading_seconds": loading,
            "completed": done.copy(),
            "checks": checks.copy(),
            "extra_denials": len(denied) - 4,
            "scores": 0,
            "backwards": 0,
            "fits": 0,
        }
        write(out / "capture_status.json", result)
        print(
            {k: v for k, v in result.items() if k not in ("completed", "checks")},
            flush=True,
        )
        return result

    def budget():
        """Stop at the declared next-call boundary, with no cap widening."""
        if (
            calls >= m["max_prefix_calls"]
            or time.monotonic() - start >= m["max_seconds"]
        ):
            raise TimeoutError("Fixed native cap reached")

    def forbidden(*_):
        raise AssertionError("Downstream MLP24 or later operation called")

    handles = [
        module.register_forward_pre_hook(forbidden)
        for module in (
            ob.model.model.layers[24].mlp,
            *ob.model.model.layers[25:],
            ob.model.lm_head,
        )
    ]
    try:
        for j in jobs:
            prefix = arrays(out / "private_prefix" / (j["key"] + ".npz"))
            for q, key in enumerate(j["query_keys"]):
                budget()
                state, grain, pos = inputs(out, {"key": key})
                kw = cached_suffix_kwargs(ob.model, state)
                calls += 1
                x = pre_mlp(ob.model, state, kw, block=24).detach().cpu().numpy()[0]
                if not np.isfinite(x).all():
                    raise ValueError("Nonfinite primal")
                np.savez_compressed(out / "private_capture" / (key + ".npz"), x=x)
                if j["index"] == 0:
                    budget()
                    calls += 1
                    check_x, d = capture_jvp(
                        ob.model,
                        state,
                        kw,
                        body_direction(state, grain, pos, 0),
                        block=24,
                    )
                    c = vector_check(prefix["d"][q, :1], d.cpu().numpy())
                    c.update(
                        key=key,
                        primal_max=float(abs(check_x.cpu().numpy()[0] - x).max()),
                    )
                    checks.append(c)
                    if not c["pass"] or c["primal_max"] > 1e-5:
                        raise ValueError("PreMLP24 input parity failed")
                done.append(key)
                status("RUNNING")
        if time.monotonic() - start > m["max_seconds"]:
            raise TimeoutError("Completed boundary exceeds declared cap")
        result = status("COMPLETE")
        write(
            out / "capture_freeze.json",
            {
                "runtime": result,
                "block": 24,
                "weights_sha256": weights_hash,
                "files": {str(p.relative_to(out)): sha(p) for p in files}
                | {wp.name: weights_hash},
            },
        )
    except TimeoutError:
        status("PAUSED_RESOURCE_BOUNDARY")
    except Exception:
        status("ERROR")
        raise
    finally:
        for h in handles:
            h.remove()


def predict(out, name):
    """Freeze full crossed local increments before opening target increments."""
    m = verify(out)
    cf = read(out / "capture_freeze.json")
    if cf["runtime"]["status"] != "COMPLETE" or cf["block"] != 24:
        raise ValueError("Complete block24 capture required")
    verify_hashes(out, cf["files"])
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    wp = out / "private_weights.npz"
    weights_hash = sha(wp)
    weights = {k: np.asarray(v, float) for k, v in arrays(wp).items()}
    files = [dst / (j["key"] + ".npz") for j in jobs]
    reads = [out / "private_prefix" / p.name for p in files]
    reads += [
        out / "private_capture" / (key + ".npz")
        for j in jobs
        for key in j["query_keys"]
    ]
    denied = guard(reads, files + [dst / "prediction_freeze.json"])
    start = time.monotonic()
    for j, f in zip(jobs, files, strict=True):
        if time.monotonic() - start > m["numeric_max_seconds"]:
            raise TimeoutError("Numeric cap reached")
        a = arrays(out / "private_prefix" / f.name)
        x = np.stack(
            [
                arrays(out / "private_capture" / (k + ".npz"))["x"]
                for k in j["query_keys"]
            ]
        )
        values = exchange(x, a["d"], weights)
        split = decompose(values)
        if abs(split["carried"] + split["operator"] - split["contrast"]).max() > 1e-10:
            raise ValueError("Symmetric split failed")
        np.savez_compressed(f, exchange=values, **split)
    seconds = time.monotonic() - start
    if seconds > m["numeric_max_seconds"]:
        raise TimeoutError("Numeric cap exceeded")
    write(
        dst / "prediction_freeze.json",
        {
            "status": "COMPLETE",
            "seconds": seconds,
            "block": 24,
            "weights_sha256": weights_hash,
            "target_reads": 0,
            "extra_denials": len(denied) - 4,
            "native_calls": 0,
            "fits": 0,
            "files": {str(p.relative_to(dst)): sha(p) for p in files},
        },
    )
    print({"status": "PREDICTED", "seconds": seconds, "target_reads": 0}, flush=True)


def evaluate(out, name, predictions):
    """Open native MLP24 targets only after freezing crossed numerical predictions."""
    verify(out)
    freeze = read(out / predictions / "prediction_freeze.json")
    verify_hashes(out / predictions, freeze["files"])
    nf = read(NATIVE / "prediction_freeze.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    rows, checks = [], []
    for j in read(out / "private_jobs.json"):
        p = arrays(out / predictions / (j["key"] + ".npz"))
        a = arrays(out / "private_prefix" / (j["key"] + ".npz"))
        link = arrays(SOURCE / "private_inputs" / (j["key"] + ".npz"))
        targets = []
        for q, key in enumerate(j["query_keys"]):
            f = NATIVE / "private_predictions" / (key + ".npz")
            if sha(f) != nf["files"][str(f.relative_to(NATIVE))]:
                raise ValueError("Target changed")
            native = arrays(f)["updates"][:, 15].astype(float)
            check = vector_check(native, p["exchange"][q, q].sum(0))
            checks.append({"key": key, **check})
            if not check["pass"]:
                raise ValueError("Actual full-vector diagonal failed")
            targets.append(native)
        for m, mode in enumerate(MODES):
            w = a["shared"][m]
            for factor in range(4):
                row = {
                    "key": j["key"],
                    "cohort": j["cohort"],
                    "mode": mode,
                    "factor": f"whole:F{factor + 1:02d}",
                    "final_T": float(link["transport"][m, factor]),
                    "native": float((targets[0][factor] - targets[1][factor]) @ w),
                }
                for part in ("carried", "operator", "contrast"):
                    row[part] = float(p[part][:, factor].sum(0) @ w)
                    for term, label in enumerate(("input", "gate")):
                        row[part + "_" + label] = float(p[part][term, factor] @ w)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(dst / "private_actions.csv", index=False)
    summaries = []
    for keys in (("cohort", "mode"), ("cohort", "mode", "factor")):
        result = []
        for key, g in df.groupby(list(keys), sort=True):
            accounting = contrast_accounting(
                g.carried.to_numpy(), g.operator.to_numpy()
            )
            rec = dict(zip(keys, key, strict=True)) | {
                k.replace("reader", "carried").replace("transport", "operator"): v
                for k, v in accounting.items()
            }
            energy = np.mean(g.final_T.to_numpy() ** 2)
            for part in ("carried", "operator", "contrast"):
                rec[part + "_final_T_share"] = float(
                    np.mean(g[part] * g.final_T) / energy
                )
            rec["native_absolute_max"] = float(abs(g.native - g.contrast).max())
            rec["sum_error"] = float(abs(g.carried + g.operator - g.contrast).max())
            rec["authors"] = g.key.nunique()
            result.append(rec)
        filename = "summary.csv" if len(keys) == 2 else "factors.csv"
        pd.DataFrame(result).to_csv(dst / filename, index=False)
        summaries.append(filename)
    write(dst / "private_checks.json", checks)
    files = [
        dst / n for n in ["private_actions.csv", "private_checks.json", *summaries]
    ]
    write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "rows": len(df),
            "pairs": 12,
            "diagonal_vectors": 96,
            "checks_pass": all(c["pass"] for c in checks),
            "vector_relative_max": max(c["relative_max"] for c in checks),
            "vector_absolute_max": max(c["absolute_max"] for c in checks),
            "files": {p.name: sha(p) for p in files},
        },
    )
    print(read(dst / "summary.json"), flush=True)


def main():
    """Run construction, bounded capture, target-denied prediction or evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture", "predict", "evaluate"))
    parser.add_argument(
        "--output", type=Path, default=BASE / "logic_query_counteraction_v1"
    )
    parser.add_argument("--name", default="predictions")
    parser.add_argument("--predictions", default="predictions")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage in ("prepare", "capture"):
            globals()[args.stage](args.output)
        elif args.stage == "predict":
            predict(args.output, args.name)
        else:
            evaluate(args.output, args.name, args.predictions)


if __name__ == "__main__":
    main()
