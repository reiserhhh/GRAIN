#!/usr/bin/env python3
"""NEXT53: matched source value/response learning, reusing frozen NEXT52 runtime."""

from __future__ import annotations

import argparse
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
from grain_research.vector_mechanisms.compiled_query_transfer import QUERIES
from grain_research.vector_mechanisms.response_aware_state import (
    OBJECTIVES,
    fit_response_maps,
)
from grain_research.vector_mechanisms.source_state_transport import (
    ARMS,
    body_directions,
    source_set,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_state_transport as previous
from scripts.run_grain_source_state_transport import (
    BASE,
    SOURCE,
    arrays,
    forbidden,
    frozen,
    read,
    ref,
    selected,
    sha,
    verify,
    verify_hashes,
    write,
)

OLD = BASE / "logic_source_state_transport_v1"
PLAN = ROOT / "docs/GRAIN_RESPONSE_AWARE_STATE_PLAN_V1.md"


def prepare(out):
    """Freeze exact previous assets and source-only directional targets in separate packets."""
    old = verify(OLD)
    fm = frozen(OLD, "models")
    prior = read(OLD / "analysis/summary.json")
    verify_hashes(OLD / "analysis", prior["files"])
    jobs = read(OLD / "private_jobs.json")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    for name in (
        "private_inputs",
        "private_fixed",
        "private_weights.npz",
        "private_jobs.json",
    ):
        (out / name).symlink_to(OLD / name)
    (out / "private_base_models").symlink_to(OLD / "models")
    (out / "private_training").mkdir()
    files = [out / n for n in old["prepared"] if not n.startswith("private_training/")]
    files += [out / "private_base_models" / n for n in fm["files"]]
    inputs = [
        OLD / "manifest.json",
        OLD / "models/freeze.json",
        OLD / "analysis/summary.json",
    ]
    inputs += [OLD / "models" / n for n in fm["files"]]
    inputs += [OLD / "analysis" / n for n in prior["files"]]
    for j in jobs:
        if j["cohort"] != "source4":
            continue
        for key in j["query_keys"]:
            source = SOURCE / "private_capture" / (key + ".npz")
            y = selected(OLD / "private_training" / (key + ".npz"), ("y",))["y"]
            path = out / "private_training" / (key + ".npz")
            np.savez(path, y=y, dy=selected(source, ("d",))["d"])
            files.append(path)
            inputs.append(source)
    inputs += [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/response_aware_state.py",
        ROOT / "tests/test_grain_response_aware_state.py",
        ROOT / "scripts/run_grain_source_state_transport.py",
    ]
    lineage = dict(old["inputs"])
    lineage.update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
    manifest = {
        "protocol": "GRAIN_RESPONSE_AWARE_STATE_V1",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "rank": 32,
        "alpha": 0.01,
        "max_fits": 20,
        "max_seconds": 180,
        "prepare_seconds": time.monotonic() - started,
        "native_calls": 0,
        "human_labels_read": False,
        "objectives": list(OBJECTIVES),
        "amendment": "Pre-outcome matched loss normalization control",
        "inputs": lineage,
        "prepared": {str(p.relative_to(out)): sha(p) for p in files},
    }
    write(out / "manifest.json", manifest)
    # Read-only common assets; each objective gets separate models and immutable predictions.
    for objective in OBJECTIVES:
        child = out / objective
        child.mkdir()
        for name in (
            "private_inputs",
            "private_fixed",
            "private_weights.npz",
            "private_jobs.json",
            "private_base_models",
            "private_training",
        ):
            (child / name).symlink_to(out / name)
        write(child / "manifest.json", manifest)
    print(
        {
            "status": "PREPARED",
            "seconds": manifest["prepare_seconds"],
            "objectives": list(OBJECTIVES),
        },
        flush=True,
    )


def fit(out):
    """Twenty bounded source-only solves, with the same feature assets for both losses."""
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    source = [j for j in jobs if j["cohort"] == "source4"]
    refs = {ref(j): source_set(j["cohort"], j["index"]) for j in jobs}
    for objective in OBJECTIVES:
        (out / objective / "models").mkdir(exist_ok=False)
    paths = [
        out / o / "models" / f"{r}_{a}.npz"
        for o in OBJECTIVES
        for r in sorted(refs)
        for a in ARMS[1:]
    ]
    reads = [
        out / d / (k + ".npz")
        for j in source
        for k in j["query_keys"]
        for d in ("private_inputs", "private_training")
    ]
    reads += [
        out / "private_base_models" / f"{r}_{a}.npz"
        for r in sorted(refs)
        for a in ARMS[1:]
    ]
    writes = (
        paths
        + [out / "fit_freeze.json", out / "source_training_losses.csv"]
        + [out / o / "models/freeze.json" for o in OBJECTIVES]
    )
    denied = install(
        reads, writes, forbidden(), [ROOT / "grain_research", ROOT / "scripts"]
    )
    started = time.monotonic()
    items = []
    for j in source:
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            data = arrays(out / "private_inputs" / (k + ".npz"))
            targets = arrays(out / "private_training" / (k + ".npz"))
            items.append(
                dict(
                    index=j["index"],
                    query=q,
                    x=data["state"],
                    d=body_directions(data),
                    **targets,
                )
            )
    training = []
    for r in sorted(refs):
        for arm in ARMS[1:]:
            base = arrays(out / "private_base_models" / f"{r}_{arm}.npz")
            maps, losses, stats = fit_response_maps(items, refs[r], base)
            for objective, model in maps.items():
                np.savez(out / objective / "models" / f"{r}_{arm}.npz", **model)
            for objective, values in losses.items():
                training.append(
                    dict(
                        reference=r,
                        arm=arm,
                        objective=objective,
                        sources=len(refs[r]),
                        source_R_X=stats["rx"],
                        source_R_D=stats["rd"],
                        **values,
                    )
                )
            if time.monotonic() - started + m["prepare_seconds"] > m["max_seconds"]:
                raise TimeoutError("Twenty-solve source budget exceeded")
    pd.DataFrame(training).to_csv(out / "source_training_losses.csv", index=False)
    elapsed = time.monotonic() - started
    for o in OBJECTIVES:
        ps = [p for p in paths if p.parent.parent.name == o]
        write(
            out / o / "models/freeze.json",
            {
                "status": "COMPLETE",
                "seconds": elapsed,
                "fits": 10,
                "sources": refs,
                "native_calls": 0,
                "human_labels_read": False,
                "denial_probes": 4,
                "extra_denials": len(denied) - 4,
                "shared_fit_stage_seconds": True,
                "files": {p.name: sha(p) for p in ps},
            },
        )
    result = {
        "status": "COMPLETE",
        "seconds": elapsed,
        "fits": len(paths),
        "denial_probes": 4,
        "extra_denials": len(denied) - 4,
        "native_calls": 0,
        "human_labels_read": False,
        "files": {
            str(p.relative_to(out)): sha(p)
            for p in paths + [out / "source_training_losses.csv"]
        },
    }
    write(out / "fit_freeze.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def compare(out, name, analysis):
    """Compare the original primal method, normalization control and joint learning."""
    verify(out)
    folder = out / name
    folder.mkdir(exist_ok=False)
    summaries = {}
    files = []
    for filename in (
        "query_summary.csv",
        "factor_summary.csv",
        "contrast_summary.csv",
        "contrast_factors.csv",
        "geometry_summary.csv",
        "geometry_factors.csv",
    ):
        frames = []
        for objective, base in (
            ("historical_primal", OLD / "analysis"),
            *((o, out / o / analysis) for o in OBJECTIVES),
        ):
            freeze = read(base / "summary.json")
            verify_hashes(base, freeze["files"])
            tab = pd.read_csv(base / filename)
            if objective != "historical_primal":
                tab = tab[tab.arm.isin(ARMS[1:])]
            frames.append(tab.assign(objective=objective))
        tab = pd.concat(frames, ignore_index=True)
        tab.to_csv(folder / filename, index=False)
        summaries[filename] = tab
        files.append(filename)
    g = summaries["geometry_summary.csv"]
    q = summaries["query_summary.csv"]
    idx = ["cohort", "arm", "query_name"]
    value, joint = (g[g.objective == o].set_index(idx) for o in OBJECTIVES)
    changes = value.index.to_frame(index=False)
    for measure in ("state_mse_over_zero", "tangent_mse_over_zero"):
        changes[measure + "_joint_over_matched"] = (
            joint[measure] / value[measure]
        ).to_numpy()
    vq, jq = (q[q.objective == o].set_index(idx) for o in OBJECTIVES)
    changes["endpoint_sse_joint_over_matched"] = (
        (jq.action_mse_over_zero / vq.action_mse_over_zero)
        .reindex(value.index)
        .to_numpy()
    )
    changes.to_csv(folder / "matched_changes.csv", index=False)
    files.append("matched_changes.csv")
    decisions = {}
    for arm in ARMS[1:]:
        direction = changes[(changes.cohort == "new8") & (changes.arm == arm)]
        endpoint = q[(q.cohort == "new8") & (q.arm == arm) & (q.objective == "joint")]
        decisions[arm] = {
            "directional_improvement": bool(
                len(direction) == 2
                and (direction.tangent_mse_over_zero_joint_over_matched < 1).all()
            ),
            "component_replacement": bool(
                len(endpoint) == 2 and (endpoint.action_mse_over_zero < 1).all()
            ),
        }
    result = {
        "status": "COMPLETE",
        "decisions": decisions,
        "native_calls": 0,
        "human_labels_read": False,
        "files": {n: sha(folder / n) for n in files},
    }
    write(folder / "summary.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def main():
    """Keep fit/prediction/evaluation in isolated processes; enforce a shared compute cap."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "fit", "predict", "evaluate", "compare")
    )
    p.add_argument("--objective", choices=OBJECTIVES)
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_response_aware_state_v1"
    )
    p.add_argument("--name")
    p.add_argument("--predictions", default="predictions")
    p.add_argument("--analysis", default="analysis")
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(a.output)
        elif a.stage == "fit":
            fit(a.output)
        elif a.stage == "compare":
            compare(a.output, a.name or "comparison", a.analysis)
        else:
            if a.objective is None:
                p.error("An objective is required for predict/evaluate")
            child = a.output / a.objective
            if a.stage == "evaluate":
                previous.evaluate(child, a.name or "analysis", a.predictions)
            else:
                name = a.name or "predictions"
                m = read(a.output / "manifest.json")
                used = 0.0
                if name == "predictions":
                    used = (
                        m["prepare_seconds"]
                        + read(a.output / "fit_freeze.json")["seconds"]
                    )
                for o in OBJECTIVES:
                    freeze = a.output / o / name / "freeze.json"
                    if freeze.exists():
                        used += read(freeze)["seconds"]
                remaining = m["max_seconds"] - used
                if remaining <= 0:
                    raise TimeoutError("Combined construction budget already exhausted")

                def timeout(_signum, _frame):
                    """Stop a stage before exceeding the remaining joint wall budget."""
                    raise TimeoutError("Combined numerical budget exhausted")

                signal.signal(signal.SIGALRM, timeout)
                signal.setitimer(signal.ITIMER_REAL, remaining)
                try:
                    previous.predict(child, name)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
