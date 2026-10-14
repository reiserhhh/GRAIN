#!/usr/bin/env python3
"""Freeze, predict, execute finite native actions, then analyze without refitting."""

from __future__ import annotations

import argparse
import inspect
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.query_moment_actions import (
    DIRECTIONS,
    action_specs,
    array_hash,
    body_action,
    odd_even,
    patch_body,
)
from grain_research.query_moment_operator import apply_operator, observation_moments
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_query_moment_operator import (
    ARMS,
    BASE,
    INPUT,
    NATIVE,
    QUERIES,
    arrays,
    directions,
    load_models,
    read,
    sha,
    write,
)

PRIOR = BASE / "mechanism_todo_v1/query_moment_r1"
DEFAULT = BASE / "mechanism_todo_v1/query_moment_finite_r1"
PLAN = ROOT / "docs/GRAIN_QUERY_MOMENT_FINITE_ACTION_PLAN_V1.md"
CONFIG = ROOT / "configs/grain_query_moment_finite_actions.json"


def pure(key):
    """Deserialize only the archived forward-input schema."""
    data = arrays(INPUT / "private_inputs" / (key + ".npz"))
    if set(data) != {"state", "grain", "positions", "cos", "sin", "mask"}:
        raise ValueError("Unexpected pure-input fields")
    if data["state"].dtype != np.float32:
        raise ValueError("Pinned FP32 state required")
    patch_body(data["state"], data["positions"], data["state"][data["positions"]])
    return data


def verify(out):
    """Verify all plan, source-model, code and numeric-input bindings."""
    contract = read(out / "contract.json")
    for name, expected in contract["inputs"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed finite-action input: " + name)
    return contract


def prepare(out):
    """Freeze the finite task without fitting or reading application target values."""
    cfg, prior = read(CONFIG), read(PRIOR / "contract.json")
    if read(PRIOR / "audit.json")["status"].upper() != "PASS":
        raise ValueError("Audited source-compiled packet required")
    jobs = [j for j in prior["jobs"] if j["cohort"] == cfg["cohort"]]
    if len(jobs) != 8 or list(QUERIES) != cfg["queries"]:
        raise ValueError("Frozen reused8/two-query contract required")
    specs = action_specs(cfg["epsilons"])
    if len(jobs) * len(QUERIES) * len(specs) != cfg["max_calls"]:
        raise ValueError("Call budget disagrees with exhaustive action list")
    load_models(PRIOR)
    files = [
        PLAN,
        CONFIG,
        Path(__file__),
        PRIOR / "contract.json",
        PRIOR / "audit.json",
        PRIOR / "models/freeze.json",
        NATIVE / "manifest.json",
        NATIVE / "observer.json",
    ]
    files += [
        ROOT / name
        for name in (
            "grain_research/query_moment_actions.py",
            "grain_research/query_moment_operator.py",
            "grain_research/vector_mechanisms/native_material_reader.py",
            "grain_research/vector_mechanisms/selection_endpoint.py",
            "grain_research/vector_mechanisms/context_carrier.py",
            "grain_research/vector_mechanisms/stage_file_guard.py",
            "scripts/run_grain_query_moment_operator.py",
            "scripts/run_grain_complete_transport_jvp.py",
            "scripts/audit_grain_query_moment_finite_actions.py",
        )
    ]
    files += [PRIOR / "models" / (q + ".npz") for q in QUERIES]
    for job in jobs:
        first = pure(job["query_keys"][0])
        for key in job["query_keys"]:
            data = pure(key)
            np.testing.assert_array_equal(first["grain"], data["grain"])
            np.testing.assert_array_equal(
                first["state"][first["positions"]], data["state"][data["positions"]]
            )
            files += [INPUT / "private_inputs" / (key + ".npz")]
            # Hashing opaque target files is not a target-value read/selection.
            files += [
                NATIVE / "private_packets" / (key + ext) for ext in (".npz", ".json")
            ]
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "contract.json",
        {
            **cfg,
            "jobs": jobs,
            "actions": specs,
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "prepared_time_ns": time.time_ns(),
        },
    )
    print(
        {"stage": "prepare", "authors": len(jobs), "calls": cfg["max_calls"]},
        flush=True,
    )


def apply(out):
    """Freeze explicit finite predictions before permitting current native targets."""
    contract, models = verify(out), load_models(PRIOR)
    dst = out / "application"
    dst.mkdir(exist_ok=False)
    target_paths = [
        NATIVE / "private_packets" / (key + ext)
        for j in contract["jobs"]
        for key in j["query_keys"]
        for ext in (".json", ".npz")
    ]
    reads = [ROOT / p for p in contract["inputs"] if "private_packets" not in p]
    outputs = [
        dst / n
        for n in ("private_predictions.csv", "private_observations.csv", "freeze.json")
    ]
    denied = install(
        reads, outputs, target_paths, [ROOT / "scripts", ROOT / "grain_research"]
    )
    rows, obs = [], []
    start = time.process_time()
    for job in contract["jobs"]:
        for query, key in zip(QUERIES, job["query_keys"], strict=True):
            data = pure(key)
            g, h = data["grain"], data["state"][data["positions"]]
            ds = directions({"h": h.astype(float), "g": g}, models)
            model = models[query]
            y0, fields = apply_operator(g, h, model)
            m0 = observation_moments(g, h, model)
            for spec in contract["actions"]:
                changed = body_action(g, h, ds, spec)
                full = patch_body(data["state"], data["positions"], changed)
                values = apply_operator(g, changed, model)[0]
                moments = observation_moments(g, changed, model)
                meta = {
                    "author_key": job["key"],
                    "query_key": key,
                    "query": query,
                    **spec,
                }
                obs.append(
                    {
                        **meta,
                        "body_sha256": array_hash(changed),
                        "full_sha256": array_hash(full),
                        "relative_displacement": np.linalg.norm(
                            changed.astype(float) - h
                        )
                        / np.linalg.norm(h.astype(float)),
                        **{
                            n + "_relative_change": np.linalg.norm(moments[n] - m0[n])
                            / max(np.linalg.norm(m0[n]), 1e-12)
                            for n in m0
                        },
                    }
                )
                for ai, arm in enumerate(ARMS):
                    rows.append(
                        {
                            **meta,
                            "arm": arm,
                            "value": values[ai],
                            "delta": values[ai] - y0[ai],
                            "linearized_actual_patch": np.sum(
                                fields[ai] * (changed.astype(float) - h)
                            ),
                        }
                    )
        if time.process_time() - start > contract["max_application_cpu_seconds"]:
            raise TimeoutError("Application CPU cap reached; no native stage")
    pd.DataFrame(rows).to_csv(outputs[0], index=False)
    pd.DataFrame(obs).to_csv(outputs[1], index=False)
    write(
        outputs[2],
        {
            "files": {p.name: sha(p) for p in outputs[:2]},
            "rows": len(rows),
            "cpu_seconds": time.process_time() - start,
            "frozen_time_ns": time.time_ns(),
            "target_denial_probes": len(target_paths),
            "extra_denials": len(denied) - len(target_paths),
            "fits": 0,
            "native_calls": 0,
        },
    )
    print(
        {
            "stage": "apply",
            "rows": len(rows),
            "cpu_seconds": time.process_time() - start,
        },
        flush=True,
    )


def check_predictions(out):
    """Require immutable explicit predictions before native calls or target access."""
    freeze = read(out / "application/freeze.json")
    for name, expected in freeze["files"].items():
        if sha(out / "application" / name) != expected:
            raise ValueError("Frozen application changed")
    if freeze["extra_denials"] or freeze["native_calls"] or freeze["fits"]:
        raise ValueError("Unexpected prediction-stage access or fitting")
    return freeze


def native(out):
    """Replay the pinned native suffix; persist every finite value under hard caps."""
    import torch
    from transformers.models.qwen2 import modeling_qwen2

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from grain_research.vector_mechanisms.selection_endpoint import endpoint
    from scripts.run_grain_complete_transport_jvp import load_observer

    contract, models = verify(out), load_models(PRIOR)
    freeze = check_predictions(out)
    manifest = read(NATIVE / "manifest.json")
    if (
        sha(Path(inspect.getfile(modeling_qwen2)))
        != manifest["qwen_implementation_sha256"]
    ):
        raise ValueError("Pinned native implementation changed")
    dst = out / "native"
    dst.mkdir(exist_ok=False)
    stats = {
        "status": "LOADING",
        "calls": 0,
        "native_seconds": 0.0,
        "completed_contexts": [],
        "files": {},
        "started_time_ns": time.time_ns(),
        "prediction_frozen_time_ns": freeze["frozen_time_ns"],
        "fits": 0,
        "new_backward_calls": 0,
    }
    write(dst / "status.json", stats)

    def alarm(_sig, _frame):
        raise TimeoutError("Frozen observer loading cap reached")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(contract["max_loading_seconds"])
    try:
        ob, loading = load_observer(manifest)
    except Exception as error:
        stats.update(
            status="FAILED_LOADING", error=type(error).__name__ + ": " + str(error)
        )
        write(dst / "status.json", stats)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    stats["loading_seconds"] = loading
    torch.set_grad_enabled(False)
    ids = torch.tensor(manifest["alphabet"]["token_ids"], device="mps")
    contrast = torch.tensor(manifest["c"], device="mps", dtype=torch.float32)
    bound = pd.read_csv(out / "application/private_observations.csv").set_index(
        ["query_key", "action"]
    )
    try:
        for job in contract["jobs"]:
            for query, key in zip(QUERIES, job["query_keys"], strict=True):
                data = pure(key)
                h, g = data["state"][data["positions"]], data["grain"]
                ds = directions({"h": h.astype(float), "g": g}, models)
                x0 = torch.tensor(data["state"][None], device="mps")
                kwargs = cached_suffix_kwargs(ob.model, x0)
                for value, name in zip(
                    kwargs["position_embeddings"], ("cos", "sin"), strict=True
                ):
                    np.testing.assert_array_equal(value.cpu().numpy()[0], data[name])
                np.testing.assert_array_equal(
                    kwargs["attention_mask"][0, 0, -1].cpu().numpy(), data["mask"]
                )
                archived = float(
                    read(NATIVE / "private_packets" / (key + ".json"))["contrast"]
                )
                packet = {
                    "query_key": key,
                    "author_key": job["key"],
                    "query": query,
                    "archived_baseline": archived,
                    "rows": [],
                }
                path = dst / (key + ".json")
                for spec in contract["actions"]:
                    if (
                        stats["calls"] >= contract["max_calls"]
                        or stats["native_seconds"] >= contract["max_native_seconds"]
                    ):
                        raise TimeoutError("Frozen native call/time cap reached")
                    changed = body_action(g, h, ds, spec)
                    full = patch_body(data["state"], data["positions"], changed)
                    expected = bound.loc[(key, spec["action"]), :]
                    if (
                        array_hash(changed) != expected.body_sha256
                        or array_hash(full) != expected.full_sha256
                    ):
                        raise ValueError(
                            "Native intervention differs from frozen prediction"
                        )
                    x = torch.tensor(full[None], device="mps")
                    torch.mps.synchronize()
                    start = time.monotonic()
                    stats["calls"] += 1
                    try:
                        value = float(
                            endpoint(
                                ob.model,
                                x,
                                kwargs,
                                ids,
                                contrast,
                                block=contract["suffix_block"],
                            ).item()
                        )
                    finally:
                        torch.mps.synchronize()
                        stats["native_seconds"] += time.monotonic() - start
                    if not np.isfinite(value):
                        raise ArithmeticError("Nonfinite native endpoint")
                    packet["rows"].append(
                        {
                            **spec,
                            "value": value,
                            "body_sha256": array_hash(changed),
                            "full_sha256": array_hash(full),
                        }
                    )
                    write(path, packet)
                    stats["files"][path.name] = sha(path)
                    stats["status"] = "RUNNING"
                    write(dst / "status.json", stats)
                    if (
                        spec["kind"] == "baseline"
                        and abs(value - archived) > contract["replay_score_atol"]
                    ):
                        raise ArithmeticError(
                            "Unchanged native baseline differs from archived target"
                        )
                stats["completed_contexts"].append(key)
                write(dst / "status.json", stats)
                print(
                    {
                        "stage": "native",
                        "contexts": len(stats["completed_contexts"]),
                        "calls": stats["calls"],
                        "native_seconds": stats["native_seconds"],
                    },
                    flush=True,
                )
        stats["status"] = "COMPLETE"
    except TimeoutError as error:
        stats.update(status="PAUSED_BUDGET", error=str(error))
        raise
    except Exception as error:
        stats.update(status="FAILED", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        stats["finished_time_ns"] = time.time_ns()
        write(dst / "status.json", stats)


def make_tables(out):
    """Derive all finite/stress and odd/even comparisons, including evaluation-only tangents."""
    contract = verify(out)
    check_predictions(out)
    models = load_models(PRIOR)
    pred = pd.read_csv(out / "application/private_predictions.csv")
    records, components, stresses, replays = [], [], [], []
    for job in contract["jobs"]:
        for query, key in zip(QUERIES, job["query_keys"], strict=True):
            path = out / "native" / (key + ".json")
            if not path.exists():
                continue
            packet = read(path)
            native_values = {r["action"]: r["value"] for r in packet["rows"]}
            if "baseline" not in native_values:
                continue
            f0 = native_values["baseline"]
            replays.append(
                {"query_key": key, "error": f0 - packet["archived_baseline"]}
            )
            data = pure(key)
            h, g = data["state"][data["positions"]], data["grain"]
            ds = directions({"h": h.astype(float), "g": g}, models)
            j = arrays(NATIVE / "private_packets" / (key + ".npz"))[
                "body_gradient"
            ].astype(float)
            pp = pred[pred.query_key == key]
            for spec in contract["actions"][1:]:
                action = spec["action"]
                if action not in native_values:
                    continue
                changed = body_action(g, h, ds, spec)
                actual = native_values[action] - f0
                tangent = float(np.sum(j * (changed.astype(float) - h)))
                for row in pp[pp.action == action].itertuples():
                    item = {
                        "author_key": job["key"],
                        "query": query,
                        **spec,
                        "arm": row.arm,
                        "actual": actual,
                        "predicted": row.delta,
                        "native_tangent": tangent,
                    }
                    (stresses if spec["kind"] == "stress" else records).append(item)
            for direction in DIRECTIONS:
                for e in contract["epsilons"]:
                    names = [f"{direction}_{e:.6g}_{s:+d}" for s in (1, -1)]
                    if not all(name in native_values for name in names):
                        continue
                    actual_odd, actual_even = odd_even(
                        f0, *[native_values[n] for n in names]
                    )
                    tangent_values = []
                    for sign in (1, -1):
                        hh = body_action(
                            g,
                            h,
                            ds,
                            {
                                "kind": "finite",
                                "direction": direction,
                                "epsilon": e,
                                "sign": sign,
                            },
                        )
                        tangent_values.append(np.sum(j * (hh.astype(float) - h)))
                    for arm in (*ARMS, "native_tangent"):
                        vv = (
                            tangent_values
                            if arm == "native_tangent"
                            else [
                                float(
                                    pp[(pp.action == n) & (pp.arm == arm)].delta.iloc[0]
                                )
                                for n in names
                            ]
                        )
                        po, pe = odd_even(0.0, *vv)
                        components.append(
                            {
                                "author_key": job["key"],
                                "query": query,
                                "direction": direction,
                                "epsilon": e,
                                "arm": arm,
                                "actual_odd": actual_odd,
                                "predicted_odd": po,
                                "actual_even": actual_even,
                                "predicted_even": pe,
                            }
                        )
    return [pd.DataFrame(x) for x in (records, components, stresses, replays)]


def metrics(components, contract):
    """Paired author bootstrap; repeated actions are never independent participants."""
    rows = []
    groups = {
        "all7": list(DIRECTIONS),
        "F4": list(DIRECTIONS[:4]),
        "within2": list(DIRECTIONS[4:6]),
        "random": [DIRECTIONS[6]],
    }
    for query in QUERIES:
        for e in contract["epsilons"]:
            for group, names in groups.items():
                panel = components[
                    (components["query"] == query)
                    & (components.epsilon == e)
                    & components.direction.isin(names)
                ]
                for component in ("odd", "even"):
                    if panel.empty:
                        continue
                    p = panel.copy()
                    p["mse"] = (
                        p["predicted_" + component] - p["actual_" + component]
                    ) ** 2
                    wide = (
                        p.groupby(["author_key", "arm"])
                        .mse.mean()
                        .unstack("arm")
                        .dropna()
                    )
                    rng = np.random.default_rng(contract["bootstrap_seed"])
                    ix = rng.integers(
                        len(wide), size=(contract["bootstrap_samples"], len(wide))
                    )
                    baseline = wide.quadratic.to_numpy()
                    actual = panel[panel.arm == "quadratic"][
                        "actual_" + component
                    ].to_numpy()
                    for arm in wide:
                        v = wide[arm].to_numpy()
                        improvement = np.sqrt(baseline[ix].mean(1)) - np.sqrt(
                            v[ix].mean(1)
                        )
                        rows.append(
                            {
                                "query": query,
                                "epsilon": e,
                                "group": group,
                                "component": component,
                                "arm": arm,
                                "authors": len(wide),
                                "rmse": np.sqrt(v.mean()),
                                "actual_rms": np.sqrt(np.mean(actual**2)),
                                "rmse_reduction_vs_quadratic": np.sqrt(baseline.mean())
                                - np.sqrt(v.mean()),
                                "delta_ci_low": np.quantile(improvement, 0.025),
                                "delta_ci_high": np.quantile(improvement, 0.975),
                                "positive_author_count": int(np.sum(v < baseline)),
                            }
                        )
    return pd.DataFrame(rows)


def analyze(out):
    """Save exhaustive evaluation with local utility and global sufficiency separated."""
    contract = verify(out)
    tables = make_tables(out)
    dst = out / "evaluation"
    dst.mkdir(exist_ok=False)
    names = (
        "private_actions.csv",
        "private_components.csv",
        "private_stresses.csv",
        "replays.csv",
    )
    for name, table in zip(names, tables, strict=True):
        table.to_csv(dst / name, index=False)
    mm = metrics(tables[1], contract)
    mm.to_csv(dst / "metrics.csv", index=False)
    obs = pd.read_csv(out / "application/private_observations.csv")
    stress_summary = []
    for (query, action), p in tables[2].groupby(["query", "action"]):
        actual = p[p.arm == "quadratic"].actual.to_numpy()
        ob = obs[(obs["query"] == query) & (obs.action == action)]
        stress_summary.append(
            {
                "query": query,
                "action": action,
                "authors": len(actual),
                "native_delta_rms": np.sqrt(np.mean(actual**2)),
                "native_abs_min": np.min(abs(actual)),
                "native_abs_max": np.max(abs(actual)),
                "nonzero_above_replay_tolerance": int(
                    np.sum(abs(actual) > contract["replay_score_atol"])
                ),
                "explicit_abs_max": float(p.predicted.abs().max()),
                "relative_displacement_mean": ob.relative_displacement.mean(),
                "max_observation_relative_change": ob[
                    [n for n in ob if n.endswith("_relative_change")]
                ]
                .to_numpy()
                .max(),
            }
        )
    pd.DataFrame(stress_summary).to_csv(dst / "stress_summary.csv", index=False)
    summary = {
        "protocol": contract["protocol"],
        "native_status": read(out / "native/status.json"),
        "counts": {n: len(t) for n, t in zip(names, tables, strict=True)},
        "max_baseline_error": tables[3].error.abs().max(),
        "stress": stress_summary,
        "scope": "Frozen source4, historically exposed application8; hidden-state endpoint mechanism, not new personality accuracy",
        "no_new_fit": True,
        "no_new_backward": True,
    }
    write(dst / "summary.json", summary)
    write(
        dst / "freeze.json",
        {"files": {p.name: sha(p) for p in sorted(dst.iterdir()) if p.is_file()}},
    )
    print(
        {
            "stage": "analyze",
            "metric_rows": len(mm),
            "counts": summary["counts"],
            "max_baseline_error": summary["max_baseline_error"],
        },
        flush=True,
    )


def main():
    """Run one isolated stage; application guards are process-local and irreversible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "apply", "native", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
