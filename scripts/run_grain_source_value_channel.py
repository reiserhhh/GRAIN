#!/usr/bin/env python3
"""NEXT76: source value function, target-denied application, complete K/V account."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms import source_value_channel as fn
from grain_research.vector_mechanisms.source_query_selection import source_query
from grain_research.vector_mechanisms.source_state_transport import (
    body_directions,
    source_set,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_response_coordinates as previous

old, prior = previous.old, previous.prev
STATE, SOURCE, CAPTURE = previous.STATE, previous.SOURCE, old.SOURCE
OUT = old.BASE / "logic_source_value_channel_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_VALUE_CHANNEL_PLAN_V1.md"
CAP = 180
TABLES = (
    "value_metrics.csv",
    "query_actions.csv",
    "factor_actions.csv",
    "contrast_actions.csv",
    "scalar_metrics.csv",
    "component_summary.csv",
)
ALL_ARMS = (
    *fn.ARMS,
    *[f"{s}__native_value" for s in fn.SELECTORS],
    "source_Q__native_value",
)


def limit(start, used):
    """Enforce one cumulative numeric-stage budget, including replay."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("NEXT76 cumulative CPU budget exceeded")


def finish(out, folder, names, start, **extra):
    """Freeze explicit artifacts after a complete bounded stage."""
    result = {
        "status": "COMPLETE",
        "cpu_seconds": time.process_time() - start,
        "files": {n: old.sha(out / folder / n) for n in names},
        **extra,
    }
    old.write(out / folder / "freeze.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def prepare(out):
    """Pin old evidence and export g-only inputs, not current V/dV/K/Q."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT75 required")
    old.verify_hashes(CAPTURE, old.read(CAPTURE / "capture_freeze.json")["files"])
    old.verify_hashes(
        CAPTURE / "predictions",
        old.read(CAPTURE / "predictions/prediction_freeze.json")["files"],
    )
    jobs = old.read(previous.OUT / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Exact twelve-author panel required")
    out.mkdir(exist_ok=False)
    (out / "private_readers").mkdir()
    old.write(out / "private_jobs.json", jobs)
    prepared = [out / "private_jobs.json"]
    inputs = dict(m["inputs"])
    inputs.update(
        {str((previous.OUT / n).relative_to(ROOT)): h for n, h in m["prepared"].items()}
    )
    for folder in ("models", "predictions"):
        f = prior.freeze(previous.OUT, folder)
        inputs.update(
            {
                str((previous.OUT / folder / n).relative_to(ROOT)): h
                for n, h in f["files"].items()
            }
        )
    paths = [
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        Path(previous.__file__),
        ROOT / "tests/test_grain_source_value_channel.py",
        CAPTURE / "capture_freeze.json",
        CAPTURE / "private_weights.npz",
        CAPTURE / "predictions/prediction_freeze.json",
        previous.OUT / "audit.json",
    ]
    for j in jobs:
        paths.append(CAPTURE / "predictions" / (j["key"] + ".npz"))
        for key in j["query_keys"]:
            src = prior.OUT / "private_values" / (key + ".npz")
            dst = out / "private_readers" / (key + ".npz")
            np.savez(dst, g=old.arrays(src)["g"])
            prepared.append(dst)
            paths += [src, CAPTURE / "private_capture" / (key + ".npz")]
    inputs.update({str(p.relative_to(ROOT)): old.sha(p) for p in paths})
    result = {
        "protocol": "GRAIN_SOURCE_VALUE_CHANNEL_V1",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "prepare_cpu_seconds": time.process_time() - start,
        "inputs": inputs,
        "prepared": {str(p.relative_to(out)): old.sha(p) for p in prepared},
        "max_fits": 10,
        "max_cpu_seconds": CAP,
        "alpha": 0.01,
        "native_values_retained": False,
        "fixed_native_readout_retained": True,
        "H16_retained": True,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    old.write(out / "manifest.json", result)
    print(
        {"status": "PREPARED", "cpu_seconds": result["prepare_cpu_seconds"]}, flush=True
    )


def fit(out):
    """Ten source V/dV solves; new8 payloads and current actions inaccessible."""
    old.verify(out)
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    src = [j for j in jobs if j["cohort"] == "source4"]
    refs = {prior.reference(j): source_set(j["cohort"], j["index"]) for j in jobs}
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [f"{r}_{q}.npz" for r in sorted(refs) for q in old.QUERIES]
    reads = [STATE / "private_weights.npz"]
    reads += [
        STATE / "private_inputs" / (k + ".npz") for j in src for k in j["query_keys"]
    ]
    reads += [
        CAPTURE / "private_capture" / (k + ".npz") for j in src for k in j["query_keys"]
    ]
    reads += [previous.OUT / "models" / n for n in names]
    new = next(j for j in jobs if j["cohort"] == "new8")
    probes = [
        STATE / "private_inputs" / (new["query_keys"][0] + ".npz"),
        CAPTURE / "private_capture" / (new["query_keys"][0] + ".npz"),
        previous.OUT / "predictions" / (src[0]["query_keys"][0] + ".npz"),
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "source_training.csv", "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    weights, items = old.arrays(reads[0]), []
    for j in src:
        for q, key in zip(old.QUERIES, j["query_keys"], strict=True):
            data = old.arrays(STATE / "private_inputs" / (key + ".npz"))
            c = old.arrays(CAPTURE / "private_capture" / (key + ".npz"))
            d = body_directions(data)
            v0, dv0 = fn.values(data["state"], d, weights)
            v, dv = fn.values(c["x"], c["d"], weights)
            items.append(
                {
                    "index": j["index"],
                    "query": q,
                    "data": data,
                    "d": d,
                    "v0": v0,
                    "dv0": dv0,
                    "v": v,
                    "dv": dv,
                }
            )
    training = []
    for r, members in sorted(refs.items()):
        for q in old.QUERIES:
            base = old.arrays(previous.OUT / "models" / f"{r}_{q}.npz")
            model, tr = fn.fit(items, members, q, base)
            np.savez(dst / f"{r}_{q}.npz", **model)
            training.append(
                dict(reference=r, source_authors=len(members), query=q, **tr)
            )
            limit(start, used)
    pd.DataFrame(training).to_csv(dst / "source_training.csv", index=False)
    finish(
        out,
        "models",
        [*names, "source_training.csv"],
        start,
        fits=len(training),
        denial_probes=len(probes),
        extra_denials=len(denied) - len(probes),
        new8_reads=0,
    )


def predict(out, folder):
    """Apply four complete local functions using H16, source maps and g-only."""
    old.verify(out)
    fm = prior.freeze(out, "models")
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [k + ".npz" for j in jobs for k in j["query_keys"]]
    reads = [STATE / "private_weights.npz"]
    reads += [STATE / "private_inputs" / n for n in names] + [
        out / "private_readers" / n for n in names
    ]
    reads += [
        root / "models" / n
        for root in (out, previous.OUT)
        for n in fm["files"]
        if n.endswith(".npz")
    ]
    probes = [
        prior.OUT / "private_values" / names[0],
        SOURCE / "private_inputs" / names[0],
        CAPTURE / "private_capture" / names[0],
        previous.OUT / "predictions" / names[0],
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    weights, errors, bias_errors = old.arrays(reads[0]), [], []
    for j in jobs:
        for q, key in zip(old.QUERIES, j["query_keys"], strict=True):
            data = old.arrays(STATE / "private_inputs" / (key + ".npz"))
            reader = old.arrays(out / "private_readers" / (key + ".npz"))
            name = f"{prior.reference(j)}_{q}.npz"
            sm, vm = (
                old.arrays(previous.OUT / "models" / name),
                old.arrays(out / "models" / name),
            )
            for model in (sm, vm):
                if str(model["query"]) != q or not np.array_equal(
                    model["members"], source_set(j["cohort"], j["index"])
                ):
                    raise ValueError("Source/query split mismatch")
            d = body_directions(data)
            p = fn.predict(data, d, weights, sm, vm, reader)
            np.savez(dst / (key + ".npz"), **p)
            bias_errors.append(float(np.max(abs(p["bias_actions"]))))
            if j["cohort"] == "source4" and j["index"] == 0:
                for f in range(4):
                    scores = [
                        fn.predict(
                            dict(data, state=data["state"] + e * d[f]),
                            d,
                            weights,
                            sm,
                            vm,
                            reader,
                        )["scalar"]
                        for e in (-1e-5, 1e-5)
                    ]
                    errors.extend(
                        abs(
                            (scores[1] - scores[0]) / 2e-5 - p["actions"][:, 2, :, f]
                        ).ravel()
                    )
            limit(start, used)
    if len(errors) != 64 or max(errors) > 1e-7 or max(bias_errors) > 1e-9:
        raise ValueError("Compatible total derivative or constant-bias check failed")
    finish(
        out,
        folder,
        names,
        start,
        fd_count=len(errors),
        fd_max_error=float(max(errors)),
        bias_action_max_error=max(bias_errors),
        denial_probes=len(probes),
        extra_denials=len(denied) - len(probes),
        native_value_reads=0,
        target_reads=0,
        native_values_retained=False,
        fixed_native_readout_retained=True,
    )


def scalar_metrics(y, p):
    """Use the same finite-sample statistics, labeled as local scalar fidelity."""
    return {k.replace("action", "scalar"): v for k, v in old.fidelity(y, p).items()}


def summarize(groups, value_rows):
    """Retain every selector/value/component/query/mode cell in aggregate tables."""
    qr, fr, cr, sr, cs = [], [], [], [], []
    for cohort, rows in groups.items():
        a, scalars, target, ts = [np.stack([r[k] for r in rows]) for k in range(4)]
        for ti, tname in enumerate(("source_Q_fixed", "native_Q_fixed")):
            y, sy = target[:, :, ti], ts[:, :, ti]
            for ai, arm in enumerate(ALL_ARMS):
                for mi, mode in enumerate(old.MODES):
                    meta = {
                        "cohort": cohort,
                        "authors": len(rows),
                        "target": tname,
                        "arm": arm,
                        "mode": mode,
                    }
                    for ci, comp in enumerate(fn.COMPONENTS):
                        cm = meta | {"component": comp}
                        cr.append(
                            cm
                            | old.fidelity(
                                (y[:, 0, ci, mi] - y[:, 1, ci, mi]).ravel(),
                                (a[:, 0, ai, ci, mi] - a[:, 1, ai, ci, mi]).ravel(),
                            )
                        )
                        for qi, q in enumerate(old.QUERIES):
                            qm = cm | {"query": q}
                            qr.append(
                                qm
                                | old.fidelity(
                                    y[:, qi, ci, mi].ravel(),
                                    a[:, qi, ai, ci, mi].ravel(),
                                )
                            )
                            for f in range(4):
                                fr.append(
                                    qm
                                    | {"factor": f"whole:F{f + 1:02d}"}
                                    | old.fidelity(
                                        y[:, qi, ci, mi, f], a[:, qi, ai, ci, mi, f]
                                    )
                                )
                    for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
                        actual = (
                            sy[:, qi, mi] if qi < 2 else sy[:, 0, mi] - sy[:, 1, mi]
                        )
                        pred = (
                            scalars[:, qi, ai, mi]
                            if qi < 2
                            else scalars[:, 0, ai, mi] - scalars[:, 1, ai, mi]
                        )
                        sr.append(meta | {"query": q} | scalar_metrics(actual, pred))
        for ai, arm in enumerate(ALL_ARMS):
            for mi, mode in enumerate(old.MODES):
                for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
                    parts = (
                        a[:, qi, ai, :, mi]
                        if qi < 2
                        else a[:, 0, ai, :, mi] - a[:, 1, ai, :, mi]
                    )
                    k, v, total = (parts[:, c].ravel() for c in range(3))
                    ek, ev, cross = np.mean(k * k), np.mean(v * v), 2 * np.mean(k * v)
                    np.testing.assert_allclose(
                        np.mean(total * total), ek + ev + cross, atol=1e-10
                    )
                    cs.append(
                        {
                            "cohort": cohort,
                            "authors": len(rows),
                            "arm": arm,
                            "mode": mode,
                            "query": q,
                            "key_rms": float(np.sqrt(ek)),
                            "value_rms": float(np.sqrt(ev)),
                            "total_rms": float(np.sqrt(np.mean(total * total))),
                            "signed_cross_energy": float(cross),
                            "total_over_sum_energy": float(
                                np.mean(total * total) / (ek + ev)
                            )
                            if ek + ev
                            else 0.0,
                        }
                    )
    vm = (
        pd.DataFrame(value_rows)
        .groupby(["cohort", "query", "value_arm"], as_index=False)
        .mean()
    )
    return [vm, *[pd.DataFrame(r) for r in (qr, fr, cr, sr, cs)]]


def evaluate(out, folder):
    """Read native values only after prediction freeze; compare matching K/V totals."""
    old.verify(out)
    prior.freeze(out, folder)
    used, start = prior.prior_cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    weights = old.arrays(STATE / "private_weights.npz")
    qw = old.arrays(CAPTURE / "private_weights.npz")
    groups, metrics, errors = {c: [] for c in ("source4", "new8")}, [], []
    for j in old.read(out / "private_jobs.json"):
        acts, scores, targets, target_scores = [], [], [], []
        original = old.arrays(CAPTURE / "predictions" / (j["key"] + ".npz"))["actions"]
        for qi, (q, key) in enumerate(zip(old.QUERIES, j["query_keys"], strict=True)):
            data = old.arrays(STATE / "private_inputs" / (key + ".npz"))
            cap = old.arrays(CAPTURE / "private_capture" / (key + ".npz"))
            e = old.arrays(SOURCE / "private_inputs" / (key + ".npz"))
            reader = old.arrays(out / "private_readers" / (key + ".npz"))
            sm = old.arrays(previous.OUT / "models" / f"{prior.reference(j)}_{q}.npz")
            p = old.arrays(out / folder / (key + ".npz"))
            v, dv = fn.values(cap["x"], cap["d"], weights)
            np.testing.assert_array_equal(v, e["v"])
            d = body_directions(data)
            comp = []
            for s in fn.SELECTORS:
                ell, dell = fn.selector(data, d, weights, sm, s)
                comp.append(fn.readout(ell, dell, data["mask"], v, dv, reader))
            t = []
            for anchor in (sm["anchor"], source_query(cap["x"][-1], qw)):
                ell, dell = fn.scalar.logits(
                    anchor, e["k"], e["dk"], e, float(e["scale"])
                )
                t.append(fn.readout(ell, dell, e["mask"], v, dv, reader))
            errors.extend(
                [
                    float(np.max(abs(t[1]["actions"][0] - original[qi, :, 1]))),
                    float(np.max(abs(t[1]["actions"][1] - original[qi, :, 2]))),
                ]
            )
            snapshots = (
                old.arrays(prior.OUT / "predictions" / (key + ".npz")),
                old.arrays(previous.OUT / "predictions" / (key + ".npz")),
            )
            for si in range(2):
                frozen = {
                    n: snapshots[si][n][0] if si == 0 else snapshots[si][n]
                    for n in ("scalar", "actions")
                }
                errors.extend(
                    [
                        float(np.max(abs(comp[si]["scalar"] - frozen["scalar"]))),
                        float(np.max(abs(comp[si]["actions"][0] - frozen["actions"]))),
                    ]
                )
            comp.append(t[0])
            acts.append(
                np.concatenate((p["actions"], np.stack([r["actions"] for r in comp])))
            )
            scores.append(
                np.concatenate((p["scalar"], np.stack([r["scalar"] for r in comp])))
            )
            targets.append(np.stack([r["actions"] for r in t]))
            target_scores.append(np.stack([r["scalar"] for r in t]))
            use = fn.scalar.allowed(data["mask"], len(v))
            for vi, va in enumerate(fn.VALUES):
                av, ad = p["v"][vi][use], p["dv"][vi][:, use]
                tv, td = v[use], dv[:, use]
                metrics.append(
                    {
                        "cohort": j["cohort"],
                        "query": q,
                        "value_arm": va,
                        "authors": 4 if j["cohort"] == "source4" else 8,
                        "value_mse": float(np.mean((av - tv) ** 2)),
                        "value_sse_zero": float(
                            np.sum((av - tv) ** 2) / np.sum(tv * tv)
                        ),
                        "dvalue_mse": float(np.mean((ad - td) ** 2)),
                        "dvalue_sse_zero": float(
                            np.sum((ad - td) ** 2) / np.sum(td * td)
                        ),
                        "predicted_value_rms": float(np.sqrt(np.mean(av * av))),
                        "actual_value_rms": float(np.sqrt(np.mean(tv * tv))),
                        "bias_column_scalar_rms": float(
                            np.sqrt(np.mean(p["bias_scalar"] ** 2))
                        )
                        if vi
                        else 0.0,
                    }
                )
        groups[j["cohort"]].append(
            tuple(np.stack(x) for x in (acts, scores, targets, target_scores))
        )
        limit(start, used)
    if max(errors) > 1e-9:
        raise ValueError("Native K/V partial or old comparator replay failed")
    tables = summarize(groups, metrics)
    for name, table in zip(TABLES, tables, strict=True):
        table.to_csv(dst / name, index=False)
    limit(start, used)
    finish(
        out,
        dst.name,
        TABLES,
        start,
        counts=[len(t) for t in tables],
        native_replay_max_error=max(errors),
    )


def audit(out):
    """Verify arithmetic, exact replay and declared access/budget without efficacy inflation."""
    old.verify(out)
    fm, a, b, aa, bb = [
        prior.freeze(out, n)
        for n in ("models", "predictions", "replay", "analysis", "replay_analysis")
    ]
    if fm["fits"] != 10 or len(a["files"]) != 24 or set(a["files"]) != set(b["files"]):
        raise ValueError("Fixed inventory mismatch")
    for name in a["files"]:
        x, y = [old.arrays(out / n / name) for n in ("predictions", "replay")]
        if set(x) != set(y):
            raise ValueError("Replay field mismatch")
        for field in x:
            np.testing.assert_array_equal(x[field], y[field])
        np.testing.assert_array_equal(
            x["actions"][:, 2], x["actions"][:, 0] + x["actions"][:, 1]
        )
    if aa["files"] != bb["files"] or aa["counts"] != [8, 336, 1344, 168, 168, 84]:
        raise ValueError("Complete table replay/count mismatch")
    if prior.prior_cpu(out) > CAP or any(f["extra_denials"] for f in (fm, a, b)):
        raise ValueError("Budget/access boundary exceeded")
    result = {
        "status": "PASS",
        "fits": 10,
        "packets": 24,
        "csv_replay": 6,
        "total_cpu_seconds": prior.prior_cpu(out),
        "fd_count": a["fd_count"],
        "fd_max_error": a["fd_max_error"],
        "bias_action_max_error": a["bias_action_max_error"],
        "native_replay_max_error": aa["native_replay_max_error"],
        "native_values_retained": False,
        "fixed_native_readout_retained": True,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run each stage in a separate process with one numeric thread."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "fit", "predict", "evaluate", "audit"),
    )
    p.add_argument("--folder", default="predictions", choices=("predictions", "replay"))
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage in ("predict", "evaluate"):
            globals()[a.stage](a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
