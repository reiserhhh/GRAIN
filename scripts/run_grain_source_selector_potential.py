#!/usr/bin/env python3
"""NEXT74: bounded source fit, target-denied application and independent evaluation."""

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
from grain_research.vector_mechanisms import source_selector_potential as fn
from grain_research.vector_mechanisms.source_state_transport import (
    body_directions,
    source_set,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_query_selection as old

OUT = old.BASE / "logic_source_selector_potential_v1"
STATE = old.BASE / "logic_source_state_transport_v1"
SOURCE = old.BASE / "logic_source_query_selection_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_SELECTOR_POTENTIAL_PLAN_V1.md"
CAP = 120
TABLES = (
    "selector_metrics.csv",
    "query_actions.csv",
    "factor_actions.csv",
    "contrast_actions.csv",
)


def reference(j):
    """Select the unchanged author-disjoint source reference."""
    return "exclude" + str(j["index"]) if j["cohort"] == "source4" else "all4"


def freeze(out, folder):
    """Verify an immutable stage inventory before using any of its arrays."""
    f = old.read(out / folder / "freeze.json")
    if f["status"] != "COMPLETE":
        raise ValueError("Complete stage required")
    old.verify_hashes(out / folder, f["files"])
    return f


def prior_cpu(out):
    """Account for completed primary and replay stages against one shared cap."""
    total = old.read(out / "manifest.json")["prepare_cpu_seconds"]
    for folder in ("models", "predictions", "analysis", "replay", "replay_analysis"):
        path = out / folder / "freeze.json"
        if path.exists():
            total += old.read(path)["cpu_seconds"]
    return total


def limit(start, used):
    """Fail without further fitting if the finite computation budget is exhausted."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("Cumulative 120 CPU second NEXT74 budget exceeded")


def prepare(out):
    """Copy only current V/g and register exact H16/layout/source addresses."""
    start = time.process_time()
    old.verify(STATE)
    old.verify(SOURCE)
    fm = freeze(STATE, "models")
    jobs = old.read(STATE / "private_jobs.json")
    if jobs != old.read(SOURCE / "private_jobs.json") or len(jobs) != 12:
        raise ValueError("Exact paired twelve-author jobs required")
    out.mkdir(exist_ok=False)
    (out / "private_values").mkdir()
    prepared, inputs = (
        [],
        [
            STATE / "manifest.json",
            SOURCE / "manifest.json",
            STATE / "models/freeze.json",
        ],
    )
    old.write(out / "private_jobs.json", jobs)
    prepared.append(out / "private_jobs.json")
    for j in jobs:
        for key in j["query_keys"]:
            p, e = (
                STATE / "private_inputs" / (key + ".npz"),
                SOURCE / "private_inputs" / (key + ".npz"),
            )
            data, native = old.arrays(p), old.arrays(e)
            for name in ("cos", "sin", "mask"):
                np.testing.assert_array_equal(data[name], native[name])
            if len(data["state"]) != len(native["k"]):
                raise ValueError("Token alignment failure")
            path = out / "private_values" / (key + ".npz")
            np.savez(path, v=native["v"], g=native["g"])
            prepared.append(path)
            inputs.extend([p, e])
    refs = sorted({reference(j) for j in jobs})
    inputs += [STATE / "models" / (r + "_causal_context.npz") for r in refs]
    if any(p.name not in fm["files"] for p in inputs[-len(refs) :]):
        raise ValueError("Source base model inventory failure")
    inputs += [
        STATE / "private_weights.npz",
        SOURCE / "private_source_bank.npz",
        PLAN,
        Path(__file__),
        ROOT / "tests/test_grain_source_selector_potential.py",
    ]
    inputs += [
        ROOT / "grain_research/vector_mechanisms" / (n + ".py")
        for n in (
            "source_selector_potential",
            "source_state_transport",
            "source_query_selection",
            "fixed_seed_attention",
            "upstream_attention_channels",
            "native_material_reader",
            "stage_file_guard",
        )
    ]
    inputs += [
        ROOT / "scripts" / (n + ".py")
        for n in (
            "run_grain_source_query_selection",
            "run_grain_compiled_query_transfer",
        )
    ]
    cpu = time.process_time() - start
    if cpu > CAP:
        raise TimeoutError("Preparation budget exceeded")
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_SELECTOR_POTENTIAL_V1",
            "prepare_cpu_seconds": cpu,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": {str(p.relative_to(ROOT)): old.sha(p) for p in inputs},
            "prepared": {str(p.relative_to(out)): old.sha(p) for p in prepared},
            "max_fits": 20,
            "max_cpu_seconds": CAP,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "native_values_retained": True,
        },
    )
    print({"status": "PREPARED", "cpu_seconds": cpu, "value_packets": 24}, flush=True)


def fit(out):
    """Twenty split-specific query fits; target/new8 files denied in this process."""
    old.verify(out)
    used, start = prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    src = [j for j in jobs if j["cohort"] == "source4"]
    refs = {reference(j): source_set(j["cohort"], j["index"]) for j in jobs}
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [
        f"{r}_{q}_{o}.npz"
        for r in sorted(refs)
        for q in old.QUERIES
        for o in fn.OBJECTIVES
    ]
    reads = [STATE / "private_weights.npz", SOURCE / "private_source_bank.npz"]
    reads += [
        base / "private_inputs" / (k + ".npz")
        for j in src
        for k in j["query_keys"]
        for base in (STATE, SOURCE)
    ]
    reads += [STATE / "models" / (r + "_causal_context.npz") for r in refs]
    new = next(j for j in jobs if j["cohort"] == "new8")
    probes = [
        STATE / "private_inputs" / (new["query_keys"][0] + ".npz"),
        SOURCE / "private_inputs" / (new["query_keys"][0] + ".npz"),
        old.SOURCE / "predictions" / (src[0]["key"] + ".npz"),
    ]
    denied = install(
        reads,
        [dst / n for n in names] + [dst / "freeze.json", dst / "source_losses.csv"],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    weights, bank = old.arrays(reads[0]), old.arrays(reads[1])
    items = []
    for j in src:
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True):
            data = old.arrays(STATE / "private_inputs" / (k + ".npz"))
            native = old.arrays(SOURCE / "private_inputs" / (k + ".npz"))
            d = body_directions(data)
            k0, dk0 = fn.identity_keys(data, d, weights)
            items.append(
                {
                    "index": j["index"],
                    "query": q,
                    "data": data,
                    "d": d,
                    "k0": k0,
                    "dk0": dk0,
                    "k": native["k"],
                    "dk": native["dk"],
                }
            )
    training = []
    for r, members in sorted(refs.items()):
        base = old.arrays(STATE / "models" / (r + "_causal_context.npz"))
        for q in old.QUERIES:
            cohort, index = ("new8", 0) if r == "all4" else ("source4", int(r[-1]))
            anchors, check = old.anchors(bank, cohort, index, q)
            if check != members:
                raise ValueError("Split query anchor contamination")
            models, losses = fn.fit(
                items, members, q, base, anchors[0], float(weights["scale"])
            )
            for o, model in models.items():
                np.savez(dst / f"{r}_{q}_{o}.npz", **model)
                training.append(
                    {
                        "reference": r,
                        "query": q,
                        "objective": o,
                        "source_authors": len(members),
                        **losses[o],
                    }
                )
            limit(start, used)
    pd.DataFrame(training).to_csv(dst / "source_losses.csv", index=False)
    names += ["source_losses.csv"]
    result = {
        "status": "COMPLETE",
        "fits": len(training),
        "cpu_seconds": time.process_time() - start,
        "files": {n: old.sha(dst / n) for n in names},
        "denial_probes": len(probes),
        "extra_denials": len(denied) - len(probes),
        "current_Q_reads": 0,
        "new8_reads": 0,
    }
    old.write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def predict(out, folder):
    """Freeze all scalar paths before opening current native K/dK or target actions."""
    old.verify(out)
    fm = freeze(out, "models")
    used, start = prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    keys = [k for j in jobs for k in j["query_keys"]]
    names = [k + ".npz" for k in keys]
    reads = [STATE / "private_weights.npz"]
    reads += [STATE / "private_inputs" / n for n in names] + [
        out / "private_values" / n for n in names
    ]
    reads += [out / "models" / n for n in fm["files"] if n.endswith(".npz")]
    probes = [
        SOURCE / "private_inputs" / names[0],
        old.SOURCE / "private_capture" / names[0],
        old.SOURCE / "predictions" / (jobs[0]["key"] + ".npz"),
    ]
    denied = install(
        reads,
        [dst / n for n in names] + [dst / "freeze.json"],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    w = old.arrays(reads[0])
    errors, checks = [], []
    for j in jobs:
        for q, key in zip(old.QUERIES, j["query_keys"], strict=True):
            data = old.arrays(STATE / "private_inputs" / (key + ".npz"))
            v = old.arrays(out / "private_values" / (key + ".npz"))
            d = body_directions(data)
            k0 = fn.identity_keys(data, d, w)
            models = [
                old.arrays(out / "models" / f"{reference(j)}_{q}_{o}.npz")
                for o in fn.OBJECTIVES
            ]
            identity = dict(models[0], coef=np.zeros_like(models[0]["coef"]))
            packets = []
            for model in [identity, *models]:
                if (
                    not np.array_equal(
                        model["members"], source_set(j["cohort"], j["index"])
                    )
                    or str(model["query"]) != q
                ):
                    raise ValueError("Prediction source/query reference failure")
                ell, dell = fn.potential(data, d, w, model, k0)
                packet = fn.readout(ell, dell, data["mask"], v) | {
                    "logits": ell,
                    "dlogits": dell,
                }
                packets.append(packet)
                if j["cohort"] == "source4" and j["index"] == 0:
                    for f in range(4):
                        scores = []
                        for t in (-1e-5, 1e-5):
                            moved = dict(data, state=data["state"] + t * d[f])
                            le, dl = fn.potential(moved, d, w, model)
                            scores.append(fn.readout(le, dl, data["mask"], v)["scalar"])
                        errors.extend(
                            abs(
                                (scores[1] - scores[0]) / 2e-5 - packet["actions"][:, f]
                            ).tolist()
                        )
            np.savez(
                dst / (key + ".npz"),
                **{n: np.stack([p[n] for p in packets]) for n in packets[0]},
            )
            checks.append(
                {"key": key, "sources": source_set(j["cohort"], j["index"]), "query": q}
            )
            limit(start, used)
    if len(errors) != 48 or max(errors) > 1e-7:
        raise ValueError("Same-function finite difference failed")
    result = {
        "status": "COMPLETE",
        "cpu_seconds": time.process_time() - start,
        "files": {n: old.sha(dst / n) for n in names},
        "checks": checks,
        "fd_count": len(errors),
        "fd_max_error": max(errors),
        "fits": 0,
        "target_reads": 0,
        "native_current_key_reads": 0,
        "current_Q_reads": 0,
        "native_values_retained": True,
        "new_neural_calls": 0,
        "denial_probes": len(probes),
        "extra_denials": len(denied) - len(probes),
    }
    old.write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k not in ("files", "checks")}, flush=True)


def evaluate(out, folder):
    """Evaluate complete frozen candidates, preserving author-level sampling units."""
    old.verify(out)
    freeze(out, folder)
    freeze(SOURCE, "predictions")
    old.verify_hashes(
        old.SOURCE / "predictions",
        old.read(old.SOURCE / "predictions/prediction_freeze.json")["files"],
    )
    used, start = prior_cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    bank = old.arrays(SOURCE / "private_source_bank.npz")
    rows, grouped, max_error = [], {c: [] for c in ("source4", "new8")}, 0.0
    for j in old.read(out / "private_jobs.json"):
        actions = []
        prior = old.arrays(SOURCE / "predictions" / (j["key"] + ".npz"))["actions"][
            :, 0
        ]
        y = old.arrays(old.SOURCE / "predictions" / (j["key"] + ".npz"))["actions"][
            :, :, 1
        ]
        for qi, (q, key) in enumerate(zip(old.QUERIES, j["query_keys"], strict=True)):
            native = old.arrays(SOURCE / "private_inputs" / (key + ".npz"))
            anchor, _ = old.anchors(bank, j["cohort"], j["index"], q)
            ell, dell = fn.logits(
                anchor[0], native["k"], native["dk"], native, float(native["scale"])
            )
            t = fn.readout(
                ell, dell, native["mask"], {n: native[n] for n in ("v", "g")}
            )
            max_error = max(max_error, float(abs(t["actions"] - prior[qi]).max()))
            p = old.arrays(out / folder / (key + ".npz"))
            actions.append(np.concatenate((p["actions"], t["actions"][None]), axis=0))
            use = fn.allowed(native["mask"], len(ell))
            tl, td = (
                fn.gauge(ell, native["mask"])[use],
                fn.gauge(dell, native["mask"])[:, use],
            )
            for a, arm in enumerate(fn.ARMS):
                pl, dpred = p["logits"][a][use], p["dlogits"][a][:, use]
                kl = np.sum(
                    t["attention"][use]
                    * (t["logattention"][use] - p["logattention"][a][use]),
                    axis=0,
                ).mean()
                rows.append(
                    {
                        "cohort": j["cohort"],
                        "query": q,
                        "arm": arm,
                        "attention_kl": float(kl),
                        "centered_logit_mse": float(np.mean((pl - tl) ** 2)),
                        "centered_logit_sse_zero": float(
                            np.sum((pl - tl) ** 2) / np.sum(tl**2)
                        ),
                        "dlogit_mse": float(np.mean((dpred - td) ** 2)),
                        "dlogit_sse_zero": float(
                            np.sum((dpred - td) ** 2) / np.sum(td**2)
                        ),
                    }
                )
        grouped[j["cohort"]].append((np.stack(actions), y))
    if max_error > 1e-10:
        raise ValueError("Original key-channel comparator does not replay")
    selector = (
        pd.DataFrame(rows).groupby(["cohort", "query", "arm"], as_index=False).mean()
    )
    selector["authors"] = selector["cohort"].map({"source4": 4, "new8": 8})
    qr, fr, cr = [], [], []
    for cohort, materials in grouped.items():
        a, native = [np.stack([v[i] for v in materials]) for i in range(2)]
        for target_name, y in (
            ("source_Q_native_key", a[:, :, -1]),
            ("native_Q_native_key", native),
        ):
            for ai, arm in enumerate((*fn.ARMS, "source_Q_native_key")):
                for mi, mode in enumerate(old.MODES):
                    meta = {
                        "cohort": cohort,
                        "authors": len(materials),
                        "target": target_name,
                        "arm": arm,
                        "mode": mode,
                    }
                    cr.append(
                        meta
                        | old.fidelity(
                            (y[:, 0, mi] - y[:, 1, mi]).ravel(),
                            (a[:, 0, ai, mi] - a[:, 1, ai, mi]).ravel(),
                        )
                    )
                    for qi, q in enumerate(old.QUERIES):
                        qm = meta | {"query": q}
                        qr.append(
                            qm
                            | old.fidelity(
                                y[:, qi, mi].ravel(), a[:, qi, ai, mi].ravel()
                            )
                        )
                        for f in range(4):
                            fr.append(
                                qm
                                | {"factor": f"whole:F{f + 1:02d}"}
                                | old.fidelity(y[:, qi, mi, f], a[:, qi, ai, mi, f])
                            )
    tables = [selector, *[pd.DataFrame(r) for r in (qr, fr, cr)]]
    for name, table in zip(TABLES, tables, strict=True):
        table.to_csv(dst / name, index=False)
    limit(start, used)
    result = {
        "status": "COMPLETE",
        "cpu_seconds": time.process_time() - start,
        "files": {n: old.sha(dst / n) for n in TABLES},
        "counts": [len(t) for t in tables],
        "teacher_replay_max_error": max_error,
    }
    old.write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def audit(out):
    """Exact replay and bounded-source/derivative checks, not an efficacy gate."""
    old.verify(out)
    fm = freeze(out, "models")
    a, b = freeze(out, "predictions"), freeze(out, "replay")
    if set(a["files"]) != set(b["files"]) or len(a["files"]) != 24 or fm["fits"] != 20:
        raise ValueError("Fit or prediction inventory mismatch")
    for name in a["files"]:
        x, y = [old.arrays(out / f / name) for f in ("predictions", "replay")]
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])
    aa, bb = freeze(out, "analysis"), freeze(out, "replay_analysis")
    if aa["files"] != bb["files"] or aa["counts"] != [12, 64, 256, 32]:
        raise ValueError("Exact aggregate replay/cardinality failed")
    if any(f["extra_denials"] for f in (fm, a, b)) or prior_cpu(out) > CAP:
        raise ValueError("File-boundary or cumulative budget failure")
    result = {
        "status": "PASS",
        "fits": 20,
        "packets": 24,
        "csv_replay": 4,
        "total_cpu_seconds": prior_cpu(out),
        "fd_count": a["fd_count"],
        "fd_max_error": a["fd_max_error"],
        "teacher_replay_max_error": aa["teacher_replay_max_error"],
        "current_native_keys_in_prediction": False,
        "current_Q_read": False,
        "new_neural_calls": 0,
        "human_labels_read": False,
        "native_values_retained": True,
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run each stage in a separate process; never refit from evaluation results."""
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
