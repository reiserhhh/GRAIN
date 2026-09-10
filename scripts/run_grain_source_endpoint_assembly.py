#!/usr/bin/env python3
"""NEXT77: zero-fit source endpoint assembly and two-link component accounting."""

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
from grain_research.vector_mechanisms import source_endpoint_assembly as fn
from grain_research.vector_mechanisms.selection_endpoint import (
    key_vectors,
    source_reader,
)
from grain_research.vector_mechanisms.source_query_selection import source_query
from grain_research.vector_mechanisms.source_state_transport import (
    body_directions,
    source_set,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_value_channel as previous
from scripts.run_grain_selection_endpoint import frozen
from scripts.run_grain_source_state_transport import selected

old, prior = previous.old, previous.prior
STATE, SOURCE, CAPTURE = previous.STATE, previous.SOURCE, previous.CAPTURE
SELECTOR = previous.previous.OUT
ENDPOINT = old.BASE / "logic_selection_endpoint_v1"
OUT = old.BASE / "logic_source_endpoint_assembly_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_ENDPOINT_ASSEMBLY_PLAN_V1.md"
CAP = 60
TABLES = (
    "query_actions.csv",
    "factor_actions.csv",
    "contrast_actions.csv",
    "scalar_metrics.csv",
    "component_summary.csv",
    "reader_error_account.csv",
    "vector_metrics.csv",
)
COUNTS = [144, 576, 72, 72, 72, 288, 108]


def limit(start, used):
    """Bound all numeric stages, including one exact replay."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("NEXT77 cumulative CPU budget exceeded")


def freeze(out, folder):
    """Check file hashes and the exact stage inventory, excluding Apple metadata."""
    result = prior.freeze(out, folder)
    names = {
        p.name
        for p in (out / folder).iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if names != set(result["files"]) | {"freeze.json"}:
        raise ValueError("Unexpected stage files")
    return result


def prepare(out):
    """Rebuild only source readers; verify exact old source membership and bytes."""
    start = time.process_time()
    manifest = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT76 required")
    jobs = old.read(previous.OUT / "private_jobs.json")
    if len(jobs) != 12 or jobs != old.read(ENDPOINT / "private_jobs.json"):
        raise ValueError("Exact paired source4/new8 inventory required")
    out.mkdir(exist_ok=False)
    (out / "private_readers").mkdir()
    old.write(out / "private_jobs.json", jobs)
    prepared = [out / "private_jobs.json"]
    inputs = dict(manifest["inputs"])
    inputs.update(
        {
            str((previous.OUT / n).relative_to(ROOT)): h
            for n, h in manifest["prepared"].items()
        }
    )
    for root, folders in (
        (previous.OUT, ("models",)),
        (SELECTOR, ("models",)),
        (ENDPOINT, ("capture_source", "capture_new", "vectors", "predictions")),
    ):
        for folder in folders:
            f = frozen(root, folder)
            inputs.update(
                {
                    str((root / folder / n).relative_to(ROOT)): h
                    for n, h in f["files"].items()
                }
            )
    bank = {
        (j["index"], q): selected(ENDPOINT / "capture_source" / (k + ".npz"), ("r",))[
            "r"
        ]
        for j in jobs
        if j["cohort"] == "source4"
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True)
    }
    checks = []
    for j in jobs:
        rows = []
        for q in old.QUERIES:
            r, members = source_reader(bank, j["cohort"], j["index"], q)
            rows.append(r)
            checks.append(
                {
                    "cohort": j["cohort"],
                    "index": j["index"],
                    "query": q,
                    "members": members,
                }
            )
        pair = np.stack(rows)
        np.testing.assert_array_equal(
            pair,
            selected(ENDPOINT / "predictions" / (j["key"] + ".npz"), ("readers",))[
                "readers"
            ],
        )
        fixed = STATE / "private_fixed" / (j["key"] + ".npz")
        np.testing.assert_array_equal(pair, selected(fixed, ("r",))["r"])
        inputs[str(fixed.relative_to(ROOT))] = old.sha(fixed)
        p = out / "private_readers" / (j["key"] + ".npz")
        np.savez(
            p, r=fn.readers(pair), members=np.array(source_set(j["cohort"], j["index"]))
        )
        prepared.append(p)
    paths = [
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        ROOT / "tests/test_grain_source_endpoint_assembly.py",
        previous.OUT / "audit.json",
        STATE / "private_weights.npz",
        CAPTURE / "private_weights.npz",
    ]
    inputs.update({str(p.relative_to(ROOT)): old.sha(p) for p in paths})
    result = {
        "protocol": "GRAIN_SOURCE_ENDPOINT_ASSEMBLY_V1",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "prepare_cpu_seconds": time.process_time() - start,
        "inputs": inputs,
        "prepared": {str(p.relative_to(out)): old.sha(p) for p in prepared},
        "source_checks": checks,
        "max_fits": 0,
        "max_cpu_seconds": CAP,
        "H16_retained": True,
        "current_readout_retained": False,
        "current_native_values_retained": False,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    old.write(out / "manifest.json", result)
    print(
        {
            "status": "PREPARED",
            "cpu_seconds": result["prepare_cpu_seconds"],
            "source_reader_checks": len(checks),
        },
        flush=True,
    )


def predict(out, folder):
    """Use H16 and frozen source functions only; deny current g and native targets."""
    old.verify(out)
    fm = prior.freeze(previous.OUT, "models")
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [k + ".npz" for j in jobs for k in j["query_keys"]]
    reads = [STATE / "private_weights.npz"] + [
        STATE / "private_inputs" / n for n in names
    ]
    reads += [out / "private_readers" / (j["key"] + ".npz") for j in jobs]
    reads += [
        root / "models" / n
        for root in (previous.OUT, SELECTOR)
        for n in fm["files"]
        if n.endswith(".npz")
    ]
    probes = [
        previous.OUT / "private_readers" / names[0],
        CAPTURE / "private_capture" / names[0],
        SOURCE / "private_inputs" / names[0],
        ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        previous.OUT / "predictions" / names[0],
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    weights, errors = old.arrays(reads[0]), []
    if "q" in weights or "q_bias" in weights:
        raise ValueError("Q matrices forbidden in application")
    for j in jobs:
        reader = old.arrays(out / "private_readers" / (j["key"] + ".npz"))
        members = source_set(j["cohort"], j["index"])
        np.testing.assert_array_equal(reader["members"], members)
        for qi, (q, key) in enumerate(zip(old.QUERIES, j["query_keys"], strict=True)):
            data = old.arrays(STATE / "private_inputs" / (key + ".npz"))
            name = f"{prior.reference(j)}_{q}.npz"
            sm, vm = (
                old.arrays(SELECTOR / "models" / name),
                old.arrays(previous.OUT / "models" / name),
            )
            for model in (sm, vm):
                np.testing.assert_array_equal(model["members"], members)
                if str(model["query"]) != q:
                    raise ValueError("Query membership mismatch")
            d, r = body_directions(data), reader["r"][qi]
            p = fn.predict(data, d, weights, sm, vm, r)
            np.savez(dst / (key + ".npz"), **p)
            if j["cohort"] == "source4" and j["index"] == 0:
                for f in range(4):
                    s = [
                        fn.predict(
                            dict(data, state=data["state"] + e * d[f]),
                            d,
                            weights,
                            sm,
                            vm,
                            r,
                        )["scalar"]
                        for e in (-1e-5, 1e-5)
                    ]
                    errors.extend(
                        abs((s[1] - s[0]) / 2e-5 - p["actions"][:, 2, :, f]).ravel()
                    )
            limit(start, used)
    if len(errors) != 64 or max(errors) > 1e-7:
        raise ValueError("Full scalar finite difference failed")
    previous.finish(
        out,
        folder,
        names,
        start,
        fd_count=len(errors),
        fd_max_error=float(max(errors)),
        denial_probes=len(probes),
        extra_denials=len(denied) - len(probes),
        fits=0,
        current_gradient_reads=0,
        current_value_reads=0,
        target_reads=0,
    )


def qview(x, index):
    """Return either query or the predeclared paired N-minus-confirmation contrast."""
    return x[:, index] if index < 2 else x[:, 0] - x[:, 1]


def summarize(groups):
    """Publish all arms and paired contrasts, retaining both errors and their cross term."""
    qr, fr, cr, sr, cs, er, vr = [], [], [], [], [], [], []
    for cohort, rows in groups.items():
        x = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        for ai, arm in enumerate(fn.ARMS):
            for mi, mode in enumerate(fn.MODES):
                meta = {
                    "cohort": cohort,
                    "authors": len(rows),
                    "arm": arm,
                    "mode": mode,
                }
                for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
                    a = qview(x["actions"], qi)[:, ai, :, mi]
                    y = qview(x["target"], qi)[:, :, mi]
                    transported = qview(x["transported"], qi)[:, ai, :, mi]
                    for ci, component in enumerate(fn.COMPONENTS):
                        cm = meta | {"component": component}
                        metric = old.fidelity(y[:, ci].ravel(), a[:, ci].ravel())
                        if qi < 2:
                            qr.append(cm | {"query": q} | metric)
                            for f in range(4):
                                fr.append(
                                    cm
                                    | {"query": q, "factor": f"whole:F{f + 1:02d}"}
                                    | old.fidelity(y[:, ci, f], a[:, ci, f])
                                )
                        else:
                            cr.append(cm | metric)
                        er.append(
                            cm
                            | {"query": q}
                            | fn.error_metrics(y[:, ci], transported[:, ci], a[:, ci])
                        )
                    sy, sp, st = (
                        qview(x["target_scalar"], qi)[:, mi],
                        qview(x["scalar"], qi)[:, ai, mi],
                        qview(x["scalar_transported"], qi)[:, ai, mi],
                    )
                    sr.append(meta | {"query": q} | previous.scalar_metrics(sy, sp))
                    er.append(
                        meta
                        | {"query": q, "component": "scalar"}
                        | fn.error_metrics(sy, st, sp)
                    )
                    k, v, total = (a[:, ci].ravel() for ci in range(3))
                    ek, ev, cross = np.mean(k * k), np.mean(v * v), 2 * np.mean(k * v)
                    np.testing.assert_allclose(
                        np.mean(total * total), ek + ev + cross, atol=1e-10
                    )
                    cs.append(
                        meta
                        | {
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
            for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
                for ci, component in enumerate(fn.COMPONENTS):
                    y = qview(x["target_vectors"], qi)[:, ci]
                    p = qview(x["vectors"], qi)[:, ai, ci]
                    metric = {
                        k.replace("action", "vector"): v
                        for k, v in old.fidelity(y.ravel(), p.ravel()).items()
                    }
                    vr.append(
                        {
                            "cohort": cohort,
                            "authors": len(rows),
                            "arm": arm,
                            "query": q,
                            "component": component,
                        }
                        | metric
                    )
    return [pd.DataFrame(r) for r in (qr, fr, cr, sr, cs, er, vr)]


def evaluate(out, folder):
    """Open native endpoint readers only after frozen application outputs exist."""
    old.verify(out)
    freeze(out, folder)
    used, start = prior.prior_cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    w, qw = (
        old.arrays(STATE / "private_weights.npz"),
        old.arrays(CAPTURE / "private_weights.npz"),
    )
    groups, errors, account_errors = {"source4": [], "new8": []}, [], []
    for j in old.read(out / "private_jobs.json"):
        rc = fn.readers(
            np.stack(
                [
                    selected(
                        ENDPOINT
                        / (
                            "capture_source"
                            if j["cohort"] == "source4"
                            else "capture_new"
                        )
                        / (k + ".npz"),
                        ("r",),
                    )["r"]
                    for k in j["query_keys"]
                ]
            )
        )
        rb = old.arrays(out / "private_readers" / (j["key"] + ".npz"))["r"]
        old_prediction = old.arrays(ENDPOINT / "predictions" / (j["key"] + ".npz"))
        rows = []
        for qi, (q, key) in enumerate(zip(old.QUERIES, j["query_keys"], strict=True)):
            p = old.arrays(out / folder / (key + ".npz"))
            cap = old.arrays(CAPTURE / "private_capture" / (key + ".npz"))
            e = old.arrays(SOURCE / "private_inputs" / (key + ".npz"))
            sm = old.arrays(SELECTOR / "models" / f"{prior.reference(j)}_{q}.npz")
            v, dv = fn.value.values(cap["x"], cap["d"], w)
            np.testing.assert_array_equal(v, e["v"])
            anchors = (sm["anchor"], source_query(cap["x"][-1], qw))
            native = []
            for anchor in anchors:
                ell, dell = fn.value.scalar.logits(
                    anchor, e["k"], e["dk"], e, float(e["scale"])
                )
                native.append(fn.vectors(ell, dell, e["mask"], v, dv, w["o"]))
            evidence = {k: v for k, v in e.items() if k != "g"}
            for anchor, packet in zip(anchors, native, strict=True):
                check = key_vectors(anchor[None], evidence, w["o"])[0]
                errors.append(float(np.max(abs(check - packet["vectors"][0]))))
            errors.append(
                float(
                    np.max(
                        abs(
                            native[0]["vectors"][0]
                            - old.arrays(ENDPOINT / "vectors" / (key + ".npz"))[
                                "vectors"
                            ][0]
                        )
                    )
                )
            )
            errors.append(
                float(
                    np.max(
                        abs(
                            native[0]["vectors"][0] @ rb[qi, 0]
                            - old_prediction["actions"][qi, 0]
                        )
                    )
                )
            )
            vectors = np.concatenate(
                (p["vectors"], np.stack([a["vectors"] for a in native]))
            )
            primal = np.concatenate(
                (p["primal"], np.stack([a["primal"] for a in native]))
            )
            actions, scalars = (
                np.einsum("acfh,mh->acmf", vectors, rb[qi]),
                primal @ rb[qi].T,
            )
            errors.extend(
                [
                    float(np.max(abs(actions[:4] - p["actions"]))),
                    float(np.max(abs(scalars[:4] - p["scalar"]))),
                ]
            )
            transported = np.einsum("acfh,mh->acmf", vectors, rc[qi])
            ts = fn.contract(native[1], rc[qi])
            direct_reader = np.einsum("acfh,mh->acmf", vectors, rb[qi] - rc[qi])
            account_errors.append(
                float(np.max(abs(actions - transported - direct_reader)))
            )
            rows.append(
                {
                    "actions": actions,
                    "transported": transported,
                    "scalar": scalars,
                    "scalar_transported": primal @ rc[qi].T,
                    "target": ts["actions"],
                    "target_scalar": ts["scalar"],
                    "vectors": vectors,
                    "target_vectors": native[1]["vectors"],
                }
            )
            limit(start, used)
        groups[j["cohort"]].append({k: np.stack([r[k] for r in rows]) for k in rows[0]})
    if max(errors) > 1e-9 or max(account_errors) > 1e-9:
        raise ValueError("Old key replay or exact reader error identity failed")
    tables = summarize(groups)
    if [len(t) for t in tables] != COUNTS or tables[5].identity_error.max() > 1e-9:
        raise ValueError("Table inventory or squared error identity failed")
    for name, table in zip(TABLES, tables, strict=True):
        table.to_csv(dst / name, index=False)
    limit(start, used)
    previous.finish(
        out,
        dst.name,
        TABLES,
        start,
        counts=COUNTS,
        old_key_replay_max_error=max(errors),
        reader_identity_max_error=max(account_errors),
        squared_error_identity_max_error=float(tables[5].identity_error.max()),
    )


def audit(out):
    """Check exact numerical replay, no-fit routing and the complete published grid."""
    manifest = old.verify(out)
    a, b, aa, bb = [
        freeze(out, n) for n in ("predictions", "replay", "analysis", "replay_analysis")
    ]
    if (
        len(a["files"]) != 24
        or set(a["files"]) != set(b["files"])
        or aa["files"] != bb["files"]
        or aa["counts"] != COUNTS
    ):
        raise ValueError("Exact output inventory/replay failed")
    for name in a["files"]:
        x, y = [old.arrays(out / n / name) for n in ("predictions", "replay")]
        if set(x) != set(y):
            raise ValueError("Replay fields mismatch")
        for k in x:
            np.testing.assert_array_equal(x[k], y[k])
        np.testing.assert_array_equal(
            x["vectors"][:, 2], x["vectors"][:, 0] + x["vectors"][:, 1]
        )
        np.testing.assert_allclose(
            x["actions"][:, 2], x["actions"][:, 0] + x["actions"][:, 1], atol=1e-10
        )
    if (
        any(
            f["extra_denials"]
            or f["fits"]
            or f["current_gradient_reads"]
            or f["target_reads"]
            for f in (a, b)
        )
        or prior.prior_cpu(out) > CAP
    ):
        raise ValueError("Application boundary or budget exceeded")
    result = {
        "status": "PASS",
        "fits": 0,
        "packets": 24,
        "csv_replay": len(TABLES),
        "total_cpu_seconds": prior.prior_cpu(out),
        "fd_count": a["fd_count"],
        "fd_max_error": a["fd_max_error"],
        "old_key_replay_max_error": aa["old_key_replay_max_error"],
        "reader_identity_max_error": aa["reader_identity_max_error"],
        "squared_error_identity_max_error": aa["squared_error_identity_max_error"],
        "source_reader_checks": len(manifest["source_checks"]),
        "H16_retained": True,
        "current_readout_retained": False,
        "current_native_values_retained": False,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run immutable stages in separate single-threaded numeric processes."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", required=True, choices=("prepare", "predict", "evaluate", "audit")
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
