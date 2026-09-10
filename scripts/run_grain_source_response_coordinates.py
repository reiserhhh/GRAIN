#!/usr/bin/env python3
"""NEXT75: source-response coordinates, with isolated application and replay."""

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
from grain_research.vector_mechanisms import source_response_coordinates as fn
from grain_research.vector_mechanisms import source_selector_potential as scalar
from grain_research.vector_mechanisms.source_state_transport import (
    body_directions,
    source_set,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_selector_potential as prev

old, STATE, SOURCE = prev.old, prev.STATE, prev.SOURCE
OUT = old.BASE / "logic_source_response_coordinates_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_RESPONSE_COORDINATES_PLAN_V1.md"
CAP = 180
ARMS = (*scalar.ARMS, "response_coordinates_joint")
TABLES = prev.TABLES


def limit(start, used):
    """Apply one cumulative stage CPU cap including the numerical replay."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("NEXT75 cumulative 180 CPU second budget exceeded")


def prepare(out):
    """Freeze prior assets and new code without reading any new clinical labels."""
    start = time.process_time()
    m = old.verify(prev.OUT)
    if old.read(prev.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT74 required")
    fm, fp, fa = [
        prev.freeze(prev.OUT, n) for n in ("models", "predictions", "analysis")
    ]
    jobs = old.read(prev.OUT / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Exact twelve-author inventory required")
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    inputs = dict(m["inputs"])
    for name, val in m["prepared"].items():
        inputs[str((prev.OUT / name).relative_to(ROOT))] = val
    for name, freeze in (("models", fm), ("predictions", fp), ("analysis", fa)):
        for file, val in freeze["files"].items():
            inputs[str((prev.OUT / name / file).relative_to(ROOT))] = val
    code = [
        PLAN,
        Path(__file__),
        ROOT / "tests/test_grain_source_response_coordinates.py",
        ROOT / "grain_research/vector_mechanisms/source_response_coordinates.py",
        Path(prev.__file__),
        prev.OUT / "audit.json",
    ]
    inputs.update({str(p.relative_to(ROOT)): old.sha(p) for p in code})
    result = {
        "protocol": "GRAIN_SOURCE_RESPONSE_COORDINATES_V1",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "prepare_cpu_seconds": time.process_time() - start,
        "inputs": inputs,
        "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
        "max_fits": 10,
        "source_capacity_accounts": 10,
        "max_cpu_seconds": CAP,
        "rank": 32,
        "alpha": 0.01,
        "pinv_rtol": fn.PINV_RTOL,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    old.write(out / "manifest.json", result)
    print(
        {"status": "PREPARED", "cpu_seconds": result["prepare_cpu_seconds"]}, flush=True
    )


def fit(out):
    """Ten old-source accounts, five shared bases and ten fixed joint fits."""
    old.verify(out)
    used, start = prev.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    source = [j for j in jobs if j["cohort"] == "source4"]
    refs = {prev.reference(j): source_set(j["cohort"], j["index"]) for j in jobs}
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    models = [f"{r}_{q}.npz" for r in refs for q in old.QUERIES]
    tables = ["source_capacity.csv", "source_training.csv", "basis_account.json"]
    reads = [STATE / "private_weights.npz", SOURCE / "private_source_bank.npz"]
    reads += [
        root / "private_inputs" / (key + ".npz")
        for j in source
        for key in j["query_keys"]
        for root in (STATE, SOURCE)
    ]
    reads += [
        prev.OUT / "models" / f"{r}_{q}_joint.npz" for r in refs for q in old.QUERIES
    ]
    new = next(j for j in jobs if j["cohort"] == "new8")
    probes = [
        STATE / "private_inputs" / (new["query_keys"][0] + ".npz"),
        SOURCE / "private_inputs" / (new["query_keys"][0] + ".npz"),
        old.SOURCE / "predictions" / (source[0]["key"] + ".npz"),
    ]
    denied = install(
        reads,
        [dst / n for n in models + tables + ["freeze.json"]],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    weights, bank = old.arrays(reads[0]), old.arrays(reads[1])
    items = []
    for j in source:
        for q, key in zip(old.QUERIES, j["query_keys"], strict=True):
            data = old.arrays(STATE / "private_inputs" / (key + ".npz"))
            native = old.arrays(SOURCE / "private_inputs" / (key + ".npz"))
            d = body_directions(data)
            k0, dk0 = scalar.identity_keys(data, d, weights)
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
    accounts, training, bases = [], [], {}
    for r, members in sorted(refs.items()):
        cohort, index = ("new8", 0) if r == "all4" else ("source4", int(r[-1]))
        anchors = {q: old.anchors(bank, cohort, index, q)[0][0] for q in old.QUERIES}
        for q in old.QUERIES:
            model = old.arrays(prev.OUT / "models" / f"{r}_{q}_joint.npz")
            np.testing.assert_array_equal(model["anchor"], anchors[q])
            s = scalar.source_statistics(
                items, members, q, model, anchors[q], float(weights["scale"])
            )
            accounts.append(
                {
                    "reference": r,
                    "source_authors": len(members),
                    "query": q,
                    **fn.capacity(s, model["coef"]),
                }
            )
        base, bases[r] = fn.build_basis(
            items, members, anchors, float(weights["scale"])
        )
        for q in old.QUERIES:
            model, tr = fn.fit_joint(
                items, members, q, base, anchors[q], float(weights["scale"])
            )
            np.savez(dst / f"{r}_{q}.npz", **model)
            training.append(
                {"reference": r, "source_authors": len(members), "query": q, **tr}
            )
        limit(start, used)
    pd.DataFrame(accounts).to_csv(dst / tables[0], index=False)
    pd.DataFrame(training).to_csv(dst / tables[1], index=False)
    old.write(dst / tables[2], bases)
    result = {
        "status": "COMPLETE",
        "fits": len(training),
        "source_capacity_accounts": len(accounts),
        "source_bases": len(bases),
        "cpu_seconds": time.process_time() - start,
        "files": {n: old.sha(dst / n) for n in models + tables},
        "denial_probes": len(probes),
        "extra_denials": len(denied) - len(probes),
        "new8_reads": 0,
        "current_Q_reads": 0,
    }
    old.write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def predict(out, folder):
    """Apply only the new model; old predictions are not read before this freeze."""
    old.verify(out)
    fm = prev.freeze(out, "models")
    used, start = prev.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [k + ".npz" for j in jobs for k in j["query_keys"]]
    reads = [STATE / "private_weights.npz"] + [
        STATE / "private_inputs" / n for n in names
    ]
    reads += [prev.OUT / "private_values" / n for n in names]
    reads += [out / "models" / n for n in fm["files"] if n.endswith(".npz")]
    probes = [
        SOURCE / "private_inputs" / names[0],
        prev.OUT / "predictions" / names[0],
        old.SOURCE / "private_capture" / names[0],
        old.SOURCE / "predictions" / (jobs[0]["key"] + ".npz"),
    ]
    denied = install(
        reads,
        [dst / n for n in names + ["freeze.json"]],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    w = old.arrays(reads[0])
    errors, checks = [], []
    for j in jobs:
        for q, key in zip(old.QUERIES, j["query_keys"], strict=True):
            data = old.arrays(STATE / "private_inputs" / (key + ".npz"))
            v = old.arrays(prev.OUT / "private_values" / (key + ".npz"))
            model = old.arrays(out / "models" / f"{prev.reference(j)}_{q}.npz")
            if str(model["query"]) != q or not np.array_equal(
                model["members"], source_set(j["cohort"], j["index"])
            ):
                raise ValueError("Exact source/query reference failed")
            d = body_directions(data)
            ell, dell = scalar.potential(data, d, w, model)
            packet = scalar.readout(ell, dell, data["mask"], v) | {
                "logits": ell,
                "dlogits": dell,
            }
            np.savez(dst / (key + ".npz"), **packet)
            if j["cohort"] == "source4" and j["index"] == 0:
                for f in range(4):
                    scores = []
                    for eps in (-1e-5, 1e-5):
                        moved = dict(data, state=data["state"] + eps * d[f])
                        le, dl = scalar.potential(moved, d, w, model)
                        scores.append(scalar.readout(le, dl, data["mask"], v)["scalar"])
                    errors.extend(
                        abs(
                            (scores[1] - scores[0]) / 2e-5 - packet["actions"][:, f]
                        ).tolist()
                    )
            checks.append(
                {"key": key, "query": q, "sources": source_set(j["cohort"], j["index"])}
            )
            limit(start, used)
    if len(errors) != 16 or max(errors) > 1e-7:
        raise ValueError("Actual same-function derivative check failed")
    result = {
        "status": "COMPLETE",
        "cpu_seconds": time.process_time() - start,
        "files": {n: old.sha(dst / n) for n in names},
        "checks": checks,
        "fd_count": len(errors),
        "fd_max_error": max(errors),
        "target_reads": 0,
        "current_key_reads": 0,
        "current_Q_reads": 0,
        "new_neural_calls": 0,
        "native_values_retained": True,
        "denial_probes": len(probes),
        "extra_denials": len(denied) - len(probes),
    }
    old.write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k not in ("files", "checks")}, flush=True)


def evaluate(out, folder):
    """Retain every prior comparator and every new query/factor/mode action."""
    old.verify(out)
    prev.freeze(out, folder)
    prev.freeze(prev.OUT, "predictions")
    prev.freeze(SOURCE, "predictions")
    old.verify_hashes(
        old.SOURCE / "predictions",
        old.read(old.SOURCE / "predictions/prediction_freeze.json")["files"],
    )
    used, start = prev.prior_cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    bank = old.arrays(SOURCE / "private_source_bank.npz")
    rows, groups, error = [], {c: [] for c in ("source4", "new8")}, 0.0
    for j in old.read(out / "private_jobs.json"):
        actions = []
        prior = old.arrays(SOURCE / "predictions" / (j["key"] + ".npz"))["actions"][
            :, 0
        ]
        target = old.arrays(old.SOURCE / "predictions" / (j["key"] + ".npz"))[
            "actions"
        ][:, :, 1]
        for qi, (q, key) in enumerate(zip(old.QUERIES, j["query_keys"], strict=True)):
            native = old.arrays(SOURCE / "private_inputs" / (key + ".npz"))
            anchor = old.anchors(bank, j["cohort"], j["index"], q)[0][0]
            ell, dell = scalar.logits(
                anchor, native["k"], native["dk"], native, float(native["scale"])
            )
            teacher = scalar.readout(
                ell, dell, native["mask"], {n: native[n] for n in ("v", "g")}
            )
            error = max(error, float(abs(teacher["actions"] - prior[qi]).max()))
            oldp = old.arrays(prev.OUT / "predictions" / (key + ".npz"))
            newp = old.arrays(out / folder / (key + ".npz"))
            packet = {n: np.concatenate((oldp[n], newp[n][None]), axis=0) for n in oldp}
            actions.append(
                np.concatenate((packet["actions"], teacher["actions"][None]), axis=0)
            )
            use = scalar.allowed(native["mask"], len(ell))
            tl, td = (
                scalar.gauge(ell, native["mask"])[use],
                scalar.gauge(dell, native["mask"])[:, use],
            )
            for a, arm in enumerate(ARMS):
                pl, dp = packet["logits"][a][use], packet["dlogits"][a][:, use]
                kl = np.sum(
                    teacher["attention"][use]
                    * (teacher["logattention"][use] - packet["logattention"][a][use]),
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
                        "dlogit_mse": float(np.mean((dp - td) ** 2)),
                        "dlogit_sse_zero": float(
                            np.sum((dp - td) ** 2) / np.sum(td**2)
                        ),
                    }
                )
        groups[j["cohort"]].append((np.stack(actions), target))
    if error > 1e-10:
        raise ValueError("Original native key target failed replay")
    selector = (
        pd.DataFrame(rows).groupby(["cohort", "query", "arm"], as_index=False).mean()
    )
    selector["authors"] = selector["cohort"].map({"source4": 4, "new8": 8})
    qr, fr, cr = [], [], []
    for cohort, data in groups.items():
        a, native = [np.stack([v[i] for v in data]) for i in range(2)]
        for tname, y in (
            ("source_Q_native_key", a[:, :, -1]),
            ("native_Q_native_key", native),
        ):
            for ai, arm in enumerate((*ARMS, "source_Q_native_key")):
                for mi, mode in enumerate(old.MODES):
                    meta = {
                        "cohort": cohort,
                        "authors": len(data),
                        "target": tname,
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
    tables = [selector, *[pd.DataFrame(x) for x in (qr, fr, cr)]]
    baseline_error = 0.0
    for name, table in zip(TABLES, tables, strict=True):
        # Preserve the previous numerical baseline rows exactly up to CSV round-trip.
        baseline = pd.read_csv(prev.OUT / "analysis" / name)
        keys = [c for c in baseline if baseline[c].dtype == object]
        subset = (
            table[table.arm != "response_coordinates_joint"]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        baseline = baseline.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(subset[keys], baseline[keys])
        numeric = [c for c in baseline if c not in keys]
        diff = abs(subset[numeric].to_numpy() - baseline[numeric].to_numpy())
        baseline_error = max(baseline_error, float(np.nanmax(diff)))
        np.testing.assert_allclose(
            subset[numeric], baseline[numeric], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        table.to_csv(dst / name, index=False)
    limit(start, used)
    result = {
        "status": "COMPLETE",
        "cpu_seconds": time.process_time() - start,
        "files": {n: old.sha(dst / n) for n in TABLES},
        "counts": [len(t) for t in tables],
        "teacher_replay_max_error": error,
        "old_baseline_max_error": baseline_error,
    }
    old.write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def audit(out):
    """Verify replay and budgets without converting implementation checks to efficacy."""
    old.verify(out)
    fm = prev.freeze(out, "models")
    a, b = prev.freeze(out, "predictions"), prev.freeze(out, "replay")
    if (
        set(a["files"]) != set(b["files"])
        or len(a["files"]) != 24
        or fm["fits"] != 10
        or fm["source_capacity_accounts"] != 10
    ):
        raise ValueError("Fixed inventory or source-fit count failed")
    for name in a["files"]:
        x, y = [old.arrays(out / n / name) for n in ("predictions", "replay")]
        for k in x:
            np.testing.assert_array_equal(x[k], y[k])
    aa, bb = prev.freeze(out, "analysis"), prev.freeze(out, "replay_analysis")
    if aa["files"] != bb["files"] or aa["counts"] != [16, 80, 320, 40]:
        raise ValueError("Aggregate replay or row counts failed")
    if prev.prior_cpu(out) > CAP or any(f["extra_denials"] for f in (fm, a, b)):
        raise ValueError("Cumulative budget or data boundary failed")
    result = {
        "status": "PASS",
        "fits": 10,
        "source_capacity_accounts": 10,
        "packets": 24,
        "csv_replay": 4,
        "total_cpu_seconds": prev.prior_cpu(out),
        "fd_count": a["fd_count"],
        "fd_max_error": a["fd_max_error"],
        "teacher_replay_max_error": aa["teacher_replay_max_error"],
        "old_baseline_max_error": aa["old_baseline_max_error"],
        "new_neural_calls": 0,
        "human_labels_read": False,
        "native_values_retained": True,
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Use separate processes for prepare, source fit, pure application and evaluation."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "fit", "predict", "evaluate", "audit"),
    )
    p.add_argument("--folder", default="predictions", choices=("predictions", "replay"))
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        if args.stage in ("predict", "evaluate"):
            globals()[args.stage](args.output, args.folder)
        else:
            globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
