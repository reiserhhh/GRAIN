#!/usr/bin/env python3
"""NEXT73: crossed query scalar paths with frozen source anchors and pure evidence."""

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
from grain_research.vector_mechanisms.query_selection_orientation import (
    account,
    crossed,
    path_score,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_query_partition_geometry as previous

old, SOURCE = previous.old, previous.SOURCE
OUT = old.BASE / "logic_query_selection_orientation_v1"
PLAN = ROOT / "docs/GRAIN_QUERY_SELECTION_ORIENTATION_PLAN_V1.md"
CAP = 30
TABLES = (
    "crossed_query_actions.csv",
    "crossed_contrast_actions.csv",
    "query_account.csv",
    "factor_account.csv",
    "contrast_account.csv",
)


def prepare(out):
    """Freeze permitted source4/new8 inputs and exact stage implementation."""
    start = time.process_time()
    m = old.verify(SOURCE)
    old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited canonical NEXT72 required")
    jobs = old.read(SOURCE / "private_jobs.json")
    if len(jobs) != 12 or len({j["key"] for j in jobs}) != 12:
        raise ValueError("Exact twelve paired authors required")
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    inputs = {str((SOURCE / p).relative_to(ROOT)): v for p, v in m["prepared"].items()}
    paths = [
        PLAN,
        Path(__file__),
        previous.OUT / "audit.json",
        SOURCE / "manifest.json",
        ROOT / "tests/test_grain_query_selection_orientation.py",
    ]
    paths += [
        ROOT / "grain_research/vector_mechanisms" / (n + ".py")
        for n in (
            "query_selection_orientation",
            "query_partition_geometry",
            "source_query_selection",
            "fixed_seed_attention",
            "compiled_query_transfer",
            "stage_file_guard",
        )
    ]
    paths += [
        ROOT / "scripts" / (n + ".py")
        for n in (
            "run_grain_query_partition_geometry",
            "run_grain_source_query_selection",
            "run_grain_compiled_query_transfer",
        )
    ]
    inputs.update({str(p.relative_to(ROOT)): old.sha(p) for p in paths})
    cpu = time.process_time() - start
    if cpu > CAP:
        raise TimeoutError("Prepare cap exceeded")
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_SELECTION_ORIENTATION_V1",
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "prepare_cpu_seconds": cpu,
            "fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "current_Q_read": False,
        },
    )
    print({"status": "PREPARED", "cpu_seconds": cpu, "authors": 12}, flush=True)


def predict(out, folder):
    """Freeze crossed paths/operators in a process denied access to native targets."""
    old.verify(out)
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    reads = [SOURCE / "private_source_bank.npz"] + [
        SOURCE / "private_inputs" / (k + ".npz") for j in jobs for k in j["query_keys"]
    ]
    files = [dst / (j["key"] + ".npz") for j in jobs]
    probes = [
        SOURCE / "predictions" / (jobs[0]["key"] + ".npz"),
        old.SOURCE / "predictions" / (jobs[0]["key"] + ".npz"),
    ]
    denied = install(
        reads,
        files + [dst / "freeze.json"],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    start = time.process_time()
    bank = old.arrays(reads[0])
    checks, errors = [], []
    for j, path in zip(jobs, files, strict=True):
        packets = []
        for q, key in zip(old.QUERIES, j["query_keys"], strict=True):
            anchor, members = old.anchors(bank, j["cohort"], j["index"], q)
            e = old.arrays(SOURCE / "private_inputs" / (key + ".npz"))
            packet = crossed(anchor[:2], e)
            packets.append(packet)
            checks.append({"key": j["key"], "query": q, "sources": members})
            if j["cohort"] == "source4" and j["index"] == 0:
                for a in range(2):
                    for b in range(2):
                        for f in range(4):
                            fd = (
                                path_score(anchor[:2], e, a, b, f, 1e-5)
                                - path_score(anchor[:2], e, a, b, f, -1e-5)
                            ) / 2e-5
                            errors.extend(
                                np.abs(fd - packet["actions"][a, b, :, f]).tolist()
                            )
        np.savez_compressed(
            path, **{k: np.stack([p[k] for p in packets]) for k in packets[0]}
        )
        if time.process_time() - start > CAP:
            raise TimeoutError("Prediction cap exceeded")
    if len(errors) != 64 or max(errors) > 1e-7:
        raise ValueError("Frozen scalar derivative check failed")
    r = {
        "status": "COMPLETE",
        "cpu_seconds": time.process_time() - start,
        "checks": checks,
        "files": {p.name: old.sha(p) for p in files},
        "fd_count": len(errors),
        "fd_max_error": max(errors),
        "target_reads": 0,
        "current_Q_reads": 0,
        "fits": 0,
        "neural_calls": 0,
        "denial_probes": len(probes),
        "extra_denials": len(denied) - len(probes),
    }
    old.write(dst / "freeze.json", r)
    print({k: v for k, v in r.items() if k not in ("checks", "files")}, flush=True)


def evaluate(out, folder):
    """Account for all frozen query and contrast swaps; never promote mixed cells."""
    old.verify(out)
    fr = old.read(out / folder / "freeze.json")
    if fr["status"] != "COMPLETE":
        raise ValueError("Frozen predictions required")
    old.verify_hashes(out / folder, fr["files"])
    for root, freeze in (
        (SOURCE, "predictions/freeze.json"),
        (old.SOURCE, "predictions/prediction_freeze.json"),
    ):
        old.verify_hashes(root / "predictions", old.read(root / freeze)["files"])
    start = time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    grouped = {c: [] for c in ("source4", "new8")}
    error = 0.0
    for j in old.read(out / "private_jobs.json"):
        a = old.arrays(out / folder / (j["key"] + ".npz"))["actions"]
        prior = old.arrays(SOURCE / "predictions" / (j["key"] + ".npz"))["actions"][
            :, :2
        ]
        diagonal = np.stack((a[:, 0, 0], a[:, 1, 1]), axis=1)
        error = max(error, float(np.max(np.abs(diagonal - prior))))
        np.testing.assert_allclose(diagonal, prior, atol=1e-11, rtol=1e-10)
        target = old.arrays(old.SOURCE / "predictions" / (j["key"] + ".npz"))[
            "actions"
        ][:, :, 1, :]
        grouped[j["cohort"]].append((a, target))
    query_cells, contrast_cells, qs, fs, cs = [], [], [], [], []
    for cohort, values in grouped.items():
        a = np.stack([p[0] for p in values])
        y = np.stack([p[1] for p in values])
        for m, mode in enumerate(old.MODES):
            meta = {"cohort": cohort, "mode": mode, "authors": len(values)}
            c = (a[:, 0, :, :, m] - a[:, 1, :, :, m]).transpose(1, 2, 0, 3)
            cy = y[:, 0, m] - y[:, 1, m]
            cs.append(meta | account(c, cy))
            for q, query in enumerate(old.QUERIES):
                z, t = a[:, q, :, :, m].transpose(1, 2, 0, 3), y[:, q, m]
                qm = meta | {"query": query}
                qs.append(qm | account(z, t))
                for f in range(4):
                    fs.append(
                        qm
                        | {"factor": f"whole:F{f + 1:02d}"}
                        | account(z[..., f], t[:, f])
                    )
                for i in range(2):
                    for j in range(2):
                        cell = {
                            "selection_anchor": ("own", "other")[i],
                            "direction_anchor": ("own", "other")[j],
                        }
                        query_cells.append(
                            qm | cell | old.fidelity(t.ravel(), z[i, j].ravel())
                        )
                        if q == 0:
                            contrast_cells.append(
                                meta | cell | old.fidelity(cy.ravel(), c[i, j].ravel())
                            )
    tables = (query_cells, contrast_cells, qs, fs, cs)
    for name, rows in zip(TABLES, tables, strict=True):
        pd.DataFrame(rows).to_csv(dst / name, index=False)
    s = {
        "status": "COMPLETE",
        "cpu_seconds": time.process_time() - start,
        "diagonal_max_error": error,
        "counts": [len(rows) for rows in tables],
        "files": {name: old.sha(dst / name) for name in TABLES},
    }
    old.write(dst / "summary.json", s)
    print({k: v for k, v in s.items() if k != "files"}, flush=True)


def audit(out):
    """Exact replay, scalar/diagonal identities, source exclusions and budget."""
    start = time.process_time()
    m = old.verify(out)
    fr = [old.read(out / n / "freeze.json") for n in ("predictions", "replay")]
    for n, f in zip(("predictions", "replay"), fr, strict=True):
        old.verify_hashes(out / n, f["files"])
    if set(fr[0]["files"]) != set(fr[1]["files"]):
        raise ValueError("Packet inventories differ")
    for name in fr[0]["files"]:
        p = [old.arrays(out / n / name) for n in ("predictions", "replay")]
        for k in ("actions", "operator", "anchors"):
            np.testing.assert_array_equal(p[0][k], p[1][k])
    ss = [old.read(out / n / "summary.json") for n in ("analysis", "replay_analysis")]
    for n, s in zip(("analysis", "replay_analysis"), ss, strict=True):
        old.verify_hashes(out / n, s["files"])
    if (
        ss[0]["files"] != ss[1]["files"]
        or set(ss[0]["files"]) != set(TABLES)
        or ss[0]["counts"] != [32, 16, 8, 32, 4]
    ):
        raise ValueError("Aggregate replay/cardinality failed")
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(v["cpu_seconds"] for v in fr + ss)
    )
    if cpu > CAP or len(fr[0]["files"]) != 12:
        raise ValueError("Bounded packet budget failed")
    r = {
        "status": "PASS",
        "packets": 12,
        "csv_replay": 5,
        "total_cpu_seconds": cpu,
        "diagonal_max_error": ss[0]["diagonal_max_error"],
        "fd_count": fr[0]["fd_count"],
        "fd_max_error": fr[0]["fd_max_error"],
        "fits": 0,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    old.write(out / "audit.json", r)
    print(r, flush=True)


def main():
    """Run target-separated stages; use a fresh output directory for repetitions."""
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
