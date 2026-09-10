#!/usr/bin/env python3
"""NEXT59: cached native action in directions erased by a frozen pooled input."""

from __future__ import annotations

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
from grain_research.vector_mechanisms.pooled_feature_null_response import (
    PARTS,
    response_account,
    responses,
)
from scripts.run_grain_complete_function_extraction import (
    BASE,
    ENDPOINT,
    QUERIES,
    arrays,
    basis_path,
    body_directions,
    read,
    ref,
    selected,
    sha,
    verify,
    verify_hashes,
    write,
)
from scripts.run_grain_fixed_seed_attention import context
from scripts.run_grain_reader_new_author_transport import packet_arrays

OLD = BASE / "logic_complete_function_extraction_v1"
NATIVE = BASE / "logic_fixed_seed_attention_v1"
PLAN = ROOT / "docs/GRAIN_POOLED_FEATURE_NULL_RESPONSE_PLAN_V1.md"
FACTORS = [f"whole:F{i:02d}" for i in range(1, 5)]
FIELDS = ["native", *PARTS]


def target_path(job, key):
    """Resolve old full-function targets without confusing endpoint gradient layers."""
    folder = "capture_source" if job["cohort"] == "source4" else "capture_new"
    return ENDPOINT / folder / (key + ".npz")


def prepare(out):
    """Freeze all bindings and current code before inspecting decomposed effects."""
    started = time.process_time()
    m = verify(OLD)
    verify_hashes(OLD, m["prepared"])
    verify_hashes(ROOT, m["evaluation"])
    nm, em = read(NATIVE / "manifest.json"), read(ENDPOINT / "manifest.json")
    verify_hashes(ROOT, nm["inputs"])
    if sha(NATIVE / "private_jobs.json") != nm["jobs_sha256"]:
        raise ValueError("Original native job binding changed")
    for name in ("alphabet", "c", "model_sha256"):
        if nm[name] != em[name]:
            raise ValueError("Native and endpoint target disagree: " + name)
    jobs = read(OLD / "private_jobs.json")
    native_jobs = {j["key"]: j for j in read(NATIVE / "private_jobs.json")}
    if len(jobs) != 12 or len(native_jobs) != 24:
        raise ValueError("Exact12 authors/24 native contexts required")
    files = dict(m["inputs"]) | dict(m["evaluation"]) | dict(nm["inputs"])
    files.update(
        {str((OLD / n).relative_to(ROOT)): h for n, h in m["prepared"].items()}
    )
    binding = []
    for job in jobs:
        for q, key in zip(QUERIES, job["query_keys"], strict=True):
            nj = native_jobs[key]
            if (nj["cohort"], nj["index"], nj["query"]) != (
                job["cohort"],
                job["index"],
                q,
            ):
                raise ValueError("Native material/query join differs")
            state, _, grain, pos = context(nj)
            inp = arrays(OLD / "private_inputs" / (key + ".npz"))
            for name, expected in (
                ("state", state),
                ("positions", pos),
                ("grain", grain),
            ):
                np.testing.assert_array_equal(inp[name], expected)
            packet = NATIVE / "private_packets" / (key + ".json")
            rec, a = packet_arrays(packet)
            if rec["key"] != key or a["body_gradient"].shape != (
                len(pos),
                state.shape[1],
            ):
                raise ValueError("Original H16 gradient/address binding differs")
            target = selected(target_path(job, key), ("score", "tau"))
            ds = body_directions(inp)[:, pos]
            tau = np.einsum("th,fth->f", a["body_gradient"].astype(float), ds)
            np.testing.assert_allclose(tau, target["tau"], atol=1e-5, rtol=3e-4)
            np.testing.assert_allclose(
                rec["contrast"], target["score"], atol=1e-5, rtol=3e-4
            )
            b = selected(basis_path(OLD, ref(job)), ("basis",))["basis"]
            orth = float(np.max(np.abs(b.T @ b - np.eye(32))))
            if orth > 1e-10:
                raise ValueError("Frozen FP64 basis precision is insufficient")
            binding.append(
                {
                    "query_key": key,
                    "tau_error": float(np.max(np.abs(tau - target["tau"]))),
                    "score_error": abs(float(rec["contrast"] - target["score"])),
                    "orthogonality_error": orth,
                    "body_tokens": len(pos),
                }
            )
            for p in (packet, packet.with_suffix(".npz")):
                files[str(p.relative_to(ROOT))] = sha(p)
    extra = [
        OLD / "manifest.json",
        NATIVE / "manifest.json",
        NATIVE / "private_jobs.json",
        ENDPOINT / "manifest.json",
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/pooled_feature_null_response.py",
        ROOT / "tests/test_grain_pooled_feature_null_response.py",
    ]
    files.update({str(p.relative_to(ROOT)): sha(p) for p in extra})
    out.mkdir(exist_ok=False)
    write(out / "private_bindings.json", binding)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_POOLED_FEATURE_NULL_RESPONSE_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "inputs": files,
            "binding_sha256": sha(out / "private_bindings.json"),
            "prepare_cpu_seconds": time.process_time() - started,
            "max_cpu_seconds": 60,
            "fits": 0,
            "neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print(
        {
            "status": "PREPARED",
            "contexts": len(binding),
            "tau_max_error": max(x["tau_error"] for x in binding),
            "prepare_cpu_seconds": time.process_time() - started,
        }
    )


def aggregates(frame):
    """Keep every factor and pooled query, treating authors as the actual units."""
    rows = []
    for keys in (["cohort", "query_name"], ["cohort", "query_name", "factor"]):
        for key, g in frame.groupby(keys, sort=True):
            meta = dict(zip(keys, key, strict=True))
            meta.setdefault("factor", "all4")
            rows.append(
                meta | {"authors": g.author_key.nunique()} | response_account(g[FIELDS])
            )
    return pd.DataFrame(rows)


def analyze(out, folder):
    """Compute one fixed native-response decomposition on already permitted caches."""
    start = time.process_time()
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    if sha(out / "private_bindings.json") != m["binding_sha256"]:
        raise ValueError("Preflight bindings changed")
    jobs = read(OLD / "private_jobs.json")
    rows, geometry = [], []
    checks = {
        "response_sum_error": 0.0,
        "reconstruction_relative": 0.0,
        "feature_null_relative": 0.0,
        "energy_relative": 0.0,
        "orthogonal_inner_relative": 0.0,
    }
    for job in jobs:
        u = selected(basis_path(OLD, ref(job)), ("basis",))["basis"]
        for q, key in zip(QUERIES, job["query_keys"], strict=True):
            inp = arrays(OLD / "private_inputs" / (key + ".npz"))
            g = selected(
                NATIVE / "private_packets" / (key + ".npz"), ("body_gradient",)
            )["body_gradient"]
            for f, d in zip(
                FACTORS, body_directions(inp)[:, inp["positions"]], strict=True
            ):
                value, geom = responses(g, d, u)
                meta = {
                    "author_key": job["key"],
                    "cohort": job["cohort"],
                    "query_name": q,
                    "factor": f,
                }
                rows.append(meta | value)
                geometry.append(meta | geom)
                norm = max(1.0, float(np.linalg.norm(d)))
                en = max(1.0, geom["total_energy"])
                checks["response_sum_error"] = max(
                    checks["response_sum_error"], value["sum_error"]
                )
                checks["reconstruction_relative"] = max(
                    checks["reconstruction_relative"],
                    geom["reconstruction_error"] / norm,
                )
                checks["feature_null_relative"] = max(
                    checks["feature_null_relative"],
                    max(geom[n + "_feature_error"] for n in PARTS)
                    / max(1.0, geom["feature_derivative_norm"]),
                )
                checks["energy_relative"] = max(
                    checks["energy_relative"], geom["energy_identity_error"] / en
                )
                checks["orthogonal_inner_relative"] = max(
                    checks["orthogonal_inner_relative"],
                    max(abs(v) for k, v in geom.items() if k.endswith("_inner")) / en,
                )
    if len(rows) != 96 or max(checks.values()) > 1e-8:
        raise ValueError("Frozen96 actions or numerical identities failed")
    raw, geo = pd.DataFrame(rows), pd.DataFrame(geometry)
    groups = ["cohort", "query_name", "factor"]
    energy = (
        geo.groupby(groups, sort=True)[
            ["total_energy", *[n + "_energy" for n in PARTS]]
        ]
        .sum()
        .reset_index()
    )
    for n in PARTS:
        energy[n + "_fraction"] = energy[n + "_energy"] / energy.total_energy.replace(
            0, np.nan
        )
    keys = ["cohort", "author_key", "factor"]
    a, b = [
        raw[raw.query_name == q].set_index(keys).sort_index()[FIELDS] for q in QUERIES
    ]
    if not a.index.equals(b.index):
        raise ValueError("Query contrast author/factor pairing differs")
    contrast = (a - b).reset_index()
    contrast["query_name"] = "neuroticism-minus-confirmation_seeking"
    tables = {
        "private_responses": raw,
        "private_geometry": geo,
        "responses": aggregates(raw),
        "geometry": energy,
        "query_contrasts": aggregates(contrast),
    }
    dst = out / folder
    dst.mkdir(exist_ok=False)
    for n, df in tables.items():
        df.to_csv(dst / (n + ".csv"), index=False)
    cpu = time.process_time() - start
    if cpu + m["prepare_cpu_seconds"] > m["max_cpu_seconds"]:
        raise TimeoutError("Finite cache budget exceeded")
    write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "checks": checks,
            "unique_authors": 12,
            "contexts": 24,
            "actions": 96,
            "fits": 0,
            "neural_calls": 0,
            "human_labels_read": False,
            "files": {n + ".csv": sha(dst / (n + ".csv")) for n in tables},
        },
    )
    print({"status": "COMPLETE", "cpu_seconds": cpu, "checks": checks})


def audit(out):
    """Verify exact replay plus a fixed-unit independent least-squares projection."""
    start = time.process_time()
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    a, b = [read(out / f / "summary.json") for f in ("analysis", "replay")]
    if (
        a["status"] != b["status"]
        or a["status"] != "COMPLETE"
        or a["files"] != b["files"]
    ):
        raise ValueError("Exact complete CSV replay required")
    for f, s in (("analysis", a), ("replay", b)):
        verify_hashes(out / f, s["files"])
    j = read(OLD / "private_jobs.json")[0]
    key = j["query_keys"][0]
    inp = arrays(OLD / "private_inputs" / (key + ".npz"))
    u = selected(basis_path(OLD, ref(j)), ("basis",))["basis"]
    g = selected(NATIVE / "private_packets" / (key + ".npz"), ("body_gradient",))[
        "body_gradient"
    ].astype(float)
    raw = pd.read_csv(out / "analysis/private_responses.csv")
    raw = raw[(raw.author_key == j["key"]) & (raw.query_name == QUERIES[0])].set_index(
        "factor"
    )
    err = 0.0
    for f, d in zip(FACTORS, body_directions(inp)[:, inp["positions"]], strict=True):
        mean = d.mean(axis=0)
        coeff = np.linalg.lstsq(u, mean, rcond=None)[0]
        vp = u @ coeff
        expected = [
            float(g.sum(axis=0) @ vp),
            float(np.vdot(g, d - mean)),
            float(g.sum(axis=0) @ (mean - vp)),
        ]
        err = max(
            err,
            float(
                np.max(
                    np.abs(
                        np.asarray(expected) - raw.loc[f, list(PARTS)].to_numpy(float)
                    )
                )
            ),
        )
    if err > 1e-9:
        raise ValueError("Independent projection/contraction differs")
    cpu = time.process_time() - start
    total = m["prepare_cpu_seconds"] + a["cpu_seconds"] + b["cpu_seconds"] + cpu
    if total > m["max_cpu_seconds"]:
        raise TimeoutError("Total60s cache budget exceeded")
    result = {
        "status": "PASS",
        "csvs_replayed": len(a["files"]),
        "independent_response_error": err,
        "cpu_seconds": cpu,
        "total_measured_cpu_seconds": total,
        "fits": 0,
        "neural_calls": 0,
        "human_labels_read": False,
    }
    write(out / "replay_audit.json", result)
    print(result)


def main():
    """Separate frozen cache preflight, one experiment/replay and independent audit."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "analyze", "audit"), required=True)
    p.add_argument("--folder", choices=("analysis", "replay"), default="analysis")
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_pooled_feature_null_response_v1"
    )
    args = p.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == "analyze":
            analyze(args.output, args.folder)
        else:
            globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
