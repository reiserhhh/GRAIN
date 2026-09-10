#!/usr/bin/env python3
"""NEXT51: frozen selection vectors, source suffix readers, then endpoint evaluation."""

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
from grain_research.vector_mechanisms.compiled_query_transfer import QUERIES, fidelity
from grain_research.vector_mechanisms.selection_endpoint import (
    account,
    key_vectors,
    source_reader,
)
from grain_research.vector_mechanisms.source_query_selection import (
    ARMS,
    anchors,
    source_query,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_source_query_selection import (
    BASE,
    SOURCE,
    arrays,
    read,
    sha,
    verify,
    verify_hashes,
    write,
)

ANCHOR = BASE / "logic_source_query_selection_v1"
NATIVE = BASE / "logic_fixed_seed_attention_v1"
PLAN = ROOT / "docs/GRAIN_SELECTION_ENDPOINT_PLAN_V1.md"


def prepare(out):
    """Freeze permitted numerical inputs and exact prior native boundary paths."""
    am = verify(ANCHOR)
    if read(ANCHOR / "analysis/summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete NEXT50 required")
    jobs = read(ANCHOR / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Exact twelve pairs required")
    nm = read(NATIVE / "manifest.json")
    out.mkdir(exist_ok=False)
    (out / "private_evidence").mkdir()
    files, inputs = (
        [],
        [PLAN, Path(__file__), NATIVE / "manifest.json", ANCHOR / "manifest.json"],
    )
    start = time.monotonic()
    for j in jobs:
        for key in j["query_keys"]:
            p = ANCHOR / "private_inputs" / (key + ".npz")
            e = arrays(p)
            dst = out / "private_evidence" / (key + ".npz")
            np.savez(dst, **{k: v for k, v in e.items() if k != "g"})
            files.append(dst)
            inputs.extend([p, SOURCE / "private_capture" / (key + ".npz")])
    wpath = SOURCE / "private_weights.npz"
    np.savez(out / "private_output.npz", o=arrays(wpath)["o"])
    bankpath = ANCHOR / "private_source_bank.npz"
    np.savez(out / "private_source_bank.npz", **arrays(bankpath))
    write(out / "private_jobs.json", jobs)
    files.extend(
        [
            out / "private_output.npz",
            out / "private_source_bank.npz",
            out / "private_jobs.json",
        ]
    )
    inputs.extend([wpath, bankpath, SOURCE / "capture_freeze.json"])
    for stem in (
        "selection_endpoint",
        "source_query_selection",
        "fixed_seed_attention",
        "compiled_query_transfer",
        "stage_file_guard",
        "context_carrier",
        "trace",
    ):
        inputs.append(ROOT / "grain_research/vector_mechanisms" / (stem + ".py"))
    inputs.extend(
        [
            ROOT / "tests/test_grain_selection_endpoint.py",
            ROOT / "scripts/run_grain_complete_transport_jvp.py",
        ]
    )
    # Preserve all upstream files verified by NEXT50, including its r2 amendment.
    lineage = dict(am["inputs"])
    lineage.update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SELECTION_ENDPOINT_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "block": 22,
            "alphabet": nm["alphabet"],
            "c": nm["c"],
            "model_sha256": nm["model_sha256"],
            "max_native_seconds": 180,
            "max_reverse_pairs": 24,
            "max_jvps": 2,
            "max_fd": 4,
            "max_numeric_seconds": 60,
            "fd_epsilon": 0.05,
            "jvp_atol": 1e-5,
            "jvp_rtol": 3e-4,
            "fd_atol": 2e-4,
            "fd_rtol": 0.03,
            "inputs": lineage,
            "prepared": {str(p.relative_to(out)): sha(p) for p in files},
            "human_labels_read": False,
            "fits": 0,
            "prepare_seconds": time.monotonic() - start,
        },
    )
    print({"status": "PREPARED", "pairs": len(jobs)}, flush=True)


def frozen(out, folder):
    """Verify a completed stage and every exact output filename."""
    f = read(out / folder / "freeze.json")
    if f["status"] != "COMPLETE":
        raise ValueError("Stage incomplete: " + folder)
    verify_hashes(out / folder, f["files"])
    return f


def vectors(out, name):
    """Freeze full candidate increments using source Q and pure evidence only."""
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    paths = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    reads = [out / "private_source_bank.npz", out / "private_output.npz"] + [
        out / "private_evidence" / p.name for p in paths
    ]
    probes = [
        SOURCE / "private_capture/new8_a00_neuroticism.npz",
        SOURCE / "predictions/new8_a00.npz",
        out / "capture_new/new8_a00_neuroticism.npz",
        ANCHOR / "analysis/private_actions.csv",
    ]
    denied = install(
        reads,
        paths + [dst / "freeze.json"],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    start = time.monotonic()
    bank, wo = arrays(reads[0]), arrays(reads[1])["o"]
    for j in jobs:
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            a, _ = anchors(bank, j["cohort"], j["index"], q)
            e = arrays(out / "private_evidence" / (key + ".npz"))
            np.savez_compressed(dst / (key + ".npz"), vectors=key_vectors(a, e, wo))
            if time.monotonic() - start > m["max_numeric_seconds"]:
                raise TimeoutError("Vector construction cap exceeded")
    result = {
        "status": "COMPLETE",
        "seconds": time.monotonic() - start,
        "native_calls": 0,
        "fits": 0,
        "current_Q_reads": 0,
        "target_reads": 0,
        "denial_probes": 4,
        "extra_denials": len(denied) - 4,
        "files": {p.name: sha(p) for p in paths},
    }
    write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def capture(out, cohort):
    """Run a bounded real suffix; sources precede the frozen scalar construction."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from grain_research.vector_mechanisms.selection_endpoint import (
        endpoint,
        endpoint_gradients,
    )
    from scripts.run_grain_complete_transport_jvp import load_observer

    m = verify(out)
    frozen(out, "vectors")
    prior = {"native_seconds": 0.0, "reverse_pairs": 0, "jvps": 0, "fd_forwards": 0}
    if cohort == "new8":
        prior = frozen(out, "capture_source")
        frozen(out, "predictions")
    folder = "capture_source" if cohort == "source4" else "capture_new"
    dst = out / folder
    dst.mkdir(exist_ok=False)
    jobs = [j for j in read(out / "private_jobs.json") if j["cohort"] == cohort]
    ob, loading = load_observer(m)
    torch.set_grad_enabled(False)
    torch.func.jvp(
        lambda x: x * x, (torch.ones(2, device="mps"),), (torch.ones(2, device="mps"),)
    )
    ids = torch.tensor(m["alphabet"]["token_ids"], device="mps")
    c = torch.tensor(m["c"], device="mps", dtype=torch.float32)
    counters = {
        k: prior[k] for k in ("native_seconds", "reverse_pairs", "jvps", "fd_forwards")
    }
    paths, checks = [], []
    started = time.monotonic()

    def timed(fn, kind):
        """Count all real suffix calls against the same cumulative compute limit."""
        limits = {
            "reverse_pairs": "max_reverse_pairs",
            "jvps": "max_jvps",
            "fd_forwards": "max_fd",
        }
        if (
            counters["native_seconds"] >= m["max_native_seconds"]
            or counters[kind] >= m[limits[kind]]
        ):
            raise TimeoutError("Native compute boundary reached")
        torch.mps.synchronize()
        start = time.monotonic()
        counters[kind] += 1
        try:
            return fn()
        finally:
            torch.mps.synchronize()
            counters["native_seconds"] += time.monotonic() - start

    def status(kind):
        """Keep a terminal/partial native record even when parity or budget fails."""
        r = {
            "status": kind,
            **counters,
            "loading_seconds": loading,
            "wall_seconds": time.monotonic() - started,
            "fits": 0,
            "human_labels_read": False,
            "checks": checks.copy(),
            "files": {p.name: sha(p) for p in paths},
        }
        write(dst / "status.json", r)
        print({k: v for k, v in r.items() if k not in ("checks", "files")}, flush=True)
        return r

    try:
        for j in jobs:
            for q, key in zip(QUERIES, j["query_keys"], strict=True):
                cap = arrays(SOURCE / "private_capture" / (key + ".npz"))
                state = torch.tensor(cap["x"][None], device="mps")
                kw = cached_suffix_kwargs(ob.model, state)
                for actual, name in zip(
                    kw["position_embeddings"], ("cos", "sin"), strict=True
                ):
                    np.testing.assert_array_equal(actual.cpu().numpy()[0], cap[name])
                np.testing.assert_array_equal(
                    kw["attention_mask"][0, 0, -1].cpu().numpy(), cap["mask"]
                )
                value, g, r, post = timed(
                    lambda state=state, kw=kw: endpoint_gradients(
                        ob.model, state, kw, ids, c
                    ),
                    "reverse_pairs",
                )
                grad = g.cpu().numpy()[0].astype(float)
                tau = np.einsum("th,fth->f", grad, cap["d"])
                p = dst / (key + ".npz")
                np.savez_compressed(
                    p, r=r.cpu().numpy()[0], tau=tau, score=value.cpu().numpy()
                )
                paths.append(p)
                if j["index"] == 0 and q == QUERIES[0]:
                    delta = torch.tensor(cap["d"][0:1], device="mps")
                    _, deriv = timed(
                        lambda state=state, kw=kw, delta=delta: torch.func.jvp(
                            lambda h: endpoint(ob.model, h, kw, ids, c),
                            (state,),
                            (delta,),
                        ),
                        "jvps",
                    )
                    tangent = float(deriv.cpu())
                    np.testing.assert_allclose(
                        tangent, tau[0], rtol=m["jvp_rtol"], atol=m["jvp_atol"]
                    )
                    u = arrays(out / "vectors" / (key + ".npz"))["vectors"][0, 0]
                    direction = torch.zeros_like(post)
                    direction[:, -1] = torch.tensor(u, device="mps", dtype=post.dtype)
                    eps = m["fd_epsilon"]
                    plus = timed(
                        lambda post=post, eps=eps, direction=direction, kw=kw: endpoint(
                            ob.model, post + eps * direction, kw, ids, c, post=True
                        ),
                        "fd_forwards",
                    )
                    minus = timed(
                        lambda post=post, eps=eps, direction=direction, kw=kw: endpoint(
                            ob.model, post - eps * direction, kw, ids, c, post=True
                        ),
                        "fd_forwards",
                    )
                    fd = float(((plus - minus) / (2 * eps)).cpu())
                    dot = float(r.cpu().numpy()[0].astype(float) @ u)
                    checks.append(
                        {
                            "unit": key,
                            "jvp": tangent,
                            "reverse": float(tau[0]),
                            "fd": fd,
                            "dot": dot,
                        }
                    )
                    np.testing.assert_allclose(
                        fd, dot, rtol=m["fd_rtol"], atol=m["fd_atol"]
                    )
                del g, post, grad, state, r
                status("RUNNING")
        if counters["native_seconds"] > m["max_native_seconds"]:
            raise TimeoutError("Native cap exceeded after final call")
        result = status("COMPLETE")
        write(dst / "freeze.json", result)
    except TimeoutError:
        status("PAUSED_RESOURCE_BOUNDARY")
    except Exception:
        status("ERROR")
        raise


def predict(out, name, vector_name):
    """Use only frozen source gradients and candidate vectors for all scalar outputs."""
    m = verify(out)
    frozen(out, vector_name)
    sf = frozen(out, "capture_source")
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    sources = [
        (j, q, k)
        for j in jobs
        if j["cohort"] == "source4"
        for q, k in zip(QUERIES, j["query_keys"], strict=True)
    ]
    reads = [out / "capture_source" / (k + ".npz") for _, _, k in sources]
    reads += [out / vector_name / (k + ".npz") for j in jobs for k in j["query_keys"]]
    paths = [dst / (j["key"] + ".npz") for j in jobs]
    probes = [
        out / "capture_new/new8_a00_neuroticism.npz",
        SOURCE / "predictions/new8_a00.npz",
        SOURCE / "private_capture/new8_a00_neuroticism.npz",
        ANCHOR / "analysis/private_actions.csv",
    ]
    denied = install(
        reads,
        paths + [dst / "freeze.json"],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    start = time.monotonic()
    bank = {
        (j["index"], q): arrays(out / "capture_source" / (key + ".npz"))["r"]
        for j, q, key in sources
    }
    checks = []
    for j, p in zip(jobs, paths, strict=True):
        values, rs = [], []
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            r, members = source_reader(bank, j["cohort"], j["index"], q)
            u = arrays(out / vector_name / (key + ".npz"))["vectors"]
            values.append(u @ r)
            rs.append(r)
            checks.append({"author_key": j["key"], "query_name": q, "sources": members})
        a = np.stack(values)
        np.savez_compressed(p, actions=a, contrasts=a[0] - a[1], readers=np.stack(rs))
    seconds = time.monotonic() - start
    if seconds > m["max_numeric_seconds"]:
        raise TimeoutError("Numerical prediction cap exceeded")
    result = {
        "status": "COMPLETE",
        "seconds": seconds,
        "source_capture_seconds": sf["native_seconds"],
        "native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "current_gradient_reads": 0,
        "denial_probes": 4,
        "extra_denials": len(denied) - 4,
        "checks": checks,
        "files": {p.name: sha(p) for p in paths},
    }
    write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k not in ("checks", "files")}, flush=True)


def evaluate(out, name, prediction_name, vector_name):
    """Open native current targets only after source-only predictions are frozen."""
    verify(out)
    for stage in (prediction_name, vector_name, "capture_source", "capture_new"):
        frozen(out, stage)
    dst = out / name
    dst.mkdir(exist_ok=False)
    jobs = read(out / "private_jobs.json")
    w = arrays(SOURCE / "private_weights.npz")
    rows = []
    for j in jobs:
        p = arrays(out / prediction_name / (j["key"] + ".npz"))
        for qi, (q, key) in enumerate(zip(QUERIES, j["query_keys"], strict=True)):
            cap = arrays(SOURCE / "private_capture" / (key + ".npz"))
            current = source_query(cap["x"][-1], w)
            e = arrays(out / "private_evidence" / (key + ".npz"))
            actual = key_vectors(current[None], e, w["o"])[0]
            native = arrays(
                out
                / ("capture_source" if j["cohort"] == "source4" else "capture_new")
                / (key + ".npz")
            )
            uhat = arrays(out / vector_name / (key + ".npz"))["vectors"]
            kappa = actual @ native["r"]
            result = account(kappa, uhat @ native["r"], p["actions"][qi], native["tau"])
            # The direct vector formula independently checks the reader-error term.
            np.testing.assert_allclose(
                result["reader_error"],
                uhat @ (p["readers"][qi] - native["r"]),
                atol=1e-12,
            )
            for a, arm in enumerate(ARMS):
                for f in range(4):
                    rows.append(
                        {
                            "author_key": j["key"],
                            "cohort": j["cohort"],
                            "query_name": q,
                            "arm": arm,
                            "factor": f"whole:F{f + 1:02d}",
                            **{k: float(v[a, f]) for k, v in result.items()},
                        }
                    )
    df = pd.DataFrame(rows)
    if (
        len(df) != 288
        or df.author_key.nunique() != 12
        or df.duplicated(["author_key", "query_name", "arm", "factor"]).any()
    ):
        raise ValueError("Exact action/author inventory mismatch")
    names = list(account(0, 0, 0, 0))
    idx = ["author_key", "cohort", "arm", "factor"]
    cf = (
        df[df.query_name == QUERIES[0]].set_index(idx)[names]
        - df[df.query_name == QUERIES[1]].set_index(idx)[names]
    )
    cf = cf.reset_index()
    if len(cf) != 144 or cf[names].isna().any().any():
        raise ValueError("Exact paired contrast inventory mismatch")
    df.to_csv(dst / "private_actions.csv", index=False)
    cf.to_csv(dst / "private_contrasts.csv", index=False)
    files = ["private_actions.csv", "private_contrasts.csv"]
    summaries = []
    for frame, groups, fn in (
        (df, ["cohort", "arm", "query_name"], "query_summary.csv"),
        (df, ["cohort", "arm", "query_name", "factor"], "factor_summary.csv"),
        (cf, ["cohort", "arm"], "contrast_summary.csv"),
        (cf, ["cohort", "arm", "factor"], "contrast_factors.csv"),
    ):
        values = []
        for key, g in frame.groupby(groups, sort=True):
            meta = dict(zip(groups, key, strict=True)) | {
                "authors": g.author_key.nunique()
            }
            y = g.actual.to_numpy()
            for prediction in ("transported", "compiled"):
                values.append(
                    meta
                    | {"readout": prediction}
                    | fidelity(y, g[prediction].to_numpy())
                )
        tab = pd.DataFrame(values)
        tab.to_csv(dst / fn, index=False)
        summaries.append(tab)
        files.append(fn)
    budget = frozen(out, "capture_new")
    qtab = summaries[0]
    primary = qtab[
        (qtab.cohort == "new8") & (qtab.arm == ARMS[0]) & (qtab.readout == "compiled")
    ]
    erows = []
    for level, frame, groups in (
        ("action", df, ["cohort", "arm", "query_name"]),
        ("contrast", cf, ["cohort", "arm"]),
    ):
        for key, g in frame.groupby(groups, sort=True):
            a, b = g.selection_error.to_numpy(), g.reader_error.to_numpy()
            total = g.total_error.to_numpy()
            np.testing.assert_allclose(total, a + b, atol=1e-12)
            den = float(np.mean(g.tau**2))
            erows.append(
                dict(zip(groups, key, strict=True))
                | {
                    "level": level,
                    "authors": g.author_key.nunique(),
                    "actions": len(g),
                    "selection_mse": float(np.mean(a * a)),
                    "reader_mse": float(np.mean(b * b)),
                    "cross_term": float(2 * np.mean(a * b)),
                    "total_mse": float(np.mean(total * total)),
                    "native_tau_rms": float(np.sqrt(den)),
                    "key_rms": float(np.sqrt(np.mean(g.actual**2))),
                    "key_projection_on_tau": float(np.mean(g.actual * g.tau) / den),
                    "physical_patch_error_over_omission": float(
                        np.sum(a * a) / np.sum(g.actual**2)
                    ),
                    "compiled_swap_error_over_omission": float(
                        np.sum(total * total) / np.sum(g.actual**2)
                    ),
                    "physical_patch_r": float(
                        np.corrcoef(g.tau, g.physical_patch)[0, 1]
                    ),
                    "compiled_swap_r": float(np.corrcoef(g.tau, g.compiled_swap)[0, 1]),
                }
            )
    pd.DataFrame(erows).to_csv(dst / "error_accounting.csv", index=False)
    files.append("error_accounting.csv")
    result = {
        "status": "COMPLETE",
        "actions": len(df),
        "contrasts": len(cf),
        "primary_retained": bool(
            len(primary) == 2 and (primary.action_mse_over_zero < 1).all()
        ),
        "native_seconds": budget["native_seconds"],
        "reverse_pairs": budget["reverse_pairs"],
        "jvps": budget["jvps"],
        "fd_forwards": budget["fd_forwards"],
        "fits": 0,
        "human_labels_read": False,
        "files": {n: sha(dst / n) for n in files},
    }
    write(dst / "summary.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def main():
    """Run each freeze/capture stage as a separate bounded process."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=(
            "prepare",
            "vectors",
            "capture-source",
            "predict",
            "capture-new",
            "evaluate",
        ),
    )
    p.add_argument("--output", type=Path, default=BASE / "logic_selection_endpoint_v1")
    p.add_argument("--name")
    p.add_argument("--vectors", default="vectors")
    p.add_argument("--predictions", default="predictions")
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(a.output)
        elif a.stage == "vectors":
            vectors(a.output, a.name or "vectors")
        elif a.stage == "predict":
            predict(a.output, a.name or "predictions", a.vectors)
        elif a.stage.startswith("capture-"):
            capture(a.output, "source4" if a.stage == "capture-source" else "new8")
        else:
            evaluate(a.output, a.name or "analysis", a.predictions, a.vectors)


if __name__ == "__main__":
    main()
