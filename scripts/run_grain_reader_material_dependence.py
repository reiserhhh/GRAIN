#!/usr/bin/env python3
"""NEXT81: fixed-last source reading and four GRAIN BODY-to-reader interventions."""

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
from grain_research.vector_mechanisms import reader_material_dependence as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_reader_motion as previous

old, ep, prior = previous.old, previous.ep, previous.prior
OUT = old.BASE / "logic_reader_material_dependence_v1"
PLAN = ROOT / "docs/GRAIN_READER_MATERIAL_DEPENDENCE_PLAN_V1.md"
ETA, CAP = 0.01, 90
FIELDS = (*old.QUERIES, "N_minus_confirmation")
TABLES = (
    "reader_dispersion.csv",
    "clamp_change.csv",
    "body_reader_slopes.csv",
    "body_reader_projections.csv",
    "clamp_reader_projections.csv",
    "input_checks.csv",
)
COUNTS = [6, 3, 12, 260, 130, 2]


def prepare(out):
    """Pin the source-only intervention before exposing any new reader results."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT80 required")
    jobs = [
        j
        for j in old.read(previous.OUT / "private_jobs.json")
        if j["cohort"] == "source4"
    ]
    if [j["index"] for j in jobs] != list(range(4)):
        raise ValueError("Frozen ordered four-source panel required")
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    lasts = np.stack(
        [
            np.stack(
                [
                    old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))["state"][-1]
                    for k in j["query_keys"]
                ]
            )
            for j in jobs
        ]
    )
    np.savez(
        out / "private_reference.npz",
        last=lasts.astype(float).mean(axis=0).astype(lasts.dtype),
    )
    inputs = dict(m["inputs"])
    for folder in ("capture", "models"):
        frozen = ep.freeze(previous.OUT, folder)
        inputs.update(
            {
                str((previous.OUT / folder / n).relative_to(ROOT)): h
                for n, h in frozen["files"].items()
            }
        )
    for p in (
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        previous.OUT / "audit.json",
        ROOT / "tests/test_grain_reader_material_dependence.py",
    ):
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_READER_MATERIAL_DEPENDENCE_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {
                n: old.sha(out / n)
                for n in ("private_jobs.json", "private_reference.npz")
            },
            "prepare_cpu_seconds": time.process_time() - start,
            **{k: m[k] for k in ("model_sha256", "alphabet", "c")},
            "max_calls": 78,
            "max_native_seconds": 120,
            "max_loading_seconds": 300,
            "max_cpu_seconds": CAP,
            "eta": ETA,
            "human_labels_read": False,
            "fit_count": 0,
            "authors": 4,
        },
    )
    print({"status": "PREPARED", "authors": 4}, flush=True)


def capture(out):
    """Bound native source interventions and preserve progress/cost on any failure."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from scripts.run_grain_complete_transport_jvp import load_observer

    m = old.verify(out)
    jobs = old.read(out / "private_jobs.json")
    last = old.arrays(out / "private_reference.npz")["last"]
    dst = out / "capture"
    dst.mkdir(exist_ok=False)
    sp = out / "native_status.json"
    stats = {
        "status": "LOADING",
        "calls": 0,
        "native_seconds": 0.0,
        "loading_seconds": 0.0,
        "completed": 0,
        "human_labels_read": False,
        "recipient_reads": 0,
    }
    old.write(sp, stats)

    def alarm(_sig, _frame):
        raise TimeoutError("Frozen native budget exceeded")

    handler = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(m["max_loading_seconds"])
    tload = time.monotonic()
    try:
        ob, loading = load_observer(m)
    except BaseException:
        stats.update(status="FAILED_LOADING", loading_seconds=time.monotonic() - tload)
        old.write(sp, stats)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
    stats["loading_seconds"] = loading
    torch.set_grad_enabled(False)
    ids = torch.tensor(m["alphabet"]["token_ids"], device="mps")
    contrast = torch.tensor(m["c"], device="mps", dtype=torch.float32)
    keys = [k for j in jobs for k in j["query_keys"]]
    reads = [ep.STATE / "private_inputs" / (k + ".npz") for k in keys]
    reads += [previous.OUT / "capture" / ("parity_" + k + ".npz") for k in keys]
    names = [f"a{i:02d}.npz" for i in range(4)] + [
        "half.npz",
        "checks.json",
        "parity.npz",
    ]
    probes = [
        ep.STATE / "private_inputs/new8_a00_neuroticism.npz",
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")] + [sp],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    paths = []
    checks = {
        "parity_relative": [],
        "parity_max": [],
        "halfstep_relative": None,
        "clamp_last_checks": 0,
        "perturbation_last_checks": 0,
        "paired_body_equal_checks": 0,
    }

    def status(kind):
        stats.update(
            status=kind,
            files={p.name: old.sha(p) for p in paths},
            extra_denials=len(denied) - len(probes),
        )
        old.write(sp, stats)
        print({k: v for k, v in stats.items() if k != "files"}, flush=True)

    def timed(data):
        if (
            stats["calls"] >= m["max_calls"]
            or stats["native_seconds"] >= m["max_native_seconds"]
        ):
            raise TimeoutError("Native call/time cap reached")
        x = torch.tensor(data["state"][None], device="mps")
        kw = cached_suffix_kwargs(ob.model, x)
        for a, k in zip(kw["position_embeddings"], ("cos", "sin"), strict=True):
            np.testing.assert_array_equal(a.cpu().numpy()[0], data[k])
        np.testing.assert_array_equal(
            kw["attention_mask"][0, 0, -1].cpu().numpy(), data["mask"]
        )
        torch.mps.synchronize()
        start = time.monotonic()
        stats["calls"] += 1
        try:
            return previous.fn.native_reader(ob.model, x, kw, ids, contrast)
        finally:
            torch.mps.synchronize()
            stats["native_seconds"] += time.monotonic() - start

    def pair(data, ds, fi, eta):
        result = []
        scores = []
        for qi, (x, d) in enumerate(zip(data, ds, strict=True)):
            qr, qs = [], []
            for sign in (-1, 1):
                changed = fn.perturb(x, d[fi], sign * eta)
                np.testing.assert_array_equal(changed["state"][-1], last[qi])
                checks["perturbation_last_checks"] += 1
                score, r = timed(changed)
                qr.append(r)
                qs.append(score)
            result.append(qr)
            scores.append(qs)
        return np.asarray(result), np.asarray(scores)

    try:
        for j in jobs:
            i = j["index"]
            data = [
                old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
                for k in j["query_keys"]
            ]
            orig = [
                old.arrays(previous.OUT / "capture" / ("parity_" + k + ".npz"))
                for k in j["query_keys"]
            ]
            if i == 0:
                parity = []
                for x, oldr in zip(data, orig, strict=True):
                    score, r = timed(x)
                    rel = float(
                        np.linalg.norm(r - oldr["r"]) / np.linalg.norm(oldr["r"])
                    )
                    mx = float(np.max(abs(r - oldr["r"])))
                    checks["parity_relative"].append(rel)
                    checks["parity_max"].append(mx)
                    parity.append(r)
                    if rel > 1e-3 or mx > 1e-4:
                        raise ValueError("Native endpoint parity failed")
                p = dst / "parity.npz"
                np.savez(p, r=np.stack(parity))
                paths.append(p)
                status("PARITY_PASSED")
            ds = [ep.body_directions(x) for x in data]
            np.testing.assert_array_equal(data[0]["positions"], data[1]["positions"])
            np.testing.assert_array_equal(
                data[0]["state"][data[0]["positions"]],
                data[1]["state"][data[1]["positions"]],
            )
            np.testing.assert_array_equal(
                ds[0][:, data[0]["positions"]], ds[1][:, data[1]["positions"]]
            )
            checks["paired_body_equal_checks"] += 1
            fixed = [fn.clamp(x, last[q]) for q, x in enumerate(data)]
            base_r, base_scores = [], []
            for q, x in enumerate(fixed):
                np.testing.assert_array_equal(x["state"][-1], last[q])
                checks["clamp_last_checks"] += 1
                score, r = timed(x)
                base_r.append(r)
                base_scores.append(score)
            captures, scores = [], []
            for fi in range(4):
                r, s = pair(fixed, ds, fi, ETA)
                captures.append(r)
                scores.append(s)
                if i == 0 and fi == 0:
                    half, hs = pair(fixed, ds, fi, ETA / 2)
                    a, b = fn.query_slopes(r, ETA), fn.query_slopes(half, ETA / 2)
                    error = float(
                        np.linalg.norm(a[2] - b[2]) / max(np.linalg.norm(a[2]), 1e-30)
                    )
                    checks["halfstep_relative"] = error
                    p = dst / "half.npz"
                    np.savez(p, r=half, score=hs)
                    paths.append(p)
                    if error > 0.20:
                        raise ValueError(
                            "Frozen differential secant sensitivity gate failed"
                        )
            p = dst / f"a{i:02d}.npz"
            np.savez(
                p,
                original_r=np.stack([x["r"] for x in orig]),
                original_score=np.asarray([x["score"] for x in orig]),
                clamped_r=np.stack(base_r),
                clamped_score=np.asarray(base_scores),
                perturbed_r=np.stack(captures),
                perturbed_score=np.stack(scores),
                last=last,
                original_last=np.stack([x["state"][-1] for x in data]),
                body_count=np.asarray([len(x["positions"]) for x in data]),
            )
            paths.append(p)
            stats["completed"] += 1
            status("RUNNING")
        if stats["calls"] != 78 or stats["native_seconds"] > m["max_native_seconds"]:
            raise ValueError("Native inventory/time violation")
        old.write(dst / "checks.json", checks)
        paths.append(dst / "checks.json")
        status("COMPLETE")
        old.write(
            dst / "freeze.json",
            {
                "status": "COMPLETE",
                "files": {p.name: old.sha(p) for p in paths},
                "calls": stats["calls"],
                "native_seconds": stats["native_seconds"],
                "loading_seconds": loading,
                "extra_denials": len(denied) - len(probes),
                "checks": checks,
            },
        )
    except BaseException:
        status("FAILED")
        raise


def tables(packet, old_basis, motion_basis):
    """Aggregate exact finite-panel accounts without treating directions as authors."""
    orig, fixed, slope = packet["original"], packet["fixed"], packet["slopes"]
    variants = {"original": orig, "clamped": fixed}
    dr, changes, sr, proj, cp, inp = [], [], [], [], [], []
    for variant, rows in variants.items():
        for qi, field in enumerate(FIELDS):
            dr.append(dict(variant=variant, query=field, **fn.dispersion(rows[:, qi])))
    for qi, field in enumerate(FIELDS):
        a, b = orig[:, qi], fixed[:, qi]
        changes.append(
            {
                "query": field,
                "authors": 4,
                "change_l2_rms": float(np.sqrt(np.mean(np.sum((b - a) ** 2, axis=1)))),
                "change_over_original_l2": float(np.sqrt(fn.ratio(b - a, a))),
                "clamped_over_original_dispersion": fn.ratio(
                    b - b.mean(axis=0), a - a.mean(axis=0)
                ),
            }
        )
        for fi in range(4):
            s = slope[:, fi, qi]
            sr.append(
                {
                    "query": field,
                    "factor": f"F{fi + 1:02d}",
                    "authors": 4,
                    "slope_l2_rms": float(np.sqrt(np.mean(np.sum(s * s, axis=1)))),
                    "step_over_reader_l2": float(ETA * np.sqrt(fn.ratio(s, b))),
                    "outside_original_range": fn.energy_outside(s, old_basis),
                    "outside_context_extended_range": fn.energy_outside(
                        s, motion_basis
                    ),
                    "mean_direction_energy_fraction": 1
                    - fn.dispersion(s)["floor_over_energy"],
                }
            )
    # Fixed ORIGINAL evidence vectors are projection rulers, not new clamped model targets.
    p, u = packet["primal"].mean(axis=1), packet["vectors"].mean(axis=1)
    for ai, arm in enumerate(("native_original", *ep.fn.ARMS[:4])):
        rulers = {"scalar": p[:, ai]}
        for ci, component in enumerate(ep.fn.COMPONENTS):
            for fi in range(4):
                rulers[f"{component}_F{fi + 1:02d}"] = u[:, ai, ci, fi]
        for observable, ruler in rulers.items():
            for variant, rows in variants.items():
                val = np.einsum("ih,ih->i", rows[:, 2], ruler)
                cp.append(
                    dict(
                        arm=arm,
                        observable=observable,
                        variant=variant,
                        **fn.summarize_scalars(val),
                    )
                )
            for fi in range(4):
                val = np.einsum("ih,ih->i", slope[:, fi, 2], ruler)
                proj.append(
                    dict(
                        arm=arm,
                        observable=observable,
                        factor=f"F{fi + 1:02d}",
                        **fn.summarize_scalars(val),
                    )
                )
    for qi, q in enumerate(old.QUERIES):
        a, b = packet["original_last"][:, qi], packet["last"][:, qi]
        inp.append(
            {
                "query": q,
                "authors": 4,
                "last_max_difference": float(abs(b - b[:1]).max()),
                "clamp_displacement_over_last_l2": float(np.sqrt(fn.ratio(b - a, a))),
                "body_count_mean": float(packet["body_count"][:, qi].mean()),
            }
        )
    return [pd.DataFrame(x) for x in (dr, changes, sr, proj, cp, inp)]


def analyze(out, folder):
    """Replay source-only vector geometry and fixed-evidence projections; zero fits."""
    old.verify(out)
    ep.freeze(out, "capture")
    used, start = prior.prior_cpu(out), time.process_time()
    dst = out / folder
    dst.mkdir(exist_ok=False)
    jobs = old.read(out / "private_jobs.json")
    w, qw = [old.arrays(x / "private_weights.npz") for x in (ep.STATE, ep.CAPTURE)]
    packet = {
        k: []
        for k in (
            "original",
            "fixed",
            "slopes",
            "primal",
            "vectors",
            "last",
            "original_last",
            "body_count",
        )
    }
    for j in jobs:
        c = old.arrays(out / "capture" / f"a{j['index']:02d}.npz")
        for name, field in (("original", "original_r"), ("fixed", "clamped_r")):
            r = c[field]
            packet[name].append(np.concatenate((r, (r[0] - r[1])[None])))
        packet["slopes"].append(
            np.stack([fn.query_slopes(r, ETA) for r in c["perturbed_r"]])
        )
        native = previous.previous.previous.native_packets(j, w, qw)
        program = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        for field in ("primal", "vectors"):
            packet[field].append(
                np.concatenate(
                    (native[field][:, 1:2], np.stack([x[field] for x in program])),
                    axis=1,
                )
            )
        for field in ("last", "original_last", "body_count"):
            packet[field].append(c[field])
    packet = {k: np.stack(v) for k, v in packet.items()}
    model = old.arrays(previous.OUT / "models/all4.npz")
    results = tables(packet, previous.fn.basis(model), previous.fn.basis(model, True))
    if [len(x) for x in results] != COUNTS:
        raise ValueError("Incomplete aggregate panel")
    np.savez(dst / "private_accounts.npz", **packet)
    for name, t in zip(TABLES, results, strict=True):
        t.to_csv(dst / name, index=False)
    if used + time.process_time() - start > CAP:
        raise TimeoutError("Cumulative numeric cap")
    ep.previous.finish(
        out,
        folder,
        [*TABLES, "private_accounts.npz"],
        start,
        counts=COUNTS,
        source_authors=4,
        fit_count=0,
        native_calls=0,
        human_labels_read=False,
    )


def audit(out):
    """Verify finite costs, input clamps, source inventory and exact numerical replay."""
    old.verify(out)
    cap, a, b = [ep.freeze(out, x) for x in ("capture", "analysis", "replay_analysis")]
    for name in TABLES:
        if a["files"][name] != b["files"][name]:
            raise ValueError("Aggregate replay mismatch")
    pa, pb = [
        old.arrays(out / f / "private_accounts.npz")
        for f in ("analysis", "replay_analysis")
    ]
    for key in pa:
        np.testing.assert_array_equal(pa[key], pb[key])
    checks = cap["checks"]
    if (
        cap["calls"] != 78
        or cap["native_seconds"] > 120
        or cap["extra_denials"]
        or len(cap["files"]) != 7
        or len(a["files"]) != 7
        or a["counts"] != COUNTS
        or checks["clamp_last_checks"] != 8
        or checks["perturbation_last_checks"] != 68
        or checks["paired_body_equal_checks"] != 4
        or checks["halfstep_relative"] > 0.20
        or prior.prior_cpu(out) > CAP
    ):
        raise ValueError("Finite capture/check/budget mismatch")
    np.testing.assert_array_equal(
        pa["last"], np.broadcast_to(pa["last"][:1], pa["last"].shape)
    )
    r = dict(
        status="PASS",
        authors=4,
        native_calls=cap["calls"],
        native_seconds=cap["native_seconds"],
        loading_seconds=cap["loading_seconds"],
        numeric_cpu_seconds=prior.prior_cpu(out),
        fit_count=0,
        public_tables=len(TABLES),
        parity_relative_max=max(checks["parity_relative"]),
        halfstep_relative=checks["halfstep_relative"],
        body_pair_checks=4,
        human_labels_read=False,
        **{k: checks[k] for k in ("clamp_last_checks", "perturbation_last_checks")},
    )
    old.write(out / "audit.json", r)
    print(r, flush=True)


def main():
    """Each stage is immutable and separate; native capture is never blindly replayed."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", required=True, choices=("prepare", "capture", "analyze", "audit")
    )
    p.add_argument(
        "--folder", choices=("analysis", "replay_analysis"), default="analysis"
    )
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage == "analyze":
            analyze(a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
