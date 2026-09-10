#!/usr/bin/env python3
"""NEXT80: source reader-motion teaching and fixed numerical endpoint application."""

import argparse
import inspect
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
from grain_research.vector_mechanisms import source_reader_motion as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_context_differential_reader as previous

old, prior, ep = previous.old, previous.prior, previous.ep
OUT = old.BASE / "logic_source_reader_motion_v1"
NATIVE = old.BASE / "logic_fixed_seed_attention_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_READER_MOTION_PLAN_V1.md"
TABLES = (*previous.TABLES, "source_motion_summary.csv")
COUNTS = [576, 1728, 288, 104, 96, 6, 2, 6, 10]
CAP = 90


def limit(start, used):
    """Bound numerical stages separately from the native intervention budget."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("NEXT80 numerical budget exceeded")


def prepare(out):
    """Pin the prior panel, native observer, source packages and exact code."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT79 required")
    native = old.read(NATIVE / "manifest.json")
    from transformers.models.qwen2 import modeling_qwen2

    if (
        old.sha(Path(inspect.getfile(modeling_qwen2)))
        != native["qwen_implementation_sha256"]
    ):
        raise ValueError("Frozen native implementation changed")
    out.mkdir(exist_ok=False)
    jobs = old.read(previous.OUT / "private_jobs.json")
    old.write(out / "private_jobs.json", jobs)
    inputs = dict(m["inputs"])
    inputs.update(
        {str((previous.OUT / n).relative_to(ROOT)): h for n, h in m["prepared"].items()}
    )
    for folder in ("models", "predictions", "analysis"):
        fr = ep.freeze(previous.OUT, folder)
        inputs.update(
            {
                str((previous.OUT / folder / n).relative_to(ROOT)): h
                for n, h in fr["files"].items()
            }
        )
    paths = [
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        NATIVE / "manifest.json",
        NATIVE / "observer.json",
        previous.OUT / "audit.json",
        ROOT / "tests/test_grain_source_reader_motion.py",
        ROOT / "scripts/run_grain_complete_transport_jvp.py",
        ROOT / "grain_research/vector_mechanisms/context_carrier.py",
    ]
    inputs.update({str(p.relative_to(ROOT)): old.sha(p) for p in paths})
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_READER_MOTION_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "prepare_cpu_seconds": time.process_time() - start,
            "model_sha256": native["model_sha256"],
            "alphabet": native["alphabet"],
            "c": native["c"],
            "max_calls": 76,
            "max_native_seconds": 120,
            "max_loading_seconds": 300,
            "max_cpu_seconds": CAP,
            "source_references": 5,
            "coefficient_solves": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED"}, flush=True)


def capture(out):
    """Capture only source perturbations, preserving partial cost and failure state."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from scripts.run_grain_complete_transport_jvp import load_observer

    m = old.verify(out)
    jobs = old.read(out / "private_jobs.json")
    src = {j["index"]: j for j in jobs if j["cohort"] == "source4"}
    refs = {prior.reference(j): ep.source_set(j["cohort"], j["index"]) for j in jobs}
    dst = out / "capture"
    dst.mkdir(exist_ok=False)
    stats = {
        "status": "LOADING",
        "calls": 0,
        "native_seconds": 0.0,
        "loading_seconds": 0.0,
        "completed": 0,
        "human_labels_read": False,
        "recipient_reads": 0,
    }
    status_path = out / "native_status.json"
    old.write(status_path, stats)

    def alarm(_sig, _frame):
        raise TimeoutError("Native observer load cap exceeded")

    handler = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(m["max_loading_seconds"])
    loading_start = time.monotonic()
    try:
        ob, loading = load_observer(m)
    except BaseException:
        stats.update(
            status="FAILED_LOADING", loading_seconds=time.monotonic() - loading_start
        )
        old.write(status_path, stats)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
    stats["loading_seconds"] = loading
    torch.set_grad_enabled(False)
    ids = torch.tensor(m["alphabet"]["token_ids"], device="mps")
    contrast = torch.tensor(m["c"], device="mps", dtype=torch.float32)
    source_keys = [k for j in src.values() for k in j["query_keys"]]
    reads = [
        root / (k + ".npz")
        for k in source_keys
        for root in (ep.STATE / "private_inputs", ep.ENDPOINT / "capture_source")
    ]
    reads += [previous.OUT / "models" / (ref + ".npz") for ref in refs]
    names = ["parity_" + k + ".npz" for k in source_keys]
    names += [f"{ref}_a{i:02d}.npz" for ref, members in refs.items() for i in members]
    names += ["all4_a00_half.npz", "checks.json"]
    probes = [
        ep.STATE / "private_inputs/new8_a00_neuroticism.npz",
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        previous.OUT / "analysis/score_metrics.csv",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")] + [status_path],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    paths = []
    checks = {
        "parity_relative": [],
        "parity_max": [],
        "halfstep_relative": None,
        "skipped_directions": 0,
    }

    def status(kind):
        stats.update(
            status=kind,
            files={p.name: old.sha(p) for p in paths},
            extra_denials=len(denied) - len(probes),
        )
        old.write(status_path, stats)
        print({k: v for k, v in stats.items() if k != "files"}, flush=True)

    def timed(data):
        if (
            stats["calls"] >= m["max_calls"]
            or stats["native_seconds"] >= m["max_native_seconds"]
        ):
            raise TimeoutError("Frozen native cap reached")
        state = torch.tensor(data["state"][None], device="mps")
        kw = cached_suffix_kwargs(ob.model, state)
        for a, k in zip(kw["position_embeddings"], ("cos", "sin"), strict=True):
            np.testing.assert_array_equal(a.cpu().numpy()[0], data[k])
        np.testing.assert_array_equal(
            kw["attention_mask"][0, 0, -1].cpu().numpy(), data["mask"]
        )
        torch.mps.synchronize()
        start = time.monotonic()
        stats["calls"] += 1
        try:
            return fn.native_reader(ob.model, state, kw, ids, contrast)
        finally:
            torch.mps.synchronize()
            stats["native_seconds"] += time.monotonic() - start

    def pair(data, v, eps):
        return np.stack(
            [
                np.stack([timed(fn.shifted(x, v, e))[1] for e in (-eps, eps)])
                for x in data
            ]
        )

    try:
        bank = {
            i: [
                old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
                for k in j["query_keys"]
            ]
            for i, j in src.items()
        }
        for i, j in src.items():
            for data, k in zip(bank[i], j["query_keys"], strict=True):
                score, r = timed(data)
                target = ep.selected(
                    ep.ENDPOINT / "capture_source" / (k + ".npz"), ("r",)
                )["r"].astype(float)
                rel = float(np.linalg.norm(r - target) / np.linalg.norm(target))
                mx = float(np.max(abs(r - target)))
                checks["parity_relative"].append(rel)
                checks["parity_max"].append(mx)
                p = dst / ("parity_" + k + ".npz")
                np.savez(p, r=r, score=np.asarray(score))
                paths.append(p)
                if rel > 1e-3 or mx > 1e-4:
                    raise ValueError("H16 suffix/post22 reader parity failed")
        status("PARITY_PASSED")
        for ref in sorted(refs):
            model = old.arrays(previous.OUT / "models" / (ref + ".npz"))
            np.testing.assert_array_equal(model["members"], refs[ref])
            v, eps = fn.directions(model)
            for pos, i in enumerate(model["members"]):
                if not eps or not np.any(v[pos]):
                    r = np.zeros((2, 2, len(v[pos])))
                    checks["skipped_directions"] += 1
                else:
                    r = pair(bank[int(i)], v[pos], eps)
                p = dst / f"{ref}_a{i:02d}.npz"
                np.savez(
                    p, r=r, v=v[pos], epsilon=np.asarray(eps), member=np.asarray(i)
                )
                paths.append(p)
                if ref == "all4" and i == 0:
                    half = (
                        pair(bank[int(i)], v[pos], eps / 2)
                        if eps and np.any(v[pos])
                        else r.copy()
                    )
                    j1, j2 = (
                        (fn.secant(r, eps), fn.secant(half, eps / 2))
                        if eps
                        else (np.zeros_like(v[pos]), np.zeros_like(v[pos]))
                    )
                    error = float(
                        np.linalg.norm(j1 - j2) / max(np.linalg.norm(j1), 1e-30)
                    )
                    checks["halfstep_relative"] = error
                    hp = dst / "all4_a00_half.npz"
                    np.savez(hp, r=half)
                    paths.append(hp)
                    if error > 0.20:
                        raise ValueError(
                            "Frozen finite-difference sensitivity gate failed"
                        )
                stats["completed"] += 1
                status("RUNNING")
        if stats["native_seconds"] > m["max_native_seconds"] or stats["calls"] > 76:
            raise TimeoutError("Native cap exceeded by last completed call")
        old.write(dst / "checks.json", checks)
        paths.append(dst / "checks.json")
        status("COMPLETE")
        old.write(
            dst / "freeze.json",
            {
                "status": "COMPLETE",
                "files": {p.name: old.sha(p) for p in paths},
                "native_seconds": stats["native_seconds"],
                "loading_seconds": loading,
                "calls": stats["calls"],
                "extra_denials": len(denied) - len(probes),
                "checks": checks,
            },
        )
    except BaseException:
        status("FAILED")
        raise


def models(out):
    """Compile one secant column per source/reference with no regression solves."""
    old.verify(out)
    ep.freeze(out, "capture")
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    refs = {prior.reference(j) for j in jobs}
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = []
    for ref in sorted(refs):
        model = old.arrays(previous.OUT / "models" / (ref + ".npz"))
        v, eps = fn.directions(model)
        js = []
        for pos, i in enumerate(model["members"]):
            data = old.arrays(out / "capture" / f"{ref}_a{i:02d}.npz")
            np.testing.assert_array_equal(data["v"], v[pos])
            if int(data["member"]) != i or float(data["epsilon"]) != eps:
                raise ValueError("Source capture membership/direction changed")
            js.append(
                fn.secant(data["r"], eps)
                if eps and np.any(v[pos])
                else np.zeros_like(v[pos])
            )
        name = ref + ".npz"
        np.savez(dst / name, **model, v=v, j=np.stack(js), epsilon=np.asarray(eps))
        names.append(name)
        limit(start, used)
    ep.previous.finish(
        out, "models", names, start, source_constructions=5, coefficient_solves=0
    )


def predict(out, folder):
    """Apply all three source reader fields with no current native target access."""
    old.verify(out)
    fm = ep.freeze(out, "models")
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    keys = [k for j in jobs for k in j["query_keys"]]
    reads = [ep.STATE / "private_weights.npz"] + [
        out / "models" / n for n in fm["files"]
    ]
    reads += [
        root / (k + ".npz")
        for k in keys
        for root in (ep.STATE / "private_inputs", ep.OUT / "predictions")
    ]
    reads += [previous.OUT / "predictions" / n for n in names]
    reads += [
        root / "models" / f"{prior.reference(j)}_{q}.npz"
        for j in jobs
        for q in old.QUERIES
        for root in (ep.SELECTOR, ep.previous.OUT)
    ]
    probes = [
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        out / "capture/all4_a00.npz",
        ep.SOURCE / "private_inputs" / (keys[0] + ".npz"),
        ep.CAPTURE / "private_capture" / (keys[0] + ".npz"),
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    weights = old.arrays(ep.STATE / "private_weights.npz")
    if {"q", "q_bias"} & set(weights):
        raise ValueError("Native Q forbidden")
    fd, baseline, body = [], [], 0
    for job in jobs:
        model = old.arrays(out / "models" / (prior.reference(job) + ".npz"))
        np.testing.assert_array_equal(
            model["members"], ep.source_set(job["cohort"], job["index"])
        )
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in job["query_keys"]
        ]
        c = np.mean([x["state"][-1] for x in data], axis=0)
        ds = [ep.body_directions(x) for x in data]
        for d in ds:
            if np.any(d[:, -1]):
                raise ValueError("BODY must not move context")
            body += 1
        r = np.stack([fn.read(model, c, v) for v in fn.VARIANTS])
        packets = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in job["query_keys"]
        ]
        primal, vectors = [
            np.stack([x[k] for x in packets]) for k in ("primal", "vectors")
        ]
        ps = [fn.base.project(primal, vectors, rv) for rv in r]
        result = {k: np.stack([x[k] for x in ps]) for k in ps[0]}
        original = old.arrays(previous.OUT / "predictions" / (job["key"] + ".npz"))
        for k in result:
            baseline.append(float(np.max(abs(result[k][:2] - original[k]))))
        if job["cohort"] == "source4" and job["index"] == 0:
            for qi, q in enumerate(old.QUERIES):
                name = f"{prior.reference(job)}_{q}.npz"
                sm, vm = [
                    old.arrays(root / "models" / name)
                    for root in (ep.SELECTOR, ep.previous.OUT)
                ]
                for f in range(4):
                    scores = []
                    for epsilon in (-1e-5, 1e-5):
                        changed = dict(
                            data[qi], state=data[qi]["state"] + epsilon * ds[qi][f]
                        )
                        cx = (changed["state"][-1] + data[1 - qi]["state"][-1]) / 2
                        px = ep.fn.predict(
                            changed, ds[qi], weights, sm, vm, ep.fn.readers(r[2])[qi]
                        )
                        scores.append(
                            px["primal"] @ ep.fn.readers(fn.read(model, cx))[qi].T
                        )
                    fd.extend(
                        abs(
                            (scores[1] - scores[0]) / 2e-5
                            - result["actions"][2, qi, :, 2, :, f]
                        ).ravel()
                    )
        diag = fn.support_account(model, c)
        np.savez(
            dst / (job["key"] + ".npz"),
            **result,
            readers=r,
            diagnostics=np.asarray(list(diag.values())),
            diagnostic_fields=np.asarray(list(diag)),
        )
        limit(start, used)
    if len(fd) != 64 or max(fd) > 1e-7 or max(baseline) > 1e-10 or body != 24:
        raise ValueError("Field scalar derivative or baseline replay failed")
    ep.previous.finish(
        out,
        folder,
        names,
        start,
        fd_count=len(fd),
        fd_max_error=float(max(fd)),
        baseline_max_error=max(baseline),
        body_context_checks=body,
        native_reads=0,
        target_reads=0,
        extra_denials=len(denied) - len(probes),
    )


def summarize(groups):
    """Reuse the frozen two-variant statistic definitions without mutating them."""
    variants = (
        "scalar",
        "actions",
        "scalar_terms",
        "action_terms",
        "projected_scalar",
        "projected_actions",
    )
    tables = []
    for vi in (1, 2):
        paired = {}
        for cohort, rows in groups.items():
            paired[cohort] = [
                dict(
                    row,
                    **{k: row[k][[0, vi]] for k in variants},
                    ranges=[row["ranges"][0], row["ranges"][vi]],
                )
                for row in rows
            ]
        current = previous.summarize(paired)
        if vi == 1:
            tables = current
        else:
            for ti, t in enumerate(current):
                if "variant" not in t.columns:
                    continue
                add = t[t.variant == "context_kernel"].copy()
                add["variant"] = "source_motion"
                tables[ti] = pd.concat((tables[ti], add), ignore_index=True)
    return tables


def source_motion_summary(out, native_sources, jobs):
    """Project source secants on native and all four frozen source-program vectors."""
    groups = {}
    sources = {j["index"]: j for j in jobs if j["cohort"] == "source4"}
    program = {
        i: [old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]]
        for i, j in sources.items()
    }
    for ref in sorted({prior.reference(j) for j in jobs}):
        model = old.arrays(out / "models" / (ref + ".npz"))
        basis = fn.basis(model)
        group = "source4_all" if ref == "all4" else "source3_LOO"
        for pos, i in enumerate(model["members"]):
            slope = model["j"][pos]
            outside = slope - (slope @ basis.T) @ basis
            p, u = native_sources[int(i)]
            ps = np.concatenate(
                (p[:, None, :], np.stack([x["primal"] for x in program[int(i)]])),
                axis=1,
            )
            us = np.concatenate(
                (
                    u[:, None, :, :, :],
                    np.stack([x["vectors"] for x in program[int(i)]]),
                ),
                axis=1,
            )
            for ai, arm in enumerate(("native", *ep.fn.ARMS[:4])):
                groups.setdefault((group, arm), []).append(
                    {
                        "slope2": float(slope @ slope),
                        "outside2": float(outside @ outside),
                        "scalar": float(ps[:, ai].mean(axis=0) @ slope),
                        "action": us[:, ai].mean(axis=0) @ slope,
                        "epsilon": float(model["epsilon"]),
                    }
                )
    rows = []
    for (group, arm), values in groups.items():
        energy = sum(x["slope2"] for x in values)
        row = {
            "reference_design": group,
            "arm": arm,
            "distinct_sources": 4,
            "source_reference_units": len(values),
            "slope_l2_rms": float(np.sqrt(np.mean([x["slope2"] for x in values]))),
            "outside_energy_fraction": sum(x["outside2"] for x in values) / energy
            if energy
            else 0.0,
            "scalar_sensitivity_rms": float(
                np.sqrt(np.mean([x["scalar"] ** 2 for x in values]))
            ),
            "scalar_step_effect_rms": float(
                np.sqrt(np.mean([(x["epsilon"] * x["scalar"]) ** 2 for x in values]))
            ),
        }
        actions = np.stack([x["action"] for x in values])
        for ci, component in enumerate(ep.fn.COMPONENTS):
            row[component + "_sensitivity_rms"] = float(
                np.sqrt(np.mean(actions[:, ci] ** 2))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(out, folder):
    """Use frozen targets to separate response utility from source range expansion."""
    old.verify(out)
    ep.freeze(out, folder)
    used, start = prior.prior_cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    w, qw = (
        old.arrays(ep.STATE / "private_weights.npz"),
        old.arrays(ep.CAPTURE / "private_weights.npz"),
    )
    groups, names = {"source4": [], "new8": []}, []
    native_sources = {}
    jobs = old.read(out / "private_jobs.json")
    for j in jobs:
        p = old.arrays(out / folder / (j["key"] + ".npz"))
        model = old.arrays(out / "models" / (prior.reference(j) + ".npz"))
        native = previous.previous.native_packets(j, w, qw)
        rc = np.stack(
            [
                ep.selected(
                    ep.ENDPOINT
                    / ("capture_source" if j["cohort"] == "source4" else "capture_new")
                    / (k + ".npz"),
                    ("r",),
                )["r"]
                for k in j["query_keys"]
            ]
        )
        primal, vectors = native["primal"][:, 1:2], native["vectors"][:, 1:2]
        if j["cohort"] == "source4":
            native_sources[j["index"]] = (primal[:, 0], vectors[:, 0])
        target = fn.base.project(primal, vectors, rc)
        row = {k: p[k] for k in ("scalar", "actions", "scalar_terms", "action_terms")}
        row.update(
            {
                "target_" + k: (v[:, 0] if k in ("scalar", "actions") else v[0])
                for k, v in target.items()
            }
        )
        rd = p["readers"][:, 0] - p["readers"][:, 1]
        row["projected_scalar"] = rd @ primal.mean(axis=0)[0]
        row["projected_actions"] = np.einsum("cfh,vh->vcf", vectors.mean(axis=0)[0], rd)
        name = "private_" + j["key"] + ".npz"
        np.savez(dst / name, **row)
        names.append(name)
        row["ranges"] = [
            fn.range_account(model, rc[0] - rc[1], d, vi == 2)
            for vi, d in enumerate(rd)
        ]
        row["diagnostics"] = {
            str(k): float(v)
            for k, v in zip(p["diagnostic_fields"], p["diagnostics"], strict=True)
        }
        groups[j["cohort"]].append(row)
        limit(start, used)
    tables = [*summarize(groups), source_motion_summary(out, native_sources, jobs)]
    if [len(t) for t in tables] != COUNTS:
        raise ValueError("Incomplete fixed panel")
    errors = []
    for name, t in zip(TABLES, tables, strict=True):
        if name == "source_motion_summary.csv":
            t.to_csv(dst / name, index=False)
            continue
        prior_table = pd.read_csv(previous.OUT / "analysis" / name)
        comp = t[t.variant != "source_motion"] if "variant" in t.columns else t
        for col in prior_table:
            if prior_table[col].dtype.kind in "if":
                errors.append(
                    float(
                        np.max(abs(comp[col].to_numpy() - prior_table[col].to_numpy()))
                    )
                )
            elif list(comp[col]) != list(prior_table[col]):
                raise ValueError("Old panel ordering changed")
        t.to_csv(dst / name, index=False)
    if max(errors) > 1e-8:
        raise ValueError("Prior statistic reproduction failed")
    limit(start, used)
    ep.previous.finish(
        out,
        dst.name,
        [*TABLES, *names],
        start,
        counts=COUNTS,
        baseline_metric_max_error=max(errors),
    )


def audit(out):
    """Check all finite artifacts and cached replay, not scientific promotion."""
    old.verify(out)
    c = ep.freeze(out, "capture")
    fm = ep.freeze(out, "models")
    a, b, aa, bb = [
        ep.freeze(out, n)
        for n in ("predictions", "replay", "analysis", "replay_analysis")
    ]
    for first, second, fa, fb in (
        ("predictions", "replay", a, b),
        ("analysis", "replay_analysis", aa, bb),
    ):
        if set(fa["files"]) != set(fb["files"]):
            raise ValueError("Replay inventory mismatch")
        for name in fa["files"]:
            if name.endswith(".npz"):
                x, y = [old.arrays(out / f / name) for f in (first, second)]
                if set(x) != set(y):
                    raise ValueError("Replay fields differ")
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k])
            elif fa["files"][name] != fb["files"][name]:
                raise ValueError("CSV replay mismatch")
    if (
        len(c["files"]) != 26
        or len(fm["files"]) != 5
        or len(a["files"]) != 12
        or len(aa["files"]) != 21
        or aa["counts"] != COUNTS
        or c["extra_denials"]
        or c["calls"] > 76
        or c["native_seconds"] > 120
        or any(
            f["extra_denials"] or f["native_reads"] or f["target_reads"] for f in (a, b)
        )
        or prior.prior_cpu(out) > CAP
    ):
        raise ValueError("Finite budget/access/inventory violation")
    result = {
        "status": "PASS",
        "source_constructions": 5,
        "coefficient_solves": 0,
        "human_labels_read": False,
        "native_calls": c["calls"],
        "native_seconds": c["native_seconds"],
        "loading_seconds": c["loading_seconds"],
        "parity_relative_max": max(c["checks"]["parity_relative"]),
        "halfstep_relative": c["checks"]["halfstep_relative"],
        "public_tables": 9,
        "application_packets": 12,
        "body_context_checks": a["body_context_checks"],
        "fd_count": a["fd_count"],
        "fd_max_error": a["fd_max_error"],
        "baseline_max_error": a["baseline_max_error"],
        "baseline_metric_max_error": aa["baseline_metric_max_error"],
        "numeric_cpu_seconds": prior.prior_cpu(out),
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Each immutable stage runs in its own process; native calls are never replayed."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "capture", "models", "predict", "evaluate", "audit"),
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
