#!/usr/bin/env python3
"""NEXT84: two source-only output-span fits and immutable ten-field comparison."""

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
from grain_research.vector_mechanisms import source_observable_span as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_observable_reader_teaching as previous

old, ep, prior = previous.old, previous.ep, previous.prior
base = fn.prior.base
OUT = old.BASE / "logic_source_observable_span_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_OBSERVABLE_SPAN_PLAN_V1.md"
VARIANTS = (*previous.VARIANTS, *fn.VARIANTS)
TABLES = (*previous.TABLES, "source_extension.csv", "new_direction_usage.csv")
COUNTS = [36, 4, 240, 720, 2880, 130, 10, 960, 4, 52, 2, 2]
CAP = 120
VARIANT_KEYS = (
    "scalar",
    "actions",
    "correction",
    "diagonal",
    "readers",
    "reader_derivatives",
)


def limit(start, used):
    """Bound numerical CPU independently of recipient outcomes."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("NEXT84 cumulative numerical CPU cap")


def unique_rulers(model):
    """Validate seven observations per source author and retain twenty unique rows."""
    authors = previous.observation_authors(model["groups"])
    blocks = model["source_rulers"][np.arange(4) * 2]
    if blocks.shape[:2] != (4, 5):
        raise ValueError("Four source authors, five rulers each required")
    np.testing.assert_array_equal(model["source_rulers"], blocks[authors])
    return blocks.reshape(20, -1)


def prepare(out):
    """Pin source fits, all comparisons and prior target packets before fitting."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT83 required")
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", old.read(previous.OUT / "private_jobs.json"))
    inputs = dict(m["inputs"])
    for folder in ("models", "predictions", "analysis"):
        fr = ep.freeze(previous.OUT, folder)
        inputs.update(
            {
                str((previous.OUT / folder / n).relative_to(ROOT)): h
                for n, h in fr["files"].items()
            }
        )
    for p in (
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        previous.OUT / "audit.json",
        ROOT / "tests/test_grain_source_observable_span.py",
    ):
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_OBSERVABLE_SPAN_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "prepare_cpu_seconds": time.process_time() - start,
            "source_authors": 4,
            "recipient_authors": 8,
            "observations": 28,
            "source_unique_rulers": 20,
            "max_fits": 2,
            "max_cpu_seconds": CAP,
            "native_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED"}, flush=True)


def models(out):
    """Fit from frozen NEXT83 source-only model packages, without native captures."""
    old.verify(out)
    used, start = prior.prior_cpu(out), time.process_time()
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [v + ".npz" for v in fn.VARIANTS]
    reads = [previous.OUT / "models" / (v + ".npz") for v in fn.prior.VARIANTS]
    probes = [
        previous.OUT / "analysis/private_new8_a00.npz",
        ep.STATE / "private_inputs/new8_a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    for i, (path, name) in enumerate(zip(reads, names, strict=True)):
        model = old.arrays(path)
        result = fn.fit(model, unique_rulers(model), bool(i))
        if result["output_basis"].shape[1] > 48 or result["theta"].size > 1344:
            raise ValueError("Fixed source-span size exceeded")
        np.savez(dst / name, **result)
        limit(start, used)
    ep.previous.finish(
        out,
        "models",
        names,
        start,
        coefficient_solves=2,
        observations=28,
        extra_denials=len(denied) - len(probes),
        human_labels_read=False,
    )


def finite_differences(data, ds, models, readers, result, weights):
    """Check both anchored and moving-reader calculus on one unchanged author."""
    fa, fd, eps = [], [], 1e-5
    for f in range(4):
        anchored, diagonal = [], []
        for e in (-eps, eps):
            changed = [
                x | {"state": x["state"] + e * d[f]}
                for x, d in zip(data, ds, strict=True)
            ]
            cb, cdb = base.features(changed, ds)
            pp = []
            for qi, q in enumerate(old.QUERIES):
                sm, vm = [
                    old.arrays(root / "models" / f"all4_{q}.npz")
                    for root in (ep.SELECTOR, ep.previous.OUT)
                ]
                pp.append(
                    ep.fn.predict(
                        changed[qi],
                        ds[qi],
                        weights,
                        sm,
                        vm,
                        ep.fn.readers(readers[0])[qi],
                    )["primal"]
                )
            pp = np.stack(pp)
            aa, dd = [], []
            for vi, m in enumerate(models, 8):
                cr, _ = base.readers(m, cb, cdb, m["scale"], "grain_field")
                aa.append(np.einsum("qah,qmh->qam", pp, ep.fn.readers(readers[vi])))
                dd.append(np.einsum("qah,qmh->qam", pp, ep.fn.readers(cr)))
            anchored.append(np.stack(aa))
            diagonal.append(np.stack(dd))
        fa.extend(
            abs(
                (anchored[1] - anchored[0]) / (2 * eps)
                - result["actions"][8:, :, :, 2, :, f]
            ).ravel()
        )
        fd.extend(
            abs(
                (diagonal[1] - diagonal[0]) / (2 * eps)
                - result["diagonal"][8:, :, :, :, f]
            ).ravel()
        )
    return fa, fd


def predict(out, folder):
    """Apply ten fixed comparators with H16 only; do not read recipient targets."""
    old.verify(out)
    fr = ep.freeze(out, "models")
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "new8"]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    reads = [out / "models" / n for n in fr["files"]] + [
        ep.STATE / "private_weights.npz"
    ]
    reads += [
        root / (k + ".npz")
        for j in jobs
        for k in j["query_keys"]
        for root in (ep.STATE / "private_inputs", ep.OUT / "predictions")
    ]
    reads += [previous.OUT / "predictions" / n for n in names]
    reads += [
        root / "models" / f"all4_{q}.npz"
        for q in old.QUERIES
        for root in (ep.SELECTOR, ep.previous.OUT)
    ]
    probes = [
        previous.OUT / "analysis/private_new8_a00.npz",
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    weights = old.arrays(ep.STATE / "private_weights.npz")
    if {"q", "q_bias"} & set(weights):
        raise ValueError("Current Q forbidden in application")
    fa, fd, parity, comparator = [], [], [], []
    for j in jobs:
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        ds = [ep.body_directions(d) for d in data]
        b, db = base.features(data, ds)
        saved = old.arrays(previous.OUT / "predictions" / (j["key"] + ".npz"))
        packet = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        primal, vectors = [
            np.stack([p[k] for p in packet]) for k in ("primal", "vectors")
        ]
        rr, dr = list(saved["readers"]), list(saved["reader_derivatives"])
        for m in ms:
            r, d = base.readers(m, b, db, m["scale"], "grain_field")
            rr.append(r)
            dr.append(d)
        rr, dr = np.stack(rr), np.stack(dr)
        ps = [
            previous.previous.projections(primal, vectors, r, d)
            for r, d in zip(rr, dr, strict=True)
        ]
        result = {k: np.stack([p[k] for p in ps]) for k in ps[0]}
        for k in result:
            comparator.append(float(abs(result[k][:8] - saved[k]).max()))
            parity.append(float(abs(result[k][8] - saved[k][6]).max()))
        if j["index"] == 0:
            fa, fd = finite_differences(data, ds, ms, rr, result, weights)
        np.savez(dst / (j["key"] + ".npz"), **result, readers=rr, reader_derivatives=dr)
        limit(start, used)
    if (
        len(fa) != 128
        or len(fd) != 128
        or max(fa + fd) > 1e-6
        or max(parity) > 1e-8
        or max(comparator) > 1e-10
    ):
        raise ValueError("Finite difference or comparator parity failed")
    ep.previous.finish(
        out,
        folder,
        names,
        start,
        anchored_fd_count=len(fa),
        diagonal_fd_count=len(fd),
        anchored_fd_max=float(max(fa)),
        diagonal_fd_max=float(max(fd)),
        control_max=max(parity),
        comparator_max=max(comparator),
        native_reads=0,
        target_reads=0,
        extra_denials=len(denied) - len(probes),
    )


def new_tables(rows, models):
    """Adapt only the last two slots to the frozen eight-slot summary interface."""
    indices = [0, 1, 2, 3, 4, 5, 8, 9]
    view = [
        {k: v[indices] if k in VARIANT_KEYS else v for k, v in r.items()} for r in rows
    ]
    remap = dict(zip(fn.prior.VARIANTS, fn.VARIANTS, strict=True))
    tables = []
    for t in previous.summarize(view, models):
        t = t[t["variant"].isin(remap)].copy()
        t["variant"] = t["variant"].map(remap)
        tables.append(t)
    return tables


def summarize(rows, models, old_tables):
    """Keep all prior aggregates and append two new fields plus signed usage."""
    tables = [
        pd.concat((a, b), ignore_index=True)
        for a, b in zip(old_tables, new_tables(rows, models), strict=True)
    ]
    extension, usage = [], []
    native = np.stack([r["native_r"][0] - r["native_r"][1] for r in rows])
    for vi, (variant, m) in enumerate(zip(fn.VARIANTS, models, strict=True), 8):
        extension.append(
            {
                "variant": variant,
                "authors": 4,
                "old_output_rank": m["original_output_basis"].shape[1],
                "new_output_rank": m["output_basis"].shape[1],
                **{
                    k: float(m[k])
                    for k in (
                        "ruler_coverage_relative",
                        "extension_rank_tolerance",
                        "old_gradient_norm",
                        "missing_gradient_norm",
                        "extra_coefficient_energy",
                        "extra_source_field_energy",
                        "old_objective",
                        "new_objective",
                    )
                },
            }
        )
        pred = np.stack([r["readers"][vi, 0] - r["readers"][vi, 1] for r in rows])
        usage.append(
            {"variant": variant, "authors": 8, **fn.direction_usage(pred, native, m)}
        )
    return [*tables, pd.DataFrame(extension), pd.DataFrame(usage)]


def evaluate(out, folder):
    """Open cached native evaluation arrays only after application freeze."""
    old.verify(out)
    ep.freeze(out, folder)
    used, start = prior.prior_cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    rows, names = [], []
    for j in old.read(out / "private_jobs.json"):
        if j["cohort"] != "new8":
            continue
        name = "private_" + j["key"] + ".npz"
        row = old.arrays(out / folder / (j["key"] + ".npz")) | ep.selected(
            previous.OUT / "analysis" / name,
            (
                "native_r",
                "native_primal",
                "native_vectors",
                "target_scalar",
                "target_actions",
            ),
        )
        np.savez(dst / name, **row)
        rows.append(row)
        names.append(name)
        limit(start, used)
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    before = [pd.read_csv(previous.OUT / "analysis" / n) for n in previous.TABLES]
    tables = summarize(rows, ms, before)
    if [len(t) for t in tables] != COUNTS:
        raise ValueError("Incomplete comparison inventory")
    for name, t in zip(TABLES, tables, strict=True):
        t.to_csv(dst / name, index=False)
    limit(start, used)
    ep.previous.finish(out, dst.name, [*TABLES, *names], start, counts=COUNTS)


def audit(out):
    """Check source budget, exact replay, unchanged comparator tables and read scope."""
    old.verify(out)
    m = ep.freeze(out, "models")
    stages = {
        f: ep.freeze(out, f)
        for f in ("predictions", "analysis", "replay", "replay_analysis")
    }
    for left, right in (("predictions", "replay"), ("analysis", "replay_analysis")):
        a, b = stages[left], stages[right]
        if set(a["files"]) != set(b["files"]):
            raise ValueError("Replay inventory differs")
        for name in a["files"]:
            if name.endswith(".npz"):
                x, y = [old.arrays(out / f / name) for f in (left, right)]
                if set(x) != set(y):
                    raise ValueError("Replay keys differ")
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k])
            elif a["files"][name] != b["files"][name]:
                raise ValueError("CSV replay differs")
    for n in previous.TABLES:
        a, b = [pd.read_csv(p / "analysis" / n) for p in (previous.OUT, out)]
        pd.testing.assert_frame_equal(
            a,
            b.iloc[: len(a)].reset_index(drop=True),
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    p = stages["predictions"]
    if (
        len(m["files"]) != 2
        or m["coefficient_solves"] != 2
        or m["extra_denials"]
        or len(p["files"]) != 8
        or stages["analysis"]["counts"] != COUNTS
        or prior.prior_cpu(out) > CAP
    ):
        raise ValueError("Budget/inventory failure")
    for f in ("predictions", "replay"):
        if any(stages[f][k] for k in ("extra_denials", "native_reads", "target_reads")):
            raise ValueError("Unexpected application reads")
    result = {
        "status": "PASS",
        "source_authors": 4,
        "recipient_authors": 8,
        "coefficient_solves": 2,
        "observations": 28,
        "native_calls": 0,
        "human_labels_read": False,
        "numeric_cpu_seconds": prior.prior_cpu(out),
        "public_tables": len(TABLES),
        **{
            k: p[k]
            for k in (
                "anchored_fd_count",
                "diagonal_fd_count",
                "anchored_fd_max",
                "diagonal_fd_max",
                "control_max",
                "comparator_max",
            )
        },
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Execute one immutable stage per process."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=(
            "prepare",
            "models",
            "predict",
            "evaluate",
            "replay",
            "replay-evaluate",
            "audit",
        ),
        required=True,
    )
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        if args.stage in ("predict", "replay"):
            predict(args.output, "predictions" if args.stage == "predict" else "replay")
        elif args.stage in ("evaluate", "replay-evaluate"):
            evaluate(
                args.output, "predictions" if args.stage == "evaluate" else "replay"
            )
        else:
            {"prepare": prepare, "models": models, "audit": audit}[args.stage](
                args.output
            )


if __name__ == "__main__":
    main()
