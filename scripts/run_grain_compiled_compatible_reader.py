#!/usr/bin/env python3
"""NEXT87: matched teaching for the evidence actually available at application."""

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
from grain_research.vector_mechanisms import compiled_compatible_reader as fn
from grain_research.vector_mechanisms import teaching_composition as account
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_conditional_teaching as previous

old, ep, p82, coord = previous.old, previous.ep, previous.p82, previous.coord
p85 = previous.previous
OUT = old.BASE / "logic_compiled_compatible_reader_v1"
PLAN = ROOT / "docs/GRAIN_COMPILED_COMPATIBLE_READER_PLAN_V1.md"
SOURCE_MODEL = p85.OUT / "models/moving_evidence_coordinates.npz"
CAP, EPS = 60, p85.EPS
CORE = previous.CORE
TABLES = (
    *CORE,
    "source_fit.csv",
    "conditioning.csv",
    "coordinate_prediction.csv",
    "signed_errors.csv",
)
COUNTS = [480, 1440, 5760, 260, 20, 1920, 10, 2, 20, 32]


def limit(start, used):
    """Stop on the frozen numerical budget, not on outcome quality."""
    if used + time.process_time() - start > CAP:
        raise TimeoutError("NEXT87 numerical CPU cap")


def prepare(out):
    """Pin code and source/application roles before constructing new teachers."""
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT86 required")
    inputs = dict(m["inputs"])
    jobs = old.read(previous.OUT / "private_jobs.json")
    helper = p82.motion.previous.previous
    native_paths = [ep.CAPTURE / "private_weights.npz"]
    native_paths += [
        root / (key + ".npz")
        for j in jobs
        if j["cohort"] == "source4"
        for key in j["query_keys"]
        for root in (
            ep.CAPTURE / "private_capture",
            ep.SOURCE / "private_inputs",
            ep.ENDPOINT / "capture_source",
        )
    ]
    native_paths += [
        ep.SELECTOR / "models" / f"{helper.prior.reference(j)}_{q}.npz"
        for j in jobs
        if j["cohort"] == "source4"
        for q in old.QUERIES
    ]
    for path in native_paths:
        inputs[str(path.relative_to(ROOT))] = old.sha(path)
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
        Path(account.__file__),
        previous.OUT / "audit.json",
        ROOT / "tests/test_grain_compiled_compatible_reader.py",
        ROOT / "scripts/report_grain_compiled_compatible_reader.py",
    ):
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    failure = OUT.with_name(OUT.name + "_aborted_precision") / "failure.json"
    prior_attempt = None
    if failure.exists():
        inputs[str(failure.relative_to(ROOT))] = old.sha(failure)
        prior_attempt = old.read(failure)
    start = time.process_time()
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", old.read(previous.OUT / "private_jobs.json"))
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_COMPILED_COMPATIBLE_READER_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "prepare_cpu_seconds": time.process_time() - start,
            "source_authors": 4,
            "recipient_authors": 8,
            "max_fits": 2,
            "max_cpu_seconds": CAP,
            "source_compiled_query_calls": 8,
            "application_fd_query_calls_each": 16,
            "native_calls": 0,
            "human_labels_read": False,
            "prior_attempt": prior_attempt,
            "teaching": "matched original4 values; exact old weights/lambda/units; all4 compiled source evidence",
        },
    )
    print({"status": "PREPARED"}, flush=True)


def teachers(out):
    """Reconstruct source native local observations and all4 compiled evidence."""
    old.verify(out)
    used, start = p85.cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "source4"]
    dst = out / "teachers"
    dst.mkdir(exist_ok=False)
    helper = p82.motion.previous.previous
    reads = [
        p85.BASELINE,
        SOURCE_MODEL,
        *p85.compiled_paths(),
        ep.CAPTURE / "private_weights.npz",
        p85.OUT / "teachers/private_teachers.npz",
    ]
    reads += [
        root / (k + ".npz")
        for j in jobs
        for k in j["query_keys"]
        for root in (
            ep.STATE / "private_inputs",
            ep.CAPTURE / "private_capture",
            ep.SOURCE / "private_inputs",
            ep.ENDPOINT / "capture_source",
        )
    ]
    reads += [
        ep.SELECTOR / "models" / f"{helper.prior.reference(j)}_{q}.npz"
        for j in jobs
        for q in old.QUERIES
    ]
    probes = [
        p85.OUT / "analysis/private_new8_a00.npz",
        ep.STATE / "private_inputs/new8_a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in ("private_teachers.npz", "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    baseline, source = old.arrays(p85.BASELINE), old.arrays(SOURCE_MODEL)
    old_teacher = old.arrays(p85.OUT / "teachers/private_teachers.npz")
    qw = old.arrays(ep.CAPTURE / "private_weights.npz")
    compiler = p85.Compiler(8)
    current, native, readers, points = [], [], [], []
    for i, j in enumerate(jobs):
        if j["index"] != i:
            raise ValueError("Source author order changed")
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        ds = [ep.body_directions(x) for x in data]
        p = compiler(data, ds)
        n = helper.native_packets(j, compiler.weights, qw)
        current.append(
            np.concatenate((p["primal"][:, 3, None], p["vectors"][:, 3, 2]), axis=1)
        )
        native.append(
            np.concatenate((n["primal"][:, 1, None], n["vectors"][:, 1, 2]), axis=1)
        )
        readers.append(
            np.stack(
                [
                    old.arrays(ep.ENDPOINT / "capture_source" / (k + ".npz"))["r"]
                    for k in j["query_keys"]
                ]
            )
        )
        points.append(
            fn.base.flatten(fn.base.features(data), baseline["scale"], "grain_field")
        )
        limit(start, used)
    current, native, readers = [
        np.stack(a).astype(float) for a in (current, native, readers)
    ]
    t = fn.teaching(
        current, native, readers, baseline["m"], baseline["d0"], source["units"]
    )
    np.testing.assert_allclose(points, source["x"][fn.ORIGINAL], atol=1e-12)
    np.testing.assert_allclose(t["frame"], old_teacher["original_frames"], atol=1e-12)
    np.testing.assert_allclose(
        readers[:, 0] - readers[:, 1] - baseline["d0"],
        baseline["target"][fn.ORIGINAL],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        t["native_coordinates"], source["target"][fn.ORIGINAL], atol=1e-10
    )
    np.savez(
        dst / "private_teachers.npz",
        **t,
        current=current,
        native=native,
        native_readers=readers,
    )
    ep.previous.finish(
        out,
        "teachers",
        ["private_teachers.npz"],
        start,
        compiled_query_calls=compiler.calls,
        native_cached_query_reconstructions=8,
        target_state_binding=True,
        extra_denials=len(denied) - len(probes),
        native_calls=0,
    )


def models(out):
    """Fit only the two original-source value targets, never renormalizing."""
    old.verify(out)
    ep.freeze(out, "teachers")
    used, start = p85.cpu(out), time.process_time()
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [v + ".npz" for v in fn.VARIANTS]
    path = out / "teachers/private_teachers.npz"
    probes = [
        p85.OUT / "analysis/private_new8_a00.npz",
        p85.OUT / "predictions/new8_a00.npz",
    ]
    denied = install(
        [SOURCE_MODEL, path],
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    source, t = old.arrays(SOURCE_MODEL), old.arrays(path)
    for name, key in zip(
        names, ("native_coordinates", "compatible_coordinates"), strict=True
    ):
        np.savez(dst / name, **fn.fit(source, t[key]))
        limit(start, used)
    ep.previous.finish(
        out,
        "models",
        names,
        start,
        source_solves=2,
        coefficients_each=20,
        extra_denials=len(denied) - len(probes),
        human_labels_read=False,
    )


def predict(out, folder):
    """Apply both matched functions with cached frames, then freeze predictions."""
    old.verify(out)
    ep.freeze(out, "models")
    used, start = p85.cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "new8"]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    reads = [
        p85.BASELINE,
        *p85.compiled_paths(),
        *[out / "models" / (v + ".npz") for v in fn.VARIANTS],
    ]
    reads += [
        root / (k + ".npz")
        for j in jobs
        for k in j["query_keys"]
        for root in (ep.STATE / "private_inputs", ep.OUT / "predictions")
    ]
    reads += [p85.OUT / "predictions" / n for n in names]
    probes = [
        p85.OUT / "analysis/private_new8_a00.npz",
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    baseline = old.arrays(p85.BASELINE)
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    compiler = p85.Compiler(16)
    fa, fd = [], []
    for j in jobs:
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        ds = [ep.body_directions(x) for x in data]
        b, db = fn.base.features(data, ds)
        x, dx = [fn.base.flatten(a, baseline["scale"], "grain_field") for a in (b, db)]
        saved = old.arrays(p85.OUT / "predictions" / (j["key"] + ".npz"))
        r, dr = saved["frame"], saved["frame_derivatives"]
        packets = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        primal, vectors = [
            np.stack([p[k] for p in packets]) for k in ("primal", "vectors")
        ]
        rr, dd, cs, cds, checks = [], [], [], [], []
        for m in ms:
            rv, rd, c, check = coord.readers(baseline, m, b, db, r, dr)
            c0, dc = fn.base.field(m, x, dx)
            np.testing.assert_array_equal(c, c0)
            rr.append(rv)
            dd.append(rd)
            cs.append(c)
            cds.append(dc)
            checks.append(check)
        rr, dd = map(np.stack, (rr, dd))
        pp = [
            p82.projections(primal, vectors, a, d) for a, d in zip(rr, dd, strict=True)
        ]
        result = {k: np.stack([p[k] for p in pp]) for k in pp[0]}
        if j["index"] == 0:
            for f in range(4):
                av, zv = [], []
                for step in (-EPS, EPS):
                    state = p85.changed(data, [d[f] for d in ds], step)
                    p = compiler(state, ds)
                    blocks = fn.base.features(state)
                    changed_r = [
                        coord.value_readers(
                            baseline, m, blocks, p["frame"] / m["units"][:, None]
                        )
                        for m in ms
                    ]
                    av.append(
                        np.stack(
                            [
                                np.einsum("qah,qmh->qam", p["primal"], ep.fn.readers(a))
                                for a in rr
                            ]
                        )
                    )
                    zv.append(
                        np.stack(
                            [
                                np.einsum("qah,qmh->qam", p["primal"], ep.fn.readers(a))
                                for a in changed_r
                            ]
                        )
                    )
                fa.extend(
                    abs(
                        (av[1] - av[0]) / (2 * EPS)
                        - result["actions"][:, :, :, 2, :, f]
                    ).ravel()
                )
                fd.extend(
                    abs(
                        (zv[1] - zv[0]) / (2 * EPS) - result["diagonal"][:, :, :, :, f]
                    ).ravel()
                )
        np.savez(
            dst / (j["key"] + ".npz"),
            **result,
            readers=rr,
            reader_derivatives=dd,
            coordinate_values=np.stack(cs),
            coordinate_derivatives=np.stack(cds),
            frame=r,
            frame_derivatives=dr,
            lift_checks=np.stack(checks),
        )
        limit(start, used)
    if len(fa) != 128 or len(fd) != 128 or max(fa + fd) > 1e-6:
        raise ValueError("Whole-function or anchored finite difference failed")
    ep.previous.finish(
        out,
        folder,
        names,
        start,
        compiled_calls=compiler.calls,
        anchored_fd_count=len(fa),
        diagonal_fd_count=len(fd),
        anchored_fd_max=float(max(fa)),
        diagonal_fd_max=float(max(fd)),
        native_reads=0,
        target_reads=0,
        extra_denials=len(denied) - len(probes),
    )


def summarize(rows, ms, baseline, prior_tables, programs):
    """Append all cells and retain both target definitions and signed errors."""
    z = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    p = {k: np.stack([r[k] for r in programs]) for k in programs[0]}
    sc, ac, fc, pr, vr, diag, sf, cond, cp, se = [[] for _ in TABLES]
    native_d = z["native_r"][:, 0] - z["native_r"][:, 1]
    rulers = {"scalar": z["native_primal"].mean(axis=1)}
    for ki, k in enumerate(ep.fn.COMPONENTS):
        for f in range(4):
            rulers[f"{k}_F{f + 1:02d}"] = z["native_vectors"].mean(axis=1)[:, ki, f]
    t = fn.teaching(
        np.concatenate((p["primal"][:, :, 3, None], p["vectors"][:, :, 3, 2]), axis=2),
        np.concatenate(
            (z["native_primal"][:, :, None], z["native_vectors"][:, :, 2]), axis=2
        ),
        z["native_r"],
        baseline["m"],
        baseline["d0"],
        ms[0]["units"],
    )
    np.testing.assert_allclose(t["frame"], z["frame"], atol=1e-12)
    for vi, (variant, m) in enumerate(zip(fn.VARIANTS, ms, strict=True)):
        meta = {"variant": variant, "cohort": "new8", "authors": 8}
        d = z["readers"][:, vi, 0] - z["readers"][:, vi, 1]
        error = d - native_d
        vr.append(
            meta
            | {
                "vector_error2": float(np.sum(error**2)),
                "error_over_native": float(np.sum(error**2) / np.sum(native_d**2)),
                "centered_error2": float(np.sum((error - error.mean(axis=0)) ** 2)),
            }
        )
        for name, ruler in rulers.items():
            y, pred = [np.einsum("nh,nh->n", a, ruler) for a in (native_d, d)]
            pr.append(meta | {"observable": name} | p82.metric(y, pred))
        cond.append(
            {
                "variant": variant,
                "authors": 4,
                "observations": 4,
                **{
                    k: float(m[k])
                    for k in (
                        "coefficients",
                        "regularizer",
                        "regularized_condition",
                        "normal_residual_relative",
                    )
                },
                "source_weight": float(m["weights"][0]),
            }
        )
        for f in range(5):
            name = "scalar" if f == 0 else f"total_F{f:02d}"
            sf.append(
                {
                    "variant": variant,
                    "authors": 4,
                    "coordinate": name,
                    **p82.metric(m["target"][:, f], m["fitted"][:, f]),
                }
            )
            for target in ("native_coordinates", "compatible_coordinates"):
                cp.append(
                    meta
                    | {"coordinate": name, "target": target}
                    | p82.metric(t[target][:, f], z["coordinate_values"][:, vi, f])
                )
        for ai, arm in enumerate(ep.fn.ARMS[:4]):
            for oi, obs in enumerate(("scalar", *ep.fn.COMPONENTS)):
                current = (
                    p["primal"][:, :, ai, None]
                    if oi == 0
                    else p["vectors"][:, :, ai, oi - 1]
                )
                native = (
                    z["native_primal"][:, :, None]
                    if oi == 0
                    else z["native_vectors"][:, :, oi - 1]
                )
                a = account.decompose(
                    current, native, z["readers"][:, vi], z["native_r"], baseline["m"]
                )
                se.append(meta | {"arm": arm, "observable": obs} | account.energies(a))
                expected = (
                    (z["scalar"][:, vi, 0, ai, 0] - z["scalar"][:, vi, 1, ai, 0])[
                        :, None
                    ]
                    if oi == 0
                    else z["actions"][:, vi, 0, ai, oi - 1, 0]
                    - z["actions"][:, vi, 1, ai, oi - 1, 0]
                )
                actual = (
                    (z["target_scalar"][:, 0, 0] - z["target_scalar"][:, 1, 0])[:, None]
                    if oi == 0
                    else z["target_actions"][:, 0, oi - 1, 0]
                    - z["target_actions"][:, 1, oi - 1, 0]
                )
                np.testing.assert_allclose(a["prediction"], expected, atol=1e-10)
                np.testing.assert_allclose(a["target"], actual, atol=1e-10)
        for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
            s, a, cor, dd = [
                p82.qview(z[k][:, vi], qi)
                for k in ("scalar", "actions", "correction", "diagonal")
            ]
            sy, ay = [p82.qview(z[k], qi) for k in ("target_scalar", "target_actions")]
            for ai, arm in enumerate(ep.fn.ARMS[:4]):
                for mi, mode in enumerate(ep.fn.MODES):
                    mm = meta | {"query": q, "arm": arm, "mode": mode}
                    sc.append(mm | p82.metric(sy[:, mi], s[:, ai, mi]))
                    for ki, k in enumerate(ep.fn.COMPONENTS):
                        cm = mm | {"component": k}
                        ac.append(cm | p82.metric(ay[:, ki, mi], a[:, ai, ki, mi]))
                        for f in range(4):
                            fc.append(
                                cm
                                | {"factor": f"F{f + 1:02d}"}
                                | p82.metric(ay[:, ki, mi, f], a[:, ai, ki, mi, f])
                            )
                    for f in range(4):
                        anchor, extra, diagonal = (
                            a[:, ai, 2, mi, f],
                            cor[:, ai, mi, f],
                            dd[:, ai, mi, f],
                        )
                        np.testing.assert_allclose(diagonal, anchor + extra, atol=1e-10)
                        diag.append(
                            mm
                            | {
                                "factor": f"F{f + 1:02d}",
                                "anchored_rms": float(np.sqrt(np.mean(anchor**2))),
                                "reader_correction_rms": float(
                                    np.sqrt(np.mean(extra**2))
                                ),
                                "diagonal_rms": float(np.sqrt(np.mean(diagonal**2))),
                                "signed_cross_energy": float(
                                    2 * np.mean(anchor * extra)
                                ),
                                "correction_mean": float(extra.mean()),
                            }
                        )
    core = [
        pd.concat((a, pd.DataFrame(b)), ignore_index=True)
        for a, b in zip(prior_tables, (sc, ac, fc, pr, vr, diag), strict=True)
    ]
    return [*core, *[pd.DataFrame(a) for a in (sf, cond, cp, se)]]


def evaluate(out, folder):
    """Open cached native arrays only after both new predictions are frozen."""
    old.verify(out)
    ep.freeze(out, folder)
    used, start = p85.cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    rows, names, programs = [], [], []
    for j in old.read(out / "private_jobs.json"):
        if j["cohort"] != "new8":
            continue
        n = "private_" + j["key"] + ".npz"
        row = old.arrays(out / folder / (j["key"] + ".npz")) | ep.selected(
            p85.OUT / "analysis" / n,
            (
                "native_r",
                "native_primal",
                "native_vectors",
                "target_scalar",
                "target_actions",
            ),
        )
        np.savez(dst / n, **row)
        rows.append(row)
        names.append(n)
        pair = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        programs.append(
            {k: np.stack([a[k] for a in pair]) for k in ("primal", "vectors")}
        )
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    tables = summarize(
        rows,
        ms,
        old.arrays(p85.BASELINE),
        [pd.read_csv(previous.OUT / "analysis" / n) for n in CORE],
        programs,
    )
    if [len(t) for t in tables] != COUNTS:
        raise ValueError("Finite table inventory differs")
    for n, t in zip(TABLES, tables, strict=True):
        t.to_csv(dst / n, index=False)
    limit(start, used)
    ep.previous.finish(out, dst.name, [*TABLES, *names], start, counts=COUNTS)


def audit(out):
    """Validate exact replay, matched source design, budgets and retained tables."""
    old.verify(out)
    stages = {
        f: ep.freeze(out, f)
        for f in (
            "teachers",
            "models",
            "predictions",
            "analysis",
            "replay",
            "replay_analysis",
        )
    }
    for left, right in (("predictions", "replay"), ("analysis", "replay_analysis")):
        a, b = stages[left], stages[right]
        if set(a["files"]) != set(b["files"]):
            raise ValueError("Replay inventory differs")
        for n in a["files"]:
            if n.endswith(".npz"):
                x, y = [old.arrays(out / f / n) for f in (left, right)]
                if set(x) != set(y):
                    raise ValueError("Replay keys differ")
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k])
            elif a["files"][n] != b["files"][n]:
                raise ValueError("CSV replay differs")
    for n in CORE:
        a, b = [pd.read_csv(root / "analysis" / n) for root in (previous.OUT, out)]
        pd.testing.assert_frame_equal(
            a,
            b.iloc[: len(a)].reset_index(drop=True),
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    for f in ("teachers", "models", "predictions", "replay"):
        if stages[f]["extra_denials"]:
            raise ValueError("Unexpected file reads")
    if (
        stages["teachers"]["compiled_query_calls"] != 8
        or stages["models"]["source_solves"] != 2
        or stages["analysis"]["counts"] != COUNTS
        or p85.cpu(out) > CAP
    ):
        raise ValueError("Frozen budget or inventory differs")
    source = old.arrays(SOURCE_MODEL)
    for v in fn.VARIANTS:
        m = old.arrays(out / "models" / (v + ".npz"))
        for k in ("x", "v", "kinds", "weights"):
            np.testing.assert_array_equal(m[k], source[k][fn.ORIGINAL])
        np.testing.assert_array_equal(m["units"], source["units"])
        assert m["regularizer"] == source["regularizer"]
    for f in ("predictions", "replay"):
        p = stages[f]
        if p["compiled_calls"] != 16 or p["native_reads"] or p["target_reads"]:
            raise ValueError("Application contract differs")
    result = {
        "status": "PASS",
        "source_authors": 4,
        "recipient_authors": 8,
        "source_solves": 2,
        "coefficients_each": 20,
        "native_calls": 0,
        "source_compiled_query_calls": 8,
        "native_cached_query_reconstructions": 8,
        "local_compiled_calls_each_application": 16,
        "human_labels_read": False,
        "numeric_cpu_seconds": p85.cpu(out),
        "public_tables": len(TABLES),
        **{
            k: stages["predictions"][k]
            for k in (
                "anchored_fd_count",
                "diagonal_fd_count",
                "anchored_fd_max",
                "diagonal_fd_max",
            )
        },
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run one immutable stage in a fresh process, with one numerical thread."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "teachers",
            "models",
            "predict",
            "evaluate",
            "replay",
            "replay-evaluate",
            "audit",
        ),
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
            {
                "prepare": prepare,
                "teachers": teachers,
                "models": models,
                "audit": audit,
            }[args.stage](args.output)


if __name__ == "__main__":
    main()
