#!/usr/bin/env python3
"""NEXT88 numerical application, matched evaluation and exact replay checks."""

import time

import numpy as np
import pandas as pd

from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_diversity_teaching as run

old, ep, p85, prior = run.old, run.ep, run.p85, run.previous
fn, coord, p82 = run.fn, prior.coord, prior.p82
base = fn.previous.base


def predict(out, folder):
    """Apply both12-source readers without recipient native targets or capture."""
    old.verify(out)
    ep.freeze(out, "models")
    used, start = p85.cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "new8"]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    reads = [p85.BASELINE, *p85.compiled_paths()]
    reads += [out / "models" / (v + ".npz") for v in fn.VARIANTS]
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
        out / "capture/a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [run.ROOT / "scripts", run.ROOT / "grain_research"],
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
        b, db = base.features(data, ds)
        x, dx = [base.flatten(a, baseline["scale"], "grain_field") for a in (b, db)]
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
            c0, dc = base.field(m, x, dx)
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
                for step in (-prior.EPS, prior.EPS):
                    state = p85.changed(data, [d[f] for d in ds], step)
                    p = compiler(state, ds)
                    blocks = base.features(state)
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
                        (av[1] - av[0]) / (2 * prior.EPS)
                        - result["actions"][:, :, :, 2, :, f]
                    ).ravel()
                )
                fd.extend(
                    abs(
                        (zv[1] - zv[0]) / (2 * prior.EPS)
                        - result["diagonal"][:, :, :, :, f]
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
        run.limit(start, used)
    if len(fa) != 128 or len(fd) != 128 or max(fa + fd) > 1e-6:
        raise ValueError("Anchored or whole-function finite difference failed")
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


def adapt_tables(tables, old_counts):
    """Relabel only newly appended matched rows; preserve all20 prior readers."""
    names = dict(zip(prior.fn.VARIANTS, fn.VARIANTS, strict=True))
    for i, t in enumerate(tables):
        start = old_counts[i] if i < len(run.CORE) else 0
        t.loc[t.index[start:], "variant"] = t.iloc[start:].variant.replace(names)
    tables[6]["authors"] = 12
    tables[7]["authors"] = 12
    tables[7]["observations"] = 12
    return tables


def summarize(rows, models, baseline, old_tables, programs):
    """Use the unchanged NEXT87 metric definitions, with source12 metadata."""
    return adapt_tables(
        prior.summarize(rows, models, baseline, old_tables, programs),
        [len(t) for t in old_tables],
    )


def evaluate(out, folder):
    """Evaluate on the same cached eight authors only after prediction freeze."""
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
        [pd.read_csv(prior.OUT / "analysis" / n) for n in run.CORE],
        programs,
    )
    if [len(t) for t in tables] != run.COUNTS:
        raise ValueError("Finite table inventory differs")
    for n, t in zip(run.TABLES, tables, strict=True):
        t.to_csv(dst / n, index=False)
    run.limit(start, used)
    ep.previous.finish(out, dst.name, [*run.TABLES, *names], start, counts=run.COUNTS)


def audit(out):
    """Audit completed source design, fixed model contract and exact replay."""
    manifest = old.verify(out)
    stages = {
        f: ep.freeze(out, f)
        for f in (
            "capture",
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
    for n in run.CORE:
        a, b = [pd.read_csv(root / "analysis" / n) for root in (prior.OUT, out)]
        pd.testing.assert_frame_equal(
            a,
            b.iloc[: len(a)].reset_index(drop=True),
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    for f in ("teachers", "models", "predictions", "replay"):
        if stages[f]["extra_denials"]:
            raise ValueError("Unexpected file access")
    cap = stages["capture"]
    if cap["complete_source_authors"] != 8 or cap["native_seconds"] > run.NATIVE_CAP:
        raise ValueError("Incomplete or over-budget source capture")
    for k, count in manifest["native_limits"].items():
        if cap[k] != count:
            raise ValueError("Native call inventory differs")
    if (
        stages["teachers"]["compiled_query_calls"] != 16
        or stages["models"]["source_solves"] != 2
        or stages["analysis"]["counts"] != run.COUNTS
        or p85.cpu(out) > run.CAP
    ):
        raise ValueError("Fixed numerical contract differs")
    source, teach = (
        old.arrays(prior.SOURCE_MODEL),
        old.arrays(out / "teachers/private_teachers.npz"),
    )
    for v, target in zip(
        fn.VARIANTS, ("native_coordinates", "compatible_coordinates"), strict=True
    ):
        m = old.arrays(out / "models" / (v + ".npz"))
        np.testing.assert_array_equal(m["x"], teach["points"])
        np.testing.assert_array_equal(m["target"], teach[target])
        np.testing.assert_array_equal(
            m["weights"], np.repeat(source["weights"][fn.previous.ORIGINAL[0]], 12)
        )
        np.testing.assert_array_equal(m["units"], source["units"])
        np.testing.assert_array_equal(m["fixed_frame"], source["fixed_frame"])
        if m["regularizer"] != source["regularizer"] or m["coefficients"] != 60:
            raise ValueError("Source parameter contract differs")
    for f in ("predictions", "replay"):
        p = stages[f]
        if p["compiled_calls"] != 16 or p["native_reads"] or p["target_reads"]:
            raise ValueError("Application contract differs")
    result = {
        "status": "PASS",
        "source_authors": 12,
        "additional_sources": 8,
        "recipient_authors": 8,
        "source_solves": 2,
        "coefficients_each": 60,
        "human_labels_read": False,
        "numeric_cpu_seconds": p85.cpu(out),
        "native_seconds": cap["native_seconds"],
        "native_calls": {k: cap[k] for k in manifest["native_limits"]},
        "source_compiled_query_calls": 16,
        "compiled_calls_including_replay": 48,
        "public_tables": len(run.TABLES),
        "prior_readers_retained": 20,
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
