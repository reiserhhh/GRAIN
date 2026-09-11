#!/usr/bin/env python3
"""NEXT86: nested source teaching functions, same-material application and audit."""

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
from grain_research.vector_mechanisms import conditional_teaching as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_evidence_coordinate_reader as previous

old, ep, p82, coord = previous.old, previous.ep, previous.p82, previous.fn
OUT = old.BASE / "logic_conditional_teaching_v1"
PLAN = ROOT / "docs/GRAIN_CONDITIONAL_TEACHING_INNOVATION_PLAN_V1.md"
CAP, EPS = 120, previous.EPS
CORE = (
    "component_scores.csv",
    "component_actions.csv",
    "factor_actions.csv",
    "reader_projection.csv",
    "reader_vector.csv",
    "diagonal_actions.csv",
)
TABLES = (
    *CORE,
    "source_loss.csv",
    "conditioning.csv",
    "coordinate_prediction.csv",
    "coordinate_innovations.csv",
    "function_innovations.csv",
)
COUNTS = [432, 1296, 5184, 234, 18, 1728, 18, 6, 30, 20, 384]


def cpu(out):
    """Canonical numerical stage time, including exact application replay."""
    return previous.cpu(out)


def limit(start, used):
    """Stop by fixed compute budget rather than observed evaluation quality."""
    if used + time.process_time() - start > CAP:
        raise TimeoutError("NEXT86 numerical CPU cap reached")


def prepare(out):
    """Pin the prior immutable source/application packages and this experiment."""
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT85 required")
    inputs = dict(m["inputs"])
    for folder in ("teachers", "models", "predictions", "analysis"):
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
        ROOT / "tests/test_grain_conditional_teaching.py",
        ROOT / "scripts/report_grain_conditional_teaching.py",
    ):
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    start = time.process_time()
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", old.read(previous.OUT / "private_jobs.json"))
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONDITIONAL_TEACHING_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "prepare_cpu_seconds": time.process_time() - start,
            "source_authors": 4,
            "recipient_authors": 8,
            "max_direct_source_solves": 6,
            "max_cpu_seconds": CAP,
            "assimilation_order": ["value", "clamped_BODY", "original_context"],
            "weights": "exact frozen NEXT85 weights, never renormalized per subset",
            "native_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED"}, flush=True)


def models(out):
    """Construct six nested source functions and verify each by a direct solve."""
    old.verify(out)
    used, start = cpu(out), time.process_time()
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [v + ".npz" for v in fn.VARIANTS]
    paths = [previous.OUT / "models" / (v + ".npz") for v in coord.VARIANTS]
    probes = [
        previous.OUT / "predictions/new8_a00.npz",
        previous.OUT / "analysis/private_new8_a00.npz",
    ]
    denied = install(
        paths,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    parity = []
    for fi, path in enumerate(paths):
        source = old.arrays(path)
        ms = fn.nested_models(source)
        gap = float(
            np.linalg.norm(ms[-1]["beta"] - source["beta"])
            / max(np.linalg.norm(source["beta"]), np.finfo(float).tiny)
        )
        if gap > 1e-7:
            raise ValueError("Full model differs from frozen NEXT85")
        parity.append(gap)
        for si, m in enumerate(ms):
            m["full_beta_relative"] = np.asarray(gap)
            np.savez(dst / names[fi * 3 + si], **m)
            limit(start, used)
    ep.previous.finish(
        out,
        "models",
        names,
        start,
        direct_source_solves=6,
        conditional_stages=6,
        full_beta_relative_max=max(parity),
        extra_denials=len(denied) - len(probes),
        human_labels_read=False,
    )


def predict(out, folder):
    """Apply source-only functions using cached H16 and frames, then freeze."""
    old.verify(out)
    ep.freeze(out, "models")
    used, start = cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "new8"]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    reads = [
        previous.BASELINE,
        *previous.compiled_paths(),
        *[out / "models" / (v + ".npz") for v in fn.VARIANTS],
    ]
    reads += [
        root / (k + ".npz")
        for j in jobs
        for k in j["query_keys"]
        for root in (ep.STATE / "private_inputs", ep.OUT / "predictions")
    ]
    reads += [previous.OUT / "predictions" / n for n in names]
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
    baseline = old.arrays(previous.BASELINE)
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    compiler = previous.Compiler(16)
    fa, fd, parity, coordinate_parity = [], [], [], []
    for j in jobs:
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        ds = [ep.body_directions(x) for x in data]
        b, db = fn.base.features(data, ds)
        x, dx = [fn.base.flatten(t, baseline["scale"], "grain_field") for t in (b, db)]
        saved = old.arrays(previous.OUT / "predictions" / (j["key"] + ".npz"))
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
        rr, dd, cs = [np.stack(a) for a in (rr, dd, cs)]
        values = [
            p82.projections(primal, vectors, a, d) for a, d in zip(rr, dd, strict=True)
        ]
        result = {k: np.stack([p[k] for p in values]) for k in values[0]}
        for ni, oi, ci in ((2, 10, 0), (5, 11, 1)):
            for k in result:
                parity.append(float(abs(result[k][ni] - saved[k][oi]).max()))
            for current, key in ((rr, "readers"), (dd, "reader_derivatives")):
                parity.append(float(abs(current[ni] - saved[key][oi]).max()))
            coordinate_parity.append(
                float(abs(cs[ni] - saved["coordinate_values"][ci]).max())
            )
        row = fn.base.gram(
            x[None],
            np.zeros_like(x)[None],
            [False],
            ms[0]["x"],
            ms[0]["v"],
            ms[0]["kinds"],
        )
        parts = np.stack(
            [
                np.stack(
                    [
                        (row[:, m["groups"] == g] @ m["beta"][m["groups"] == g])[0]
                        for g in range(3)
                    ]
                )
                for m in (ms[2], ms[5])
            ]
        )
        np.testing.assert_allclose(parts.sum(axis=1), cs[[2, 5]], atol=1e-10)
        if j["index"] == 0:
            for f in range(4):
                aa, zz = [], []
                for step in (-EPS, EPS):
                    state = previous.changed(data, [d[f] for d in ds], step)
                    p = compiler(state, ds)
                    blocks = fn.base.features(state)
                    av, zv = [], []
                    for vi, m in enumerate(ms):
                        changed_r = coord.value_readers(
                            baseline, m, blocks, p["frame"] / m["units"][:, None]
                        )
                        av.append(
                            np.einsum(
                                "qah,qmh->qam", p["primal"], ep.fn.readers(rr[vi])
                            )
                        )
                        zv.append(
                            np.einsum(
                                "qah,qmh->qam", p["primal"], ep.fn.readers(changed_r)
                            )
                        )
                    aa.append(np.stack(av))
                    zz.append(np.stack(zv))
                fa.extend(
                    abs(
                        (aa[1] - aa[0]) / (2 * EPS)
                        - result["actions"][:, :, :, 2, :, f]
                    ).ravel()
                )
                fd.extend(
                    abs(
                        (zz[1] - zz[0]) / (2 * EPS) - result["diagonal"][:, :, :, :, f]
                    ).ravel()
                )
        np.savez(
            dst / (j["key"] + ".npz"),
            **result,
            readers=rr,
            reader_derivatives=dd,
            coordinate_values=cs,
            coordinate_derivatives=np.stack(cds),
            raw_group_parts=parts,
            frame=r,
            frame_derivatives=dr,
            lift_checks=np.stack(checks),
        )
        limit(start, used)
    if (
        len(fa) != 384
        or len(fd) != 384
        or max(fa + fd) > 1e-6
        or max(parity) > 1e-8
        or max(coordinate_parity) > 1e-8
    ):
        raise ValueError("Nested function FD or frozen full-reader parity failed")
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
        full_function_parity_max=max(parity),
        coordinate_parity_max=max(coordinate_parity),
        native_reads=0,
        target_reads=0,
        extra_denials=len(denied) - len(probes),
    )


def summarize(rows, models, baseline, prior_tables):
    """Compute every declared cell without selecting a function from new8 outcomes."""
    z = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    sc, ac, fc, pr, vr, diag, losses, conditions, cp, ci, fi = [[] for _ in TABLES]
    native = z["native_r"][:, 0] - z["native_r"][:, 1]
    rulers = {"scalar": z["native_primal"].mean(axis=1)}
    for component, name in enumerate(ep.fn.COMPONENTS):
        for f in range(4):
            rulers[f"{name}_F{f + 1:02d}"] = z["native_vectors"].mean(axis=1)[
                :, component, f
            ]
    for vi, (variant, m) in enumerate(zip(fn.VARIANTS, models, strict=True)):
        frame, level = divmod(vi, 3)
        meta = {"variant": variant, "cohort": "new8", "authors": 8}
        d = z["readers"][:, vi, 0] - z["readers"][:, vi, 1]
        error = d - native
        vr.append(
            meta
            | {
                "vector_error2": float(np.sum(error**2)),
                "error_over_native": float(np.sum(error**2) / np.sum(native**2)),
                "centered_error2": float(np.sum((error - error.mean(axis=0)) ** 2)),
            }
        )
        for name, ruler in rulers.items():
            y, p = [np.einsum("ih,ih->i", a, ruler) for a in (native, d)]
            pr.append(meta | {"observable": name} | p82.metric(y, p))
        for gi, group in enumerate(fn.base.GROUPS):
            losses.append(
                {
                    "variant": variant,
                    "authors": 4,
                    "group": group,
                    "observations": int(np.sum(m["groups"] == gi)),
                    "included_in_fit": gi <= level,
                    "residual_sse_over_target": float(m["group_loss"][gi]),
                    "zero_target_group": bool(m["zero_target_groups"][gi]),
                }
            )
        conditions.append(
            {
                "variant": variant,
                "authors": 4,
                "observations": int(m["selected"].sum()),
                "coefficients": int(m["coefficients"]),
                **{
                    k: float(m[k])
                    for k in (
                        "schur_min_eigenvalue",
                        "schur_condition",
                        "direct_relative",
                        "normal_residual_relative",
                        "full_beta_relative",
                        "regularizer",
                        "regularization_penalty",
                        "innovation_target_rms",
                    )
                },
            }
        )
        r = z["frame"] if frame else np.broadcast_to(m["fixed_frame"], z["frame"].shape)
        actual = np.einsum("nkh,nh->nk", r, native - baseline["d0"])
        pred = z["coordinate_values"][:, vi]
        increment = pred if level == 0 else pred - z["coordinate_values"][:, vi - 1]
        for f in range(5):
            name = "scalar" if f == 0 else f"total_F{f:02d}"
            raw = z["raw_group_parts"][:, frame, level, f]
            cp.append(
                {
                    "variant": variant,
                    "authors": 8,
                    "coordinate": name,
                    **p82.metric(actual[:, f], pred[:, f]),
                    "raw_group_rms": float(np.sqrt(np.mean(raw**2))),
                    "conditional_increment_rms": float(
                        np.sqrt(np.mean(increment[:, f] ** 2))
                    ),
                }
            )
            if level:
                ci.append(
                    {
                        "frame": fn.FRAMES[frame],
                        "authors": 8,
                        "coordinate": name,
                        "update": f"{fn.LEVELS[level - 1]}_to_{fn.LEVELS[level]}",
                        "raw_added_group_rms": float(np.sqrt(np.mean(raw**2))),
                        **fn.error_change(
                            actual[:, f],
                            z["coordinate_values"][:, vi - 1, f],
                            pred[:, f],
                        ),
                    }
                )
        for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
            s, a, cor, dd = [
                p82.qview(z[k][:, vi], qi)
                for k in ("scalar", "actions", "correction", "diagonal")
            ]
            sy, ay = [p82.qview(z[k], qi) for k in ("target_scalar", "target_actions")]
            if level:
                before_s, before_a = [
                    p82.qview(z[k][:, vi - 1], qi) for k in ("scalar", "actions")
                ]
            for ai, arm in enumerate(ep.fn.ARMS[:4]):
                for mi, mode in enumerate(ep.fn.MODES):
                    mm = meta | {"query": q, "arm": arm, "mode": mode}
                    sc.append(mm | p82.metric(sy[:, mi], s[:, ai, mi]))
                    if level:
                        im = {
                            "frame": fn.FRAMES[frame],
                            "authors": 8,
                            "query": q,
                            "arm": arm,
                            "mode": mode,
                            "update": f"{fn.LEVELS[level - 1]}_to_{fn.LEVELS[level]}",
                        }
                        fi.append(
                            im
                            | {"observable": "scalar"}
                            | fn.error_change(
                                sy[:, mi], before_s[:, ai, mi], s[:, ai, mi]
                            )
                        )
                    for component, name in enumerate(ep.fn.COMPONENTS):
                        cm = mm | {"component": name}
                        ac.append(
                            cm
                            | p82.metric(ay[:, component, mi], a[:, ai, component, mi])
                        )
                        if level:
                            fi.append(
                                im
                                | {"observable": name}
                                | fn.error_change(
                                    ay[:, component, mi],
                                    before_a[:, ai, component, mi],
                                    a[:, ai, component, mi],
                                )
                            )
                        for f in range(4):
                            fc.append(
                                cm
                                | {"factor": f"F{f + 1:02d}"}
                                | p82.metric(
                                    ay[:, component, mi, f], a[:, ai, component, mi, f]
                                )
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
        pd.concat((old_table, pd.DataFrame(extra)), ignore_index=True)
        for old_table, extra in zip(
            prior_tables, (sc, ac, fc, pr, vr, diag), strict=True
        )
    ]
    return [*core, *[pd.DataFrame(t) for t in (losses, conditions, cp, ci, fi)]]


def evaluate(out, folder):
    """Open old evaluation-only native arrays after the new predictions freeze."""
    old.verify(out)
    ep.freeze(out, folder)
    used, start = cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    rows, names = [], []
    for j in old.read(out / "private_jobs.json"):
        if j["cohort"] != "new8":
            continue
        n = "private_" + j["key"] + ".npz"
        row = old.arrays(out / folder / (j["key"] + ".npz")) | ep.selected(
            previous.OUT / "analysis" / n,
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
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    tables = summarize(
        rows,
        ms,
        old.arrays(previous.BASELINE),
        [pd.read_csv(previous.OUT / "analysis" / n) for n in CORE],
    )
    if [len(t) for t in tables] != COUNTS:
        raise ValueError("Finite comparison inventory differs")
    for n, t in zip(TABLES, tables, strict=True):
        t.to_csv(dst / n, index=False)
    limit(start, used)
    ep.previous.finish(out, dst.name, [*TABLES, *names], start, counts=COUNTS)


def audit(out):
    """Require exact replay, fixed budgets, prior comparisons and zero target reads."""
    old.verify(out)
    stages = {
        f: ep.freeze(out, f)
        for f in ("models", "predictions", "analysis", "replay", "replay_analysis")
    }
    for left, right in (("predictions", "replay"), ("analysis", "replay_analysis")):
        a, b = stages[left], stages[right]
        if set(a["files"]) != set(b["files"]):
            raise ValueError("Replay file inventory differs")
        for n in a["files"]:
            if n.endswith(".npz"):
                x, y = [old.arrays(out / f / n) for f in (left, right)]
                if set(x) != set(y):
                    raise ValueError("Replay array keys differ")
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
    if (
        stages["models"]["direct_source_solves"] != 6
        or stages["models"]["conditional_stages"] != 6
        or stages["analysis"]["counts"] != COUNTS
        or cpu(out) > CAP
    ):
        raise ValueError("Frozen budget or inventory differs")
    for f in ("models", "predictions", "replay"):
        if stages[f]["extra_denials"]:
            raise ValueError("Unexpected stage reads")
    for f in ("predictions", "replay"):
        p = stages[f]
        if p["compiled_calls"] != 16 or p["native_reads"] or p["target_reads"]:
            raise ValueError("Application input contract differs")
    p = stages["predictions"]
    result = {
        "status": "PASS",
        "source_authors": 4,
        "recipient_authors": 8,
        "nested_functions": 6,
        "direct_source_solves": 6,
        "native_calls": 0,
        "human_labels_read": False,
        "numeric_cpu_seconds": cpu(out),
        "public_tables": len(TABLES),
        "local_compiled_calls_each": 16,
        **{
            k: p[k]
            for k in (
                "anchored_fd_count",
                "diagonal_fd_count",
                "anchored_fd_max",
                "diagonal_fd_max",
                "full_function_parity_max",
                "coordinate_parity_max",
            )
        },
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run one immutable stage per process with one numerical thread."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
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
            {"prepare": prepare, "models": models, "audit": audit}[args.stage](
                args.output
            )


if __name__ == "__main__":
    main()
