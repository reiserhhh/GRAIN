#!/usr/bin/env python3
"""NEXT83: source-functional reader teaching, immutable application and range account."""

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
from grain_research.vector_mechanisms import observable_reader_teaching as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_material_reader_field as previous

old, ep, prior = previous.old, previous.ep, previous.prior
OUT = old.BASE / "logic_observable_reader_teaching_v1"
PLAN = ROOT / "docs/GRAIN_OBSERVABLE_READER_TEACHING_PLAN_V1.md"
VARIANTS = (*previous.VARIANTS, *fn.VARIANTS)
TABLES = (
    "source_fit.csv",
    "source_geometry.csv",
    "component_scores.csv",
    "component_actions.csv",
    "factor_actions.csv",
    "reader_projection.csv",
    "reader_vector.csv",
    "diagonal_actions.csv",
    "range_vector.csv",
    "range_projection.csv",
)
COUNTS = [18, 2, 192, 576, 2304, 104, 8, 768, 2, 26]
CAP = 90


def limit(start, used):
    """Bound cumulative numerical work without using recipient outcomes to stop."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("NEXT83 cumulative numerical CPU cap")


def observation_authors(groups):
    """Bind the frozen mixed-observation layout to its four source authors."""
    np.testing.assert_array_equal(groups, np.repeat(np.arange(3), [8, 16, 4]))
    return np.concatenate(
        (np.repeat(np.arange(4), 2), np.repeat(np.arange(4), 4), np.arange(4))
    )


def prepare(out):
    """Pin upstream teachers, prediction targets and code before either source fit."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT82 required")
    out.mkdir(exist_ok=False)
    jobs = old.read(previous.OUT / "private_jobs.json")
    old.write(out / "private_jobs.json", jobs)
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
        ROOT / "tests/test_grain_observable_reader_teaching.py",
    ):
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_OBSERVABLE_READER_TEACHING_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "prepare_cpu_seconds": time.process_time() - start,
            "source_authors": 4,
            "recipient_authors": 8,
            "observations": 28,
            "source_rulers_per_author": 5,
            "max_fits": 2,
            "max_cpu_seconds": CAP,
            "native_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED"}, flush=True)


def models(out):
    """Teach two matched fields using only source reader targets and original rulers."""
    old.verify(out)
    used, start = prior.prior_cpu(out), time.process_time()
    src = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "source4"]
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [v + ".npz" for v in fn.VARIANTS]
    model_path = previous.OUT / "models/grain_field.npz"
    reads = [
        model_path,
        ep.STATE / "private_weights.npz",
        ep.CAPTURE / "private_weights.npz",
    ]
    reads += [
        root / (k + ".npz")
        for j in src
        for k in j["query_keys"]
        for root in (ep.CAPTURE / "private_capture", ep.SOURCE / "private_inputs")
    ]
    reads += [
        ep.SELECTOR / "models" / f"{prior.reference(j)}_{q}.npz"
        for j in src
        for q in old.QUERIES
    ]
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
    m = old.arrays(model_path)
    weights, qw = [
        old.arrays(r / "private_weights.npz") for r in (ep.STATE, ep.CAPTURE)
    ]
    rulers = []
    for i, j in enumerate(src):
        if j["index"] != i:
            raise ValueError("Source order changed")
        native = previous.motion.previous.previous.native_packets(j, weights, qw)
        p = native["primal"][:, 1].mean(axis=0)
        b = native["vectors"][:, 1, 2].mean(axis=0)
        rulers.append(np.concatenate((p[None], b)))
    rulers = np.stack(rulers)[observation_authors(m["groups"])]
    for i, name in enumerate(names):
        result = fn.fit(m, rulers, bool(i))
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


def predict(out, folder):
    """Read source packages and H16 only, preserving all six earlier comparators."""
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
        b, db = fn.base.features(data, ds)
        old_pred = old.arrays(previous.OUT / "predictions" / (j["key"] + ".npz"))
        packet = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        primal, vectors = [
            np.stack([p[k] for p in packet]) for k in ("primal", "vectors")
        ]
        rr, dr = list(old_pred["readers"]), list(old_pred["reader_derivatives"])
        for m in ms:
            r, d = fn.base.readers(m, b, db, m["scale"], "grain_field")
            rr.append(r)
            dr.append(d)
        rr, dr = np.stack(rr), np.stack(dr)
        ps = [
            previous.projections(primal, vectors, r, d)
            for r, d in zip(rr, dr, strict=True)
        ]
        result = {k: np.stack([p[k] for p in ps]) for k in ps[0]}
        for k in result:
            comparator.append(float(abs(result[k][:6] - old_pred[k]).max()))
            parity.append(float(abs(result[k][6] - old_pred[k][5]).max()))
        if j["index"] == 0:
            eps = 1e-5
            for f in range(4):
                av, dv = [], []
                for e in (-eps, eps):
                    changed = [
                        dict(x, state=x["state"] + e * d[f])
                        for x, d in zip(data, ds, strict=True)
                    ]
                    cb, cdb = fn.base.features(changed, ds)
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
                                ep.fn.readers(rr[0])[qi],
                            )["primal"]
                        )
                    pp = np.stack(pp)
                    aa, dd = [], []
                    for vi, m in enumerate(ms, 6):
                        cr, _ = fn.base.readers(m, cb, cdb, m["scale"], "grain_field")
                        aa.append(np.einsum("qah,qmh->qam", pp, ep.fn.readers(rr[vi])))
                        dd.append(np.einsum("qah,qmh->qam", pp, ep.fn.readers(cr)))
                    av.append(np.stack(aa))
                    dv.append(np.stack(dd))
                fa.extend(
                    abs(
                        (av[1] - av[0]) / (2 * eps)
                        - result["actions"][6:, :, :, 2, :, f]
                    ).ravel()
                )
                fd.extend(
                    abs(
                        (dv[1] - dv[0]) / (2 * eps) - result["diagonal"][6:, :, :, :, f]
                    ).ravel()
                )
        np.savez(dst / (j["key"] + ".npz"), **result, readers=rr, reader_derivatives=dr)
        limit(start, used)
    if (
        len(fa) != 128
        or len(fd) != 128
        or max(fa + fd) > 1e-6
        or max(parity) > 1e-8
        or max(comparator) > 1e-10
    ):
        raise ValueError("Finite difference or fixed-control reproduction failed")
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


def summarize(rows, models):
    """Summarize eight fixed fields at the author level, retaining all factors."""
    z = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    fit, geo, sc, ac, fc, pr, vr, diag, rv, rp = [[] for _ in TABLES]
    for variant, m in zip(fn.VARIANTS, models, strict=True):
        for li, loss in enumerate(fn.LOSSES):
            for gi, group in enumerate(fn.base.GROUPS):
                fit.append(
                    {
                        "variant": variant,
                        "authors": 4,
                        "role": "source_training",
                        "loss": loss,
                        "group": group,
                        "observations": int(np.sum(m["groups"] == gi)),
                        "residual_sse_over_target": float(m["source_losses"][li, gi]),
                        "zero_target_group": bool(m["source_zero_energy"][li, gi]),
                    }
                )
        geo.append(
            {
                "variant": variant,
                "authors": 4,
                "observations": 28,
                "output_rank": m["output_basis"].shape[1],
                "kernel_rank": m["a"].shape[1],
                "coefficients": int(m["coefficient_count"]),
                "normal_residual_relative": float(m["normal_residual_relative"]),
                "prior_beta_relative": float(m["prior_beta_relative"]),
                "kernel_reconstruction_relative": float(
                    m["kernel_reconstruction_relative"]
                ),
                "target_outside_relative": float(m["target_outside_relative"]),
                "regularizer": float(m["regularizer"]),
                "regularization_penalty": float(m["regularization_penalty"]),
            }
        )
    native_d = z["native_r"][:, 0] - z["native_r"][:, 1]
    rulers = {"scalar": z["native_primal"].mean(axis=1)}
    for ci, c in enumerate(ep.fn.COMPONENTS):
        for f in range(4):
            rulers[f"{c}_F{f + 1:02d}"] = z["native_vectors"].mean(axis=1)[:, ci, f]
    for vi, variant in enumerate(VARIANTS):
        meta = {"variant": variant, "cohort": "new8", "authors": 8}
        pred_d = z["readers"][:, vi, 0] - z["readers"][:, vi, 1]
        error = pred_d - native_d
        vr.append(
            meta
            | {
                "vector_error2": float(np.sum(error**2)),
                "error_over_native": float(np.sum(error**2) / np.sum(native_d**2)),
                "centered_error2": float(np.sum((error - error.mean(axis=0)) ** 2)),
            }
        )
        for name, ruler in rulers.items():
            y, p = [np.einsum("ih,ih->i", d, ruler) for d in (native_d, pred_d)]
            pr.append(meta | {"observable": name} | previous.metric(y, p))
        if vi >= 6:
            m = models[vi - 6]
            inside, outside = fn.split_error(
                pred_d, native_d, m["d0"], m["output_basis"]
            )
            rv.append(meta | fn.error_account(inside, outside))
            for name, ruler in rulers.items():
                a, b = [np.einsum("ih,ih->i", e, ruler) for e in (inside, outside)]
                rp.append(meta | {"observable": name} | fn.error_account(a, b))
        for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
            s, a, cor, dd = [
                previous.qview(z[k][:, vi], qi)
                for k in ("scalar", "actions", "correction", "diagonal")
            ]
            sy, ay = [
                previous.qview(z[k], qi) for k in ("target_scalar", "target_actions")
            ]
            for ai, arm in enumerate(ep.fn.ARMS[:4]):
                for mi, mode in enumerate(ep.fn.MODES):
                    mm = meta | {"query": q, "arm": arm, "mode": mode}
                    sc.append(mm | previous.metric(sy[:, mi], s[:, ai, mi]))
                    for ci, c in enumerate(ep.fn.COMPONENTS):
                        cm = mm | {"component": c}
                        ac.append(cm | previous.metric(ay[:, ci, mi], a[:, ai, ci, mi]))
                        for f in range(4):
                            fc.append(
                                cm
                                | {"factor": f"F{f + 1:02d}"}
                                | previous.metric(ay[:, ci, mi, f], a[:, ai, ci, mi, f])
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
    return [pd.DataFrame(t) for t in (fit, geo, sc, ac, fc, pr, vr, diag, rv, rp)]


def evaluate(out, folder):
    """Open exact old target arrays only after freezing both new applications."""
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
        p = old.arrays(out / folder / (j["key"] + ".npz"))
        t = ep.selected(
            previous.OUT / "analysis" / name,
            (
                "native_r",
                "native_primal",
                "native_vectors",
                "target_scalar",
                "target_actions",
            ),
        )
        row = p | t
        np.savez(dst / name, **row)
        rows.append(row)
        names.append(name)
        limit(start, used)
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    tables = summarize(rows, ms)
    if [len(t) for t in tables] != COUNTS:
        raise ValueError("Comparison inventory incomplete")
    for name, t in zip(TABLES, tables, strict=True):
        t.to_csv(dst / name, index=False)
    limit(start, used)
    ep.previous.finish(out, dst.name, [*TABLES, *names], start, counts=COUNTS)


def audit(out):
    """Verify stage manifests, read boundaries and exact deterministic replay."""
    old.verify(out)
    m = ep.freeze(out, "models")
    stages = {
        f: ep.freeze(out, f)
        for f in ("predictions", "analysis", "replay", "replay_analysis")
    }
    for left, right in (("predictions", "replay"), ("analysis", "replay_analysis")):
        a, b = stages[left], stages[right]
        if set(a["files"]) != set(b["files"]):
            raise ValueError("Replay file inventory differs")
        for name in a["files"]:
            if name.endswith(".npz"):
                x, y = [old.arrays(out / f / name) for f in (left, right)]
                if set(x) != set(y):
                    raise ValueError("Replay array inventory differs")
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k])
            elif a["files"][name] != b["files"][name]:
                raise ValueError("CSV replay differs")
    p = stages["predictions"]
    if (
        len(m["files"]) != 2
        or m["coefficient_solves"] != 2
        or m["extra_denials"]
        or len(p["files"]) != 8
        or stages["analysis"]["counts"] != COUNTS
        or prior.prior_cpu(out) > CAP
    ):
        raise ValueError("Finite budget or inventory mismatch")
    for f in ("predictions", "replay"):
        if any(stages[f][k] for k in ("extra_denials", "native_reads", "target_reads")):
            raise ValueError("Unexpected application reads")
    result = dict(
        status="PASS",
        source_authors=4,
        recipient_authors=8,
        coefficient_solves=2,
        observations=28,
        native_calls=0,
        human_labels_read=False,
        numeric_cpu_seconds=prior.prior_cpu(out),
        public_tables=len(TABLES),
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
    )
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run a single bounded immutable stage per command."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "models", "predict", "evaluate", "audit"),
    )
    p.add_argument("--folder", choices=("predictions", "replay"), default="predictions")
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage in ("predict", "evaluate"):
            globals()[a.stage](a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
