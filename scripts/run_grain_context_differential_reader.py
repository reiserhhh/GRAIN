#!/usr/bin/env python3
"""NEXT79: one source-normalized context field, with frozen component targets."""

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
from grain_research.vector_mechanisms import context_differential_reader as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_endpoint_query_contrast as previous

ep = previous.previous
old, prior = previous.old, previous.prior
OUT = old.BASE / "logic_context_differential_reader_v1"
PLAN = ROOT / "docs/GRAIN_CONTEXT_DIFFERENTIAL_READER_PLAN_V1.md"
CAP = 90
TABLES = (
    "score_metrics.csv",
    "factor_metrics.csv",
    "term_metrics.csv",
    "joint_account.csv",
    "error_account.csv",
    "reader_metrics.csv",
    "weight_diagnostics.csv",
    "range_account.csv",
)
COUNTS = [384, 1152, 192, 72, 64, 4, 2, 4]


def limit(start, used):
    """Keep the entire numeric card, including replay, inside its fixed CPU cap."""
    if time.process_time() - start + used > CAP:
        raise TimeoutError("NEXT79 cumulative CPU budget exceeded")


def prepare(out):
    """Pin previous source/target provenance without estimating on recipients."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT78 required")
    out.mkdir(exist_ok=False)
    jobs = old.read(previous.OUT / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Exact source4/new8 panel required")
    old.write(out / "private_jobs.json", jobs)
    inputs = dict(m["inputs"])
    inputs.update(
        {str((previous.OUT / n).relative_to(ROOT)): h for n, h in m["prepared"].items()}
    )
    for root, folders in ((previous.OUT, ("analysis",)), (ep.OUT, ("predictions",))):
        for folder in folders:
            fr = ep.freeze(root, folder)
            inputs.update(
                {
                    str((root / folder / n).relative_to(ROOT)): h
                    for n, h in fr["files"].items()
                }
            )
    paths = [
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        ROOT / "tests/test_grain_context_differential_reader.py",
        previous.OUT / "audit.json",
    ]
    inputs.update({str(p.relative_to(ROOT)): old.sha(p) for p in paths})
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONTEXT_DIFFERENTIAL_READER_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "prepare_cpu_seconds": time.process_time() - start,
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "source_constructions": 5,
            "coefficient_solves": 0,
            "max_cpu_seconds": CAP,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED"}, flush=True)


def models(out):
    """Construct five references using only source contexts and source readers."""
    old.verify(out)
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = old.read(out / "private_jobs.json")
    source = [j for j in jobs if j["cohort"] == "source4"]
    refs = {prior.reference(j): ep.source_set(j["cohort"], j["index"]) for j in jobs}
    if len(refs) != 5:
        raise ValueError("Five source references required")
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [r + ".npz" for r in refs]
    reads = [
        root / (k + ".npz")
        for j in source
        for k in j["query_keys"]
        for root in (ep.STATE / "private_inputs", ep.ENDPOINT / "capture_source")
    ]
    probes = [
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        ep.STATE / "private_inputs/new8_a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    bank = {}
    for j in source:
        c = [
            ep.selected(ep.STATE / "private_inputs" / (k + ".npz"), ("state",))[
                "state"
            ][-1]
            for k in j["query_keys"]
        ]
        r = [
            ep.selected(ep.ENDPOINT / "capture_source" / (k + ".npz"), ("r",))["r"]
            for k in j["query_keys"]
        ]
        bank[j["index"]] = (np.mean(c, axis=0), np.stack(r))
    for ref, members in refs.items():
        np.savez(dst / (ref + ".npz"), **fn.build(bank, members))
        limit(start, used)
    ep.previous.finish(
        out,
        "models",
        names,
        start,
        source_constructions=5,
        coefficient_solves=0,
        extra_denials=len(denied) - len(probes),
    )


def predict(out, folder):
    """Use paired H16 and source functions only, denying current native outcomes."""
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
    reads += [ep.OUT / "private_readers" / n for n in names]
    reads += [
        root / "models" / f"{prior.reference(j)}_{q}.npz"
        for j in jobs
        for q in old.QUERIES
        for root in (ep.SELECTOR, ep.previous.OUT)
    ]
    probes = [
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        ep.CAPTURE / "private_capture" / (keys[0] + ".npz"),
        ep.SOURCE / "private_inputs" / (keys[0] + ".npz"),
        previous.OUT / "analysis/term_metrics.csv",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    weights = old.arrays(ep.STATE / "private_weights.npz")
    if {"q", "q_bias"} & set(weights):
        raise ValueError("Native Q forbidden in application")
    base_errors, fd_errors, body_checks = [], [], 0
    for j in jobs:
        model = old.arrays(out / "models" / (prior.reference(j) + ".npz"))
        np.testing.assert_array_equal(
            model["members"], ep.source_set(j["cohort"], j["index"])
        )
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        d = [ep.body_directions(x) for x in data]
        for dx in d:
            if np.any(dx[:, -1]):
                raise ValueError("BODY directions must exclude context row")
            body_checks += 1
        c = np.mean([x["state"][-1] for x in data], axis=0)
        r = np.stack([fn.read(model, c, conditional=bool(v)) for v in range(2)])
        rb = old.arrays(ep.OUT / "private_readers" / (j["key"] + ".npz"))["r"][:, 0]
        base_errors.append(float(np.max(abs(r[0] - rb))))
        oldp = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        primal, vectors = [
            np.stack([p[k] for p in oldp]) for k in ("primal", "vectors")
        ]
        projected = [fn.project(primal, vectors, rv) for rv in r]
        result = {k: np.stack([p[k] for p in projected]) for k in projected[0]}
        for k in ("scalar", "actions"):
            base_errors.append(
                float(np.max(abs(result[k][0] - np.stack([p[k] for p in oldp]))))
            )
        if j["cohort"] == "source4" and j["index"] == 0:
            for qi, q in enumerate(old.QUERIES):
                name = f"{prior.reference(j)}_{q}.npz"
                sm, vm = [
                    old.arrays(root / "models" / name)
                    for root in (ep.SELECTOR, ep.previous.OUT)
                ]
                for f in range(4):
                    scalars = []
                    for eps in (-1e-5, 1e-5):
                        changed = dict(
                            data[qi], state=data[qi]["state"] + eps * d[qi][f]
                        )
                        cx = (changed["state"][-1] + data[1 - qi]["state"][-1]) / 2
                        px = ep.fn.predict(
                            changed, d[qi], weights, sm, vm, ep.fn.readers(r[0])[qi]
                        )
                        scalars.append(
                            np.stack(
                                [
                                    px["primal"]
                                    @ ep.fn.readers(fn.read(model, cx, bool(v)))[qi].T
                                    for v in range(2)
                                ]
                            )
                        )
                    fd_errors.extend(
                        abs(
                            (scalars[1] - scalars[0]) / 2e-5
                            - result["actions"][:, qi, :, 2, :, f]
                        ).ravel()
                    )
        diag = fn.diagnostics(model, c)
        np.savez(
            dst / (j["key"] + ".npz"),
            **result,
            readers=r,
            source_weights=fn.weights(model, c)[0],
            diagnostics=np.array(list(diag.values())),
            diagnostic_fields=np.array(list(diag)),
        )
        limit(start, used)
    if (
        max(base_errors) > 1e-9
        or len(fd_errors) != 128
        or max(fd_errors) > 1e-7
        or body_checks != 24
    ):
        raise ValueError("Baseline replay or full scalar BODY derivative failed")
    ep.previous.finish(
        out,
        folder,
        names,
        start,
        baseline_max_error=max(base_errors),
        fd_count=len(fd_errors),
        fd_max_error=float(max(fd_errors)),
        body_context_checks=body_checks,
        extra_denials=len(denied) - len(probes),
        native_reads=0,
        target_reads=0,
    )


def center(x):
    """Center only over authors, keeping fixed GRAIN factor columns separate."""
    return x - x.mean(axis=0, keepdims=True)


def summarize(groups):
    """Keep every model, query, mode and factor, plus direct and projected errors."""
    score, factor, term, joint, error, reader, diagnostic, ranges = ([] for _ in TABLES)
    for cohort, rows in groups.items():
        base = {"cohort": cohort, "authors": len(rows)}
        arrays = {
            k: np.stack([r[k] for r in rows])
            for k in rows[0]
            if isinstance(rows[0][k], np.ndarray)
        }
        for oi, obs in enumerate(fn.contrast.OBSERVABLES):
            y = (
                arrays["target_scalar_terms"]
                if oi == 0
                else arrays["target_action_terms"][:, oi - 1]
            )
            joint.append(
                base
                | {
                    "variant": "native_target",
                    "arm": "native_target",
                    "observable": obs,
                }
                | fn.contrast.moments(y)
            )
        for vi, variant in enumerate(fn.VARIANTS):
            meta = base | {"variant": variant}
            for ai, arm in enumerate(ep.fn.ARMS[:4]):
                m = meta | {"arm": arm}
                for oi, obs in enumerate(fn.contrast.OBSERVABLES):
                    if oi == 0:
                        p, y = arrays["scalar"][:, vi, :, ai], arrays["target_scalar"]
                        pt, yt = (
                            arrays["scalar_terms"][:, vi, ai],
                            arrays["target_scalar_terms"],
                        )
                    else:
                        p, y = (
                            arrays["actions"][:, vi, :, ai, oi - 1],
                            arrays["target_actions"][:, :, oi - 1],
                        )
                        pt, yt = (
                            arrays["action_terms"][:, vi, ai, oi - 1],
                            arrays["target_action_terms"][:, oi - 1],
                        )
                    mm = m | {"observable": obs}
                    joint.append(mm | fn.contrast.moments(pt))
                    error.append(mm | fn.contrast.error_account(yt, pt))
                    for ti, name in enumerate(fn.contrast.TERMS):
                        term.append(
                            mm | {"term": name} | previous.metric(yt[:, ti], pt[:, ti])
                        )
                    for mi, mode in enumerate(ep.fn.MODES):
                        for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
                            yy = y[:, qi, mi] if qi < 2 else y[:, 0, mi] - y[:, 1, mi]
                            pp = p[:, qi, mi] if qi < 2 else p[:, 0, mi] - p[:, 1, mi]
                            mq = mm | {"mode": mode, "query": q}
                            score.append(
                                mq
                                | previous.metric(yy, pp)
                                | {
                                    "centered_mse": float(
                                        np.mean((center(pp) - center(yy)) ** 2)
                                    )
                                }
                            )
                            if oi:
                                for fi in range(4):
                                    factor.append(
                                        mq
                                        | {"factor": f"whole:F{fi + 1:02d}"}
                                        | previous.metric(yy[:, fi], pp[:, fi])
                                    )
            ra = [r["ranges"][vi] for r in rows]
            total = sum(x["vector_error2"] for x in ra)
            native = sum(x["native_norm2"] for x in ra)
            rr = meta | {
                "vector_error_over_native": sum(x["vector_error2"] for x in ra) / native
            }
            for oi, obs in enumerate(fn.contrast.OBSERVABLES):
                p = (
                    arrays["projected_scalar"][:, vi]
                    if oi == 0
                    else arrays["projected_actions"][:, vi, oi - 1]
                )
                y = (
                    arrays["target_scalar_terms"][:, 1]
                    if oi == 0
                    else arrays["target_action_terms"][:, oi - 1, 1]
                )
                rr.update({obs + "_" + k: v for k, v in previous.metric(y, p).items()})
                rr[obs + "_centered_mse"] = float(np.mean((center(p) - center(y)) ** 2))
            reader.append(rr)
            ranges.append(
                meta
                | {
                    "affine_rank_mean": float(np.mean([x["rank"] for x in ra])),
                    "vector_error2": total,
                    "orthogonal_lower_bound2": sum(
                        x["orthogonal_lower_bound2"] for x in ra
                    ),
                    "in_span_error2": sum(x["in_span_error2"] for x in ra),
                    "irreducible_fraction_of_vector_error": sum(
                        x["orthogonal_lower_bound2"] for x in ra
                    )
                    / total,
                }
            )
        ds = pd.DataFrame([r["diagnostics"] for r in rows])
        dg = dict(base)
        for col in ds:
            dg.update(
                {
                    col + "_" + agg: float(getattr(ds[col], agg)())
                    for agg in ("mean", "min", "max")
                }
            )
        diagnostic.append(dg)
    return [
        pd.DataFrame(x)
        for x in (score, factor, term, joint, error, reader, diagnostic, ranges)
    ]


def evaluate(out, folder):
    """Open native readers after frozen application; no recipient fitting follows."""
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
    for j in old.read(out / "private_jobs.json"):
        p = old.arrays(out / folder / (j["key"] + ".npz"))
        model = old.arrays(out / "models" / (prior.reference(j) + ".npz"))
        native = previous.native_packets(j, w, qw)
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
        target = fn.project(primal, vectors, rc)
        row = {k: p[k] for k in ("scalar", "actions", "scalar_terms", "action_terms")}
        row.update(
            {
                "target_" + k: (v[:, 0] if k in ("scalar", "actions") else v[0])
                for k, v in target.items()
            }
        )
        nd = rc[0] - rc[1]
        rd = p["readers"][:, 0] - p["readers"][:, 1]
        row["projected_scalar"] = rd @ primal.mean(axis=0)[0]
        row["projected_actions"] = np.einsum("cfh,vh->vcf", vectors.mean(axis=0)[0], rd)
        name = "private_" + j["key"] + ".npz"
        names.append(name)
        np.savez(dst / name, **row)
        row["ranges"] = [fn.range_account(model, nd, d) for d in rd]
        row["diagnostics"] = {
            str(k): float(v)
            for k, v in zip(p["diagnostic_fields"], p["diagnostics"], strict=True)
        }
        groups[j["cohort"]].append(row)
        limit(start, used)
    tables = summarize(groups)
    if [len(t) for t in tables] != COUNTS:
        raise ValueError("Incomplete evaluation grid")
    oldtable = pd.read_csv(previous.OUT / "analysis/term_metrics.csv")
    errs = []
    for row in tables[2][tables[2].variant == "constant"].to_dict("records"):
        match = oldtable
        for key in ("cohort", "arm", "observable", "term"):
            match = match[match[key] == row[key]]
        if len(match) != 1:
            raise ValueError("Nonunique NEXT78 baseline lookup")
        errs += [
            abs(row[k] - float(match.iloc[0][k]))
            for k in ("r", "mse_over_zero", "rmse")
        ]
    if len(errs) != 288 or max(errs) > 1e-8:
        raise ValueError("NEXT78 baseline replay failed")
    for name, table in zip(TABLES, tables, strict=True):
        table.to_csv(dst / name, index=False)
    limit(start, used)
    ep.previous.finish(
        out,
        dst.name,
        [*TABLES, *names],
        start,
        counts=COUNTS,
        baseline_metric_max_error=max(errs),
    )


def audit(out):
    """Verify exact replay, baseline reproduction, isolation and finite-card cost."""
    old.verify(out)
    fm = ep.freeze(out, "models")
    a, b, aa, bb = [
        ep.freeze(out, n)
        for n in ("predictions", "replay", "analysis", "replay_analysis")
    ]
    if (
        len(fm["files"]) != 5
        or len(a["files"]) != 12
        or len(aa["files"]) != 20
        or aa["counts"] != COUNTS
    ):
        raise ValueError("Incomplete frozen inventory")
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
                    raise ValueError("Replay field mismatch")
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k])
            elif fa["files"][name] != fb["files"][name]:
                raise ValueError("Exact CSV replay failed")
    if (
        fm["extra_denials"]
        or any(
            f["extra_denials"] or f["native_reads"] or f["target_reads"] for f in (a, b)
        )
        or prior.prior_cpu(out) > CAP
    ):
        raise ValueError("Access or budget failure")
    result = {
        "status": "PASS",
        "source_constructions": 5,
        "coefficient_solves": 0,
        "new_neural_calls": 0,
        "human_labels_read": False,
        "public_tables": 8,
        "application_packets": 12,
        "body_context_checks": a["body_context_checks"],
        "baseline_max_error": a["baseline_max_error"],
        "baseline_metric_max_error": aa["baseline_metric_max_error"],
        "fd_count": a["fd_count"],
        "fd_max_error": a["fd_max_error"],
        "total_cpu_seconds": prior.prior_cpu(out),
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run each guarded immutable stage in a fresh one-thread process."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "models", "predict", "evaluate", "audit"),
    )
    p.add_argument("--folder", default="predictions", choices=("predictions", "replay"))
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        if args.stage in ("predict", "evaluate"):
            globals()[args.stage](args.output, args.folder)
        else:
            globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
