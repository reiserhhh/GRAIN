#!/usr/bin/env python3
"""NEXT89 aggregate evaluation, moving-common error account and replay audit."""

import time

import numpy as np
import pandas as pd

from scripts import run_grain_joint_query_functional_reader as run

old, ep, p85, fn = run.old, run.ep, run.p85, run.fn
p82 = run.previous.previous.p82


def summarize(
    rows, model, baseline, teaching, prior_tables, programs, prior_rows, prior_errors
):
    """Retain the previous22 readers and append all joint-reader observations."""
    z = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    p = {k: np.stack([r[k] for r in programs]) for k in programs[0]}
    sc, ac, fc, pr, vr, diag, sf, cp, geo, pm, se, cond, rp = [[] for _ in run.TABLES]
    meta = {"variant": fn.VARIANT, "cohort": "new8", "authors": 8}
    nr = z["native_r"]
    rr = z["readers"][:, 0]
    nd = nr[:, 0] - nr[:, 1]
    d = rr[:, 0] - rr[:, 1]
    error = d - nd
    vr.append(
        meta
        | {
            "vector_error2": float(np.sum(error**2)),
            "error_over_native": float(np.sum(error**2) / np.sum(nd**2)),
            "centered_error2": float(np.sum((error - error.mean(axis=0)) ** 2)),
        }
    )
    rulers = {"scalar": z["native_primal"].mean(axis=1)}
    for ki, k in enumerate(ep.fn.COMPONENTS):
        for f in range(4):
            rulers[f"{k}_F{f + 1:02d}"] = z["native_vectors"].mean(axis=1)[:, ki, f]
    for name, ruler in rulers.items():
        y, pred = [np.einsum("nh,nh->n", a, ruler) for a in (nd, d)]
        pr.append(meta | {"observable": name} | p82.metric(y, pred))
    t = fn.teaching(
        np.concatenate((p["primal"][:, :, 3, None], p["vectors"][:, :, 3, 2]), axis=2),
        np.concatenate(
            (z["native_primal"][:, :, None], z["native_vectors"][:, :, 2]), axis=2
        ),
        nr,
        baseline,
        model["units"],
    )
    np.testing.assert_allclose(t["frame"], z["frame"], atol=1e-12)
    conditions = []
    for cohort, frames in (("source12", teaching["frame"]), ("new8", z["frame"])):
        cs = np.asarray(
            [[fn.coord.frame_geometry(a)["condition"] for a in pair] for pair in frames]
        )
        conditions.append(cs)
        for qi, q in enumerate(old.QUERIES):
            geo.append(
                {
                    "cohort": cohort,
                    "authors": len(frames),
                    "query": q,
                    "rank": 5,
                    "condition_min": float(cs[:, qi].min()),
                    "condition_max": float(cs[:, qi].max()),
                    "condition_median": float(np.median(cs[:, qi])),
                }
            )
    for qi, q in enumerate(old.QUERIES):
        for f in range(5):
            name = "scalar" if f == 0 else f"total_F{f:02d}"
            sf.append(
                {"variant": fn.VARIANT, "authors": 12, "query": q, "coordinate": name}
                | p82.metric(
                    model["target"][:, qi * 5 + f], model["fitted"][:, qi * 5 + f]
                )
            )
            for target in ("native_coordinates", "compatible_coordinates"):
                cp.append(
                    meta
                    | {"query": q, "coordinate": name, "target": target}
                    | p82.metric(
                        t[target][:, qi, f], z["coordinate_values"][:, 0, qi, f]
                    )
                )
    for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
        s, a, cor, dd = [
            p82.qview(z[k][:, 0], qi)
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
                            "reader_correction_rms": float(np.sqrt(np.mean(extra**2))),
                            "diagonal_rms": float(np.sqrt(np.mean(diagonal**2))),
                            "signed_cross_energy": float(2 * np.mean(anchor * extra)),
                            "correction_mean": float(extra.mean()),
                        }
                    )
    for ai, arm in enumerate(ep.fn.ARMS[:4]):
        for oi, obs in enumerate(("scalar", *ep.fn.COMPONENTS)):
            cur = (
                p["primal"][:, :, ai, None]
                if oi == 0
                else p["vectors"][:, :, ai, oi - 1]
            )
            nat = (
                z["native_primal"][:, :, None]
                if oi == 0
                else z["native_vectors"][:, :, oi - 1]
            )
            acc = fn.decompose(cur, nat, rr, nr)
            expected = (
                (z["scalar"][:, 0, 0, ai, 0] - z["scalar"][:, 0, 1, ai, 0])[:, None]
                if oi == 0
                else z["actions"][:, 0, 0, ai, oi - 1, 0]
                - z["actions"][:, 0, 1, ai, oi - 1, 0]
            )
            np.testing.assert_allclose(acc["prediction"], expected, atol=1e-10)
            se.append(meta | {"arm": arm, "observable": obs} | fn.energies(acc))
    combined = {
        k: np.concatenate((np.stack([a[k] for a in prior_rows]), z[k]), axis=1)
        for k in ("scalar", "actions", "readers")
    }
    for vi, variant in enumerate((*run.previous.fn.VARIANTS, fn.VARIANT)):
        mr = meta | {"variant": variant}
        diff = combined["readers"][:, vi] - nr
        mean_error = diff.mean(axis=1)
        difference_error = diff[:, 0] - diff[:, 1]
        whole = float(np.sum(diff**2))
        mean_energy = float(2 * np.sum(mean_error**2))
        difference_energy = float(np.sum(difference_error**2) / 2)
        np.testing.assert_allclose(whole, mean_energy + difference_energy, atol=1e-10)
        rp.append(
            mr
            | {
                "pair_error2": whole,
                "common_pair_error2": mean_energy,
                "difference_pair_error2": difference_energy,
                "pair_error_over_native": whole / float(np.sum(nr**2)),
            }
        )
        for ai, arm in enumerate(ep.fn.ARMS[:4]):
            for mi, mode in enumerate(ep.fn.MODES):
                for oi, obs in enumerate(("scalar", *ep.fn.COMPONENTS)):
                    pred = (
                        combined["scalar"][:, vi, :, ai, mi].mean(axis=1)
                        if oi == 0
                        else combined["actions"][:, vi, :, ai, oi - 1, mi].mean(axis=1)
                    )
                    actual = (
                        z["target_scalar"][:, :, mi].mean(axis=1)
                        if oi == 0
                        else z["target_actions"][:, :, oi - 1, mi].mean(axis=1)
                    )
                    pm.append(
                        mr
                        | {"arm": arm, "mode": mode, "observable": obs}
                        | p82.metric(actual, pred)
                    )
    cond.append(
        {
            "variant": fn.VARIANT,
            "authors": 12,
            "observations": 12,
            "source_weight": float(model["weights"][0]),
            **{
                k: float(model[k])
                for k in (
                    "coefficients",
                    "regularizer",
                    "regularized_condition",
                    "normal_residual_relative",
                )
            },
        }
    )
    core = [
        pd.concat((old_table, pd.DataFrame(new)), ignore_index=True)
        for old_table, new in zip(prior_tables, (sc, ac, fc, pr, vr, diag), strict=True)
    ]
    return [
        *core,
        pd.DataFrame(sf),
        pd.DataFrame(cp),
        pd.DataFrame(geo),
        pd.DataFrame(pm),
        pd.concat((prior_errors, pd.DataFrame(se)), ignore_index=True),
        pd.DataFrame(cond),
        pd.DataFrame(rp),
    ]


def evaluate(out, folder):
    """Open native evaluation targets only after the joint predictions freeze."""
    old.verify(out)
    ep.freeze(out, folder)
    used, start = p85.cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    rows, names, programs, prior_rows = [], [], [], []
    for j in old.read(out / "private_jobs.json"):
        if j["cohort"] != "new8":
            continue
        n = "private_" + j["key"] + ".npz"
        prior_row = old.arrays(run.previous.OUT / "analysis" / n)
        row = old.arrays(out / folder / (j["key"] + ".npz")) | {
            k: prior_row[k]
            for k in (
                "native_r",
                "native_primal",
                "native_vectors",
                "target_scalar",
                "target_actions",
            )
        }
        np.savez(dst / n, **row)
        rows.append(row)
        names.append(n)
        prior_rows.append(prior_row)
        pair = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        programs.append(
            {k: np.stack([p[k] for p in pair]) for k in ("primal", "vectors")}
        )
    tables = summarize(
        rows,
        old.arrays(out / "models" / (fn.VARIANT + ".npz")),
        old.arrays(p85.BASELINE),
        old.arrays(out / "teachers/private_teachers.npz"),
        [pd.read_csv(run.previous.OUT / "analysis" / n) for n in run.CORE],
        programs,
        prior_rows,
        pd.read_csv(run.previous.OUT / "analysis/signed_errors.csv"),
    )
    if [len(t) for t in tables] != run.COUNTS:
        raise ValueError("Finite table inventory differs")
    for n, t in zip(run.TABLES, tables, strict=True):
        t.to_csv(dst / n, index=False)
    run.limit(start, used)
    ep.previous.finish(out, dst.name, [*run.TABLES, *names], start, counts=run.COUNTS)


def audit(out):
    """Verify stage isolation, fixed source parameters and exact complete replay."""
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
                    raise ValueError("Replay array keys differ")
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k])
            elif a["files"][n] != b["files"][n]:
                raise ValueError("CSV replay differs")
    for n in run.CORE:
        a, b = [pd.read_csv(root / "analysis" / n) for root in (run.previous.OUT, out)]
        pd.testing.assert_frame_equal(
            a,
            b.iloc[: len(a)].reset_index(drop=True),
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    m, s = [
        old.arrays(p)
        for p in (out / "models" / (fn.VARIANT + ".npz"), run.SOURCE_MODEL)
    ]
    for k in ("x", "v", "kinds", "weights", "regularizer", "units"):
        np.testing.assert_array_equal(m[k], s[k])
    t = old.arrays(out / "teachers/private_teachers.npz")
    np.testing.assert_array_equal(
        m["target"], t["compatible_coordinates"].reshape(12, 10)
    )
    for f in ("teachers", "models", "predictions", "replay"):
        if stages[f]["extra_denials"]:
            raise ValueError("Unexpected stage access")
    for f in ("predictions", "replay"):
        a = stages[f]
        if a["compiled_calls"] != 128 or a["native_reads"] or a["target_reads"]:
            raise ValueError("Application contract differs")
    if (
        stages["models"]["source_solves"] != 1
        or m["coefficients"] != 120
        or stages["analysis"]["counts"] != run.COUNTS
        or p85.cpu(out) > run.CAP
    ):
        raise ValueError("Fixed solve or budget contract differs")
    result = {
        "status": "PASS",
        "source_authors": 12,
        "recipient_authors": 8,
        "source_solves": 1,
        "coefficients": 120,
        "compiled_calls_including_replay": 256,
        "numeric_cpu_seconds": p85.cpu(out),
        "native_calls": 0,
        "human_labels_read": False,
        "prior_readers_retained": 22,
        "public_tables": len(run.TABLES),
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
