#!/usr/bin/env python3
"""NEXT82: three source-only derivative-kernel reader fields and exact local calculus."""

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
from grain_research.vector_mechanisms import material_reader_field as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_reader_material_dependence as previous

old, ep, prior, motion = previous.old, previous.ep, previous.prior, previous.previous
OUT = old.BASE / "logic_material_reader_field_v1"
PLAN = ROOT / "docs/GRAIN_MATERIAL_READER_FIELD_PLAN_V1.md"
VARIANTS = (*motion.fn.VARIANTS, *fn.CONTRACTS)
CAP = 90
TABLES = (
    "source_fit.csv",
    "source_geometry.csv",
    "component_scores.csv",
    "component_actions.csv",
    "factor_actions.csv",
    "reader_projection.csv",
    "reader_vector.csv",
    "diagonal_actions.csv",
    "input_support.csv",
)
COUNTS = [9, 3, 144, 432, 1728, 78, 6, 576, 3]


def limit(start, used):
    """Apply a cumulative CPU cap to all finite numerical stages."""
    if used + time.process_time() - start > CAP:
        raise TimeoutError("NEXT82 cumulative numeric cap")


def prepare(out):
    """Pin prior code/source/target bytes; only source data teach the new fields."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT81 required")
    out.mkdir(exist_ok=False)
    jobs = old.read(motion.OUT / "private_jobs.json")
    old.write(out / "private_jobs.json", jobs)
    inputs = dict(m["inputs"])
    for root, folders in (
        (previous.OUT, ("capture",)),
        (motion.OUT, ("models", "predictions")),
    ):
        for folder in folders:
            fr = ep.freeze(root, folder)
            inputs.update(
                {
                    str((root / folder / n).relative_to(ROOT)): h
                    for n, h in fr["files"].items()
                }
            )
    for p in (
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        previous.OUT / "audit.json",
        ROOT / "tests/test_grain_material_reader_field.py",
    ):
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_MATERIAL_READER_FIELD_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "prepare_cpu_seconds": time.process_time() - start,
            "source_authors": 4,
            "recipient_authors": 8,
            "source_values": 8,
            "source_body_derivatives": 16,
            "source_context_derivatives": 4,
            "max_solves": 3,
            "max_cpu_seconds": CAP,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED"}, flush=True)


def models(out):
    """Build exactly three joint-input fields, using only four source authors."""
    old.verify(out)
    used, start = prior.prior_cpu(out), time.process_time()
    src = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "source4"]
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [c + ".npz" for c in fn.CONTRACTS]
    reads = [
        ep.STATE / "private_inputs" / (k + ".npz") for j in src for k in j["query_keys"]
    ]
    reads += [previous.OUT / "capture" / f"a{j['index']:02d}.npz" for j in src]
    reads += [motion.OUT / "models/all4.npz"]
    probes = [
        ep.STATE / "private_inputs/new8_a00_neuroticism.npz",
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    old_model = old.arrays(motion.OUT / "models/all4.npz")
    points, tangents, target, originals = [], [], [], []
    rows = {g: [] for g in range(3)}
    original_readers = []
    for j in src:
        i = j["index"]
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        cap = old.arrays(previous.OUT / "capture" / f"a{i:02d}.npz")
        fixed = [previous.fn.clamp(x, cap["last"][q]) for q, x in enumerate(data)]
        b = fn.features(data)
        originals.append(b)
        zero = [np.zeros_like(x) for x in b]
        fixed_b, fixed_db = fn.features(fixed, [ep.body_directions(x) for x in data])
        original_readers.append(cap["original_r"])
        rows[0].append((b, zero, cap["original_r"][0] - cap["original_r"][1]))
        # Both views teach values; only original views set the reference units.
        rows[0].append((fixed_b, zero, cap["clamped_r"][0] - cap["clamped_r"][1]))
        for f in range(4):
            rows[1].append(
                (
                    fixed_b,
                    [x[f] for x in fixed_db],
                    previous.fn.query_slopes(cap["perturbed_r"][f], previous.ETA)[2],
                )
            )
        ds = [np.zeros((1, *x["state"].shape)) for x in data]
        for d in ds:
            d[0, -1] = old_model["v"][i]
        _, db = fn.features(data, ds)
        rows[2].append((b, [x[0] for x in db], old_model["j"][i]))
    # Values alternate original/clamp; all normalization uses the originals list only.
    d0 = np.mean([r[0] - r[1] for r in original_readers], axis=0)
    np.testing.assert_allclose(d0, old_model["d"].mean(axis=0), atol=1e-12)
    groups = []
    for gi in range(3):
        for b, db, y in rows[gi]:
            points.append(b)
            tangents.append(db)
            target.append(y - d0 if gi == 0 else y)
            groups.append(gi)
    groups = np.asarray(groups)
    target = np.stack(target)
    if len(target) != 28 or list(np.bincount(groups)) != [8, 16, 4]:
        raise ValueError("Frozen source observation inventory violated")
    scale = fn.scales(originals)
    for contract in fn.CONTRACTS:
        x, v = [
            np.stack([fn.flatten(b, scale, contract) for b in rows_])
            for rows_ in (points, tangents)
        ]
        model = fn.fit(x, v, groups > 0, target, groups, d0, old_model["m"])
        model.update(
            scale=scale,
            groups=groups,
            target=target,
            source_original_x=np.stack(
                [fn.flatten(b, scale, contract) for b in originals]
            ),
        )
        np.savez(dst / (contract + ".npz"), **model)
        limit(start, used)
    ep.previous.finish(
        out,
        "models",
        names,
        start,
        coefficient_solves=3,
        observations=28,
        extra_denials=len(denied) - len(probes),
        human_labels_read=False,
    )


def projections(primal, vectors, r, dr):
    """Separate anchored H partials from the diagonal derivative of the same values."""
    p = motion.fn.base.project(primal, vectors, r)
    drm = np.stack((dr, np.zeros_like(dr)), axis=1)
    correction = np.einsum("qah,qmfh->qamf", primal, drm)
    return {
        "scalar": p["scalar"],
        "actions": p["actions"],
        "correction": correction,
        "diagonal": p["actions"][:, :, 2] + correction,
    }


def predict(out, folder):
    """Apply three new and three frozen comparator fields to new8 without targets."""
    old.verify(out)
    fm = ep.freeze(out, "models")
    used, start = prior.prior_cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "new8"]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    reads = [out / "models" / n for n in fm["files"]] + [
        ep.STATE / "private_weights.npz"
    ]
    reads += [
        root / (k + ".npz")
        for j in jobs
        for k in j["query_keys"]
        for root in (ep.STATE / "private_inputs", ep.OUT / "predictions")
    ]
    reads += [motion.OUT / "predictions" / n for n in names]
    reads += [
        root / "models" / f"all4_{q}.npz"
        for q in old.QUERIES
        for root in (ep.SELECTOR, ep.previous.OUT)
    ]
    probes = [
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        previous.OUT / "capture/a00.npz",
        ep.CAPTURE / "private_capture" / (jobs[0]["query_keys"][0] + ".npz"),
    ]
    denied = install(
        reads,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    models = [old.arrays(out / "models" / (c + ".npz")) for c in fn.CONTRACTS]
    weights = old.arrays(ep.STATE / "private_weights.npz")
    if {"q", "q_bias"} & set(weights):
        raise ValueError("Current Q forbidden")
    fd_a, fd_d, base_errors = [], [], []
    for j in jobs:
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        ds = [ep.body_directions(x) for x in data]
        b, db = fn.features(data, ds)
        prior_pred = old.arrays(motion.OUT / "predictions" / (j["key"] + ".npz"))
        packets = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        primal, vectors = [
            np.stack([x[k] for x in packets]) for k in ("primal", "vectors")
        ]
        rr, dd = (
            list(prior_pred["readers"]),
            [np.zeros((2, 4, primal.shape[-1])) for _ in range(3)],
        )
        support = []
        for model, contract in zip(models, fn.CONTRACTS, strict=True):
            r, dr = fn.readers(model, b, db, model["scale"], contract)
            rr.append(r)
            dd.append(dr)
            x = fn.flatten(b, model["scale"], contract)
            d2 = np.sum((model["source_original_x"] - x) ** 2, axis=1)
            support.append(
                [float(d2.min()), float(d2.mean()), float(np.exp(-d2 / 2).max())]
            )
        rr, dd = np.stack(rr), np.stack(dd)
        results = [
            projections(primal, vectors, r, d) for r, d in zip(rr, dd, strict=True)
        ]
        result = {k: np.stack([p[k] for p in results]) for k in results[0]}
        for k in ("scalar", "actions"):
            base_errors.append(float(abs(result[k][:3] - prior_pred[k]).max()))
        if j["index"] == 0:
            eps = 1e-5
            for f in range(4):
                anchored, diagonal = [], []
                for e in (-eps, eps):
                    changed = [
                        dict(x, state=x["state"] + e * d[f])
                        for x, d in zip(data, ds, strict=True)
                    ]
                    cb, cdb = fn.features(changed, ds)
                    px = []
                    for qi, q in enumerate(old.QUERIES):
                        sm, vm = [
                            old.arrays(root / "models" / f"all4_{q}.npz")
                            for root in (ep.SELECTOR, ep.previous.OUT)
                        ]
                        px.append(
                            ep.fn.predict(
                                changed[qi],
                                ds[qi],
                                weights,
                                sm,
                                vm,
                                ep.fn.readers(rr[0])[qi],
                            )["primal"]
                        )
                    px = np.stack(px)
                    av, dv = [], []
                    for vi, (model, contract) in enumerate(
                        zip(models, fn.CONTRACTS, strict=True), 3
                    ):
                        cr, _ = fn.readers(model, cb, cdb, model["scale"], contract)
                        av.append(np.einsum("qah,qmh->qam", px, ep.fn.readers(rr[vi])))
                        dv.append(np.einsum("qah,qmh->qam", px, ep.fn.readers(cr)))
                    anchored.append(np.stack(av))
                    diagonal.append(np.stack(dv))
                fd_a.extend(
                    abs(
                        (anchored[1] - anchored[0]) / (2 * eps)
                        - result["actions"][3:, :, :, 2, :, f]
                    ).ravel()
                )
                fd_d.extend(
                    abs(
                        (diagonal[1] - diagonal[0]) / (2 * eps)
                        - result["diagonal"][3:, :, :, :, f]
                    ).ravel()
                )
        np.savez(
            dst / (j["key"] + ".npz"),
            **result,
            readers=rr,
            reader_derivatives=dd,
            support=np.asarray(support),
        )
        limit(start, used)
    if (
        len(fd_a) != 192
        or len(fd_d) != 192
        or max(fd_a) > 1e-6
        or max(fd_d) > 1e-6
        or max(base_errors) > 1e-10
    ):
        raise ValueError("Anchored/diagonal FD or old prediction parity failed")
    ep.previous.finish(
        out,
        folder,
        names,
        start,
        anchored_fd_count=len(fd_a),
        diagonal_fd_count=len(fd_d),
        anchored_fd_max=float(max(fd_a)),
        diagonal_fd_max=float(max(fd_d)),
        baseline_max=max(base_errors),
        native_reads=0,
        target_reads=0,
        extra_denials=len(denied) - len(probes),
    )


def metric(y, p):
    """Use prior fidelity definitions plus author-centered error on the same columns."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return old.fidelity(y.ravel(), p.ravel()) | {
        "author_centered_mse": float(
            np.mean(((y - y.mean(axis=0)) - (p - p.mean(axis=0))) ** 2)
        )
    }


def qview(x, qi):
    """Query view for author-first paired arrays, preserving direction columns."""
    return x[:, qi] if qi < 2 else x[:, 0] - x[:, 1]


def summarize(rows, models):
    """Keep all score/action cells and diagonal derivatives without false target matching."""
    z = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    fit, geo, sc, ac, fc, pr, vr, diag, sup = [], [], [], [], [], [], [], [], []
    for contract, m in zip(fn.CONTRACTS, models, strict=True):
        for gi, group in enumerate(fn.GROUPS):
            fit.append(
                {
                    "contract": contract,
                    "role": "source_training",
                    "authors": 4,
                    "group": group,
                    "observations": int(np.sum(m["groups"] == gi)),
                    "residual_sse_over_target": float(m["group_loss"][gi]),
                    "zero_target_group": bool(m["zero_target_groups"][gi]),
                }
            )
        geo.append(
            {
                "contract": contract,
                "authors": 4,
                "observations": 28,
                "regularizer": float(m["regularizer"]),
                "kernel_rank": int(m["kernel_rank"]),
                "kernel_min_eigenvalue": float(m["kernel_min_eigenvalue"]),
                "regularized_condition": float(m["regularized_condition"]),
                "zero_operator_rows": int(m["zero_operator_rows"]),
                "skipped_blocks": int(np.sum(m["scale"] == 0)),
            }
        )
    for vi, variant in enumerate(VARIANTS):
        pred_d = z["readers"][:, vi, 0] - z["readers"][:, vi, 1]
        target_d = z["native_r"][:, 0] - z["native_r"][:, 1]
        vr.append(
            {
                "variant": variant,
                "cohort": "new8",
                "authors": 8,
                "vector_error2": float(np.sum((pred_d - target_d) ** 2)),
                "error_over_native": float(
                    np.sum((pred_d - target_d) ** 2) / np.sum(target_d**2)
                ),
                "centered_error2": float(
                    np.sum(
                        ((pred_d - pred_d.mean(0)) - (target_d - target_d.mean(0))) ** 2
                    )
                ),
            }
        )
        rulers = {"scalar": z["native_primal"].mean(axis=1)}
        for ci, c in enumerate(ep.fn.COMPONENTS):
            for f in range(4):
                rulers[f"{c}_F{f + 1:02d}"] = z["native_vectors"].mean(axis=1)[:, ci, f]
        for observable, ruler in rulers.items():
            y = np.einsum("ih,ih->i", target_d, ruler)
            p = np.einsum("ih,ih->i", pred_d, ruler)
            pr.append(
                dict(
                    variant=variant,
                    cohort="new8",
                    authors=8,
                    observable=observable,
                    **metric(y, p),
                )
            )
        for qi, q in enumerate((*old.QUERIES, "N_minus_confirmation")):
            s = qview(z["scalar"][:, vi], qi)
            sy = qview(z["target_scalar"], qi)
            a = qview(z["actions"][:, vi], qi)
            ay = qview(z["target_actions"], qi)
            cor = qview(z["correction"][:, vi], qi)
            dd = qview(z["diagonal"][:, vi], qi)
            for ai, arm in enumerate(ep.fn.ARMS[:4]):
                for mi, mode in enumerate(ep.fn.MODES):
                    meta = {
                        "variant": variant,
                        "cohort": "new8",
                        "authors": 8,
                        "query": q,
                        "arm": arm,
                        "mode": mode,
                    }
                    sc.append(meta | metric(sy[:, mi], s[:, ai, mi]))
                    for ci, c in enumerate(ep.fn.COMPONENTS):
                        cm = meta | {"component": c}
                        ac.append(cm | metric(ay[:, ci, mi], a[:, ai, ci, mi]))
                        for f in range(4):
                            fc.append(
                                cm
                                | {"factor": f"F{f + 1:02d}"}
                                | metric(ay[:, ci, mi, f], a[:, ai, ci, mi, f])
                            )
                    for f in range(4):
                        anchored = a[:, ai, 2, mi, f]
                        extra = cor[:, ai, mi, f]
                        diagonal = dd[:, ai, mi, f]
                        np.testing.assert_allclose(
                            diagonal, anchored + extra, atol=1e-10
                        )
                        diag.append(
                            meta
                            | {
                                "factor": f"F{f + 1:02d}",
                                "anchored_rms": float(np.sqrt(np.mean(anchored**2))),
                                "reader_correction_rms": float(
                                    np.sqrt(np.mean(extra**2))
                                ),
                                "diagonal_rms": float(np.sqrt(np.mean(diagonal**2))),
                                "signed_cross_energy": float(
                                    2 * np.mean(anchored * extra)
                                ),
                                "correction_mean": float(extra.mean()),
                            }
                        )
    for vi, contract in enumerate(fn.CONTRACTS):
        v = z["support"][:, vi]
        sup.append(
            {
                "contract": contract,
                "authors": 8,
                "cohort": "new8",
                "nearest_distance2_mean": float(v[:, 0].mean()),
                "source_distance2_mean": float(v[:, 1].mean()),
                "nearest_kernel_mean": float(v[:, 2].mean()),
            }
        )
    return [pd.DataFrame(x) for x in (fit, geo, sc, ac, fc, pr, vr, diag, sup)]


def evaluate(out, folder):
    """Open old recipient targets only after the new predictions are frozen."""
    old.verify(out)
    ep.freeze(out, folder)
    used, start = prior.prior_cpu(out), time.process_time()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "new8"]
    w, qw = [old.arrays(r / "private_weights.npz") for r in (ep.STATE, ep.CAPTURE)]
    rows, names = [], []
    for j in jobs:
        p = old.arrays(out / folder / (j["key"] + ".npz"))
        native = motion.previous.previous.native_packets(j, w, qw)
        r = np.stack(
            [
                ep.selected(ep.ENDPOINT / "capture_new" / (k + ".npz"), ("r",))[
                    "r"
                ].astype(float)
                for k in j["query_keys"]
            ]
        )
        primal, vectors = native["primal"][:, 1], native["vectors"][:, 1]
        target = motion.fn.base.project(primal[:, None], vectors[:, None], r)
        row = p | {
            "native_r": r,
            "native_primal": primal,
            "native_vectors": vectors,
            "target_scalar": target["scalar"][:, 0],
            "target_actions": target["actions"][:, 0],
        }
        name = "private_" + j["key"] + ".npz"
        np.savez(dst / name, **row)
        names.append(name)
        rows.append(row)
        limit(start, used)
    models_ = [old.arrays(out / "models" / (c + ".npz")) for c in fn.CONTRACTS]
    results = summarize(rows, models_)
    if [len(x) for x in results] != COUNTS:
        raise ValueError("Incomplete comparison panel")
    for name, t in zip(TABLES, results, strict=True):
        t.to_csv(dst / name, index=False)
    limit(start, used)
    ep.previous.finish(out, dst.name, [*TABLES, *names], start, counts=COUNTS)


def audit(out):
    """Verify all frozen outputs, exact cached replay and the finite resource contract."""
    old.verify(out)
    m = ep.freeze(out, "models")
    stages = {
        n: ep.freeze(out, n)
        for n in ("predictions", "analysis", "replay", "replay_analysis")
    }
    for first, second in (("predictions", "replay"), ("analysis", "replay_analysis")):
        a, b = stages[first], stages[second]
        if set(a["files"]) != set(b["files"]):
            raise ValueError("Replay inventory differs")
        for name in a["files"]:
            if name.endswith(".npz"):
                x, y = [old.arrays(out / f / name) for f in (first, second)]
                for k in x:
                    np.testing.assert_array_equal(x[k], y[k])
            elif a["files"][name] != b["files"][name]:
                raise ValueError("CSV replay differs")
    if (
        len(m["files"]) != 3
        or m["coefficient_solves"] != 3
        or m["extra_denials"]
        or len(stages["predictions"]["files"]) != 8
        or stages["analysis"]["counts"] != COUNTS
        or any(
            stages[n][k]
            for n in ("predictions", "replay")
            for k in ("extra_denials", "native_reads", "target_reads")
        )
        or prior.prior_cpu(out) > CAP
    ):
        raise ValueError("Finite contract violation")
    p = stages["predictions"]
    result = dict(
        status="PASS",
        source_authors=4,
        recipient_authors=8,
        coefficient_solves=3,
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
                "baseline_max",
            )
        },
    )
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run each immutable stage separately with one cached replay and no neural calls."""
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
