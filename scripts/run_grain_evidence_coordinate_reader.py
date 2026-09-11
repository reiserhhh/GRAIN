#!/usr/bin/env python3
"""NEXT85: source-only fixed/moving evidence coordinates and coherent reader lift."""

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
from grain_research.vector_mechanisms import evidence_coordinate_reader as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_observable_span as previous

old, ep = previous.old, previous.ep
p83 = previous.previous
p82 = p83.previous
p81 = p82.previous
OUT = old.BASE / "logic_evidence_coordinate_reader_v1"
PLAN = ROOT / "docs/GRAIN_EVIDENCE_COORDINATE_READER_PLAN_V1.md"
BASELINE = previous.OUT / "models/expanded_observable_teaching.npz"
VARIANTS = (*previous.VARIANTS, *fn.VARIANTS)
TABLES = (
    *previous.TABLES,
    "coordinate_fit.csv",
    "frame_geometry.csv",
    "coordinate_prediction.csv",
    "lift_checks.csv",
    "reader_correction.csv",
    "frame_derivatives.csv",
)
COUNTS = [36, 4, 288, 864, 3456, 156, 12, 1152, 4, 52, 2, 2, 6, 6, 10, 4, 2, 25]
CAP, EPS = 180, 1e-5


def cpu(out):
    """Include teacher construction as well as both cached application passes."""
    value = old.read(out / "manifest.json")["prepare_cpu_seconds"]
    for folder in (
        "teachers",
        "models",
        "predictions",
        "analysis",
        "replay",
        "replay_analysis",
    ):
        path = out / folder / "freeze.json"
        if path.exists():
            value += float(old.read(path)["cpu_seconds"])
    return value


def limit(start, used):
    """Apply a recipient-outcome-independent cumulative CPU cap."""
    if used + time.process_time() - start > CAP:
        raise TimeoutError("NEXT85 numerical CPU cap")


def compiled_paths():
    """List the same source-only weights and four all4 program packages."""
    return [
        ep.STATE / "private_weights.npz",
        *[
            root / "models" / f"all4_{q}.npz"
            for q in old.QUERIES
            for root in (ep.SELECTOR, ep.previous.OUT)
        ],
    ]


class Compiler:
    """Count local matrix-program evaluations, never invoking a neural suffix."""

    def __init__(self, cap):
        self.weights = old.arrays(ep.STATE / "private_weights.npz")
        if {"q", "q_bias"} & set(self.weights):
            raise ValueError("Current Q forbidden")
        self.models = [
            [
                old.arrays(root / "models" / f"all4_{q}.npz")
                for root in (ep.SELECTOR, ep.previous.OUT)
            ]
            for q in old.QUERIES
        ]
        self.calls, self.cap = 0, cap

    def __call__(self, data, directions):
        packets = []
        for x, ds, (sm, vm) in zip(data, directions, self.models, strict=True):
            if self.calls >= self.cap:
                raise TimeoutError("Local compiled-program call cap")
            self.calls += 1
            packets.append(
                ep.fn.predict(
                    x, ds, self.weights, sm, vm, np.zeros((2, x["state"].shape[1]))
                )
            )
        result = {k: np.stack([p[k] for p in packets]) for k in ("primal", "vectors")}
        result["frame"] = fn.frame(result["primal"], result["vectors"])
        fn.frame_geometry(result["frame"])
        return result


def changed(data, directions, step):
    """Keep all GRAIN addresses/weights fixed while moving declared H16 rows."""
    return [
        x | {"state": x["state"].astype(float) + step * d}
        for x, d in zip(data, directions, strict=True)
    ]


def prepare(out):
    """Pin plans/code and exact prior source/new8 material before calculations."""
    start = time.process_time()
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT84 required")
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
        ROOT / "tests/test_grain_evidence_coordinate_reader.py",
    ):
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_EVIDENCE_COORDINATE_READER_V1",
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
            "source_call_cap": 192,
            "application_derivative_call_cap": 128,
            "application_fd_call_cap": 64,
            "body_target_state": "all4_last_clamped",
            "native_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED"}, flush=True)


def teachers(out):
    """Bind original/context and clamped/BODY targets; compute frame product terms."""
    old.verify(out)
    used, start = cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "source4"]
    dst = out / "teachers"
    dst.mkdir(exist_ok=False)
    motion_path = p82.motion.OUT / "models/all4.npz"
    reads = [BASELINE, motion_path, *compiled_paths()]
    reads += [
        ep.STATE / "private_inputs" / (k + ".npz")
        for j in jobs
        for k in j["query_keys"]
    ]
    reads += [p81.OUT / "capture" / f"a{j['index']:02d}.npz" for j in jobs]
    probes = [
        previous.OUT / "analysis/private_new8_a00.npz",
        ep.STATE / "private_inputs/new8_a00_neuroticism.npz",
    ]
    denied = install(
        reads,
        [dst / n for n in ("private_teachers.npz", "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    baseline, motion = old.arrays(BASELINE), old.arrays(motion_path)
    compiler = Compiler(192)
    points, tangents, frames, derivatives = [[] for _ in range(4)]
    groups = {i: [] for i in range(3)}
    originals, clamps, fd_abs, fd_rel = [], [], [], []
    for j in jobs:
        i = j["index"]
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        cap = old.arrays(p81.OUT / "capture" / f"a{i:02d}.npz")
        fixed = [p81.fn.clamp(x, cap["last"][q]) for q, x in enumerate(data)]
        ds = [ep.body_directions(x) for x in data]
        original, clamped = [compiler(x, ds)["frame"] for x in (data, fixed)]
        originals.append(original)
        clamps.append(clamped)
        for state, r in ((data, original), (fixed, clamped)):
            b = fn.base.features(state)
            groups[0].append((b, [np.zeros_like(x) for x in b], r, np.zeros_like(r)))
        abs_i, rel_i = [], []
        for f in range(5):
            state = fixed if f < 4 else data
            direction = (
                [d[f] for d in ds]
                if f < 4
                else [np.zeros_like(x["state"], dtype=float) for x in data]
            )
            if f == 4:
                for d in direction:
                    d[-1] = motion["v"][i]
            b, db = fn.base.features(state, [d[None] for d in direction])
            estimates = []
            for eps in (EPS, EPS / 2):
                values = [
                    compiler(changed(state, direction, step), ds)["frame"]
                    for step in (-eps, eps)
                ]
                estimates.append((values[1] - values[0]) / (2 * eps))
            gap = np.linalg.norm(estimates[0] - estimates[1], axis=1)
            rel = gap / np.maximum(np.linalg.norm(estimates[1], axis=1), 1e-6)
            if rel.max() > 0.01:
                raise ValueError("Source frame derivative half-step check failed")
            abs_i.append(gap)
            rel_i.append(rel)
            groups[1 if f < 4 else 2].append(
                (b, [x[0] for x in db], clamped if f < 4 else original, estimates[0])
            )
            limit(start, used)
        fd_abs.append(abs_i)
        fd_rel.append(rel_i)
    units = np.sqrt(np.mean(np.sum(np.stack(originals) ** 2, axis=2), axis=0))
    if not np.isfinite(units).all() or np.any(units <= 0):
        raise ValueError("Positive source frame units required")
    for gi in range(3):
        for b, db, r, dr in groups[gi]:
            points.append(fn.base.flatten(b, baseline["scale"], "grain_field"))
            tangents.append(fn.base.flatten(db, baseline["scale"], "grain_field"))
            frames.append(r / units[:, None])
            derivatives.append(dr / units[:, None])
    np.testing.assert_allclose(points, baseline["x"], atol=1e-12)
    np.testing.assert_allclose(tangents, baseline["v"], atol=1e-12)
    result = {
        "frames": np.stack(frames),
        "frame_derivatives": np.stack(derivatives),
        "original_frames": np.stack(originals) / units[None, :, None],
        "clamped_frames": np.stack(clamps) / units[None, :, None],
        "units": units,
        "fd_abs": np.asarray(fd_abs),
        "fd_relative": np.asarray(fd_rel),
    }
    result["fixed_frame"] = result["original_frames"].mean(axis=0)
    fn.frame_geometry(result["fixed_frame"])
    np.savez(dst / "private_teachers.npz", **result)
    ep.previous.finish(
        out,
        "teachers",
        ["private_teachers.npz"],
        start,
        compiled_calls=compiler.calls,
        max_fd_relative=float(result["fd_relative"].max()),
        extra_denials=len(denied) - len(probes),
        target_state_binding=True,
        native_calls=0,
    )


def models(out):
    """Exactly two coordinate solves from the pinned source teaching package."""
    old.verify(out)
    ep.freeze(out, "teachers")
    used, start = cpu(out), time.process_time()
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    names = [v + ".npz" for v in fn.VARIANTS]
    paths = [BASELINE, out / "teachers/private_teachers.npz"]
    probes = [
        previous.OUT / "analysis/private_new8_a00.npz",
        ep.STATE / "private_inputs/new8_a00_neuroticism.npz",
    ]
    denied = install(
        paths,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    baseline, t = [old.arrays(p) for p in paths]
    for moving, name in enumerate(names):
        r = (
            t["frames"]
            if moving
            else np.broadcast_to(t["fixed_frame"], t["frames"].shape)
        )
        dr = t["frame_derivatives"] if moving else np.zeros_like(r)
        y = fn.observation_coordinates(baseline["target"], r, dr, baseline["groups"])
        m = fn.fit_coordinates(baseline, y)
        m.update(fixed_frame=t["fixed_frame"], units=t["units"], moving=bool(moving))
        checks = []
        for i in range(28):
            x, dx = baseline["x"][i], baseline["v"][i]
            db, ddb = fn.base.field(baseline, x, dx)
            c, dc = fn.base.field(m, x, dx)
            _, _, check = fn.lift(db, ddb, r[i], dr[i][None], c, dc, baseline["d0"])
            checks.append(check)
        m["source_lift_checks"] = np.max(checks, axis=0)
        np.savez(dst / name, **m)
        limit(start, used)
    ep.previous.finish(
        out,
        "models",
        names,
        start,
        coefficient_solves=2,
        extra_denials=len(denied) - len(probes),
        human_labels_read=False,
    )


def predict(out, folder):
    """Read current H16 and source programs, never native recipient reader targets."""
    old.verify(out)
    ep.freeze(out, "models")
    used, start = cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "new8"]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    reads = [
        BASELINE,
        *compiled_paths(),
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
    baseline = old.arrays(BASELINE)
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    units = ms[0]["units"][:, None]
    compiler = Compiler(128)
    fa, fd, comparator = [], [], []
    for j in jobs:
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        ds = [ep.body_directions(x) for x in data]
        b, db = fn.base.features(data, ds)
        packets = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        primal, vectors = [
            np.stack([p[k] for p in packets]) for k in ("primal", "vectors")
        ]
        r = fn.frame(primal, vectors) / units
        fn.frame_geometry(r)
        derivatives, perturbed = [], []
        for f in range(4):
            values = []
            for step in (-EPS, EPS):
                state = changed(data, [d[f] for d in ds], step)
                values.append((state, compiler(state, ds)))
            derivatives.append(
                (values[1][1]["frame"] - values[0][1]["frame"]) / (2 * EPS) / units
            )
            perturbed.append(values)
        dr = np.stack(derivatives)
        saved = old.arrays(previous.OUT / "predictions" / (j["key"] + ".npz"))
        rr, dd = list(saved["readers"]), list(saved["reader_derivatives"])
        coords, checks = [], []
        for m in ms:
            rv, rd, c, check = fn.readers(baseline, m, b, db, r, dr)
            rr.append(rv)
            dd.append(rd)
            coords.append(c)
            checks.append(check)
        rr, dd = np.stack(rr), np.stack(dd)
        ps = [
            p82.projections(primal, vectors, v, d) for v, d in zip(rr, dd, strict=True)
        ]
        result = {k: np.stack([p[k] for p in ps]) for k in ps[0]}
        for k in result:
            comparator.append(float(abs(result[k][:10] - saved[k]).max()))
        if j["index"] == 0:
            for f, values in enumerate(perturbed):
                aa, zz = [], []
                for state, p in values:
                    blocks = fn.base.features(state)
                    av, zv = [], []
                    for vi, m in enumerate(ms, 10):
                        changed_r = fn.value_readers(
                            baseline, m, blocks, p["frame"] / units
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
                        - result["actions"][10:, :, :, 2, :, f]
                    ).ravel()
                )
                fd.extend(
                    abs(
                        (zz[1] - zz[0]) / (2 * EPS)
                        - result["diagonal"][10:, :, :, :, f]
                    ).ravel()
                )
        np.savez(
            dst / (j["key"] + ".npz"),
            **result,
            readers=rr,
            reader_derivatives=dd,
            frame=r,
            frame_derivatives=dr,
            coordinate_values=np.stack(coords),
            lift_checks=np.stack(checks),
        )
        limit(start, used)
    if (
        len(fa) != 128
        or len(fd) != 128
        or max(fa + fd) > 1e-6
        or max(comparator) > 1e-10
    ):
        raise ValueError("Whole-field FD or old-comparator parity failed")
    ep.previous.finish(
        out,
        folder,
        names,
        start,
        compiled_calls=compiler.calls,
        fd_reused_calls=16,
        anchored_fd_count=len(fa),
        diagonal_fd_count=len(fd),
        anchored_fd_max=float(max(fa)),
        diagonal_fd_max=float(max(fd)),
        comparator_max=max(comparator),
        native_reads=0,
        target_reads=0,
        extra_denials=len(denied) - len(probes),
    )


def summarize(rows, ms, baseline, teaching, old_tables, restricted_models):
    """Reuse the frozen metric code without pretending moving fields have fixed range."""
    indices = [10, 11, 2, 3, 4, 5, 6, 7]
    view = [
        {k: v[indices] if k in previous.VARIANT_KEYS else v for k, v in row.items()}
        for row in rows
    ]
    extra = p83.summarize(view, restricted_models)
    mapping = dict(zip(p83.VARIANTS[:2], fn.VARIANTS, strict=True))
    tables = list(old_tables)
    for idx in (2, 3, 4, 5, 6, 7):
        t = extra[idx][extra[idx]["variant"].isin(mapping)].copy()
        t["variant"] = t["variant"].map(mapping)
        tables[idx] = pd.concat((tables[idx], t), ignore_index=True)
    fit, geo, cp, checks, correction, derivatives = [[] for _ in range(6)]
    native = np.stack([r["native_r"][0] - r["native_r"][1] for r in rows])
    frames = np.stack([r["frame"] for r in rows])
    for vi, (variant, m) in enumerate(zip(fn.VARIANTS, ms, strict=True)):
        for gi, group in enumerate(fn.base.GROUPS):
            fit.append(
                {
                    "variant": variant,
                    "authors": 4,
                    "group": group,
                    "observations": int(np.sum(m["groups"] == gi)),
                    "residual_sse_over_target": float(m["group_loss"][gi]),
                    "zero_target_group": bool(m["zero_target_groups"][gi]),
                    "normal_residual_relative": float(m["normal_residual_relative"]),
                    "regularizer": float(m["regularizer"]),
                    "penalty": float(m["regularization_penalty"]),
                    "coefficients": int(m["coefficients"]),
                }
            )
        for cohort, values in (
            ("source_original", teaching["original_frames"]),
            ("source_clamped", teaching["clamped_frames"]),
            ("new8", frames),
        ):
            use = values if vi else np.broadcast_to(m["fixed_frame"], values.shape)
            g = [fn.frame_geometry(r) for r in use]
            geo.append(
                {
                    "variant": variant,
                    "cohort": cohort,
                    "authors": len(use),
                    "rank": 5,
                    "condition_max": max(x["condition"] for x in g),
                    "min_singular_value": min(x["singular_values"][-1] for x in g),
                    "max_singular_value": max(x["singular_values"][0] for x in g),
                }
            )
        use = frames if vi else np.broadcast_to(m["fixed_frame"], frames.shape)
        actual = np.einsum("nkh,nh->nk", use, native - baseline["d0"])
        pred = np.stack([r["coordinate_values"][vi] for r in rows])
        for f in range(5):
            cp.append(
                {
                    "variant": variant,
                    "authors": 8,
                    "coordinate": "scalar" if f == 0 else f"total_F{f:02d}",
                    **p82.metric(actual[:, f], pred[:, f]),
                }
            )
        for cohort, ch in (
            ("source", m["source_lift_checks"]),
            ("new8", np.max([r["lift_checks"][vi] for r in rows], axis=0)),
        ):
            checks.append(
                {
                    "variant": variant,
                    "authors": 4 if cohort == "source" else 8,
                    "cohort": cohort,
                    **dict(
                        zip(
                            (
                                "coordinate_residual_max",
                                "derivative_residual_max",
                                "null_change_max",
                            ),
                            [float(x) for x in ch],
                            strict=True,
                        )
                    ),
                }
            )
        new_d, old_d = [
            np.stack([r["readers"][i, 0] - r["readers"][i, 1] for r in rows])
            for i in (10 + vi, 9)
        ]
        delta, error = new_d - old_d, old_d - native
        correction.append(
            {
                "variant": variant,
                "authors": 8,
                "change_energy": float(np.sum(delta**2)),
                "old_error2": float(np.sum(error**2)),
                "signed_cross": float(2 * np.sum(delta * error)),
                "new_error2": float(np.sum((error + delta) ** 2)),
            }
        )
    for d in range(5):
        for f in range(5):
            derivatives.append(
                {
                    "authors": 4,
                    "direction": f"BODY_F{d + 1:02d}" if d < 4 else "context",
                    "ruler": "scalar" if f == 0 else f"total_F{f:02d}",
                    "absolute_gap_max": float(teaching["fd_abs"][:, d, f].max()),
                    "relative_gap_max": float(teaching["fd_relative"][:, d, f].max()),
                }
            )
    return [
        *tables,
        *[pd.DataFrame(x) for x in (fit, geo, cp, checks, correction, derivatives)],
    ]


def evaluate(out, folder):
    """Evaluate only after predictions freeze, using exact old native arrays."""
    old.verify(out)
    ep.freeze(out, folder)
    used, start = cpu(out), time.process_time()
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
    ms = [old.arrays(out / "models" / (v + ".npz")) for v in fn.VARIANTS]
    restricted = [
        old.arrays(p83.OUT / "models" / (v + ".npz")) for v in p83.fn.VARIANTS
    ]
    tables = summarize(
        rows,
        ms,
        old.arrays(BASELINE),
        old.arrays(out / "teachers/private_teachers.npz"),
        [pd.read_csv(previous.OUT / "analysis" / n) for n in previous.TABLES],
        restricted,
    )
    if [len(t) for t in tables] != COUNTS:
        raise ValueError("Incomplete comparison inventory")
    for name, table in zip(TABLES, tables, strict=True):
        table.to_csv(dst / name, index=False)
    limit(start, used)
    ep.previous.finish(out, dst.name, [*TABLES, *names], start, counts=COUNTS)


def audit(out):
    """Check all stage budgets, unchanged old comparisons and exact cache replay."""
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
    for n in previous.TABLES:
        a, b = [pd.read_csv(root / "analysis" / n) for root in (previous.OUT, out)]
        pd.testing.assert_frame_equal(
            a,
            b.iloc[: len(a)].reset_index(drop=True),
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    if (
        stages["teachers"]["compiled_calls"] != 176
        or stages["models"]["coefficient_solves"] != 2
        or stages["analysis"]["counts"] != COUNTS
        or cpu(out) > CAP
    ):
        raise ValueError("Finite budget or inventory differs")
    for f in ("teachers", "models", "predictions", "replay"):
        if stages[f]["extra_denials"]:
            raise ValueError("Unexpected reads")
    for f in ("predictions", "replay"):
        if (
            stages[f]["compiled_calls"] != 128
            or stages[f]["fd_reused_calls"] != 16
            or stages[f]["native_reads"]
            or stages[f]["target_reads"]
        ):
            raise ValueError("Application contract differs")
    result = {
        "status": "PASS",
        "source_authors": 4,
        "recipient_authors": 8,
        "fits": 2,
        "source_compiled_calls": 176,
        "application_compiled_calls_each": 128,
        "native_calls": 0,
        "human_labels_read": False,
        "numeric_cpu_seconds": cpu(out),
        "public_tables": len(TABLES),
        "source_fd_relative_max": stages["teachers"]["max_fd_relative"],
        **{
            k: stages["predictions"][k]
            for k in (
                "anchored_fd_count",
                "diagonal_fd_count",
                "anchored_fd_max",
                "diagonal_fd_max",
                "comparator_max",
            )
        },
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run one immutable stage per process."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
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
            {
                "prepare": prepare,
                "teachers": teachers,
                "models": models,
                "audit": audit,
            }[args.stage](args.output)


if __name__ == "__main__":
    main()
