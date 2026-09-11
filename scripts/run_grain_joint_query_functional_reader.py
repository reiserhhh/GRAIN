#!/usr/bin/env python3
"""NEXT89 source12 paired absolute teaching, numeric application and replay."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms import joint_query_reader as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_diversity_teaching as previous

old, ep, p85 = previous.old, previous.ep, previous.p85
OUT = old.BASE / "logic_joint_query_functional_reader_v1"
PLAN = ROOT / "docs/GRAIN_JOINT_QUERY_FUNCTIONAL_READER_PLAN_V1.md"
SOURCE_MODEL = previous.OUT / "models/original12_native_coordinates.npz"
SOURCE_TEACHERS = previous.OUT / "teachers/private_teachers.npz"
CORE = previous.CORE
TABLES = (
    *CORE,
    "source_fit.csv",
    "coordinate_prediction.csv",
    "frame_geometry.csv",
    "pair_mean.csv",
    "signed_errors.csv",
    "conditioning.csv",
    "reader_pair.csv",
)
COUNTS = [552, 1656, 6624, 299, 23, 2208, 10, 20, 4, 96, 48, 1, 3]
CAP, EPS = 120, 1e-5


def limit(start, used):
    """Bound numeric work including the exact replay, never based on outcomes."""
    if used + time.process_time() - start > CAP:
        raise TimeoutError("NEXT89 cumulative numerical CPU budget")


def prepare(out):
    """Freeze code, source package and old evaluation-only membership."""
    m = old.verify(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT88 required")
    inputs = dict(m["inputs"])
    for folder in ("capture", "teachers", "models", "predictions", "analysis"):
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
        ROOT / "scripts/report_grain_joint_query_functional_reader.py",
        ROOT / "scripts/analyze_grain_joint_query_reader.py",
        ROOT / "tests/test_grain_joint_query_reader.py",
    ):
        inputs[str(p.relative_to(ROOT))] = old.sha(p)
    start = time.process_time()
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", old.read(previous.OUT / "private_jobs.json"))
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_JOINT_QUERY_FUNCTIONAL_READER_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "prepare_cpu_seconds": time.process_time() - start,
            "source_authors": 12,
            "recipient_authors": 8,
            "source_solves": 1,
            "coefficients": 120,
            "max_cpu_seconds": CAP,
            "compiled_query_calls_per_application": 128,
            "native_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED"}, flush=True)


def teachers(out):
    """Construct source-only absolute and native-coordinate targets."""
    old.verify(out)
    used, start = p85.cpu(out), time.process_time()
    dst = out / "teachers"
    dst.mkdir(exist_ok=False)
    probes = [
        previous.OUT / "analysis/private_new8_a00.npz",
        p85.OUT / "predictions/new8_a00.npz",
    ]
    denied = install(
        [SOURCE_MODEL, SOURCE_TEACHERS, p85.BASELINE],
        [dst / n for n in ("private_teachers.npz", "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    source, old_t, b = [
        old.arrays(p) for p in (SOURCE_MODEL, SOURCE_TEACHERS, p85.BASELINE)
    ]
    t = fn.teaching(
        old_t["current"], old_t["native"], old_t["native_readers"], b, source["units"]
    )
    np.testing.assert_allclose(
        t["target_values"][:, 0] - t["target_values"][:, 1],
        old_t["contrast"],
        atol=1e-12,
    )
    np.savez(dst / "private_teachers.npz", **t, points=old_t["points"])
    limit(start, used)
    ep.previous.finish(
        out,
        "teachers",
        ["private_teachers.npz"],
        start,
        source_authors=12,
        extra_denials=len(denied) - len(probes),
        compiled_calls=0,
        native_calls=0,
    )


def models(out):
    """One fixed source12 solve for ten functional observations."""
    old.verify(out)
    ep.freeze(out, "teachers")
    used, start = p85.cpu(out), time.process_time()
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    name = fn.VARIANT + ".npz"
    probes = [
        previous.OUT / "analysis/private_new8_a00.npz",
        p85.OUT / "predictions/new8_a00.npz",
    ]
    denied = install(
        [SOURCE_MODEL, out / "teachers/private_teachers.npz"],
        [dst / name, dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    s, t = old.arrays(SOURCE_MODEL), old.arrays(out / "teachers/private_teachers.npz")
    np.savez(dst / name, **fn.fit(s, t["points"], t["compatible_coordinates"]))
    limit(start, used)
    ep.previous.finish(
        out,
        "models",
        [name],
        start,
        source_solves=1,
        coefficients=120,
        extra_denials=len(denied) - len(probes),
        human_labels_read=False,
    )


def predict(out, folder):
    """Apply without native targets; compute per-query rather than mean frames."""
    old.verify(out)
    ep.freeze(out, "models")
    used, start = p85.cpu(out), time.process_time()
    jobs = [j for j in old.read(out / "private_jobs.json") if j["cohort"] == "new8"]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".npz" for j in jobs]
    paths = [
        p85.BASELINE,
        out / "models" / (fn.VARIANT + ".npz"),
        *p85.compiled_paths(),
    ]
    paths += [
        root / (k + ".npz")
        for j in jobs
        for k in j["query_keys"]
        for root in (ep.STATE / "private_inputs", ep.OUT / "predictions")
    ]
    probes = [
        previous.OUT / "analysis/private_new8_a00.npz",
        ep.ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        SOURCE_TEACHERS,
    ]
    denied = install(
        paths,
        [dst / n for n in (*names, "freeze.json")],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    b, m = old.arrays(p85.BASELINE), old.arrays(out / "models" / (fn.VARIANT + ".npz"))
    compiler = p85.Compiler(128)
    fa, fd = [], []
    for ji, j in enumerate(jobs):
        data = [
            old.arrays(ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        ds = [ep.body_directions(d) for d in data]
        blocks, tangents = fn.base.features(data, ds)
        packets = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        p, v = [np.stack([z[k] for z in packets]) for k in ("primal", "vectors")]
        frame = fn.frames(p, v, m["units"])
        changes, derivatives = [], []
        for f in range(4):
            pair = []
            for step in (-EPS, EPS):
                state = p85.changed(data, [d[f] for d in ds], step)
                pack = compiler(state, ds)
                rr = fn.frames(pack["primal"], pack["vectors"], m["units"])
                pair.append((state, pack, rr))
            derivatives.append((pair[1][2] - pair[0][2]) / (2 * EPS))
            changes.append(pair)
        dr = np.stack(derivatives, axis=1)
        rr, rd, c, dc, checks = fn.readers(b, m, blocks, tangents, frame, dr)
        result = fn.projections(p, v, rr, rd)
        if ji == 0:
            for f, pair in enumerate(changes):
                aa, zz = [], []
                for state, pack, rf in pair:
                    rv = fn.value_readers(b, m, fn.base.features(state), rf)
                    aa.append(np.einsum("qah,qmh->qam", pack["primal"], fn.modes(rr)))
                    zz.append(np.einsum("qah,qmh->qam", pack["primal"], fn.modes(rv)))
                fa.extend(
                    abs(
                        (aa[1] - aa[0]) / (2 * EPS) - result["actions"][:, :, 2, :, f]
                    ).ravel()
                )
                fd.extend(
                    abs(
                        (zz[1] - zz[0]) / (2 * EPS) - result["diagonal"][:, :, :, f]
                    ).ravel()
                )
        np.savez(
            dst / (j["key"] + ".npz"),
            **{k: a[None] for k, a in result.items()},
            readers=rr[None],
            reader_derivatives=rd[None],
            coordinate_values=c[None],
            coordinate_derivatives=dc[None],
            frame=frame,
            frame_derivatives=dr,
            lift_checks=checks,
            frame_conditions=np.asarray(
                [fn.coord.frame_geometry(a)["condition"] for a in frame]
            ),
        )
        limit(start, used)
        if ji == 1:
            elapsed = time.process_time() - start
            estimate = elapsed * 4 * (2 if folder == "predictions" else 1)
            print(
                {
                    "completed_authors": 2,
                    "cpu_seconds": elapsed,
                    "remaining_pass_estimate": estimate,
                    "compiled_calls": compiler.calls,
                },
                flush=True,
            )
            if used + estimate > CAP:
                raise TimeoutError(
                    "Cost-only two-author gate; complete units preserved"
                )
    if len(fa) != 64 or len(fd) != 64 or max(fa + fd) > 1e-6:
        raise ValueError("Anchored/shared/full-function derivative check failed")
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


def main():
    """Isolate each source/application/evaluation stage in a fresh process."""
    from scripts import analyze_grain_joint_query_reader as analysis

    parser = argparse.ArgumentParser()
    parser.add_argument(
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
    parser.add_argument("--output", type=Path, default=OUT)
    a = parser.parse_args()
    with threadpool_limits(limits=1):
        if a.stage in ("predict", "replay"):
            predict(a.output, "predictions" if a.stage == "predict" else "replay")
        elif a.stage in ("evaluate", "replay-evaluate"):
            analysis.evaluate(
                a.output, "predictions" if a.stage == "evaluate" else "replay"
            )
        else:
            {
                "prepare": prepare,
                "teachers": teachers,
                "models": models,
                "audit": analysis.audit,
            }[a.stage](a.output)


if __name__ == "__main__":
    main()
