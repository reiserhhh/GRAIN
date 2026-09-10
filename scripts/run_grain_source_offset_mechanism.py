#!/usr/bin/env python3
"""NEXT56: two source-defined ingredient functions through the complete suffix."""

from __future__ import annotations

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
from grain_research.vector_mechanisms.source_offset import (
    CANDIDATES,
    NEW_CANDIDATES,
    effect_summary,
    family,
    ingredient_effects,
    offset_geometry,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_whole_path_assembly import (
    BASE,
    ENDPOINT,
    QUERIES,
    array_hash,
    arrays,
    body_directions,
    fidelity,
    forbidden,
    frozen,
    paired_queries,
    read,
    ref,
    reverse_response,
    score_metrics,
    sha,
    source_set,
    verify,
    verify_hashes,
    write,
)

OLD = BASE / "logic_whole_path_assembly_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_OFFSET_MECHANISM_PLAN_V1.md"


def prepare(out):
    """Freeze source-only functions and pure H16 inputs before new suffix calls."""
    old, pf = verify(OLD), frozen(OLD, "predictions")
    jobs = read(OLD / "private_jobs.json")
    if len(jobs) != 12 or len(pf["files"]) != 24:
        raise ValueError("Twelve paired materials and complete NEXT55 required")
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    (out / "private_inputs").symlink_to(OLD / "private_inputs")
    (out / "private_models").symlink_to(OLD / "private_models")
    write(out / "private_jobs.json", jobs)
    files = [out / "private_jobs.json"]
    for r in sorted({ref(j) for j in jobs}):
        files.append(out / "private_models" / f"joint_{r}_pointwise.npz")
    for j in jobs:
        model = arrays(out / "private_models" / f"joint_{ref(j)}_pointwise.npz")
        for k in j["query_keys"]:
            path = out / "private_inputs" / (k + ".npz")
            data = arrays(path)
            if set(data) != {"state", "grain", "positions", "cos", "sin", "mask"}:
                raise ValueError("Pure H16/layout packet required")
            family(data["state"].astype(float), body_directions(data), model)
            files.append(path)
    lineage = dict(old["inputs"])
    for p in (
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/source_offset.py",
        ROOT / "tests/test_grain_source_offset.py",
        OLD / "manifest.json",
        OLD / "predictions/freeze.json",
    ):
        lineage[str(p.relative_to(ROOT))] = sha(p)
    m = {
        **{
            k: old[k]
            for k in (
                "alphabet",
                "c",
                "model_sha256",
                "fd_epsilon",
                "fd_atol",
                "fd_rtol",
                "jvp_atol",
                "jvp_rtol",
            )
        },
        "protocol": "GRAIN_SOURCE_OFFSET_MECHANISM_V1",
        "block": 22,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidates": CANDIDATES,
        "new_candidates": NEW_CANDIDATES,
        "max_reverse_pairs": 48,
        "max_jvps": 2,
        "max_fd_forwards": 4,
        "max_native_seconds": 120,
        "max_numeric_seconds": 60,
        "inputs": lineage,
        "prepared": {str(p.relative_to(out)): sha(p) for p in files},
        "evaluation": old["evaluation"],
        "reused_candidates": {
            str((OLD / "predictions" / n).relative_to(ROOT)): h
            for n, h in pf["files"].items()
        },
        "prepare_seconds": time.monotonic() - started,
        "fits": 0,
        "human_labels_read": False,
    }
    write(out / "manifest.json", m)
    print(
        {"status": "PREPARED", "units": 24, "maps": 5, "seconds": m["prepare_seconds"]},
        flush=True,
    )


def predict(out):
    """Run two compatible new functions once, preserving partial results on a cap."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from grain_research.vector_mechanisms.selection_endpoint import endpoint
    from scripts.run_grain_complete_transport_jvp import load_observer

    m, dst = verify(out), out / "predictions"
    dst.mkdir(exist_ok=False)
    jobs = read(out / "private_jobs.json")
    observer, loading = load_observer(m)
    torch.set_grad_enabled(False)
    torch.func.jvp(
        lambda a: a * a, (torch.ones(2, device="mps"),), (torch.ones(2, device="mps"),)
    )
    ids = torch.tensor(m["alphabet"]["token_ids"], device="mps")
    c = torch.tensor(m["c"], device="mps", dtype=torch.float32)
    paths = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    denied = install(
        [out / n for n in m["prepared"]],
        paths + [dst / "status.json", dst / "freeze.json"],
        forbidden(),
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    counters = {"native_seconds": 0.0, "reverse_pairs": 0, "jvps": 0, "fd_forwards": 0}
    started, completed, checks = time.monotonic(), [], []

    def status(kind):
        wall = time.monotonic() - started
        r = {
            "status": kind,
            **counters,
            "wall_seconds": wall,
            "numeric_seconds": wall - counters["native_seconds"] + m["prepare_seconds"],
            "loading_seconds": loading,
            "completed_units": len(completed),
            "fits": 0,
            "human_labels_read": False,
            "current_target_reads": 0,
            "reused_output_reads": 0,
            "denial_probes": 4,
            "extra_denials": len(denied) - 4,
            "checks": checks.copy(),
            "files": {p.name: sha(p) for p in completed},
        }
        write(dst / "status.json", r)
        print(
            {
                k: r[k]
                for k in (
                    "status",
                    "completed_units",
                    "reverse_pairs",
                    "native_seconds",
                    "numeric_seconds",
                )
            },
            flush=True,
        )
        return r

    def timed(fn, kind):
        if (
            counters["native_seconds"] >= m["max_native_seconds"]
            or counters[kind] >= m["max_" + kind]
        ):
            raise TimeoutError("Native compute cap reached")
        torch.mps.synchronize()
        start = time.monotonic()
        counters[kind] += 1
        try:
            return fn()
        finally:
            torch.mps.synchronize()
            counters["native_seconds"] += time.monotonic() - start

    try:
        for j in jobs:
            model = arrays(out / "private_models" / f"joint_{ref(j)}_pointwise.npz")
            for q, key in zip(QUERIES, j["query_keys"], strict=True):
                data = arrays(out / "private_inputs" / (key + ".npz"))
                x, d = data["state"].astype(float), body_directions(data)
                f = family(x, d, model)
                scores, gradients, actions, xhash, dhash = [], [], [], [], []
                for candidate in NEW_CANDIDATES:
                    mx, md = f[candidate]
                    state = torch.tensor(mx[None], device="mps", dtype=torch.float32)
                    directions = torch.tensor(md, device="mps", dtype=torch.float32)
                    kw = cached_suffix_kwargs(observer.model, state)
                    for actual, name in zip(
                        kw["position_embeddings"], ("cos", "sin"), strict=True
                    ):
                        np.testing.assert_array_equal(
                            actual.cpu().numpy()[0], data[name]
                        )
                    np.testing.assert_array_equal(
                        kw["attention_mask"][0, 0, -1].cpu().numpy(), data["mask"]
                    )
                    value, gradient, tau = timed(
                        lambda state=state, directions=directions, kw=kw: (
                            reverse_response(
                                observer.model, state, directions, kw, ids, c
                            )
                        ),
                        "reverse_pairs",
                    )
                    scores.append(value)
                    gradients.append(gradient)
                    actions.append(tau)
                    xhash.append(array_hash(mx.astype(np.float32)))
                    dhash.append(array_hash(md.astype(np.float32)))
                    check = (
                        "offset_only"
                        if j["cohort"] == "source4"
                        else "nonintercept_only"
                    )
                    if j["index"] == 0 and q == QUERIES[0] and candidate == check:
                        _, deriv = timed(
                            lambda state=state, directions=directions, kw=kw: (
                                torch.func.jvp(
                                    lambda z: endpoint(observer.model, z, kw, ids, c),
                                    (state,),
                                    (directions[:1],),
                                )
                            ),
                            "jvps",
                        )
                        jvp, eps, fs = float(deriv.cpu()), m["fd_epsilon"], []
                        np.testing.assert_allclose(
                            jvp, tau[0], atol=m["jvp_atol"], rtol=m["jvp_rtol"]
                        )
                        for sign in (1, -1):
                            px = torch.tensor(
                                family(x + sign * eps * d[0], d, model)[candidate][0][
                                    None
                                ],
                                device="mps",
                                dtype=torch.float32,
                            )
                            v = timed(
                                lambda px=px, kw=kw: endpoint(
                                    observer.model, px, kw, ids, c
                                ),
                                "fd_forwards",
                            )
                            fs.append(float(v.cpu()))
                        fd = (fs[0] - fs[1]) / (2 * eps)
                        checks.append(
                            {
                                "cohort": j["cohort"],
                                "candidate": candidate,
                                "query": q,
                                "factor": "whole:F01",
                                "reverse": float(tau[0]),
                                "jvp": jvp,
                                "fd": fd,
                            }
                        )
                        np.testing.assert_allclose(
                            fd, tau[0], atol=m["fd_atol"], rtol=m["fd_rtol"]
                        )
                p = dst / (key + ".npz")
                np.savez(
                    p,
                    score=np.asarray(scores),
                    tau=np.asarray(actions),
                    gradient=np.asarray(gradients),
                    x32_hash=np.asarray(xhash),
                    d32_hash=np.asarray(dhash),
                )
                completed.append(p)
                r = status("RUNNING")
                if (
                    r["native_seconds"] > m["max_native_seconds"]
                    or r["numeric_seconds"] > m["max_numeric_seconds"]
                ):
                    raise TimeoutError("Execution boundary reached")
        if (
            counters["reverse_pairs"],
            counters["jvps"],
            counters["fd_forwards"],
            len(completed),
        ) != (48, 2, 4, 24):
            raise ValueError("New native inventory differs")
        write(dst / "freeze.json", status("COMPLETE"))
    except TimeoutError:
        status("PAUSED_RESOURCE_BOUNDARY")
    except Exception:
        status("ERROR")
        raise


def evaluate(out, name):
    """Join frozen complete outputs, compare ingredients and never recalibrate."""
    m = verify(out)
    frozen(out, "predictions")
    verify_hashes(ROOT, m["evaluation"])
    verify_hashes(ROOT, m["reused_candidates"])
    dst = out / name
    dst.mkdir(exist_ok=False)
    jobs = read(out / "private_jobs.json")
    targets = {}
    for j in jobs:
        folder = "capture_source" if j["cohort"] == "source4" else "capture_new"
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            targets[j["cohort"], j["index"], q] = arrays(
                ENDPOINT / folder / (k + ".npz")
            )
    scores, actions, geometry, energy = [], [], [], []
    for j in jobs:
        model = arrays(out / "private_models" / f"joint_{ref(j)}_pointwise.npz")
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            data = arrays(out / "private_inputs" / (k + ".npz"))
            x, d = data["state"].astype(float), body_directions(data)
            f = family(x, d, model)
            old, p = (
                arrays(OLD / "predictions" / (k + ".npz")),
                arrays(out / "predictions" / (k + ".npz")),
            )
            packets = [(old, 0), (p, 0), (p, 1), (old, 2)]
            native = targets[j["cohort"], j["index"], q]
            const = float(
                np.mean(
                    [
                        targets["source4", i, q]["score"]
                        for i in source_set(j["cohort"], j["index"])
                    ]
                )
            )
            meta = {"author_key": j["key"], "cohort": j["cohort"], "query_name": q}
            energy.append(meta | offset_geometry(x, model, f["joint"][0]))
            for a, (p, ai) in zip(CANDIDATES, packets, strict=True):
                mx, md = f[a]
                if array_hash(mx.astype(np.float32)) != str(
                    p["x32_hash"][ai]
                ) or array_hash(md.astype(np.float32)) != str(p["d32_hash"][ai]):
                    raise ValueError("Original or new composed state changed")
                scores.append(
                    meta
                    | {
                        "candidate": a,
                        "actual": float(native["score"]),
                        "predicted": p["score"][ai],
                        "constant": const,
                    }
                )
                for fi in range(4):
                    actions.append(
                        meta
                        | {
                            "candidate": a,
                            "factor": f"whole:F{fi + 1:02d}",
                            "actual": native["tau"][fi],
                            "predicted": p["tau"][ai, fi],
                        }
                    )
                g = p["gradient"][ai].astype(float)
                geo = meta | {
                    "candidate": a,
                    "tangent_norm_ratio": float(np.linalg.norm(md) / np.linalg.norm(d)),
                    "tangent_change_relative": float(
                        np.linalg.norm(md - d) / np.linalg.norm(d)
                    ),
                }
                for index, label in ((0, "identity"), (2, "joint")):
                    reference = old["gradient"][index].astype(float)
                    geo.update(
                        {
                            f"gradient_norm_over_{label}": float(
                                np.linalg.norm(g) / np.linalg.norm(reference)
                            ),
                            f"gradient_cosine_with_{label}": float(
                                np.sum(g * reference)
                                / (np.linalg.norm(g) * np.linalg.norm(reference))
                            ),
                            f"gradient_relative_difference_to_{label}": float(
                                np.linalg.norm(g - reference)
                                / np.linalg.norm(reference)
                            ),
                        }
                    )
                geometry.append(geo)
    sd, ad = pd.DataFrame(scores), pd.DataFrame(actions)
    keys = ["author_key", "cohort", "candidate"]
    sc, ac = (
        paired_queries(sd, keys, ["actual", "predicted", "constant"]),
        paired_queries(ad, keys + ["factor"], ["actual", "predicted"]),
    )
    if (len(sd), len(ad), len(sc), len(ac)) != (96, 384, 48, 192):
        raise ValueError("Four-function paired inventory differs")
    files, tables = [], {}

    def save(tab, n):
        tab.to_csv(dst / (n + ".csv"), index=False)
        files.append(n + ".csv")
        tables[n] = tab

    for tab, n in (
        (sd, "scores"),
        (ad, "actions"),
        (sc, "score_contrasts"),
        (ac, "action_contrasts"),
    ):
        save(tab, "private_" + n)
    for rows, n, group in (
        (energy, "offset_geometry", ["cohort", "query_name"]),
        (geometry, "gradient_geometry", ["cohort", "query_name", "candidate"]),
    ):
        tab = pd.DataFrame(rows)
        save(tab, "private_" + n)
        save(tab.drop(columns="author_key").groupby(group).mean().reset_index(), n)
    definitions = (
        (sd, ["cohort", "query_name"], "scores", True),
        (sc, ["cohort"], "score_contrasts", True),
        (ad, ["cohort", "query_name"], "actions", False),
        (ad, ["cohort", "query_name", "factor"], "factor_actions", False),
        (ac, ["cohort"], "action_contrasts", False),
        (ac, ["cohort", "factor"], "factor_contrasts", False),
    )
    for tab, group, n, is_score in definitions:
        rows = []
        for k, g in tab.groupby(group + ["candidate"], sort=True):
            r = (
                score_metrics(g.actual, g.predicted, g.constant)
                if is_score
                else fidelity(g.actual.to_numpy(), g.predicted.to_numpy())
            )
            rows.append(
                dict(zip(group + ["candidate"], k, strict=True))
                | r
                | {"authors": g.author_key.nunique()}
            )
        save(pd.DataFrame(rows), n)
        unit = ["author_key", *group] + (
            ["factor"] if not is_score and "factor" not in group else []
        )
        effects = []
        for k, g in tab.groupby(unit, sort=True):
            v = g.set_index("candidate").loc[list(CANDIDATES)]
            np.testing.assert_allclose(v.actual, v.actual.iloc[0], atol=0)
            a = ingredient_effects(v.predicted.to_numpy(), np.asarray(v.actual.iloc[0]))
            effects.append(
                dict(zip(unit, k, strict=True)) | {f: float(z) for f, z in a.items()}
            )
        ef = pd.DataFrame(effects)
        save(ef, "private_" + n + "_effects")
        fields = list(ingredient_effects(np.zeros(4), np.asarray(0.0)))
        rows = []
        for k, g in ef.groupby(group, sort=True):
            k = k if isinstance(k, tuple) else (k,)
            rows.append(
                dict(zip(group, k, strict=True))
                | effect_summary({f: g[f].to_numpy() for f in fields})
                | {"authors": g.author_key.nunique()}
            )
        save(pd.DataFrame(rows), n + "_effects")
    decisions = {}
    for a in CANDIDATES:
        s = tables["scores"].query("cohort=='new8' and candidate==@a")
        r = tables["actions"].query("cohort=='new8' and candidate==@a")
        assert len(s) == len(r) == 2
        decisions[a] = {
            "score_usable": bool(
                ((s.score_r > 0) & (s.score_mse_over_source_constant < 1)).all()
            ),
            "response_usable": bool(
                ((r.action_r > 0) & (r.action_mse_over_zero < 1)).all()
            ),
        }
    result = {
        "status": "COMPLETE",
        "score_rows": 96,
        "action_rows": 384,
        "decisions": decisions,
        "new_native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "files": {n: sha(dst / n) for n in files},
    }
    write(dst / "summary.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_source_offset_mechanism_v1"
    )
    p.add_argument("--name", default="analysis")
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(a.output)
        elif a.stage == "predict":
            predict(a.output)
        else:
            evaluate(a.output, a.name)
