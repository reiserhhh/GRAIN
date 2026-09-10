#!/usr/bin/env python3
"""NEXT55: frozen source maps inside the complete fixed-logit suffix."""

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
from grain_research.vector_mechanisms.compiled_query_transfer import QUERIES, fidelity
from grain_research.vector_mechanisms.source_state_transport import (
    apply_map,
    array_hash,
    body_directions,
    source_set,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.whole_path_assembly import (
    CANDIDATES,
    error_account,
    paired_queries,
    reverse_response,
    score_metrics,
)
from scripts.run_grain_source_state_transport import (
    BASE,
    ENDPOINT,
    arrays,
    forbidden,
    frozen,
    read,
    ref,
    sha,
    verify,
    verify_hashes,
    write,
)

OLD = BASE / "logic_response_aware_state_v1"
PLAN = ROOT / "docs/GRAIN_WHOLE_PATH_ASSEMBLY_PLAN_V1.md"


def prepare(out):
    """Freeze exact prior maps and pure inputs; export hashes, never current targets."""
    old = verify(OLD)
    nm = read(ENDPOINT / "manifest.json")
    jobs = read(OLD / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Exact twelve frozen paired materials required")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    (out / "private_inputs").symlink_to(OLD / "private_inputs")
    (out / "private_expected").mkdir()
    (out / "private_models").mkdir()
    write(out / "private_jobs.json", jobs)
    files = [out / "private_jobs.json"]
    inputs = [OLD / "manifest.json", ENDPOINT / "manifest.json", PLAN, Path(__file__)]
    refs = sorted({ref(j) for j in jobs})
    for objective in CANDIDATES[1:]:
        fm = frozen(OLD / objective, "models")
        frozen(OLD / objective, "predictions")
        for r in refs:
            name = r + "_pointwise.npz"
            if name not in fm["files"]:
                raise ValueError("Fixed pointwise map missing")
            src = OLD / objective / "models" / name
            dst = out / "private_models" / (objective + "_" + name)
            dst.symlink_to(src)
            files.append(dst)
            inputs.append(src)
        inputs += [
            OLD / objective / n / "freeze.json" for n in ("models", "predictions")
        ]
    for j in jobs:
        for key in j["query_keys"]:
            data = out / "private_inputs" / (key + ".npz")
            if set(arrays(data)) != {
                "state",
                "grain",
                "positions",
                "cos",
                "sin",
                "mask",
            }:
                raise ValueError("Only pure H16/GRAIN/layout fields required")
            p = [OLD / o / "predictions" / (key + ".npz") for o in CANDIDATES[1:]]
            a, b = [arrays(x) for x in p]
            dst = out / "private_expected" / (key + ".npz")
            np.savez(
                dst,
                **{
                    f: np.asarray([a[f][0], a[f][1], b[f][1]])
                    for f in ("x_hash", "d_hash")
                },
            )
            files += [data, dst]
            inputs += p
    evaluation = {}
    for folder in ("capture_source", "capture_new"):
        f = frozen(ENDPOINT, folder)
        evaluation.update(
            {
                str((ENDPOINT / folder / n).relative_to(ROOT)): h
                for n, h in f["files"].items()
            }
        )
        inputs.append(ENDPOINT / folder / "freeze.json")
    for p in (
        "grain_research/vector_mechanisms/whole_path_assembly.py",
        "grain_research/vector_mechanisms/selection_endpoint.py",
        "grain_research/vector_mechanisms/context_carrier.py",
        "grain_research/vector_mechanisms/stage_file_guard.py",
        "scripts/run_grain_complete_transport_jvp.py",
        "tests/test_grain_whole_path_assembly.py",
    ):
        inputs.append(ROOT / p)
    lineage = dict(old["inputs"])
    lineage.update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
    m = {
        "protocol": "GRAIN_WHOLE_PATH_ASSEMBLY_V1",
        "candidates": CANDIDATES,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "block": 22,
        "alphabet": nm["alphabet"],
        "c": nm["c"],
        "model_sha256": nm["model_sha256"],
        "max_native_seconds": 180,
        "max_numeric_seconds": 60,
        "max_reverse_pairs": 72,
        "max_jvps": 2,
        "max_fd_forwards": 4,
        **{
            n: nm[n]
            for n in ("fd_epsilon", "fd_atol", "fd_rtol", "jvp_atol", "jvp_rtol")
        },
        "inputs": lineage,
        "prepared": {str(p.relative_to(out)): sha(p) for p in files},
        "evaluation": evaluation,
        "prepare_seconds": time.monotonic() - started,
        "fits": 0,
        "human_labels_read": False,
    }
    write(out / "manifest.json", m)
    print(
        {
            "status": "PREPARED",
            "units": 24,
            "maps": 10,
            "seconds": m["prepare_seconds"],
        },
        flush=True,
    )


def predict(out):
    """Run each declared candidate once; stop and retain progress at any boundary."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from grain_research.vector_mechanisms.selection_endpoint import endpoint
    from scripts.run_grain_complete_transport_jvp import load_observer

    m = verify(out)
    dst = out / "predictions"
    dst.mkdir(exist_ok=False)
    jobs = read(out / "private_jobs.json")
    observer, loading = load_observer(m)
    torch.set_grad_enabled(False)
    # Framework initialization only, not a model forward or a scientific observation.
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
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    counters = {"native_seconds": 0.0, "reverse_pairs": 0, "jvps": 0, "fd_forwards": 0}
    started, completed, checks = time.monotonic(), [], []

    def save_status(status):
        """Write only actual completed packets and accumulated native call costs."""
        wall = time.monotonic() - started
        result = {
            "status": status,
            **counters,
            "wall_seconds": wall,
            "numeric_seconds": wall - counters["native_seconds"] + m["prepare_seconds"],
            "loading_seconds": loading,
            "completed_units": len(completed),
            "fits": 0,
            "human_labels_read": False,
            "current_target_reads": 0,
            "denial_probes": 4,
            "extra_denials": len(denied) - 4,
            "checks": checks.copy(),
            "files": {p.name: sha(p) for p in completed},
        }
        write(dst / "status.json", result)
        print(
            {
                k: result[k]
                for k in (
                    "status",
                    "completed_units",
                    "reverse_pairs",
                    "jvps",
                    "fd_forwards",
                    "native_seconds",
                    "numeric_seconds",
                )
            },
            flush=True,
        )
        return result

    def timed(fn, kind):
        """Bound all candidate native calls including parity probes, with no retry."""
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
            models = {
                o: arrays(out / "private_models" / f"{o}_{ref(j)}_pointwise.npz")
                for o in CANDIDATES[1:]
            }
            for q, key in zip(QUERIES, j["query_keys"], strict=True):
                data = arrays(out / "private_inputs" / (key + ".npz"))
                expected = arrays(out / "private_expected" / (key + ".npz"))
                x, d = data["state"].astype(float), body_directions(data)
                values, gradients, responses, state_hashes, tangent_hashes = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for ai, candidate in enumerate(CANDIDATES):
                    mx, md = (
                        (x, d) if ai == 0 else apply_map(x, d, models[candidate])[:2]
                    )
                    if array_hash(mx) != str(expected["x_hash"][ai]) or array_hash(
                        md
                    ) != str(expected["d_hash"][ai]):
                        raise ValueError("Frozen same-function reconstruction changed")
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
                    values.append(value)
                    gradients.append(gradient)
                    responses.append(tau)
                    state_hashes.append(array_hash(mx.astype(np.float32)))
                    tangent_hashes.append(array_hash(md.astype(np.float32)))
                    if candidate == "joint" and j["index"] == 0 and q == QUERIES[0]:
                        _, deriv = timed(
                            lambda state=state, directions=directions, kw=kw: (
                                torch.func.jvp(
                                    lambda h: endpoint(observer.model, h, kw, ids, c),
                                    (state,),
                                    (directions[:1],),
                                )
                            ),
                            "jvps",
                        )
                        jvp = float(deriv.cpu())
                        np.testing.assert_allclose(
                            jvp, tau[0], atol=m["jvp_atol"], rtol=m["jvp_rtol"]
                        )
                        eps = m["fd_epsilon"]
                        finite = []
                        for sign in (1, -1):
                            # Check the full composition from the original H16 perturbation.
                            perturbed = apply_map(
                                x + sign * eps * d[0], d, models[candidate]
                            )[0]
                            px = torch.tensor(
                                perturbed[None], device="mps", dtype=torch.float32
                            )
                            v = timed(
                                lambda px=px, kw=kw: endpoint(
                                    observer.model, px, kw, ids, c
                                ),
                                "fd_forwards",
                            )
                            finite.append(float(v.cpu()))
                        fd = (finite[0] - finite[1]) / (2 * eps)
                        checks.append(
                            {
                                "cohort": j["cohort"],
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
                    del state, directions, gradient, mx, md
                p = dst / (key + ".npz")
                np.savez(
                    p,
                    score=np.asarray(values),
                    tau=np.asarray(responses),
                    gradient=np.asarray(gradients),
                    x32_hash=np.asarray(state_hashes),
                    d32_hash=np.asarray(tangent_hashes),
                )
                completed.append(p)
                status = save_status("RUNNING")
                if (
                    status["native_seconds"] > m["max_native_seconds"]
                    or status["numeric_seconds"] > m["max_numeric_seconds"]
                ):
                    raise TimeoutError("Native/numerical compute boundary reached")
        if (
            counters["reverse_pairs"],
            counters["jvps"],
            counters["fd_forwards"],
            len(completed),
        ) != (72, 2, 4, 24):
            raise ValueError("Complete native inventory differs")
        result = save_status("COMPLETE")
        write(dst / "freeze.json", result)
    except TimeoutError:
        save_status("PAUSED_RESOURCE_BOUNDARY")
    except Exception:
        save_status("ERROR")
        raise


def evaluate(out, name):
    """Compare frozen whole outputs to old native targets, never refit or recenter."""
    m = verify(out)
    frozen(out, "predictions")
    verify_hashes(ROOT, m["evaluation"])
    dst = out / name
    dst.mkdir(exist_ok=False)
    jobs = read(out / "private_jobs.json")
    targets = {}
    for j in jobs:
        folder = "capture_source" if j["cohort"] == "source4" else "capture_new"
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            targets[j["cohort"], j["index"], q] = arrays(
                ENDPOINT / folder / (key + ".npz")
            )
    scores, actions = [], []
    for j in jobs:
        members = source_set(j["cohort"], j["index"])
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            native = targets[j["cohort"], j["index"], q]
            constant = float(
                np.mean([targets["source4", i, q]["score"] for i in members])
            )
            p = arrays(out / "predictions" / (key + ".npz"))
            if p["score"].shape != (3,) or p["tau"].shape != (3, 4):
                raise ValueError("Candidate/factor inventory differs")
            for ai, candidate in enumerate(CANDIDATES):
                row = {
                    "author_key": j["key"],
                    "cohort": j["cohort"],
                    "query_name": q,
                    "candidate": candidate,
                }
                scores.append(
                    row
                    | {
                        "actual": float(native["score"]),
                        "predicted": p["score"][ai],
                        "constant": constant,
                    }
                )
                for f in range(4):
                    actions.append(
                        row
                        | {
                            "factor": f"whole:F{f + 1:02d}",
                            "actual": native["tau"][f],
                            "predicted": p["tau"][ai, f],
                        }
                    )
    sd, ad = pd.DataFrame(scores), pd.DataFrame(actions)
    keys = ["author_key", "cohort", "candidate"]
    sc = paired_queries(sd, keys, ["actual", "predicted", "constant"])
    ac = paired_queries(ad, keys + ["factor"], ["actual", "predicted"])
    if (len(sd), len(ad), len(sc), len(ac)) != (72, 288, 36, 144):
        raise ValueError("Whole score/response paired rows differ")
    files, summary_tables = [], {}

    def save(tab, filename):
        """Save deterministic aggregate or explicitly private tables."""
        tab.to_csv(dst / filename, index=False)
        files.append(filename)

    for frame, n in (
        (sd, "scores"),
        (ad, "actions"),
        (sc, "score_contrasts"),
        (ac, "action_contrasts"),
    ):
        save(frame, "private_" + n + ".csv")
    for frame, group, n, score in (
        (sd, ["cohort", "query_name", "candidate"], "scores", True),
        (sc, ["cohort", "candidate"], "score_contrasts", True),
        (ad, ["cohort", "query_name", "candidate"], "actions", False),
        (ad, ["cohort", "query_name", "candidate", "factor"], "factor_actions", False),
        (ac, ["cohort", "candidate"], "action_contrasts", False),
        (ac, ["cohort", "candidate", "factor"], "factor_contrasts", False),
    ):
        rows = []
        for key, g in frame.groupby(group, sort=True):
            metrics = (
                score_metrics(g.actual, g.predicted, g.constant)
                if score
                else fidelity(g.actual.to_numpy(), g.predicted.to_numpy())
            )
            rows.append(
                dict(zip(group, key, strict=True))
                | metrics
                | {"authors": g.author_key.nunique()}
            )
        summary_tables[n] = pd.DataFrame(rows)
        save(summary_tables[n], n + ".csv")
    for frame, group, n in (
        (sd, ["cohort", "query_name"], "score_error_changes"),
        (sc, ["cohort"], "score_contrast_error_changes"),
        (ad, ["cohort", "query_name"], "action_error_changes"),
        (ad, ["cohort", "query_name", "factor"], "factor_error_changes"),
        (ac, ["cohort"], "action_contrast_error_changes"),
        (ac, ["cohort", "factor"], "factor_contrast_error_changes"),
    ):
        rows = []
        for key, g in frame.groupby(group, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            idx = ["author_key"] + (["factor"] if "factor" in g else [])
            slots = {
                a: g[g.candidate == a].set_index(idx).sort_index() for a in CANDIDATES
            }
            for a, b in (
                ("identity", "normalized_value"),
                ("identity", "joint"),
                ("normalized_value", "joint"),
            ):
                if not slots[a].index.equals(slots[b].index):
                    raise ValueError("Candidate pairing mismatch")
                np.testing.assert_array_equal(slots[a].actual, slots[b].actual)
                r = error_account(
                    slots[a].actual, slots[a].predicted, slots[b].predicted
                )
                rows.append(
                    dict(zip(group, key, strict=True))
                    | {"baseline": a, "candidate": b}
                    | r
                )
        save(pd.DataFrame(rows), n + ".csv")
    decisions = {}
    for a in CANDIDATES:
        s = summary_tables["scores"].query("cohort == 'new8' and candidate == @a")
        r = summary_tables["actions"].query("cohort == 'new8' and candidate == @a")
        if len(s) != 2 or len(r) != 2:
            raise ValueError("Two-query decision inventory differs")
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
        "score_rows": 72,
        "action_rows": 288,
        "decisions": decisions,
        "new_fits": 0,
        "new_native_calls": 0,
        "human_labels_read": False,
        "files": {n: sha(dst / n) for n in files},
    }
    write(dst / "summary.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def main():
    """Keep input preparation, candidate execution and target evaluation separate."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    p.add_argument("--output", type=Path, default=BASE / "logic_whole_path_assembly_v1")
    p.add_argument("--name", default="analysis")
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(a.output)
        elif a.stage == "predict":
            predict(a.output)
        else:
            evaluate(a.output, a.name)


if __name__ == "__main__":
    main()
