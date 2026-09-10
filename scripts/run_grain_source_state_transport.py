#!/usr/bin/env python3
"""NEXT52: source primal-state maps with their own exact directional derivatives."""

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
from grain_research.vector_mechanisms.selection_endpoint import key_vectors
from grain_research.vector_mechanisms.source_query_selection import (
    anchors,
    evidence_inputs,
)
from grain_research.vector_mechanisms.source_state_transport import (
    ARMS,
    apply_map,
    array_hash,
    body_directions,
    fit_maps,
    source_set,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_selection_endpoint import frozen
from scripts.run_grain_source_query_selection import (
    BASE,
    SOURCE,
    arrays,
    read,
    sha,
    verify,
    verify_hashes,
    write,
)

ANCHOR = BASE / "logic_source_query_selection_v1"
ENDPOINT = BASE / "logic_selection_endpoint_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_STATE_TRANSPORT_PLAN_V1.md"


def ref(j):
    """Name a fixed source reference, never a query/outcome-selected model."""
    return f"exclude{j['index']}" if j["cohort"] == "source4" else "all4"


def selected(path, fields):
    """Read only named NPZ fields, avoiding co-located current target matrices."""
    with np.load(path, allow_pickle=False) as z:
        return {n: z[n] for n in fields}


def prepare(out):
    """Export pure H16/layout inputs and keep source-only primal targets separate."""
    old = verify(ENDPOINT)
    frozen(ENDPOINT, "predictions")
    result = read(ENDPOINT / "analysis/summary.json")
    verify_hashes(ENDPOINT / "analysis", result["files"])
    if result["status"] != "COMPLETE":
        raise ValueError("Complete NEXT51 required")
    jobs = read(ANCHOR / "private_jobs.json")
    if len(jobs) != 12 or sum(j["cohort"] == "source4" for j in jobs) != 4:
        raise ValueError("Exact source4/new8 inventory required")
    out.mkdir(exist_ok=False)
    for folder in ("private_inputs", "private_training", "private_fixed"):
        (out / folder).mkdir()
    started = time.monotonic()
    inputs, files = [], []
    bank = arrays(ANCHOR / "private_source_bank.npz")
    inputs += [ANCHOR / "private_source_bank.npz", ENDPOINT / "predictions/freeze.json"]
    for j in jobs:
        prior = ENDPOINT / "predictions" / (j["key"] + ".npz")
        readers = selected(prior, ("readers",))["readers"]
        inputs.append(prior)
        qs = []
        for query, key in zip(QUERIES, j["query_keys"], strict=True):
            hpath = SOURCE / "private_inputs" / (key + ".npz")
            cpath = SOURCE / "private_capture" / (key + ".npz")
            data = arrays(hpath)
            layout = selected(cpath, ("cos", "sin", "mask"))
            if set(data) != {"state", "grain", "positions"}:
                raise ValueError("Pure H16 input schema required")
            if layout["mask"].shape != (len(data["state"]),):
                raise ValueError("Current fixed token layout mismatch")
            path = out / "private_inputs" / (key + ".npz")
            np.savez(path, **data, **layout)
            files.append(path)
            inputs.extend([hpath, cpath])
            if j["cohort"] == "source4":
                path = out / "private_training" / (key + ".npz")
                np.savez(path, y=selected(cpath, ("x",))["x"])
                files.append(path)
            q, members = anchors(bank, j["cohort"], j["index"], query)
            if members != source_set(j["cohort"], j["index"]):
                raise ValueError("Source anchor/map mismatch")
            qs.append(q[0])
        path = out / "private_fixed" / (j["key"] + ".npz")
        np.savez(path, q=np.stack(qs), r=readers)
        files.append(path)
    wpath = SOURCE / "private_weights.npz"
    path = out / "private_weights.npz"
    # The current query projection is deliberately not part of this program.
    w = arrays(wpath)
    np.savez(path, **{k: v for k, v in w.items() if k not in ("q", "q_bias")})
    files.append(path)
    inputs.append(wpath)
    write(out / "private_jobs.json", jobs)
    files.append(out / "private_jobs.json")
    inputs += [
        PLAN,
        Path(__file__),
        ROOT / "tests/test_grain_source_state_transport.py",
    ]
    for stem in (
        "source_state_transport",
        "selection_endpoint",
        "source_query_selection",
        "upstream_attention_channels",
        "native_material_reader",
        "stage_file_guard",
    ):
        inputs.append(ROOT / "grain_research/vector_mechanisms" / (stem + ".py"))
    lineage = dict(old["inputs"])
    lineage.update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
    seconds = time.monotonic() - started
    if seconds > 180:
        raise TimeoutError("Preparation budget exceeded")
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_STATE_TRANSPORT_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "rank": 32,
            "alpha": 0.01,
            "seed": 52,
            "max_fits": 10,
            "max_seconds": 180,
            "prepare_seconds": seconds,
            "native_calls": 0,
            "human_labels_read": False,
            "current_inputs": "H16, GRAIN BODY directions, fixed token layout only",
            "storage": "Lossless Xhat=X16+ZB, Dhat=D+DZB with exact full-array hashes",
            "inputs": lineage,
            "prepared": {str(p.relative_to(out)): sha(p) for p in files},
        },
    )
    print({"status": "PREPARED", "seconds": seconds, "materials": 24}, flush=True)


def finish(dst, paths, started, denied, **extra):
    """Freeze exactly completed outputs and computational counters."""
    result = {
        "status": "COMPLETE",
        "seconds": time.monotonic() - started,
        "native_calls": 0,
        "human_labels_read": False,
        "denial_probes": 4,
        "extra_denials": len(denied) - 4,
        **extra,
        "files": {p.name: sha(p) for p in paths},
    }
    write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k != "files"}, flush=True)


def forbidden():
    """Representative current intermediate/endpoint targets that must stay closed."""
    return [
        SOURCE / "private_capture/new8_a00_neuroticism.npz",
        ENDPOINT / "capture_new/new8_a00_neuroticism.npz",
        ENDPOINT / "analysis/private_actions.csv",
        SOURCE / "predictions/new8_a00.npz",
    ]


def fit(out):
    """Fit ten source-only primal maps, with no target-current state access."""
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    src = [j for j in jobs if j["cohort"] == "source4"]
    refs = {ref(j): source_set(j["cohort"], j["index"]) for j in jobs}
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    paths = [dst / f"{r}_{a}.npz" for r in sorted(refs) for a in ARMS[1:]]
    reads = [
        out / d / (k + ".npz")
        for j in src
        for k in j["query_keys"]
        for d in ("private_inputs", "private_training")
    ]
    denied = install(
        reads,
        paths + [dst / "freeze.json"],
        forbidden(),
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    started = time.monotonic()
    items = []
    for j in src:
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            items.append(
                {
                    "index": j["index"],
                    "query": q,
                    "x": arrays(out / "private_inputs" / (k + ".npz"))["state"],
                    "y": arrays(out / "private_training" / (k + ".npz"))["y"],
                }
            )
    for r in sorted(refs):
        maps = fit_maps(items, refs[r], m["rank"], m["alpha"], m["seed"])
        for a, model in maps.items():
            np.savez(dst / f"{r}_{a}.npz", **model)
        if m["prepare_seconds"] + time.monotonic() - started > m["max_seconds"]:
            raise TimeoutError("Source map budget exceeded; no grid/retry")
    finish(
        dst,
        paths,
        started,
        denied,
        fits=len(paths),
        sources=refs,
        source_materials=len(items),
        source_tokens=sum(len(i["x"]) for i in items),
    )


def predict(out, name):
    """Freeze factorized states, exact tangents and endpoints before target evaluation."""
    m = verify(out)
    fm = frozen(out, "models")
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    paths = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    reads = [out / "private_weights.npz"]
    reads += [out / "models" / n for n in fm["files"]]
    reads += [out / "private_inputs" / p.name for p in paths]
    reads += [out / "private_fixed" / (j["key"] + ".npz") for j in jobs]
    denied = install(
        reads,
        paths + [dst / "freeze.json"],
        forbidden(),
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    started = time.monotonic()
    weights = arrays(reads[0])
    for j in jobs:
        fixed = arrays(out / "private_fixed" / (j["key"] + ".npz"))
        models = {a: arrays(out / "models" / f"{ref(j)}_{a}.npz") for a in ARMS[1:]}
        for qi, k in enumerate(j["query_keys"]):
            data = arrays(out / "private_inputs" / (k + ".npz"))
            x, d = data["state"].astype(float), body_directions(data)
            payload, vectors, actions, xhash, dhash = {}, [], [], [], []
            for arm in ARMS:
                if arm == "identity":
                    xx, dd = x, d
                else:
                    xx, dd, z, dz = apply_map(x, d, models[arm])
                    payload[arm + "_z"], payload[arm + "_dz"] = z, dz
                xhash.append(array_hash(xx))
                dhash.append(array_hash(dd))
                cap = {n: data[n] for n in ("cos", "sin", "mask")} | {"x": xx, "d": dd}
                evidence = evidence_inputs(cap, weights, np.zeros((2, x.shape[1])))
                del evidence["g"]
                u = key_vectors(fixed["q"][qi][None], evidence, weights["o"])[0]
                vectors.append(u)
                actions.append(u @ fixed["r"][qi])
            np.savez(
                dst / (k + ".npz"),
                **payload,
                vectors=np.stack(vectors),
                actions=np.stack(actions),
                x_hash=np.asarray(xhash),
                d_hash=np.asarray(dhash),
            )
            seconds = m["prepare_seconds"] + fm["seconds"] + time.monotonic() - started
            if seconds > m["max_seconds"]:
                raise TimeoutError("Construction/prediction budget exceeded")
    finish(
        dst,
        paths,
        started,
        denied,
        fits=0,
        current_target_reads=0,
        primary_upstream_seconds=m["prepare_seconds"] + fm["seconds"],
    )


def reconstruct(data, packet, models, ai):
    """Rebuild exact frozen candidate arrays, without estimating from opened targets."""
    x, d = data["state"].astype(float), body_directions(data)
    arm = ARMS[ai]
    if arm != "identity":
        x = x + packet[arm + "_z"] @ models[arm]["coef"]
        d = d + packet[arm + "_dz"] @ models[arm]["coef"]
    if array_hash(x) != str(packet["x_hash"][ai]) or array_hash(d) != str(
        packet["d_hash"][ai]
    ):
        raise ValueError("Frozen full state/direction reconstruction changed")
    return x, d


def evaluate(out, name, predictions):
    """Open verified native targets only after all candidate outputs are frozen."""
    verify(out)
    frozen(out, "models")
    pf = frozen(out, predictions)
    prior = read(ENDPOINT / "analysis/summary.json")
    verify_hashes(ENDPOINT / "analysis", prior["files"])
    targets = pd.read_csv(ENDPOINT / "analysis/private_actions.csv")
    targets = targets[targets.arm == "source_own_Q"].set_index(
        ["author_key", "query_name", "factor"]
    )
    if len(targets) != 96 or not targets.index.is_unique:
        raise ValueError("Unique NEXT51 endpoint targets required")
    dst = out / name
    dst.mkdir(exist_ok=False)
    rows, geometry = [], []
    for j in read(out / "private_jobs.json"):
        models = {a: arrays(out / "models" / f"{ref(j)}_{a}.npz") for a in ARMS[1:]}
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            p = arrays(out / predictions / (k + ".npz"))
            data = arrays(out / "private_inputs" / (k + ".npz"))
            native = selected(SOURCE / "private_capture" / (k + ".npz"), ("x", "d"))
            base_d = body_directions(data)
            xden = float(np.mean(native["x"].astype(float) ** 2))
            xbase = float(
                np.mean((data["state"].astype(float) - native["x"].astype(float)) ** 2)
            )
            for ai, arm in enumerate((*ARMS, "native_input_reference")):
                meta = {
                    "author_key": j["key"],
                    "cohort": j["cohort"],
                    "query_name": q,
                    "arm": arm,
                    "eligible": arm in ARMS,
                }
                if arm in ARMS:
                    xx, dd = reconstruct(data, p, models, ai)
                    xmse = float(np.mean((xx - native["x"]) ** 2))
                    for f in range(4):
                        y = native["d"][f].astype(float)
                        dmse = float(np.mean((dd[f] - y) ** 2))
                        geometry.append(
                            meta
                            | {
                                "factor": f"whole:F{f + 1:02d}",
                                "tokens": len(xx),
                                "state_mse_over_zero": xmse / xden,
                                "state_mse_over_identity": xmse / xbase,
                                "tangent_mse_over_zero": dmse / float(np.mean(y * y)),
                                "tangent_mse_over_identity": dmse
                                / float(np.mean((base_d[f] - y) ** 2)),
                                "final_query_tangent_norm": float(
                                    np.linalg.norm(dd[f, -1])
                                ),
                                "native_final_query_tangent_norm": float(
                                    np.linalg.norm(y[-1])
                                ),
                            }
                        )
                for f in range(4):
                    factor = f"whole:F{f + 1:02d}"
                    target = targets.loc[j["key"], q, factor]
                    predicted = p["actions"][ai, f] if arm in ARMS else target.compiled
                    rows.append(
                        meta
                        | {
                            "factor": factor,
                            "actual": float(target.actual),
                            "predicted": float(predicted),
                        }
                    )
    df, gf = pd.DataFrame(rows), pd.DataFrame(geometry)
    idx = ["author_key", "cohort", "arm", "factor", "eligible"]
    cf = (
        df[df.query_name == QUERIES[0]].set_index(idx)[["actual", "predicted"]]
        - df[df.query_name == QUERIES[1]].set_index(idx)[["actual", "predicted"]]
    ).reset_index()
    if len(df) != 384 or len(cf) != 192 or cf.isna().any().any():
        raise ValueError("Fixed paired inventory mismatch")
    files = []

    def save(frame, filename):
        """Write a deterministic table for replay comparison."""
        frame.to_csv(dst / filename, index=False)
        files.append(filename)

    save(df, "private_actions.csv")
    save(cf, "private_contrasts.csv")
    save(gf, "private_geometry.csv")
    summaries = []
    for frame, groups, fn in (
        (df, ["cohort", "arm", "query_name"], "query_summary.csv"),
        (df, ["cohort", "arm", "query_name", "factor"], "factor_summary.csv"),
        (cf, ["cohort", "arm"], "contrast_summary.csv"),
        (cf, ["cohort", "arm", "factor"], "contrast_factors.csv"),
    ):
        tab = pd.DataFrame(
            [
                dict(zip(groups, key, strict=True))
                | {
                    "authors": g.author_key.nunique(),
                    "eligible": bool(g.eligible.all()),
                }
                | fidelity(g.actual.to_numpy(), g.predicted.to_numpy())
                for key, g in frame.groupby(groups, sort=True)
            ]
        )
        save(tab, fn)
        summaries.append(tab)
    cols = [c for c in gf if c.endswith(("_zero", "_identity", "_norm"))]
    # Equal author/material averages, not a pooled token-row pseudo sample.
    tab = (
        gf.groupby(["cohort", "arm", "query_name"], sort=True)[cols]
        .mean()
        .reset_index()
    )
    save(tab, "geometry_summary.csv")
    save(
        gf.groupby(["cohort", "arm", "query_name", "factor"], sort=True)[cols]
        .mean()
        .reset_index(),
        "geometry_factors.csv",
    )
    primary = summaries[0][summaries[0].cohort == "new8"]
    decision = {
        arm: bool(
            len(primary[primary.arm == arm]) == 2
            and (primary[primary.arm == arm].action_mse_over_zero < 1).all()
        )
        for arm in ARMS
    }
    write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "retained": decision,
            "actions": len(df),
            "contrasts": len(cf),
            "maps": 10,
            "native_calls": 0,
            "human_labels_read": False,
            "prediction_seconds": pf["seconds"],
            "files": {n: sha(dst / n) for n in files},
        },
    )
    print({"status": "COMPLETE", "retained": decision, "actions": len(df)}, flush=True)


def main():
    """Run isolated source-fit, current-predict and post-freeze evaluation stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "predict", "evaluate"))
    parser.add_argument(
        "--output", type=Path, default=BASE / "logic_source_state_transport_v1"
    )
    parser.add_argument("--name")
    parser.add_argument("--predictions", default="predictions")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == "prepare":
            prepare(args.output)
        elif args.stage == "fit":
            fit(args.output)
        elif args.stage == "predict":
            predict(args.output, args.name or "predictions")
        else:
            evaluate(args.output, args.name or "analysis", args.predictions)


if __name__ == "__main__":
    main()
