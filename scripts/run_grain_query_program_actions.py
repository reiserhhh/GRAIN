#!/usr/bin/env python3
"""Run M07 source-frozen query-operation and independent-computation comparisons."""

from __future__ import annotations

import argparse
import collections
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    digest,
    load_functions,
    quality,
    read_json,
    tokenize_world,
    write_json,
)
from grain_research.vector_mechanisms.navigation_runtime import encode_texts
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit

DEFAULT = BASE / "mechanism_todo_v1/query_program_actions_r1"
M06 = BASE / "mechanism_todo_v1/shared_query_actions_r1"
OLD = BASE / "navigation_v1/NAV-10/reference_routes_r1"
SPAN = BASE / "navigation_v1/NAV-02/source_span_r1"
BUNDLE = BASE / "navigation_v1/NAV-11/runtime_export_r1/bundle"
CONFIG = ROOT / "configs/grain_query_program_actions.json"
PLAN = ROOT / "docs/GRAIN_QUERY_PROGRAM_ACTION_PLAN_V1.md"


def code_files():
    """Freeze the scientific runner and all newly introduced arithmetic."""
    return [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/query_program_actions.py",
        CONFIG,
        PLAN,
    ]


def prepare(out):
    """Reuse construction-only programs; freeze public inputs before any new outputs."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from scripts.run_grain_source_span_extension import load_extension

    if out.exists():
        raise FileExistsError("Choose a fresh packet directory")
    cfg, old = read_json(CONFIG), read_json(OLD / "construction_freeze.json")
    original = read_json(OUTPUT / "manifest.json")
    if digest(MODEL / "model.safetensors") != original["model_sha256"]:
        raise ValueError("Fixed observer changed")
    for n, sha in old["files"].items():
        if digest(OLD / n) != sha:
            raise ValueError("NAV10 construction artifact changed")
    models = load_extension(SPAN)
    if set(cfg["queries"]) != set(models):
        raise ValueError("Unexpected source queries")
    refs = load_functions(OUTPUT / "real_reference.npz")
    out.mkdir(parents=True)
    bundle = out / "bundle"
    package = bundle / "grain_program"
    package.mkdir(parents=True)
    modules = [
        "__init__.py",
        "query_program_actions.py",
        "navigation_runtime.py",
        "projection_program.py",
        "context_program.py",
        "affine_program.py",
        "functional_profiles.py",
        "state_geometry.py",
        "state_transport.py",
        "source_span.py",
    ]
    sources = {}

    def copy(source, target):
        sources[str(source.relative_to(ROOT))] = digest(source)
        shutil.copy2(source, target)

    for n in modules:
        copy(ROOT / "grain_research/vector_mechanisms" / n, package / n)
    prior = read_json(BUNDLE / "manifest.json")
    for n in (
        "tokenizer.json",
        "embedding_table.npy",
        "context.safetensors",
        "config.json",
    ):
        if digest(BUNDLE / n) != prior["files"][n]:
            raise ValueError("Portable source asset changed")
        copy(BUNDLE / n, bundle / n)
    copy(OUTPUT / "real_reference.npz", bundle / "reader.npz")
    if digest(bundle / "reader.npz") != old["readers"]["R_real"]:
        raise ValueError("Source readers disagree")
    specs = read_json(BUNDLE / "queries.json")
    for q in cfg["queries"]:
        specs[q]["anchor"] = refs[q].anchor
        copy(SPAN / f"{q}.npz", bundle / f"state_{q}.npz")
        copy(OLD / f"R_real_A_source_{q}.npz", bundle / f"compiled_{q}.npz")
        for i in range(2):
            for kind in ("token", "context"):
                meta, _ = checked_arrays(OLD / f"{kind}_{q}_{i}.json")
                if meta["query"] != q or meta["source_split"] != "construction":
                    raise ValueError("Chart is not a source-only artifact")
                copy(OLD / f"{kind}_{q}_{i}.npz", bundle / f"{kind}_{q}_{i}.npz")
    write_json(bundle / "queries.json", specs)
    common = [
        "tokenizer.json",
        "queries.json",
        *["grain_program/" + n for n in modules],
    ]
    deps = {
        "token": common
        + ["embedding_table.npy"]
        + [f"token_{q}_{i}.npz" for q in specs for i in range(2)],
        "context": common
        + ["context.safetensors", "config.json", "reader.npz"]
        + [f"{kind}_{q}.npz" for q in specs for kind in ("state", "compiled")]
        + [f"context_{q}_{i}.npz" for q in specs for i in range(2)],
    }
    files = {
        str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*") if p.is_file()
    }
    write_json(
        bundle / "manifest.json",
        {"files": files, "dependencies": deps, "source_hashes": sources},
    )
    worlds = [
        {**w, "id": "m06_" + w["id"], "family": "m06", "texts": [w["body"]]}
        for w in read_json(M06 / "worlds.json")
    ]
    for f in cfg["binding_frames"]:
        for high in (0, 1):
            a, b = (f["low"], f["high"]) if high == 0 else (f["high"], f["low"])
            for order in (0, 1):
                clauses = [f'I said: "{a}"', f'My colleague said: "{b}"']
                if order:
                    clauses.reverse()
                worlds.append(
                    {
                        "id": f"bind_{f['id']}_{high}_{order}",
                        "family": "binding",
                        "target": f["target"],
                        "frame": f["id"],
                        "self_high": high,
                        "order": order,
                        "texts": [f["prefix"] + "\n" + "\n".join(clauses)],
                    }
                )
    simple = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design, inputs, alphabets = [], [], {}
    for w in worlds:
        for q in cfg["queries"]:
            ids, mask = encode_texts(
                simple,
                w["texts"],
                specs[q]["recipe"],
                specs[q]["question_tokens"],
                maximum=1024,
            )
            _, _, enc, omask, alphabet = tokenize_world(tok, w["texts"], q)
            if ids != enc["input_ids"] or not np.array_equal(mask, omask):
                raise ValueError("Portable and teacher prompt bytes differ")
            i = len(inputs)
            inputs.append({"id": str(i), "texts": w["texts"], "query": q})
            design.append(
                {
                    **{k: v for k, v in w.items() if k not in ("body", "texts", "id")},
                    "id": str(i),
                    "world_id": w["id"],
                    "query": q,
                    "ids": ids,
                    "mask": mask.tolist(),
                }
            )
            alphabets[q] = alphabet
    for n, value in (
        ("inputs.json", inputs),
        ("design.json", design),
        ("worlds.json", worlds),
        ("config.json", cfg),
    ):
        write_json(out / n, value)
    write_json(
        out / "manifest.json",
        {
            "protocol": "M07_QUERY_PROGRAM_ACTION_V1",
            "source_reader": "R_real",
            "inputs": {
                n: digest(out / n)
                for n in ("inputs.json", "design.json", "worlds.json", "config.json")
            },
            "bundle_manifest_sha256": digest(bundle / "manifest.json"),
            "code": {str(p.relative_to(ROOT)): digest(p) for p in code_files()},
            "source": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    OLD / "construction_freeze.json",
                    SPAN / "construction_freeze.json",
                    M06 / "manifest.json",
                    M06 / "worlds.json",
                )
            },
            "model_sha256": original["model_sha256"],
            "alphabets": alphabets,
            "rows": len(inputs),
            "new_training": False,
            "new_human_labels": False,
        },
    )
    print(
        {"status": "PREPARED", "rows": len(inputs), "worlds": len(worlds)}, flush=True
    )


def verify(out):
    """Check the pre-run plan, code, raw inputs and source-export identities."""
    m = read_json(out / "manifest.json")
    for section in ("code", "source"):
        for n, sha in m[section].items():
            if digest(ROOT / n) != sha:
                raise ValueError("Frozen " + section + " changed: " + n)
    for n, sha in m["inputs"].items():
        if digest(out / n) != sha:
            raise ValueError("Raw packet changed")
    if digest(out / "bundle/manifest.json") != m["bundle_manifest_sha256"]:
        raise ValueError("Runtime manifest changed")
    return m


def cold(out, mode):
    """Run from another cwd with teacher/source repositories and network denied."""
    verify(out)
    dest = out / ("cold_" + mode)
    if dest.exists():
        raise FileExistsError(
            "Existing cold run retained; inspect it instead of overwriting"
        )
    wrapper = """
import json, os, pathlib, sys
bundle, inp, output, root, parent, mode = sys.argv[1:]
bundle, inp, output, root, parent = [pathlib.Path(x).resolve() for x in (bundle,inp,output,root,parent)]
denied=[]
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        raise PermissionError('Network forbidden during cold scoring')
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
        return
    p=pathlib.Path(os.fsdecode(args[0])).resolve()
    blocked=p.is_relative_to(root) or p.is_relative_to(parent)
    allowed=p.is_relative_to(bundle) or p==inp or p.is_relative_to(output) or p.is_relative_to(parent/'.venv')
    if blocked and not allowed:
        denied.append(str(p)); raise PermissionError('Teacher/source repository access forbidden')
sys.addaudithook(guard)
sys.path.insert(0,str(bundle))
from grain_program.query_program_actions import cold_run
cold_run(bundle,inp,output,mode,device='mps')
(output/'access_guard.json').write_text(json.dumps({'active':True,'blocked_source_attempts':len(denied),'network_forbidden':True}))
"""
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="grain-m07-") as cwd:
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                wrapper,
                str((out / "bundle").resolve()),
                str((out / "inputs.json").resolve()),
                str(dest.resolve()),
                str(ROOT),
                str(ROOT.parent / "project persona"),
                mode,
            ],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=660,
        )
    (out / f"cold_{mode}.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f"Cold {mode} failed; see saved log")
    print(run.stdout, flush=True)


def teacher(out):
    """Collect a fixed comparator only after both independent prediction freezes."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    m = verify(out)
    for mode in ("token", "context"):
        check_cold(out, mode)
    if digest(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Teacher changed")
    dest = out / "teacher"
    dest.mkdir(exist_ok=True)
    done = out / "teacher_status.json"
    if done.exists() and read_json(done)["status"] == "COMPLETE":
        print(read_json(done))
        return
    torch.set_num_threads(2)
    start = time.monotonic()
    obs = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=1024
    )
    obs.model.requires_grad_(False)
    refs = load_functions(out / "bundle/reader.npz")
    for d in read_json(out / "design.json"):
        path = dest / f"input_{int(d['id']):03d}.json"
        if path.exists():
            checked_arrays(path)
            continue
        if time.monotonic() - start > 600:
            write_json(done, {"status": "PAUSED_BOUND", "new_calls": obs.calls})
            return
        state, _, native = capture_boundary(obs, d["ids"], m["alphabets"][d["query"]])
        h = state[0].float().cpu().numpy()
        mask = np.asarray(d["mask"], bool)
        val = refs[d["query"]].parts(h[mask], h[-1])["score"]
        save_unit(path, {"id": d["id"], "native": native, "affine": val}, state=h)
    write_json(
        done,
        {
            "status": "COMPLETE",
            "new_calls": obs.calls,
            "rows": m["rows"],
            "seconds": time.monotonic() - start,
        },
    )
    print(read_json(done), flush=True)


def check_cold(out, mode):
    """Verify complete raw-text predictions, checked imports and retained file hashes."""
    dest = out / ("cold_" + mode)
    for n, sha in read_json(dest / "freeze.json").items():
        if digest(dest / n) != sha:
            raise ValueError("Cold prediction changed")
    guard, deps = (
        read_json(dest / "access_guard.json"),
        read_json(dest / "dependencies.json"),
    )
    if not guard["active"] or guard["blocked_source_attempts"]:
        raise ValueError("Invalid cold input boundary")
    if mode == "token" and (
        deps["torch_imported"]
        or deps["transformers_imported"]
        or deps["transformer_forward_calls"]
    ):
        raise ValueError("Token mode executed a neural program")
    return deps


def analyze(out):
    """Compare frozen levels, finite effects, query carriers and source binding."""
    from grain_research.vector_mechanisms.context_program import score_context_charts
    from grain_research.vector_mechanisms.query_program_actions import arrays

    m = verify(out)
    deps = {mode: check_cold(out, mode) for mode in ("token", "context")}
    if read_json(out / "teacher_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete the bounded teacher comparison first")
    d = pd.DataFrame(read_json(out / "design.json"))
    a = pd.DataFrame(read_json(out / "cold_token/predictions.json"))
    b = pd.DataFrame(read_json(out / "cold_context/predictions.json"))
    f = d.merge(
        a.drop(columns=["query", "tokens", "arrays"]), on="id", validate="one_to_one"
    )
    f = f.merge(
        b.drop(columns=["query", "tokens", "arrays"]), on="id", validate="one_to_one"
    )
    native, affine, mass = [], [], []
    max_replay, audit_checks = 0.0, 0
    charts = {
        q: [arrays(out / f"bundle/context_{q}_{i}.npz") for i in range(2)]
        for q in m["alphabets"]
    }
    specs = read_json(out / "bundle/queries.json")
    for _, r in f.iterrows():
        idx = int(r.id)
        tm, _ = checked_arrays(out / f"teacher/input_{idx:03d}.json")
        native.append(tm["native"]["conditional_score"])
        affine.append(tm["affine"])
        mass.append(tm["native"]["candidate_mass"])
        z = arrays(out / f"cold_context/input_{idx:03d}.npz")
        old = score_context_charts(
            charts[r.query], z["hidden"], z["mask"], specs[r.query]["anchor"]
        )["mean"]["score"]
        err = abs(old - r.context_all)
        max_replay = max(max_replay, err)
        if err > 2e-6:
            raise ValueError("Carrier decomposition changed the old B reader")
        if not np.array_equal(z["ids"], r.ids) or not np.array_equal(
            z["mask"], r["mask"]
        ):
            raise ValueError("Cold prompt and teacher differ")
        audit_checks += 2
    f["native"], f["upper_affine"], f["candidate_mass"] = native, affine, mass
    methods = [
        "token_uniform",
        "token_semantic_order",
        "A_state",
        "A_compiled",
        "context_all",
        "context_C",
        "context_PQ",
    ]
    levels = []
    for (family, query), g in f.groupby(["family", "query"]):
        for target in ("native", "upper_affine"):
            for method in methods:
                levels.append(
                    {
                        "family": family,
                        "query": query,
                        "target": target,
                        "method": method,
                        **quality(g[target], g[method]),
                    }
                )
    effects = []
    quantities = [*methods, "native", "upper_affine", "B_C", "B_P", "B_Q", "A_P", "A_Q"]
    for _, lo in f.iterrows():
        if lo.family == "m06":
            for factor in ("arousal", "dejection"):
                if lo[factor] != 0:
                    continue
                other = "dejection" if factor == "arousal" else "arousal"
                g = f[
                    (f.family == "m06")
                    & (f.frame == lo.frame)
                    & (f.owner == lo.owner)
                    & (f["query"] == lo["query"])
                    & (f[other] == lo[other])
                    & (f[factor] == 1)
                ]
                hi = g.iloc[0]
                effects.append(
                    {
                        "kind": factor,
                        "frame": lo.frame,
                        "owner": lo.owner,
                        "query": lo["query"],
                        "base": lo.id,
                        "donor": hi.id,
                        **{n: hi[n] - lo[n] for n in quantities},
                    }
                )
        else:
            for factor, kind in (("self_high", "binding"), ("order", "order")):
                if lo[factor] != 0:
                    continue
                other = "order" if factor == "self_high" else "self_high"
                g = f[
                    (f.family == "binding")
                    & (f.frame == lo.frame)
                    & (f["query"] == lo["query"])
                    & (f[other] == lo[other])
                    & (f[factor] == 1)
                ]
                hi = g.iloc[0]
                effects.append(
                    {
                        "kind": kind,
                        "frame": lo.frame,
                        "owner": "mixed",
                        "query": lo["query"],
                        "target": lo.target,
                        "base": lo.id,
                        "donor": hi.id,
                        "bag_equal": collections.Counter(lo.ids)
                        == collections.Counter(hi.ids),
                        **{n: hi[n] - lo[n] for n in quantities},
                    }
                )
    e = pd.DataFrame(effects)
    metrics = []
    for (kind, query), g in e.groupby(["kind", "query"]):
        for method in methods:
            metrics.append(
                {
                    "kind": kind,
                    "query": query,
                    "method": method,
                    **quality(g.native, g[method]),
                    "mean_native_delta": g.native.mean(),
                    "mean_program_delta": g[method].mean(),
                    "native_rms": np.sqrt(np.mean(g.native**2)),
                    "program_rms": np.sqrt(np.mean(g[method] ** 2)),
                    "sign_agreement": np.mean((g.native > 0) == (g[method] > 0)),
                }
            )
    f.drop(columns=["ids", "mask"]).to_csv(out / "levels.csv", index=False)
    pd.DataFrame(levels).to_csv(out / "level_metrics.csv", index=False)
    e.to_csv(out / "effects.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "effect_metrics.csv", index=False)
    e.groupby(["kind", "frame", "owner", "query"])[quantities].mean().to_csv(
        out / "operation_map.csv"
    )
    audit = {
        "status": "PASS",
        "raw_ids_and_old_B_checks": audit_checks,
        "max_old_B_error": max_replay,
        "max_A_equivalence_error": float(abs(f.A_state - f.A_compiled).max()),
        "dependencies": deps,
        "independent_researcher": False,
        "new_human_labels": False,
    }
    write_json(out / "audit.json", audit)
    summary = {
        "status": "COMPLETE",
        "rows": len(f),
        "effects": len(e),
        "teacher": read_json(out / "teacher_status.json"),
        "min_candidate_mass": float(f.candidate_mass.min()),
        "binding_bag_equal": int(e[e.kind == "binding"].bag_equal.sum()),
        "binding_pairs": int((e.kind == "binding").sum()),
        "new_training": False,
    }
    write_json(out / "summary.json", summary)
    names = [
        "levels.csv",
        "level_metrics.csv",
        "effects.csv",
        "effect_metrics.csv",
        "operation_map.csv",
        "audit.json",
        "summary.json",
    ]
    write_json(out / "analysis_freeze.json", {n: digest(out / n) for n in names})
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "token", "context", "teacher", "analyze")
    )
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage in ("token", "context"):
        cold(args.out, args.stage)
    else:
        globals()[args.stage](args.out)
