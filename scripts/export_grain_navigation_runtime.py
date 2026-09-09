#!/usr/bin/env python3
"""NAV-11: package frozen arithmetic and test raw-text replay with source access denied."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    read_json,
    tokenize_world,
    write_json,
)
from grain_research.vector_mechanisms.navigation_runtime import encode_texts
from grain_research.vector_mechanisms.state_geometry import compile_projection
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_state_transport import load_operators

DEFAULT = BASE / "navigation_v1/NAV-11/runtime_export_r1"


def prepare(out):
    """Copy local, construction-only artifacts; bundle contains no author datasets."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    if not out.resolve().is_relative_to((BASE / "navigation_v1").resolve()):
        raise ValueError("Keep the private replay packet in ignored navigation outputs")
    if out.exists():
        raise FileExistsError("Use a fresh export ID")
    token = BASE / "navigation_v1/NAV-03/input_pullback_r1"
    context = BASE / "navigation_v1/NAV-03/context_pullback_r2"
    state = BASE / "navigation_v1/NAV-02/state_transport_r1"
    tp, cp = read_json(token / "program.json"), read_json(context / "program.json")
    _sp, operators, fs = load_operators(state)
    if digest(token / "embedding_table.npy") != tp["embedding_sha256"]:
        raise ValueError("Static table changed")
    if digest(context / "context.safetensors") != cp["weights_sha256"]:
        raise ValueError("Context weights changed")
    qfs = load_functions(OUTPUT / "synthetic_reference.npz")
    out.mkdir(parents=True)
    bundle = out / "bundle"
    bundle.mkdir()
    for sub in ("grain_runtime", "token_charts", "context_charts", "compiled"):
        (bundle / sub).mkdir()
    modules = [
        "__init__.py",
        "navigation_runtime.py",
        "projection_program.py",
        "context_program.py",
        "state_geometry.py",
        "state_transport.py",
    ]
    for name in modules:
        shutil.copy2(
            ROOT / "grain_research/vector_mechanisms" / name,
            bundle / "grain_runtime" / name,
        )
    shutil.copy2(ROOT / "scripts/run_grain_navigation_runtime.py", bundle / "run.py")
    shutil.copy2(
        ROOT / "docs/GRAIN_NAVIGATION_RUNTIME_BUILD_V1.md", bundle / "BUILD_AND_SCOPE.md"
    )
    shutil.copy2(token / "embedding_table.npy", bundle / "embedding_table.npy")
    for name in ("context.safetensors", "config.json", "tokenizer.json"):
        shutil.copy2(state / name, bundle / name)
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    simple = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    specs = {}
    for q in QUERIES:
        t, spans, _enc, _mask, _alphabet = tokenize_world(
            tok, ["GRAIN_SENTINEL_A", "GRAIN_SENTINEL_B"], q
        )
        recipe = {
            "prefix": t[: spans[0][0]],
            "separator": t[spans[0][1] : spans[1][0]].replace(
                "Excerpt 2:", "Excerpt {index}:"
            ),
            "suffix": t[spans[1][1] :],
        }
        recipe["question_offset_in_suffix"] = (
            recipe["suffix"].rindex("\nQuestion: ") + 1
        )
        specs[q] = {
            "recipe": recipe,
            "question_tokens": tp["question_tokens"][q],
            "anchor": tp["anchors"][q],
            "direction": "F positive" if q == "TF_cont" else "higher nominal target",
        }
        if not tp["anchors"][q] == cp["anchors"][q] == qfs[q].anchor == fs[q].anchor:
            raise ValueError("Candidate construction references differ")
        for n in (1, 2, 12):
            bodies = [
                f"Text {i}: I said 'Question: x'. 日本語も含む。" for i in range(n)
            ]
            _text, _spans, original, original_mask, _alphabet = tokenize_world(
                tok, bodies, q
            )
            ids, mask = encode_texts(simple, bodies, recipe, tp["question_tokens"][q])
            if ids != original["input_ids"] or not np.array_equal(mask, original_mask):
                raise ValueError("Portable raw-text rendering differs")
        for i in range(2):
            for folder, filename, dest in (
                (token, f"{q}_{i}", bundle / "token_charts" / f"{q}_{i}.npz"),
                (context, f"{q}_{i}_3", bundle / "context_charts" / f"{q}_{i}.npz"),
            ):
                meta, _z = checked_arrays(folder / "charts" / (filename + ".json"))
                if meta["source_split"] != "construction" or meta["query"] != q:
                    raise ValueError("Nonconstruction chart")
                shutil.copy2(folder / "charts" / (filename + ".npz"), dest)
        np.savez_compressed(
            bundle / "compiled" / f"{q}.npz", **compile_projection(operators[q], fs[q])
        )
    write_json(bundle / "queries.json", specs)
    code = ["run.py", *["grain_runtime/" + n for n in modules]]
    common = [*code, "queries.json", "tokenizer.json"]
    deps = {
        "token": common
        + ["embedding_table.npy"]
        + [f"token_charts/{q}_{i}.npz" for q in QUERIES for i in range(2)],
        "context4": common
        + ["context.safetensors", "config.json"]
        + [f"context_charts/{q}_{i}.npz" for q in QUERIES for i in range(2)],
    }
    deps["state1"] = (
        common
        + ["context.safetensors", "config.json"]
        + [f"compiled/{q}.npz" for q in QUERIES]
    )
    deps["state2"] = deps["state1"]
    files = {
        str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*") if p.is_file()
    }
    write_json(
        bundle / "manifest.json",
        {
            "protocol": "NAV11_PORTABLE_PROGRAM_V1",
            "files": files,
            "program_dependencies": deps,
            "runtime_classes": {
                "token": "transformer-free; pretrained token table and synthetic Jacobian charts",
                "context4": "four extracted Transformer blocks plus explicit derivative projections",
                "state1": "four extracted blocks plus compiled linear-kernel state operator",
                "state2": "four extracted blocks plus compiled quadratic-kernel state operator",
            },
            "source_program_hashes": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    token / "program.json",
                    context / "program.json",
                    state / "program.json",
                )
            },
            "construction_used_frozen_LLM": True,
            "construction_used_human_labels": False,
            "compiled_state_parameters_are_negative_transfer_candidates": True,
            "original_LLM_or_PQ_required_runtime": False,
            "private_author_data_in_bundle": False,
            "distribution": "local research artifact; pretrained-weight licensing still applies",
            "installed_versions": {
                n: importlib.metadata.version(n)
                for n in (
                    "numpy",
                    "scipy",
                    "tokenizers",
                    "threadpoolctl",
                    "safetensors",
                    "torch",
                    "transformers",
                )
            },
        },
    )
    # Replays stay outside the transferable bundle: two real authors and three held worlds.
    natural = BASE / "navigation_v1/NAV-04/natural_evidence_r1"
    nw = [
        w
        for w in read_json(natural / "private_worlds.json")
        if w["condition"] == "original"
    ][:2]
    sw = [
        w
        for w in read_json(OUTPUT / "worlds.json")
        if w["split"] == "held_scene"
        and w["actor"] == "self"
        and w["opportunity"] == "voluntary"
        and w["realization"] == 0
    ]
    token_world = pd.read_csv(token / "predictions.csv").query(
        "method == 'semantic_order'"
    )
    ctx_world = pd.read_csv(
        BASE / "navigation_v1/NAV-10/context_worlds_r2/private_predictions.csv"
    ).query("method == 'context4_mean'")
    st_world = pd.read_csv(
        BASE / "navigation_v1/NAV-10/state_worlds_r1/predictions.csv"
    )
    prog_nat = pd.read_csv(
        BASE / "navigation_v1/NAV-04/natural_programs_r1/predictions.csv"
    )
    state_nat = pd.read_csv(
        BASE / "navigation_v1/NAV-10/state_natural_r1/predictions.csv"
    )
    examples, expected = [], []
    for i, (w, is_natural) in enumerate(
        [(w, True) for w in nw] + [(w, False) for w in sw]
    ):
        for q in QUERIES:
            key = f"case{i}_{q}"
            examples.append({"id": key, "query": q, "texts": w["texts"]})
            pools = {
                "token": prog_nat.query("readout == 'token_semantic_order'")
                if is_natural
                else token_world,
                "context4": prog_nat.query("readout == 'context4_mean'")
                if is_natural
                else ctx_world,
                "state1": (state_nat if is_natural else st_world).query(
                    "readout == 'transport1'"
                ),
                "state2": (state_nat if is_natural else st_world).query(
                    "readout == 'transport2'"
                ),
            }
            for program, data in pools.items():
                row = data[(data.world_id == w["world_id"]) & (data["query"] == q)]
                if len(row) != 1:
                    raise ValueError("Replay expectation is missing or duplicated")
                expected.append(
                    {"id": key, "program": program, "score": float(row.iloc[0].score)}
                )
    (out / "private_inputs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in examples),
        encoding="utf-8",
    )
    write_json(out / "expected.json", expected)
    write_json(
        out / "export.json",
        {
            "status": "EXPORTED",
            "bundle_manifest_sha256": digest(bundle / "manifest.json"),
            "input_sha256": digest(out / "private_inputs.jsonl"),
            "expected_sha256": digest(out / "expected.json"),
            "replay_rows_per_program": len(examples),
            "absolute_score_tolerance": 1e-5,
            "source_access_during_replay": "deny other source repo paths; bundle and explicit input/output only",
            "file_bytes": sum((bundle / n).stat().st_size for n in files),
            "teacher_states_supplied": False,
            "human_criteria_joined": False,
        },
    )
    print(
        {"status": "EXPORTED", "bundle": str(bundle), "replay_rows": len(examples)},
        flush=True,
    )


def replay(out, program):
    """Execute the exported CLI from another cwd with source access denied by audit hook."""
    m = read_json(out / "export.json")
    bundle = out / "bundle"
    for path, key in (
        (bundle / "manifest.json", "bundle_manifest_sha256"),
        (out / "private_inputs.jsonl", "input_sha256"),
        (out / "expected.json", "expected_sha256"),
    ):
        if digest(path) != m[key]:
            raise ValueError("Frozen replay changed")
    dest = out / f"replay_{program}"
    # This guard denies both repositories, except the exact bundle/input/output.
    wrapper = """
import json, os, pathlib, runpy, sys
bundle, inp, output, root, parent, program = sys.argv[1:]
bundle, inp, output, root, parent = [pathlib.Path(x).resolve() for x in (bundle, inp, output, root, parent)]
denied = []
def audit(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = pathlib.Path(os.fsdecode(args[0])).resolve()
    forbidden = p.is_relative_to(root) or p.is_relative_to(parent)
    allowed = p.is_relative_to(bundle) or p == inp or p.is_relative_to(output) or p.is_relative_to(parent / '.venv')
    if forbidden and not allowed:
        denied.append(str(p))
        raise PermissionError('Source-repository access denied during portable replay')
sys.addaudithook(audit)
sys.path.insert(0, str(bundle))
sys.argv = [str(bundle / 'run.py'), '--bundle', str(bundle), '--input', str(inp), '--output', str(output), '--program', program, '--device', 'cpu' if program == 'token' else 'mps']
runpy.run_path(str(bundle / 'run.py'), run_name='__main__')
(output / 'source_access_guard.json').write_text(json.dumps({'blocked_source_access_attempts': len(denied), 'guard_active': True}))
"""
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1"
    )
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="grain-runtime-replay-") as cwd:
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                wrapper,
                str(bundle.resolve()),
                str((out / "private_inputs.jsonl").resolve()),
                str(dest.resolve()),
                str(ROOT),
                str(ROOT.parent / "project persona"),
                program,
            ],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    (out / f"replay_{program}.log").write_text(
        run.stdout + run.stderr, encoding="utf-8"
    )
    if run.returncode:
        raise RuntimeError(f"Portable replay failed; inspect replay_{program}.log")
    actual = pd.DataFrame(read_json(dest / "predictions.json"))
    expected = pd.DataFrame(read_json(out / "expected.json")).query(
        "program == @program"
    )
    merged = actual.merge(
        expected,
        on=["id", "program"],
        suffixes=("_actual", "_expected"),
        validate="one_to_one",
    )
    if len(merged) != m["replay_rows_per_program"]:
        raise ValueError("Portable replay membership changed")
    merged["abs_error"] = abs(merged.score_actual - merged.score_expected)
    merged.to_csv(dest / "comparison.csv", index=False)
    deps = read_json(dest / "dependencies.json")
    guard = read_json(dest / "source_access_guard.json")
    independence = guard["blocked_source_access_attempts"] == 0
    if program == "token":
        independence &= not deps["torch_imported"] and not deps["transformers_imported"]
    numerical = float(merged.abs_error.max()) <= m["absolute_score_tolerance"]
    result = {
        "status": "PASS" if numerical and independence else "FAIL",
        "program": program,
        "rows": len(merged),
        "max_abs_error": float(merged.abs_error.max()),
        "dependency_boundary_pass": bool(independence),
        "numerical_pass": bool(numerical),
        "full_LLM_runtime": False,
        "human_labels_accessed": False,
    }
    write_json(dest / "summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "replay"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument(
        "--program", choices=("token", "context4", "state1", "state2"), default="token"
    )
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.out)
    else:
        replay(a.out, a.program)
