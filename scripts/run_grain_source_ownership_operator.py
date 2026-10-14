#!/usr/bin/env python3
"""HM76: freeze source group operations, capture fresh vectors, test transport."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_source_ownership_operator as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.vector_mechanisms.navigation import MODEL, digest
from scripts.run_grain_math_span_binding import hidden_tensor, load_observer
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/source_ownership_operator_r3"
SPEC = ROOT / "configs/grain_source_ownership_operator_hm76.json"
SOURCE = ROOT / "results/grain_math_response_v1/ownership_composition_r1"
COLD = ROOT / "results/grain_math_response_v1/cold_relational_measurement_r1"
STAGES = ("body", *["query_" + q for q in M.TARGETS])
CODE = (
    "docs/GRAIN_SOURCE_OWNERSHIP_OPERATOR_PLAN_HM76.md",
    "configs/grain_source_ownership_operator_hm76.json",
    "grain_research/construction/math_source_ownership_operator.py",
    "scripts/run_grain_source_ownership_operator.py",
    "scripts/verify_grain_source_ownership_operator.py",
    "tests/test_grain_source_ownership_operator.py",
    "grain_research/construction/math_answer_recoding_mechanism.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
    "grain_research/vector_mechanisms/trace.py",
)


def read(path):
    """Read a named JSON asset without interpreting its status as evidence."""
    return json.loads(path.read_text())


def write(path, value):
    """Atomically persist actual outputs/checkpoints without overwriting old roots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def export(path, rows):
    """Write complete fixed-schema numeric accounts for every evaluated pair."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal(directory, names):
    """Bind the bytes actually consumed/produced, not merely their existence."""
    return {str(p.relative_to(directory)): digest(p) for p in names}


def files(directory, pattern="*"):
    """Enumerate real assets, excluding ExFAT/macOS AppleDouble sidecars."""
    return sorted(
        p for p in directory.glob(pattern) if p.is_file() and not p.name.startswith("._")
    )


def prepare():
    """Seal fresh materials and protocol before fitting or application reads."""
    from transformers import AutoTokenizer

    if OUT.exists():
        raise FileExistsError("HM76 root exists; do not overwrite or restart")
    spec = read(SPEC)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = read(ROOT / "configs/grain_math_passive_event_composition_v1.json")[
        "system"
    ]
    rows, prefixes, order_lookup, proofs = [], {}, {}, []
    for doc in M.fresh_documents(spec):
        for query in M.TARGETS:
            text, ids, part, _, _ = render(
                doc["body"], query, "A", 1, system, tokenizer
            )
            prefix = ids[: part["body"][-1] + 1]
            if doc["doc_id"] in prefixes and prefixes[doc["doc_id"]] != prefix:
                raise ValueError("Fresh body depends on future query")
            prefixes[doc["doc_id"]] = prefix
            row = doc | {
                "unit": len(rows),
                "query": query,
                "input_ids": ids,
                "partition": part,
                "rendered": text,
            }
            order_lookup[doc["pair_id"], doc["order"], doc["corner"], query] = row
            rows.append(row)
    for row in rows:
        if row["order"]:
            continue
        twin = order_lookup[row["pair_id"], 1, row["corner"], row["query"]]
        if Counter(row["input_ids"]) != Counter(twin["input_ids"]):
            raise ValueError("Fresh sentence reversal changed token bag")
        if row["corner"] % 2 == 0:
            partner = order_lookup[row["pair_id"], 0, row["corner"] + 1, row["query"]]
            edits = [
                i
                for i, (a, b) in enumerate(
                    zip(row["input_ids"], partner["input_ids"], strict=True)
                )
                if a != b
            ]
            if len(edits) != 2 or any(i not in row["partition"]["body"] for i in edits):
                raise ValueError("Fresh g partners changed more than two body actors")
        proofs.append(
            {"forward": row["unit"], "reverse": twin["unit"], "same_token_bag": True}
        )
    if len(rows) != spec["max_calls"]:
        raise ValueError("Frozen call count mismatch")
    for folder in ("00/code", "01/models", "02/units", "03"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    write(OUT / "00/rows.json", rows)
    write(OUT / "00/proofs.json", proofs)
    for name in CODE:
        target = OUT / "00/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    inputs = [SPEC, MODEL / "config.json"]
    for old in (SOURCE, COLD):
        inputs.extend(
            old / name
            for name in ("00/rows.json", "01/raw_seal.json", "verification.json")
        )
    write(
        OUT / "00/manifest.json",
        {
            "packet": "HM76",
            "spec": spec,
            "code": {name: digest(ROOT / name) for name in CODE},
            "inputs": {str(p): digest(p) for p in inputs},
            "assets": seal(OUT, [OUT / "00/rows.json", OUT / "00/proofs.json"]),
            "model_weights_not_separately_hashed": True,
            "fresh_bodies": 96,
            "fresh_captures": 576,
            "source_answers_used": False,
            "application_fit": False,
            "existing_application_already_seen_by_other_protocols": True,
            "original_scope_complete": False,
        },
    )
    write(
        OUT / "run_status.json",
        {
            "status": "PREPARED",
            "attempts": 0,
            "completed": 0,
            "in_flight": None,
            "wall_seconds": 0,
        },
    )
    print(
        "HM76 prepared:96 fresh bodies/576 early cuts; no upper reader or answers",
        flush=True,
    )


def check():
    """Reject changed protocol, source identities, or frozen fresh materials."""
    manifest = read(OUT / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(OUT / "00/code" / name) != sha:
            raise ValueError("Code drift: " + name)
    for name, sha in manifest["inputs"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Input identity drift: " + name)
    for name, sha in manifest["assets"].items():
        if digest(OUT / name) != sha:
            raise ValueError("Fresh material drift: " + name)
    return manifest


def load_groups(root, selection):
    """Load paired raw endpoints for fit/evaluation; never consume native answers."""
    groups, used = {}, {}
    fresh = root == OUT
    prefix = "02" if fresh else "01"
    raw_seal = (
        read(root / f"{prefix}/raw_seal.json")
        if fresh
        else read(root / "01/raw_seal.json")
    )
    for row in read(root / "00/rows.json"):
        if not selection(row):
            continue
        bank = (
            ("fresh_reverse" if row["order"] else "fresh_forward")
            if fresh
            else (
                "source" + str(row["family"])
                if "family" in row
                else "cold_forward"
                if row["form"] == "paraphrase"
                else "cold_reverse"
            )
        )
        path = root / f"{prefix}/units/unit_{row['unit']:04d}.json"
        for p in (path, path.with_suffix(".npz")):
            if digest(p) != raw_seal[str(p.relative_to(root))]:
                raise ValueError("Capture seal drift: " + str(p))
            used[str(p)] = digest(p)
        meta = read(path)
        for k in ("unit", "doc_id", "focal", "companion", "corner", "query"):
            if meta[k] != row[k]:
                raise ValueError("Capture object binding mismatch: " + k)
        if digest(path.with_suffix(".npz")) != meta["array_sha256"]:
            raise ValueError("Capture array/metadata mismatch")
        x, y, g = row["coordinates"]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            vectors = {"query_" + row["query"]: a["query_end"].astype(float)}
            if row["query"] == M.TARGETS[0]:
                vectors["body"] = a["body_end"].astype(float)
        for stage, h in vectors.items():
            if h.shape != (1536,) or not np.isfinite(h).all():
                raise ValueError("Fixed hidden coordinate changed")
            key = (bank, stage, row["focal"], row["companion"], x, y)
            if g in groups.setdefault(key, {}):
                raise ValueError("Duplicate ownership partner")
            groups[key][g] = h
    if any(set(value) != {-1, 1} for value in groups.values()):
        raise ValueError("Incomplete evaluator orbit")
    return groups, used


def calibrate():
    """Fit and seal source operations before reading held-out vector assets."""
    check()
    if (OUT / "01/model_seal.json").exists():
        raise FileExistsError("Source models already frozen; no refit")
    groups, used = load_groups(SOURCE, lambda r: r["family"] == 0)
    diagnostics = []
    for stage in STAGES:
        pairs = [value for key, value in sorted(groups.items()) if key[1] == stage]
        plus, minus = np.array([p[1] for p in pairs]), np.array([p[-1] for p in pairs])
        if len(plus) != 144:
            raise ValueError("Source coverage differs from the full fixed design")
        models, info = M.fit(plus, minus)
        for method, model in models.items():
            np.savez_compressed(OUT / f"01/models/{stage}__{method}.npz", **model)
            source_accounts = M.summarize(M.pair_accounts(model, plus, minus))
            diagnostics.append(
                {
                    "stage": stage,
                    "method": method,
                    **info,
                    **source_accounts,
                    "projector_idempotence_factor_error": float(
                        np.max(
                            abs(
                                model["right"] @ model["left"]
                                - np.eye(model["left"].shape[1])
                            )
                        )
                    ),
                    "projector_operator_norm": float(
                        np.linalg.svd(model["right"], compute_uv=False)[0]
                    ),
                }
            )
    write(OUT / "01/diagnostics.json", diagnostics)
    write(OUT / "01/source_inputs.json", used)
    assets = files(OUT / "01/models", "*.npz") + [
        OUT / "01/diagnostics.json",
        OUT / "01/source_inputs.json",
    ]
    write(OUT / "01/model_seal.json", seal(OUT, assets))
    write(
        OUT / "checkpoint.json",
        {"status": "SOURCE_FROZEN", "application_vectors_read": False},
    )
    print("HM76 source-frozen operations:", len(diagnostics), flush=True)


def models():
    """Load only sealed source coefficients, never reestimate an application origin."""
    check()
    for name, sha in read(OUT / "01/model_seal.json").items():
        if digest(OUT / name) != sha:
            raise ValueError("Source operator drift: " + name)
    result = {}
    for path in files(OUT / "01/models", "*.npz"):
        stage, method = path.stem.split("__")
        with np.load(path, allow_pickle=False) as a:
            result[stage, method] = {k: a[k] for k in a.files}
    return result


class CutReached(Exception):
    """Mark a successful cut capture rather than an uncertain forward error."""


def capture(observer, row):
    """Capture one new input; no layer17+ or native answer is allowed."""
    import torch

    state, upper = {}, []

    def stop(_module, _args, output):
        h = hidden_tensor(output)[0].detach().cpu().numpy()
        state["body_end"] = h[row["partition"]["body"][-1]].copy()
        state["query_end"] = h[-1].copy()
        raise CutReached

    def guard(_module, _args):
        upper.append(True)
        raise RuntimeError("Upper reader forbidden")

    handles = [
        observer.model.model.layers[16].register_forward_hook(stop),
        observer.model.model.layers[17].register_forward_pre_hook(guard),
    ]
    tokens = torch.tensor(
        [row["input_ids"]], device=next(observer.model.parameters()).device
    )
    try:
        try:
            with torch.inference_mode():
                observer.model.model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    return_dict=True,
                )
        except CutReached:
            pass
        if upper or set(state) != {"body_end", "query_end"}:
            raise ValueError("Incomplete early cut or forbidden upper execution")
        return state
    finally:
        for handle in handles:
            handle.remove()


def collect():
    """Save one actual early cut at a time; do not retry uncertain calls."""
    manifest = check()
    models()
    status = read(OUT / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError("Stopped/in-flight call requires review; no retry")
    if len(files(OUT / "02/units", "unit_*.json")) != status["completed"]:
        raise ValueError("Capture/status count mismatch")
    started, previous = time.monotonic(), status["wall_seconds"]
    try:
        observer = load_observer()
        actual, old = observer.manifest(), read(SOURCE / "01/observer.json")
        for key in (
            "model",
            "torch",
            "transformers",
            "dtype",
            "device",
            "hidden_size",
            "layers",
            "attention",
            "training",
            "template_sha256",
        ):
            if actual[key] != old[key]:
                raise ValueError("Observer mismatch: " + key)
        write(OUT / "02/observer.json", actual)
        for row in read(OUT / "00/rows.json"):
            path = OUT / f"02/units/unit_{row['unit']:04d}.json"
            if path.exists():
                continue
            elapsed = previous + time.monotonic() - started
            if (
                status["attempts"] >= manifest["spec"]["max_calls"]
                or elapsed >= manifest["spec"]["max_seconds"]
            ):
                raise TimeoutError("Fixed HM76 attempt/wall bound")
            status.update(
                status="COLLECTING", in_flight=row["unit"], wall_seconds=elapsed
            )
            status["attempts"] += 1
            write(OUT / "run_status.json", status)
            arrays = capture(observer, row)
            np.savez_compressed(path.with_suffix(".npz"), **arrays)
            write(
                path,
                {
                    k: row[k]
                    for k in ("unit", "doc_id", "focal", "companion", "corner", "query")
                }
                | {"array_sha256": digest(path.with_suffix(".npz")), "upper_blocks": 0},
            )
            status.update(
                completed=status["completed"] + 1,
                in_flight=None,
                wall_seconds=previous + time.monotonic() - started,
            )
            write(OUT / "run_status.json", status)
            if status["completed"] % 96 == 0:
                print(status, flush=True)
        assets = files(OUT / "02/units") + [OUT / "02/observer.json"]
        write(OUT / "02/raw_seal.json", seal(OUT, assets))
        status.update(
            status="COLLECTED", wall_seconds=previous + time.monotonic() - started
        )
        write(OUT / "run_status.json", status)
    except Exception as exc:
        status.update(
            status="STOPPED",
            error=f"{type(exc).__name__}: {exc}",
            wall_seconds=previous + time.monotonic() - started,
        )
        write(OUT / "run_status.json", status)
        raise


def analyze():
    """Apply one-vector operators and only then compare independent orbit partners."""
    source_models = models()
    if (OUT / "03/output_seal.json").exists():
        raise FileExistsError("Primary HM76 analysis already sealed")
    if read(OUT / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Actual fresh acquisition not terminal")
    groups, inputs = load_groups(SOURCE, lambda _r: True)
    for root in (COLD, OUT):
        part, used = load_groups(root, lambda _r: True)
        if set(groups) & set(part):
            raise ValueError("Duplicate source/application identity")
        groups.update(part)
        inputs.update(used)
    rows, summaries, max_identity = [], [], 0.0
    for bank in (
        "source0",
        "source1",
        "cold_forward",
        "cold_reverse",
        "fresh_forward",
        "fresh_reverse",
    ):
        for (stage, method), model in source_models.items():
            items = [
                (key, value)
                for key, value in sorted(groups.items())
                if key[:2] == (bank, stage)
            ]
            plus, minus = (
                np.array([v[1] for _, v in items]),
                np.array([v[-1] for _, v in items]),
            )
            accounts = M.pair_accounts(model, plus, minus)
            direct = (
                np.sum(
                    (np.array([M.operate(model, h) for h in plus]) - minus) ** 2, axis=1
                )
                + np.sum(
                    (np.array([M.operate(model, h) for h in minus]) - plus) ** 2, axis=1
                )
            ) / 2
            max_identity = max(
                max_identity, float(np.max(abs(direct - accounts["operator_error_sq"])))
            )
            for index, (key, _value) in enumerate(items):
                rows.append(
                    {
                        "bank": bank,
                        "stage": stage,
                        "method": method,
                        "focal": key[2],
                        "companion": key[3],
                        "x": key[4],
                        "y": key[5],
                        **{
                            k: float(v[index])
                            for k, v in accounts.items()
                            if v.ndim == 1
                        },
                    }
                )
            summaries.append(
                {
                    "bank": bank,
                    "stage": stage,
                    "method": method,
                    **M.summarize(accounts),
                }
            )
    export(OUT / "03/pair_accounts.csv", rows)
    export(OUT / "03/summary.csv", summaries)
    write(OUT / "03/input_seal.json", inputs)
    write(
        OUT / "03/summary.json",
        {
            "packet": "HM76",
            "summaries": summaries,
            "single_vector_residual_identity_max_error": max_identity,
            "fresh_actual_calls": read(OUT / "run_status.json")["attempts"],
            "source_answers_used": False,
            "application_fits": 0,
            "upper_reader_calls": 0,
            "original_scope_complete": False,
        },
    )
    write(OUT / "03/output_seal.json", seal(OUT, files(OUT / "03")))
    write(
        OUT / "checkpoint.json",
        {
            "status": "ANALYZED",
            "active_processes": [],
            "original_scope_complete": False,
        },
    )
    print(
        {
            "status": "ANALYZED",
            "pairs": len(rows),
            "summaries": len(summaries),
            "residual_identity_max_error": max_identity,
        },
        flush=True,
    )


def main():
    """Separate preparation, calibration, acquisition and application evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "calibrate", "collect", "analyze"))
    phase = parser.parse_args().phase
    if phase == "collect":
        with collector_guard(OUT):
            collect()
    else:
        {"prepare": prepare, "calibrate": calibrate, "analyze": analyze}[phase]()


if __name__ == "__main__":
    main()
