#!/usr/bin/env python3
"""Construct new-expression conditional fields, then inspect author consequences."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_action_expression as A
from grain_research.construction import math_expression_response as M
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_recursive_collector as C
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from scripts import run_grain_action_expression_hm115 as PREV
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_expression_response_hm119.json"
CODE = tuple(
    dict.fromkeys(
        (
            *PREV.CODE,
            "configs/grain_expression_response_hm119.json",
            "docs/GRAIN_EXPRESSION_RESPONSE_CONSTRUCTION_PLAN_HM119.md",
            "grain_research/construction/math_expression_response.py",
            "grain_research/construction/math_recursive_collector.py",
            "scripts/run_grain_expression_response_hm119.py",
            "scripts/score_grain_expression_response_cold_hm119.py",
            "tests/test_grain_math_expression_response.py",
        )
    )
)


def save(path, **arrays):
    """Create, never replace, a numeric scientific artifact."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def check(out):
    """Check frozen rules, source addresses and numeric input seals."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Freeze construction without opening old current states or application answers."""
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve old/failed outputs")
    spec = H.load(SPEC)
    previous = ROOT / spec["previous"]
    PREV.check(previous)
    if (
        H.load(previous / "verification.json")["status"]
        != "VERIFIED_FINITE_HM115_NOT_GLOBAL"
    ):
        raise ValueError("Verified frozen HM115 required")
    rows, reader = (
        H.load(previous / "00/rows.json"),
        H.load(previous / "00/bundle.json"),
    )
    if (
        len(rows) != spec["max_context_forwards"]
        or reader["targets"] != spec["targets"]
    ):
        raise ValueError("All fixed rows and targets required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    focus_ids = [
        tokenizer.encode(text, add_special_tokens=False) for text in spec["focus_texts"]
    ]
    current = rows[0]["current"]
    if (
        any(row["current"] != current for row in rows)
        or len(focus_ids[0]) != len(focus_ids[1])
        or len(focus_ids[0]) != len(current)
        or sum(a != b for a, b in zip(*focus_ids, strict=True)) != 1
        or any([row["input_ids"][i] for i in current] != focus_ids[0] for row in rows)
    ):
        raise ValueError("One-token focus change at identical addresses required")
    config = H.load(MODEL / "config.json")
    dim = config["hidden_size"] // config["num_attention_heads"]
    total = len(rows[0]["input_ids"])
    if any(len(row["input_ids"]) != total for row in rows):
        raise ValueError("Fixed history/current length required")
    with safe_open(
        str(MODEL / "model.safetensors"), framework="pt", device="cpu"
    ) as weights:
        embeddings = weights.get_tensor("model.embed_tokens.weight")
        seeds = embeddings[torch.tensor(focus_ids)].float().numpy()
    inv = 1 / (
        config["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    freq = torch.outer(torch.arange(total, dtype=torch.float32), inv)
    angles = torch.cat((freq, freq), dim=1)
    for name in ("00", "01/units", "02", "03"):
        (out / name).mkdir(parents=True)
    save(
        out / "00/coordinates.npz",
        seeds=seeds,
        cos=angles.cos().numpy(),
        sin=angles.sin().numpy(),
    )
    H.write(out / "00/rows.json", rows)
    H.write(out / "00/reader.json", reader)
    names = ("00/manifest.json", "00/rows.json", "00/bundle.json", "verification.json")
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {name: H.sha(ROOT / name) for name in CODE},
            "source": {
                str((previous / name).relative_to(ROOT)): H.sha(previous / name)
                for name in names
            },
            "inputs": H.seals(out, out / "00"),
            "config": config,
            "weights": {
                name: C.stream_sha(MODEL / name)
                for name in ("config.json", "tokenizer.json", "model.safetensors")
            },
            "focus_ids": focus_ids,
            "current": current,
            "history_length": min(current),
            "constructor_application_answers_or_current_input": False,
            "original_scope_complete": False,
        },
    )
    write_json(
        out / "01/status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
            "compressed_bytes": 0,
        },
    )
    print("HM119 PREPARED", len(rows), "history-only inputs", flush=True)


class _HistoryReady(Exception):
    """Stop at the last history-only block, before current or target computation."""


def capture_history(observer, row, coordinates, blocks=17):
    """Capture causal histories only; reject current, upper-block and logit calls."""
    import torch

    end = min(row["current"])
    ids = row["input_ids"][:end]
    pad = np.asarray(row["padding"], int)
    if np.any(pad >= end):
        raise ValueError("Padding outside history")
    allowed = np.tri(end, dtype=bool)
    allowed[:, pad] = False
    if not allowed.any(1).all():
        raise ValueError("Empty history attention support")
    saved, handles = {}, []
    trace = {
        "completed_blocks": [],
        "masked_blocks": [],
        "forbidden_calls": [],
        "history_tokens": end,
        "current_tokens_executed": 0,
    }
    layers = observer.model.model.layers
    dim = layers[0].self_attn.head_dim

    def projection(index, name):
        def remember(_module, _args, output):
            saved[index, name] = (
                output[0].detach().float().cpu().numpy().copy().reshape(end, -1, dim)
            )

        return remember

    def mask(index):
        def replace(_module, args, kwargs):
            actual = kwargs.get("attention_mask")
            if actual is None or actual.shape[-2:] != allowed.shape:
                raise ValueError("Expected explicit causal history mask")
            denied = torch.as_tensor(~allowed, device=actual.device)[None, None]
            altered = actual.clone().masked_fill(denied, torch.finfo(actual.dtype).min)
            if not torch.equal(altered[0, 0] == 0, ~denied[0, 0]):
                raise ValueError("History mask mismatch")
            if index == 0:
                for name, value in zip(
                    ("cos", "sin"), kwargs["position_embeddings"], strict=True
                ):
                    error = np.max(
                        abs(value[0].detach().cpu().numpy() - coordinates[name][:end])
                    )
                    if error > 1e-6:
                        raise ValueError("Frozen rotary coordinates incompatible")
            trace["masked_blocks"].append(index)
            return args, {**kwargs, "attention_mask": altered}

        return replace

    def complete(index):
        def stop(_module, _args, _output):
            trace["completed_blocks"].append(index)
            if index == blocks - 1:
                raise _HistoryReady

        return stop

    def forbidden(name):
        def reject(_module, _args):
            trace["forbidden_calls"].append(name)
            raise RuntimeError("Forbidden module: " + name)

        return reject

    try:
        for index in range(blocks):
            attention = layers[index].self_attn
            handles.append(
                attention.register_forward_pre_hook(mask(index), with_kwargs=True)
            )
            handles.append(layers[index].register_forward_hook(complete(index)))
            for name in ("k", "v"):
                handles.append(
                    getattr(attention, name + "_proj").register_forward_hook(
                        projection(index, name)
                    )
                )
        for name, module in (
            ("block17", layers[blocks]),
            ("final_norm", observer.model.model.norm),
            ("lm_head", observer.model.lm_head),
        ):
            handles.append(module.register_forward_pre_hook(forbidden(name)))
        tokens = torch.tensor([ids], device=next(observer.model.parameters()).device)
        with torch.no_grad():
            try:
                observer.model.model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    return_dict=True,
                )
            except _HistoryReady:
                pass
            else:
                raise RuntimeError("History capture did not stop at declared boundary")
    finally:
        for handle in handles:
            handle.remove()
    observer.calls += 1
    if (
        trace["completed_blocks"] != list(range(blocks))
        or trace["masked_blocks"] != list(range(blocks))
        or trace["forbidden_calls"]
    ):
        raise RuntimeError("Incomplete history trace")
    return {
        name: np.stack([saved[i, name] for i in range(blocks)]) for name in ("k", "v")
    }, trace


def collect(out):
    """Persist finite atomic captures, never automatically retry an uncertain call."""
    m = check(out)
    for name, digest in m["weights"].items():
        if C.stream_sha(MODEL / name) != digest:
            raise ValueError("Frozen model changed")
    path, rows = out / "01/status.json", H.load(out / "00/rows.json")
    status = H.load(path)
    if (
        status["state"] not in ("PREPARED", "COLLECTING")
        or status["pending"] is not None
    ):
        raise ValueError("Terminal/uncertain capture is not retried")
    with np.load(out / "00/coordinates.npz", allow_pickle=False) as z:
        coordinates = {name: z[name] for name in ("cos", "sin")}
    with collector_guard(out):
        observer = load_observer()
        for index in range(status["completed"], len(rows)):
            if (
                status["model_seconds"] >= m["spec"]["max_context_model_seconds"]
                or status["compressed_bytes"] >= m["spec"]["max_context_bytes"]
            ):
                status["state"] = "BUDGET_STOP"
                write_json(path, status)
                raise TimeoutError("Fixed finite capture budget; no renewal")
            status.update(
                state="COLLECTING", pending=index, attempts=status["attempts"] + 1
            )
            write_json(path, status)
            begun = time.monotonic()
            try:
                arrays, trace = capture_history(
                    observer, rows[index], coordinates, m["spec"]["blocks"]
                )
            except Exception:
                status.update(
                    state="FAILED",
                    model_seconds=status["model_seconds"] + time.monotonic() - begun,
                )
                write_json(path, status)
                raise
            seconds = time.monotonic() - begun
            raw = out / f"01/units/unit_{index:04d}.npz"
            save(raw, **arrays)
            H.write(
                raw.with_suffix(".json"),
                {"trace": trace, "sha256": H.sha(raw), "seconds": seconds},
            )
            status.update(
                completed=index + 1,
                pending=None,
                model_seconds=status["model_seconds"] + seconds,
                compressed_bytes=status["compressed_bytes"] + raw.stat().st_size,
            )
            write_json(path, status)
            if (index + 1) % 48 == 0:
                print("HM119 history", status, flush=True)
        status["state"] = (
            "COLLECTED"
            if status["model_seconds"] <= m["spec"]["max_context_model_seconds"]
            and status["compressed_bytes"] <= m["spec"]["max_context_bytes"]
            else "BUDGET_STOP"
        )
        write_json(path, status)
        H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
        if status["state"] != "COLLECTED":
            raise TimeoutError("Capture cap crossed at final atomic boundary")
    print("HM119 COLLECTED", status, flush=True)


def compile_all(out):
    """Export only numeric contexts and frozen reader operations to a cold child."""
    m = check(out)
    if H.load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Complete contexts required")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows = H.load(out / "00/rows.json")
    contexts = [out / f"01/units/unit_{i:04d}.npz" for i in range(len(rows))]
    valid = np.ones((len(rows), m["history_length"]), bool)
    for i, row in enumerate(rows):
        valid[i, row["padding"]] = False
    descriptor = {
        "contexts": [str(p.resolve()) for p in contexts],
        "valid": valid.tolist(),
        "coordinates": str((out / "00/coordinates.npz").resolve()),
        "weights": str(MODEL / "model.safetensors"),
        "config": m["config"],
        "reader": H.load(out / "00/reader.json"),
        "endpoint": max(m["current"]),
        **{name: m["spec"][name] for name in ("device", "blocks", "batch_size")},
        "max_seconds": m["spec"]["max_cold_seconds"],
        "hashes": {
            str(p.resolve()): C.stream_sha(p)
            for p in (
                *contexts,
                out / "00/coordinates.npz",
                MODEL / "model.safetensors",
                *(ROOT / name for name in CODE),
            )
        },
    }
    H.write(out / "02/input.json", descriptor)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_expression_response_cold_hm119.py"),
            "--input",
            str(out / "02/input.json"),
            "--output",
            str(out / "02/constructed.npz"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    H.write(
        out / "02/process.json",
        {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode:
        raise RuntimeError("Cold construction failed; preserve results, no rescue")
    H.write(out / "02/construction_seal.json", H.seals(out, out / "02"))
    print("HM119 CONSTRUCTED 576 current states; no answer/reference input", flush=True)


def load_constructed(out):
    """Recover addressed fields only after their construction has been sealed."""
    H.verify_hashes(out, H.load(out / "02/construction_seal.json"))
    rows = H.load(out / "00/rows.json")
    with np.load(out / "02/constructed.npz", allow_pickle=False) as z:
        raw = {name: z[name] for name in z.files}
    n = np.empty((3, 2, 6, 2, 2, 2, 2, 2, 6))
    seen = set()
    for row, value in zip(rows, raw["readings"], strict=True):
        index = tuple(row[name] for name in ("variant", "wording", "family")) + tuple(
            (row[name] + 1) // 2 for name in ("condition", "polarity", "owner")
        )
        if index in seen:
            raise ValueError("Duplicate constructed condition address")
        seen.add(index)
        n[index] = value
    if len(seen) != 288:
        raise ValueError("Incomplete scientific field")
    return rows, raw, n


def audit_exports(out, fields, targets):
    """Check all exported addresses and numbers, rejecting missing or duplicate rows."""
    error, counts = 0.0, {}
    for name in ("conditional_response_vectors", "qualified_author_pairing"):
        seen = set()
        with (out / f"03/{name}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                v, w = int(row["variant"]), int(row["wording"])
                family, q = targets.index(row["family"]), targets.index(row["query"])
                mode = ("uniform", "weighted").index(row["mode"])
                if name == "conditional_response_vectors":
                    c, o = (
                        (int(row["condition"]) + 1) // 2,
                        ("B", "A").index(row["actor"]),
                    )
                    address = (v, w, family, c, o, mode, q)
                    expected = {
                        "constructed_Gamma": fields["gamma"][address],
                        "source_Gamma": fields["gamma"][0, 0, family, c, o, mode, q],
                    }
                else:
                    address = (v, w, family, mode, q)
                    expected = {
                        "qualified_pairing_contrast": fields[
                            "qualified_family_pairing"
                        ][address],
                        "source_pairing_contrast": fields["qualified_family_pairing"][
                            0, 0, family, mode, q
                        ],
                        "other_actor_diagnostic_NOT_self": fields[
                            "other_actor_pairing_diagnostic"
                        ][address],
                    }
                if address in seen:
                    raise ValueError("Duplicate exported response address")
                seen.add(address)
                for field, value in expected.items():
                    error = max(error, abs(float(row[field]) - value))
        counts[name] = len(seen)
    if counts != {
        "conditional_response_vectors": 1728,
        "qualified_author_pairing": 432,
    }:
        raise ValueError("Incomplete conditional response/author export")
    return error, counts


def analyze(out):
    """Inspect full conditional response, not tune the constructor to references."""
    m = check(out)
    rows, raw, n = load_constructed(out)
    previous = ROOT / m["spec"]["previous"]
    PREV.check(previous)
    H.verify_hashes(previous, H.load(previous / "01/raw_seal.json"))
    errors = {"k": 0.0, "v": 0.0, "readings": 0.0}
    bundle = H.load(out / "00/reader.json")
    for i, row in enumerate(rows):
        old = PREV.raw_unit(previous, row["id"])
        for name in ("k", "v"):
            errors[name] = max(
                errors[name], float(np.max(abs(raw[name][i, 0] - old[name])))
            )
        readings = R.read_current(old["k"], old["v"], max(row["current"]), bundle)
        errors["readings"] = max(
            errors["readings"], float(np.max(abs(readings - raw["readings"][i, 0])))
        )
    compatible = (
        max(errors["k"], errors["v"]) <= m["spec"]["compatibility_kv_atol"]
        and errors["readings"] <= m["spec"]["compatibility_reading_atol"]
    )
    H.write(
        out / "03/compatibility.json",
        {
            "errors": errors,
            "passed": compatible,
            "reference_opened_only_after_construction_seal": True,
            "no_tuning_or_retry": True,
        },
    )
    fields = M.response_fields(n)
    identities = M.verify_response_identities(n, fields)
    save(out / "03/fields.npz", readings=n, **fields)
    comparisons, details = [], {}
    for sector, mode, (variant, wording) in itertools.product(
        (
            "actor_even_focus_even",
            "actor_even_focus_odd",
            "actor_odd_focus_even",
            "actor_odd_focus_odd",
        ),
        range(2),
        ((0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ):
        source, application = (
            fields[sector][0, 0, ..., mode, :],
            fields[sector][variant, wording, ..., mode, :],
        )
        result = A.direction_match(source, application)
        key = f"{sector}:m{mode}:v{variant}:w{wording}"
        details[key] = result
        comparisons.append(
            {
                "sector": sector,
                "mode": ("uniform", "weighted")[mode],
                "variant": variant,
                "wording": wording,
                "source_RMS": float(np.sqrt(np.mean(source**2))),
                "application_RMS": float(np.sqrt(np.mean(application**2))),
                **{
                    name: result.get(name)
                    for name in (
                        "status",
                        "matched_mean_cosine",
                        "matching_rank_with_ties",
                        "permutation_mean",
                        "matched_minus_permutation_mean",
                    )
                },
            }
        )
    response_rows, author_rows = [], []
    for variant, wording, family, c, o, mode, q in np.ndindex(3, 2, 6, 2, 2, 2, 6):
        gamma = fields["gamma"][variant, wording, family, c, o, mode, q]
        response_rows.append(
            {
                "variant": variant,
                "wording": wording,
                "family": m["spec"]["targets"][family],
                "condition": 2 * c - 1,
                "actor": ("B", "A")[o],
                "mode": ("uniform", "weighted")[mode],
                "query": m["spec"]["targets"][q],
                "constructed_Gamma": float(gamma),
                "source_Gamma": float(fields["gamma"][0, 0, family, c, o, mode, q]),
            }
        )
    for variant, wording, family, mode, q in np.ndindex(3, 2, 6, 2, 6):
        value = fields["qualified_family_pairing"][variant, wording, family, mode, q]
        author_rows.append(
            {
                "variant": variant,
                "wording": wording,
                "family": m["spec"]["targets"][family],
                "mode": ("uniform", "weighted")[mode],
                "query": m["spec"]["targets"][q],
                "qualified_pairing_contrast": float(value),
                "source_pairing_contrast": float(
                    fields["qualified_family_pairing"][0, 0, family, mode, q]
                ),
                "other_actor_diagnostic_NOT_self": float(
                    fields["other_actor_pairing_diagnostic"][
                        variant, wording, family, mode, q
                    ]
                ),
            }
        )
    H.table(out / "03/correspondence.csv", comparisons)
    H.write(out / "03/correspondence_details.json", details)
    H.table(out / "03/conditional_response_vectors.csv", response_rows)
    H.table(out / "03/qualified_author_pairing.csv", author_rows)
    H.write(
        out / "03/summary.json",
        {
            "status": "ANALYZED_PENDING_VERIFICATION"
            if compatible
            else "COMPATIBILITY_FAILED_NOT_RESCUED",
            "compatibility": errors,
            "identities": identities,
            "comparisons": comparisons,
            "context_history_forwards": 288,
            "current_only_numeric_executions": 576,
            "native_answer_calls": 0,
            "fits": 0,
            "conditional_response_rows": len(response_rows),
            "qualified_author_rows": len(author_rows),
            "original_scope_complete": False,
        },
    )
    H.write(out / "03/output_seal.json", H.seals(out, out / "03"))
    print("HM119 ANALYZED", errors, identities, flush=True)
    if not compatible:
        raise ValueError("Arithmetic compatibility failed; no precision rescue")


def verify(out):
    """Replay every primitive reading and account with independent explicit loops."""
    m = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    H.verify_hashes(out, H.load(out / "03/output_seal.json"))
    rows, raw, n = load_constructed(out)
    source = H.load(out / "00/reader.json")
    errors = {"readings": 0.0, "fields": 0.0, "csv": 0.0}
    from grain_research.construction.math_cold_evidence import rotate

    for i, row in enumerate(rows):
        receipt = H.load(out / f"01/units/unit_{i:04d}.json")
        if (
            receipt["sha256"] != H.sha(out / f"01/units/unit_{i:04d}.npz")
            or receipt["trace"]["completed_blocks"] != list(range(17))
            or receipt["trace"]["current_tokens_executed"] != 0
            or receipt["trace"]["forbidden_calls"]
        ):
            raise ValueError("History-only execution trace invalid")
        for focus, q in np.ndindex(2, 6):
            k, v = raw["k"][i, focus].astype(float), raw["v"][i, focus].astype(float)
            query = rotate(
                np.asarray(source["raw_queries"])[q],
                max(row["current"]) + source["gaps"][q],
                source["frequencies"],
                source["scaling"],
            )
            uniform = weighted = 0.0
            for head, direction in enumerate(np.asarray(source["directions"])[q]):
                logits = k[:, head // (len(query) // k.shape[1])] @ query[head]
                probability = np.exp(logits - logits.max())
                probability /= probability.sum()
                content = v[:, head // (len(query) // v.shape[1])] @ direction
                uniform += content.mean()
                weighted += sum(
                    float(a * b) for a, b in zip(probability, content, strict=True)
                )
            errors["readings"] = max(
                errors["readings"],
                abs(uniform - raw["readings"][i, focus, 0, q]),
                abs(weighted - raw["readings"][i, focus, 1, q]),
            )
    fields = M.response_fields(n)
    with np.load(out / "03/fields.npz", allow_pickle=False) as z:
        for name, value in fields.items():
            errors["fields"] = max(
                errors["fields"], float(np.max(abs(value - z[name])))
            )
    identities = M.verify_response_identities(n, fields)
    errors["csv"], csv_counts = audit_exports(out, fields, m["spec"]["targets"])
    if (
        max((*errors.values(), *identities.values())) > 1e-10
        or not H.load(out / "03/compatibility.json")["passed"]
    ):
        raise ValueError("Finite numeric verification failed")
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM119_NOT_GLOBAL",
            "errors": errors,
            "identities": identities,
            "csv_rows": csv_counts,
            "context_forwards": len(rows),
            "numeric_current_executions": len(rows) * 2,
            "native_answers": 0,
            "fits": 0,
            "same_agent_alternate_formula_not_third_party": True,
            "original_scope_complete": False,
            "permissions": m["spec"]["permissions"],
        },
    )
    print("HM119 VERIFIED", errors, identities, flush=True)


def main():
    """Expose finite ordered steps; no autonomous repeat or global completion claim."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "collect", "compile", "analyze", "verify")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {
        "prepare": prepare,
        "collect": collect,
        "compile": compile_all,
        "analyze": analyze,
        "verify": verify,
    }[args.action](args.output)


if __name__ == "__main__":
    main()
