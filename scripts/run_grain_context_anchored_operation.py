#!/usr/bin/env python3
"""Freeze HM77 source operations, acquire fresh joint states and account transport."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_context_anchored_operation as M
from grain_research.construction import math_source_ownership_operator as S
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_source_ownership_operator as B
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/context_anchored_operation_r1"
SPEC = ROOT / "configs/grain_context_anchored_operation_hm77.json"
SOURCE, COLD, OLD = B.SOURCE, B.COLD, B.OUT
read, write, export, digest, seal, files = (
    B.read,
    B.write,
    B.export,
    B.digest,
    B.seal,
    B.files,
)
CODE = tuple(
    dict.fromkeys(
        (
            "docs/GRAIN_CONTEXT_ANCHORED_OPERATION_PLAN_HM77.md",
            "configs/grain_context_anchored_operation_hm77.json",
            "grain_research/construction/math_context_anchored_operation.py",
            "scripts/run_grain_context_anchored_operation.py",
            "scripts/verify_grain_context_anchored_operation.py",
            "tests/test_grain_context_anchored_operation.py",
            *B.CODE,
        )
    )
)


def prepare():
    """Bind protocol/new texts before source construction or application reads."""
    from transformers import AutoTokenizer

    if OUT.exists():
        raise FileExistsError("HM77 root exists; preserve it, do not restart")
    spec = read(SPEC)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = read(ROOT / "configs/grain_math_passive_event_composition_v1.json")[
        "system"
    ]
    rows, lookup, prefixes = [], {}, {}
    for doc in S.fresh_documents(spec):
        for query in S.TARGETS:
            text, ids, part, _, _ = render(
                doc["body"], query, "A", 1, system, tokenizer
            )
            prefix = ids[: part["body"][-1] + 1]
            if doc["doc_id"] in prefixes and prefixes[doc["doc_id"]] != prefix:
                raise ValueError("Body prefix depends on future query")
            prefixes[doc["doc_id"]] = prefix
            row = doc | {
                "unit": len(rows),
                "query": query,
                "input_ids": ids,
                "partition": part,
                "rendered": text,
            }
            lookup[doc["pair_id"], doc["order"], doc["corner"], query] = row
            rows.append(row)
    proofs = []
    for row in rows:
        partner = lookup[row["pair_id"], row["order"], row["corner"] ^ 1, row["query"]]
        edits = [
            i
            for i, (a, b) in enumerate(
                zip(row["input_ids"], partner["input_ids"], strict=True)
            )
            if a != b
        ]
        if len(edits) != 2 or any(i not in row["partition"]["body"] for i in edits):
            raise ValueError("Ownership edit changed more than two body actors")
        twin = lookup[row["pair_id"], 1 - row["order"], row["corner"], row["query"]]
        if Counter(row["input_ids"]) != Counter(twin["input_ids"]):
            raise ValueError("Sentence order changed token bag")
        proofs.append(
            {
                "unit": row["unit"],
                "partner": partner["unit"],
                "reverse": twin["unit"],
                "actor_edits": edits,
            }
        )
    if len(rows) != spec["max_calls"]:
        raise ValueError("Fixed capture count mismatch")
    for folder in ("00/code", "01/models", "02/units", "03"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    write(OUT / "00/rows.json", rows)
    write(OUT / "00/proofs.json", proofs)
    for name in CODE:
        target = OUT / "00/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    inputs = [SPEC, MODEL / "config.json"]
    for root in (SOURCE, COLD, OLD):
        prefix = "02" if root == OLD else "01"
        inputs.extend(
            root / name
            for name in ("00/rows.json", f"{prefix}/raw_seal.json", "verification.json")
        )
    write(
        OUT / "00/manifest.json",
        {
            "packet": "HM77",
            "spec": spec,
            "code": {name: digest(ROOT / name) for name in CODE},
            "inputs": {str(p): digest(p) for p in inputs},
            "assets": seal(OUT, [OUT / "00/rows.json", OUT / "00/proofs.json"]),
            "application_fit": False,
            "source_answers_used": False,
            "existing_application_result_after": True,
            "fresh_bodies": 96,
            "model_weights_not_separately_hashed": True,
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
    print("HM77 prepared: one conditional mechanism, 576 fixed early cuts", flush=True)


def check():
    """Verify immutable code/material bytes, not merely output-file presence."""
    manifest = read(OUT / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(OUT / "00/code" / name) != sha:
            raise ValueError("Frozen code changed: " + name)
    for name, sha in manifest["inputs"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Input changed: " + name)
    for name, sha in manifest["assets"].items():
        if digest(OUT / name) != sha:
            raise ValueError("New material changed: " + name)
    return manifest


def load_joint(root, *, source_only=False):
    """Read body AND query from the SAME forward; partners only enter evaluation."""
    if source_only and root != SOURCE:
        raise ValueError("Source fit cannot load application root")
    prefix = "02" if root in (OUT, OLD) else "01"
    hashes = read(root / f"{prefix}/raw_seal.json")
    result, used = {}, {}
    for row in read(root / "00/rows.json"):
        if source_only and row["family"] != 0:
            continue
        if root == SOURCE:
            bank = "source" + str(row["family"])
        elif root == COLD:
            bank = "cold_forward" if row["form"] == "paraphrase" else "cold_reverse"
        else:
            bank = ("fresh" if root == OUT else "hm76") + (
                "_reverse" if row["order"] else "_forward"
            )
        path = root / f"{prefix}/units/unit_{row['unit']:04d}.json"
        for p in (path, path.with_suffix(".npz")):
            if digest(p) != hashes[str(p.relative_to(root))]:
                raise ValueError("Raw capture drift: " + str(p))
            used[str(p)] = digest(p)
        meta = read(path)
        for field in ("unit", "doc_id", "focal", "companion", "corner", "query"):
            if meta[field] != row[field]:
                raise ValueError("Wrong captured object: " + field)
        if meta["array_sha256"] != digest(path.with_suffix(".npz")):
            raise ValueError("Array/metadata binding mismatch")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            b, q = a["body_end"].astype(float), a["query_end"].astype(float)
        if any(h.shape != (1536,) or not np.isfinite(h).all() for h in (b, q)):
            raise ValueError("Hidden dimensions/nonfinite values changed")
        x, y, g = row["coordinates"]
        key = bank, row["query"], row["focal"], row["companion"], x, y
        if g in result.setdefault(key, {}):
            raise ValueError("Duplicate joint-state partner")
        result[key][g] = (b, q)
    if any(set(value) != {-1, 1} for value in result.values()):
        raise ValueError("Incomplete joint-state evaluation orbit")
    return result, used


def calibrate():
    """Construct all six fixed source maps before loading any application states."""
    check()
    if (OUT / "01/model_seal.json").exists():
        raise FileExistsError("Source models already sealed; no refit")
    groups, used = load_joint(SOURCE, source_only=True)
    diagnostics = []
    for query in S.TARGETS:
        items = [v for k, v in sorted(groups.items()) if k[1] == query]
        bp, qp = map(np.array, zip(*(v[1] for v in items), strict=True))
        bm, qm = map(np.array, zip(*(v[-1] for v in items), strict=True))
        if len(bp) != 144:
            raise ValueError("Incomplete source design")
        fitted, info = M.fit(bp, bm, qp, qm)
        for method, model in fitted.items():
            np.savez_compressed(OUT / f"01/models/{query}__{method}.npz", **model)
        diagnostics.append({"query": query, **info})
    write(OUT / "01/diagnostics.json", diagnostics)
    write(OUT / "01/source_inputs.json", used)
    write(
        OUT / "01/model_seal.json",
        seal(
            OUT,
            files(OUT / "01/models", "*.npz")
            + [OUT / "01/diagnostics.json", OUT / "01/source_inputs.json"],
        ),
    )
    write(
        OUT / "checkpoint.json",
        {"status": "SOURCE_FROZEN", "application_vectors_read": False},
    )
    print(
        "HM77 source maps frozen: 6 anchors, orthogonal/oblique both retained",
        flush=True,
    )


def models():
    """Load sealed source coefficients without adjusting application references."""
    check()
    for name, sha in read(OUT / "01/model_seal.json").items():
        if digest(OUT / name) != sha:
            raise ValueError("Source model drift: " + name)
    result = {}
    for path in files(OUT / "01/models", "*.npz"):
        query, method = path.stem.split("__")
        with np.load(path, allow_pickle=False) as a:
            result[query, method] = {k: a[k] for k in a.files}
    return result


def collect():
    """Capture only early joint states; retain failed/in-flight calls without retry."""
    manifest = check()
    models()
    status = read(OUT / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError("Uncertain/stopped call: review instead of retry")
    if len(files(OUT / "02/units", "unit_*.json")) != status["completed"]:
        raise ValueError("Actual capture count mismatch")
    start, previous = time.monotonic(), status["wall_seconds"]
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
                raise ValueError("Observer changed: " + key)
        write(OUT / "02/observer.json", actual)
        for row in read(OUT / "00/rows.json"):
            path = OUT / f"02/units/unit_{row['unit']:04d}.json"
            if path.exists():
                continue
            elapsed = previous + time.monotonic() - start
            if (
                status["attempts"] >= manifest["spec"]["max_calls"]
                or elapsed >= manifest["spec"]["max_seconds"]
            ):
                raise TimeoutError("Bounded HM77 acquisition stop")
            status.update(
                status="COLLECTING", in_flight=row["unit"], wall_seconds=elapsed
            )
            status["attempts"] += 1
            write(OUT / "run_status.json", status)
            arrays = B.capture(observer, row)
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
                wall_seconds=previous + time.monotonic() - start,
            )
            write(OUT / "run_status.json", status)
            if status["completed"] % 96 == 0:
                print(status, flush=True)
        write(
            OUT / "02/raw_seal.json",
            seal(OUT, files(OUT / "02/units") + [OUT / "02/observer.json"]),
        )
        status.update(
            status="COLLECTED", wall_seconds=previous + time.monotonic() - start
        )
        write(OUT / "run_status.json", status)
    except Exception as exc:
        status.update(
            status="STOPPED",
            error=f"{type(exc).__name__}: {exc}",
            wall_seconds=previous + time.monotonic() - start,
        )
        write(OUT / "run_status.json", status)
        raise


def all_groups():
    """Join immutable object identities, never silently overwrite material banks."""
    groups, inputs = {}, {}
    for root in (SOURCE, COLD, OLD, OUT):
        part, used = load_joint(root)
        if set(groups) & set(part):
            raise ValueError("Repeated application object")
        groups.update(part)
        inputs.update(used)
    return groups, inputs


def analyze():
    """Apply all fixed arms and retain the independent three-cause account."""
    fitted = models()
    if (OUT / "03/output_seal.json").exists():
        raise FileExistsError("Primary analysis sealed; do not replace it")
    if read(OUT / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Fresh capture has not completed")
    groups, inputs = all_groups()
    rows, summaries, max_direct, max_group, max_null = [], [], 0.0, 0.0, 0.0
    for bank in sorted({k[0] for k in groups}):
        for query in S.TARGETS:
            items = [
                (k, v) for k, v in sorted(groups.items()) if k[:2] == (bank, query)
            ]
            bp, qp = map(np.array, zip(*(v[1] for _, v in items), strict=True))
            bm, qm = map(np.array, zip(*(v[-1] for _, v in items), strict=True))
            for arm in M.METHODS:
                conditioned = arm.startswith("anchored")
                method = arm.split("_", 1)[1]
                model = fitted[query, method]
                accounts = M.accounts(model, bp, bm, qp, qm, conditioned=conditioned)
                direct = []
                for i in range(len(bp)):
                    b1, q1 = M.operate(model, bp[i], qp[i], conditioned=conditioned)
                    _b2, q2 = M.operate(model, bm[i], qm[i], conditioned=conditioned)
                    direct.append(
                        ((q1 - qm[i]) @ (q1 - qm[i]) + (q2 - qp[i]) @ (q2 - qp[i])) / 2
                    )
                    again_b, again_q = M.operate(model, b1, q1, conditioned=conditioned)
                    max_group = max(
                        max_group,
                        float(np.max(abs(again_b - bp[i]))),
                        float(np.max(abs(again_q - qp[i]))),
                    )
                    if arm == "anchored_oblique":
                        plain_q = M.operate(model, bp[i], qp[i], conditioned=False)[1]
                        max_null = max(max_null, float(np.max(abs(q1 - plain_q))))
                max_direct = max(
                    max_direct,
                    float(
                        np.max(abs(np.array(direct) - accounts["operator_error_sq"]))
                    ),
                )
                for i, (key, _) in enumerate(items):
                    rows.append(
                        {
                            "bank": bank,
                            "query": query,
                            "arm": arm,
                            "focal": key[2],
                            "companion": key[3],
                            "x": key[4],
                            "y": key[5],
                            **{k: float(v[i]) for k, v in accounts.items()},
                        }
                    )
                summaries.append(
                    {"bank": bank, "query": query, "arm": arm, **M.summarize(accounts)}
                )
    export(OUT / "03/pair_accounts.csv", rows)
    export(OUT / "03/summary.csv", summaries)
    write(OUT / "03/input_seal.json", inputs)
    write(
        OUT / "03/summary.json",
        {
            "packet": "HM77",
            "summaries": summaries,
            "direct_account_max_error": max_direct,
            "joint_involution_max_error": max_group,
            "oblique_anchor_no_effect_max_error": max_null,
            "fresh_actual_forwards": read(OUT / "run_status.json")["attempts"],
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
            "direct_account_max_error": max_direct,
            "joint_group_max_error": max_group,
            "oblique_anchor_no_effect": max_null,
        },
        flush=True,
    )


def main():
    """Keep source fitting, actual acquisition and application evaluation separate."""
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
