#!/usr/bin/env python3
"""Twelve-author frozen P/Q readouts against complete values, edits and anchors."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context_carrier import (
    query_region_mask,
    split_context_state,
)
from grain_research.vector_mechanisms.full_state_edit import (
    align_excerpts,
    align_states,
    capture_boundary,
    replay_suffix,
)
from grain_research.vector_mechanisms.level_response import (
    affine_parts,
    atlas_readout,
    paired_quality,
    response_ledger,
)
from grain_research.vector_mechanisms.query_functional import (
    matrix_response,
    matrix_variants,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_multiquery_sensitivity import load_panel
from scripts.run_grain_full_state_edit_connection import BRIDGE, SOURCE, token_view
from scripts.run_grain_multiquery_sensitivity import (
    QUERIES,
    native_alphabet,
    query_prompt,
)
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

FULL = BASE / "full_state_edit_connection_v1"
FUNCTIONAL = BASE / "query_region_functional_v1"
PLAN = ROOT / "docs/GRAIN_FRAMEWORK_CLOSURE_PLAN_V1.md"


def question_mask(tokenizer, bodies, query):
    """Return the actual fixed question region under the frozen chat template."""
    text, spans, encoded = token_view(tokenizer, bodies, query)
    user, _ = query_prompt(bodies, query)
    start = text.index(user) + user.rindex("\nQuestion: ") + 1
    return text, spans, encoded, query_region_mask(encoded["offset_mapping"], start)


def freeze_reference(out, records):
    """Freeze two-reference means plus all four old local charts, excluding new authors."""
    lookup = {r[0]["author_id"]: r for r in records if r[0]["view"] == 0}
    original = {a["index"]: a for a in read(FULL / "private_inputs.json")}
    design = read(FUNCTIONAL / "private_design.json")
    result, lineage, fixed_tokens = {}, {}, {}
    with np.load(FUNCTIONAL / "reference_matrices.npz", allow_pickle=False) as matrices:
        for q in QUERIES:
            zs, hs, gradients, gs, scores, refs, token_ids = [], [], [], [], [], [], []
            for d in sorted(
                (d for d in design if d["query"] == q), key=lambda d: d["index"]
            ):
                parent = ROOT / d["parent_packet"]
                node = (
                    FUNCTIONAL / "private_packets" / f"unit_{d['index']:02d}_{q}.json"
                )
                for p, digest in (
                    (parent, d["packet_sha256"]),
                    (parent.with_suffix(".npz"), d["arrays_sha256"]),
                ):
                    if sha(p) != digest:
                        raise ValueError("Old reference input changed")
                    lineage[str(p.relative_to(ROOT))] = digest
                node_meta = read(node)
                if sha(node.with_suffix(".npz")) != node_meta["arrays_sha256"]:
                    raise ValueError("Old integrated matrix changed")
                lineage[str(node.relative_to(ROOT))] = sha(node)
                lineage[str(node.with_suffix(".npz").relative_to(ROOT))] = sha(
                    node.with_suffix(".npz")
                )
                mask = np.asarray(d["question_mask"], bool)
                with np.load(parent.with_suffix(".npz"), allow_pickle=False) as a:
                    mapping = a["mapping"][mask]
                    if np.any(mapping < 0):
                        raise ValueError("Question token cannot be a fresh boundary")
                    zs.append(a["full_state"][mapping].astype(float))
                    hs.append(a["full_state"][-1].astype(float))
                with np.load(node.with_suffix(".npz"), allow_pickle=False) as a:
                    gs.append(a["integrated"])
                _, packet, vectors = lookup[d["author_id"]]
                gradients.append(vectors[f"{q}__gradient"].astype(float))
                scores.append(
                    packet["queries"][q]["readouts"]["full"]["conditional_score"]
                )
                refs.append(d["role"] == "reference")
                token_ids.append(
                    np.asarray(original[d["index"]]["queries"][q]["full_ids"])[mapping]
                )
            if (
                len(zs) != 4
                or sum(refs) != 2
                or any(not np.array_equal(token_ids[0], t) for t in token_ids)
            ):
                raise ValueError("Fixed four-author query references required")
            zs, hs, gradients, gs, scores, refs = map(
                np.asarray, (zs, hs, gradients, gs, scores, refs)
            )
            if not np.allclose(gs[refs].mean(0), matrices[q], rtol=0, atol=1e-14):
                raise ValueError("M7 reference matrix mismatch")
            arrays = {
                "Z": zs[refs].mean(0),
                "h": hs[refs].mean(0),
                "G": matrices[q],
                "g": gradients[refs].mean(0),
                "b": scores[refs].mean(),
                "chart_Z": zs,
                "chart_h": hs,
                "chart_G": gs,
                "chart_g": gradients,
                "chart_b": scores,
            }
            result.update({f"{q}__{k}": v for k, v in arrays.items()})
            fixed_tokens[q] = token_ids[0].tolist()
    np.savez_compressed(out / "reference.npz", **result)
    return lineage, fixed_tokens


def prepare(out):
    """Freeze membership, formula objects and identities before any new inference."""
    source, records = load_panel(SOURCE)
    split = read(BRIDGE / "split_manifest.json")
    if (
        sha(BRIDGE / "split_manifest.json")
        != read(BRIDGE / "manifest.json")["split_sha256"]
    ):
        raise ValueError("Old split changed")
    old = {r["author_id"] for r in read(FULL / "private_inputs.json")}
    groups = [[], []]
    for uid, ref in zip(split["authors"], split["reference_mask"], strict=True):
        if uid not in old:
            groups[0 if ref else 1].append(uid)
    key = lambda u: hashlib.sha256(f"{split['salt']}|{u}".encode()).hexdigest()
    selected = [
        u
        for pair in zip(*(sorted(g, key=key)[:6] for g in groups), strict=True)
        for u in pair
    ]
    if len(selected) != 12 or set(selected) & old:
        raise ValueError("Require twelve disjoint transport authors")
    lookup = {r[0]["author_id"]: r for r in records if r[0]["view"] == 0}
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    lineage, fixed_tokens = freeze_reference(out, records)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design = []
    for i, uid in enumerate(selected):
        row, packet, _ = lookup[uid]
        kept = [j for j in range(len(row["texts"])) if j not in row["e1"]]
        for q in QUERIES:
            full, fs, fe, fm = question_mask(tokenizer, row["texts"], q)
            edited, es, ee, em = question_mask(
                tokenizer, [row["texts"][j] for j in kept], q
            )
            mapping, audit = align_excerpts(full, edited, fs, es, kept, fe, ee)
            if not np.array_equal(np.flatnonzero(fm), mapping[em]):
                raise ValueError("Question region alignment differs")
            if not np.array_equal(
                np.asarray(fe["input_ids"])[fm], fixed_tokens[q]
            ) or not np.array_equal(np.asarray(ee["input_ids"])[em], fixed_tokens[q]):
                raise ValueError("Changed question tokens")
            alphabet = native_alphabet(q, tokenizer)
            if alphabet != packet["queries"][q]["alphabet"]:
                raise ValueError("Changed scoring alphabet")
            design.append(
                {
                    "index": i,
                    "author_id": uid,
                    "query": q,
                    "full_ids": fe["input_ids"],
                    "edited_ids": ee["input_ids"],
                    "full_mask": fm.tolist(),
                    "edited_mask": em.tolist(),
                    "mapping": mapping.tolist(),
                    "alignment": audit,
                    "alphabet": alphabet,
                    "source_full": packet["queries"][q]["readouts"]["full"][
                        "conditional_score"
                    ],
                    "source_edit": packet["queries"][q]["readouts"]["e1"][
                        "conditional_score"
                    ],
                }
            )
    write(out / "private_design.json", design)
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/level_response.py",
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
        ROOT / "grain_research/vector_mechanisms/context_carrier.py",
        ROOT / "grain_research/vector_mechanisms/query_functional.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_COMPLETE_LEVEL_RESPONSE_V1",
            "authors": 12,
            "units": 36,
            "reference_authors": 2,
            "atlas_authors": 4,
            "max_calls": 288,
            "max_seconds": 1200,
            "full_calls": 72,
            "suffix_calls": 216,
            "gradient_calls": 0,
            "labels_read": False,
            "model_sha256": source["model_sha256"],
            "reference_sha256": sha(out / "reference.npz"),
            "input_sha256": sha(out / "private_design.json"),
            "source_manifest_sha256": sha(SOURCE / "manifest.json"),
            "source_material_sha256": source["material_sha256"],
            "source_split_sha256": sha(BRIDGE / "split_manifest.json"),
            "lineage": lineage,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "question_tokens": fixed_tokens,
            "replay_tolerance": 2e-6,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed_units": 0},
    )
    print(
        {
            "status": "PREPARED",
            "authors": 12,
            "units": 36,
            "max_calls": 288,
            "labels_read": False,
        },
        flush=True,
    )


def verify(out, code=False):
    """Verify frozen objects; code checks apply to execution rather than later analysis."""
    m = read(out / "manifest.json")
    for name, digest in (
        ("reference.npz", m["reference_sha256"]),
        ("private_design.json", m["input_sha256"]),
    ):
        if sha(out / name) != digest:
            raise ValueError("Frozen input changed")
    if code:
        for name, digest in m["code_hashes"].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Frozen implementation changed: {name}")
    return m


def run(out):
    """Acquire complete units under an explicit call/time ceiling, with zero gradients."""
    m, prior = verify(out, code=True), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete; no new calls")
        return
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Model changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    current, old = observer.manifest(), read(SOURCE / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if current[key] != old[key]:
            raise ValueError(f"Observer changed: {key}")
    write(out / "observer.json", current)
    with np.load(out / "reference.npz", allow_pickle=False) as a:
        ref = dict(a)
    start, complete = time.monotonic(), 0

    def status(state):
        """Retain actual compute counters, including failed or resumed work."""
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": complete,
            "planned_units": 36,
            "labels_read": False,
            "gradient_calls": 0,
        }
        write(out / "run_status.json", value)
        print(value, flush=True)

    try:
        for d in read(out / "private_design.json"):
            q = d["query"]
            path = out / "private_packets" / f"unit_{d['index']:02d}_{q}.json"
            if path.exists():
                cached = read(path)
                if (
                    cached["author_id"] != d["author_id"]
                    or sha(path.with_suffix(".npz")) != cached["arrays_sha256"]
                ):
                    raise ValueError("Completed packet changed")
                complete += 1
                continue
            if (
                prior["calls"] + observer.calls + 8 > m["max_calls"]
                or prior["seconds"] + time.monotonic() - start > m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            h, kw, native = capture_boundary(observer, d["full_ids"], d["alphabet"])
            he, kwe, edit = capture_boundary(observer, d["edited_ids"], d["alphabet"])
            replay = replay_suffix(observer, he, kwe, d["alphabet"])[
                "conditional_score"
            ]
            errors = {
                "same_run_edit": abs(replay - edit["conditional_score"]),
                "source_full": abs(native["conditional_score"] - d["source_full"]),
                "source_edit": abs(edit["conditional_score"] - d["source_edit"]),
            }
            if max(errors.values()) > m["replay_tolerance"]:
                raise ValueError(f"Native replay mismatch: {errors}")
            aligned = align_states(h, he, d["mapping"])
            em, fm = (
                np.asarray(d["edited_mask"], bool),
                np.asarray(d["full_mask"], bool),
            )
            ep = aligned.clone()
            ep[:, :-1] = he[:, :-1]
            cells = {}
            for name, state in (
                ("f00", aligned),
                ("f10", split_context_state(aligned, he, em, "E")),
                ("f01", split_context_state(aligned, he, em, "P")),
                ("f11", ep),
            ):
                cells[name] = replay_suffix(observer, state, kwe, d["alphabet"])[
                    "conditional_score"
                ]
            anchor = h.clone()
            anchor[:, fm] = torch.as_tensor(
                ref[f"{q}__Z"], dtype=h.dtype, device=h.device
            )
            anchor[:, -1] = torch.as_tensor(
                ref[f"{q}__h"], dtype=h.dtype, device=h.device
            )
            anchor_score = replay_suffix(observer, anchor, kw, d["alphabet"])[
                "conditional_score"
            ]
            np.savez_compressed(
                path.with_suffix(".npz"),
                full_state=h[0].cpu().numpy(),
                edited_state=he[0].cpu().numpy(),
                mapping=np.asarray(d["mapping"]),
                full_P=h[0, fm].cpu().numpy(),
                edited_P=he[0, em].cpu().numpy(),
                full_Q=h[0, -1].cpu().numpy(),
                edited_Q=he[0, -1].cpu().numpy(),
            )
            write(
                path,
                {
                    "index": d["index"],
                    "author_id": d["author_id"],
                    "query": q,
                    "full": native["conditional_score"],
                    "edited": edit["conditional_score"],
                    "anchor_score": anchor_score,
                    "cells": cells,
                    "ledger": response_ledger(
                        native["conditional_score"], edit["conditional_score"], **cells
                    ),
                    "replay_errors": errors,
                    "calls": 8,
                    "arrays_sha256": sha(path.with_suffix(".npz")),
                },
            )
            complete += 1
            del h, he, aligned, anchor, ep, state, kw, kwe
            if complete % 3 == 0:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def analyze(out):
    """Use frozen formulas only; separate local P effects from complete objectives."""
    verify(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete paired panel required")
    with np.load(out / "reference.npz", allow_pickle=False) as a:
        ref = dict(a)
    predictions, ledger, diagnostics = [], [], []
    for d in read(out / "private_design.json"):
        q = d["query"]
        path = out / "private_packets" / f"unit_{d['index']:02d}_{q}.json"
        r = read(path)
        if sha(path.with_suffix(".npz")) != r["arrays_sha256"]:
            raise ValueError("State cache changed")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            zf, ze, hf, he = [
                z[k].astype(float) for k in ("full_P", "edited_P", "full_Q", "edited_Q")
            ]
        get = lambda k, query=q: ref[f"{query}__{k}"]
        full = affine_parts(zf, hf, get("Z"), get("h"), get("G"), get("g"))
        edited = affine_parts(ze, he, get("Z"), get("h"), get("G"), get("g"))
        b = float(get("b"))
        for name in ("P", "Q", "PQ"):
            for target, actual, predicted, base in (
                ("level", r["full"], b + full[name], b),
                ("natural", r["edited"] - r["full"], edited[name] - full[name], 0),
                ("anchor", r["anchor_score"] - r["full"], -full[name], 0),
            ):
                predictions.append(
                    {
                        "index": d["index"],
                        "query": q,
                        "method": name,
                        "target": target,
                        "actual": actual,
                        "predicted": predicted,
                        "baseline": base,
                    }
                )
        variants, _ = matrix_variants(get("G"))
        for name, matrix in variants.items():
            predictions.append(
                {
                    "index": d["index"],
                    "query": q,
                    "method": name,
                    "target": "local_P",
                    "actual": r["ledger"]["P_allocated"],
                    "predicted": matrix_response(matrix, ze - zf),
                    "baseline": 0,
                }
            )
        charts = []
        for zp, hp in ((zf, hf), (ze, he)):
            charts.append(
                atlas_readout(
                    zp,
                    hp,
                    get("chart_Z"),
                    get("chart_h"),
                    get("chart_G"),
                    get("chart_g"),
                    get("chart_b"),
                )
            )
        for target, actual, predicted, base in (
            ("level", r["full"], charts[0]["score"], b),
            (
                "natural",
                r["edited"] - r["full"],
                charts[1]["score"] - charts[0]["score"],
                0,
            ),
        ):
            predictions.append(
                {
                    "index": d["index"],
                    "query": q,
                    "method": "atlas",
                    "target": target,
                    "actual": actual,
                    "predicted": predicted,
                    "baseline": base,
                }
            )
        diagnostics.append(
            {
                "index": d["index"],
                "query": q,
                "anchor_distance": float(
                    np.sqrt(np.sum((zf - get("Z")) ** 2) + np.sum((hf - get("h")) ** 2))
                ),
                "anchor_residual": r["anchor_score"] - r["full"] + full["PQ"],
                "chart_full": charts[0]["chart"],
                "chart_edit": charts[1]["chart"],
                "full_outside_unit": int(not 0 <= b + full["PQ"] <= 1),
            }
        )
        ledger.append(dict(index=d["index"], query=q, **r["ledger"]))
    frame = pd.DataFrame(predictions)
    frame.to_csv(out / "private_predictions.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "private_ledger.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "private_diagnostics.csv", index=False)
    metrics = []
    for (q, target, method), g in frame.groupby(["query", "target", "method"]):
        metrics.append(
            dict(
                query=q,
                target=target,
                method=method,
                **paired_quality(
                    g.actual.to_numpy(), g.predicted.to_numpy(), g.baseline.to_numpy()
                ),
            )
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "scope": "12-author frozen construction, no labels or new gradients",
            "cost": read(out / "run_status.json"),
            "reference_sha256": sha(out / "reference.npz"),
            "prediction_sha256": sha(out / "private_predictions.csv"),
            "metrics": metrics,
            "diagnostics": pd.DataFrame(diagnostics)
            .groupby("query")
            .agg(
                outside_unit=("full_outside_unit", "sum"),
                anchor_distance_mean=("anchor_distance", "mean"),
            )
            .reset_index()
            .to_dict("records"),
        },
    )
    view = pd.DataFrame(metrics)
    print(
        view[view.method.isin(["PQ", "atlas", "frozen_full"])][
            ["query", "target", "method", "rmse", "baseline_rmse", "mse_ratio", "r"]
        ].to_string(index=False),
        flush=True,
    )


def main():
    """Run explicit offline phases; private materials remain in ignored results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "analyze"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "complete_level_response_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private output must remain under ignored results")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "run": run, "analyze": analyze}[args.phase](out)


if __name__ == "__main__":
    main()
