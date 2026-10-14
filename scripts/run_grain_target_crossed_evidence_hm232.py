#!/usr/bin/env python3
"""Run one complete source-criterion xreader-criterion intervention, without fit."""

import argparse
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_comparison_geometry as G
from grain_research.construction import math_target_crossed_evidence as M
from grain_research.vector_mechanisms.direct_traits import BIG5
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_target_crossed_evidence_hm232.json"
DEFAULT = ROOT / "results/grain_math_response_v1/target_crossed_evidence_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TARGET_CROSSED_EVIDENCE_PLAN_HM232.md",
    "grain_research/construction/math_target_crossed_evidence.py",
    "grain_research/construction/math_full_attention_rows.py",
    "scripts/run_grain_target_crossed_evidence_hm232.py",
    "tests/test_grain_target_crossed_evidence.py",
    "grain_research/construction/math_author_comparison_geometry.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
)


def arrays(path):
    """Read one registered numeric artifact without object pickle."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def body_support(tokenizer, row):
    """Recover exact author-body addresses using verified native token offsets."""
    text = tokenizer.decode(row, skip_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != row:
        raise ValueError("Decoded body offsets do not round-trip native tokens")
    start = text.index("Writer A:\n")
    end = text.index("\nEvidence ends here.", start)
    body = [
        i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start
    ]
    if not body or body != list(range(body[0], body[-1] + 1)) or body[0] < 1:
        raise ValueError("Complete contiguous two-writer body required")
    return body


def prepare(out):
    """Freeze all evidence/criteria/recipients before new mechanism observations."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement of prior or uncertain packet")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    old = R.read(parent / "units.json")
    if len(old) != 335 or (parent / "ACTIVE.lock").exists():
        raise ValueError("Only the completed original comparison observer")
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    units = []
    for i in range(64):
        chunk = old[5 * i : 5 * (i + 1)]
        if [u["target"] for u in chunk] != list(range(5)):
            raise ValueError("Changed source target order")
        rows = [u["input_ids"] for u in chunk]
        bodies = [body_support(tok, r) for r in rows]
        if any(b != bodies[0] for b in bodies):
            raise ValueError("Source criteria changed author body addresses")
        units.append(
            {
                "edge": chunk[0]["edge"],
                "order": chunk[0]["order"],
                "left": chunk[0]["left"],
                "right": chunk[0]["right"],
                "input_ids": rows,
                "body": bodies[0],
            }
        )
    receiver = {"input_ids": []}
    for task in BIG5:
        tail = G.render(["empty"] * 12, ["empty"] * 12, task).split(
            "Evidence ends here.", 1
        )[1]
        user = "Writer A:\n\nWriter B:\n\nEvidence ends here." + tail
        ids = tok.apply_chat_template(
            [
                {"role": "system", "content": G.SYSTEM},
                {"role": "user", "content": user},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if not ids or any(not isinstance(i, int) for i in ids):
            raise ValueError("Native recipient token IDs required")
        receiver["input_ids"].append(list(ids))
    rows = [r[:-1] for u in units for r in u["input_ids"]] + [
        r[:-1] for r in receiver["input_ids"]
    ]
    count = 0
    for j in range(1, min(map(len, rows)) + 1):
        if len({r[-j] for r in rows}) != 1:
            break
        count = j
    for u in (*units, receiver):
        support = M.suffix_support(u["input_ids"], count)
        if "body" in u and min(support[0]) <= max(u["body"]):
            raise ValueError("Carrier overlaps source evidence")
    out.mkdir(parents=True, exist_ok=False)
    (out / "capture").mkdir()
    (out / "cold").mkdir()
    R.write(out / "units.json", units)
    R.write(out / "receiver.json", receiver)
    R.write(out / "bindings.json", R.read(parent / "bindings.json"))
    R.archive(out / "head.npz", **arrays(parent / "head.npz"))
    files = [
        parent / p
        for p in (
            "manifest.json",
            "units.json",
            "bindings.json",
            "head.npz",
            "verification_seal.json",
        )
    ]
    files += [parent / "batches" / f"{i:03d}.npz" for i in range(64)]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "suffix_tokens": count,
            "alphabet": pm["alphabet"],
            "model_files": pm["model_files"],
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
            "neural_compiler_and_reader": True,
            "cold_does_not_read_source_text_or_application_labels": True,
        },
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "events": 288,
            "edges": 32,
            "orders": 2,
            "source_criteria": list(BIG5),
            "recipient_criteria": list(BIG5),
            "suffix_tokens": count,
            "source_lengths": [list(map(len, u["input_ids"])) for u in units],
            "recipient_lengths": list(map(len, receiver["input_ids"])),
            "source_capture_batches": 128,
            "cold_batches": 386,
            "new_author_texts": 0,
            "source_answers_or_application_y_in_capsules": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "units.json",
            "receiver.json",
            "bindings.json",
            "head.npz",
            "input_quality.json",
        ),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0, "model_seconds": 0.0})
    print(f"HM232 PREPARED:128capture+386cold,5x5cross,suffix={count}", flush=True)


def observer_for(manifest):
    """Use the cached frozen observer, never download or change weights."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    for name, digest in manifest["model_files"].items():
        if R.digest(MODEL / name) != digest:
            raise ValueError("Changed frozen model")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    cfg = manifest["config"]
    stamp = time.monotonic()
    obs = DecoderObserver.load(
        str(MODEL),
        device=cfg["device"],
        dtype=cfg["dtype"],
        max_tokens=cfg["max_tokens"],
    )
    return obs, time.monotonic() - stamp


def capture(out):
    """Encode native/closed source increments once, retaining every signed account."""
    import torch

    m = R.check(out)
    cfg, count = m["config"], m["suffix_tokens"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No capture retry of a terminal or uncertain packet")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    elapsed, completed, pending, files, capsules, replay = 0.0, 0, None, [], [], []
    try:
        obs, load = observer_for(m)
        alpha = m["alphabet"]
        if not np.array_equal(
            obs.model.lm_head.weight.detach()[alpha].float().cpu().numpy(),
            arrays(out / "head.npz")["weights"],
        ):
            raise ValueError("Changed output alphabet/head")
        for i, unit in enumerate(R.read(out / "units.json")):
            banks = []
            for mode in ("native", "closed"):
                if elapsed >= cfg["max_model_seconds"]:
                    raise RuntimeError(
                        "Finite model cap reached; preserve partial source"
                    )
                pending = [i, mode]
                H.status(
                    out,
                    {
                        "status": "CAPTURING",
                        "pid": os.getpid(),
                        "completed_batches": completed,
                        "pending": pending,
                        "model_seconds": elapsed,
                    },
                )
                stamp = time.monotonic()
                try:
                    logits, states, values, receipts = M.observe(
                        obs, unit, count, closed=mode == "closed"
                    )
                finally:
                    elapsed += time.monotonic() - stamp
                selected = logits[:, alpha]
                if mode == "native":
                    old = arrays(ROOT / cfg["parent"] / "batches" / f"{i:03d}.npz")
                    error = float(
                        np.max(
                            abs(
                                np.diff(selected, axis=1)
                                - np.diff(old["logits"], axis=1)
                            )
                        )
                    )
                    replay.append(error)
                else:
                    error = None
                stem = f"capture/{i:03d}_{mode}"
                R.archive(
                    out / (stem + ".npz"),
                    logits=selected,
                    states=states,
                    values=values,
                    normalizer=logsumexp(logits, -1),
                )
                R.write(
                    out / (stem + ".json"),
                    {
                        "unit": i,
                        "mode": mode,
                        "receipts": receipts,
                        "native_log_odds_replay_error": error,
                    },
                )
                files.extend((stem + ".npz", stem + ".json"))
                if error is not None and error > cfg["native_output_replay_tolerance"]:
                    raise ValueError(
                        "Actual observation replay failed; no precision search"
                    )
                banks.append(values)
                completed += 1
                pending = None
                del logits, states, selected
                torch.mps.empty_cache()
            capsules.append(banks[0] - banks[1])
            print(f"HM232 CAPTURE {i + 1}/64;modelsec={elapsed:.3f}", flush=True)
        R.archive(out / "capsules.npz", values=np.stack(capsules))
        R.write(
            out / "capture_summary.json",
            {
                "batches": completed,
                "model_seconds": elapsed,
                "load_seconds": load,
                "native_log_odds_replay_max": max(replay),
                "capsule_shape": list(np.stack(capsules).shape),
            },
        )
        R.seal(
            out, "captured_seal.json", (*files, "capsules.npz", "capture_summary.json")
        )
        H.status(
            out,
            {
                "status": "CAPTURED",
                "completed_batches": completed,
                "model_seconds": elapsed,
                "pid": None,
            },
        )
    except BaseException as error:
        H.status(
            out,
            {
                "status": "FAILED",
                "completed_batches": completed,
                "pending": pending,
                "model_seconds": elapsed,
                "error": str(error),
                "pid": None,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def cold(out):
    """Read only exported numeric banks in a separate, excerpt-free process."""
    import torch

    m = R.check(out, ("captured_seal.json",))
    if R.read(out / "runtime.json")["status"] != "CAPTURED":
        raise ValueError("No cold retry")
    cfg, count, alpha = m["config"], m["suffix_tokens"], m["alphabet"]
    elapsed = R.read(out / "capture_summary.json")["model_seconds"]
    capsules = arrays(out / "capsules.npz")["values"]
    receiver = R.read(out / "receiver.json")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    completed, pending, files, source_elapsed = 0, None, [], elapsed
    try:
        obs, load = observer_for(m)
        jobs = [(-1, -1), (-1, 0)] + [(i, r) for i in range(64) for r in range(6)]
        baseline, baseline_logits = None, None
        for i, r in jobs:
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError(
                    "Finite model cap reached; preserve partial cold matrix"
                )
            pending = [i, r]
            H.status(
                out,
                {
                    "status": "COLD_RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "pending": pending,
                    "model_seconds": elapsed,
                },
            )
            if i < 0:
                donor = None if r < 0 else np.zeros_like(capsules[0])
            else:
                v = capsules[i]
                selected = v[:, r : r + 1] if r < 5 else v.mean(1, keepdims=True)
                donor = np.broadcast_to(selected, v.shape).copy() * cfg["dose"]
            stamp = time.monotonic()
            try:
                logits, states, _, receipts = M.observe(
                    obs, receiver, count, donor=donor
                )
            finally:
                elapsed += time.monotonic() - stamp
            hashes = [x["prefix_KV_hash"] for x in receipts]
            if baseline is None:
                baseline = hashes
                baseline_logits = logits[:, alpha].copy()
            elif baseline != hashes:
                raise ValueError("Cold import changed the pre-final recipient")
            if (
                i < 0
                and r == 0
                and not np.array_equal(logits[:, alpha], baseline_logits)
            ):
                raise ValueError("Zero import changed the native recipient")
            stem = (
                ("cold/baseline" if r < 0 else "cold/zero")
                if i < 0
                else f"cold/{i:03d}_{r}"
            )
            R.archive(
                out / (stem + ".npz"),
                logits=logits[:, alpha],
                states=states,
                normalizer=logsumexp(logits, -1),
            )
            R.write(
                out / (stem + ".json"),
                {"source_unit": i, "donor_criterion": r, "receipts": receipts},
            )
            files.extend((stem + ".npz", stem + ".json"))
            completed += 1
            pending = None
            if i >= 0 and r == 5:
                print(f"HM232 COLD {i + 1}/64;modelsec={elapsed:.3f}", flush=True)
            del logits, states
            torch.mps.empty_cache()
        R.write(
            out / "cold_summary.json",
            {
                "batches": completed,
                "model_seconds": elapsed - source_elapsed,
                "total_model_seconds": elapsed,
                "load_seconds": load,
                "prefix_preserved": True,
            },
        )
        R.seal(out, "cold_seal.json", (*files, "cold_summary.json"))
        H.status(
            out,
            {
                "status": "COLLECTED",
                "completed_batches": 128 + completed,
                "model_seconds": elapsed,
                "pid": None,
            },
        )
    except BaseException as error:
        H.status(
            out,
            {
                "status": "FAILED",
                "completed_cold_batches": completed,
                "pending": pending,
                "model_seconds": elapsed,
                "error": str(error),
                "pid": None,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def rms(x):
    """Finite response magnitude, not a psychological variance fraction."""
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def analyze(out):
    """Describe all crossed actions after collection, with no outcome-based fit."""
    R.check(out, ("captured_seal.json", "cold_seal.json"))
    readings = np.empty((32, 2, 6, 5))
    source = np.empty((32, 2, 2, 5))
    for i in range(64):
        e, order = divmod(i, 2)
        for r in range(6):
            z = arrays(out / "cold" / f"{i:03d}_{r}.npz")["logits"]
            readings[e, order, r] = z[:, 0] - z[:, 1]
        for mode, name in enumerate(("native", "closed")):
            z = arrays(out / "capture" / f"{i:03d}_{name}.npz")["logits"]
            source[e, order, mode] = z[:, 0] - z[:, 1]
    a = M.crossed_accounts(readings)
    native = (source[:, 0, 0] - source[:, 1, 0]) / 2
    native_increment = (
        (source[:, 0, 0] - source[:, 0, 1]) - (source[:, 1, 0] - source[:, 1, 1])
    ) / 2
    R.archive(
        out / "analysis.npz",
        readings=readings,
        source_readings=source,
        native=native,
        native_increment=native_increment,
        **a,
    )
    rows = []
    for e in range(32):
        for q in range(5):
            for r in range(5):
                rows.append(
                    {
                        "edge": e,
                        "reader": q,
                        "source": r,
                        "odd_action": float(a["matrix"][e, q, r]),
                        "common_action": float(a["common_action"][e, q]),
                        "source_specific_action": float(a["departure"][e, q, r]),
                        "interaction": float(a["interaction"][e, q, r]),
                        "native_odd": float(native[e, q]),
                        "native_body_increment": float(native_increment[e, q]),
                    }
                )
    R.table(out / "crossed_actions.csv", rows)
    cells = []
    for q in range(5):
        for r in range(5):
            t, d = a["matrix"][:, q, r], a["departure"][:, q, r]
            cells.append(
                {
                    "reader": q,
                    "source": r,
                    "mean": float(t.mean()),
                    "action_RMS": rms(t),
                    "specific_RMS": rms(d),
                    "native_inner_product": float(np.mean(t * native[:, q])),
                    "increment_inner_product": float(
                        np.mean(t * native_increment[:, q])
                    ),
                }
            )
    R.table(out / "target_matrix.csv", cells)
    matching = [
        c["increment_inner_product"] for c in cells if c["reader"] == c["source"]
    ]
    other = [c["increment_inner_product"] for c in cells if c["reader"] != c["source"]]
    summary = {
        "source_batches": 128,
        "cold_batches": 386,
        "endpoints": 2570,
        "source_criterion_departure_RMS": rms(a["departure"]),
        "crossed_interaction_RMS": rms(a["interaction"]),
        "common_action_RMS": rms(a["common_action"]),
        "total_action_RMS": rms(a["matrix"]),
        "even_action_RMS": rms(a["even"]),
        "matched_minus_other_native_increment_contact": float(
            np.mean(matching) - np.mean(other)
        ),
        "source_specific_RMS_by_reader": [rms(a["departure"][:, q]) for q in range(5)],
        "all25_cells_retained": True,
        "no_accuracy_or_human_trait_claim": True,
        "edges_share_authors_not_independent_population_samples": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        ("analysis.npz", "crossed_actions.csv", "target_matrix.csv", "summary.json"),
    )
    print(summary, flush=True)


def verify(out):
    """Use alternate arithmetic on every stored action and the fixed output head."""
    R.check(out, ("captured_seal.json", "cold_seal.json", "analyzed_seal.json"))
    a = arrays(out / "analysis.npz")
    error = float(
        np.max(
            abs(a["matrix"] - a["grand"] - a["reader"] - a["source"] - a["interaction"])
        )
    )
    direct, head = 0.0, 0.0
    w = arrays(out / "head.npz")["weights"]
    for i in range(64):
        e, order = divmod(i, 2)
        for r in range(6):
            z = arrays(out / "cold" / f"{i:03d}_{r}.npz")
            direct = max(
                direct,
                float(
                    np.max(
                        abs(
                            z["logits"][:, 0]
                            - z["logits"][:, 1]
                            - a["readings"][e, order, r]
                        )
                    )
                ),
            )
            head = max(
                head,
                float(
                    np.max(
                        abs(
                            z["states"].astype(float)
                            @ (w[0].astype(float) - w[1].astype(float))
                            - a["readings"][e, order, r]
                        )
                    )
                ),
            )
    record = {
        "PASS": error < 1e-12 and direct == 0 and head < 1e-4,
        "decomposition_error": error,
        "direct_error": direct,
        "head_error": head,
        "same_agent_not_blind": True,
        "all25_cells": True,
    }
    R.write(out / "verification.json", record)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not record["PASS"]:
        raise ValueError("Recorded verification failure; no precision rescue")
    print(record, flush=True)


def main():
    """Each stage is explicit; terminal/uncertain runs cannot be retried here."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=("prepare", "capture", "cold", "analyze", "verify", "check")
    )
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    if a.stage == "check":
        R.check(
            a.output,
            (
                "captured_seal.json",
                "cold_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        )
        print("HM232 all seals PASS;not full original-scope completion", flush=True)
    else:
        globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
