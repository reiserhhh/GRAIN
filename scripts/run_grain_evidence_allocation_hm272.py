#!/usr/bin/env python3
"""One finite source-linked evidence composition experiment; no accuracy search."""

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_join as M
from grain_research.construction import math_behavior_predictive as B
from grain_research.construction import math_evidence_allocation as A
from grain_research.construction import math_join_encoding as E
from grain_research.construction import math_late_condition as L
from grain_research.construction.math_linked_join import contact
from scripts import run_grain_join_encoding_hm271 as P
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.run_grain_event_anchor_field_hm267 import selected

CONFIG = "configs/grain_evidence_allocation_hm272.json"
DEFAULT = ROOT / "results/grain_math_response_v1/evidence_allocation_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVIDENCE_ALLOCATION_PLAN_HM272.md",
    "grain_research/construction/math_evidence_allocation.py",
    "scripts/run_grain_evidence_allocation_hm272.py",
    "tests/test_grain_evidence_allocation.py",
)


def prepare(out):
    """Freeze the existing same-event panel and new assembly operation beforehand."""
    if out.exists():
        raise FileExistsError("No overwrite of attempted science")
    cfg = R.read(ROOT / CONFIG)
    if cfg["event_share"] != 0.5:
        raise ValueError("One equal-bank operation, no composition grid")
    pm = R.check(
        ROOT / cfg["parent"],
        ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
    )
    sm = R.check(
        ROOT / cfg["source_parent"], ("computed_seal.json", "verification_seal.json")
    )
    parent, source = ROOT / cfg["parent"], ROOT / cfg["source_parent"]
    inputs = {**pm["inputs"], **sm["inputs"]}
    for base, names in (
        (
            parent,
            (
                "cases.json",
                "bank.json",
                "event_bindings.json",
                "observed_sums.npz",
                "collected_seal.json",
                "verification.json",
            ),
        ),
        (source, ("joint_actions.npz", "computed_seal.json")),
    ):
        for name in names:
            path = base / name
            inputs[str(path.relative_to(ROOT))] = R.digest(path)
    cases, bank = R.read(parent / "cases.json"), R.read(parent / "bank.json")
    if len(cases) != 24 or len(bank) != 50:
        raise ValueError("Entire same-evidence battery required")
    out.mkdir(parents=True)
    (out / "captures").mkdir()
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {
                **pm["code"],
                **sm["code"],
                **{p: R.digest(ROOT / p) for p in SCIENCE},
            },
            "inputs": inputs,
            "model_weights_sha256": pm["model_weights_sha256"],
            "original_scope_complete": False,
        },
    )
    R.write(out / "cases.json", cases)
    R.write(out / "bank.json", bank)
    R.seal(out, "prepared_seal.json", ("cases.json", "bank.json"))
    H.status(out, {"state": "PREPARED", "done_batches": 0, "model_seconds": 0})
    print("HM272 PREPARED one equal-bank reader; no model observation", flush=True)


def collect(out):
    """Observe the declared reader once with unchanged isolated event encoders."""
    import torch

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No repeat of attempted collection")
    cfg = manifest["config"]
    cases, bank = R.read(out / "cases.json"), R.read(out / "bank.json")
    basis = [
        selected(
            ROOT / cfg["basis_parent"] / f"basis/{i:02d}.npz", ("covectors", "bias")
        )
        for i in range(50)
    ]
    old = selected(ROOT / cfg["parent"] / "observed_sums.npz", ("chosen", "partition"))
    done, seconds, pending, files, receipts, encoders = 0, 0.0, None, [], [], []
    errors = {
        "native": 0.0,
        "head": 0.0,
        "prefix_mutation": 0.0,
        "closed_weight": 0.0,
        "outside": 0.0,
        "bank_balance": 0.0,
        "row_mass": 0.0,
    }
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))

    def mark(state, error=None):
        """Record a live address without extending the fixed budget."""
        H.status(
            out,
            {
                "state": state,
                "pid": os.getpid(),
                "done_batches": done,
                "model_seconds": seconds,
                "pending": pending,
                "error": error,
            },
        )

    def timed(fn):
        """Include failed calls and controls in model time."""
        nonlocal seconds
        if seconds >= cfg["max_model_seconds"]:
            raise TimeoutError("1200-model-second cap; no automatic continuation")
        began = time.monotonic()
        try:
            result = fn()
            torch.mps.synchronize()
            return result
        finally:
            seconds += time.monotonic() - began

    try:
        mark("RUNNING")
        observer = J.offline_model(manifest)
        for case in cases:
            u = case["unit_id"]
            for order in range(2):
                for corner, prefix in enumerate(case["prefixes"][order]):
                    addr = case["supports"][order][corner]
                    pending = {
                        "unit": u,
                        "order": order,
                        "corner": corner,
                        "operation": "same_isolated_encoding",
                    }
                    mark("RUNNING")
                    cache, layers = timed(
                        partial(
                            E.encode,
                            observer,
                            prefix,
                            addr["body_start"],
                            addr["first_end"],
                        )
                    )
                    encoders.append({**pending, "layers": layers})
                    before = L.cache_arrays(cache)
                    errors["closed_weight"] = max(
                        errors["closed_weight"], *(r["closed_weight"] for r in layers)
                    )
                    if u == corner == 0:
                        pending = {
                            "unit": u,
                            "order": order,
                            "corner": corner,
                            "operation": "unmodified_reader_guard",
                        }
                        mark("RUNNING")
                        guard = timed(
                            partial(
                                B.observe_candidates,
                                observer,
                                cache,
                                case["suffix"],
                                [r["target_ids"] for r in bank[:2]],
                            )
                        )
                        actual = np.array(
                            [r["chosen"].sum() - r["partition"].sum() for r in guard]
                        )
                        previous = (old["chosen"] - old["partition"])[
                            u, order, corner, :2
                        ]
                        errors["native"] = max(
                            errors["native"], float(np.max(abs(actual - previous)))
                        )
                        name = f"control_{order}.npz"
                        R.archive(out / name, native_actual=actual, native_old=previous)
                        files.append(name)
                    for first in range(0, 50, cfg["batch_size"]):
                        items = bank[first : first + cfg["batch_size"]]
                        pending = {
                            "unit": u,
                            "order": order,
                            "corner": corner,
                            "start": first,
                            "operation": "equal_bank_reader",
                        }
                        mark("RUNNING")
                        rows, layers = timed(
                            partial(
                                A.observe,
                                observer,
                                cache,
                                case["suffix"],
                                [r["target_ids"] for r in items],
                                addr,
                            )
                        )
                        receipts.append({**pending, "layers": layers})
                        for layer in layers:
                            for dst, src in (
                                ("outside", "outside_error"),
                                ("bank_balance", "bank_balance_error"),
                                ("row_mass", "row_mass_error"),
                            ):
                                errors[dst] = max(errors[dst], layer[src])
                        for item, row in zip(items, rows, strict=True):
                            b = basis[item["item"]]
                            if any(not np.array_equal(row[k], b[k]) for k in b):
                                raise ValueError("Changed complete candidate covectors")
                            replay = (
                                np.sum(
                                    row["states"].astype(float)
                                    * b["covectors"].astype(float),
                                    -1,
                                )
                                + b["bias"]
                            )
                            errors["head"] = max(
                                errors["head"],
                                float(np.max(abs(replay - row["chosen"]))),
                            )
                        name = f"captures/{done:04d}.npz"
                        R.archive(
                            out / name,
                            chosen=np.concatenate([r["chosen"] for r in rows]),
                            partition=np.concatenate([r["partition"] for r in rows]),
                            first_states=np.stack([r["states"][0] for r in rows]),
                            first_covectors=np.stack([r["covectors"][0] for r in rows]),
                            first_bias=np.array([r["bias"][0] for r in rows]),
                            offsets=np.cumsum(
                                [0] + [len(r["target_ids"]) for r in items]
                            ),
                            items=np.array([r["item"] for r in items]),
                            unit=u,
                            order=order,
                            corner=corner,
                            start=first,
                        )
                        files.append(name)
                        done += 1
                        pending = None
                        if (
                            max(errors.values()) > cfg["native_tolerance"]
                            or errors["outside"]
                            or errors["closed_weight"]
                        ):
                            raise ArithmeticError(
                                "Operation contract failed; no precision rescue"
                            )
                        mark("RUNNING")
                    after = L.cache_arrays(cache)
                    errors["prefix_mutation"] = max(
                        errors["prefix_mutation"],
                        *(float(np.max(abs(after[k] - v))) for k, v in before.items()),
                    )
                    if errors["prefix_mutation"]:
                        raise ArithmeticError("Reader changed frozen prefix")
                    del cache, before, after
            print(
                json.dumps({"unit": u, "done_batches": done, "model_seconds": seconds}),
                flush=True,
            )
        R.write(out / "hook_receipts.json", receipts)
        R.write(out / "encoder_receipts.json", encoders)
        R.write(
            out / "collection_quality.json",
            {
                "batches": done,
                "sequences": 9600,
                "prefixes": 192,
                "model_seconds": seconds,
                "errors": errors,
                "new_API_generation_training_private_calls": 0,
            },
        )
        files += [
            "hook_receipts.json",
            "encoder_receipts.json",
            "collection_quality.json",
        ]
        R.seal(out, "collected_seal.json", files)
        mark("COMPLETE")
    except Exception as exc:
        mark("PARTIAL_BUDGET" if isinstance(exc, TimeoutError) else "FAILED", repr(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


def analyze(out):
    """Compare explicit unit composition to native bank selection on SAME sources."""
    cfg = R.check(out, ("collected_seal.json",))["config"]
    declared, head = P.sums(out)
    native = selected(
        ROOT / cfg["parent"] / "observed_sums.npz", ("chosen", "partition")
    )
    bank = R.read(out / "bank.json")
    lengths = np.array([len(r["target_ids"]) for r in bank])
    accounts = {
        name: M.interaction_accounts(z["chosen"], z["partition"], lengths, bank)
        for name, z in (("native", native), ("declared", declared))
    }
    arrays, rows, diagonal, levels = {}, [], {}, {}
    with np.load(
        ROOT / cfg["source_parent"] / "joint_actions.npz", allow_pickle=False
    ) as source:
        for scale in ("sum", "per_token"):
            for kind in ("balanced_key", "common_support", "equal_item_key"):
                parts = A.split(
                    accounts["native"][scale][kind], accounts["declared"][scale][kind]
                )
                for part, value in parts.items():
                    arrays[f"{scale}/{kind}/{part}"] = value
                    levels[f"{scale}/{kind}/{part}"] = {
                        "mean_by_order": value.mean(0).tolist(),
                        "rms_by_order": np.sqrt(np.mean(value**2, 0)).tolist(),
                    }
                if kind == "equal_item_key":
                    continue
                for assignment in ("actual", "fixed_null"):
                    for space in ("E5", "GRAIN15"):
                        for view in ("pooled_comments", "registered_windows"):
                            for component in (
                                "linked",
                                "profile_product",
                                "linkage_correction",
                            ):
                                name = f"{assignment}/{space}/{view}/{component}"
                                d = source[name]
                                for part, value in parts.items():
                                    matrices = [
                                        contact(d[:, o], value[:, o]) for o in range(2)
                                    ]
                                    diagonal[f"{scale}/{kind}/{name}/{part}"] = [
                                        np.diag(x).tolist() for x in matrices
                                    ]
                                    for order, matrix in enumerate(matrices):
                                        for q, r in np.ndindex(5, 5):
                                            rows.append(
                                                {
                                                    "scale": scale,
                                                    "native_kind": kind,
                                                    "assignment": assignment,
                                                    "space": space,
                                                    "source_view": view,
                                                    "source_component": component,
                                                    "reader_part": part,
                                                    "order": order,
                                                    "source_target": q,
                                                    "native_target": r,
                                                    "contact": float(matrix[q, r]),
                                                }
                                            )
    R.archive(out / "observed_sums.npz", **declared)
    R.archive(out / "stage_actions.npz", **arrays)
    R.table(out / "all_contacts.csv", rows)
    R.write(
        out / "summary.json",
        {
            "matched_contacts": diagonal,
            "behavior_actions": levels,
            "all_signed_contacts": len(rows),
            "first_token_head_error": head,
            "collection": R.read(out / "collection_quality.json"),
            "not_accuracy_or_psychological_fraction": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        ("observed_sums.npz", "stage_actions.npz", "all_contacts.csv", "summary.json"),
    )
    print(json.dumps({"contacts": len(rows), "head_error": head}), flush=True)


def verify(out):
    """Re-sum literal observations and independently replay ALL signed contacts."""
    cfg = R.check(out, ("collected_seal.json", "analyzed_seal.json"))["config"]
    raw, head = P.sums(out)
    saved = selected(out / "observed_sums.npz", ("chosen", "partition"))
    literal = max(float(np.max(abs(raw[k] - saved[k]))) for k in raw)
    reconstruction, alternate, count = 0.0, 0.0, 0
    import csv

    with (
        np.load(out / "stage_actions.npz", allow_pickle=False) as stages,
        np.load(
            ROOT / cfg["source_parent"] / "joint_actions.npz", allow_pickle=False
        ) as sources,
    ):
        for scale in ("sum", "per_token"):
            for kind in ("balanced_key", "common_support", "equal_item_key"):
                key = f"{scale}/{kind}/"
                reconstruction = max(
                    reconstruction,
                    float(
                        np.max(
                            abs(
                                stages[key + "native"]
                                - stages[key + "declared"]
                                - stages[key + "allocation_increment"]
                            )
                        )
                    ),
                )
        with (out / "all_contacts.csv").open() as f:
            for row in csv.DictReader(f):
                d = sources[
                    "/".join(
                        row[k]
                        for k in (
                            "assignment",
                            "space",
                            "source_view",
                            "source_component",
                        )
                    )
                ]
                h = stages[
                    "/".join(row[k] for k in ("scale", "native_kind", "reader_part"))
                ]
                o, q, r = [
                    int(row[k]) for k in ("order", "source_target", "native_target")
                ]
                replay = (
                    sum(float(d[u, o, q]) * float(h[u, o, r]) for u in range(24)) / 24
                )
                alternate = max(alternate, abs(replay - float(row["contact"])))
                count += 1
    hooks = R.read(out / "hook_receipts.json")
    encoders = R.read(out / "encoder_receipts.json")
    valid = (
        len(hooks) == 1344
        and len(encoders) == 192
        and all(
            [r["layer"] for r in h["layers"]] == list(range(28))
            for h in hooks + encoders
        )
    )
    quality = R.read(out / "collection_quality.json")
    passed = (
        valid
        and literal == 0
        and reconstruction < 1e-12
        and alternate < 1e-12
        and max(quality["errors"].values()) < cfg["native_tolerance"]
        and head < cfg["native_tolerance"]
        and quality["errors"]["outside"]
        == quality["errors"]["prefix_mutation"]
        == quality["errors"]["closed_weight"]
        == 0
    )
    result = {
        "PASS": passed,
        "candidate_records": 9600,
        "layer_operation_receipts": 1344 * 28,
        "literal_sum_error": literal,
        "reconstruction_error": reconstruction,
        "alternate_contacts": count,
        "alternate_contact_error": alternate,
        "integrity": quality["errors"],
        "same_agent_not_blind": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not passed:
        raise ArithmeticError("Failed audit retained without rescoring")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM272 seals PASS; no model experiment")
    else:
        globals()[args.stage](args.out)
