#!/usr/bin/env python3
"""One same-evidence encoder/reader discriminator, without accuracy selection."""

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
from grain_research.construction import math_join_encoding as E
from grain_research.construction import math_late_condition as L
from grain_research.construction.math_linked_join import contact
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.run_grain_event_anchor_field_hm267 import selected

CONFIG = "configs/grain_join_encoding_hm271.json"
DEFAULT = ROOT / "results/grain_math_response_v1/join_encoding_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_JOIN_ENCODING_PLAN_HM271.md",
    "grain_research/construction/math_join_encoding.py",
    "scripts/run_grain_join_encoding_hm271.py",
    "tests/test_grain_join_encoding.py",
    "tests/test_grain_continuation_routes.py",
)


def prepare(out):
    """Recover exact two-block addresses from frozen tokens, without opening bodies."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite or repeat of a finite packet")
    cfg = R.read(ROOT / CONFIG)
    parent, source = ROOT / cfg["parent"], ROOT / cfg["source_parent"]
    pm = R.check(parent, ("analyzed_seal.json", "verification_seal.json"))
    sm = R.check(source, ("computed_seal.json", "verification_seal.json"))
    tok = AutoTokenizer.from_pretrained(str(J.MODEL), local_files_only=True)
    header, suffix = B.observation_parts(tok, [0])
    header = header[:-1]
    separator = tok.encode("\n\n", add_special_tokens=False)
    cases, bindings, bank = [
        R.read(parent / p) for p in ("cases.json", "bindings.json", "bank.json")
    ]
    if len(cases) != 24 or len(bindings) != 24 or len(bank) != 50 or not separator:
        raise ValueError("Complete existing joint battery required")
    for u, case in enumerate(cases):
        row = bindings[u]
        if case["unit_id"] != u or row["unit_id"] != u or case["suffix"] != suffix:
            raise ValueError("Changed event/suffix identity")
        supports = []
        for order in range(2):
            support = []
            for corner, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                sizes = [
                    row["events"][w]["own_tokens" if s == 0 else "foreign_tokens"]
                    for w, s in enumerate((a, b))
                ]
                if order:
                    sizes.reverse()
                prefix = case["prefixes"][order][corner]
                start, end = len(header), len(header) + sizes[0]
                second = end + len(separator)
                if (
                    prefix[:start] != header
                    or prefix[end:second] != separator
                    or len(prefix) != second + sizes[1]
                ):
                    raise ValueError("Literal segment boundaries changed")
                if order:
                    original = case["prefixes"][0][corner]
                    first = original[start : start + sizes[1]]
                    last = original[start + sizes[1] + len(separator) :]
                    if prefix[start:end] != last or prefix[second:] != first:
                        raise ValueError("Registered reverse-order tokens changed")
                support.append(
                    {
                        "body_start": start,
                        "first_end": end,
                        "second_start": second,
                        "prefix_length": len(prefix),
                    }
                )
            supports.append(support)
        case["supports"] = supports
    for order in range(2):
        addr = cases[0]["supports"][order][0]
        block = cases[0]["prefixes"][order][0][addr["body_start"] : addr["first_end"]]
        if len(set(block)) < 2:
            raise ValueError(
                "Predeclared audit rotation must actually change first evidence"
            )
    inputs = {**pm["inputs"], **sm["inputs"]}
    for base, names in (
        (parent, ("cases.json", "bank.json", "bindings.json", "observed_sums.npz")),
        (source, ("joint_actions.npz", "summary.json")),
    ):
        for name in names:
            path = base / name
            inputs[str(path.relative_to(ROOT))] = R.digest(path)
    for i in range(50):
        path = ROOT / cfg["basis_parent"] / f"basis/{i:02d}.npz"
        inputs[str(path.relative_to(ROOT))] = R.digest(path)
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
            "no_raw_private_or_questionnaire_read": True,
            "original_scope_complete": False,
        },
    )
    R.write(out / "cases.json", cases)
    R.write(out / "bank.json", bank)
    R.write(out / "event_bindings.json", bindings)
    R.seal(
        out, "prepared_seal.json", ("cases.json", "bank.json", "event_bindings.json")
    )
    H.status(out, {"state": "PREPARED", "done_batches": 0, "model_seconds": 0})
    print("PREPARED192 exact prefix addresses; no pretrained observation", flush=True)


def collect(out):
    """Observe one isolated-encoding arm; keep native reading and all candidate tokens."""
    import torch

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No retry/resume of an attempted scientific collection")
    cfg = manifest["config"]
    cases, bank = R.read(out / "cases.json"), R.read(out / "bank.json")
    basis = [
        selected(
            ROOT / cfg["basis_parent"] / f"basis/{i:02d}.npz", ("covectors", "bias")
        )
        for i in range(50)
    ]
    lock, files, done, seconds, pending = out / "ACTIVE.lock", [], 0, 0.0, None
    errors = {
        "closed_weight": 0.0,
        "prefix_mutation": 0.0,
        "head": 0.0,
        "native": 0.0,
        "header_first_cache": 0.0,
        "later_isolation": 0.0,
    }
    receipts = []
    with lock.open("x") as f:
        f.write(str(os.getpid()))

    def mark(state, error=None):
        """Keep a live address and cumulative cap without restarting failed science."""
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
        """Charge every actual model call, including failed controls."""
        nonlocal seconds
        if seconds >= cfg["max_model_seconds"]:
            raise TimeoutError("Fixed1200-model-second cap exhausted")
        began = time.monotonic()
        try:
            value = fn()
            torch.mps.synchronize()
            return value
        finally:
            seconds += time.monotonic() - began

    try:
        mark("RUNNING")
        observer = J.offline_model(manifest)
        native_sums = selected(
            ROOT / cfg["parent"] / "observed_sums.npz", ("chosen", "partition")
        )
        for case in cases:
            u = case["unit_id"]
            for order in range(2):
                for corner, prefix in enumerate(case["prefixes"][order]):
                    addr = case["supports"][order][corner]
                    start, end = addr["body_start"], addr["first_end"]
                    pending = {
                        "unit": u,
                        "order": order,
                        "corner": corner,
                        "operation": "isolated_encode",
                    }
                    mark("RUNNING")
                    cache, layers = timed(
                        partial(E.encode, observer, prefix, start, end)
                    )
                    receipts.append({**pending, "layers": layers})
                    before = L.cache_arrays(cache)
                    if u == corner == 0:
                        pending = {
                            "unit": u,
                            "order": order,
                            "corner": corner,
                            "operation": "audit_controls",
                        }
                        mark("RUNNING")
                        normal = timed(partial(L.encode_context, observer, prefix))
                        full = L.cache_arrays(normal)
                        shifted = (
                            prefix[:start]
                            + prefix[start + 1 : end]
                            + prefix[start : start + 1]
                            + prefix[end:]
                        )
                        other, _ = timed(
                            partial(E.encode, observer, shifted, start, end)
                        )
                        rotated = L.cache_arrays(other)
                        errors["header_first_cache"] = max(
                            errors["header_first_cache"],
                            *(
                                float(
                                    np.max(
                                        abs(before[k][:, :, :end] - full[k][:, :, :end])
                                    )
                                )
                                for k in before
                            ),
                        )
                        errors["later_isolation"] = max(
                            errors["later_isolation"],
                            *(
                                float(
                                    np.max(
                                        abs(
                                            before[k][:, :, end:]
                                            - rotated[k][:, :, end:]
                                        )
                                    )
                                )
                                for k in before
                            ),
                        )
                        rows = timed(
                            partial(
                                B.observe_candidates,
                                observer,
                                normal,
                                case["suffix"],
                                [r["target_ids"] for r in bank[:2]],
                            )
                        )
                        actual = np.array(
                            [r["chosen"].sum() - r["partition"].sum() for r in rows]
                        )
                        old = (
                            native_sums["chosen"][u, order, corner, :2]
                            - native_sums["partition"][u, order, corner, :2]
                        )
                        errors["native"] = max(
                            errors["native"], float(np.max(abs(actual - old)))
                        )
                        name = f"control_{order}.npz"
                        R.archive(
                            out / name,
                            native_actual=actual,
                            native_old=old,
                            isolated_later=before["keys"][:, :, end:],
                            rotated_later=rotated["keys"][:, :, end:],
                        )
                        files.append(name)
                        del normal, other, full, rotated
                    for first in range(0, 50, cfg["batch_size"]):
                        items = bank[first : first + cfg["batch_size"]]
                        pending = {
                            "unit": u,
                            "order": order,
                            "corner": corner,
                            "start": first,
                            "operation": "native_behavior_read",
                        }
                        mark("RUNNING")
                        rows = timed(
                            partial(
                                B.observe_candidates,
                                observer,
                                cache,
                                case["suffix"],
                                [r["target_ids"] for r in items],
                            )
                        )
                        for item, row in zip(items, rows, strict=True):
                            b = basis[item["item"]]
                            if any(not np.array_equal(row[k], b[k]) for k in b):
                                raise ValueError(
                                    "Candidate vocabulary covector changed"
                                )
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
                            or errors["later_isolation"]
                            or errors["header_first_cache"]
                        ):
                            raise ArithmeticError(
                                "Instrument contract failed; retain without precision rescue"
                            )
                        mark("RUNNING")
                    after = L.cache_arrays(cache)
                    errors["prefix_mutation"] = max(
                        errors["prefix_mutation"],
                        *(float(np.max(abs(after[k] - v))) for k, v in before.items()),
                    )
                    if errors["prefix_mutation"]:
                        raise ValueError(
                            "Native suffix mutated isolated evidence cache"
                        )
                    del cache, before, after
            print(
                json.dumps({"unit": u, "done_batches": done, "model_seconds": seconds}),
                flush=True,
            )
        R.write(out / "hook_receipts.json", receipts)
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
        files += ["hook_receipts.json", "collection_quality.json"]
        R.seal(out, "collected_seal.json", files)
        mark("COMPLETE")
    except Exception as exc:
        mark("PARTIAL_BUDGET" if isinstance(exc, TimeoutError) else "FAILED", repr(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


def sums(out):
    """Re-sum exactly the registered captures, excluding filesystem metadata."""
    seal = R.read(out / "collected_seal.json")
    captures = sorted(p for p in seal if p.startswith("captures/"))
    if len(captures) != 1344:
        raise ValueError("Incomplete full candidate battery")
    arrays = {k: np.full((24, 2, 4, 50), np.nan) for k in ("chosen", "partition")}
    head_error = 0.0
    for path in captures:
        with np.load(out / path, allow_pickle=False) as z:
            u, order, corner, first = [
                int(z[k]) for k in ("unit", "order", "corner", "start")
            ]
            items, offsets = z["items"], z["offsets"]
            if (
                not np.array_equal(items, np.arange(first, min(first + 8, 50)))
                or offsets[0] != 0
                or offsets[-1] != len(z["chosen"])
            ):
                raise ValueError("Changed native observation addresses")
            for j, item in enumerate(items):
                a, b = offsets[j : j + 2]
                if not np.isnan(arrays["chosen"][u, order, corner, item]):
                    raise ValueError("Repeated candidate observation")
                for key, array in arrays.items():
                    array[u, order, corner, item] = z[key][a:b].sum()
            replay = (
                np.sum(
                    z["first_states"].astype(float)
                    * z["first_covectors"].astype(float),
                    -1,
                )
                + z["first_bias"]
            )
            head_error = max(
                head_error, float(np.max(abs(replay - z["chosen"][offsets[:-1]])))
            )
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Missing/failed candidate values cannot be imputed")
    return arrays, head_error


def analyze(out):
    """Contact frozen SOURCE components with both actual computational stages."""
    cfg = R.check(out, ("collected_seal.json",))["config"]
    iso, head_error = sums(out)
    native = selected(
        ROOT / cfg["parent"] / "observed_sums.npz", ("chosen", "partition")
    )
    bank = R.read(out / "bank.json")
    lengths = np.array([len(r["target_ids"]) for r in bank])
    accounts = {
        "native": M.interaction_accounts(
            native["chosen"], native["partition"], lengths, bank
        ),
        "isolated": M.interaction_accounts(
            iso["chosen"], iso["partition"], lengths, bank
        ),
    }
    arrays, rows, diagonal, levels = {}, [], {}, {}
    with np.load(
        ROOT / cfg["source_parent"] / "joint_actions.npz", allow_pickle=False
    ) as source:
        for scale in ("sum", "per_token"):
            for kind in ("balanced_key", "common_support", "equal_item_key"):
                parts = E.split(
                    accounts["native"][scale][kind], accounts["isolated"][scale][kind]
                )
                for part, value in parts.items():
                    arrays[f"{scale}/{kind}/{part}"] = value
                    levels[f"{scale}/{kind}/{part}"] = {
                        "mean_by_order": value.mean(0).tolist(),
                        "rms_by_order": np.sqrt(np.mean(value**2, 0)).tolist(),
                        "order_odd_rms": np.sqrt(
                            np.mean(((value[:, 0] - value[:, 1]) / 2) ** 2, 0)
                        ).tolist(),
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
                                        contact(d[:, order], value[:, order])
                                        for order in range(2)
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
    R.archive(out / "observed_sums.npz", **iso)
    R.archive(out / "stage_actions.npz", **arrays)
    R.table(out / "all_contacts.csv", rows)
    R.write(
        out / "summary.json",
        {
            "matched_contacts": diagonal,
            "behavior_actions": levels,
            "all_signed_contacts": len(rows),
            "first_token_head_error": head_error,
            "collection": R.read(out / "collection_quality.json"),
            "not_accuracy_human_behavior_or_unique_mediation": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        ("observed_sums.npz", "stage_actions.npz", "all_contacts.csv", "summary.json"),
    )
    print(
        json.dumps(
            {
                "analyzed": True,
                "contacts": len(rows),
                "first_token_head_error": head_error,
            }
        ),
        flush=True,
    )


def verify(out):
    """Audit literal captures, all layer support and signed contact reconstruction."""
    cfg = R.check(out, ("collected_seal.json", "analyzed_seal.json"))["config"]
    raw, head = sums(out)
    saved = selected(out / "observed_sums.npz", ("chosen", "partition"))
    literal = max(float(np.max(abs(raw[k] - saved[k]))) for k in raw)
    hooks = R.read(out / "hook_receipts.json")
    if len(hooks) != 192 or any(
        [r["layer"] for r in h["layers"]] != list(range(28))
        or any(r["closed_weight"] for r in h["layers"])
        for h in hooks
    ):
        raise ValueError("Incomplete actual all-layer encoding support receipts")
    reconstruction = 0.0
    with np.load(out / "stage_actions.npz", allow_pickle=False) as z:
        for scale in ("sum", "per_token"):
            for kind in ("balanced_key", "common_support", "equal_item_key"):
                prefix = f"{scale}/{kind}/"
                diff = (
                    z[prefix + "native"]
                    - z[prefix + "isolated"]
                    - z[prefix + "encoding_increment"]
                )
                reconstruction = max(reconstruction, float(np.max(abs(diff))))
    summary = R.read(out / "summary.json")
    contacts = summary["matched_contacts"]
    contact_error = 0.0
    for name, value in contacts.items():
        if name.endswith("/native"):
            base = name.removesuffix("native")
            diff = (
                np.asarray(value)
                - np.asarray(contacts[base + "isolated"])
                - np.asarray(contacts[base + "encoding_increment"])
            )
            contact_error = max(contact_error, float(np.max(abs(diff))))
    quality = R.read(out / "collection_quality.json")
    passed = (
        literal == 0
        and reconstruction < 1e-12
        and contact_error < 1e-12
        and head < cfg["native_tolerance"]
        and max(quality["errors"].values()) < cfg["native_tolerance"]
    )
    result = {
        "PASS": passed,
        "candidate_records": 9600,
        "actual_all_layer_receipts": 192 * 28,
        "literal_sum_error": literal,
        "stage_reconstruction_error": reconstruction,
        "contact_reconstruction_error": contact_error,
        "first_token_head_error": head,
        "integrity": quality["errors"],
        "same_agent_not_blind": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not passed:
        raise ArithmeticError("Audit failed; retain without new science/retry")
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
        print("HM271 registered seals PASS; no experiment started")
    else:
        globals()[args.stage](args.out)
