#!/usr/bin/env python3
"""Encode public contexts once, then observe complete text under late criteria."""

import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_predictive_link as A
from grain_research.construction import math_late_condition as M
from scripts import run_grain_conditional_likelihood_hm152 as L
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_late_condition_hm258.json"
DEFAULT = ROOT / "results/grain_math_response_v1/late_condition_observer_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_LATE_CONDITION_PLAN_HM258.md",
    "grain_research/construction/math_late_condition.py",
    "scripts/run_grain_late_condition_hm258.py",
    "scripts/score_grain_late_condition_cold.py",
    "tests/test_grain_late_condition.py",
    "grain_research/construction/math_author_predictive_link.py",
    "scripts/run_grain_conditional_likelihood_hm152.py",
    "scripts/run_grain_joint_persona_style_hm150.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
    "grain_research/vector_mechanisms/trace.py",
)


def prepare(out):
    """Freeze same-event late queries without reading application questionnaires."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement or restart of a prior packet")
    cfg = R.read(ROOT / CONFIG)
    if (
        cfg["authors"],
        cfg["directional_units"],
        cfg["hypotheses"],
        cfg["batch_size"],
    ) != (24, 48, 11, 4):
        raise ValueError("Registered finite scope changed")
    parent = ROOT / cfg["parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    old, bindings = (
        R.read(parent / "cases.json"),
        R.read(parent / "private_bindings.json"),
    )
    if len(old) != 48 or len(bindings) != 48:
        raise ValueError("All historical author directions are required")
    tok = AutoTokenizer.from_pretrained(
        J.MODEL, local_files_only=True, trust_remote_code=False
    )
    cases = []
    for case, binding in zip(old, bindings, strict=True):
        if case["unit_id"] != binding["unit_id"]:
            raise ValueError("Evidence address changed")
        target = tok.encode(binding["target"]["body"], add_special_tokens=False)
        if target != case["target_ids"]:
            raise ValueError("Full actual target token sequence changed")
        prefixes, suffixes = [], None
        for name in cfg["contexts"]:
            prefix, tail = M.observation_parts(
                tok, tok.encode(binding[name]["body"], add_special_tokens=False)
            )
            if suffixes is not None and suffixes != tail:
                raise ValueError("Context changed the target query itself")
            prefixes.append(prefix)
            suffixes = tail
        if (
            max(len(p) + len(s) + len(target) - 1 for p in prefixes for s in suffixes)
            > cfg["max_input_tokens"]
        ):
            raise ValueError("Complete raw input exceeds cap; no truncation")
        cases.append(
            {
                "unit_id": case["unit_id"],
                "author": case["author"],
                "context_window": case["context_window"],
                "prefixes": prefixes,
                "suffixes": suffixes,
                "target_ids": target,
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    (out / "prefixes").mkdir()
    source = S.selected(
        parent / "source_observations.npz",
        ("fields", "choice", "coordinates", "panel_indices"),
    )
    early = S.selected(parent / "actions.npz", ("stranger",))
    R.archive(out / "source_observations.npz", **source, early=early["stranger"])
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "prepared_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "cases.json",
            "private_bindings.json",
            "source_observations.npz",
            "actions.npz",
        )
    ]
    inputs += sorted(J.MODEL.glob("*.json")) + sorted(J.MODEL.glob("*.txt"))
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "model_weights_sha256": pm["model_weights_sha256"],
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.write(out / "cases.json", cases)
    quality = {
        "authors": 24,
        "directions": 2,
        "prefix_encodes": 96,
        "suffix_batches": 288,
        "conditional_sequences": 1056,
        "native_spot_sequences": 4,
        "target_length_range": [
            min(len(c["target_ids"]) for c in cases),
            max(len(c["target_ids"]) for c in cases),
        ],
        "max_input_tokens": max(
            len(p) + len(s) + len(c["target_ids"]) - 1
            for c in cases
            for p in c["prefixes"]
            for s in c["suffixes"]
        ),
        "exact_venue_matches": sum(b["exact_venue_match"] for b in bindings),
        "future_target_description_not_in_source_prefix": True,
        "prefix_only_public_actual_context": True,
        "new_humans_generation_labels_API_private_training": 0,
        "historical_exploration_not_holdout": True,
        "early_late_difference_not_pure_mediation": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("cases.json", "source_observations.npz", "input_quality.json"),
    )
    H.status(
        out,
        {
            "state": "PREPARED",
            "done_batches": 0,
            "prefixes": 0,
            "model_seconds": 0,
            "pending": None,
        },
    )
    print(json.dumps(quality), flush=True)


def collect(out):
    """Run one bounded cache-reuse study; retain every failure and signed output."""
    import torch

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No retry/resume of partial or terminal science")
    cfg, cases = manifest["config"], R.read(out / "cases.json")
    seconds, done, prefixes = 0.0, 0, 0
    head_error, native_error, immutable_error = 0.0, 0.0, 0.0
    files, native_rows, prefix_rows = [], [], []
    lock = out / "ACTIVE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    pending = None

    def mark(state):
        """Atomically expose the exact live finite operation and cumulative cost."""
        H.status(
            out,
            {
                "state": state,
                "pid": os.getpid(),
                "done_batches": done,
                "prefixes": prefixes,
                "model_seconds": seconds,
                "pending": pending,
            },
        )

    def timed(operation):
        """Charge failed calls too, and never renew the single scientific cap."""
        nonlocal seconds
        if seconds >= cfg["max_model_seconds"]:
            raise TimeoutError("Single cumulative cap; no renewal")
        began = time.monotonic()
        try:
            result = operation()
            torch.mps.synchronize()
            return result
        finally:
            seconds += time.monotonic() - began

    try:
        mark("RUNNING")
        observer = J.offline_model(manifest)
        for case in cases:
            basis = None
            for mode, prefix in enumerate(case["prefixes"]):
                pending = {
                    "unit": case["unit_id"],
                    "context": mode,
                    "operation": "target_blind_encode",
                }
                mark("RUNNING")
                cache = timed(partial(M.encode_context, observer, prefix))
                before = M.cache_arrays(cache)
                prefix_name = f"prefixes/{case['unit_id']:02d}_{mode}.npz"
                R.archive(out / prefix_name, **before)
                files.append(prefix_name)
                prefixes += 1
                for start in range(0, 11, 4):
                    suffixes = case["suffixes"][start : start + 4]
                    pending = {
                        "unit": case["unit_id"],
                        "context": mode,
                        "start": start,
                        "operation": "reuse_prefix",
                    }
                    mark("RUNNING")
                    result = timed(
                        partial(
                            M.observe_suffixes,
                            observer,
                            cache,
                            suffixes,
                            case["target_ids"],
                        )
                    )
                    if basis is None:
                        basis = {k: result[k] for k in ("covectors", "bias")}
                        basis_name = f"captures/basis_{case['unit_id']:02d}.npz"
                        R.archive(out / basis_name, **basis)
                        files.append(basis_name)
                    if any(not np.array_equal(result[k], basis[k]) for k in basis):
                        raise ValueError("Actual target output covectors changed")
                    replay = (
                        np.sum(
                            result["states"].astype(float)
                            * basis["covectors"].astype(float),
                            -1,
                        )
                        + basis["bias"]
                    )
                    head_error = max(
                        head_error, float(np.max(abs(replay - result["chosen"])))
                    )
                    if case["unit_id"] == 0 and start == 0:
                        specs = [(prefix + s, case["target_ids"]) for s in suffixes[:2]]
                        native = timed(partial(L.observe, observer, specs))
                        expected = native["chosen"].astype(float) - native[
                            "partition"
                        ].astype(float)
                        actual = result["chosen"][:2].astype(float) - result[
                            "partition"
                        ][:2].astype(float)
                        error = float(np.max(abs(expected - actual)))
                        native_error = max(native_error, error)
                        native_name = f"native_spot_{mode}.npz"
                        R.archive(out / native_name, cached=actual, full=expected)
                        files.append(native_name)
                        native_rows.append(
                            {
                                "context": mode,
                                "sequences": 2,
                                "max_probability_account_error": error,
                            }
                        )
                        if error > cfg["native_tolerance"]:
                            raise ArithmeticError(
                                "Cache reuse changes the declared native observation"
                            )
                    name = f"captures/{done:04d}.npz"
                    R.archive(
                        out / name,
                        chosen=result["chosen"],
                        partition=result["partition"],
                        states=result["states"],
                        unit=case["unit_id"],
                        mode=mode,
                        start=start,
                    )
                    files.append(name)
                    done += 1
                    pending = None
                    mark("RUNNING")
                after = M.cache_arrays(cache)
                error = max(float(np.max(abs(after[k] - before[k]))) for k in before)
                immutable_error = max(immutable_error, error)
                if error != 0:
                    raise ArithmeticError(
                        "Later targets changed the shared source memory"
                    )
                prefix_rows.append(
                    {
                        "unit": case["unit_id"],
                        "mode": mode,
                        "tokens": len(prefix),
                        "layers": len(before["keys"]),
                        "suffixes_using_same_cache": 11,
                        "mutation_error": error,
                    }
                )
                del cache, before, after
            print(
                json.dumps(
                    {
                        "unit": case["unit_id"],
                        "batches": done,
                        "prefixes": prefixes,
                        "model_seconds": seconds,
                    }
                ),
                flush=True,
            )
        quality = {
            "prefix_encodes": prefixes,
            "suffix_batches": done,
            "conditional_sequences": 1056,
            "native_spot_sequences": 4,
            "native_error": native_error,
            "head_error": head_error,
            "prefix_mutation_error": immutable_error,
            "model_seconds": seconds,
            "no_generation_answers_or_fitting": True,
        }
        R.write(out / "collection_quality.json", quality)
        R.table(out / "prefix_reuse_receipts.csv", prefix_rows)
        R.table(out / "native_receipts.csv", native_rows)
        files += [
            "collection_quality.json",
            "prefix_reuse_receipts.csv",
            "native_receipts.csv",
        ]
        R.seal(out, "collected_seal.json", files)
        mark("COMPLETE")
        lock.unlink()
        print(json.dumps(quality), flush=True)
    except Exception as exc:
        mark("PARTIAL_BUDGET" if isinstance(exc, TimeoutError) else "FAILED")
        R.write(
            out / "failure.json",
            {"error": repr(exc), "no_automatic_retry": True, "pending": pending},
        )
        raise


def observations(out):
    """Load every complete addressed result, never fill a missing cell with zero."""
    cases, readings = R.read(out / "cases.json"), []
    for case in cases:
        chunks = {key: [] for key in ("chosen", "partition")}
        for mode in range(2):
            by_mode = {key: [] for key in chunks}
            for local, start in enumerate(range(0, 11, 4)):
                batch = 6 * case["unit_id"] + 3 * mode + local
                with np.load(
                    out / f"captures/{batch:04d}.npz", allow_pickle=False
                ) as z:
                    if (int(z["unit"]), int(z["mode"]), int(z["start"])) != (
                        case["unit_id"],
                        mode,
                        start,
                    ):
                        raise ValueError("Conditional evidence address changed")
                    for key in chunks:
                        by_mode[key].append(z[key].astype(float))
            for key, values in chunks.items():
                values.append(np.concatenate(by_mode[key]))
        readings.append({key: np.stack(values) for key, values in chunks.items()})
    return cases, readings


def analyze(out):
    """Construct cold accounts and all signed SOURCE contacts, without selection."""
    R.check(out, ("collected_seal.json",))
    cases, readings = observations(out)
    effects, neutral, units, rows, token_rows = [], [], [], [], []
    for case, value in zip(cases, readings, strict=True):
        result = M.accounts(value["chosen"], value["partition"])
        effects.append(result["effect"].sum(-1))
        neutral.append(float(result["neutral"].sum()))
        units.append(
            {
                "unit_id": case["unit_id"],
                "chosen": value["chosen"].sum(-1).tolist(),
                "partition": value["partition"].sum(-1).tolist(),
            }
        )
        for q, target in enumerate(A.TARGETS):
            rows.append(
                {
                    "unit": case["unit_id"],
                    "author": case["author"],
                    "context_window": case["context_window"],
                    "target": target,
                    "nats": float(result["effect"][q].sum()),
                    "numerator": float(result["numerator"][q].sum()),
                    "normalizer": float(result["normalizer"][q].sum()),
                    "nats_per_token": float(result["effect"][q].mean()),
                }
            )
            for t in range(len(case["target_ids"])):
                token_rows.append(
                    {
                        "unit": case["unit_id"],
                        "target": target,
                        "position": t,
                        "nats": float(result["effect"][q, t]),
                        "numerator": float(result["numerator"][q, t]),
                        "normalizer": float(result["normalizer"][q, t]),
                    }
                )
    effects = np.asarray(effects)
    R.archive(out / "actions.npz", effect=effects, neutral=neutral)
    R.table(out / "action_accounts.csv", rows)
    R.table(out / "token_accounts.csv", token_rows)
    R.write(out / "cold_input.json", {"version": "HM258-v1", "units": units})
    cold = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_late_condition_cold.py"),
                "--input",
                str(out / "cold_input.json"),
            ],
            text=True,
        )
    )
    if cold["neural_runtime_loaded"]:
        raise ValueError("Cold observer loaded a neural runtime")
    R.write(out / "cold_output.json", cold)
    source = S.selected(
        out / "source_observations.npz", ("fields", "choice", "coordinates", "early")
    )
    contacts = {}
    lengths = np.asarray([len(c["target_ids"]) for c in cases])
    for name, reading in (("late", effects), ("early", source["early"])):
        for scale, y in (("sum", reading), ("per_token", reading / lengths[:, None])):
            for part, x in (
                ("full", source["fields"]),
                ("choice", source["choice"]),
                ("within", source["fields"] - source["choice"]),
            ):
                contacts[f"{name}/{part}/{scale}"] = A.contact(
                    x.reshape(2, 24, 5).mean(0), y.reshape(2, 24, 5).mean(0)
                ).tolist()
                for direction in range(2):
                    contacts[f"{name}/{part}/{scale}/direction{direction}"] = A.contact(
                        x.reshape(2, 24, 5)[direction], y.reshape(2, 24, 5)[direction]
                    ).tolist()
        contacts[f"{name}/GRAIN15/sum"] = A.contact(
            source["coordinates"].reshape(2, 24, 15).mean(0),
            reading.reshape(2, 24, 5).mean(0),
        ).tolist()
    R.write(out / "contacts.json", contacts)
    full, choice, within = [
        np.asarray(contacts[f"late/{k}/sum"]) for k in ("full", "choice", "within")
    ]
    summary = {
        "authors": 24,
        "target_order": list(A.TARGETS),
        "all_signed_contacts_retained": True,
        "target_blind_source_cache_reused": True,
        "late_mean_by_direction": effects.reshape(2, 24, 5).mean(1).tolist(),
        "late_rms_by_direction": np.sqrt(
            np.mean(effects.reshape(2, 24, 5) ** 2, 1)
        ).tolist(),
        "author_positive_count": (effects.reshape(2, 24, 5).mean(0) > 0)
        .sum(0)
        .tolist(),
        "full_contact_diagonal": np.diag(full).tolist(),
        "choice_contact_diagonal": np.diag(choice).tolist(),
        "within_contact_diagonal": np.diag(within).tolist(),
        "contact_by_direction": [
            np.diag(contacts[f"late/full/sum/direction{d}"]).tolist() for d in range(2)
        ],
        "choice_within_error": float(np.max(abs(full - choice - within))),
        "neutral_context_mean_by_direction": np.asarray(neutral)
        .reshape(2, 24)
        .mean(1)
        .tolist(),
        "early_late_not_mediated_fraction_or_accuracy": True,
        "not_human_posterior_or_new_mathematical_theorem": True,
        "collection": R.read(out / "collection_quality.json"),
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "actions.npz",
            "action_accounts.csv",
            "token_accounts.csv",
            "cold_input.json",
            "cold_output.json",
            "contacts.json",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Independently re-sum signed tokens, output heads and full SOURCE accounts."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    cases, readings = observations(out)
    cold = R.read(out / "cold_output.json")["rows"]
    error, head_error, count = 0.0, 0.0, 0
    effects = []
    for case, value, row in zip(cases, readings, cold, strict=True):
        ll = value["chosen"] - value["partition"]
        literal = []
        for q in range(5):
            h, l = 2 * q + 1, 2 * q + 2
            total = sum(
                ll[0, h, t] - ll[0, l, t] - ll[1, h, t] + ll[1, l, t]
                for t in range(ll.shape[-1])
            )
            error = max(error, abs(total - row["effect"][q]))
            literal.append(total)
            count += 1
        effects.append(literal)
        with np.load(
            out / f"captures/basis_{case['unit_id']:02d}.npz", allow_pickle=False
        ) as basis:
            w, bias = basis["covectors"].astype(float), basis["bias"].astype(float)
        for mode in range(2):
            for local in range(3):
                with np.load(
                    out / f"captures/{6 * case['unit_id'] + 3 * mode + local:04d}.npz",
                    allow_pickle=False,
                ) as z:
                    replay = (
                        np.einsum(
                            "btd,td->bt", z["states"].astype(float), w, optimize=False
                        )
                        + bias
                    )
                    head_error = max(
                        head_error, float(np.max(abs(replay - z["chosen"])))
                    )
    source = S.selected(out / "source_observations.npz", ("fields", "choice"))
    effects = np.asarray(effects).reshape(2, 24, 5).mean(0)
    contacts = R.read(out / "contacts.json")
    contact_error, contact_count = 0.0, 0
    for part, values in (
        ("full", source["fields"]),
        ("choice", source["choice"]),
        ("within", source["fields"] - source["choice"]),
    ):
        values = values.reshape(2, 24, 5).mean(0)
        for p in range(5):
            for q in range(5):
                direct = (
                    sum(
                        (values[u, p] - values[:, p].mean())
                        * (effects[u, q] - effects[:, q].mean())
                        for u in range(24)
                    )
                    / 24
                )
                contact_error = max(
                    contact_error, abs(direct - contacts[f"late/{part}/sum"][p][q])
                )
                contact_count += 1
    if max(error, head_error) > 1e-3 or contact_error > 1e-10:
        raise ArithmeticError("Numerical implementation not faithful at declared scale")
    result = {
        "PASS": True,
        "literal_four_corners": count,
        "literal_contacts": contact_count,
        "cold_error": float(error),
        "head_error": head_error,
        "contact_error": float(contact_error),
        "same_agent_alternate_not_blind": True,
        "new_model_calls": 0,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)


def main():
    """Expose finite stages; check never launches or repairs a science collector."""
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
        print("ALL_SEALS_PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
