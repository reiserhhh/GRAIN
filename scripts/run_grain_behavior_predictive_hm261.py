#!/usr/bin/env python3
"""One trait-name-free, public behavior-completion measurement experiment."""

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
from grain_research.construction import math_behavior_predictive as M
from grain_research.construction import math_continuation_routes as C
from grain_research.construction import math_late_condition as L
from scripts import run_grain_conditional_likelihood_hm152 as N
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_behavior_predictive_hm261.json"
DEFAULT = ROOT / "results/grain_math_response_v1/behavior_predictive_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_BEHAVIOR_PREDICTIVE_PLAN_HM261.md",
    "docs/GRAIN_BEHAVIOR_PREDICTIVE_MEASUREMENT_THEORY_V1.md",
    "grain_research/construction/math_behavior_predictive.py",
    "scripts/run_grain_behavior_predictive_hm261.py",
    "scripts/score_grain_behavior_predictive_cold.py",
    "tests/test_grain_behavior_predictive.py",
)


def prepare(out):
    """Freeze complete neutral observation units before any new native outcomes."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement/restart of old experiment")
    cfg = R.read(ROOT / CONFIG)
    if (cfg["authors"], cfg["directional_uses"], cfg["items"], cfg["batch_size"]) != (
        24,
        48,
        50,
        8,
    ):
        raise ValueError("Predeclared scope changed")
    parent = ROOT / cfg["parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    old = R.read(parent / "cases.json")
    if len(old) != 48 or [r["author"] for r in old] != list(range(24)) * 2:
        raise ValueError("Same24 authors/two directions required")
    if [r["unit_id"] for r in old] != list(range(48)) or any(
        len(r["prefixes"]) != 2 for r in old
    ):
        raise ValueError("Every addressed own/foreign observation unit is required")
    bank = M.item_bank(R.read(ROOT / cfg["items_config"]))
    tok = AutoTokenizer.from_pretrained(
        J.MODEL, local_files_only=True, trust_remote_code=False
    )
    empty, _ = L.observation_parts(tok, [])
    end = tok.convert_tokens_to_ids("<|im_end|>")
    for item in bank:
        item["target_ids"] = tok.encode(
            item["completion"], add_special_tokens=False
        ) + [end]
    cases = []
    for row in old:
        prefixes, suffix = [], None
        for prior in row["prefixes"]:
            if prior[: len(empty)] != empty or len(prior) <= len(empty):
                raise ValueError("Unknown historical context extraction boundary")
            context = prior[len(empty) :]
            prefix, tail = M.observation_parts(tok, context)
            if prefix[-len(context) :] != context or (
                suffix is not None and tail != suffix
            ):
                raise ValueError("Neutral task changed raw context IDs or query")
            if (
                max(len(prefix) + len(tail) + len(i["target_ids"]) - 1 for i in bank)
                > cfg["max_input_tokens"]
            ):
                raise ValueError("Full candidate/context exceeds cap; no truncation")
            prefixes.append(prefix)
            suffix = tail
        cases.append(
            {
                "unit_id": row["unit_id"],
                "author": row["author"],
                "context_window": row["context_window"],
                "prefixes": prefixes,
                "suffix": suffix,
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    (out / "basis").mkdir()
    inputs = dict(pm["inputs"])
    inputs.update(
        {
            str((parent / n).relative_to(ROOT)): R.digest(parent / n)
            for n in (
                "manifest.json",
                "collected_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
                "cases.json",
                "source_observations.npz",
            )
        }
    )
    inputs[cfg["items_config"]] = R.digest(ROOT / cfg["items_config"])
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {**pm["code"], **{p: R.digest(ROOT / p) for p in SCIENCE}},
            "inputs": inputs,
            "model_weights_sha256": pm["model_weights_sha256"],
            "no_raw_body_or_application_questionnaire_reads": True,
            "original_scope_complete": False,
        },
    )
    R.write(out / "cases.json", cases)
    R.write(out / "bank.json", bank)
    R.archive(
        out / "source_observations.npz",
        **S.selected(
            parent / "source_observations.npz", ("fields", "choice", "coordinates")
        ),
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("cases.json", "bank.json", "source_observations.npz"),
    )
    H.status(out, {"state": "PREPARED", "done_batches": 0, "model_seconds": 0})
    print("PREPARED_BEHAVIOR_FORWARD", flush=True)


def collect(out):
    """Teacher-force every full public candidate once under one nonrenewable cap."""
    import torch

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No automatic resume or retry of terminal/partial science")
    cfg, bank, cases = (
        manifest["config"],
        R.read(out / "bank.json"),
        R.read(out / "cases.json"),
    )
    seconds, done, prefixes, pending = 0.0, 0, 0, None
    files, guards = [], []
    errors = {"head": 0.0, "native": 0.0, "prefix_mutation": 0.0}
    lock = out / "ACTIVE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))

    def mark(state, error=None):
        """Expose current actual operation and checkpoint without rewriting results."""
        H.status(
            out,
            {
                "state": state,
                "pid": os.getpid(),
                "done_batches": done,
                "prefixes": prefixes,
                "model_seconds": seconds,
                "pending": pending,
                "error": error,
            },
        )

    def timed(fn):
        """Charge failed calls too; a scientific cap is never silently renewed."""
        nonlocal seconds
        if seconds >= cfg["max_model_seconds"]:
            raise TimeoutError("One model-time cap exhausted; preserve partial work")
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
        for case in cases:
            for mode, prefix in enumerate(case["prefixes"]):
                pending = {
                    "unit": case["unit_id"],
                    "mode": mode,
                    "operation": "neutral_encode",
                }
                mark("RUNNING")
                cache = timed(partial(L.encode_context, observer, prefix))
                before = L.cache_arrays(cache)
                prefixes += 1
                for start in range(0, 50, cfg["batch_size"]):
                    items = bank[start : start + cfg["batch_size"]]
                    targets = [i["target_ids"] for i in items]
                    pending = {
                        "unit": case["unit_id"],
                        "mode": mode,
                        "start": start,
                        "operation": "complete_behavior",
                    }
                    mark("RUNNING")
                    rows = timed(
                        partial(
                            M.observe_candidates,
                            observer,
                            cache,
                            case["suffix"],
                            targets,
                        )
                    )
                    for item, row in zip(items, rows, strict=True):
                        basis_path = out / f"basis/{item['item']:02d}.npz"
                        basis = {k: row[k] for k in ("covectors", "bias")}
                        if case["unit_id"] == 0 and mode == 0:
                            R.archive(basis_path, **basis)
                            files.append(str(basis_path.relative_to(out)))
                        else:
                            known = S.selected(basis_path, ("covectors", "bias"))
                            if any(
                                not np.array_equal(v, known[k])
                                for k, v in basis.items()
                            ):
                                raise ValueError("Behavioral output covectors changed")
                        replay = (
                            np.sum(
                                row["states"].astype(float)
                                * row["covectors"].astype(float),
                                -1,
                            )
                            + row["bias"]
                        )
                        errors["head"] = max(
                            errors["head"], float(np.max(abs(replay - row["chosen"])))
                        )
                    if case["unit_id"] == 0 and start == 0:
                        native = timed(
                            partial(
                                N.observe,
                                observer,
                                [(prefix + case["suffix"], t) for t in targets[:2]],
                            )
                        )
                        maximum = max(len(t) for t in targets[:2])
                        cached = np.full((2, maximum), np.nan)
                        full = np.full_like(cached, np.nan)
                        for i, target in enumerate(targets[:2]):
                            cached[i, : len(target)] = (
                                rows[i]["chosen"] - rows[i]["partition"]
                            )
                            full[i, : len(target)] = native["chosen"][
                                i, : len(target)
                            ].astype(float) - native["partition"][
                                i, : len(target)
                            ].astype(float)
                        err = float(np.nanmax(abs(cached - full)))
                        errors["native"] = max(errors["native"], err)
                        guard_name = f"native_guard_{mode}.npz"
                        R.archive(out / guard_name, cached=cached, full=full)
                        files.append(guard_name)
                        guards.append({"mode": mode, "sequences": 2, "error": err})
                        if err > cfg["native_tolerance"]:
                            raise ArithmeticError(
                                "Cache/native observation disagreement"
                            )
                    name = f"captures/{done:04d}.npz"
                    R.archive(
                        out / name,
                        **{
                            k: np.concatenate([r[k] for r in rows])
                            for k in ("states", "chosen", "partition")
                        },
                        offsets=np.cumsum([0] + [len(t) for t in targets]),
                        items=np.array([i["item"] for i in items]),
                        unit=case["unit_id"],
                        mode=mode,
                        start=start,
                    )
                    files.append(name)
                    done += 1
                    pending = None
                    mark("RUNNING")
                after = L.cache_arrays(cache)
                errors["prefix_mutation"] = max(
                    errors["prefix_mutation"],
                    max(float(np.max(abs(after[k] - v))) for k, v in before.items()),
                )
                if (
                    errors["prefix_mutation"]
                    or errors["head"] > cfg["native_tolerance"]
                ):
                    raise ArithmeticError(
                        "Observation integrity failed, not precision search"
                    )
                del cache, before, after
            print(
                json.dumps(
                    {"unit": case["unit_id"], "batches": done, "model_seconds": seconds}
                ),
                flush=True,
            )
        quality = {
            "batches": done,
            "sequences": 4800,
            "prefixes": prefixes,
            "model_seconds": seconds,
            "errors": errors,
            "guards": guards,
        }
        R.write(out / "collection_quality.json", quality)
        files.append("collection_quality.json")
        R.seal(out, "collected_seal.json", files)
        mark("COMPLETE")
    except Exception as exc:
        mark("PARTIAL_BUDGET" if isinstance(exc, TimeoutError) else "FAILED", repr(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


def read_sums(out):
    """Bind every ragged word account to its original author, context and item."""
    cfg = R.read(out / "manifest.json")["config"]
    bank = R.read(out / "bank.json")
    values = {k: np.full((48, 2, 50), np.nan) for k in ("chosen", "partition")}
    sequence, tokens = 0, 0
    for file in sorted((out / "captures").glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            u, mode, start = [int(z[k]) for k in ("unit", "mode", "start")]
            expected = list(range(start, min(start + cfg["batch_size"], 50)))
            if z["items"].tolist() != expected:
                raise ValueError("Ragged item addresses changed")
            offsets = z["offsets"]
            if (
                len(offsets) != len(expected) + 1
                or offsets[0] != 0
                or offsets[-1] != len(z["chosen"])
            ):
                raise ValueError("Incomplete token offsets")
            for i, item in enumerate(expected):
                a, b = map(int, offsets[i : i + 2])
                if b - a != len(bank[item]["target_ids"]) or not np.isnan(
                    values["chosen"][u, mode, item]
                ):
                    raise ValueError("Missing, copied or truncated candidate")
                for key, account in values.items():
                    account[u, mode, item] = float(z[key][a:b].astype(float).sum())
                sequence += 1
                tokens += b - a
    if sequence != 4800 or not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Partial observations are not a complete behavioral field")
    return values, tokens


def analyze(out):
    """Compile the frozen measuring function and all SOURCE correspondences once."""
    manifest = R.check(out, ("collected_seal.json",))
    cfg, bank = manifest["config"], R.read(out / "bank.json")
    sums, tokens = read_sums(out)
    packet = {
        "version": "HM261-v1",
        **{k: v.tolist() for k, v in sums.items()},
        "bank": bank,
        "lengths": [len(i["target_ids"]) for i in bank],
    }
    result = M.cold_score(packet)
    R.write(out / "cold_packet.json", packet)
    R.write(out / "cold_numeric.json", result)
    source = S.selected(
        out / "source_observations.npz", ("fields", "choice", "coordinates")
    )
    contacts, intervals, rows, identities = {}, {}, [], {}
    for scale, values in result["accounts"].items():
        b, u, q = [
            np.asarray(values[k])
            for k in ("balanced_key", "common_support", "equal_item_key")
        ]
        identities[scale] = float(
            np.max(abs(q - b - np.asarray(values["imbalance"]) * u))
        )
        for name, y in (
            ("balanced_key", b),
            ("common_support", u),
            ("equal_item_key", q),
        ):
            for part, x in (
                ("full", source["fields"]),
                ("choice", source["choice"]),
                ("within", source["fields"] - source["choice"]),
                ("GRAIN15", source["coordinates"]),
            ):
                xx, yy = x.reshape(2, 24, -1), y.reshape(2, 24, 5)
                for direction in ("pooled", 0, 1):
                    a, c = (
                        (xx.mean(0), yy.mean(0))
                        if direction == "pooled"
                        else (xx[direction], yy[direction])
                    )
                    contacts[f"{name}/{part}/{scale}/{direction}"] = C.contacts(
                        a, c
                    ).tolist()
            intervals[f"{name}/{scale}"] = C.diagonal_intervals(
                source["fields"].reshape(2, 24, 5).mean(0),
                y.reshape(2, 24, 5).mean(0),
                cfg["bootstrap_seed"],
                cfg["bootstrap_draws"],
            ).tolist()
            pooled = np.diag(contacts[f"{name}/full/{scale}/pooled"])
            for target, value in enumerate(pooled):
                rows.append(
                    {
                        "account": name,
                        "scale": scale,
                        "target": cfg["targets"][target],
                        "pooled": float(value),
                        "forward": contacts[f"{name}/full/{scale}/0"][target][target],
                        "reverse": contacts[f"{name}/full/{scale}/1"][target][target],
                    }
                )
    item_delta = np.asarray(result["item_delta"])
    item_contacts = C.contacts(
        source["fields"].reshape(2, 24, 5).mean(0),
        item_delta.reshape(2, 24, 50).mean(0),
    )
    R.write(out / "contacts.json", contacts)
    R.write(
        out / "intervals.json",
        {"marginal_conditional_exploratory": True, "values": intervals},
    )
    R.table(out / "source_contacts.csv", rows)
    R.table(
        out / "item_contacts.csv",
        [
            {
                "item": j,
                "key_target": bank[j]["target"],
                "pole": bank[j]["pole"],
                **{t: float(item_contacts[i, j]) for i, t in enumerate(cfg["targets"])},
            }
            for j in range(50)
        ],
    )
    summary = {
        "authors": 24,
        "directional_uses": 48,
        "items": 50,
        "scored_tokens": tokens,
        "primary_balanced_key_contacts": np.diag(
            contacts["balanced_key/full/sum/pooled"]
        ).tolist(),
        "primary_marginal_intervals": intervals["balanced_key/sum"],
        "common_support_contacts": np.diag(
            contacts["common_support/full/sum/pooled"]
        ).tolist(),
        "equal_item_contacts": np.diag(
            contacts["equal_item_key/full/sum/pooled"]
        ).tolist(),
        "identity_errors": identities,
        "candidate_log_mass_range": [
            float(np.min(result["candidate_log_mass"])),
            float(np.max(result["candidate_log_mass"])),
        ],
        "collection": R.read(out / "collection_quality.json"),
        "not_author_endorsement_probability_r_AUC_or_human_validity": True,
        "not_new_math_or_implicit_persona_Bayes_inverse": True,
        "original_scope_complete": False,
    }
    if max(identities.values()) > 1e-10:
        raise ArithmeticError("Complete signed key identity failed")
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "cold_packet.json",
            "cold_numeric.json",
            "contacts.json",
            "intervals.json",
            "source_contacts.csv",
            "item_contacts.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Alternative scalar word/key/SOURCE accounts; no decoder calls or refit."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    bank, packet = R.read(out / "bank.json"), R.read(out / "cold_packet.json")
    chosen, partition = np.asarray(packet["chosen"]), np.asarray(packet["partition"])
    reconstructed, tokens, maximum = np.empty_like(chosen), 0, 0.0
    for file in sorted((out / "captures").glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            u, mode = int(z["unit"]), int(z["mode"])
            for i, j in enumerate(z["items"]):
                a, b = map(int, z["offsets"][i : i + 2])
                reconstructed[u, mode, j] = sum(
                    float(z["chosen"][t]) - float(z["partition"][t])
                    for t in range(a, b)
                )
                basis = S.selected(out / f"basis/{j:02d}.npz", ("covectors", "bias"))
                replay = (
                    np.einsum(
                        "td,td->t",
                        z["states"][a:b].astype(float),
                        basis["covectors"].astype(float),
                    )
                    + basis["bias"]
                )
                maximum = max(maximum, float(np.max(abs(replay - z["chosen"][a:b]))))
                tokens += b - a
    error = float(np.max(abs(reconstructed - (chosen - partition))))
    direct = np.empty((48, 5))
    for unit in range(48):
        for target in range(5):
            poles = [
                [r["item"] for r in bank if r["target"] == target and r["pole"] == p]
                for p in (1, -1)
            ]
            means = [
                sum(
                    reconstructed[unit, 0, j] - reconstructed[unit, 1, j]
                    for j in indices
                )
                / len(indices)
                for indices in poles
            ]
            direct[unit, target] = (means[0] - means[1]) / 2
    result = R.read(out / "cold_numeric.json")
    error = max(
        error, float(np.max(abs(direct - result["accounts"]["sum"]["balanced_key"])))
    )
    fields = (
        S.selected(out / "source_observations.npz", ("fields",))["fields"]
        .reshape(2, 24, 5)
        .mean(0)
    )
    yy = direct.reshape(2, 24, 5).mean(0)
    stored = np.asarray(R.read(out / "contacts.json")["balanced_key/full/sum/pooled"])
    contact_error = 0.0
    for p in range(5):
        for q in range(5):
            f, y = fields[:, p], yy[:, q]
            mf, my = sum(map(float, f)) / 24, sum(map(float, y)) / 24
            literal = (
                sum((float(f[i]) - mf) * (float(y[i]) - my) for i in range(24)) / 24
            )
            contact_error = max(contact_error, abs(literal - stored[p, q]))
    destination = out / "standalone_cold.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_behavior_predictive_cold.py"),
            "--input",
            str(out / "cold_packet.json"),
            "--output",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cold = R.read(destination)
    if cold != result or error > 1e-10 or contact_error > 1e-10 or maximum > 1e-3:
        raise ArithmeticError(
            "Finite literal/cold contract failed; no precision rescue"
        )
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "sequences": 4800,
            "tokens": tokens,
            "primary_source_contacts": 25,
            "word_key_error": error,
            "source_contact_error": contact_error,
            "head_error": maximum,
            "standalone_cold_exact": True,
            "same_agent_not_blind": True,
            "new_model_calls": 0,
        },
    )
    R.seal(out, "verification_seal.json", ("standalone_cold.json", "verification.json"))
    print("VERIFIED_BEHAVIOR_FORWARD", flush=True)


def main():
    """Expose bounded stages; check never launches or resumes an experiment."""
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
