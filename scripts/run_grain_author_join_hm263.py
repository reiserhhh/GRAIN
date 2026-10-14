#!/usr/bin/env python3
"""One bounded, outcome-unselected author-join behavior mechanism experiment."""

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
from grain_research.construction import math_author_join as M
from grain_research.construction import math_behavior_predictive as B
from grain_research.construction import math_late_condition as L
from grain_research.construction import math_observed_family as F
from grain_research.construction import math_trait_anchor as A
from scripts import run_grain_conditional_likelihood_hm152 as N
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_author_join_hm263.json"
DEFAULT = ROOT / "results/grain_math_response_v1/author_join_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_AUTHOR_JOIN_PLAN_HM263.md",
    "grain_research/construction/math_author_join.py",
    "scripts/run_grain_author_join_hm263.py",
    "tests/test_grain_author_join.py",
)


def prepare(out):
    """Bind public event pairs, match without outcomes, then freeze joint inputs."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite/restart of an experiment")
    cfg = R.read(ROOT / CONFIG)
    parent, address = ROOT / cfg["parent"], ROOT / cfg["address_parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    am = R.check(
        address, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    acfg = am["config"]
    material = ROOT / acfg["material"]
    mm = R.read(material / "material_manifest.json")
    if mm["meps_used"] or mm["authors"] != 624:
        raise ValueError("Not the authorized public624 material")
    for name in ("private_material.json", "private_arrays.npz"):
        if R.digest(material / name) != mm["files"][name]:
            raise ValueError("Changed public material identity")
    panel = S.selected(address / "source_observations.npz", ("panel_indices",))[
        "panel_indices"
    ]
    if len(panel) != 24:
        raise ValueError("Original24 authors required")
    filter_ = (
        "[$idx[] as $i | .bundles[$i] | {author_id,source_group,views:"
        "[.views[0:2][] | [.[] | {author_id,body,comment_id,subreddit,timestamp}]]}]"
    )
    records = json.loads(
        subprocess.check_output(
            [
                "jq",
                "--argjson",
                "idx",
                json.dumps(panel.tolist()),
                filter_,
                str(material / "private_material.json"),
            ],
            text=True,
        )
    )
    identities = json.loads(
        subprocess.check_output(
            [
                "jq",
                "[.[]|{unit_id,author,context_window,own:{comment_id:.own.comment_id}}]",
                str(address / "private_bindings.json"),
            ],
            text=True,
        )
    )
    if any(
        r["source_group"] != "big5_only" or any(len(v) != 12 for v in r["views"])
        for r in records
    ):
        raise ValueError("Changed public panel/view family")
    tok = AutoTokenizer.from_pretrained(
        J.MODEL, local_files_only=True, trust_remote_code=False
    )
    tokenized = [
        [
            [tok.encode(e["body"], add_special_tokens=False) for e in v]
            for v in r["views"]
        ]
        for r in records
    ]
    header, tail = B.observation_parts(tok, [0])
    header = header[:-1]
    old = R.read(parent / "cases.json")
    slots = np.full((24, 2), -1, int)
    for row, identity in zip(old, identities, strict=True):
        u, w = row["author"], row["context_window"]
        if (row["unit_id"], u, w) != (
            identity["unit_id"],
            identity["author"],
            identity["context_window"],
        ):
            raise ValueError("Historical event addresses changed")
        candidates = [
            i
            for i, e in enumerate(records[u]["views"][w])
            if e["comment_id"] == identity["own"]["comment_id"]
        ]
        if len(candidates) != 1 or row["prefixes"][0][: len(header)] != header:
            raise ValueError("Unbound historical own comment")
        slots[u, w] = candidates[0]
        if (
            row["prefixes"][0][len(header) :] != tokenized[u][w][candidates[0]]
            or row["suffix"] != tail
        ):
            raise ValueError("Neutral observation or raw comment tokens changed")
    with np.load(material / "private_arrays.npz", allow_pickle=False) as z:
        author_ids = z["author_ids"][panel]
        vectors = z["comments"][panel, :24].astype(float).reshape(24, 2, 12, 1024)
    if not np.array_equal(author_ids, [r["author_id"] for r in records]):
        raise ValueError("Public numeric/text author binding changed")
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    lengths = np.array([[list(map(len, v)) for v in r] for r in tokenized])
    bundle = R.read(ROOT / acfg["anonymous_bundle"])
    conditions = np.array(
        [
            [
                [int(e["subreddit"] in bundle["context_rule"]["top"]) for e in v]
                for v in r["views"]
            ]
            for r in records
        ]
    )
    out.mkdir(parents=True, exist_ok=False)
    try:
        donors, foreign_slots, costs = M.joint_matching(
            vectors,
            lengths,
            conditions,
            slots,
            cfg["matching_absolute_tokens"],
            cfg["matching_relative_tokens"],
        )
    except ValueError as exc:
        R.write(
            out / "design_failure.json",
            {
                "state": "STOP_INPUT_DESIGN",
                "reason": str(exc),
                "authors_required": 24,
                "new_model_forwards": 0,
                "no_threshold_relaxation": True,
            },
        )
        raise
    bank = R.read(parent / "bank.json")
    if len(bank) != 50 or any(
        tok.encode(r["completion"], add_special_tokens=False)
        + [tok.convert_tokens_to_ids("<|im_end|>")]
        != r["target_ids"]
        for r in bank
    ):
        raise ValueError("Changed complete behavior bank")
    separator = tok.encode("\n\n", add_special_tokens=False)
    cases, bindings = [], []
    delta_vectors, delta_coordinates, dphi, dnull = [], [], [], []
    law_path = ROOT / acfg["meaning_parent"] / "source_law.json"
    null_path = ROOT / acfg["meaning_parent"] / "permuted_law.json"
    law, null = R.read(law_path), R.read(null_path)
    A.verify_law(law)
    A.verify_law(null)
    coordinates = vectors @ np.asarray(bundle["directions"], float)
    cells = 4 * conditions + F.states(
        coordinates.reshape(-1, 15), law["coordinate_cutoffs"]
    ).reshape(24, 2, 12)
    fields = np.asarray(law["probabilities"]) - np.asarray(law["prior"])
    null_fields = np.asarray(null["probabilities"]) - np.asarray(null["prior"])
    for u, v in enumerate(donors):
        segments, addresses, ds, dn, dv, dg = [], [], [], [], [], []
        for w in range(2):
            own_slot, foreign_slot = int(slots[u, w]), int(foreign_slots[u, w])
            own, foreign = (
                records[u]["views"][w][own_slot],
                records[v]["views"][w][foreign_slot],
            )
            if (
                own["author_id"] == foreign["author_id"]
                or own["comment_id"] == foreign["comment_id"]
            ):
                raise ValueError("Self donor or repeated event")
            segments.append([tokenized[u][w][own_slot], tokenized[v][w][foreign_slot]])
            addresses.append(
                {
                    "window": w,
                    "own_id": own["comment_id"],
                    "foreign_id": foreign["comment_id"],
                    "own_slot": own_slot,
                    "foreign_slot": foreign_slot,
                    "condition": int(conditions[u, w, own_slot]),
                    "own_tokens": int(lengths[u, w, own_slot]),
                    "foreign_tokens": int(lengths[v, w, foreign_slot]),
                    "exact_venue_match": own["subreddit"] == foreign["subreddit"],
                }
            )
            ds.append(fields[cells[u, w, own_slot]] - fields[cells[v, w, foreign_slot]])
            dn.append(
                null_fields[cells[u, w, own_slot]]
                - null_fields[cells[v, w, foreign_slot]]
            )
            dv.append(vectors[u, w, own_slot] - vectors[v, w, foreign_slot])
            dg.append(coordinates[u, w, own_slot] - coordinates[v, w, foreign_slot])
        prefixes = []
        for order in range(2):
            group = []
            for a, b in cfg["corners"]:
                pair = [segments[0][a], segments[1][b]]
                if order:
                    pair = pair[::-1]
                prefix, suffix = B.observation_parts(tok, pair[0] + separator + pair[1])
                if (
                    suffix != tail
                    or max(
                        len(prefix) + len(tail) + len(i["target_ids"]) - 1 for i in bank
                    )
                    > cfg["max_input_tokens"]
                ):
                    raise ValueError(
                        "Changed suffix or full joined context exceeds fixed cap"
                    )
                group.append(prefix)
            prefixes.append(group)
        cases.append({"unit_id": u, "prefixes": prefixes, "suffix": tail})
        bindings.append(
            {
                "unit_id": u,
                "own_author": u,
                "donor_author": int(v),
                "joint_match_cost": float(costs[u]),
                "events": addresses,
            }
        )
        dphi.append(ds)
        dnull.append(dn)
        delta_vectors.append(dv)
        delta_coordinates.append(dg)
    (out / "captures").mkdir()
    inputs = {**pm["inputs"], **am["inputs"]}
    for path in (
        parent / "cases.json",
        parent / "bank.json",
        address / "private_bindings.json",
        address / "source_observations.npz",
        law_path,
        null_path,
    ):
        inputs[str(path.relative_to(ROOT))] = R.digest(path)
    for i in range(50):
        path = (
            ROOT
            / cfg["parent"].split("/analysis_recovery_r1")[0]
            / f"basis/{i:02d}.npz"
        )
        inputs[str(path.relative_to(ROOT))] = R.digest(path)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {
                **pm["code"],
                **am["code"],
                **{p: R.digest(ROOT / p) for p in SCIENCE},
            },
            "inputs": inputs,
            "model_weights_sha256": pm["model_weights_sha256"],
            "cohort": "Historical24 public authors;new joined inputs,not new people",
            "questionnaire_values_read": False,
            "original_scope_complete": False,
        },
    )
    R.write(out / "cases.json", cases)
    R.write(out / "bank.json", bank)
    R.write(out / "bindings.json", bindings)
    R.archive(
        out / "source_actions.npz",
        actual=np.asarray(dphi),
        historical_null=np.asarray(dnull),
        full_e5=np.asarray(delta_vectors),
        grain15=np.asarray(delta_coordinates),
        donors=donors,
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "dependent_donor_edges": 24,
            "joined_prefixes": 192,
            "candidates": 9600,
            "binary_condition_match": 48,
            "exact_venue_matches": sum(
                e["exact_venue_match"] for r in bindings for e in r["events"]
            ),
            "maximum_segment_length_difference": max(
                abs(e["own_tokens"] - e["foreign_tokens"])
                for r in bindings
                for e in r["events"]
            ),
            "no_raw_private_or_application_labels": True,
            "matching_uses_behavior_scores": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "cases.json",
            "bank.json",
            "bindings.json",
            "source_actions.npz",
            "input_quality.json",
        ),
    )
    H.status(out, {"state": "PREPARED", "model_seconds": 0, "done_batches": 0})
    print(json.dumps(R.read(out / "input_quality.json")), flush=True)


def collect(out):
    """Observe all new joints once; retain partial work without automatic retry."""
    import torch

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No resume/retry of an already attempted finite collection")
    cfg, cases, bank = (
        manifest["config"],
        R.read(out / "cases.json"),
        R.read(out / "bank.json"),
    )
    parent = ROOT / cfg["parent"].split("/analysis_recovery_r1")[0]
    basis = [
        S.selected(parent / f"basis/{i:02d}.npz", ("covectors", "bias"))
        for i in range(50)
    ]
    done, seconds, pending = 0, 0.0, None
    errors, files = {"native": 0.0, "head": 0.0, "prefix_mutation": 0.0}, []
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))

    def mark(state, error=None):
        """Save current address and cap use independently of scientific arrays."""
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
        """Charge even failed calls to one fixed scientific time cap."""
        nonlocal seconds
        if seconds >= cfg["max_model_seconds"]:
            raise TimeoutError("Fixed900-second cap exhausted")
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
            for order in range(2):
                for corner, prefix in enumerate(case["prefixes"][order]):
                    pending = {
                        "unit": case["unit_id"],
                        "order": order,
                        "corner": corner,
                        "operation": "encode",
                    }
                    mark("RUNNING")
                    cache = timed(partial(L.encode_context, observer, prefix))
                    before = L.cache_arrays(cache)
                    for start in range(0, 50, cfg["batch_size"]):
                        items = bank[start : start + cfg["batch_size"]]
                        targets = [i["target_ids"] for i in items]
                        pending = {
                            "unit": case["unit_id"],
                            "order": order,
                            "corner": corner,
                            "start": start,
                            "operation": "complete_behavior",
                        }
                        mark("RUNNING")
                        rows = timed(
                            partial(
                                B.observe_candidates,
                                observer,
                                cache,
                                case["suffix"],
                                targets,
                            )
                        )
                        for item, row in zip(items, rows, strict=True):
                            old_basis = basis[item["item"]]
                            if any(
                                not np.array_equal(row[k], old_basis[k])
                                for k in ("covectors", "bias")
                            ):
                                raise ValueError("Behavioral target covectors changed")
                            replay = (
                                np.sum(
                                    row["states"].astype(float)
                                    * old_basis["covectors"].astype(float),
                                    -1,
                                )
                                + old_basis["bias"]
                            )
                            errors["head"] = max(
                                errors["head"],
                                float(np.max(abs(replay - row["chosen"]))),
                            )
                        if case["unit_id"] == 0 and start == 0:
                            native = timed(
                                partial(
                                    N.observe,
                                    observer,
                                    [(prefix + case["suffix"], t) for t in targets[:2]],
                                )
                            )
                            diffs = []
                            for i, target in enumerate(targets[:2]):
                                diffs.extend(
                                    (
                                        rows[i]["chosen"]
                                        - rows[i]["partition"]
                                        - (
                                            native["chosen"][i, : len(target)]
                                            - native["partition"][i, : len(target)]
                                        ).astype(float)
                                    ).tolist()
                                )
                            errors["native"] = max(
                                errors["native"], float(np.max(abs(np.asarray(diffs))))
                            )
                            name = f"native_guard_{order}_{corner}.npz"
                            R.archive(out / name, differences=np.asarray(diffs))
                            files.append(name)
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
                            order=order,
                            corner=corner,
                            start=start,
                        )
                        files.append(name)
                        done += 1
                        pending = None
                        if max(errors.values()) > cfg["native_tolerance"]:
                            raise ArithmeticError(
                                "Measurement integrity failed;no precision rescue"
                            )
                        mark("RUNNING")
                    after = L.cache_arrays(cache)
                    errors["prefix_mutation"] = max(
                        errors["prefix_mutation"],
                        max(
                            float(np.max(abs(after[k] - v))) for k, v in before.items()
                        ),
                    )
                    if errors["prefix_mutation"]:
                        raise ValueError("Joint prefix mutated during observation")
                    del cache, before, after
            print(
                json.dumps(
                    {"unit": case["unit_id"], "batches": done, "model_seconds": seconds}
                ),
                flush=True,
            )
        R.write(
            out / "collection_quality.json",
            {
                "batches": done,
                "sequences": 9600,
                "prefixes": 192,
                "model_seconds": seconds,
                "errors": errors,
            },
        )
        files.append("collection_quality.json")
        R.seal(out, "collected_seal.json", files)
        mark("COMPLETE")
    except Exception as exc:
        mark("PARTIAL_BUDGET" if isinstance(exc, TimeoutError) else "FAILED", repr(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


def read_sums(out):
    """Enumerate only sealed registered captures, never AppleDouble glob files."""
    seals, bank = R.read(out / "collected_seal.json"), R.read(out / "bank.json")
    result = {k: np.full((24, 2, 4, 50), np.nan) for k in ("chosen", "partition")}
    captures = sorted(p for p in seals if p.startswith("captures/"))
    if len(captures) != 1344:
        raise ValueError("Incomplete registered joint observation")
    for path in captures:
        with np.load(out / path, allow_pickle=False) as z:
            u, order, corner, start = [
                int(z[k]) for k in ("unit", "order", "corner", "start")
            ]
            expected = list(range(start, min(start + 8, 50)))
            if (
                z["items"].tolist() != expected
                or len(z["offsets"]) != len(expected) + 1
            ):
                raise ValueError("Changed joint candidate addresses")
            if z["offsets"][0] != 0 or z["offsets"][-1] != len(z["chosen"]):
                raise ValueError("Partial token account")
            for i, item in enumerate(expected):
                a, b = map(int, z["offsets"][i : i + 2])
                if b - a != len(bank[item]["target_ids"]) or not np.isnan(
                    result["chosen"][u, order, corner, item]
                ):
                    raise ValueError("Copied or truncated complete candidate")
                for key, values in result.items():
                    values[u, order, corner, item] = z[key][a:b].astype(float).sum()
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("Missing joint observation cannot be imputed")
    return result


def analyze(out):
    """Retain all finite signed joins and source contacts; select no better order."""
    R.check(out, ("collected_seal.json",))
    bank = R.read(out / "bank.json")
    values = read_sums(out)
    lengths = np.array([len(i["target_ids"]) for i in bank])
    accounts = M.interaction_accounts(
        values["chosen"], values["partition"], lengths, bank
    )
    source = S.selected(out / "source_actions.npz", ("actual", "historical_null"))
    R.archive(out / "observed_sums.npz", **values)
    arrays, summary, contacts, rows = {}, {}, {}, []
    for scale, result in accounts.items():
        arrays.update({f"{scale}_{k}": v for k, v in result.items()})
        summary[scale] = {}
        for key in ("balanced_key", "common_support", "equal_item_key"):
            y = result[key]
            summary[scale][key] = {
                "mean_by_order": y.mean(0).tolist(),
                "rms_by_order": np.sqrt(np.mean(y * y, 0)).tolist(),
                "positive_edges_by_order": np.sum(y > 0, 0).tolist(),
                "both_orders_positive_edges": np.sum(np.all(y > 0, 1), 0).tolist(),
                "both_orders_negative_edges": np.sum(np.all(y < 0, 1), 0).tolist(),
            }
            for law, actions in source.items():
                for order in range(2):
                    contacts[f"{scale}/{key}/{law}/{order}"] = M.source_contacts(
                        actions, y[:, order]
                    ).tolist()
            for u in range(24):
                for order in range(2):
                    for q in range(5):
                        rows.append(
                            {
                                "scale": scale,
                                "account": key,
                                "unit": u,
                                "order": order,
                                "target": B.TARGETS[q],
                                "interaction": float(y[u, order, q]),
                            }
                        )
        summary[scale]["balanced_order_even_rms"] = np.sqrt(
            np.mean(result["order_even"] ** 2, 0)
        ).tolist()
        summary[scale]["balanced_order_odd_rms"] = np.sqrt(
            np.mean(result["order_odd"] ** 2, 0)
        ).tolist()
        summary[scale]["chosen_normalizer_identity_error"] = float(
            np.max(
                abs(
                    result["item_interaction"]
                    - result["chosen_interaction"]
                    + result["normalizer_interaction"]
                )
            )
        )
        summary[scale]["key_common_identity_error"] = float(
            np.max(
                abs(
                    result["equal_item_key"]
                    - result["balanced_key"]
                    - result["imbalance"] * result["common_support"]
                )
            )
        )
    R.archive(out / "interaction_accounts.npz", **arrays)
    R.write(out / "contacts.json", contacts)
    R.table(out / "interaction_by_author.csv", rows)
    R.write(
        out / "summary.json",
        {
            "accounts": summary,
            "quality": R.read(out / "collection_quality.json"),
            "input_quality": R.read(out / "input_quality.json"),
            "dependent_edges_not_independent_humans": True,
            "not_accuracy_or_psychological_validity": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "observed_sums.npz",
            "interaction_accounts.npz",
            "contacts.json",
            "interaction_by_author.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary["sum"], allow_nan=False), flush=True)


def verify(out):
    """Use alternative four-corner arithmetic and direct SOURCE scalar products."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    sums = S.selected(out / "observed_sums.npz", ("chosen", "partition"))
    saved = S.selected(
        out / "interaction_accounts.npz",
        (
            "sum_item_interaction",
            "sum_balanced_key",
            "sum_common_support",
            "sum_equal_item_key",
        ),
    )
    ell = sums["chosen"] - sums["partition"]
    direct = (ell[:, :, 0] - ell[:, :, 1]) - (ell[:, :, 2] - ell[:, :, 3])
    maximum = float(np.max(abs(direct - saved["sum_item_interaction"])))
    bank = R.read(out / "bank.json")
    for u in range(24):
        for order in range(2):
            for q in range(5):
                high = [r["item"] for r in bank if r["target"] == q and r["pole"] == 1]
                low = [r["item"] for r in bank if r["target"] == q and r["pole"] == -1]
                h, l = (
                    sum(direct[u, order, i] for i in high) / len(high),
                    sum(direct[u, order, i] for i in low) / len(low),
                )
                maximum = max(
                    maximum,
                    abs((h - l) / 2 - saved["sum_balanced_key"][u, order, q]),
                    abs((h + l) / 2 - saved["sum_common_support"][u, order, q]),
                )
    source, contacts = (
        S.selected(out / "source_actions.npz", ("actual", "historical_null")),
        R.read(out / "contacts.json"),
    )
    contact_max = 0.0
    for law, x in source.items():
        for order in range(2):
            known = np.asarray(contacts[f"sum/balanced_key/{law}/{order}"])
            for p in range(5):
                for r in range(5):
                    for q in range(5):
                        value = (
                            sum(
                                float(x[u, 0, p])
                                * float(x[u, 1, r])
                                * float(saved["sum_balanced_key"][u, order, q])
                                for u in range(24)
                            )
                            / 24
                        )
                        contact_max = max(contact_max, abs(value - known[p, r, q]))
    if maximum > 1e-9 or contact_max > 1e-12:
        raise ArithmeticError("Literal numerical audit failed")
    R.write(
        out / "verification.json",
        {
            "mixed_account_entries": 2400,
            "balanced_common_entries": 480,
            "source_contact_entries": 500,
            "alternative_mixed_key_error": maximum,
            "alternative_source_contact_error": contact_max,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    H.status(
        out,
        {
            "state": "VERIFIED_COMPLETE_NOJOB",
            "model_seconds": R.read(out / "collection_quality.json")["model_seconds"],
            "done_batches": 1344,
        },
    )
    print(json.dumps(R.read(out / "verification.json")), flush=True)


def main():
    """Run one explicitly selected stage; never automatically restart science."""
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
        print("HM263_SEALS_PASS", flush=True)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
