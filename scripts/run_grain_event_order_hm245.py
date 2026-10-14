#!/usr/bin/env python3
"""Observe one mean-preserving order operation, with no answer or parameter fit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_companion_observation as A
from grain_research.construction import math_event_order as M
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_companion_observation_hm243 as H
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_event_order_hm245.json"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVENT_ORDER_PLAN_HM245.md",
    "grain_research/construction/math_event_order.py",
    "scripts/run_grain_event_order_hm245.py",
    "tests/test_grain_event_order.py",
    "suica_core/path_signature.py",
    *H.SCIENCE,
)
DEFAULT = ROOT / "results/grain_math_response_v1/event_order_action_r1"


def read_batch(parent, number, address):
    """Read one committed native batch after checking its identity and receipt."""
    path = parent / f"01/b{number:03d}.npz"
    receipt = R.read(path.with_suffix(".json"))
    if receipt["address"] != address or R.digest(path) != receipt["sha256"]:
        raise ValueError("Native batch address or receipt changed")
    return H.arrays(path), receipt


def render(texts, task, tok, limit):
    """Reuse the exact old template and compute its causal body boundary."""
    ids, user, _ = D.tokenize_prompt(texts, task, tok, max_tokens=limit)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": D.SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoding = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    boundary = prompt.rindex("\nEvidence ends here.")
    pos = max(
        i for i, (a, b) in enumerate(encoding["offset_mapping"]) if boundary >= b > a
    )
    if ids != encoding["input_ids"]:
        raise ValueError("Truncated or differently tokenized evidence")
    return ids, pos


def prepare(out):
    """Register full finite support and exact baseline replay before observations."""
    from transformers import AutoTokenizer

    cfg = R.read(ROOT / CONFIG)
    parent, mp = ROOT / cfg["parent"], ROOT / cfg["material_parent"]
    old = R.check(parent)
    status = R.read(parent / "status.json")
    if (
        status["state"] != "BUDGET_STOP"
        or status["completed"] != 99
        or status["pending"] is not None
    ):
        raise ValueError("Require the preserved terminal HM243, not a resumed queue")
    material = mp / "private_material.json"
    mm = R.read(mp / "manifest.json")
    if mm["arm"] != "cue_mask" or R.digest(material) != mm["material_sha256"]:
        raise ValueError("Historical public cue-mask source changed")
    lookup = {
        (r["author_id"], r["view"]): r
        for r in R.read(material)
        if r["source_group"] == "big5_only"
    }
    z = R.prepared(parent)
    basis_path = ROOT / cfg["basis"]
    basis = np.asarray(R.read(basis_path)["directions"], float)
    if basis.shape != (1024, 15):
        raise ValueError("Changed frozen GRAIN15 projection")
    p = M.validate_order(cfg["background_permutation"])
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    units_old, events_old = (
        R.read(parent / "units.json"),
        R.read(parent / "event_contracts.json"),
    )
    files = [
        parent / f
        for f in (
            "manifest.json",
            "prepared_seal.json",
            "units.json",
            "event_contracts.json",
            "observations.npz",
            "01/head.npz",
        )
    ]
    files += [material, mp / "manifest.json", basis_path]
    units, batches, contracts, lengths, token_differences = [], [], [], [], []
    mean_corners, keys, signatures, factors, backgrounds = [], [], [], [], []
    max_reduction_error, cold = 0.0, None
    # Only the two identity batches precede the 48 registered interventions.
    schedule = [(0, 0, arm, "identity") for arm in range(2)] + [
        (u, v, arm, "interior") for u, v, arm in np.ndindex(12, 2, 2)
    ]
    for u, v, arm, kind in schedule:
        uid = str(z["author_ids"][u])
        focal, other = lookup[(uid, v)], lookup[(uid, 1 - v)]
        removed = z[("selected", "control")[arm]][u, v]
        slots = [i for i in range(12) if i not in removed]
        old_texts = [focal["texts"][i] for i in slots] + other["texts"]
        new_order = np.arange(12) if kind == "identity" else p
        texts = [focal["texts"][i] for i in slots] + [
            other["texts"][i] for i in new_order
        ]
        old_batch = 8 * u + 4 * v + arm
        _, receipt = read_batch(parent, old_batch, [u, v, 0, arm])
        files += [parent / f"01/b{old_batch:03d}.{ext}" for ext in ("npz", "json")]
        if receipt["prefix_error"] > cfg["head_tolerance"]:
            raise ValueError("Baseline evidence prefix is not compatible")
        eids = [f"{uid}:{v}:{i}" for i in slots] + [
            f"{uid}:{1 - v}:{i}" for i in new_order
        ]
        old_eids = [f"{uid}:{v}:{i}" for i in slots] + [
            f"{uid}:{1 - v}:{i}" for i in range(12)
        ]
        if (
            events_old[old_batch]["event_ids"] != old_eids
            or len(set(eids)) != 20
            or sorted(eids) != sorted(old_eids)
        ):
            raise ValueError("Changed, omitted or repeated event evidence")
        if texts[:12] != old_texts[:12] or texts[-1] != old_texts[-1]:
            raise ValueError("Focal, background boundary or final event changed")
        batches.append(
            {
                "batch": len(batches),
                "old_batch": old_batch,
                "author": u,
                "view": v,
                "arm": arm,
                "kind": kind,
            }
        )
        contracts.append(
            {
                "batch": len(contracts),
                "kind": kind,
                "owner": uid,
                "old_event_ids": old_eids,
                "event_ids": eids,
            }
        )
        prefix = None
        for q, task in enumerate(cfg["targets"]):
            before_ids, before_pos = render(old_texts, task, tok, cfg["max_tokens"])
            previous = units_old[5 * old_batch + q]
            if (
                previous["input_ids"] != before_ids
                or previous["body_position"] != before_pos
            ):
                raise ValueError(
                    "Old model observation cannot be reconstructed exactly"
                )
            ids, pos = render(texts, task, tok, cfg["max_tokens"])
            if prefix is not None and prefix != ids[: pos + 1]:
                raise ValueError("Question affects shared causal evidence prefix")
            prefix = ids[: pos + 1]
            units.append(
                {
                    "author": u,
                    "view": v,
                    "companion": 0,
                    "arm": arm,
                    "kind": kind,
                    "target": q,
                    "input_ids": ids,
                    "body_position": pos,
                }
            )
            lengths.append(len(ids))
            token_differences.append(len(ids) - len(before_ids))
    for u, v in np.ndindex(12, 2):
        keys.append([u, v])
        h = z["panel_vectors"][u]
        bg = h[1 - v]
        max_reduction_error = max(
            max_reduction_error, float(np.max(abs(bg.mean(0) - bg[p].mean(0))))
        )
        # Canonical mean contract ignores order; both corners use the same sum.
        points = z["points"][u, v, 0]
        mean_corners.append(np.stack((points, points)))
        coords = h @ basis
        sa, ca = (
            [i for i in range(12) if i not in z[name][u, v]]
            for name in ("selected", "control")
        )
        account = M.ordered_interaction(coords[v, sa], coords[v, ca], coords[1 - v], p)
        signatures.append(account["corners"])
        factors.append(account["factorized_third"])
        backgrounds.append(account["background_second_difference"])
        if cold is None:
            cold = {
                "author": str(z["author_ids"][u]),
                "order": p.tolist(),
                "focal": [
                    {
                        "event_id": f"{z['author_ids'][u]}:{v}:{i}",
                        "owner": str(z["author_ids"][u]),
                        "vector": coords[v, i].tolist(),
                    }
                    for i in sa
                ],
                "background": [
                    {
                        "event_id": f"{z['author_ids'][u]}:{1 - v}:{i}",
                        "owner": str(z["author_ids"][u]),
                        "vector": coords[1 - v, i].tolist(),
                    }
                    for i in range(12)
                ],
            }
    if len(units) != cfg["new_enquiries"] or len(batches) != cfg["new_batches"]:
        raise ValueError("Registered schedule changed")
    model_files = {
        f.name: R.digest(f)
        for f in MODEL.iterdir()
        if f.is_file() and not f.name.startswith("._")
    }
    if model_files != old["model_files"]:
        raise ValueError("Model observer changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "01").mkdir()
    R.archive(
        out / "observations.npz",
        keys=np.array(keys),
        mean_corners=np.array(mean_corners),
        signature1=np.array([[[s[0] for s in bg] for bg in x] for x in signatures]),
        signature2=np.array([[[s[1] for s in bg] for bg in x] for x in signatures]),
        signature3=np.array([[[s[2] for s in bg] for bg in x] for x in signatures]),
        factorized_third=np.array(factors),
        background_second_difference=np.array(backgrounds),
        panel_coordinates=(z["panel_vectors"][:12] @ basis),
        selected=z["selected"][:12],
        control=z["control"][:12],
    )
    for name, value in (
        ("units.json", units),
        ("batch_contracts.json", batches),
        ("event_contracts.json", contracts),
        ("cold_input.json", cold),
    ):
        R.write(out / name, value)
    R.write(out / "cold_account.json", M.cold_account(cold))
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {f: R.digest(ROOT / f) for f in SCIENCE},
            "inputs": {str(f.relative_to(ROOT)): R.digest(f) for f in files},
            "model_files": model_files,
            "alphabet": old["alphabet"],
            "old_outcomes_known_not_blind": True,
            "public_material_only": True,
            "original_scope_complete": False,
        },
    )
    quality = {
        "authors": 12,
        "groups": 24,
        "old_own_batches_reused": 48,
        "new_identity_batches": 2,
        "new_order_batches": 48,
        "new_inquiries": len(units),
        "tokens_min": min(lengths),
        "tokens_max": max(lengths),
        "token_length_differences": sorted(set(token_differences)),
        "mean_roundoff_max": max_reduction_error,
        "canonical_mean_order_difference": 0,
        "fixed_background_prefix_events": 4,
        "fixed_final_background_events": 1,
        "cohort_is_old_completion_prefix": True,
        "original_scope_complete": False,
    }
    R.write(out / "quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "units.json",
            "batch_contracts.json",
            "event_contracts.json",
            "cold_input.json",
            "cold_account.json",
            "quality.json",
        ),
    )
    H.N.H.status_write(
        out,
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
        },
    )
    print(json.dumps(quality), flush=True)


def assert_terminal(out):
    """Do not analyze a live experiment or confuse a timeout with termination."""
    status = R.read(out / "status.json")
    if (
        status["state"] not in ("COLLECTED", "BUDGET_STOP", "FAILED")
        or (out / "collector.lock").exists()
    ):
        raise RuntimeError("Collector is not terminal")
    runtime = out / "runtime.json"
    if runtime.exists():
        try:
            os.kill(R.read(runtime)["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Recorded collector PID is still present")
    return status


def analyze(out):
    """Retain full planned coverage and all signed complete-group consequences."""
    m, status = R.check(out), assert_terminal(out)
    cfg, contracts = m["config"], R.read(out / "batch_contracts.json")
    parent, head = ROOT / cfg["parent"], H.arrays(out / "01/head.npz")
    old_head = H.arrays(parent / "01/head.npz")
    head_identity_error = max(float(np.max(abs(head[k] - old_head[k]))) for k in head)
    new = {}
    for b in range(status["completed"]):
        contract = contracts[b]
        key = [contract[k] for k in ("author", "view")] + [0, contract["arm"]]
        new[b] = read_batch(out, b, key)[0]
    if not all(i in new for i in (0, 1)):
        raise RuntimeError("Identity replay incomplete; no scientific promotion")
    identity_logit_error, identity_action_error = 0.0, 0.0
    identity = []
    for b in range(2):
        old, _ = read_batch(parent, contracts[b]["old_batch"], [0, 0, 0, b])
        identity_logit_error = max(
            identity_logit_error, float(np.max(abs(new[b]["logits"] - old["logits"])))
        )
        identity.append(
            A.read_logits(new[b]["logits"])["log_odds"]
            - A.read_logits(old["logits"])["log_odds"]
        )
    identity_action_error = float(np.max(abs(identity[0] - identity[1])))
    coverage, keys, logits, states, normalizers, rows = [], [], [], [], [], []
    for u, v in np.ndindex(24, 2):
        bs = [2 + 4 * u + 2 * v + arm for arm in range(2)] if u < 12 else []
        complete = bool(bs) and all(b in new for b in bs)
        coverage.append(
            {
                "author": u,
                "view": v,
                "registered_this_packet": u < 12,
                "complete": complete,
                "committed_arms": sum(b in new for b in bs),
                "old_missing_not_resumed": u >= 12,
            }
        )
        if not complete:
            continue
        old_arrays = [
            read_batch(parent, 8 * u + 4 * v + a, [u, v, 0, a])[0] for a in range(2)
        ]
        corners = [old_arrays, [new[b] for b in bs]]
        keys.append([u, v])
        for name, target in (
            ("logits", logits),
            ("states", states),
            ("normalizer", normalizers),
        ):
            target.append(np.array([[a[name] for a in bg] for bg in corners]))
    R.table(out / "coverage.csv", coverage)
    if not keys:
        raise RuntimeError("No complete registered order group; coverage retained")
    logits, states, normalizers = (
        np.array(logits),
        np.array(states),
        np.array(normalizers),
    )
    readout = A.read_logits(logits)
    acts = {k: readout[k][:, :, 0] - readout[k][:, :, 1] for k in ("score", "log_odds")}
    mixed = {k: a[:, 0] - a[:, 1] for k, a in acts.items()}
    accounts = A.head_accounts(
        states.transpose(0, 3, 1, 2, 4), head["weights"], head["bias"]
    )
    head_error = float(
        np.max(abs(accounts["logits"] - logits.transpose(0, 3, 1, 2, 4)))
    )
    numerical_error = float(np.max(abs(mixed["log_odds"] - accounts["total"])))
    observed_discrepancy = 2 * identity_action_error + 4 * head_error
    for i, (u, v) in enumerate(keys):
        for q, target in enumerate(cfg["targets"]):
            delta = float(mixed["log_odds"][i, q])
            rows.append(
                {
                    "author": u,
                    "view": v,
                    "target": target,
                    "original_action": float(acts["log_odds"][i, 0, q]),
                    "reordered_action": float(acts["log_odds"][i, 1, q]),
                    "mixed_log_odds": delta,
                    "mixed_score": float(mixed["score"][i, q]),
                    "reader": float(accounts["reader"][i, q]),
                    "representation": float(accounts["representation"][i, q]),
                    "mean_reader_action_minimax_error": abs(delta) / 2,
                    "larger_than_observed_instrument_discrepancy": abs(delta)
                    > observed_discrepancy,
                }
            )
    R.table(out / "event_order_actions.csv", rows)
    R.archive(
        out / "native_accounts.npz",
        keys=np.array(keys),
        logits=logits,
        states=states,
        normalizers=normalizers,
        mixed_log_odds=mixed["log_odds"],
        mixed_score=mixed["score"],
        original_reordered_actions=acts["log_odds"],
        reader=accounts["reader"],
        representation=accounts["representation"],
        centered_logit_mixed=accounts["mixed_centered_logits"],
    )
    z, target_rows = R.prepared(out), []
    sig_mixed = {
        k: z[k][:, 0, 0] - z[k][:, 0, 1] - z[k][:, 1, 0] + z[k][:, 1, 1]
        for k in ("signature1", "signature2", "signature3")
    }
    for q, target in enumerate(cfg["targets"]):
        x, y = acts["log_odds"][:, 0, q], acts["log_odds"][:, 1, q]
        r, h = accounts["reader"][:, q], accounts["representation"][:, q]
        target_rows.append(
            {
                "target": target,
                "groups": len(keys),
                "mixed_rms": A.rms(x - y),
                "action_sign_reversals": int(np.count_nonzero(x * y < 0)),
                "reader_rms": A.rms(r),
                "representation_rms": A.rms(h),
                "signed_account_product": float(np.mean(r * h)),
                "centered_logit_mixed_rms": A.rms(
                    accounts["mixed_centered_logits"][:, q]
                ),
                "mean_reader_minimax_error_rms": A.rms(x - y) / 2,
                "actions_above_observed_discrepancy": int(
                    np.count_nonzero(abs(x - y) > observed_discrepancy)
                ),
            }
        )
    summary = {
        "primary_state": status,
        "registered_authors": 12,
        "registered_groups": 24,
        "complete_groups": len(keys),
        "complete_authors": len({k[0] for k in keys}),
        "primary_complete": status["state"] == "COLLECTED" and len(keys) == 24,
        "unused_committed_half_groups": sum(
            r["committed_arms"] for r in coverage if not r["complete"]
        ),
        "targets": target_rows,
        "identity_logit_error": identity_logit_error,
        "identity_action_error": identity_action_error,
        "head_identity_error": head_identity_error,
        "head_error": head_error,
        "native_account_error": numerical_error,
        "instrument_check_pass": max(
            identity_logit_error, head_error, head_identity_error
        )
        <= cfg["head_tolerance"],
        "observed_discrepancy_not_population_CI": observed_discrepancy,
        "signature1_mixed_max": float(np.max(abs(sig_mixed["signature1"]))),
        "signature2_mixed_max": float(np.max(abs(sig_mixed["signature2"]))),
        "signature3_factorization_error": float(
            np.max(abs(sig_mixed["signature3"] - z["factorized_third"]))
        ),
        "signature3_mixed_rms": A.rms(sig_mixed["signature3"]),
        "order_sensitive_target_reader_not_fitted": True,
        "not_personality_accuracy_or_unique_attention_mechanism": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "coverage.csv",
            "event_order_actions.csv",
            "native_accounts.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Recompute action witnesses and Chen factors using alternate finite sums."""
    R.check(out, ("analyzed_seal.json",))
    summary, z, n = (
        R.read(out / "summary.json"),
        R.prepared(out),
        H.arrays(out / "native_accounts.npz"),
    )
    cfg, factor_error, second_error = R.read(out / "manifest.json")["config"], 0.0, 0.0
    p = np.array(cfg["background_permutation"])
    for i, (u, v) in enumerate(z["keys"]):
        h = z["panel_coordinates"][u]
        s, c = (
            [j for j in range(12) if j not in z[name][u, v]]
            for name in ("selected", "control")
        )
        a, b = h[v, s].sum(0) - h[v, c].sum(0), h[1 - v]
        db = M.second_level_direct(b) - M.second_level_direct(b[p])
        literal = np.array([a[j] * db[k] for j in range(15) for k in range(225)])
        factor_error = max(
            factor_error, float(np.max(abs(literal - z["factorized_third"][i])))
        )
        second_error = max(
            second_error,
            float(np.max(abs(M.signature_levels(b)[1] - M.second_level_direct(b)))),
        )
    odds = np.logaddexp.reduce(n["logits"][..., 5:], axis=-1) - np.logaddexp.reduce(
        n["logits"][..., :5], axis=-1
    )
    literal = odds[:, 0, 0] - odds[:, 0, 1] - odds[:, 1, 0] + odds[:, 1, 1]
    action_error = float(np.max(abs(literal - n["mixed_log_odds"])))
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__)),
            "cold",
            "--input",
            str(out / "cold_input.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cold = json.loads(process.stdout)
    expected = R.read(out / "cold_account.json")
    cold_error = max(
        float(np.max(abs(np.asarray(cold[k]) - np.asarray(expected[k]))))
        for k in ("mean_focal", "mean_background")
    )
    cold_error = max(
        cold_error,
        max(
            float(np.max(abs(np.array(a) - np.array(b))))
            for key in ("levels_original", "levels_reordered")
            for a, b in zip(cold[key], expected[key], strict=True)
        ),
    )
    checks = [
        factor_error,
        second_error,
        action_error,
        cold_error,
        summary["signature3_factorization_error"],
        summary["signature1_mixed_max"],
        summary["signature2_mixed_max"],
    ]
    result = {
        "PASS": max(checks) <= cfg["arithmetic_tolerance"]
        and summary["instrument_check_pass"],
        "literal_factor_max_error": factor_error,
        "literal_second_max_error": second_error,
        "alternate_action_max_error": action_error,
        "cold_process_error": cold_error,
        "same_agent_alternate_not_blind": True,
        "no_target_fitting": True,
        "not_psychological_validity": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)
    if not result["PASS"]:
        raise RuntimeError("Verification failed; preserve result, no scientific retry")


def main():
    """Run separate finite stages; collection is inherited without modification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check", "cold")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.stage == "cold":
        if args.input is None:
            raise ValueError("Explicit numeric input required")
        print(json.dumps(M.cold_account(R.read(args.input)), allow_nan=False))
    elif args.stage == "check":
        R.check(args.output, ("analyzed_seal.json", "verification_seal.json"))
        print("HM245 sealed stages unchanged")
    else:
        {
            "prepare": prepare,
            "collect": H.collect,
            "analyze": analyze,
            "verify": verify,
        }[args.stage](args.output)


if __name__ == "__main__":
    main()
