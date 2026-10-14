#!/usr/bin/env python3
"""One finite criterion-qualified direct/relay experiment; no score rescue."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_contract_routes as M
from grain_research.construction import math_target_contract as T
from grain_research.construction.math_prospective_evidence import bootstrap
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_contract_routes_hm222.json"
DEFAULT = ROOT / "results/grain_math_response_v1/contract_routes_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CONTRACT_ROUTES_PLAN_HM222.md",
    "grain_research/construction/math_contract_routes.py",
    "scripts/run_grain_contract_routes_hm222.py",
    "tests/test_grain_contract_routes.py",
    "tests/test_grain_query_relay_cut.py",
    "grain_research/construction/math_target_contract.py",
    "grain_research/construction/math_prospective_evidence.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "scripts/run_grain_target_contract_hm221.py",
)


def prepare(out):
    """Freeze exact historical endpoints and source token supports, not outcomes."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve prior or uncertain experiments")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    manifest = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    old = R.read(parent / "units.json")[:288]
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    batches, quality = [], []
    for b in range(48):
        group = old[6 * b : 6 * (b + 1)]
        if {(u["author"], u["arm"]) for u in group} != {(b // 2, b % 2)}:
            raise ValueError("Historical author/evidence binding changed")
        if [(u["name"], u["definition"]) for u in group] != [
            (n, d) for n in range(3) for d in range(2)
        ]:
            raise ValueError("Historical contract order changed")
        rows, supports = [], []
        for u in group:
            ids = u["input_ids"]
            decoded = tok.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            start, end = (
                decoded.index("Excerpt 1:"),
                decoded.index("Evidence ends here."),
            )
            encoded = tok(
                decoded, add_special_tokens=False, return_offsets_mapping=True
            )
            if encoded["input_ids"] != ids or len(ids) + 16 > cfg["max_tokens"]:
                raise ValueError("Exact token identity or length unsupported")
            body = [
                i
                for i, (a, z) in enumerate(encoded["offset_mapping"])
                if z > start and a < end
            ]
            supports.append(body)
            rows.append(ids)
        if any(x != supports[0] for x in supports):
            raise ValueError("Contracts change physical evidence support")
        unit = {"author": b // 2, "arm": b % 2, "input_ids": rows, "body": supports[0]}
        M.support(unit)
        batches.append(unit)
        quality.append(
            {
                "author": b // 2,
                "arm": b % 2,
                "body_tokens": len(supports[0]),
                "body_start": min(supports[0]),
                "body_end": max(supports[0]),
                "endpoint_lengths": list(map(len, rows)),
            }
        )
    jobs = [
        {"batch": b, "mode": mode, "audit": False}
        for b in range(48)
        for mode in ("closed", "direct", "relay")
    ]
    jobs += [
        {"batch": b, "mode": mode, "audit": True}
        for b in range(2)
        for mode in ("native", "closed_shift")
    ]
    if len(jobs) != cfg["new_batches"]:
        raise ValueError("Incomplete frozen experiment")
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.write(out / "units.json", batches)
    R.write(out / "jobs.json", jobs)
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "events": 288,
            "reused_native_endpoints": 288,
            "new_endpoints": 888,
            "raw_texts_or_questionnaire_fields_read": False,
            "token_ids_unchanged": True,
            "supports": quality,
            "S_T_body_length_differences": [
                quality[2 * u]["body_tokens"] - quality[2 * u + 1]["body_tokens"]
                for u in range(24)
            ],
        },
    )
    files = [
        parent / "manifest.json",
        parent / "units.json",
        parent / "bindings.json",
        parent / "head.npz",
    ]
    files += [
        parent / k
        for k in (
            "prepared_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        )
    ]
    files += [parent / "batches" / f"{i:03d}.npz" for i in range(48)]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "alphabet": manifest["alphabet"],
            "model_files": manifest["model_files"],
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "old_results_known_new_route_interactions_unobserved": True,
            "original_scope_complete": False,
        },
    )
    R.seal(out, "prepared_seal.json", ("units.json", "jobs.json", "input_quality.json"))
    H.status(out, {"status": "PREPARED", "completed_batches": 0})
    print("PREPARED HM222:144science+4audit/888newendpoints", flush=True)


def collect(out):
    """Collect each frozen graph operation once, with recoverable finite status."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No uncertain or terminal retry")
    for name, digest in manifest["model_files"].items():
        if R.digest(MODEL / name) != digest:
            raise ValueError("Frozen observer changed")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    elapsed, completed, load_seconds = 0.0, 0, 0.0
    try:
        torch.set_num_threads(2)
        stamp = time.monotonic()
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        load_seconds = time.monotonic() - stamp
        alphabet = manifest["alphabet"]["token_ids"]
        head = observer.model.lm_head.weight.detach()[alphabet].float().cpu().numpy()
        old_head = H.selected(ROOT / cfg["parent"] / "head.npz", ("weights",))[
            "weights"
        ]
        if not np.array_equal(head, old_head):
            raise ValueError("Changed digit observation head")
        R.archive(out / "head.npz", weights=head)
        units, jobs = R.read(out / "units.json"), R.read(out / "jobs.json")
        for i, job in enumerate(jobs):
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite cap; no extension")
            unit = units[job["batch"]]
            mode = job["mode"]
            if mode == "closed_shift":
                body = unit["body"]
                rows = np.array(unit["input_ids"], dtype=object)
                shifted = []
                for row in rows:
                    row = list(row)
                    replacement = np.roll([row[j] for j in body], 1)
                    for j, value in zip(body, replacement, strict=True):
                        row[j] = int(value)
                    shifted.append(row)
                unit = {**unit, "input_ids": shifted}
                r, d = 0, 0
            else:
                r, d = M.ROUTES[mode]
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "pending_batch": i,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                    "load_seconds": load_seconds,
                },
            )
            stamp = time.monotonic()
            try:
                logits, states, receipts = M.native_batch(
                    observer, unit, relay_open=r, direct_open=d
                )
                if not np.isfinite(logits).all() or not np.isfinite(states).all():
                    raise ValueError("Nonfinite observation")
                R.archive(
                    out / "batches" / f"{i:03d}.npz",
                    logits=logits[:, alphabet],
                    normalizer=logsumexp(logits, -1),
                    states=states,
                )
                R.write(
                    out / "batches" / f"{i:03d}.json",
                    {
                        "job": job,
                        "relay_open": r,
                        "direct_open": d,
                        "receipts": receipts,
                    },
                )
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "pending_batch": None,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                    "load_seconds": load_seconds,
                },
            )
            if completed % 12 == 0 or completed == len(jobs):
                print(f"HM222 {completed}/148;model_seconds={elapsed:.3f}", flush=True)
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "completed_batches": completed,
                "model_seconds": elapsed,
                "load_seconds": load_seconds,
                "new_API_train_private_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("head.npz", "runtime.json")
            + tuple(
                f"batches/{i:03d}.{ext}" for i in range(148) for ext in ("npz", "json")
            ),
        )
        print("COLLECTED HM222", elapsed, flush=True)
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "completed_batches": completed,
                "model_seconds": elapsed,
                "load_seconds": load_seconds,
                "error": repr(exc),
                "no_auto_retry": True,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def load_field(out):
    """Assemble complete route fields with inherited native endpoints unchanged."""
    cfg = R.read(out / "manifest.json")["config"]
    rows = [
        H.selected(
            ROOT / cfg["parent"] / "batches" / f"{b:03d}.npz",
            ("logits", "normalizer", "states"),
        )
        for b in range(48)
    ]
    field = {}
    for key, value in rows[0].items():
        shape = (24, 2, 2, 2, 3, 2) + value.shape[1:]
        a = np.empty(shape)
        for b in range(48):
            a[b // 2, b % 2, 1, 1] = rows[b][key].reshape((3, 2) + value.shape[1:])
            for j, mode in enumerate(("closed", "direct", "relay")):
                current = H.selected(out / "batches" / f"{3 * b + j:03d}.npz", (key,))[
                    key
                ]
                r, d = M.ROUTES[mode]
                a[b // 2, b % 2, r, d] = current.reshape((3, 2) + value.shape[1:])
        field[key] = a
    return field


def stats(values, cfg):
    """Use one fixed whole-author interval, not endpoint-level pseudo-replication."""
    x = np.asarray(values, float)
    lo, hi = bootstrap(x[:, None], cfg["bootstrap_draws"], cfg["bootstrap_seed"])[0]
    return {
        "mean": float(x.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "positive_authors": int((x > 0).sum()),
    }


def analyze(out):
    """Keep all routes, names, definitions, vector and curvature interactions."""
    cfg = R.check(out, ("collected_seal.json",))["config"]
    f = load_field(out)
    y, linear, curvature = T.responses(f["logits"])
    delta = y[:, 0] - y[:, 1]
    definition = delta[..., 0] - delta[..., 1]
    accounts = M.route_accounts(definition)
    va = f["states"][:, 0] - f["states"][:, 1]
    vd = va[..., 0, :] - va[..., 1, :]
    vector_accounts = M.route_accounts(vd)
    lin = linear[:, 0] - linear[:, 1]
    curv = curvature[:, 0] - curvature[:, 1]
    la = M.route_accounts(lin[..., 0] - lin[..., 1])
    ca = M.route_accounts(curv[..., 0] - curv[..., 1])
    mass = np.exp(logsumexp(f["logits"], -1) - f["normalizer"])
    rows = [
        {
            "name": T.NAMES[n],
            "route": name,
            "relay_open": r,
            "direct_open": d,
            **stats(definition[:, r, d, n], cfg),
        }
        for n in range(3)
        for name, (r, d) in M.ROUTES.items()
    ]
    contrasts = [
        {"name": T.NAMES[n], "account": key, **stats(value[:, n], cfg)}
        for n in range(3)
        for key, value in accounts.items()
    ]
    detail = [
        {
            "author": u,
            "route": name,
            "name": T.NAMES[n],
            "definition": criterion,
            "action": float(delta[u, r, d, n, c]),
            "linear_action": float(lin[u, r, d, n, c]),
            "curvature_action": float(curv[u, r, d, n, c]),
        }
        for u in range(24)
        for name, (r, d) in M.ROUTES.items()
        for n in range(3)
        for c, criterion in enumerate(("cooperation", "exploration"))
    ]
    R.archive(
        out / "analysis.npz",
        readings=y,
        delta=delta,
        definition=definition,
        linear_readings=linear,
        curvature_readings=curvature,
        compliance=mass,
        vector_definition=vd,
        **accounts,
        **{"vector_" + key: value for key, value in vector_accounts.items()},
    )
    R.table(out / "definition_by_route.csv", rows)
    R.table(out / "route_contrasts.csv", contrasts)
    R.table(out / "author_actions.csv", detail)
    R.write(
        out / "summary.json",
        {
            "packet": "HM222",
            "primary_anonymous_interaction": stats(accounts["interaction"][:, 2], cfg),
            "all_route_definitions": rows,
            "all_route_accounts": contrasts,
            "primary_linear_curvature": [
                float(x["interaction"][:, 2].mean()) for x in (la, ca)
            ],
            "joint_account_max_error": float(
                np.max(
                    abs(
                        accounts["joint_increment"]
                        - accounts["direct_increment"]
                        - accounts["relay_increment"]
                        - accounts["interaction"]
                    )
                )
            ),
            "linear_curvature_max_error": float(np.max(abs(y - linear - curvature))),
            "compliance_range": [float(mass.min()), float(mass.max())],
            "new_batches": 148,
            "new_endpoints": 888,
            "reused_native_endpoints": 288,
            "actual_authors": 24,
            "closed_reference_keeps_length_opportunity": True,
            "human_behavior_or_personality_accuracy": False,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "definition_by_route.csv",
            "route_contrasts.csv",
            "author_actions.csv",
            "summary.json",
        ),
    )
    print(
        "ANALYZED HM222",
        R.read(out / "summary.json")["primary_anonymous_interaction"],
        flush=True,
    )


def verify(out):
    """Recompute contrasts and actual prefix/closed controls without new runs."""
    cfg = R.check(out, ("collected_seal.json", "analyzed_seal.json"))["config"]
    field = load_field(out)
    a = H.selected(
        out / "analysis.npz",
        (
            "readings",
            "delta",
            "definition",
            "interaction",
            "compliance",
            "linear_readings",
            "curvature_readings",
            "vector_interaction",
        ),
    )
    z = field["logits"] - field["logits"].max(-1, keepdims=True)
    y = np.log(
        sum(np.exp(z[..., j]) for j in range(5, 10))
        / sum(np.exp(z[..., j]) for j in range(5))
    )
    delta = y[:, 0] - y[:, 1]
    definition = delta[..., 0] - delta[..., 1]
    interaction = (
        definition[:, 1, 1]
        - definition[:, 1, 0]
        - definition[:, 0, 1]
        + definition[:, 0, 0]
    )
    error = max(
        float(np.max(abs(x - a[k])))
        for k, x in (
            ("readings", y),
            ("delta", delta),
            ("definition", definition),
            ("interaction", interaction),
        )
    )
    error = max(
        error, float(np.max(abs(y - a["linear_readings"] - a["curvature_readings"])))
    )
    mass = sum(np.exp(field["logits"][..., j] - field["normalizer"]) for j in range(10))
    error = max(error, float(np.max(abs(mass - a["compliance"]))))
    primary = interaction[:, 2]
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    draws = primary[rng.integers(24, size=(cfg["bootstrap_draws"], 24))].mean(-1)
    expected = [primary.mean(), *np.quantile(draws, [0.025, 0.975])]
    summary = R.read(out / "summary.json")["primary_anonymous_interaction"]
    error = max(
        error,
        max(
            abs(summary[k] - float(v))
            for k, v in zip(("mean", "ci_low", "ci_high"), expected, strict=True)
        ),
    )
    if summary["positive_authors"] != int((primary > 0).sum()):
        raise ValueError("Primary sign count mismatch")
    weights = H.selected(out / "head.npz", ("weights",))["weights"].astype(float)
    head_error = float(np.max(abs(field["states"] @ weights.T - field["logits"])))
    w = weights[5:].mean(0) - weights[:5].mean(0)
    linear_delta = a["linear_readings"][:, 0] - a["linear_readings"][:, 1]
    linear_definition = linear_delta[..., 0] - linear_delta[..., 1]
    contraction = float(
        np.max(
            abs(
                a["vector_interaction"] @ w
                - M.route_accounts(linear_definition)["interaction"]
            )
        )
    )
    prefix_checked, native_error, closed_error = 0, 0.0, 0.0
    for b in range(48):
        receipts = [
            R.read(out / "batches" / f"{3 * b + j:03d}.json")["receipts"]
            for j in range(3)
        ]
        for layer in range(len(receipts[0])):
            keys = [
                (row[layer]["body_key_hash"], row[layer]["body_value_hash"])
                for row in receipts
            ]
            if len(set(keys)) != 1:
                raise ValueError("Route intervention altered native body prefix")
            if (
                receipts[0][layer]["relay_body_weight_max"] != 0
                or receipts[0][layer]["final_body_weight_max"] != 0
            ):
                raise ValueError("Closed route leaked")
            if (
                receipts[1][layer]["relay_body_weight_max"] != 0
                or receipts[2][layer]["final_body_weight_max"] != 0
            ):
                raise ValueError("Single route leaked")
            prefix_checked += 1
    for b in range(2):
        native = H.selected(
            out / "batches" / f"{144 + 2 * b:03d}.npz", ("logits", "states")
        )
        old = H.selected(
            ROOT / cfg["parent"] / "batches" / f"{b:03d}.npz", ("logits", "states")
        )
        closed = H.selected(out / "batches" / f"{3 * b:03d}.npz", ("logits", "states"))
        shifted = H.selected(
            out / "batches" / f"{145 + 2 * b:03d}.npz", ("logits", "states")
        )
        native_error = max(
            native_error, *(float(np.max(abs(native[k] - old[k]))) for k in old)
        )
        closed_error = max(
            closed_error, *(float(np.max(abs(closed[k] - shifted[k]))) for k in closed)
        )
        r0 = R.read(out / "batches" / f"{3 * b:03d}.json")["receipts"]
        r1 = R.read(out / "batches" / f"{145 + 2 * b:03d}.json")["receipts"]
        if [x["relay_KV_hash"] for x in r0] != [x["relay_KV_hash"] for x in r1]:
            raise ValueError("Closed relay depends on same-length body identities")
    if error > 1e-10 or max(head_error, contraction, native_error, closed_error) > 3e-4:
        raise ValueError(
            f"Finite verification failure:{error}/{head_error}/{contraction}/{native_error}/{closed_error}"
        )
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "max_alternate_error": error,
            "native_head_max_error": head_error,
            "vector_contraction_error": contraction,
            "native_replay_error": native_error,
            "closed_perturbation_error": closed_error,
            "prefix_layer_checks": prefix_checked,
            "literal_odds_probability_masses": 1152,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(
        "VERIFIED HM222",
        error,
        head_error,
        contraction,
        native_error,
        closed_error,
        flush=True,
    )


def main():
    """Explicit single-stage execution; no retry or research queue is launched."""
    parser = argparse.ArgumentParser()
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
        print("ALL HM222 SEALS PASS", flush=True)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
