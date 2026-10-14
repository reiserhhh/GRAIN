#!/usr/bin/env python3
"""Cross frozen SOURCE selectors with all native queries once, without fitting."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_selector_matrix as M
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_selector_matrix_hm248.json"
DEFAULT = ROOT / "results/grain_math_response_v1/selector_matrix_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SELECTOR_MATRIX_PLAN_HM248.md",
    "grain_research/construction/math_selector_matrix.py",
    "scripts/run_grain_selector_matrix_hm248.py",
    "tests/test_grain_selector_matrix.py",
    "grain_research/construction/math_prospective_evidence.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "scripts/run_grain_observed_family_hm170.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
)
KEYS = ("accounts", "subsets", "conditions", "lengths", "panel_indices")
TASKS = tuple(D.BIG5)


def batch_key(ids):
    """Bind all five exact token sequences, not prose similarity or target names."""
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def prepare(out):
    """Freeze all source-only choices and addressed prompts before new answers."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement or automatic restart")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    if cfg["targets"] != pm["config"]["targets"] or not np.array_equal(
        cfg["evaluation_direction"], M.EVALUATIVE
    ):
        raise ValueError("Changed target or competitor definition")
    z = H.selected(parent / "observations.npz", KEYS)
    old_selections, units = (
        R.read(parent / "selections.json"),
        R.read(parent / "units.json"),
    )
    if len(z["panel_indices"]) != 24 or len(units) != 360:
        raise ValueError("Historical complete panel required")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    if D.candidates("agreeableness", tok) != pm["alphabet"]:
        raise ValueError("Changed digit alphabet")
    previous = {}
    for bi in range(72):
        rows = units[bi * 5 : (bi + 1) * 5]
        u, arm = divmod(bi, 3)
        if any(
            (r["author"], r["arm"], r["target"]) != (u, arm, q)
            for q, r in enumerate(rows)
        ):
            raise ValueError("Old batch address changed")
        key = batch_key([r["input_ids"] for r in rows])
        if key in previous:
            raise ValueError("Unexpected duplicate historical batch")
        previous[key] = bi
    texts = []
    for u in range(24):
        rows = units[u * 15 : (u + 1) * 15]
        bodies = M.extract_bodies(
            tok.decode(rows[0]["input_ids"], skip_special_tokens=False)
        )
        if [len(tok.encode(t, add_special_tokens=False)) for t in bodies] != z[
            "lengths"
        ][u].tolist():
            raise ValueError("Changed body lengths after token recovery")
        old_sets = [list(range(12))] + [
            [i for i in range(12) if i not in old_selections[u][a]]
            for a in ("support", "counter")
        ]
        for arm, slots in enumerate(old_sets):
            for q, task in enumerate(TASKS):
                ids, _, _ = D.tokenize_prompt(
                    [bodies[i] for i in slots], task, tok, cfg["max_tokens"]
                )
                if ids != rows[arm * 5 + q]["input_ids"]:
                    raise ValueError(
                        "Old prompt does not replay exactly; no parser fallback"
                    )
        texts.append(bodies)
    choices, action, indices = [], [], []
    for u in range(24):
        selected = M.select_all(
            z["accounts"][u],
            z["subsets"][u],
            z["conditions"][u],
            z["lengths"][u],
            absolute=cfg["length_absolute_tolerance"],
            relative=cfg["length_relative_tolerance"],
        )
        if any(
            selected[2][k] != old_selections[u][k]
            for k in (
                "support",
                "counter",
                "support_index",
                "counter_index",
                "feasible_pairs",
            )
        ):
            raise ValueError("A-only selection changed")
        for row in selected:
            row.update(
                author=u,
                original_author_index=int(z["panel_indices"][u]),
                event_ids=old_selections[u]["event_ids"],
            )
        choices += selected
        action.append([r["source_action"] for r in selected])
        indices.append([[r["support_index"], r["counter_index"]] for r in selected])
    jobs, roles, new_keys, prompt_lengths = [], [], {}, []
    for i in range(12):
        for u in (i, i + 12):
            for selector in (0, 1, 3, 4, 2):
                row = choices[u * 5 + selector]
                for arm, removal in enumerate((row["support"], row["counter"])):
                    slots = [j for j in range(12) if j not in removal]
                    ids = [
                        D.tokenize_prompt(
                            [texts[u][j] for j in slots], task, tok, cfg["max_tokens"]
                        )[0]
                        for task in TASKS
                    ]
                    prompt_lengths += [len(t) for t in ids]
                    key = batch_key(ids)
                    if key in previous:
                        handle = {"kind": "old", "index": previous[key]}
                    elif key in new_keys:
                        handle = {"kind": "new", "index": new_keys[key]}
                    else:
                        handle = {"kind": "new", "index": len(jobs)}
                        new_keys[key] = len(jobs)
                        jobs.append(
                            {
                                "job": len(jobs),
                                "author": u,
                                "selector": selector,
                                "arm": arm,
                                "input_ids": ids,
                                "token_sha256": key,
                            }
                        )
                    roles.append(
                        {
                            "author": u,
                            "selector": selector,
                            "arm": arm,
                            "handle": handle,
                            "token_sha256": key,
                        }
                    )
    if len(jobs) > cfg["max_new_batches"] or len(roles) != 240:
        raise ValueError("Changed finite scope")
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files != pm["model_files"]:
        raise ValueError("Changed frozen observer")
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "prepared_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "observations.npz",
            "units.json",
            "selections.json",
            "head.npz",
            "analysis.npz",
        )
    ]
    inputs += [parent / "batches" / f"{i:03d}.npz" for i in range(72)]
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.archive(
        out / "observations.npz",
        panel_indices=z["panel_indices"],
        selections=np.asarray(indices),
        source_actions=np.asarray(action),
    )
    R.write(out / "choices.json", choices)
    R.write(out / "jobs.json", jobs)
    R.write(out / "roles.json", roles)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "alphabet": pm["alphabet"],
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "model_files": model_files,
            "actual_new_batches": len(jobs),
            "A_historical_known_not_prospective": True,
            "source_only_selection": True,
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "events": 288,
            "old_prompt_exact_replays": 360,
            "new_batches": len(jobs),
            "new_enquiries": len(jobs) * 5,
            "role_batches": 240,
            "old_role_reuses": sum(r["handle"]["kind"] == "old" for r in roles),
            "new_role_aliases": sum(r["handle"]["kind"] == "new" for r in roles)
            - len(jobs),
            "distinct_pair_counts_by_author": [
                len({tuple(indices[u][q]) for q in range(5)}) for u in range(24)
            ],
            "prompt_tokens_range": [min(prompt_lengths), max(prompt_lengths)],
            "application_labels_read": False,
            "new_embedding_generation_fit_API_train": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "choices.json",
            "jobs.json",
            "roles.json",
            "input_quality.json",
        ),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0, "model_seconds": 0})
    print("PREPARED HM248", len(jobs), "new batches", flush=True)


def collect(out):
    """Observe only unique NEW frozen batches within the nonrenewable cap."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg, jobs = manifest["config"], R.read(out / "jobs.json")
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No terminal/uncertain automatic retry")
    for name, h in manifest["model_files"].items():
        if R.digest(MODEL / name) != h:
            raise ValueError("Changed model file")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    elapsed, completed = 0.0, 0
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    try:
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        alpha = manifest["alphabet"]["token_ids"]
        head = observer.model.lm_head.weight.detach()[alpha].float().cpu().numpy()
        old_head = H.selected(ROOT / cfg["parent"] / "head.npz", ("weights",))[
            "weights"
        ]
        if not np.array_equal(head, old_head):
            raise ValueError("Changed historical output head")
        R.archive(out / "head.npz", weights=head)
        for bi, job in enumerate(jobs):
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite cap reached; no extension/restart")
            runtime = {
                "status": "RUNNING",
                "pid": os.getpid(),
                "pending_batch": bi,
                "completed_batches": completed,
                "model_seconds": elapsed,
            }
            H.status(out, runtime)
            pad = observer.tokenizer.pad_token_id
            if pad is None:
                pad = observer.tokenizer.eos_token_id
            width = max(map(len, job["input_ids"]))
            ids = torch.full((5, width), pad, dtype=torch.long, device=cfg["device"])
            mask, last = torch.zeros_like(ids), []
            for q, row in enumerate(job["input_ids"]):
                ids[q, : len(row)] = torch.tensor(row, device=cfg["device"])
                mask[q, : len(row)] = 1
                last.append(len(row) - 1)
            stamp = time.monotonic()
            try:
                with torch.inference_mode():
                    output = observer.model.model(
                        input_ids=ids,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    state = output.last_hidden_state[
                        torch.arange(5, device=cfg["device"]),
                        torch.tensor(last, device=cfg["device"]),
                    ]
                    logits = (
                        observer.model.lm_head(state)
                        .float()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )
                    states = state.float().cpu().numpy()
                if not np.isfinite(logits).all() or not np.isfinite(states).all():
                    raise ValueError("Nonfinite native observation")
                R.archive(
                    out / "batches" / f"{bi:03d}.npz",
                    logits=logits[:, alpha],
                    normalizer=logsumexp(logits, -1),
                    states=states,
                )
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            H.status(
                out,
                runtime
                | {
                    "pending_batch": None,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                },
            )
            if completed % 16 == 0:
                print(
                    f"HM248 observed {completed}/{len(jobs)} batches;model_seconds={elapsed:.3f}",
                    flush=True,
                )
            del output, logits, state, states
        H.status(
            out,
            {
                "status": "COLLECTED",
                "completed_batches": completed,
                "model_seconds": elapsed,
                "pid": None,
                "new_API_train_private_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("head.npz", "runtime.json")
            + tuple(f"batches/{i:03d}.npz" for i in range(len(jobs))),
        )
        print("COLLECTED HM248", elapsed, flush=True)
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "completed_batches": completed,
                "model_seconds": elapsed,
                "error": repr(exc),
                "no_auto_retry": True,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def native(out):
    """Resolve exact observation handles without treating aliases as replication."""
    manifest = R.read(out / "manifest.json")
    parent = ROOT / manifest["config"]["parent"]
    cache, roles = {}, R.read(out / "roles.json")
    shape = (24, 5, 2, 5)
    result = {
        "logits": np.empty(shape + (10,)),
        "normalizer": np.empty(shape),
        "states": np.empty(shape + (1536,)),
    }
    seen = set()
    for role in roles:
        h = role["handle"]
        key = h["kind"], h["index"]
        if key not in cache:
            location = parent if h["kind"] == "old" else out
            cache[key] = H.selected(
                location / "batches" / f"{h['index']:03d}.npz", tuple(result)
            )
        address = role["author"], role["selector"], role["arm"]
        if address in seen:
            raise ValueError("Duplicate role address")
        seen.add(address)
        for k, values in result.items():
            values[address] = cache[key][k]
    if len(seen) != 240:
        raise ValueError("Incomplete family cannot be analyzed as full scope")
    return result


def analyze(out):
    """Report every matrix cell, fixed competitor and prospective primary."""
    manifest = R.check(out, ("collected_seal.json",))
    cfg, z, n = manifest["config"], R.prepared(out), native(out)
    y, action = M.read_actions(n["logits"])
    parts = M.decompose(action)
    ci = M.bootstrap(action, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    ri = M.bootstrap(parts["remainder"], cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    pi = M.bootstrap(parts["primary"], cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    contact, banks = M.crossbank_witness(parts["interaction"])
    rows = []
    for r, selector in enumerate(cfg["targets"]):
        for q, reader in enumerate(cfg["targets"]):
            rows.append(
                {
                    "selector": selector,
                    "reader": reader,
                    "new_selector": r != 2,
                    "raw_mean": float(action[:, r, q].mean()),
                    "raw_ci_low": float(ci[r, q, 0]),
                    "raw_ci_high": float(ci[r, q, 1]),
                    "raw_positive_authors": int(np.count_nonzero(action[:, r, q] > 0)),
                    "evaluation_remainder_mean": float(
                        parts["remainder"][:, r, q].mean()
                    ),
                    "remainder_ci_low": float(ri[r, q, 0]),
                    "remainder_ci_high": float(ri[r, q, 1]),
                    "source_full_mean": float(z["source_actions"][:, r, 0, q].mean()),
                    "source_association_mean": float(
                        z["source_actions"][:, r, 1, q].mean()
                    ),
                    "source_departure_mean": float(
                        z["source_actions"][:, r, 2, q].mean()
                    ),
                }
            )
    compliance = np.exp(logsumexp(n["logits"], -1) - n["normalizer"])
    R.table(out / "selector_reader_matrix.csv", rows)
    R.table(
        out / "author_actions.csv",
        [
            {
                "author": u,
                "selector": cfg["targets"][r],
                "reader": cfg["targets"][q],
                "raw": float(action[u, r, q]),
                "evaluative": float(parts["evaluative"][u, r, q]),
                "remainder": float(parts["remainder"][u, r, q]),
                "source_departure": float(z["source_actions"][u, r, 2, q]),
            }
            for u in range(24)
            for r in range(5)
            for q in range(5)
        ],
    )
    R.archive(
        out / "analysis.npz",
        native_readings=y,
        actions=action,
        compliance=compliance,
        bank_interactions=banks,
        **parts,
    )
    primary = {
        "new_driver_diagonal_surplus_mean": float(parts["primary"].mean()),
        "ci_low": float(pi[0]),
        "ci_high": float(pi[1]),
        "positive_authors": int(np.count_nonzero(parts["primary"] > 0)),
    }
    R.write(
        out / "summary.json",
        {
            "packet": "HM248",
            "primary": primary,
            "all_cells": rows,
            "fixed_evaluation_direction": cfg["evaluation_direction"],
            "signed_crossbank_interaction_contact": contact,
            "mean_matrix_singular_values_descriptive": np.linalg.svd(
                action.mean(0), compute_uv=False
            ).tolist(),
            "evaluation_reconstruction_max_error": float(
                np.max(abs(action - parts["evaluative"] - parts["remainder"]))
            ),
            "compliance_range": [float(compliance.min()), float(compliance.max())],
            "actual_new_batches": manifest["actual_new_batches"],
            "actual_new_inquiries": manifest["actual_new_batches"] * 5,
            "old_A_known_not_new_confirmation": True,
            "new_authors": 0,
            "no_prediction_accuracy_or_new_math_claim": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "summary.json",
            "selector_reader_matrix.csv",
            "author_actions.csv",
        ),
    )
    print(json.dumps({"primary": primary, "signed_bank_contact": contact}), flush=True)


def verify(out):
    """Independently replay all selected kernel accounts, native odds and addresses."""
    manifest = R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    cfg, parent = manifest["config"], ROOT / manifest["config"]["parent"]
    z = H.selected(
        parent / "observations.npz",
        KEYS + ("source_means", "codes", "variance", "points", "vectors"),
    )
    choices = R.read(out / "choices.json")
    a = H.selected(
        out / "analysis.npz",
        ("actions", "native_readings", "evaluative", "remainder", "primary"),
    )
    n = native(out)
    errors, points = [], []
    for row in choices:
        u, r = row["author"], row["selector"]
        candidates = M.select_all(
            z["accounts"][u],
            z["subsets"][u],
            z["conditions"][u],
            z["lengths"][u],
            absolute=cfg["length_absolute_tolerance"],
            relative=cfg["length_relative_tolerance"],
        )
        if any(row[k] != candidates[r][k] for k in candidates[r]):
            raise ValueError("SOURCE selection/address replay failed")
        for index, removal in (
            (row["support_index"], row["support"]),
            (row["counter_index"], row["counter"]),
        ):
            point = z["points"][u, index]
            points.append(
                float(
                    np.max(abs(np.delete(z["vectors"][u], removal, 0).mean(0) - point))
                )
            )
            k = np.exp(
                -np.sum((z["source_means"] - point) ** 2, -1)
                / (2 * float(z["variance"]))
            ).mean(1)
            g = z["codes"]
            law = np.array([k[g == c].sum() / len(k) for c in range(32)])
            for q in range(5):
                high = ((np.arange(32) >> q) & 1).astype(bool)
                other = g & ~(1 << q)
                base = np.array(
                    [
                        k[other == (code & ~(1 << q))].mean()
                        * np.count_nonzero(g == code)
                        / len(k)
                        for code in range(32)
                    ]
                )
                f, b = (
                    np.log(law[high].sum() / law[~high].sum()),
                    np.log(base[high].sum() / base[~high].sum()),
                )
                errors.extend(
                    abs(np.array([f, b, f - b]) - z["accounts"][u, index, :, q])
                )
    literal = np.empty((24, 5, 2, 5))
    for address in np.ndindex(literal.shape):
        v = n["logits"][address]
        hi, lo = v[5:], v[:5]
        literal[address] = (
            hi.max()
            + np.log(np.exp(hi - hi.max()).sum())
            - lo.max()
            - np.log(np.exp(lo - lo.max()).sum())
        )
    odds_error = float(np.max(abs(literal - a["native_readings"])))
    action_error = float(
        np.max(abs(literal[:, :, 1] - literal[:, :, 0] - a["actions"]))
    )
    old = H.selected(parent / "analysis.npz", ("raw",))["raw"]
    old_error = float(np.max(abs(old - a["actions"][:, 2])))
    weights = H.selected(out / "head.npz", ("weights",))["weights"].astype(float)
    head_error = float(np.max(abs(n["states"].astype(float) @ weights.T - n["logits"])))
    account_error, point_error = float(max(errors)), float(max(points))
    if (
        max(account_error, point_error, odds_error, action_error, old_error)
        > cfg["source_tolerance"]
        or head_error > cfg["head_tolerance"]
    ):
        raise ValueError("Registered audit tolerance failed; no precision rescue")
    roles, jobs = R.read(out / "roles.json"), R.read(out / "jobs.json")
    units = R.read(parent / "units.json")
    for role in roles:
        h = role["handle"]
        ids = (
            [u["input_ids"] for u in units[h["index"] * 5 : (h["index"] + 1) * 5]]
            if h["kind"] == "old"
            else jobs[h["index"]]["input_ids"]
        )
        if batch_key(ids) != role["token_sha256"]:
            raise ValueError("Resolved token address mismatch")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "literal_F_B_D_accounts": len(errors),
            "source_max_error": account_error,
            "point_max_error": point_error,
            "digit_odds_max_error": odds_error,
            "action_max_error": action_error,
            "old_A_action_max_error": old_error,
            "native_head_max_error": head_error,
            "role_token_hash_checks": 240,
            "same_agent_alternate_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM248", account_error, head_error, flush=True)


def main():
    """Run a single explicit immutable stage, never automatically restart."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.output,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("ALL HM248 SEALS PASS", flush=True)
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
