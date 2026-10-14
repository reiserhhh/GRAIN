#!/usr/bin/env python3
"""Design, observe and audit one SOURCE-balanced actual-text experiment."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_balanced_evidence as M
from grain_research.construction import math_selector_matrix as S
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_balanced_evidence_hm250.json"
DEFAULT = ROOT / "results/grain_math_response_v1/balanced_evidence_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_BALANCED_EVIDENCE_PLAN_HM250.md",
    "grain_research/construction/math_balanced_evidence.py",
    "scripts/run_grain_balanced_evidence_hm250.py",
    "tests/test_grain_balanced_evidence.py",
    "grain_research/construction/math_selector_matrix.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "scripts/run_grain_observed_family_hm170.py",
)
OLD_Q = [2, 3]
KEYS = ("accounts", "subsets", "conditions", "lengths", "panel_indices")


def token_key(ids):
    """Identify both actual A/C token sequences, not their source coordinates."""
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def prepare(out):
    """Construct all positive mixtures before reading any new native answer."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement/restart")
    cfg = R.read(ROOT / CONFIG)
    parent, selector = ROOT / cfg["parent"], ROOT / cfg["selector_parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    sm = R.check(
        selector,
        ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
    )
    if (
        cfg["panel"] != list(range(6)) + list(range(12, 18))
        or cfg["source_target"] != 3
    ):
        raise ValueError("Changed finite panel/target")
    if cfg["reader_targets"] != ["agreeableness", "conscientiousness"]:
        raise ValueError("Changed reader contract")
    z = H.selected(parent / "observations.npz", KEYS)
    units = R.read(parent / "units.json")
    choices = R.read(selector / "choices.json")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    if any(D.candidates(q, tok) != pm["alphabet"] for q in cfg["reader_targets"]):
        raise ValueError("Changed digit alphabet")
    old = {}
    for bi in range(72):
        ids = [units[bi * 5 + q]["input_ids"] for q in OLD_Q]
        old.setdefault(token_key(ids), {"kind": "parent", "index": bi})
    for job in R.read(selector / "jobs.json"):
        old.setdefault(
            token_key([job["input_ids"][q] for q in OLD_Q]),
            {"kind": "selector", "index": job["job"]},
        )
    jobs, roles, designs, new, weights = [], [], [], {}, []
    source_seconds = time.monotonic()
    lengths = []
    for panel_slot, u in enumerate(cfg["panel"]):
        historical = choices[u * 5 + 3]
        count = historical["removed_condition_counts"]
        if count[0] != count[1]:
            raise ValueError("Changed old SOURCE-C condition stratum")
        design = M.construct(
            z["accounts"][u], z["subsets"][u], z["conditions"][u], count[0]
        )
        support = design["support"]
        if len(support) > cfg["max_support_per_author"]:
            raise ValueError("SOURCE support exceeds finite cap; no relaxation")
        weights.append(design["weights"])
        designs.append(
            {
                "panel_slot": panel_slot,
                "old_panel_author": u,
                "original_author_index": int(z["panel_indices"][u]),
                "support": support.tolist(),
                "candidate_indices": design["candidate_indices"].tolist(),
                "source_actions": design["source_actions"].tolist(),
                "balance_error": design["balance_error"],
                "condition_count": count[0],
                "dual": design["dual"].tolist(),
                "dual_min_slack": design["dual_min_slack"],
                "duality_gap": design["duality_gap"],
                "old_source_actions": historical["source_action"],
            }
        )
        body = S.extract_bodies(
            tok.decode(units[u * 15]["input_ids"], skip_special_tokens=False)
        )
        if [len(tok.encode(t, add_special_tokens=False)) for t in body] != z["lengths"][
            u
        ].tolist():
            raise ValueError("Changed public event tokens")
        for q, task in enumerate(D.BIG5):
            if (
                D.tokenize_prompt(body, task, tok, cfg["max_tokens"])[0]
                != units[u * 15 + q]["input_ids"]
            ):
                raise ValueError("Old full text does not replay exactly")
        for k in support:
            retained = [i for i in range(12) if i not in z["subsets"][u, k]]
            ids = [
                D.tokenize_prompt(
                    [body[i] for i in retained], task, tok, cfg["max_tokens"]
                )[0]
                for task in cfg["reader_targets"]
            ]
            lengths += [len(row) for row in ids]
            key = token_key(ids)
            if key in old:
                handle = old[key]
            elif key in new:
                handle = {"kind": "new", "index": new[key]}
            else:
                handle = {"kind": "new", "index": len(jobs)}
                new[key] = len(jobs)
                jobs.append({"job": len(jobs), "input_ids": ids, "token_sha256": key})
            roles.append(
                {
                    "author": panel_slot,
                    "subset_index": int(k),
                    "handle": handle,
                    "token_sha256": key,
                }
            )
    source_seconds = time.monotonic() - source_seconds
    if len(jobs) > cfg["max_new_batches"]:
        raise ValueError("Finite batch cap exceeded; no model calls")
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files != pm["model_files"] or model_files != sm["model_files"]:
        raise ValueError("Changed frozen observer")
    inputs = []
    for root in (parent, selector):
        inputs += [
            root / p
            for p in (
                "manifest.json",
                "prepared_seal.json",
                "collected_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
                "head.npz",
            )
        ]
    inputs += [parent / p for p in ("observations.npz", "units.json")]
    inputs += [
        selector / p
        for p in ("choices.json", "jobs.json", "roles.json", "analysis.npz")
    ]
    for h in {
        json.dumps(r["handle"], sort_keys=True)
        for r in roles
        if r["handle"]["kind"] != "new"
    }:
        handle = json.loads(h)
        root = parent if handle["kind"] == "parent" else selector
        inputs.append(root / "batches" / f"{handle['index']:03d}.npz")
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.archive(
        out / "observations.npz",
        weights=np.asarray(weights),
        accounts=z["accounts"][cfg["panel"]],
        conditions=z["conditions"][cfg["panel"]],
        subsets=z["subsets"][cfg["panel"]],
        lengths=z["lengths"][cfg["panel"]],
    )
    R.write(out / "designs.json", designs)
    R.write(out / "jobs.json", jobs)
    R.write(out / "roles.json", roles)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "model_files": model_files,
            "alphabet": pm["alphabet"],
            "original_scope_complete": False,
            "source_only_design": True,
            "source_seconds": source_seconds,
        },
    )
    quality = {
        "authors": len(designs),
        "support_by_author": [len(d["support"]) for d in designs],
        "new_batches": len(jobs),
        "new_inquiries": len(jobs) * 2,
        "role_inputs": len(roles),
        "reused_inputs": sum(r["handle"]["kind"] != "new" for r in roles),
        "max_source_balance_error": max(d["balance_error"] for d in designs),
        "source_D_C": [d["source_actions"][2][3] for d in designs],
        "prompt_tokens_range": [min(lengths), max(lengths)],
        "all_event_at_slot_marginals_balanced": True,
        "rival_source_accounts_mean_balanced_not_full_laws": True,
        "application_labels_read": False,
        "new_API_private_train_calls": 0,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "designs.json",
            "jobs.json",
            "roles.json",
            "input_quality.json",
        ),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0, "model_seconds": 0})
    print(json.dumps(quality), flush=True)


def collect(out):
    """Observe every unique supported input once under a nonrenewable finite cap."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg, jobs = manifest["config"], R.read(out / "jobs.json")
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No terminal or uncertain retry")
    for name, value in manifest["model_files"].items():
        if R.digest(MODEL / name) != value:
            raise ValueError("Model identity changed")
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
        alphabet = manifest["alphabet"]["token_ids"]
        head = observer.model.lm_head.weight.detach()[alphabet].float().cpu().numpy()
        old_head = H.selected(ROOT / cfg["parent"] / "head.npz", ("weights",))[
            "weights"
        ]
        if not np.array_equal(head, old_head):
            raise ValueError("Changed historical head")
        R.archive(out / "head.npz", weights=head)
        for bi, job in enumerate(jobs):
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite cap reached; no restart/extension")
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "pending_batch": bi,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                },
            )
            pad = observer.tokenizer.pad_token_id
            if pad is None:
                pad = observer.tokenizer.eos_token_id
            ids = torch.full(
                (2, max(map(len, job["input_ids"]))),
                pad,
                dtype=torch.long,
                device=cfg["device"],
            )
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
                        torch.arange(2, device=cfg["device"]),
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
                    logits=logits[:, alphabet],
                    normalizer=logsumexp(logits, -1),
                    states=states,
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
                },
            )
            if completed % 24 == 0:
                print(
                    f"HM250 observed {completed}/{len(jobs)};model_seconds={elapsed:.3f}",
                    flush=True,
                )
            del output, state, logits, states
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "completed_batches": completed,
                "model_seconds": elapsed,
                "new_API_private_train_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("head.npz", "runtime.json")
            + tuple(f"batches/{i:03d}.npz" for i in range(len(jobs))),
        )
        print("COLLECTED HM250", elapsed, flush=True)
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
    """Resolve reused/new exact inputs without counting role aliases as people."""
    cfg = R.read(out / "manifest.json")["config"]
    roots = {
        "new": out,
        "parent": ROOT / cfg["parent"],
        "selector": ROOT / cfg["selector_parent"],
    }
    cache, result = {}, {}
    for row in R.read(out / "roles.json"):
        h = row["handle"]
        key = h["kind"], h["index"]
        if key not in cache:
            z = H.selected(
                roots[h["kind"]] / "batches" / f"{h['index']:03d}.npz",
                ("logits", "states"),
            )
            cache[key] = (
                z if h["kind"] == "new" else {k: v[OLD_Q] for k, v in z.items()}
            )
        address = row["author"], row["subset_index"]
        if address in result:
            raise ValueError("Duplicate input address")
        result[address] = cache[key]
    return result


def analyze(out):
    """Read the complete prospective design; retain all signs and every author."""
    manifest = R.check(out, ("collected_seal.json",))
    cfg, designs = manifest["config"], R.read(out / "designs.json")
    z, readings = R.prepared(out), native(out)
    old = H.selected(ROOT / cfg["selector_parent"] / "analysis.npz", ("actions",))
    actions, rows, means = [], [], []
    for u, design in enumerate(designs):
        support = design["support"]
        logits = np.stack([readings[u, k]["logits"] for k in support])
        arm, delta, y = M.mixture_action(z["weights"][u][:, support], logits)
        means.append(arm)
        actions.append(delta)
        for j, k in enumerate(support):
            rows.append(
                {
                    "author": u,
                    "old_panel_author": design["old_panel_author"],
                    "subset": k,
                    "p_plus": float(z["weights"][u, 0, k]),
                    "p_minus": float(z["weights"][u, 1, k]),
                    "A_native": float(y[j, 0]),
                    "C_native": float(y[j, 1]),
                }
            )
    action = np.asarray(actions)
    old_action = old["actions"][cfg["panel"], 3][:, OLD_Q]
    source_new = np.array([d["source_actions"][2][3] for d in designs])
    source_old = np.array([d["old_source_actions"][2][3] for d in designs])
    ratio = source_new / source_old
    eligible = np.isfinite(ratio) & (ratio >= 0) & (ratio <= 1)
    ci = S.bootstrap(action, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    surplus = action[:, 1] - action[:, 0]
    summary = {
        "authors": len(designs),
        "source_D_C": source_new.tolist(),
        "native_mean_A_C": action.mean(0).tolist(),
        "native_CI_A_C": ci.tolist(),
        "positive_counts_A_C": (action > 0).sum(0).tolist(),
        "C_minus_A_mean": float(surplus.mean()),
        "C_minus_A_CI": S.bootstrap(
            surplus, cfg["bootstrap_draws"], cfg["bootstrap_seed"]
        ).tolist(),
        "bank_means_A_C": [a.mean(0).tolist() for a in np.split(action, 2)],
        "old_unbalanced_C_selector_mean_A_C": old_action.mean(0).tolist(),
        "source_amplitude_ratio": ratio.tolist(),
        "amplitude_comparison_eligible": eligible.tolist(),
        "eligible_balanced_minus_scaled_old_A_C": (
            action[eligible] - ratio[eligible, None] * old_action[eligible]
        ).tolist(),
        "new_batches": R.read(out / "input_quality.json")["new_batches"],
        "model_seconds": R.read(out / "runtime.json")["model_seconds"],
        "psychological_accuracy": False,
        "all_original_scope_complete": False,
    }
    R.archive(
        out / "analysis.npz",
        native_actions=action,
        arm_means=np.asarray(means),
        old_native_actions=old_action,
        source_actions=np.asarray([d["source_actions"] for d in designs]),
    )
    R.table(out / "native_accounts.csv", rows)
    R.table(
        out / "author_actions.csv",
        [
            {
                "author": u,
                "A_action": float(a[0]),
                "C_action": float(a[1]),
                "C_minus_A": float(a[1] - a[0]),
                "source_D_C": float(source_new[u]),
            }
            for u, a in enumerate(action)
        ],
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        ("analysis.npz", "native_accounts.csv", "author_actions.csv", "summary.json"),
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Use literal sums and full raw constraints; never reoptimize the frozen LP."""
    manifest = R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    cfg, z = manifest["config"], R.prepared(out)
    designs, reading = R.read(out / "designs.json"), native(out)
    result = H.selected(out / "analysis.npz", ("native_actions", "source_actions"))
    head = H.selected(out / "head.npz", ("weights",))["weights"].astype(float)
    balance_error, action_error, head_error, dual_error = 0.0, 0.0, 0.0, 0.0
    source_checks, native_checks = 0, 0
    for u, design in enumerate(designs):
        d = z["weights"][u, 0] - z["weights"][u, 1]
        # Build slot columns independently from the registered source helper.
        slot = np.zeros((495, 96))
        for k, removed in enumerate(z["subsets"][u]):
            retained = sorted(set(range(12)) - set(removed))
            for position, event in enumerate(retained):
                slot[k, event * 8 + position] = 1
        rivals = z["accounts"][u][:, :, [0, 1, 2, 4]].reshape(495, 12)
        features = np.column_stack((slot, rivals))
        for column in features.T:
            balance_error = max(
                balance_error,
                abs(math.fsum(float(w) * float(x) for w, x in zip(d, column))),
            )
            source_checks += 1
        for f in range(3):
            for q in range(5):
                observed = math.fsum(
                    float(w) * float(x) for w, x in zip(d, z["accounts"][u, :, f, q])
                )
                action_error = max(
                    action_error, abs(observed - result["source_actions"][u, f, q])
                )
                source_checks += 1
        ix = design["candidate_indices"]
        local = features[ix]
        scale = np.ptp(local, axis=0)
        scale[scale == 0] = 1
        normalized = (local - local.mean(0)) / scale
        n = len(ix)
        equality = np.vstack(
            (
                np.r_[np.ones(n), np.zeros(n)],
                np.r_[np.zeros(n), np.ones(n)],
                np.column_stack((normalized.T, -normalized.T)),
            )
        )
        objective = np.r_[-z["accounts"][u, ix, 2, 3], z["accounts"][u, ix, 2, 3]]
        dual = np.array(design["dual"])
        dual_error = max(
            dual_error,
            float(max(0, -(objective - equality.T @ dual).min())),
            abs(float(objective @ z["weights"][u][:, ix].ravel() - dual[:2].sum())),
        )
        for q in range(2):
            terms = []
            for k in design["support"]:
                a = reading[u, k]
                row = a["logits"][q]
                maximum = float(max(row))
                high = math.fsum(math.exp(float(v) - maximum) for v in row[5:])
                low = math.fsum(math.exp(float(v) - maximum) for v in row[:5])
                terms.append(float(d[k]) * (math.log(high) - math.log(low)))
                head_error = max(
                    head_error,
                    float(np.max(abs(a["states"][q].astype(float) @ head.T - row))),
                )
                native_checks += 1
            action_error = max(
                action_error, abs(math.fsum(terms) - result["native_actions"][u, q])
            )
    passed = (
        balance_error <= cfg["balance_tolerance"]
        and action_error <= 1e-9
        and dual_error <= 1e-8
        and head_error <= cfg["head_tolerance"]
    )
    evidence = {
        "passed": passed,
        "source_literal_checks": source_checks,
        "native_checks": native_checks,
        "balance_error": balance_error,
        "action_error": action_error,
        "dual_certificate_error": dual_error,
        "head_error": head_error,
        "independent_blind": False,
        "same_agent_alternate_arithmetic": True,
    }
    R.write(out / "verification.json", evidence)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(evidence), flush=True)
    if not passed:
        raise ValueError("Frozen audit failed; preserve, no science edit")


def main():
    """Use one explicit stage, leaving terminal stages immutable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
