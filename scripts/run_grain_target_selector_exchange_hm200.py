#!/usr/bin/env python3
"""One frozen natural-evidence selection intervention, without score fitting."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_same_event_multitarget as M
from grain_research.construction import math_target_selector_exchange as X
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_joint_target_binding_hm198 as H
from scripts import run_grain_same_event_multitarget_hm199 as P

SPEC = ROOT / "configs/grain_target_selector_exchange_hm200.json"
CODE = (
    "configs/grain_target_selector_exchange_hm200.json",
    "docs/GRAIN_TARGET_SELECTOR_EXCHANGE_PLAN_HM200.md",
    "grain_research/construction/math_target_selector_exchange.py",
    "scripts/run_grain_target_selector_exchange_hm200.py",
    "tests/test_grain_target_selector_exchange.py",
    "scripts/run_grain_same_event_multitarget_hm199.py",
    "scripts/run_grain_joint_target_binding_hm198.py",
    "grain_research/construction/math_same_event_multitarget.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/navigation.py",
)


def prepare(out):
    """Bind original evidence and physical body addresses before any new forwards."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite or automatic r1 restart")
    spec = H.read(SPEC)
    parent, native = ROOT / spec["parent"], ROOT / spec["native_parent"]
    manifest = P.check(parent)
    if H.read(parent / "status.json")["state"] != "COLLECTED":
        raise ValueError("Parent must be terminal")
    inputs = {}
    for name in (
        "prepared_seal.json",
        "collected_seal.json",
        "analyzed_seal.json",
        "verification_seal.json",
    ):
        seal = H.read(parent / name)
        H.check_hashes(parent, seal)
        for key in (name, *seal):
            path = parent / key
            inputs[str(path.relative_to(ROOT))] = H.sha(path)
    old, panel = (
        H.read(native / "private_material.json"),
        P.arrays(parent / "00/observations.npz"),
    )
    lookup = {(r["author_id"], r["view"]): r for r in old}
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    previous = {
        (r["author"], r["view"], r["arm"], r["target"]): r["input_ids"]
        for r in H.read(parent / "00/units.json")
    }
    units = []
    for u, uid in enumerate(panel["author_ids"]):
        for view in (0, 1):
            row = lookup[str(uid), view]
            for arm, removed in enumerate((row["selected"], row["control"])):
                texts = [t for i, t in enumerate(row["texts"]) if i not in removed]
                ids_list, body, prefix = [], None, None
                for q, target in enumerate(M.TASKS):
                    user, _ = D.render(texts, target)
                    messages = [
                        {"role": "system", "content": D.SYSTEM},
                        {"role": "user", "content": user},
                    ]
                    rendered = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    ids = tok.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                    encoded = tok(
                        rendered, add_special_tokens=False, return_offsets_mapping=True
                    )
                    anchor = rendered.index(user)
                    start = anchor + user.index("Excerpt 1:\n")
                    end = anchor + user.rindex("\nEvidence ends here.")
                    ix = [
                        i
                        for i, (a, b) in enumerate(encoded["offset_mapping"])
                        if b > a and b > start and a < end
                    ]
                    current_prefix = ids[: max(ix) + 1]
                    if (
                        ids != encoded["input_ids"]
                        or len(ids) + 16 > spec["max_tokens"]
                    ):
                        raise ValueError("Unsupported native template or token length")
                    if body is not None and (body != ix or prefix != current_prefix):
                        raise ValueError("Query changed physical evidence addresses")
                    if q != 1 and ids != previous[u, view, arm + 1, q]:
                        raise ValueError("Parent query material changed")
                    body, prefix = ix, current_prefix
                    ids_list.append(ids)
                units.append(
                    {
                        "author": u,
                        "view": view,
                        "arm": arm,
                        "body": body,
                        "input_ids": ids_list,
                    }
                )
    if len(units) != 96 or spec["new_batches"] != 194:
        raise ValueError("Complete fixed support required")
    (out / "00").mkdir(parents=True)
    (out / "01").mkdir()
    H.save_npz(
        out / "00/observations.npz",
        source_actions=P.arrays(parent / "analysis.npz")["source_actions"][:, :, 0],
        splits=panel["splits"],
    )
    H.write(out / "00/units.json", units)
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {k: H.sha(ROOT / k) for k in CODE},
            "inputs": inputs,
            "model_files": manifest["model_files"],
            "alphabet": manifest["alphabet"],
            "public_material": "Historical public PANDORA only",
            "application_labels_read": 0,
            "frozen_before_new_observations": True,
            "original_scope_complete": False,
        },
    )
    H.seal(
        out,
        "prepared_seal.json",
        ("00/observations.npz", "00/units.json", "00/manifest.json"),
    )
    H.status_write(
        out,
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
        },
    )
    print("PREPARED HM200:96units/194batches/970enquiries", flush=True)


def check(out):
    """Verify scientific input/code bytes, not only status or file presence."""
    H.check_hashes(out, H.read(out / "prepared_seal.json"))
    m = H.read(out / "00/manifest.json")
    H.check_hashes(ROOT, m["code"])
    H.check_hashes(ROOT, m["inputs"])
    return m


def collect(out):
    """Capture donors and execute each declared foreign/self batch exactly once."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    m, status = check(out), H.read(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No uncertain or terminal restart")
    H.check_hashes(MODEL, m["model_files"])
    lock = out / "collector.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    spec, units = m["spec"], H.read(out / "00/units.json")
    try:
        torch.set_num_threads(2)
        start = time.monotonic()
        observer = DecoderObserver.load(
            str(MODEL),
            device=spec["device"],
            dtype=spec["dtype"],
            max_tokens=spec["max_tokens"],
        )
        status["model_seconds"] += time.monotonic() - start
        if (
            len(observer.model.model.layers) != spec["layers"]
            or observer.model.config.num_attention_heads != spec["query_heads"]
        ):
            raise ValueError("Frozen observer architecture changed")
        alphabet = m["alphabet"]["token_ids"]
        head = observer.model.lm_head
        H.save_npz(
            out / "01/head.npz",
            weights=head.weight.detach()[alphabet].float().cpu().numpy(),
            bias=np.zeros(10)
            if head.bias is None
            else head.bias.detach()[alphabet].float().cpu().numpy(),
        )
        H.write(
            out / "runtime.json",
            {
                "pid": os.getpid(),
                "torch": torch.__version__,
                "device": spec["device"],
                "dtype": spec["dtype"],
            },
        )
        for b, unit in enumerate(units):
            donor = None
            modes = ["native"] + (["self"] if b < 2 else []) + ["foreign"]
            for mode in modes:
                if status["model_seconds"] >= spec["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Finite cap; no renewal")
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=[b, mode]
                )
                H.status_write(out, status)
                start = time.monotonic()
                try:
                    mapping = (
                        None
                        if mode == "native"
                        else (
                            list(range(5))
                            if mode == "self"
                            else spec["donor_permutation"]
                        )
                    )
                    logits, state, mass, diagnostics, conditional = X.native_batch(
                        observer, unit, mapping=mapping, donor=donor
                    )
                    if mode == "native":
                        donor = conditional
                    path = out / f"01/b{b:03d}_{mode}.npz"
                    payload = {
                        "candidate_logits": logits[:, alphabet],
                        "normalizer": logsumexp(logits, axis=1),
                        "states": state,
                        "body_masses": mass,
                    }
                    if conditional is not None:
                        payload["conditional"] = conditional
                    H.save_npz(path, **payload)
                    H.write(
                        path.with_suffix(".json"),
                        {
                            "unit": b,
                            "mode": mode,
                            "address": [unit[k] for k in ("author", "view", "arm")],
                            "sha256": H.sha(path),
                            "blocks": diagnostics,
                        },
                    )
                finally:
                    status["model_seconds"] += time.monotonic() - start
                    H.status_write(out, status)
                status.update(completed=status["completed"] + 1, pending=None)
                H.status_write(out, status)
            if (b + 1) % 8 == 0:
                print(
                    f"COLLECTED HM200 {b + 1}/96units,{status['completed']}/194batches;{status['model_seconds']:.3f}s",
                    flush=True,
                )
        H.seal(
            out,
            "collected_seal.json",
            [p.relative_to(out).as_posix() for p in sorted((out / "01").iterdir())]
            + ["runtime.json"],
        )
        status.update(state="COLLECTED", pending=None)
        H.status_write(out, status)
    except Exception as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = repr(exc)
        H.status_write(out, status)
        raise
    finally:
        lock.unlink()


def complete(out):
    """Load both modes on all addresses; refuse partial acceptance."""
    m, status = check(out), H.read(out / "status.json")
    if status["state"] != "COLLECTED" or status["completed"] != 194:
        raise RuntimeError("Collection incomplete")
    H.check_hashes(out, H.read(out / "collected_seal.json"))
    logits, normalizer, states = (
        np.empty((24, 2, 2, 2, 5, 10)),
        np.empty((24, 2, 2, 2, 5)),
        np.empty((24, 2, 2, 2, 5, 1536)),
    )
    for b, unit in enumerate(H.read(out / "00/units.json")):
        for mode in ("native", "foreign"):
            z = P.arrays(out / f"01/b{b:03d}_{mode}.npz")
            k = (
                unit["author"],
                unit["view"],
                unit["arm"],
                ("native", "foreign").index(mode),
            )
            logits[k], normalizer[k], states[k] = (
                z["candidate_logits"],
                z["normalizer"],
                z["states"],
            )
    return m, P.arrays(out / "00/observations.npz"), logits, normalizer, states


def analyze(out):
    """Keep signed response/CONTACT changes, never select a better predictor."""
    m, z, logits, normalizer, states = complete(out)
    readings = M.digit_units(logits, normalizer)
    action = readings[:, :, 0] - readings[:, :, 1]
    contacts, diagonals = [], []
    groups = ["author_mean", "view0", "view1"] + [
        "split_" + s for s in sorted(set(z["splits"].tolist()))
    ]
    for b, c, s in np.ndindex(2, 4, 3):
        x = z["source_actions"][:, :, b, :, c, s]
        for group in groups:
            xx = P.take(x, z, group)
            for mode, unit in np.ndindex(2, 3):
                yy = P.take(action[:, :, mode, :, unit], z, group)
                cv, cr = X.covariances(xx, yy), M.correlations(xx, yy)
                for r, q in np.ndindex(5, 5):
                    contacts.append(
                        {
                            "source_view": b,
                            "component": (
                                "marginal",
                                "matched",
                                "reference_shift",
                                "profile_composition",
                            )[c],
                            "space": ("full", "retained", "complement")[s],
                            "group": group,
                            "mode": ("native", "foreign")[mode],
                            "unit": ("score", "log_odds", "geometric_logit")[unit],
                            "source_target": M.TASKS[r],
                            "native_query": M.TASKS[q],
                            "covariance": float(cv[r, q]),
                            "r": P.number(cr[r, q]),
                        }
                    )
        if c == 1:
            xx = x.mean(1)
            nn, ff = (action[:, :, j, :, 2].mean(1) for j in (0, 1))
            ci = X.paired_diagonal_ci(
                xx, nn, ff, m["spec"]["bootstrap_draws"], m["spec"]["bootstrap_seed"]
            )
            d = np.diag(X.covariances(xx, ff - nn))
            rr = [
                np.diag(M.correlations(xx, action[:, :, j, :, 0].mean(1)))
                for j in (0, 1)
            ]
            for q in range(5):
                diagonals.append(
                    {
                        "source_view": b,
                        "space": ("full", "retained", "complement")[s],
                        "target": M.TASKS[q],
                        "native_score_r": P.number(rr[0][q]),
                        "foreign_score_r": P.number(rr[1][q]),
                        "geometric_covariance_change": float(d[q]),
                        "change_CI_low": float(ci[q, 0]),
                        "change_CI_high": float(ci[q, 1]),
                    }
                )
    head = P.arrays(out / "01/head.npz")
    head_error = float(
        np.max(
            abs(
                np.einsum("uvamqd,kd->uvamqk", states, head["weights"].astype(float))
                + head["bias"]
                - logits
            )
        )
    )
    self_error = 0.0
    for b in (0, 1):
        a, c = (
            P.arrays(out / f"01/b{b:03d}_{mode}.npz") for mode in ("native", "self")
        )
        self_error = max(
            self_error,
            float(np.max(abs(a["candidate_logits"] - c["candidate_logits"]))),
        )
    c_error = float(np.max(abs(logits[:, :, :, 0, 3] - logits[:, :, :, 1, 3])))
    block_receipts = [
        r for f in (out / "01").glob("b*.json") for r in H.read(f)["blocks"]
    ]
    guard = {
        k: max(r[k] for r in block_receipts)
        for k in (
            "shared_body_KV_error",
            "installed_error",
            "mass_error",
            "outside_error",
        )
    }
    if max(head_error, self_error, c_error) > 3e-4 or max(guard.values()) > 3e-4:
        raise ValueError("Native or self-selector account failure")
    P.table(out / "all_contacts.csv", contacts)
    P.table(out / "matched_diagonals.csv", diagonals)
    H.save_npz(out / "analysis.npz", readings=readings, actions=action)
    summary = {
        "authors": 24,
        "new_batches": 194,
        "new_enquiries": 970,
        "model_seconds": H.read(out / "status.json")["model_seconds"],
        "head_error": head_error,
        "two_self_logit_error": self_error,
        "all96_C_self_logit_error": c_error,
        "guards": guard,
        "foreign_conditional_tv_mean": float(
            np.mean(
                [
                    r["conditional_tv_mean"]
                    for f in (out / "01").glob("b*_foreign.json")
                    for r in H.read(f)["blocks"]
                ]
            )
        ),
        "score_action_change_RMS": np.sqrt(
            np.mean((action[:, :, 1, :, 0] - action[:, :, 0, :, 0]) ** 2, axis=(0, 1))
        ).tolist(),
        "matched_diagonals": diagonals,
        "contacts": len(contacts),
        "candidate_mass_min_mean": [
            float(readings[..., 3].min()),
            float(readings[..., 3].mean()),
        ],
        "psychological_accuracy": False,
        "native_fit": False,
        "original_scope_complete": False,
    }
    H.write(out / "summary.json", summary)
    H.seal(
        out,
        "analyzed_seal.json",
        ("all_contacts.csv", "matched_diagonals.csv", "analysis.npz", "summary.json"),
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "matched_diagonals"}),
        flush=True,
    )


def verify(out):
    """Replay native scores/contact arithmetic and actual installation receipts."""
    import csv

    _, z, logits, _normalizer, states = complete(out)
    H.check_hashes(out, H.read(out / "analyzed_seal.json"))
    head = P.arrays(out / "01/head.npz")
    expected = np.stack(
        [
            head["weights"].astype(float) @ h + head["bias"]
            for h in states.reshape(-1, 1536)
        ]
    ).reshape(logits.shape)
    head_error = float(np.max(abs(expected - logits)))
    a = P.arrays(out / "analysis.npz")
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    score = p @ (np.arange(10) / 9)
    score_error = float(
        np.max(abs((score[:, :, 0] - score[:, :, 1]) - a["actions"][..., 0]))
    )
    cv_error = 0.0
    with (out / "all_contacts.csv").open() as f:
        for row in csv.DictReader(f):
            b, c, s = (
                int(row["source_view"]),
                ("marginal", "matched", "reference_shift", "profile_composition").index(
                    row["component"]
                ),
                ("full", "retained", "complement").index(row["space"]),
            )
            r, q = (
                M.TASKS.index(row["source_target"]),
                M.TASKS.index(row["native_query"]),
            )
            j, k = (
                ("native", "foreign").index(row["mode"]),
                ("score", "log_odds", "geometric_logit").index(row["unit"]),
            )
            x = P.take(z["source_actions"][:, :, b, :, c, s], z, row["group"])[:, r]
            y = P.take(a["actions"][:, :, j, :, k], z, row["group"])[:, q]
            cv_error = max(
                cv_error, abs(np.cov(x, y, ddof=1)[0, 1] - float(row["covariance"]))
            )
    norm_error = max(
        float(np.max(abs(P.arrays(f)["conditional"].astype(float).sum(-1) - 1)))
        for f in (out / "01").glob("b*_native.npz")
    )
    if head_error > 3e-4 or max(score_error, cv_error) > 3e-12 or norm_error > 2e-6:
        raise ValueError("Alternate arithmetic verification failed")
    receipt = {
        "PASS": True,
        "native_accounts": 960,
        "contacts": 18000,
        "head_error": head_error,
        "score_error": score_error,
        "covariance_error": cv_error,
        "donor_normalization_error": norm_error,
        "same_agent_not_blind": True,
    }
    H.write(out / "verification.json", receipt)
    H.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/target_selector_exchange_r1",
    )
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
        args.stage
    ](args.output)
