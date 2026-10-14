#!/usr/bin/env python3
"""One finite SAME-evidence forward relay cut, not an accuracy search."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_query_relay_cut as Q
from grain_research.construction import math_same_event_multitarget as M
from grain_research.construction import math_target_selector_exchange as X
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_joint_target_binding_hm198 as H
from scripts import run_grain_same_event_multitarget_hm199 as P
from scripts import run_grain_target_selector_exchange_hm200 as R

SPEC = ROOT / "configs/grain_query_relay_cut_hm201.json"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_query_relay_cut_hm201.json",
            "docs/GRAIN_QUERY_RELAY_CUT_PLAN_HM201.md",
            "grain_research/construction/math_query_relay_cut.py",
            "scripts/run_grain_query_relay_cut_hm201.py",
            "tests/test_grain_query_relay_cut.py",
            *R.CODE,
        )
    )
)
UNITS = ("score", "log_odds", "geometric_logit")
COMPONENTS = ("marginal", "matched", "reference_shift", "profile_composition")
SPACES = ("full", "retained", "complement")


def prepare(out):
    """Bind sealed prior tokens, SOURCE actions and intact donors without forwards."""
    if out.exists():
        raise FileExistsError("No overwrite or automatic restart")
    spec = H.read(SPEC)
    parent = ROOT / spec["parent"]
    pm = R.check(parent)
    if H.read(parent / "status.json")["state"] != "COLLECTED":
        raise ValueError("Intact parent must be terminal")
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
    units = H.read(parent / "00/units.json")
    if len(units) != 96 or spec["new_batches"] != 194:
        raise ValueError("Full registered support required")
    for u in units:
        Q.relay_support(u)
    if spec["targets"] != list(M.TASKS):
        raise ValueError("Target order changed")
    (out / "00").mkdir(parents=True)
    (out / "01").mkdir()
    H.write(out / "00/units.json", units)
    H.save_npz(out / "00/observations.npz", **P.arrays(parent / "00/observations.npz"))
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "inputs": inputs,
            "model_files": pm["model_files"],
            "alphabet": pm["alphabet"],
            "frozen_before_new_observations": True,
            "public_material": "SAME historical authorized PANDORA token units",
            "application_labels_read": 0,
            "new_fit": False,
            "original_scope_complete": False,
        },
    )
    H.seal(
        out,
        "prepared_seal.json",
        ("00/units.json", "00/observations.npz", "00/manifest.json"),
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
    print("PREPARED HM201:96units/194batches/970enquiries", flush=True)


def collect(out):
    """Acquire two cut modes once; keep old intact donors and no silent replays."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    m, status = R.check(out), H.read(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No uncertain, failed or terminal restart")
    H.check_hashes(MODEL, m["model_files"])
    lock = out / "collector.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
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
            raise ValueError("Frozen architecture changed")
        alphabet, head = m["alphabet"]["token_ids"], observer.model.lm_head
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
            intact_donor = P.arrays(ROOT / spec["parent"] / f"01/b{b:03d}_native.npz")[
                "conditional"
            ]
            cut_donor = None
            for mode in ["native"] + (["self"] if b < 2 else []) + ["foreign"]:
                if status["model_seconds"] >= spec["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Finite cap; no renewal")
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=[b, mode]
                )
                H.status_write(out, status)
                start = time.monotonic()
                try:
                    donor = (
                        None
                        if mode == "native"
                        else cut_donor
                        if mode == "self"
                        else intact_donor
                    )
                    mapping = (
                        None
                        if mode == "native"
                        else list(range(5))
                        if mode == "self"
                        else spec["donor_permutation"]
                    )
                    logits, state, mass, receipts, conditional, relay = Q.native_batch(
                        observer,
                        unit,
                        mapping=mapping,
                        donor=donor,
                        fixed_c=mode == "foreign",
                    )
                    if mode == "native":
                        cut_donor = conditional
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
                            "blocks": receipts,
                            "relay": relay,
                        },
                    )
                finally:
                    status["model_seconds"] += time.monotonic() - start
                    H.status_write(out, status)
                status.update(completed=status["completed"] + 1, pending=None)
                H.status_write(out, status)
            if (b + 1) % 8 == 0:
                print(
                    f"COLLECTED HM201 {b + 1}/96units,{status['completed']}/194batches;{status['model_seconds']:.3f}s",
                    flush=True,
                )
        H.seal(
            out,
            "collected_seal.json",
            [
                p.relative_to(out).as_posix()
                for p in sorted((out / "01").iterdir())
                if not p.name.startswith("._")
            ]
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
    """Load every cut and historical intact corner on identical evidence units."""
    m, z, cut_l, cut_z, cut_h = R.complete(out)
    parent = ROOT / m["spec"]["parent"]
    _, pz, intact_l, intact_z, _ = R.complete(parent)
    if any(not np.array_equal(z[k], pz[k]) for k in z):
        raise ValueError("SOURCE fields or historical author roles changed")
    head, old_head = P.arrays(out / "01/head.npz"), P.arrays(parent / "01/head.npz")
    if any(not np.array_equal(head[k], old_head[k]) for k in head):
        raise ValueError("Different endpoint head")
    cut, intact = M.digit_units(cut_l, cut_z), M.digit_units(intact_l, intact_z)
    actions = np.stack(
        (intact[:, :, 0] - intact[:, :, 1], cut[:, :, 0] - cut[:, :, 1]), axis=2
    )
    return m, z, cut_l, cut_h, cut, actions, Q.factorial_actions(actions)


def analyze(out):
    """Evaluate signed factorial contacts without a best-corner selection."""
    m, z, logits, states, readings, actions, effects = complete(out)
    groups = ["author_mean", "view0", "view1"] + [
        "split_" + s for s in sorted(set(z["splits"].tolist()))
    ]
    contacts, diagonals = [], []
    for b, c, s in np.ndindex(2, 4, 3):
        x = z["source_actions"][:, :, b, :, c, s]
        for group in groups:
            xx = P.take(x, z, group)
            for effect, values in effects.items():
                for k in range(3):
                    yy = P.take(values[..., k], z, group)
                    cv, cr = X.covariances(xx, yy), M.correlations(xx, yy)
                    for r, q in np.ndindex(5, 5):
                        contacts.append(
                            {
                                "source_view": b,
                                "component": COMPONENTS[c],
                                "space": SPACES[s],
                                "group": group,
                                "effect": effect,
                                "unit": UNITS[k],
                                "source_target": M.TASKS[r],
                                "native_query": M.TASKS[q],
                                "covariance": float(cv[r, q]),
                                "r": P.number(cr[r, q]),
                            }
                        )
        if c != 1:
            continue
        xx = x.mean(1)
        corners_r = {
            name: np.diag(M.correlations(xx, effects[name][..., 0].mean(1)))
            for name in ("intact_native", "intact_foreign", "cut_native", "cut_foreign")
        }
        for contrast, delta in (
            ("relay_delta", effects["relay_delta"]),
            ("relay_under_foreign", effects["cut_foreign"] - effects["intact_foreign"]),
            ("interaction", effects["interaction"]),
        ):
            yy = delta[..., 2].mean(1)
            ci = X.paired_diagonal_ci(
                xx,
                np.zeros_like(yy),
                yy,
                m["spec"]["bootstrap_draws"],
                m["spec"]["bootstrap_seed"],
            )
            d = np.diag(X.covariances(xx, yy))
            for q in range(5):
                diagonals.append(
                    {
                        "source_view": b,
                        "space": SPACES[s],
                        "target": M.TASKS[q],
                        "contrast": contrast,
                        **{
                            name + "_r": P.number(v[q]) for name, v in corners_r.items()
                        },
                        "geometric_covariance_change": float(d[q]),
                        "CI_low": float(ci[q, 0]),
                        "CI_high": float(ci[q, 1]),
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
    self_error = max(
        float(
            np.max(
                abs(
                    P.arrays(out / f"01/b{b:03d}_native.npz")["candidate_logits"]
                    - P.arrays(out / f"01/b{b:03d}_self.npz")["candidate_logits"]
                )
            )
        )
        for b in (0, 1)
    )
    c_error = float(np.max(abs(logits[:, :, :, 0, 3] - logits[:, :, :, 1, 3])))
    receipts = [H.read(f) for f in sorted((out / "01").glob("b*.json"))]
    guards = {
        k: max(r[k] for a in receipts for r in a["blocks"])
        for k in (
            "shared_body_KV_error",
            "installed_error",
            "mass_error",
            "outside_error",
        )
    }
    relay_leak = max(r["relay_body_weight_max"] for a in receipts for r in a["relay"])
    if (
        relay_leak != 0
        or max(head_error, self_error, c_error) > 3e-4
        or max(guards.values()) > 2e-6
    ):
        raise ValueError("Declared intervention/control arithmetic failed")
    add_error = float(
        np.max(
            abs(
                effects["joint_delta"]
                - effects["relay_delta"]
                - effects["selection_delta"]
                - effects["interaction"]
            )
        )
    )
    P.table(out / "all_contacts.csv", contacts)
    P.table(out / "matched_diagonals.csv", diagonals)
    H.save_npz(out / "analysis.npz", readings=readings, actions=actions, **effects)
    summary = {
        "authors": 24,
        "new_batches": 194,
        "new_inquiries": 970,
        "model_seconds": H.read(out / "status.json")["model_seconds"],
        "head_error": head_error,
        "self_error": self_error,
        "fixed_C_error": c_error,
        "guards": guards,
        "relay_body_weight_max": relay_leak,
        "factorial_add_error": add_error,
        "score_effect_RMS": {
            k: np.sqrt(np.mean(v[..., 0] ** 2, axis=(0, 1))).tolist()
            for k, v in effects.items()
        },
        "contacts": len(contacts),
        "matched_diagonals": diagonals,
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
    """Recompute stored endpoint and signed contact accounts once, independently."""
    _, z, logits, states, _readings, actions, effects = complete(out)
    H.check_hashes(out, H.read(out / "analyzed_seal.json"))
    stored, head = P.arrays(out / "analysis.npz"), P.arrays(out / "01/head.npz")
    predicted = np.stack(
        [
            head["weights"].astype(float) @ h + head["bias"]
            for h in states.reshape(-1, 1536)
        ]
    ).reshape(logits.shape)
    head_error = float(np.max(abs(predicted - logits)))
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    score_error = float(
        np.max(
            abs(
                (p @ (np.arange(10) / 9))[:, :, 0]
                - (p @ (np.arange(10) / 9))[:, :, 1]
                - stored["actions"][:, :, 1, :, :, 0]
            )
        )
    )
    effect_error = max(
        float(np.max(abs(stored[name] - values))) for name, values in effects.items()
    )
    cv_error = 0.0
    with (out / "all_contacts.csv").open() as stream:
        for row in csv.DictReader(stream):
            b, c, s = (
                int(row["source_view"]),
                COMPONENTS.index(row["component"]),
                SPACES.index(row["space"]),
            )
            r, q, k = (
                M.TASKS.index(row["source_target"]),
                M.TASKS.index(row["native_query"]),
                UNITS.index(row["unit"]),
            )
            x = P.take(z["source_actions"][:, :, b, :, c, s], z, row["group"])[:, r]
            y = P.take(effects[row["effect"]][..., k], z, row["group"])[:, q]
            cv_error = max(
                cv_error, abs(np.cov(x, y, ddof=1)[0, 1] - float(row["covariance"]))
            )
    addition = float(
        np.max(
            abs(
                actions[:, :, 1, 1]
                - actions[:, :, 0, 0]
                - stored["relay_delta"]
                - stored["selection_delta"]
                - stored["interaction"]
            )
        )
    )
    if head_error > 3e-4 or max(score_error, effect_error, cv_error, addition) > 3e-12:
        raise ValueError("Alternate arithmetic verification failed")
    receipt = {
        "PASS": True,
        "cut_endpoint_accounts": 960,
        "contact_accounts": 72000,
        "head_error": head_error,
        "score_error": score_error,
        "effect_error": effect_error,
        "covariance_error": cv_error,
        "factorial_error": addition,
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
        default=ROOT / "results/grain_math_response_v1/query_relay_cut_r1",
    )
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
        args.stage
    ](args.output)
