#!/usr/bin/env python3
"""One body-access summary construction, without scalar endpoint fitting."""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_body_summary_innovation as I
from grain_research.construction import math_relay_value_summary as V
from grain_research.construction import math_same_event_multitarget as M
from grain_research.construction import math_target_selector_exchange as X
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_joint_target_binding_hm198 as H
from scripts import run_grain_relay_value_summary_hm202 as T
from scripts import run_grain_same_event_multitarget_hm199 as P
from scripts import run_grain_target_selector_exchange_hm200 as R

SPEC = ROOT / "configs/grain_body_summary_innovation_hm203.json"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_body_summary_innovation_hm203.json",
            "docs/GRAIN_BODY_SUMMARY_INNOVATION_PLAN_HM203.md",
            "grain_research/construction/math_body_summary_innovation.py",
            "scripts/run_grain_body_summary_innovation_hm203.py",
            "tests/test_grain_body_summary_innovation.py",
            *T.CODE,
        )
    )
)
MODES = ("native", "zero", "common", "foreign")
NEW_MODES = MODES[1:]


def prepare(out):
    """Freeze all same-event banks and scientific code before blind captures."""
    if out.exists():
        raise FileExistsError("No overwrite or automatic restart")
    spec = H.read(SPEC)
    parent = ROOT / spec["parent"]
    pm = R.check(parent)
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
            inputs[str((parent / key).relative_to(ROOT))] = H.sha(parent / key)
    units = H.read(parent / "00/units.json")
    if len(units) != 96 or spec["new_batches"] != 386:
        raise ValueError("Complete finite support required")
    if spec["targets"] != list(M.TASKS) or spec["modes"] != list(MODES):
        raise ValueError("Frozen target/mode family required")
    suffixes = [V.common_suffix(u) for u in units]
    if {len(s[0]) for s in suffixes} != {40}:
        raise ValueError("Original identical suffix support required")
    (out / "00").mkdir(parents=True)
    (out / "01").mkdir()
    H.write(out / "00/units.json", units)
    H.save_npz(out / "00/observations.npz", **P.arrays(parent / "00/observations.npz"))
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {k: H.sha(ROOT / k) for k in CODE},
            "inputs": inputs,
            "model_files": pm["model_files"],
            "alphabet": pm["alphabet"],
            "suffix_tokens": 40,
            "application_labels_read": 0,
            "frozen_before_new_observations": True,
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
    print("PREPARED HM203:96units/386batches/1930inquiries", flush=True)


def collect(out):
    """Observe B once and apply all declared increments with immutable native p."""
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
    parent = ROOT / spec["parent"]
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
        if len(observer.model.model.layers) != spec["layers"]:
            raise ValueError("Frozen architecture changed")
        head = observer.model.lm_head
        alphabet = m["alphabet"]["token_ids"]
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
            native_bank = P.arrays(parent / f"01/b{b:03d}_native.npz")
            blind_values = None
            modes = ["blind"] + (["self"] if b < 2 else []) + list(NEW_MODES)
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
                    if mode == "blind":
                        result = I.blind_batch(observer, unit)
                        blind_values = result[5]["values"]
                    else:
                        result = I.reader_batch(
                            observer,
                            unit,
                            native_bank,
                            blind_values,
                            mode,
                            spec["donor_permutation"] if mode == "foreign" else None,
                        )
                    logits, state, masses, body_receipts = result[:4]
                    payload = {
                        "candidate_logits": logits[:, alphabet],
                        "normalizer": logsumexp(logits, axis=1),
                        "states": state,
                        "body_masses": masses,
                    }
                    if mode == "blind":
                        payload["values"] = blind_values
                    path = out / f"01/b{b:03d}_{mode}.npz"
                    H.save_npz(path, **payload)
                    H.write(
                        path.with_suffix(".json"),
                        {
                            "unit": b,
                            "mode": mode,
                            "address": [unit[k] for k in ("author", "view", "arm")],
                            "sha256": H.sha(path),
                            "blocks": result[6],
                            "body": body_receipts,
                            "blind": result[7] if mode == "blind" else [],
                        },
                    )
                finally:
                    status["model_seconds"] += time.monotonic() - start
                    H.status_write(out, status)
                status.update(completed=status["completed"] + 1, pending=None)
                H.status_write(out, status)
            if (b + 1) % 8 == 0:
                print(
                    f"COLLECTED HM203 {b + 1}/96units,{status['completed']}/386batches;{status['model_seconds']:.3f}s",
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
    """Load all registered endpoint corners, reusing rather than recapturing native."""
    m, status = R.check(out), H.read(out / "status.json")
    if status["state"] != "COLLECTED" or status["completed"] != 386:
        raise RuntimeError("Collection incomplete")
    H.check_hashes(out, H.read(out / "collected_seal.json"))
    logits = np.empty((24, 2, 2, 4, 5, 10))
    norm = np.empty((24, 2, 2, 4, 5))
    states = np.empty((24, 2, 2, 4, 5, 1536))
    for b, unit in enumerate(H.read(out / "00/units.json")):
        for mode in MODES:
            folder = ROOT / m["spec"]["parent"] if mode == "native" else out
            a = P.arrays(folder / f"01/b{b:03d}_{mode}.npz")
            ix = (unit["author"], unit["view"], unit["arm"], MODES.index(mode))
            logits[ix], norm[ix], states[ix] = (
                a["candidate_logits"],
                a["normalizer"],
                a["states"],
            )
    readings = M.digit_units(logits, norm)
    actions = readings[:, :, 0] - readings[:, :, 1]
    effects = {k: actions[:, :, j] for j, k in enumerate(MODES)}
    effects.update({k + "_delta": effects[k] - effects["native"] for k in NEW_MODES})
    return (
        m,
        P.arrays(out / "00/observations.npz"),
        logits,
        states,
        readings,
        actions,
        effects,
    )


def descriptors(out, m):
    """Describe every increment and installed force without selecting heads/layers."""
    rows = []
    parent = ROOT / m["spec"]["parent"]
    for b, unit in enumerate(H.read(out / "00/units.json")):
        native = P.arrays(parent / f"01/b{b:03d}_native.npz")
        blind = P.arrays(out / f"01/b{b:03d}_blind.npz")["values"].astype(float)
        v = native["values"].astype(float)
        inc = v - blind
        common = np.broadcast_to(inc.mean(1, keepdims=True), inc.shape)
        pp = np.stack(
            [
                native["probabilities"][:, q][:, :, ri]
                for q, ri in enumerate(V.common_suffix(unit))
            ],
            axis=1,
        ).astype(float)
        groups = pp.shape[2] // v.shape[2]
        forcing = {}
        for mode in NEW_MODES:
            d = (
                I.increment_donor(
                    v,
                    blind,
                    mode,
                    m["spec"]["donor_permutation"] if mode == "foreign" else None,
                )
                - v
            )
            forcing[mode] = np.einsum(
                "lqhr,lqhrd->lqhd", pp, np.repeat(d, groups, axis=2)
            )
        for q, target in enumerate(M.TASKS):
            rows.append(
                {
                    "unit": b,
                    "author": unit["author"],
                    "view": unit["view"],
                    "arm": unit["arm"],
                    "target": target,
                    "blind_RMS": float(np.sqrt(np.mean(blind[:, q] ** 2))),
                    "increment_RMS": float(np.sqrt(np.mean(inc[:, q] ** 2))),
                    "common_increment_RMS": float(np.sqrt(np.mean(common[:, q] ** 2))),
                    "specific_increment_RMS": float(
                        np.sqrt(np.mean((inc[:, q] - common[:, q]) ** 2))
                    ),
                    **{
                        k + "_force_RMS": float(np.sqrt(np.mean(f[:, q] ** 2)))
                        for k, f in forcing.items()
                    },
                }
            )
    return rows


def analyze(out):
    """Retain all contacts and primary changes; no outcome-based model selection."""
    m, z, logits, states, readings, actions, effects = complete(out)
    groups = ["author_mean", "view0", "view1"] + [
        "split_" + s for s in sorted(set(z["splits"].tolist()))
    ]
    contacts, diagonals = [], []
    for b, c, s in np.ndindex(2, 4, 3):
        x = z["source_actions"][:, :, b, :, c, s]
        for group in groups:
            xx = P.take(x, z, group)
            for name, values in effects.items():
                for k in range(3):
                    yy = P.take(values[..., k], z, group)
                    cv, cr = X.covariances(xx, yy), M.correlations(xx, yy)
                    for r, q in np.ndindex(5, 5):
                        contacts.append(
                            {
                                "source_view": b,
                                "component": T.COMPONENTS[c],
                                "space": T.SPACES[s],
                                "group": group,
                                "effect": name,
                                "unit": T.UNITS[k],
                                "source_target": M.TASKS[r],
                                "native_query": M.TASKS[q],
                                "covariance": float(cv[r, q]),
                                "r": P.number(cr[r, q]),
                            }
                        )
        if c == 1:
            xx = x.mean(1)
            correlations = {
                name: np.diag(M.correlations(xx, effects[name][..., 0].mean(1)))
                for name in MODES
            }
            for name in (k + "_delta" for k in NEW_MODES):
                yy = effects[name][..., 2].mean(1)
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
                            "space": T.SPACES[s],
                            "target": M.TASKS[q],
                            "contrast": name,
                            **{
                                k + "_r": P.number(v[q])
                                for k, v in correlations.items()
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
    parent = ROOT / m["spec"]["parent"]
    self_error = max(
        float(
            np.max(
                abs(
                    P.arrays(parent / f"01/b{b:03d}_native.npz")["candidate_logits"]
                    - P.arrays(out / f"01/b{b:03d}_self.npz")["candidate_logits"]
                )
            )
        )
        for b in (0, 1)
    )
    c_error = float(np.max(abs(logits[:, :, :, 0, 3] - logits[:, :, :, 3, 3])))
    receipts = [H.read(p) for p in sorted((out / "01").glob("b*.json"))]
    guards = {
        k: max(r[k] for a in receipts if a["mode"] != "blind" for r in a["blocks"])
        for k in ("fixed_probability_error", "prefix_V_error")
    }
    guards["relay_body_weight_max"] = max(
        r["relay_body_weight_max"] for a in receipts for r in a["blind"]
    )
    if max(head_error, self_error, c_error, *guards.values()) > 3e-4:
        raise ValueError("Frozen increment reader or controls failed")
    old_cut = ROOT / H.read(parent / "00/manifest.json")["spec"]["parent"]
    blind_replay_error = max(
        float(
            np.max(
                abs(
                    P.arrays(old_cut / f"01/b{b:03d}_native.npz")["candidate_logits"]
                    - P.arrays(out / f"01/b{b:03d}_blind.npz")["candidate_logits"]
                )
            )
        )
        for b in range(96)
    )
    if blind_replay_error > 3e-4:
        raise ValueError("Historical blind instrument differs")
    P.table(out / "all_contacts.csv", contacts)
    P.table(out / "matched_diagonals.csv", diagonals)
    P.table(out / "increment_descriptors.csv", descriptors(out, m))
    H.save_npz(out / "analysis.npz", readings=readings, actions=actions, **effects)
    summary = {
        "authors": 24,
        "new_batches": 386,
        "new_inquiries": 1930,
        "model_seconds": H.read(out / "status.json")["model_seconds"],
        "head_error": head_error,
        "self_error": self_error,
        "foreign_C_error": c_error,
        "blind_replay_error": blind_replay_error,
        "guards": guards,
        "contacts": len(contacts),
        "matched_diagonals": diagonals,
        "score_effect_RMS": {
            k: np.sqrt(np.mean(v[..., 0] ** 2, axis=(0, 1))).tolist()
            for k, v in effects.items()
        },
        "native_fit": False,
        "psychological_accuracy": False,
        "original_scope_complete": False,
    }
    H.write(out / "summary.json", summary)
    H.seal(
        out,
        "analyzed_seal.json",
        (
            "all_contacts.csv",
            "matched_diagonals.csv",
            "increment_descriptors.csv",
            "analysis.npz",
            "summary.json",
        ),
    )
    print({k: v for k, v in summary.items() if k != "matched_diagonals"}, flush=True)


def verify(out):
    """Recalculate every endpoint and covariance by alternate scalar arithmetic."""
    _, z, logits, states, _, actions, effects = complete(out)
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
                - stored["actions"][..., 0]
            )
        )
    )
    cv_error = 0.0
    with (out / "all_contacts.csv").open() as stream:
        for row in csv.DictReader(stream):
            b, c, s = (
                int(row["source_view"]),
                T.COMPONENTS.index(row["component"]),
                T.SPACES.index(row["space"]),
            )
            r, q, k = (
                M.TASKS.index(row["source_target"]),
                M.TASKS.index(row["native_query"]),
                T.UNITS.index(row["unit"]),
            )
            x = P.take(z["source_actions"][:, :, b, :, c, s], z, row["group"])[:, r]
            y = P.take(effects[row["effect"]][..., k], z, row["group"])[:, q]
            cv_error = max(
                cv_error, abs(np.cov(x, y, ddof=1)[0, 1] - float(row["covariance"]))
            )
    delta_error = max(
        float(
            np.max(
                abs(
                    stored[k + "_delta"]
                    - (actions[:, :, MODES.index(k)] - actions[:, :, 0])
                )
            )
        )
        for k in NEW_MODES
    )
    if head_error > 3e-4 or max(score_error, cv_error, delta_error) > 3e-12:
        raise ValueError("Alternate endpoint/contact arithmetic failed")
    receipt = {
        "PASS": True,
        "endpoint_accounts": 1920,
        "contacts": 63000,
        "head_error": head_error,
        "score_error": score_error,
        "covariance_error": cv_error,
        "delta_error": delta_error,
        "same_agent_not_blind": True,
    }
    H.write(out / "verification.json", receipt)
    H.seal(out, "verification_seal.json", ("verification.json",))
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/body_summary_innovation_r1",
    )
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
        args.stage
    ](args.output)
