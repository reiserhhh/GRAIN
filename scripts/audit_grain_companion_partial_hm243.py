#!/usr/bin/env python3
"""Read terminal, complete HM243 corners without redefining its full scope."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_companion_observation as M
from grain_research.construction import math_relational_target_operator as T
from grain_research.construction import math_target_relation_operator as C
from scripts import run_grain_companion_observation_hm243 as H
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_relational_target_operator_hm207 as P

PRIMARY = H.DEFAULT
META = ROOT / "results/grain_math_response_v1/companion_metadata_audit_r1"
OUT = ROOT / "results/grain_math_response_v1/companion_partial_audit_r1"
CODE = (
    "docs/GRAIN_COMPANION_PARTIAL_AUDIT_HM243.md",
    "scripts/audit_grain_companion_partial_hm243.py",
    "tests/test_grain_companion_partial_audit.py",
)


def require_terminal(status, pid):
    """Reject a live/uncertain collector; never stop or restart it here."""
    if status["state"] not in ("BUDGET_STOP", "FAILED"):
        raise RuntimeError("Partial audit requires terminal incomplete primary")
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return
    raise RuntimeError("Recorded collector PID is still present")


def coverage(completed, batches=192):
    """Describe every planned view; only complete four-corner groups qualify."""
    if not 0 <= completed <= batches or batches != 192:
        raise ValueError("Fixed HM243 completion bounds required")
    rows, keys = [], []
    for u, v in np.ndindex(24, 2):
        first = 8 * u + 4 * v
        n = max(0, min(4, completed - first))
        rows.append({"author": u, "view": v, "corners": n, "complete": n == 4})
        if n == 4:
            keys.append((u, v))
    return rows, keys


def prepare():
    """Bind the separate readout contract before opening native results."""
    R.check(PRIMARY)
    OUT.mkdir(parents=True, exist_ok=False)
    R.write(
        OUT / "manifest.json",
        {
            "code": {p: R.digest(ROOT / p) for p in CODE},
            "inputs": {
                str(p.relative_to(ROOT)): R.digest(p)
                for p in (
                    PRIMARY / "prepared_seal.json",
                    PRIMARY / "manifest.json",
                    PRIMARY / "observations.npz",
                    META / "audit_seal.json",
                    META / "condition_accounts.csv",
                )
            },
            "native_outcomes_read_at_registration": False,
            "new_neural_forwards": 0,
            "not_full_primary_acceptance": True,
        },
    )
    print("Partial readout registered; no native outcomes opened", flush=True)


def analyze():
    """Keep every observed target/sign and missing unit after terminal stop."""
    registration = R.read(OUT / "manifest.json")
    for category in ("code", "inputs"):
        for p, digest in registration[category].items():
            if R.digest(ROOT / p) != digest:
                raise ValueError("Changed partial readout contract: " + p)
    status, runtime = R.read(PRIMARY / "status.json"), R.read(PRIMARY / "runtime.json")
    require_terminal(status, runtime["pid"])
    cfg = R.check(PRIMARY)["config"]
    rows, keys = coverage(status["completed"])
    if not keys:
        raise RuntimeError("No complete four-corner observation; no effect imputation")
    units, inputs = R.read(PRIMARY / "units.json"), {}
    logits, states, normalizers = [], [], []
    for u, v in keys:
        zz, hh, nn = [], [], []
        for c, a in np.ndindex(2, 2):
            batch = 8 * u + 4 * v + 2 * c + a
            path = PRIMARY / f"01/b{batch:03d}.npz"
            receipt = R.read(path.with_suffix(".json"))
            if (
                receipt["address"] != [u, v, c, a]
                or R.digest(path) != receipt["sha256"]
            ):
                raise ValueError("Unverified native receipt/address")
            if receipt["prefix_error"] > cfg["head_tolerance"]:
                raise ValueError("Changed causal evidence prefix")
            observed = units[5 * batch : 5 * batch + 5]
            if [x["target"] for x in observed] != list(range(5)):
                raise ValueError("Missing/reordered target in completed batch")
            z = H.arrays(path)
            zz.append(z["logits"])
            hh.append(z["states"])
            nn.append(z["normalizer"])
            for p in (path, path.with_suffix(".json")):
                inputs[str(p.relative_to(ROOT))] = R.digest(p)
        logits.append(np.array(zz).reshape(2, 2, 5, 10))
        states.append(np.array(hh).reshape(2, 2, 5, 1536))
        normalizers.append(np.array(nn).reshape(2, 2, 5))
    logits, states, normalizers = map(np.asarray, (logits, states, normalizers))
    z, head = R.prepared(PRIMARY), H.arrays(PRIMARY / "01/head.npz")
    inputs[str((PRIMARY / "01/head.npz").relative_to(ROOT))] = R.digest(
        PRIMARY / "01/head.npz"
    )
    readings = M.read_logits(logits)
    native = {k: H.mixed_targets(readings[k]) for k in ("score", "log_odds")}
    account = M.head_accounts(
        states.transpose(0, 3, 1, 2, 4), head["weights"], head["bias"]
    )
    source = H.mixed_targets(z["source_corners"])
    cov = H.mixed_targets(z["covariance_corners"])
    u, v = np.array(keys).T
    source, cov = source[:, :, u, v], cov[:, :, u, v]
    with (META / "condition_accounts.csv").open() as f:
        metadata = {(int(r["author"]), int(r["view"])): r for r in csv.DictReader(f)}
    actions, summaries, contacts = [], [], []
    for i, q in np.ndindex(len(keys), 5):
        au, av = keys[i]
        y = readings["log_odds"][i, :, :, q]
        meta = metadata[(au, av)]
        actions.append(
            {
                "author": au,
                "view": av,
                "target": cfg["targets"][q],
                "own_action": float(y[0, 0] - y[0, 1]),
                "foreign_action": float(y[1, 0] - y[1, 1]),
                "old_alone_action": float(z["old_single_actions"][au, av, q, 1]),
                "mixed_log_odds": float(native["log_odds"][i, q]),
                "mixed_score": float(native["score"][i, q]),
                "reader": float(account["reader"][i, q]),
                "representation": float(account["representation"][i, q]),
                "mixed_centered_logit_rms": M.rms(
                    account["mixed_centered_logits"][i, q]
                ),
                "native_minus_head": float(
                    native["log_odds"][i, q] - account["total"][i, q]
                ),
                "focal_condition_count_equal": meta["focal_edit_condition_count_equal"],
                "companion_condition_count_equal": meta[
                    "companion_condition_count_equal"
                ],
            }
        )
    for q, target in enumerate(cfg["targets"]):
        y = readings["log_odds"][..., q]
        own, foreign = y[:, 0, 0] - y[:, 0, 1], y[:, 1, 0] - y[:, 1, 1]
        summaries.append(
            {
                "target": target,
                "complete_views": len(keys),
                "mixed_rms": M.rms(native["log_odds"][:, q]),
                "reader_rms": M.rms(account["reader"][:, q]),
                "representation_rms": M.rms(account["representation"][:, q]),
                "signed_reader_representation_product": float(
                    np.mean(account["reader"][:, q] * account["representation"][:, q])
                ),
                "own_action_rms": M.rms(own),
                "foreign_action_rms": M.rms(foreign),
                "own_foreign_sign_reversals": int(np.sum(own * foreign < 0)),
                "own_foreign_centered_contact": M.cosine(own, foreign),
                "centered_logit_mixed_rms": M.rms(
                    account["mixed_centered_logits"][:, q]
                ),
            }
        )
    for family, values, components, targets in (
        ("uniform32_relation", source, T.COMPONENTS, cfg["targets"]),
        ("empirical_covariance", cov, C.COMPONENTS, C.TARGETS),
    ):
        for bank, component, unit in np.ndindex(2, 3, 2):
            name = ("score", "log_odds")[unit]
            for t, q in np.ndindex(len(targets), 5):
                x, y = values[bank, component, :, t], native[name][:, q]
                contacts.append(
                    {
                        "family": family,
                        "bank": bank,
                        "component": components[component],
                        "unit": name,
                        "source_target": targets[t],
                        "query_target": cfg["targets"][q],
                        "centered_contact": M.cosine(x, y),
                        "mean_signed_product": float(np.mean(x * y)),
                    }
                )
    head_error = float(np.max(abs(account["logits"] - logits.transpose(0, 3, 1, 2, 4))))
    mass = np.exp(logits - logits.max(-1, keepdims=True))
    scalar = np.log(mass[..., 5:].sum(-1) / mass[..., :5].sum(-1))
    scalar_error = float(np.max(abs(H.mixed_targets(scalar) - native["log_odds"])))
    banks, basis = P.banks(ROOT / cfg["relation_parent"])
    field_error = 0.0
    for bank, component, i in np.ndindex(2, 3, len(keys)):
        au, av = keys[i]
        pts = z["points"][au, av]
        delta = pts[0, 0, av] - pts[0, 1, av]
        f = T.field(
            banks[bank], T.COMPONENTS[component], "full", pts[0, 0, 1 - av], av, basis
        ) - T.field(
            banks[bank], T.COMPONENTS[component], "full", pts[1, 0, 1 - av], av, basis
        )
        field_error = max(
            field_error,
            float(np.max(abs(f[T.TARGET_MODES] @ delta - source[bank, component, i]))),
        )
    closure = float(np.max(abs(account["closure"])))
    verification = {
        "PASS": head_error <= cfg["head_tolerance"]
        and max(scalar_error, field_error, closure) <= cfg["arithmetic_tolerance"],
        "head_error": head_error,
        "scalar_mixed_error": scalar_error,
        "source_field_error": field_error,
        "head_closure_error": closure,
        "same_agent_alternate_not_blind": True,
    }
    summary = {
        "primary_state": status,
        "planned_batches": 192,
        "committed_batches": status["completed"],
        "used_batches": 4 * len(keys),
        "planned_views": 48,
        "complete_views": len(keys),
        "planned_authors": 24,
        "authors_with_complete_views": len(set(u.tolist())),
        "authors_with_both_views": sum(
            sum(au == k for au, _ in keys) == 2 for k in range(24)
        ),
        "primary_complete": False,
        "selection_is_serial_prefix_not_random_sample": True,
        "targets": summaries,
        "verification": verification,
        "candidate_mass_range": [
            float(np.exp(readings["log_candidate_partition"] - normalizers).min()),
            float(np.exp(readings["log_candidate_partition"] - normalizers).max()),
        ],
        "not_prediction_accuracy_or_psychological_validity": True,
        "original_scope_complete": False,
    }
    R.write(OUT / "input_receipts.json", inputs)
    R.archive(
        OUT / "accounts.npz",
        keys=np.array(keys),
        native_logits=logits,
        normalizers=normalizers,
        mixed_log_odds=native["log_odds"],
        mixed_score=native["score"],
        source_mixed=source,
        covariance_mixed=cov,
        **{
            k: account[k]
            for k in (
                "reader",
                "representation",
                "total",
                "mixed_hidden",
                "mixed_centered_logits",
            )
        },
    )
    R.table(OUT / "coverage.csv", rows)
    R.table(OUT / "native_actions.csv", actions)
    R.table(OUT / "source_contacts.csv", contacts)
    R.write(OUT / "summary.json", summary)
    R.seal(
        OUT,
        "readout_seal.json",
        (
            "manifest.json",
            "input_receipts.json",
            "accounts.npz",
            "coverage.csv",
            "native_actions.csv",
            "source_contacts.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


def main():
    """Register once, then read only a terminal incomplete collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "analyze"))
    stage = parser.parse_args().stage
    (prepare if stage == "prepare" else analyze)()


if __name__ == "__main__":
    main()
