#!/usr/bin/env python3
"""Explain fixed background edits from SOURCE laws; no model or parameter search."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_companion_observation as C
from grain_research.construction import math_conditional_evidence_valuation as M
from grain_research.construction import math_relational_diagnostic_update as L
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_conditional_evidence_valuation_hm244.json"
OUT = ROOT / "results/grain_math_response_v1/conditional_evidence_valuation_r1"
CODE = (
    CONFIG,
    "docs/GRAIN_CONDITIONAL_EVIDENCE_VALUATION_PLAN_HM244.md",
    "grain_research/construction/math_conditional_evidence_valuation.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "grain_research/construction/math_joint_target_atlas.py",
    "grain_research/construction/math_companion_observation.py",
    "scripts/run_grain_conditional_evidence_valuation_hm244.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_conditional_evidence_valuation.py",
)


def arrays(path, keys=None):
    """Read only declared numerical keys without pickle, raw text or questionnaires."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def banks(cfg):
    """Use both previously frozen banks; never alter a scale or choose a winner."""
    z = arrays(ROOT / cfg["source_parent"] / "source_banks.npz")
    result = [
        {k[3:]: v for k, v in z.items() if k.startswith(f"b{b}_")} for b in (0, 1)
    ]
    if [len(b["prototypes"]) for b in result] != [203, 183] or any(
        float(b["variance"]) != cfg["source_variance"] for b in result
    ):
        raise ValueError("Different source reference or scale")
    return result


def check_files(folder, seal):
    """Validate an existing immutable stage without rerunning its experiment."""
    for path, sha in R.read(folder / seal).items():
        if R.digest(folder / path) != sha:
            raise ValueError("Changed ancestor output: " + path)


def addressed_example(points, events, author=0, view=0):
    """Attach the original event support to one cold four-corner query."""
    corners = []
    for c in (0, 1):
        row = []
        for a in (0, 1):
            e = events[8 * author + 4 * view + 2 * c + a]
            if e["address"] != [author, view, c, a]:
                raise ValueError("Different addressed ancestor corner")
            ids = [e["event_ids"][:8], e["event_ids"][8:]]
            if view == 1:
                ids.reverse()
            row.append(
                [
                    {
                        "event_ids": ids[v],
                        "mean": points[author, view, c, a, v].tolist(),
                    }
                    for v in (0, 1)
                ]
            )
        corners.append(row)
    return {
        "frame_key": f"panel{author}/view{view}",
        "focal_view": view,
        "corners": corners,
    }


def prepare():
    """Register every source unit before new calculation, not before known native data."""
    cfg = R.read(ROOT / CONFIG)
    source, points, partial = (
        ROOT / cfg[k] for k in ("source_parent", "points_parent", "partial_parent")
    )
    R.check(source, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    check_files(points, "prepared_seal.json")
    check_files(partial, "readout_seal.json")
    files = (
        source / "manifest.json",
        source / "source_banks.npz",
        source / "built_seal.json",
        source / "verified_seal.json",
        points / "observations.npz",
        points / "event_contracts.json",
        points / "prepared_seal.json",
        partial / "accounts.npz",
        partial / "coverage.csv",
        partial / "summary.json",
        partial / "readout_seal.json",
    )
    OUT.mkdir(parents=True, exist_ok=False)
    R.write(
        OUT / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in CODE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "new_SOURCE_outputs_read": False,
            "native_outputs_previously_known": True,
            "not_blind_preregistration": True,
            "original_scope_complete": False,
        },
    )
    z = arrays(points / "observations.npz", ("points",))
    if z["points"].shape != (24, 2, 2, 2, 2, 1024):
        raise ValueError("Changed full numerical cohort")
    for u, v in np.ndindex(24, 2):
        M.validate_corners(z["points"][u, v], v)
    R.archive(OUT / "observations.npz", points=z["points"])
    R.write(
        OUT / "cold_input.json",
        addressed_example(z["points"], R.read(points / "event_contracts.json")),
    )
    R.seal(
        OUT,
        "prepared_seal.json",
        ("manifest.json", "observations.npz", "cold_input.json"),
    )
    print("Registered all48 SOURCE groups; native collection unchanged", flush=True)


def compute():
    """Evaluate exact posterior reweighting and both accounts without native reads."""
    cfg = R.check(OUT)["config"]
    points = R.prepared(OUT)["points"]
    start = time.monotonic()
    readings = [
        [M.read(bank, points[u, v], v) for u, v in np.ndindex(24, 2)]
        for bank in banks(cfg)
    ]
    result = {
        k: np.array([[r[k] for r in bank] for bank in readings]).reshape(
            2, 24, 2, *readings[0][0][k].shape
        )
        for k in readings[0][0]
    }
    if (
        max(np.max(abs(result[k])) for k in ("closure", "direct_action_error"))
        > cfg["arithmetic_tolerance"]
    ):
        raise ValueError("Exact valuation account failed; do not tune tolerance")
    R.archive(OUT / "source_accounts.npz", **result)
    R.write(
        OUT / "computed.json",
        {
            "compute_seconds": time.monotonic() - start,
            "new_model_calls": 0,
            "source_groups": 48,
        },
    )
    R.seal(OUT, "computed_seal.json", ("source_accounts.npz", "computed.json"))
    print("SOURCE conditional valuation computed once", flush=True)


def contact(a, b, tolerance):
    """Keep constant accounts unidentified, not artificial correlation values."""
    a, b = np.asarray(a), np.asarray(b)
    if max(abs(a - a.mean())) <= tolerance or max(abs(b - b.mean())) <= tolerance:
        return None
    return C.cosine(a, b)


def analyze():
    """Describe all SOURCE mechanisms; native contact is a secondary partial readout."""
    cfg = R.check(OUT, ("computed_seal.json",))["config"]
    z = arrays(OUT / "source_accounts.npz")
    old = arrays(
        ROOT / cfg["partial_parent"] / "accounts.npz",
        ("keys", "mixed_log_odds", "native_logits"),
    )
    keys = old["keys"]
    if keys.shape != (24, 2) or len(set(keys[:, 0].tolist())) != 12:
        raise ValueError("Changed native completion subset; no hidden narrowing")
    u, v = keys.T
    native = C.read_logits(old["native_logits"])["log_odds"]
    native_own, native_foreign = (
        native[:, 0, 0] - native[:, 0, 1],
        native[:, 1, 0] - native[:, 1, 1],
    )
    tol = cfg["arithmetic_tolerance"]
    rows, summary, contacts = [], [], []
    for b, law, q in np.ndindex(2, 2, 5):
        total = z["mixed"][b, :, :, law, q]
        composition = z["composition"][b, :, :, law, q]
        within = z["within_profile"][b, :, :, law, q]
        action = z["actions"][b, :, :, law, :, q]
        source_flip = action[..., 0] * action[..., 1] < -(tol**2)
        native_flip = native_own[:, q] * native_foreign[:, q] < -(tol**2)
        summary.append(
            {
                "bank": b,
                "law": L.LAWS[law],
                "target": L.TARGETS[q],
                "groups": 48,
                "mixed_rms": C.rms(total),
                "composition_rms": C.rms(composition),
                "within_profile_rms": C.rms(within),
                "signed_account_product": float(np.mean(composition * within)),
                "source_background_reversals": int(source_flip.sum()),
                "native_complete_groups": len(keys),
                "source_reversals_on_native_support": int(source_flip[u, v].sum()),
                "native_reversals": int(native_flip.sum()),
                "reversal_overlap": int(np.sum(source_flip[u, v] & native_flip)),
                "mixed_native_contact": contact(
                    total[u, v], old["mixed_log_odds"][:, q], tol
                ),
            }
        )
        for au, av in np.ndindex(24, 2):
            rows.append(
                {
                    "bank": b,
                    "author": au,
                    "view": av,
                    "law": L.LAWS[law],
                    "target": L.TARGETS[q],
                    "own_action": float(action[au, av, 0]),
                    "foreign_action": float(action[au, av, 1]),
                    "mixed": float(total[au, av]),
                    "composition": float(composition[au, av]),
                    "within_profile": float(within[au, av]),
                    "native_available": bool(np.any(np.all(keys == [au, av], axis=1))),
                }
            )
        for component in ("mixed", "composition", "within_profile"):
            for query in range(5):
                x = z[component][b, u, v, law, q]
                y = old["mixed_log_odds"][:, query]
                contacts.append(
                    {
                        "bank": b,
                        "law": L.LAWS[law],
                        "component": component,
                        "source_target": L.TARGETS[q],
                        "query_target": L.TARGETS[query],
                        "centered_contact": contact(x, y, tol),
                        "mean_signed_product": float(np.mean(x * y)),
                    }
                )
    R.table(OUT / "evidence_actions.csv", rows)
    R.table(OUT / "mechanisms_by_target.csv", summary)
    R.table(OUT / "native_contacts_all.csv", contacts)
    for b in (0, 1):
        R.write(
            OUT / f"cold_bank{b}.json",
            M.cold(banks(cfg)[b], R.read(OUT / "cold_input.json")),
        )
    R.write(
        OUT / "summary.json",
        {
            "source_authors": [203, 183],
            "source_groups": 48,
            "native_authors": 12,
            "native_groups": 24,
            "native_primary_complete": False,
            "group_units_not_independent_people": True,
            "new_neural_calls": 0,
            "account_closure_max": float(np.max(abs(z["closure"]))),
            "product_within_profile_max": float(
                np.max(abs(z["within_profile"][..., 1, :]))
            ),
            "product_profile_response_shift_max": float(
                np.max(abs(z["profile_response_shift"][..., 1, :]))
            ),
            "targets": summary,
            "not_accuracy_psychological_validity_or_new_math": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        OUT,
        "analyzed_seal.json",
        (
            "evidence_actions.csv",
            "mechanisms_by_target.csv",
            "native_contacts_all.csv",
            "summary.json",
            "cold_bank0.json",
            "cold_bank1.json",
        ),
    )
    print("All mechanisms/signs retained; native support12 authors only", flush=True)


def verify():
    """Use direct-distance Gaussian laws and a fresh cold process, not finer precision."""
    cfg = R.check(OUT, ("computed_seal.json", "analyzed_seal.json"))["config"]
    points, z = R.prepared(OUT)["points"], arrays(OUT / "source_accounts.npz")
    worst, checks = 0.0, 0
    for b, bank in enumerate(banks(cfg)):
        for u, v in np.ndindex(24, 2):
            literal = []
            for c, a in np.ndindex(2, 2):
                p = L.literal(bank, points[u, v, c, a])["profile"]
                literal.append(np.log((p @ L.BITS) / (p @ (1 - L.BITS))))
            odds = np.array(literal).reshape(2, 2, 2, 5)
            action = (odds[:, 0] - odds[:, 1]).transpose(1, 0, 2)
            worst = max(worst, float(np.max(abs(action - z["actions"][b, u, v]))))
            checks += action.size
    cold_errors = []
    for b in (0, 1):
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "cold",
                "--input",
                str(OUT / "cold_input.json"),
                "--bank",
                str(b),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        reading = json.loads(proc.stdout)
        R.write(OUT / f"cold_process_bank{b}.json", reading)
        old = R.read(OUT / f"cold_bank{b}.json")
        cold_errors.append(
            max(
                float(np.max(abs(np.array(reading[k]) - old[k])))
                for k in (
                    "actions",
                    "mixed",
                    "composition",
                    "within_profile",
                    "crossed",
                )
            )
        )
    closure = float(np.max(abs(z["closure"])))
    product = float(np.max(abs(z["within_profile"][..., 1, :])))
    ok = max(worst, closure, product, *cold_errors) <= cfg["arithmetic_tolerance"]
    R.write(
        OUT / "verification.json",
        {
            "PASS": ok,
            "literal_action_checks": checks,
            "literal_max_error": worst,
            "account_closure_max": closure,
            "product_within_profile_max": product,
            "cold_process_errors": cold_errors,
            "same_agent_alternate_not_blind": True,
        },
    )
    R.seal(
        OUT,
        "verification_seal.json",
        ("verification.json", "cold_process_bank0.json", "cold_process_bank1.json"),
    )
    if not ok:
        raise ValueError("Frozen verification failed; preserve the result")
    print(json.dumps(R.read(OUT / "verification.json")), flush=True)


def main():
    """Run immutable stages once, or a read-only addressed cold explanation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check", "cold")
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--bank", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.stage == "cold":
        cfg = R.check(OUT)["config"]
        print(
            json.dumps(
                M.cold(banks(cfg)[args.bank], R.read(args.input)), allow_nan=False
            )
        )
    elif args.stage == "check":
        R.check(
            OUT, ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")
        )
        print("All registered inputs/science and outputs unchanged")
    else:
        {"prepare": prepare, "compute": compute, "analyze": analyze, "verify": verify}[
            args.stage
        ]()


if __name__ == "__main__":
    main()
