#!/usr/bin/env python3
"""One frozen whole-target experiment, reusing old laws rather than tuning scores."""

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
from grain_research.construction import math_event_anchor_field as A
from grain_research.construction import math_event_target_order as M
from grain_research.construction import math_ordered_valuation as O
from grain_research.construction import math_source_target_law as S
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_event_anchor_field_hm267 import selected

CONFIG = "configs/grain_event_target_order_hm269.json"
DEFAULT = ROOT / "results/grain_math_response_v1/event_target_order_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVENT_TARGET_ORDER_PLAN_HM269.md",
    "grain_research/construction/math_event_target_order.py",
    "grain_research/construction/math_event_anchor_field.py",
    "grain_research/construction/math_ordered_valuation.py",
    "grain_research/construction/math_source_target_law.py",
    "scripts/run_grain_event_target_order_hm269.py",
    "scripts/score_grain_event_target_order_cold.py",
    "scripts/run_grain_event_anchor_field_hm267.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_event_target_order.py",
)
BANK_KEYS = ("source_comments", "codes", "anchors", "variance", "directions")


def source_bank(cfg):
    """Open only the previously sealed SOURCE numeric instrument."""
    return selected(ROOT / cfg["parent"] / "source_bundle.npz", BANK_KEYS)


def prepare(out):
    """Freeze full support and addressed observations before whole-law analysis."""
    if out.exists():
        raise FileExistsError("No packet replacement or rescue restart")
    cfg = R.read(ROOT / CONFIG)
    parent, cut_file = ROOT / cfg["parent"], ROOT / cfg["cut_parent"]
    R.check(parent, ("computed_seal.json", "verification_seal.json"))
    cut_seal = cut_file.parent / "compiled_seal.json"
    if R.digest(cut_file) != R.read(cut_seal)[cut_file.name]:
        raise ValueError("Changed original386-author cut metadata")
    bank = source_bank(cfg)
    cuts = np.asarray(R.read(cut_file)["label_cutoffs"], float)
    contract = M.cut_contract(bank["codes"], bank["anchors"], cuts)
    obs = selected(
        parent / "observations.npz",
        ("points", "behavior", "null_order", "panel_indices"),
    )
    if (
        bank["source_comments"].shape != (386, 24, 1024)
        or obs["points"].shape != (48, 2, 1024)
        or obs["behavior"].shape != (48, 5)
        or obs["panel_indices"].shape != (24,)
        or not np.array_equal(np.sort(obs["null_order"]), np.arange(386))
    ):
        raise ValueError("Changed complete SOURCE/SAME-event scope")
    names = [
        f"{assignment}/{space}/{family}/{kind}"
        for assignment in cfg["assignments"]
        for space in A.SPACES
        for family in A.FAMILIES
        for kind in ("probability", "anchor_mean")
    ]
    old = selected(parent / "scores.npz", names)
    if any(value.shape != (48, 2, 5) for value in old.values()):
        raise ValueError("Changed original scalar observation")
    inputs = [
        parent / name
        for name in (
            "observations.npz",
            "scores.npz",
            "source_bundle.npz",
            "computed_seal.json",
            "verification_seal.json",
        )
    ]
    inputs += [cut_file, cut_seal]
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {name: R.digest(ROOT / name) for name in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **obs, **old, cuts=cuts)
    R.write(
        out / "input_quality.json",
        contract
        | {
            "source_authors": 386,
            "source_prototypes": 9264,
            "authors": 24,
            "uses": 48,
            "source_unique_anchor_values": [
                len(np.unique(bank["anchors"][:, q])) for q in range(5)
            ],
            "old403_cutoffs_not_used": True,
            "new_neural_API_private_application_label_training_calls": 0,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print("PREPARED HM269 fullsupport/SAME48/all5targets/no model or fit", flush=True)


def compute(out):
    """Compute every target law and comparator once, retaining all signs."""
    cfg = R.check(out)["config"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "cap": cfg["compute_seconds_cap"]},
    )
    start = time.perf_counter()
    try:
        z, bank = R.prepared(out), source_bank(cfg)
        laws = M.event_weights(bank, z["points"].reshape(96, 1024))
        arrays, individual, contacts, diagonals = {}, [], [], {}
        old_error, integral_error, oriented_error = 0.0, 0.0, 0.0
        for key, flat in laws.items():
            w = flat.reshape(48, 2, 386)
            arrays[key + "/source_weights"] = w
            for assignment in cfg["assignments"]:
                order = np.arange(386) if assignment == "actual" else z["null_order"]
                a, codes = bank["anchors"][order], bank["codes"][order]
                M.cut_contract(codes, a, z["cuts"])
                bits = (codes[:, None] >> np.arange(5)) & 1
                name = assignment + "/" + key
                old_error = max(
                    old_error,
                    float(
                        np.max(
                            abs(
                                (flat @ bits).reshape(48, 2, 5)
                                - z[name + "/probability"]
                            )
                        )
                    ),
                    float(
                        np.max(
                            abs((flat @ a).reshape(48, 2, 5) - z[name + "/anchor_mean"])
                        )
                    ),
                )
                for q, target in enumerate(A.TARGETS):
                    support, atom = M.atom_laws(flat, a, q)
                    atom = atom.reshape(48, 2, len(support))
                    delta = atom[:, 0] - atom[:, 1]
                    account = M.ordered_account(support, delta)
                    prefix = name + "/" + target
                    arrays[prefix + "/support"] = support
                    arrays[prefix + "/mass"] = atom
                    arrays[prefix + "/tail"] = account["tail"]
                    integral_error = max(
                        integral_error,
                        float(
                            np.max(abs(account["mean_change"] - account["integral"]))
                        ),
                    )
                    for u in range(48):
                        certificate = O.certificate(delta[u])
                        individual.append(
                            {
                                "assignment": assignment,
                                "contract": key,
                                "use": u // 24,
                                "author_slot": u % 24,
                                "target": target,
                                "status": certificate["status"],
                                "lower": certificate["lower"],
                                "upper": certificate["upper"],
                                "binary_change": float(
                                    delta[u, support > z["cuts"][q]].sum()
                                ),
                                "mean_change": float(account["mean_change"][u]),
                                "positive_area": float(account["positive_area"][u]),
                                "negative_area": float(account["negative_area"][u]),
                            }
                        )
                    for use, ix in (
                        ("pooled", np.arange(48)),
                        ("window0", np.arange(24)),
                        ("window1", np.arange(24, 48)),
                    ):
                        for r, behavior_target in enumerate(A.TARGETS):
                            b = z["behavior"][ix, r]
                            contact = M.behavior_contact(w[ix], b)
                            signed = np.mean(delta[ix] * b[:, None], 0)
                            current = M.ordered_account(support, signed)
                            row = {
                                "assignment": assignment,
                                "contract": key,
                                "use": use,
                                "source_target": target,
                                "behavior_target": behavior_target,
                                "comparator_status": contact["status"],
                                "behavior_weight_scale": contact["scale"],
                                "status": current["certificate"]["status"],
                                "lower": current["certificate"]["lower"],
                                "upper": current["certificate"]["upper"],
                                "positive_generators": current["certificate"][
                                    "positive_generators"
                                ],
                                "negative_generators": current["certificate"][
                                    "negative_generators"
                                ],
                                "support_values": len(support),
                                "declared_cut": float(z["cuts"][q]),
                                "binary_contact": float(
                                    signed[support > z["cuts"][q]].sum()
                                ),
                                "mean_contact": float(current["mean_change"]),
                                "positive_area": float(current["positive_area"]),
                                "negative_area": float(current["negative_area"]),
                            }
                            ckey = prefix + "/" + use + "/" + behavior_target
                            arrays[ckey + "/signed_mass"] = signed
                            arrays[ckey + "/tail"] = current["tail"]
                            if contact["status"] == "OBSERVED":
                                _, oriented = M.atom_laws(
                                    np.stack((contact["plus"], contact["minus"])), a, q
                                )
                                arrays[ckey + "/oriented_probability_laws"] = oriented
                                oriented_error = max(
                                    oriented_error,
                                    float(
                                        np.max(
                                            abs(
                                                signed
                                                - contact["scale"]
                                                * (oriented[0] - oriented[1])
                                            )
                                        )
                                    ),
                                )
                            old_binary = np.mean(
                                (
                                    z[name + "/probability"][ix, 0, q]
                                    - z[name + "/probability"][ix, 1, q]
                                )
                                * b
                            )
                            old_mean = np.mean(
                                (
                                    z[name + "/anchor_mean"][ix, 0, q]
                                    - z[name + "/anchor_mean"][ix, 1, q]
                                )
                                * b
                            )
                            old_error = max(
                                old_error,
                                float(abs(row["binary_contact"] - old_binary)),
                                float(abs(row["mean_contact"] - old_mean)),
                            )
                            integral_error = max(
                                integral_error,
                                float(
                                    abs(current["mean_change"] - current["integral"])
                                ),
                            )
                            contacts.append(row)
                            if q == r:
                                diagonals[ckey] = row
        elapsed = time.perf_counter() - start
        if elapsed > cfg["compute_seconds_cap"]:
            raise TimeoutError("Nonrenewable whole-target CPU cap exceeded")
        if max(old_error, integral_error, oriented_error) > 1e-10:
            raise ValueError("Whole-target account failed; no scalar/threshold rescue")
        R.archive(out / "accounts.npz", **arrays)
        R.table(out / "ordered_contacts.csv", contacts)
        R.table(out / "individual_order.csv", individual)
        summary = {
            "state": "COMPUTED_NOJOB",
            "scientific_CPU_seconds": elapsed,
            "contacts": len(contacts),
            "individual_certificates": len(individual),
            "diagonals": diagonals,
            "old_scalar_readback_error": old_error,
            "whole_support_integral_error": integral_error,
            "oriented_probability_reconstruction_error": oriented_error,
            "no_accuracy_fit_cut_search_or_new_model": True,
            "not_human_trait_order_or_causal_effect": True,
            "original_scope_complete": False,
        }
        R.write(out / "summary.json", summary)
        R.seal(
            out,
            "computed_seal.json",
            (
                "accounts.npz",
                "ordered_contacts.csv",
                "individual_order.csv",
                "summary.json",
            ),
        )
        print(
            json.dumps({k: v for k, v in summary.items() if k != "diagonals"}),
            flush=True,
        )
        print("E5/comment actual matched target curves", flush=True)
        for use in cfg["groups"]:
            print(
                json.dumps(
                    [
                        diagonals[f"actual/E5/comment_mixture/{q}/{use}/{q}"]
                        for q in A.TARGETS
                    ]
                ),
                flush=True,
            )
    except Exception as exc:
        R.write(
            out / "failure.json",
            {"error": repr(exc), "automatic_restart_allowed": False},
        )
        raise


def verify(out):
    """Bounded literal tail accounts and a genuinely separate cold CLI process."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    z, bank = R.prepared(out), source_bank(cfg)
    keys = [
        f"{key}/source_weights"
        for key in (f"{s}/{f}" for s in A.SPACES for f in A.FAMILIES)
    ]
    saved = selected(out / "accounts.npz", keys)
    error, count = 0.0, 0
    for key in (f"{s}/{f}" for s in A.SPACES for f in A.FAMILIES):
        w = saved[key + "/source_weights"]
        for assignment in cfg["assignments"]:
            a = (
                bank["anchors"]
                if assignment == "actual"
                else bank["anchors"][z["null_order"]]
            )
            for q, target in enumerate(A.TARGETS):
                prefix = assignment + "/" + key + "/" + target
                stored = selected(
                    out / "accounts.npz", (prefix + "/support", prefix + "/tail")
                )
                support, tails = stored[prefix + "/support"], stored[prefix + "/tail"]
                for u in (0, 17, 47):
                    signed = w[u, 0] - w[u, 1]
                    direct = np.array(
                        [
                            sum(float(signed[i]) for i in range(386) if a[i, q] > t)
                            for t in support[:-1]
                        ]
                    )
                    error = max(error, float(np.max(abs(direct - tails[u]))))
                    cdf = S.cdf(w[u], a, q, support[:-1])
                    error = max(error, float(np.max(abs(direct + cdf[0] - cdf[1]))))
                    count += len(direct)
    record = {
        "frame_key": "HM269_use0",
        "after": {"event_id": "own0", "vector": z["points"][0, 0].tolist()},
        "before": {"event_id": "foreign0", "vector": z["points"][0, 1].tolist()},
    }
    R.write(out / "cold_input.json", record)
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_event_target_order_cold.py"),
            "--source-bundle",
            str(ROOT / cfg["parent"] / "source_bundle.npz"),
            "--input",
            str(out / "cold_input.json"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    cold = json.loads(run.stdout)
    for key, targets in cold["readings"].items():
        for target, d in targets.items():
            prefix = "actual/" + key + "/" + target
            tails = selected(out / "accounts.npz", (prefix + "/tail",))[
                prefix + "/tail"
            ]
            error = max(
                error, float(np.max(abs(tails[0] - d["tail_after_each_support_value"])))
            )
    if error > 1e-10:
        raise ValueError("Literal/cold account error; preserve science, do not rerun")
    R.write(out / "cold_readback.json", cold)
    R.write(
        out / "verification.json",
        {
            "state": "VERIFIED_NOJOB",
            "alternate_max_error": error,
            "literal_tail_accounts": count,
            "cold_CLI_exit": run.returncode,
            "fixed_units": [0, 17, 47],
            "audit": "same_agent_not_blind",
            "scientific_rerun": False,
        },
    )
    R.seal(
        out,
        "verification_seal.json",
        ("verification.json", "cold_input.json", "cold_readback.json"),
    )
    print(json.dumps(R.read(out / "verification.json")), flush=True)


def main():
    """Run one explicit immutable phase, with no automatic retry or continuation."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("ALL_SEALS_PASS HM269")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
