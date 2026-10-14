#!/usr/bin/env python3
"""One same-event source-linkage discriminator, without accuracy selection."""

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
from grain_research.construction import math_linked_join as M
from grain_research.construction.math_behavior_predictive import key_accounts
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_event_anchor_field_hm267 import selected

CONFIG = "configs/grain_linked_join_hm270.json"
DEFAULT = ROOT / "results/grain_math_response_v1/linked_join_r1"
BANK_KEYS = ("source_comments", "codes", "anchors", "variance", "directions")
SCIENCE = (
    CONFIG,
    "docs/GRAIN_LINKED_JOIN_PLAN_HM270.md",
    "grain_research/construction/math_linked_join.py",
    "grain_research/construction/math_event_anchor_field.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "grain_research/construction/math_behavior_predictive.py",
    "scripts/run_grain_linked_join_hm270.py",
    "scripts/score_grain_linked_join_cold.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_event_anchor_field_hm267.py",
    "tests/test_grain_linked_join.py",
)


def source_bank(cfg):
    """Read only the previously registered numerical source instrument."""
    return selected(ROOT / cfg["source_parent"] / "source_bundle.npz", BANK_KEYS)


def prepare(out):
    """Bind true native join endpoints, not a different historical stranger graph."""
    if out.exists():
        raise FileExistsError("No replacement of a frozen or failed packet")
    cfg = R.read(ROOT / CONFIG)
    source, joint, vector = [
        ROOT / cfg[k] for k in ("source_parent", "joint_parent", "vector_parent")
    ]
    R.check(source, ("computed_seal.json", "verification_seal.json"))
    R.check(joint, ("analyzed_seal.json", "verification_seal.json"))
    R.check(vector)
    bank = source_bank(cfg)
    obs = selected(
        source / "observations.npz",
        ("points", "null_order", "source_indices", "panel_indices"),
    )
    v = selected(vector / "observations.npz", ("panel_vectors", "author_ids"))
    stored = selected(joint / "source_actions.npz", ("full_e5", "donors"))
    sums = selected(joint / "observed_sums.npz", ("chosen", "partition"))
    bindings, ledger, key = (
        R.read(joint / "bindings.json"),
        R.read(source / "event_ledger.json"),
        R.read(joint / "bank.json"),
    )
    if (
        bank["source_comments"].shape != (386, 24, 1024)
        or v["panel_vectors"].shape != (24, 2, 12, 1024)
        or sums["chosen"].shape != (24, 2, 4, 50)
        or sums["partition"].shape != sums["chosen"].shape
        or np.intersect1d(obs["source_indices"], obs["panel_indices"]).size
        or not np.array_equal(np.sort(obs["null_order"]), np.arange(386))
        or len(bindings) != 24
    ):
        raise ValueError("Changed complete source/panel/native scope")
    own_addresses = {(e["author"], e["context_window"]): e["own"] for e in ledger}
    points = np.empty((24, 2, 2, 1024))
    identity_error, normalization_error = 0.0, 0.0
    for u, row in enumerate(bindings):
        if (
            row["unit_id"] != u
            or row["own_author"] != u
            or row["donor_author"] != stored["donors"][u]
        ):
            raise ValueError("Changed author graph identity")
        donor = int(row["donor_author"])
        if donor == u or len(row["events"]) != 2:
            raise ValueError("Self donor or incomplete actual join")
        ids = []
        for w, event in enumerate(row["events"]):
            address = own_addresses[(u, w)]
            if (
                event["window"] != w
                or event["own_id"] != address["event_id"]
                or str(v["author_ids"][u]) != address["author_id"]
            ):
                raise ValueError("Actual source/vector/native event address differs")
            points[u, w, 0] = v["panel_vectors"][u, w, event["own_slot"]]
            points[u, w, 1] = v["panel_vectors"][donor, w, event["foreign_slot"]]
            identity_error = max(
                identity_error,
                float(np.max(abs(points[u, w, 0] - obs["points"][w * 24 + u, 0]))),
                float(
                    np.max(
                        abs(points[u, w, 0] - points[u, w, 1] - stored["full_e5"][u, w])
                    )
                ),
            )
            ids += [event["own_id"], event["foreign_id"]]
        if len(set(ids)) != 4:
            raise ValueError("Repeated native evidence identity")
    normalization_error = float(np.max(abs(np.linalg.norm(points, axis=-1) - 1)))
    if identity_error > 1e-12 or normalization_error > 1e-12:
        raise ValueError("Numeric endpoints do not reconstruct the actual native join")
    native = key_accounts(sums["chosen"] - sums["partition"], key)
    native = {
        k: x[:, :, 0] - x[:, :, 1] - x[:, :, 2] + x[:, :, 3]
        for k, x in native.items()
        if k != "imbalance"
    }
    old = selected(
        joint / "interaction_accounts.npz", ("sum_balanced_key", "sum_common_support")
    )
    native_error = max(
        float(np.max(abs(native[k] - old[f"sum_{k}"])))
        for k in ("balanced_key", "common_support")
    )
    if native_error > 1e-12:
        raise ValueError("Full candidate keys changed")
    record = {
        "frame_key": "joint-edge-0",
        "endpoints": [
            [
                {
                    "event_id": bindings[0]["events"][w][
                        "own_id" if s == 0 else "foreign_id"
                    ],
                    "vector": points[0, w, s].tolist(),
                }
                for s in range(2)
            ]
            for w in range(2)
        ],
    }
    inputs = [
        source / p
        for p in (
            "source_bundle.npz",
            "observations.npz",
            "event_ledger.json",
            "computed_seal.json",
            "verification_seal.json",
        )
    ]
    inputs += [
        joint / p
        for p in (
            "observed_sums.npz",
            "bank.json",
            "bindings.json",
            "source_actions.npz",
            "interaction_accounts.npz",
            "analyzed_seal.json",
            "verification_seal.json",
        )
    ]
    inputs += [vector / "observations.npz", vector / "prepared_seal.json"]
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        points=points,
        balanced=native["balanced_key"],
        common=native["common_support"],
        null_order=obs["null_order"],
    )
    R.write(out / "event_bindings.json", bindings)
    R.write(out / "example_edit.json", record)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "source_comment_prototypes": 9264,
            "dependent_donor_edges": 24,
            "native_orders": 2,
            "native_corners": 4,
            "native_items": 50,
            "same_native_event_vector_error": identity_error,
            "vector_norm_error": normalization_error,
            "native_key_readback_error": native_error,
            "source_profiles_minimum_count": int(
                np.bincount(bank["codes"], minlength=32).min()
            ),
            "new_neural_API_private_training_calls": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "event_bindings.json",
            "example_edit.json",
            "input_quality.json",
        ),
    )
    print("PREPARED HM270 exact native graph, no observations recollected", flush=True)


def compute(out):
    """Run all fixed source contracts once, selecting no favorable mechanism."""
    cfg = R.check(out)["config"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "cap": cfg["compute_seconds_cap"]},
    )
    began = time.perf_counter()
    try:
        obs, bank = R.prepared(out), source_bank(cfg)
        points = M.corners(obs["points"]).reshape(192, 2, 1024)
        arrays, rows, diagonal, descriptions = {}, [], {}, {}
        for space in A.SPACES:
            for view in M.VIEWS:
                for assignment in cfg["assignments"]:
                    order = (
                        np.arange(386) if assignment == "actual" else obs["null_order"]
                    )
                    current = bank | {
                        "codes": bank["codes"][order],
                        "anchors": bank["anchors"][order],
                    }
                    reading = M.pair_readings(current, points, space, view)
                    targets = reading["targets"].reshape(24, 2, 4, 2, 5)
                    actions = M.mixed_actions(targets)
                    name = f"{assignment}/{space}/{view}"
                    arrays[name + "/targets"] = targets
                    arrays[name + "/profiles"] = reading["profile"].reshape(
                        24, 2, 4, 2, 32
                    )
                    arrays[name + "/log_factor"] = reading["log_factor"].reshape(
                        24, 2, 4, 32
                    )
                    diagonal[name] = {}
                    descriptions[name] = {}
                    for component, action in actions.items():
                        arrays[name + "/" + component] = action
                        descriptions[name][component] = {
                            "rms_by_order": np.sqrt(np.mean(action**2, 0)).tolist(),
                            "order_odd_rms": np.sqrt(
                                np.mean(((action[:, 0] - action[:, 1]) / 2) ** 2, 0)
                            ).tolist(),
                        }
                        diagonal[name][component] = []
                        for native_order in range(2):
                            for observer in ("balanced", "common"):
                                contact = M.contact(
                                    action[:, native_order],
                                    obs[observer][:, native_order],
                                )
                                if observer == "balanced":
                                    diagonal[name][component].append(
                                        np.diag(contact).tolist()
                                    )
                                for q, target in enumerate(A.TARGETS):
                                    for r, behavior in enumerate(A.TARGETS):
                                        rows.append(
                                            {
                                                "assignment": assignment,
                                                "space": space,
                                                "source_view": view,
                                                "component": component,
                                                "native_order": native_order,
                                                "observer": observer,
                                                "source_target": target,
                                                "behavior_target": behavior,
                                                "contact": float(contact[q, r]),
                                            }
                                        )
                    if time.perf_counter() - began > cfg["compute_seconds_cap"]:
                        raise TimeoutError(
                            "One finite compute cap; no rescue expansion"
                        )
        R.archive(out / "joint_actions.npz", **arrays)
        R.table(out / "all_contacts.csv", rows)
        elapsed = time.perf_counter() - began
        R.write(
            out / "summary.json",
            {
                "matched_balanced_contacts": diagonal,
                "source_actions": descriptions,
                "input_quality": R.read(out / "input_quality.json"),
                "scientific_CPU_seconds": elapsed,
                "all_signed_contacts": len(rows),
                "fit_parameter_search_or_new_model": False,
                "native_graph_edges_not_independent_people": True,
                "not_accuracy_human_behavior_or_causal_mediation": True,
                "original_scope_complete": False,
            },
        )
        R.write(
            out / "runtime.json",
            {
                "state": "COMPLETE",
                "scientific_CPU_seconds": elapsed,
                "new_model_calls": 0,
            },
        )
        R.seal(
            out,
            "computed_seal.json",
            ("joint_actions.npz", "all_contacts.csv", "summary.json", "runtime.json"),
        )
        print(json.dumps(diagonal["actual/E5/pooled_comments"]), flush=True)
    except Exception as exc:
        R.write(
            out / "runtime.json",
            {
                "state": "FAILED",
                "error": repr(exc),
                "scientific_CPU_seconds": time.perf_counter() - began,
            },
        )
        raise


def verify(out):
    """Check exact native linkage and independent Gaussian arithmetic, once."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    obs, bank = R.prepared(out), source_bank(cfg)
    saved = selected(
        out / "joint_actions.npz",
        [f"actual/{s}/{v}/targets" for s in A.SPACES for v in M.VIEWS],
    )
    points = M.corners(obs["points"])
    error, reconstruction, shared_readback = 0.0, 0.0, 0.0
    for space in A.SPACES:
        z = bank["source_comments"]
        p = points[:2]
        if space == "GRAIN15":
            q = A.projection(bank["directions"])
            p, z = p @ q, z @ q
        for view in M.VIEWS:
            bs = [z, z] if view == M.VIEWS[0] else np.split(z, 2, axis=1)
            name = f"actual/{space}/{view}"
            for u in range(2):
                for order in range(2):
                    for c in range(4):
                        # Direct coordinate distances, no dot-product/log-mixture routine.
                        k = [
                            np.exp(
                                -np.sum((bs[v] - p[u, order, c, v]) ** 2, -1)
                                / (2 * float(bank["variance"]))
                            ).mean(-1)
                            for v in range(2)
                        ]
                        g = bank["codes"]
                        prior = np.bincount(g, minlength=32) / len(g)
                        j = np.array(
                            [np.mean((k[0] * k[1])[g == code]) for code in range(32)]
                        )
                        i = np.array(
                            [
                                k[0][g == code].mean() * k[1][g == code].mean()
                                for code in range(32)
                            ]
                        )
                        weights = np.stack((prior * j, prior * i))
                        weights /= weights.sum(-1, keepdims=True)
                        direct = weights @ (
                            (np.arange(32)[:, None] >> np.arange(5)) & 1
                        )
                        error = max(
                            error,
                            float(
                                np.max(
                                    abs(direct - saved[name + "/targets"][u, order, c])
                                )
                            ),
                        )
                        if view == M.VIEWS[0]:
                            shared = A.readings(
                                A.author_log_law(
                                    p[u, order, c], z, bank["variance"]
                                ).sum(0, keepdims=True),
                                g,
                                bank["anchors"],
                            )
                            shared_readback = max(
                                shared_readback,
                                float(
                                    np.max(abs(shared["probability"][0] - direct[0]))
                                ),
                            )
            actions = M.mixed_actions(saved[name + "/targets"])
            reconstruction = max(
                reconstruction,
                float(
                    np.max(
                        abs(
                            actions["linked"]
                            - actions["profile_product"]
                            - actions["linkage_correction"]
                        )
                    )
                ),
            )
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_linked_join_cold.py"),
            "--bundle",
            str(ROOT / cfg["source_parent"] / "source_bundle.npz"),
            "--record",
            str(out / "example_edit.json"),
        ],
        text=True,
    )
    cold = json.loads(output)
    cold_error = max(
        float(
            np.max(
                abs(
                    np.asarray(cold["readings"][f"{space}/{view}"]["corner_targets"])
                    - saved[f"actual/{space}/{view}/targets"][0]
                )
            )
        )
        for space in A.SPACES
        for view in M.VIEWS
    )
    result = {
        "PASS": max(error, reconstruction, shared_readback, cold_error) < 1e-10,
        "literal_target_checks": 2 * 2 * 4 * 4 * 2 * 5,
        "direct_gaussian_max_error": error,
        "mixed_reconstruction_error": reconstruction,
        "current_shared_author_cold_readback_error": shared_readback,
        "standalone_cold_CLI_error": cold_error,
        "same_agent_not_blind": True,
    }
    R.write(out / "cold_readback.json", cold)
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("cold_readback.json", "verification.json"))
    if not result["PASS"]:
        raise ArithmeticError("Verification failed; do not refit or rerun science")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("HM270 seals PASS; no experiment started")
    else:
        globals()[args.stage](args.out)
