#!/usr/bin/env python3
"""One finite continuous SOURCE-field / same-event behavior correspondence."""

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
from grain_research.construction import math_event_anchor_field as M
from grain_research.construction import math_event_behavior as B
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_event_anchor_field_hm267.json"
DEFAULT = ROOT / "results/grain_math_response_v1/event_anchor_field_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVENT_ANCHOR_FIELD_PLAN_HM267.md",
    "grain_research/construction/math_event_anchor_field.py",
    "grain_research/construction/math_event_behavior.py",
    "scripts/run_grain_event_anchor_field_hm267.py",
    "scripts/score_grain_event_anchor_field_cold.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_event_anchor_field.py",
)
READINGS = ("probability", "anchor_mean", "log_Bayes_factor")


def selected(path, names):
    """Decompress only an explicit allowlist of numeric blocks."""
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in names}


def jq(path, expression):
    """Export event identity only, without material bodies or questionnaires."""
    return json.loads(subprocess.check_output(["jq", expression, str(path)], text=True))


def prepare(out):
    """Bind the entire registered panel to SOURCE-only continuous observations."""
    if out.exists():
        raise FileExistsError("Do not replace a frozen packet or renew its budget")
    cfg = R.read(ROOT / CONFIG)
    source, identity, events, behavior, material = [
        ROOT / cfg[k]
        for k in (
            "source_parent",
            "identity_parent",
            "event_parent",
            "behavior_parent",
            "material",
        )
    ]
    R.check(source, ("computed_seal.json",))
    R.check(behavior, ("analyzed_seal.json", "verification_seal.json"))
    bank = selected(
        source / "source_bundle.npz", ("source_means", "codes", "anchors", "variance")
    )
    ids = selected(identity / "observations.npz", ("source_indices", "panel_indices"))
    cached = selected(material / "private_arrays.npz", ("author_ids", "comments"))
    if (
        ids["source_indices"].shape != (386,)
        or ids["panel_indices"].shape != (24,)
        or np.intersect1d(ids["source_indices"], ids["panel_indices"]).size
        or cached["comments"].shape != (624, 36, 1024)
        or float(bank["variance"]) != cfg["variance"]
    ):
        raise ValueError("Changed full source/panel scope or inherited variance")
    metadata = jq(
        material / "private_material.json",
        "[.bundles[] | {author_id,event_ids:[.views[0:2][][] | .comment_id]}]",
    )
    ledger = jq(
        events / "private_bindings.json",
        "[.[] | {unit_id,author,context_window,own:{author_id:.own.author_id,event_id:.own.comment_id},foreign:{author_id:.foreign.author_id,event_id:.foreign.comment_id}}]",
    )
    if (
        len(ledger) != 48
        or len(metadata) != 624
        or any(
            row["author_id"] != str(cached["author_ids"][i])
            or len(row["event_ids"]) != 24
            for i, row in enumerate(metadata)
        )
    ):
        raise ValueError("Numeric/source event identity differs")
    source_comments = cached["comments"][ids["source_indices"], :24].astype(float)
    source_comments /= np.linalg.norm(source_comments, axis=-1, keepdims=True)
    if not np.array_equal(
        source_comments.reshape(386, 2, 12, 1024).mean(2), bank["source_means"]
    ):
        raise ValueError(
            "SOURCE means and comment observations use different encodings"
        )
    panel = ids["panel_indices"]
    addresses = {}
    for original in panel:
        row = metadata[int(original)]
        for slot, eid in enumerate(row["event_ids"]):
            key = (row["author_id"], eid)
            if key in addresses:
                raise ValueError("Duplicated source event identity")
            addresses[key] = (int(original), slot)
    points = []
    for i, row in enumerate(ledger):
        if (row["unit_id"], row["author"], row["context_window"]) != (
            i,
            i % 24,
            i // 24,
        ):
            raise ValueError("Panel use order changed")
        pair = []
        for endpoint in ("own", "foreign"):
            event = row[endpoint]
            original, slot = addresses[(event["author_id"], event["event_id"])]
            if endpoint == "own" and original != int(panel[row["author"]]):
                raise ValueError("Own evidence has a foreign author")
            if endpoint == "foreign" and original == int(panel[row["author"]]):
                raise ValueError("A stranger must not be the same author")
            v = cached["comments"][original, slot].astype(float)
            pair.append(v / np.linalg.norm(v))
        points.append(pair)
    old = R.read(behavior / "source_actions.json")
    fixed_behavior = R.read(behavior / "behavior.json")
    directions = np.asarray(R.read(ROOT / cfg["directions"])["directions"])
    M.validate(source_comments, bank["codes"], bank["anchors"], float(bank["variance"]))
    null_order = np.random.default_rng(cfg["null_seed"]).permutation(386)
    out.mkdir(parents=True)
    inputs = [
        source / "source_bundle.npz",
        source / "computed_seal.json",
        identity / "observations.npz",
        events / "private_bindings.json",
        material / "private_arrays.npz",
        material / "private_material.json",
        behavior / "source_actions.json",
        behavior / "behavior.json",
        behavior / "verification_seal.json",
        ROOT / cfg["directions"],
    ]
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
        out / "source_bundle.npz",
        source_comments=source_comments,
        codes=bank["codes"],
        anchors=bank["anchors"],
        variance=bank["variance"],
        directions=directions,
    )
    R.archive(
        out / "observations.npz",
        points=np.asarray(points),
        behavior=np.asarray(fixed_behavior["sum"]["balanced_key"]),
        old_delta=np.asarray(old["changes"]["actual"]["full"]),
        source_indices=ids["source_indices"],
        panel_indices=panel,
        null_order=null_order,
    )
    R.write(out / "event_ledger.json", ledger)
    R.write(
        out / "input_quality.json",
        {
            "SOURCE_authors": 386,
            "SOURCE_comment_prototypes": 9264,
            "panel_authors": 24,
            "same_event_uses": 48,
            "source_panel_overlap": 0,
            "historical_reused_not_lockbox": True,
            "source_mean_readback_error": 0,
            "direction_Gram_identity_error": float(
                np.max(abs(directions.T @ directions - np.eye(15)))
            ),
            "observation_covariance_preserved_not_isotropic_GRAIN": True,
            "metadata_fields": ["author_id", "event_id"],
            "new_neural_API_raw_text_held_labels_private_training_calls": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "source_bundle.npz",
            "observations.npz",
            "event_ledger.json",
            "input_quality.json",
        ),
    )
    print("PREPARED HM267:386 SOURCE/9264 comments/SAME48 event uses", flush=True)


def compute(out):
    """One finite pass; no target/outcome fit and no model or parameter selection."""
    cfg = R.check(out)["config"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "cap": cfg["compute_seconds_cap"]},
    )
    start = time.perf_counter()
    try:
        z = R.prepared(out)
        bank = selected(
            out / "source_bundle.npz",
            ("source_comments", "codes", "anchors", "variance", "directions"),
        )
        actual = M.score_events(bank, z["points"].reshape(96, 1024))
        null_bank = dict(bank)
        null_bank.update(
            codes=bank["codes"][z["null_order"]],
            anchors=bank["anchors"][z["null_order"]],
        )
        null = M.score_events(null_bank, z["points"].reshape(96, 1024))
        arrays, rows, diagonals = {}, [], {}
        for assignment, scored in (("actual", actual), ("fixed_null", null)):
            for key, reading in scored.items():
                for kind in READINGS:
                    values = reading[kind].reshape(48, 2, 5)
                    arrays[f"{assignment}/{key}/{kind}"] = values
                    delta = values[:, 0] - values[:, 1]
                    for order, ix in (
                        ("pooled", np.arange(48)),
                        ("window0", np.arange(24)),
                        ("window1", np.arange(24, 48)),
                    ):
                        contact = B.contacts(delta[ix], z["behavior"][ix])
                        for p in range(5):
                            for q in range(5):
                                rows.append(
                                    {
                                        "assignment": assignment,
                                        "field": key,
                                        "reading": kind,
                                        "use": order,
                                        "SOURCE_target": M.TARGETS[p],
                                        "behavior_target": M.TARGETS[q],
                                        "action_contact": float(
                                            contact["action"][p, q]
                                        ),
                                        "centered_covariance": float(
                                            contact["covariance"][p, q]
                                        ),
                                        "mean_product": float(contact["means"][p, q]),
                                    }
                                )
                    if kind == "probability":
                        ci = B.paired_intervals(
                            delta.reshape(2, 24, 5),
                            z["behavior"].reshape(2, 24, 5),
                            cfg["bootstrap_seed"],
                            cfg["bootstrap_draws"],
                        )
                        diagonals[f"{assignment}/{key}"] = {
                            "pooled": np.diag(
                                B.contacts(delta, z["behavior"])["action"]
                            ).tolist(),
                            "window0": np.diag(
                                B.contacts(delta[:24], z["behavior"][:24])["action"]
                            ).tolist(),
                            "window1": np.diag(
                                B.contacts(delta[24:], z["behavior"][24:])["action"]
                            ).tolist(),
                            "marginal_author_bootstrap_interval": ci.tolist(),
                        }
                arrays[f"{assignment}/{key}/profile_mass"] = reading[
                    "profile_mass"
                ].reshape(48, 2, 32)
        old_zero = np.max(abs(z["old_delta"]), axis=1) == 0
        separation = {}
        for key in actual:
            delta = (
                arrays[f"actual/{key}/probability"][:, 0]
                - arrays[f"actual/{key}/probability"][:, 1]
            )
            separation[key] = {
                "old_aliases": int(old_zero.sum()),
                "new_distinct_probability_vectors": int(
                    np.count_nonzero(np.max(abs(delta[old_zero]), axis=1))
                ),
                "alias_probability_change_RMS": np.sqrt(
                    np.mean(delta[old_zero] ** 2, 0)
                ).tolist(),
                "alias_target_behavior_contact": np.diag(
                    B.contacts(delta[old_zero], z["behavior"][old_zero])["action"]
                ).tolist(),
            }
        elapsed = time.perf_counter() - start
        if elapsed > cfg["compute_seconds_cap"]:
            raise TimeoutError("Nonrenewable finite CPU cap exceeded")
        R.archive(out / "scores.npz", **arrays)
        R.table(out / "contacts.csv", rows)
        summary = {
            "state": "COMPUTED_NOJOB",
            "scientific_CPU_seconds": elapsed,
            "primary_diagonals": diagonals,
            "old_alias_separation": separation,
            "all_5x5_cells_and_signs_retained": True,
            "no_fit_or_parameter_search": True,
            "not_prediction_accuracy_or_new_math_theorem": True,
            "original_scope_complete": False,
        }
        R.write(out / "summary.json", summary)
        R.seal(
            out, "computed_seal.json", ("scores.npz", "contacts.csv", "summary.json")
        )
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        R.write(
            out / "failure.json",
            {"error": repr(exc), "restart_or_rescue_allowed": False},
        )
        raise


def verify(out):
    """Bounded alternate Gaussian accounts and one independent cold record readback."""
    R.check(out, ("computed_seal.json",))
    z = R.prepared(out)
    bank = selected(
        out / "source_bundle.npz",
        ("source_comments", "codes", "anchors", "variance", "directions"),
    )
    scores = selected(
        out / "scores.npz",
        (
            "actual/E5/comment_mixture/probability",
            "actual/GRAIN15/comment_mixture/probability",
        ),
    )
    error = 0.0
    bits = (bank["codes"][:, None] >> np.arange(5)) & 1
    for space in M.SPACES:
        q = M.projection(bank["directions"])
        source = (
            bank["source_comments"] @ q
            if space == "GRAIN15"
            else bank["source_comments"]
        )
        for i in (0, 48, 95):
            x = z["points"].reshape(96, 1024)[i]
            point = x @ q if space == "GRAIN15" else x
            direct = np.exp(
                -np.sum((source - point) ** 2, axis=-1) / (2 * float(bank["variance"]))
            ).mean(1)
            value = direct @ bits / direct.sum()
            saved = scores[f"actual/{space}/comment_mixture/probability"].reshape(
                96, 5
            )[i]
            error = max(error, float(np.max(abs(value - saved))))
    record = {
        "frame_key": "HM267_event0",
        "events": [{"event_id": "own0", "vector": z["points"][0, 0].tolist()}],
    }
    cold = M.cold_record(bank, record)
    for space in M.SPACES:
        saved = scores[f"actual/{space}/comment_mixture/probability"][0, 0]
        error = max(
            error,
            float(
                np.max(
                    abs(
                        np.asarray(
                            cold["readings"][f"{space}/comment_mixture"]["probability"]
                        )
                        - saved
                    )
                )
            ),
        )
    if error > 1e-10:
        raise ValueError("Alternate/cold numeric identity failed; do not rescore")
    R.write(
        out / "verification.json",
        {
            "state": "VERIFIED_NOJOB",
            "alternate_max_error": error,
            "fixed_direct_points": [0, 48, 95],
            "cold_event": "own0",
            "audit": "same_agent_not_blind",
            "no_scientific_rerun": True,
        },
    )
    R.write(out / "cold_input.json", record)
    R.write(out / "cold_readback.json", cold)
    R.seal(
        out,
        "verification_seal.json",
        ("verification.json", "cold_input.json", "cold_readback.json"),
    )
    print(json.dumps(R.read(out / "verification.json")), flush=True)


def main():
    """Execute one explicit stage, never an autonomous parameter queue."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("ALL_SEALS_PASS HM267")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
