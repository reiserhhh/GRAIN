#!/usr/bin/env python3
"""One finite SAME-native-family context-relation mechanism experiment."""

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
from grain_research.construction import math_context_relation as C
from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_joint_meaning as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_context_relation_hm178.json"
DEFAULT = ROOT / "results/grain_math_response_v1/context_relation_action_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CONTEXT_RELATION_PLAN_HM178.md",
    "grain_research/construction/math_context_relation.py",
    "scripts/run_grain_context_relation_hm178.py",
    "scripts/score_grain_context_relation_cold.py",
    "scripts/verify_grain_context_relation_hm178.py",
    "tests/test_grain_context_relation.py",
)
COMPONENTS = ("independent", "common", "full", "shared_change", "specific_change")


def prepare(out):
    """Freeze exact event joins and SOURCE-only fields before native comparisons."""
    start = time.perf_counter()
    spec = R.read(ROOT / CONFIG)
    pars = {
        k: ROOT / spec[k]
        for k in ("meaning_parent", "event_parent", "panel_parent", "native_parent")
    }
    stages = {
        "meaning_parent": ("scored_seal.json", "analysis_seal.json"),
        "event_parent": ("compiled_seal.json", "cold_seal.json", "analysis_seal.json"),
        "panel_parent": ("compiled_seal.json", "analysis_seal.json"),
        "native_parent": ("scored_seal.json", "analysis_seal.json"),
    }
    inputs, code = {}, {}
    for name, par in pars.items():
        m = R.check(par, stages[name])
        code.update(m["code"])
        for stage in ("manifest.json", "prepared_seal.json", *stages[name]):
            files = (
                (stage,) if stage == "manifest.json" else (stage, *R.read(par / stage))
            )
            for f in files:
                path = str((par / f).relative_to(ROOT))
                inputs[path] = R.digest(ROOT / path)
    a, e, old = (
        R.prepared(pars[k]) for k in ("panel_parent", "event_parent", "meaning_parent")
    )
    source = old["source"]
    if (
        len(source) != 386
        or len(a["author_ids"]) != 24
        or not np.array_equal(source, a["source_indices"])
        or set(source) & set(a["application_indices"])
    ):
        raise ValueError("Changed source/panel identities")
    inv = np.argsort(e["original_units"], axis=1)
    coords = np.take_along_axis(e["coordinates"], inv[:, :, None], axis=1)
    cond = np.take_along_axis(e["conditions"], inv, axis=1)
    threshold = R.read(pars["event_parent"] / "source_law.json")["thresholds"]
    state = R.F.states(coords.reshape(-1, 15), threshold).reshape(624, 36)
    cells = 4 * cond + state
    mass = np.zeros((386, 2, 2, 8))
    for u, idx in enumerate(source):
        for v in range(2):
            for h in range(2):
                ids = np.arange(6) + 12 * v + 6 * h
                mass[u, v, h] = np.bincount(cells[idx, ids], minlength=8) / 6
    panel_cells = np.empty((2, 24, 2, 12), int)
    err = 0.0
    for u, idx in enumerate(a["application_indices"]):
        for v in range(2):
            ids = np.arange(12) + 12 * v
            err = max(
                err, float(np.max(np.abs(coords[idx, ids] - a["coordinates"][u, 1, v])))
            )
            for arm in range(2):
                panel_cells[arm, u, v] = 4 * a["conditions"][u, v] + R.F.states(
                    a["coordinates"][u, arm, v], threshold
                )
            if not np.array_equal(panel_cells[1, u, v], cells[idx, ids]):
                raise ValueError("Raw cell/event identity mismatch")
    if err != 0:
        raise ValueError("Raw coordinates do not match the SAME event frame")
    b = C.build_bundle(
        mass, old["source_targets"], spec["ipf_tolerance"], spec["ipf_max_sweeps"]
    )
    C.validate_bundle(b)
    if len(b["profiles"]) != 32:
        raise ValueError("Full target-profile scope changed")
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": code | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
            "cohort_reuse": "Historical native panel/directions;not new lockbox",
        },
    )
    R.write(out / "source_bundle.json", b)
    R.archive(
        out / "observations.npz",
        source=source,
        source_mass=mass,
        source_targets=old["source_targets"],
        panel_cells=panel_cells,
        author_ids=a["author_ids"],
        application_indices=a["application_indices"],
        selected=a["selected"],
        control=a["control"],
        splits=a["splits"],
    )
    count = 0
    with (out / "cold_input.jsonl").open("x") as sink:
        for arm, u, v in np.ndindex(2, 24, 2):
            for k, ids in enumerate(
                J.group_indices(a["selected"][u, v], a["control"][u, v])
            ):
                r = {
                    "author_key": f"{arm}/{u}/{v}/{k}",
                    "view": v,
                    "events": [
                        {
                            "event_id": f"{a['application_indices'][u]}/{12 * v + int(i)}",
                            "slot": int(i),
                            "cell": int(panel_cells[arm, u, v, i]),
                        }
                        for i in ids
                    ],
                }
                sink.write(json.dumps(r) + "\n")
                count += 1
    quality = {
        "source_authors": 386,
        "profiles": 32,
        "panel_authors": 24,
        "views": 48,
        "cold_inputs": count,
        "raw_coordinate_join_error": err,
        "old_17_plus_17_incomparable_full_native_views": 48,
        "new_declared_family": "original12 slots;disjoint first6/last6 per view",
        "mask_cell_changes": int(np.count_nonzero(panel_cells[0] != panel_cells[1])),
        "new_neural_private_label_reads": 0,
        "native_outcomes_read": False,
        "solver": b["solver_receipts"],
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        (
            "source_bundle.json",
            "observations.npz",
            "cold_input.jsonl",
            "input_quality.json",
        ),
    )
    print(json.dumps(quality))


def score(out):
    """Cold standalone process receives no native outputs or panel labels."""
    R.check(out)
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_context_relation_cold.py"),
            "--bundle",
            str(out / "source_bundle.json"),
            "--input",
            str(out / "cold_input.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        check=True,
    )
    shape = (2, 24, 2, 5, 3, 5)
    mass, mean = np.full(shape, np.nan), np.full(shape, np.nan)
    support, pairs = np.zeros(shape, bool), np.zeros(shape[:4], int)
    seen = set()
    for line in (out / "cold_output.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = tuple(map(int, r["author_key"].split("/")))
        if key in seen or r["view"] != key[2]:
            raise ValueError("Cold identity mismatch")
        seen.add(key)
        mass[key], mean[key], support[key], pairs[key] = (
            r["pair_mass_readout"],
            r["conditional_pair_average"],
            r["supported"],
            r["pairs"],
        )
    if len(seen) != 480:
        raise ValueError("Cold family incomplete")
    R.archive(
        out / "cold_accounts.npz",
        mass=mass,
        conditional_mean=mean,
        supported=support,
        pairs=pairs,
    )
    receipt = {
        "inputs": 480,
        "values": mass.size,
        "supported": int(support.sum()),
        "no_pair_inputs": int(np.count_nonzero(pairs == 0)),
        "native_fit": False,
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "cold_receipt.json", receipt)
    R.seal(
        out,
        "scored_seal.json",
        ("cold_output.jsonl", "cold_accounts.npz", "cold_receipt.json"),
    )
    print(json.dumps(receipt))


def analyze(out):
    """Describe same-event actions;never select a predictor or calibrate scale."""
    start = time.perf_counter()
    spec = R.check(out, ("scored_seal.json",))["config"]
    z, b = R.prepared(out), C.validate_bundle(R.read(out / "source_bundle.json"))
    with np.load(out / "cold_accounts.npz", allow_pickle=False) as a:
        mass, pair_counts = a["mass"], a["pairs"]
    native = ROOT / spec["native_parent"]
    nobs = R.prepared(native)
    if (
        not np.array_equal(nobs["author_ids"], z["author_ids"])
        or not np.array_equal(nobs["selected"], z["selected"])
        or not np.array_equal(nobs["control"], z["control"])
        or not np.array_equal(nobs["cells"], z["panel_cells"])
    ):
        raise ValueError("Native evidence sets/cells changed")
    with np.load(native / "composition_accounts.npz", allow_pickle=False) as a:
        logits, scores = a["native_logits"], a["native_scores"]
        old_j = a["native_log_odds_J_sum"]
    odds = logits[..., 1:] - logits[..., :1]
    nj = C.context_action(np.moveaxis(odds, 2, 0))
    score_j = C.context_action(np.moveaxis(scores, 2, 0))
    if not np.allclose(nj, old_j, rtol=0, atol=2e-12):
        raise ValueError("Historical nine native context actions changed")
    action = C.context_action(np.moveaxis(mass, 3, 0))
    count_action = C.context_action(np.moveaxis(pair_counts, 3, 0))
    opportunity = count_action[..., None] * np.asarray(b["prior"]) / 36
    specific, shared = (
        action[..., 2, :] - action[..., 1, :],
        action[..., 1, :] - action[..., 0, :],
    )
    comp = np.stack(
        [action[..., 0, :], action[..., 1, :], action[..., 2, :], shared, specific],
        axis=-2,
    )
    contact = np.empty_like(action)
    f = np.asarray(b["pair_fields"], float)
    for arm, u, v in np.ndindex(2, 24, 2):
        contact[arm, u, v] = C.contact_action(
            f[v], z["panel_cells"][arm, u, v], z["selected"][u, v], z["control"][u, v]
        )
    finite = np.isfinite(action) & np.isfinite(contact)
    contact_error = float(np.max(np.abs(action[finite] - contact[finite])))
    rows, metrics = [], []
    for arm, u, v, k, q in np.ndindex(2, 24, 2, 3, 5):
        rows.append(
            {
                "arm": ("cue_mask", "raw")[arm],
                "author_index": int(z["application_indices"][u]),
                "view": v,
                "model": M.MODELS[k],
                "target": spec["targets"][q],
                "context_action": finite_value(action[arm, u, v, k, q]),
                "opportunity_action": float(opportunity[arm, u, v, q]),
                "centered_action": finite_value(
                    action[arm, u, v, k, q] - opportunity[arm, u, v, q]
                ),
                "direct_contact": finite_value(contact[arm, u, v, k, q]),
                "native_ordinal_action": float(score_j[u, v]),
            }
        )
    endpoints = np.concatenate([score_j[..., None], nj], axis=-1)
    for arm, k, q, digit in np.ndindex(2, 5, 5, 10):
        for group in ("author_mean", "view0", "view1"):
            x, y = comp[arm, :, :, k, q], endpoints[:, :, digit]
            if group == "author_mean":
                x, y = x.mean(1), y.mean(1)
            else:
                v = int(group[-1])
                x, y = x[:, v], y[:, v]
            metrics.append(
                {
                    "arm": ("cue_mask", "raw")[arm],
                    "component": COMPONENTS[k],
                    "target": spec["targets"][q],
                    "native_endpoint": "ordinal" if digit == 0 else f"digit{digit}_vs0",
                    "group": group,
                    **C.signed_agreement(x, y),
                }
            )
    R.table(out / "context_actions.csv", rows)
    R.table(out / "signed_native_links.csv", metrics)
    R.archive(
        out / "mechanism_accounts.npz",
        actions=action,
        contacts=contact,
        components=comp,
        opportunity=opportunity,
        native_log_odds_actions=nj,
        native_ordinal_actions=score_j,
    )
    main = [
        r
        for r in metrics
        if r["arm"] == "cue_mask"
        and r["group"] == "author_mean"
        and r["native_endpoint"] == "ordinal"
    ]
    summary = {
        "question": "SOURCE profile-specific relation versus native SAME-context action",
        "main_descriptive_links": main,
        "context_contact_error": contact_error,
        "native_replay_error": float(np.max(np.abs(nj - old_j))),
        "opportunity_nonzero_views": int(np.count_nonzero(count_action[0])),
        "specific_nonzero_author_target_cells": int(
            np.count_nonzero(specific[0].mean(1))
        ),
        "new_neural_private_API_train_reads": 0,
        "native_fit": False,
        "accuracy_promotion": False,
        "original_scope_complete": False,
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "context_actions.csv",
            "signed_native_links.csv",
            "mechanism_accounts.npz",
            "summary.json",
        ),
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "main_descriptive_links"})
    )


def finite_value(x):
    """Keep missing SOURCE support missing in JSON/CSV."""
    return float(x) if np.isfinite(x) else None


def main():
    """Run an explicit once-only stage without any autonomous retry queue."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "score", "analyze"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    {"prepare": prepare, "score": score, "analyze": analyze}[a.stage](a.out)


if __name__ == "__main__":
    main()
