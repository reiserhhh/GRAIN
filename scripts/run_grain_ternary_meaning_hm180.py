#!/usr/bin/env python3
"""One finite source-qualified ternary meaning construction, not a precision grid."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_ternary_meaning as T
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_ternary_meaning_hm180.json"
DEFAULT = ROOT / "results/grain_math_response_v1/ternary_target_meaning_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TERNARY_MEANING_PLAN_HM180.md",
    "grain_research/construction/math_ternary_meaning.py",
    "scripts/run_grain_ternary_meaning_hm180.py",
    "scripts/score_grain_ternary_meaning_cold.py",
    "scripts/verify_grain_ternary_meaning_hm180.py",
    "tests/test_grain_ternary_meaning.py",
)


def prepare(out):
    """Freeze original event addresses and author folds before fitting any laws."""
    start, spec = time.perf_counter(), R.read(ROOT / CONFIG)
    parent, event = ROOT / spec["parent"], ROOT / spec["event_parent"]
    inputs, code = {}, {}
    for par, stages in (
        (parent, ("scored_seal.json", "analysis_seal.json")),
        (event, ("compiled_seal.json", "cold_seal.json", "analysis_seal.json")),
    ):
        m = R.check(par, stages)
        code.update(m["code"])
        for stage in ("manifest.json", "prepared_seal.json", *stages):
            files = (
                (stage,) if stage == "manifest.json" else (stage, *R.read(par / stage))
            )
            for f in files:
                path = str((par / f).relative_to(ROOT))
                inputs[path] = R.digest(ROOT / path)
    a, e = R.prepared(parent), R.prepared(event)
    inv = np.argsort(e["original_units"], axis=1)
    coords = np.take_along_axis(e["coordinates"], inv[:, :, None], axis=1)
    cond = np.take_along_axis(e["conditions"], inv, axis=1)
    threshold = R.read(event / "source_law.json")["thresholds"]
    cells = 4 * cond + R.F.states(coords.reshape(-1, 15), threshold).reshape(624, 36)
    source, y = a["source"], a["source_targets"]
    if len(source) != 386 or set(source) & set(a["application_indices"]):
        raise ValueError("Changed SOURCE author scope or panel overlap")
    sc = np.stack((cells[source, :12], cells[source, 12:24]), axis=1)
    mass = np.zeros((386, 2, 3, 8))
    for u, v, role in np.ndindex(386, 2, 3):
        mass[u, v, role] = (
            np.bincount(sc[u, v, 4 * role : 4 * role + 4], minlength=8) / 4
        )
    if not np.array_equal(
        a["panel_cells"][1],
        np.stack(
            (
                cells[a["application_indices"], :12],
                cells[a["application_indices"], 12:24],
            ),
            axis=1,
        ),
    ):
        raise ValueError("Raw original event identities changed")
    folds = T.source_folds(source, y)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": code | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
            "cohort_reuse": "Historical source directions/panel;not new lockbox or full-pipeline CV",
        },
    )
    R.archive(
        out / "observations.npz",
        source=source,
        source_targets=y,
        source_cells=sc,
        source_mass=mass,
        folds=folds,
        panel_cells=a["panel_cells"],
        author_ids=a["author_ids"],
        application_indices=a["application_indices"],
        selected=a["selected"],
        control=a["control"],
    )
    count = 0
    with (out / "cold_input.jsonl").open("x") as sink:
        for arm, u, v in np.ndindex(2, 24, 2):
            for k, ids in enumerate(
                J.group_indices(a["selected"][u, v], a["control"][u, v])
            ):
                record = {
                    "author_key": f"{arm}/{u}/{v}/{k}",
                    "view": v,
                    "events": [
                        {
                            "event_id": f"{a['application_indices'][u]}/{12 * v + int(i)}",
                            "slot": int(i),
                            "cell": int(a["panel_cells"][arm, u, v, i]),
                        }
                        for i in ids
                    ],
                }
                sink.write(json.dumps(record) + "\n")
                count += 1
    quality = {
        "source_authors": 386,
        "full_profiles": len(np.unique(y, axis=0)),
        "panel_authors": 24,
        "views": 2,
        "roles": [4, 4, 4],
        "source_triples": 386 * 2 * 64,
        "crossfit_fold_authors": np.bincount(folds).tolist(),
        "cold_inputs": count,
        "new_neural_calls": 0,
        "application_labels_or_native_answers": False,
        "original_slot_order_is_not_chronological": True,
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_input.jsonl", "input_quality.json"),
    )
    print(json.dumps(quality))


def compile_laws(out):
    """Compile full and fixed two-fold laws once, with no outcome-driven revision."""
    R.check(out)
    z, spec, start = R.prepared(out), R.read(ROOT / CONFIG), time.perf_counter()
    kwargs = {"tolerance": spec["ipf_tolerance"], "max_sweeps": spec["ipf_max_sweeps"]}
    b = T.build_bundle(z["source_mass"], z["source_targets"], **kwargs)
    T.validate_bundle(b)
    R.write(out / "source_bundle.json", b)
    for held in range(2):
        train = z["folds"] != held
        cf = T.build_bundle(
            z["source_mass"][train], z["source_targets"][train], **kwargs
        )
        if len(cf["profiles"]) != 32:
            raise ValueError("Fixed fold lost a full target profile")
        T.validate_bundle(cf)
        R.write(out / f"crossfit_bundle_{held}.json", cf)
    receipt = {
        "constructions": 3,
        "view_laws": 6,
        "condition_solvers": 48,
        "seconds": time.perf_counter() - start,
        "native_answer_fit": False,
    }
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_bundle.json",
            "crossfit_bundle_0.json",
            "crossfit_bundle_1.json",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def score(out):
    """Score unlabeled panel in an independent process, then held SOURCE authors."""
    R.check(out, ("compiled_seal.json",))
    start, z = time.perf_counter(), R.prepared(out)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_ternary_meaning_cold.py"),
            "--bundle",
            str(out / "source_bundle.json"),
            "--input",
            str(out / "cold_input.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        check=True,
    )
    shape = (2, 24, 2, 5, 2, 5)
    mass, means, triples = (
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.zeros(shape[:4], int),
    )
    seen = set()
    for line in (out / "cold_output.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = tuple(map(int, r["author_key"].split("/")))
        if key in seen or r["view"] != key[2]:
            raise ValueError("Cold evidence identity changed")
        seen.add(key)
        mass[key], means[key], triples[key] = (
            r["target_mass_readout"],
            r["conditional_target_average"],
            r["triples"],
        )
    if len(seen) != 480:
        raise ValueError("Cold scope incomplete")
    rows = []
    for held in range(2):
        b = R.read(out / f"crossfit_bundle_{held}.json")
        psi = np.array(
            [T.fields(b["profiles"], t) for t in np.asarray(b["profile_tables"], float)]
        )
        for u in np.flatnonzero(z["folds"] == held):
            for v in range(2):
                cells = z["source_cells"][u, v]
                values = np.array(
                    [
                        psi[v, :, :, cells[i], cells[j], cells[k]]
                        for i, j, k in itertools.product(
                            range(4), range(4, 8), range(8, 12)
                        )
                    ]
                )
                good = np.isfinite(values).all((1, 2))
                known = int(good.sum())
                for q, target in enumerate(T.TARGETS):
                    label = int(z["source_targets"][u, q])
                    loss = (
                        ((values[good, :, q] - label) ** 2).mean(0)
                        if known
                        else [None, None]
                    )
                    rows.append(
                        {
                            "author": int(z["source"][u]),
                            "held_fold": held,
                            "view": v,
                            "target": target,
                            "binary_source_label": label,
                            "supported_triples": known,
                            "total_triples": 64,
                            "complete": known == 64,
                            "reference_brier_on_common_support": loss[0],
                            "full_brier_on_common_support": loss[1],
                            "reference_minus_full": float(loss[0] - loss[1])
                            if known
                            else None,
                        }
                    )
    R.table(out / "source_crossfit_authors.csv", rows)
    R.archive(
        out / "cold_accounts.npz", mass=mass, conditional_mean=means, triples=triples
    )
    receipt = {
        "cold_inputs": len(seen),
        "target_values": int(mass.size),
        "crossfit_rows": len(rows),
        "source_authors": 386,
        "crossfit_triples": 49408,
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "scoring_receipt.json", receipt)
    R.seal(
        out,
        "scored_seal.json",
        (
            "cold_output.jsonl",
            "source_crossfit_authors.csv",
            "cold_accounts.npz",
            "scoring_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def analyze(out):
    """Decide source existence and held-author reuse separately, retaining negatives."""
    R.check(out, ("compiled_seal.json", "scored_seal.json"))
    z, b = R.prepared(out), R.read(out / "source_bundle.json")
    t = np.asarray(b["profile_tables"], float)
    stats = [T.statistics(b["profiles"], a) for a in t]
    with np.load(out / "cold_accounts.npz", allow_pickle=False) as a:
        values, triples = a["mass"], a["triples"]
    actions = (
        values[:, :, :, 2]
        - values[:, :, :, 1]
        - values[:, :, :, 3]
        + values[:, :, :, 4]
    )
    contacts = np.full(actions.shape, np.nan)
    for arm, u, v in np.ndindex(2, 24, 2):
        contacts[arm, u, v] = T.contact_action(
            T.fields(b["profiles"], t[v]),
            z["panel_cells"][arm, u, v],
            z["selected"][u, v],
            z["control"][u, v],
        )
    good = np.isfinite(actions) & np.isfinite(contacts)
    contact_error = (
        float(np.max(np.abs(actions[good] - contacts[good]))) if good.any() else None
    )
    with (out / "source_crossfit_authors.csv").open() as stream:
        raw = list(csv.DictReader(stream))
    aggregate = []
    for held, v, target in itertools.product((-1, 0, 1), range(2), T.TARGETS):
        selected = [
            r
            for r in raw
            if r["view"] == str(v)
            and r["target"] == target
            and (held == -1 or r["held_fold"] == str(held))
        ]
        complete = [r for r in selected if r["complete"] == "True"]
        d = np.array([float(r["reference_minus_full"]) for r in complete])
        aggregate.append(
            {
                "held_fold": held,
                "view": v,
                "target": target,
                "authors": len(selected),
                "complete_authors": len(complete),
                "incomplete_authors": len(selected) - len(complete),
                "mean_brier_reference_minus_full_complete": float(d.mean())
                if len(d)
                else None,
                "positive_complete_authors": int(np.sum(d > 0)),
                "negative_complete_authors": int(np.sum(d < 0)),
                "min_supported_triples": min(
                    int(r["supported_triples"]) for r in selected
                ),
            }
        )
    R.table(out / "source_crossfit_summary.csv", aggregate)
    R.archive(out / "mechanism_accounts.npz", actions=actions, contacts=contacts)
    summary = {
        "source_mechanisms": stats,
        "crossfit": aggregate,
        "contact_error": contact_error,
        "contact_comparisons": int(good.sum()),
        "unsupported_actions": int((~np.isfinite(actions)).sum()),
        "no_triple_inputs": int(np.sum(triples == 0)),
        "panel_labels_read": False,
        "native_outputs_read": False,
        "original_scope_complete": False,
        "new_capability": "coherent_source_target_triple_observer_preserving_all_profile_pair_readouts",
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        ("source_crossfit_summary.csv", "mechanism_accounts.npz", "summary.json"),
    )
    print(json.dumps(summary))


def main():
    """Run one selected finite stage; preserve partial results on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compile", "score", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        {
            "prepare": prepare,
            "compile": compile_laws,
            "score": score,
            "analyze": analyze,
        }[args.stage](args.out)
    except Exception as error:
        if args.out.exists() and not (args.out / f"{args.stage}_failure.json").exists():
            R.write(
                args.out / f"{args.stage}_failure.json",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retry": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
