#!/usr/bin/env python3
"""One actual cross-window author-relation discriminator and cold construction."""

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
from grain_research.construction import math_cross_window as W
from grain_research.construction import math_target_profile as P
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_trait_anchor_hm171 as T

CONFIG = "configs/grain_cross_window_hm176.json"
DEFAULT = ROOT / "results/grain_math_response_v1/cross_window_profile_law_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CROSS_WINDOW_PLAN_HM176.md",
    "grain_research/construction/math_cross_window.py",
    "scripts/run_grain_cross_window_hm176.py",
    "scripts/score_grain_cross_window_cold.py",
    "scripts/verify_grain_cross_window_hm176.py",
    "tests/test_grain_cross_window.py",
)


def prepare(out):
    """Authenticate old disjoint observations and compile only SOURCE labels."""
    start = time.perf_counter()
    spec = R.read(ROOT / CONFIG)
    parent, event_parent, mass_parent, index_parent = (
        ROOT / spec[k]
        for k in ("parent", "event_parent", "mass_parent", "index_parent")
    )
    manifest = R.check(parent, ("scored_seal.json", "analysis_seal.json"))
    R.check(event_parent, T.STAGES)
    T.check(mass_parent, ("compiled_seal.json", "analysis_seal.json"))
    R.check(index_parent, ("compiled_seal.json", "analysis_seal.json"))
    old_bundle = P.validate_bundle(R.read(parent / "source_bundle.json"))
    with np.load(index_parent / "observations.npz", allow_pickle=False) as a:
        source = a["source_indices"]
    z, mass = R.prepared(event_parent), T.observations(mass_parent)
    held = mass["held"]
    if (
        len(source) != spec["source_authors"]
        or len(held) != spec["application_authors"]
        or set(source) & set(held)
    ):
        raise ValueError("Changed or overlapping source/application author roles")
    pair = z["pairs"]
    state = R.F.states(
        z["coordinates"].reshape(-1, 15),
        R.read(event_parent / "source_law.json")["thresholds"],
    ).reshape(624, 36)
    cells, units = np.empty((624, 2, 17), int), np.empty((624, 2, 17), int)
    for u in range(624):
        for w in range(2):
            t = pair[(pair[:, 0] == u) & (pair[:, 1] == w), 2]
            if len(t) != spec["events_per_window"]:
                raise ValueError("Changed event window size")
            cells[u, w] = 4 * z["conditions"][u, t] + state[u, t]
            units[u, w] = z["original_units"][u, t]
        if set(units[u, 0]) & set(units[u, 1]):
            raise ValueError("Same event copied into both windows")
    direct_mass = np.stack(
        [
            np.stack([np.bincount(c, minlength=8) / 17 for c in window])
            for window in cells
        ]
    )
    p = mass["mass"][source]
    old_p, y = (
        np.asarray(old_bundle["parent_bundle"][k])
        for k in ("source_mass", "source_binary_targets")
    )
    mass_error = float(np.max(np.abs(direct_mass - mass["mass"])))
    pooled_error = float(np.max(np.abs(p.mean(1) - old_p)))
    if max(mass_error, pooled_error) > 2e-12 or not np.all(mass["counts"] == 17):
        raise ValueError("Old single-window or pooled observations disagree")
    bundle = W.build_bundle(p, y)
    inputs = dict(manifest["inputs"])
    for par, stages in (
        (parent, ("scored_seal.json", "analysis_seal.json")),
        (event_parent, T.STAGES),
        (mass_parent, ("compiled_seal.json", "analysis_seal.json")),
        (index_parent, ("compiled_seal.json", "analysis_seal.json")),
    ):
        for stage in ("manifest.json", "prepared_seal.json", *stages):
            files = (
                (stage,) if stage == "manifest.json" else (stage, *R.read(par / stage))
            )
            for name in files:
                path = str((par / name).relative_to(ROOT))
                inputs[path] = R.digest(ROOT / path)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": manifest["code"] | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
            "cohort_reuse": "historical numerical cohort;not new representation holdout/lockbox",
        },
    )
    R.write(out / "source_bundle.json", bundle)
    R.archive(
        out / "observations.npz",
        source=source,
        held=held,
        source_mass=p,
        source_targets=y,
        source_cells=cells[source],
        source_units=units[source],
        held_cells=cells[held],
        held_units=units[held],
    )
    with (out / "cold_input.jsonl").open("x") as sink:
        for u in held:
            row = {
                "author_key": f"held/{u}",
                "events": [
                    {
                        "event_id": f"{u}/{units[u, w, j]}",
                        "window": w,
                        "cell": int(cells[u, w, j]),
                    }
                    for w in range(2)
                    for j in range(17)
                ],
            }
            sink.write(json.dumps(row) + "\n")
    quality = {
        "source_authors": len(source),
        "application_authors": len(held),
        "observed_profiles": len(bundle["profile_counts"]),
        "profile_count_min": min(bundle["profile_counts"]),
        "profile_count_max": max(bundle["profile_counts"]),
        "events_per_window": 17,
        "source_cross_pairs": len(source) * 17 * 17,
        "application_cross_pairs": len(held) * 17 * 17,
        "window_mass_error": mass_error,
        "pooled_mass_error": pooled_error,
        "event_overlap": 0,
        "source_application_author_overlap": 0,
        "new_neural_or_label_file_reads": 0,
        "application_labels_read": False,
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
    """Score all historical held observations with no target/outcome input."""
    R.check(out)
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_cross_window_cold.py"),
            "--bundle",
            str(out / "source_bundle.json"),
            "--input",
            str(out / "cold_input.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        check=True,
    )
    rows = [
        json.loads(line)
        for line in (out / "cold_output.jsonl").read_text().splitlines()
    ]
    z = R.prepared(out)
    if [r["author_key"] for r in rows] != [f"held/{u}" for u in z["held"]] or any(
        r["events_per_window"] != [17, 17] for r in rows
    ):
        raise ValueError("Cold output identity/window changed")
    R.archive(
        out / "cold_accounts.npz",
        readout=np.asarray([r["pair_target_readout"] for r in rows], float),
        supported=np.asarray([r["supported"] for r in rows], bool),
    )
    R.write(
        out / "cold_receipt.json",
        {
            "rows": len(rows),
            "seconds": time.perf_counter() - start,
            "new_neural_calls": 0,
            "application_outcomes": False,
        },
    )
    R.seal(
        out,
        "scored_seal.json",
        ("cold_output.jsonl", "cold_accounts.npz", "cold_receipt.json"),
    )
    print(json.dumps(R.read(out / "cold_receipt.json")))


def rms(x):
    """Summarize signed accounts without selecting a favorable direction."""
    return float(np.sqrt(np.mean(np.square(x))))


def analyze(out):
    """Find recurrent relations and their choice/response makeup, not accuracy."""
    start = time.perf_counter()
    spec = R.check(out, ("scored_seal.json",))["config"]
    z = R.prepared(out)
    p, y = z["source_mass"], z["source_targets"]
    bundle = W.validate_bundle(R.read(out / "source_bundle.json"))
    law = np.asarray(bundle["joint_laws"])
    center = W.centered_accounts(p, y)
    condition = W.condition_accounts(law[2], law[1])
    observed, null, tail = W.matched_reference(
        p, y, spec["permutations"], spec["permutation_seed"]
    )
    prior = y.mean(0)
    delta, signed_delta = law[2] - law[1], W.signed(law[2] - law[1], prior)
    parent_bundle = R.read(ROOT / spec["parent"] / "source_bundle.json")
    old_within = P.two_event_accounts(parent_bundle)["within_profile"]
    identity = {
        "average_variance": float(
            np.max(
                np.abs(
                    (center["v0"] + center["v1"]) / 2
                    - center["common"]
                    - center["change"]
                )
            )
        ),
        "cross_symmetric": float(
            np.max(
                np.abs(center["cross_symmetric"] - center["common"] + center["change"])
            )
        ),
        "old_within_replay": float(np.max(np.abs(old_within - center["common"]))),
        "choice_response_reconstruction": float(
            np.max(
                np.abs(
                    delta
                    - condition["choice"]
                    - condition["response"]
                    - condition["unresolved"]
                )
            )
        ),
    }
    for k, name in enumerate(W.MODELS):
        early = np.array(
            [p[y[:, q] == g, 0].sum(0) / len(p) for q in range(5) for g in (0, 1)]
        ).reshape(5, 2, 8)
        late = np.array(
            [p[y[:, q] == g, 1].sum(0) / len(p) for q in range(5) for g in (0, 1)]
        ).reshape(5, 2, 8)
        identity[name + "_marginal"] = float(
            max(
                np.max(np.abs(law[k].sum(-1) - early)),
                np.max(np.abs(law[k].sum(-2) - late)),
            )
        )
    if max(identity.values()) > 2e-12:
        raise ValueError("Observation identity failed")
    joint_rows = [
        {
            "model": name,
            "trait": spec["traits"][q],
            "group": g,
            "early_cell": a,
            "late_cell": b,
            "joint_mass": float(law[k, q, g, a, b]),
        }
        for k, name in enumerate(W.MODELS)
        for q in range(5)
        for g in (0, 1)
        for a in range(8)
        for b in range(8)
    ]
    R.table(out / "joint_law_accounts.csv", joint_rows)
    names = ("all_profiles", *spec["traits"])
    R.table(
        out / "matched_reference.csv",
        [
            {
                "statistic": name,
                "observed_frobenius": float(observed[i]),
                "null_mean": float(null[:, i].mean()),
                "null_max": float(null[:, i].max()),
                "tail_rank": float(tail[i]),
                "permutations": len(null),
            }
            for i, name in enumerate(names)
        ],
    )
    cr = []
    for q, trait in enumerate(spec["traits"]):
        for g in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    sl = (q, g, slice(4 * a, 4 * a + 4), slice(4 * b, 4 * b + 4))
                    c, r, u = (
                        condition[k][sl] for k in ("choice", "response", "unresolved")
                    )
                    cr.append(
                        {
                            "trait": trait,
                            "group": g,
                            "early_condition": a,
                            "late_condition": b,
                            "conditional_supported": bool(
                                condition["conditional_supported"][q, g, a, b]
                            ),
                            "choice_norm_squared": float(np.sum(c * c)),
                            "response_norm_squared": float(np.sum(r * r)),
                            "signed_cross_twice": float(2 * np.sum(c * r)),
                            "unresolved_norm_squared": float(np.sum(u * u)),
                            "total_norm_squared": float(np.sum(delta[sl] ** 2)),
                        }
                    )
    R.table(out / "condition_accounts.csv", cr)
    with np.load(out / "cold_accounts.npz", allow_pickle=False) as a:
        readout, support = a["readout"], a["supported"]
    cold_rows, differences = [], {}
    for q, trait in enumerate(spec["traits"]):
        for k, name in enumerate(W.MODELS):
            for i, u in enumerate(z["held"]):
                cold_rows.append(
                    {
                        "author_key": f"held/{u}",
                        "model": name,
                        "trait": trait,
                        "supported": bool(support[i, k, q]),
                        "pair_target_readout": float(readout[i, k, q])
                        if support[i, k, q]
                        else None,
                    }
                )
        ok = support[:, 1, q] & support[:, 2, q]
        differences[trait] = {
            "supported_authors": int(ok.sum()),
            "same_minus_other_rms": rms(readout[ok, 2, q] - readout[ok, 1, q])
            if ok.any()
            else None,
            "same_minus_other_mean": float(
                np.mean(readout[ok, 2, q] - readout[ok, 1, q])
            )
            if ok.any()
            else None,
        }
    R.table(out / "cold_readout_accounts.csv", cold_rows)
    R.archive(
        out / "mechanism_accounts.npz",
        joint_laws=law,
        same_other=delta,
        signed_same_other=signed_delta,
        **center,
        **condition,
        matched_observed=observed,
        matched_null=null,
        matched_tail=tail,
    )
    sym = center["cross_symmetric"][0].sum(0)
    summary = {
        "status": "FINITE_MECHANISM_COMPLETE_NOT_SCOPE_COMPLETE",
        "source_authors": len(p),
        "source_cross_pairs": len(p) * 17 * 17,
        "laws": list(W.MODELS),
        "identities": identity,
        "within_profile_cross_rms": rms(center["cross"]),
        "same_other_joint_rms": rms(delta),
        "common_rms": rms(center["common"]),
        "change_rms": rms(center["change"]),
        "signed_target_cross_rms": [rms(v) for v in W.signed(center["cross"], prior)],
        "cross_symmetric_eigenvalues": np.linalg.eigvalsh(sym).tolist(),
        "matched_tail_ranks": dict(zip(names, tail.tolist(), strict=True)),
        "condition_choice_rms": rms(condition["choice"]),
        "condition_response_rms": rms(condition["response"]),
        "condition_unresolved_rms": rms(condition["unresolved"]),
        "condition_signed_cross_twice": float(
            2 * np.sum(condition["choice"] * condition["response"])
        ),
        "unobserved_condition_blocks": int((~condition["conditional_supported"]).sum()),
        "cold_readout_differences": differences,
        "supported_cold_values": int(support.sum()),
        "cold_value_count": int(support.size),
        "application_labels_read": False,
        "accuracy_optimization": False,
        "psychological_cause_identified": False,
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "joint_law_accounts.csv",
            "matched_reference.csv",
            "condition_accounts.csv",
            "cold_readout_accounts.csv",
            "mechanism_accounts.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def main():
    """Run one explicit stage, never automatically advance a queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "score": score, "analyze": analyze}[args.stage](args.out)


if __name__ == "__main__":
    main()
