#!/usr/bin/env python3
"""Execute one actual-history memory discriminator,not a predictive search."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_observed_path_memory as M
from scripts import run_grain_author_observation_channel_hm234 as H
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_observed_path_memory_hm235.json"
DEFAULT = ROOT / "results/grain_math_response_v1/observed_path_memory_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_OBSERVED_PATH_MEMORY_PLAN_HM235.md",
    "grain_research/construction/math_observed_path_memory.py",
    "grain_research/construction/math_joint_meaning.py",
    "grain_research/construction/math_joint_evidence.py",
    "scripts/run_grain_observed_path_memory_hm235.py",
    "tests/test_grain_observed_path_memory.py",
)
STAGES = ("computed_seal.json", "cold_seal.json", "verification_seal.json")


def clean(value):
    """Serialize missing numbers as null,not neutral or successful measurements."""
    a = np.asarray(value)
    return np.where(np.isfinite(a), a, None).tolist()


def prepare(out):
    """Bind real paths and old SOURCE assignments before computing new outcomes."""
    cfg = R.read(ROOT / CONFIG)
    source = ROOT / cfg["source_parent"]
    paths = ROOT / cfg["paths_parent"]
    events = ROOT / cfg["event_parent"]
    H.check(source, ("cold_seal.json", "analyzed_seal.json", "verification_seal.json"))
    files = []
    for parent, names in (
        (source, ("observations.npz", "source_bundles.json")),
        (paths, ("observations.npz",)),
        (events, ("observations.npz",)),
    ):
        S.parent_seal(parent, "prepared_seal.json", names)
        files.extend(
            str((parent / name).relative_to(ROOT))
            for name in ("prepared_seal.json",) + names
        )
    p = S.selected(paths / "observations.npz", ("paths", "held"))
    e = S.selected(events / "observations.npz", ("times", "original_units"))
    ix = S.selected(source / "observations.npz", ("source_indices",))["source_indices"]
    bundles = R.read(source / "source_bundles.json")
    mass, y = J.validate_bundle(bundles["actual"])
    _, shuffled = J.validate_bundle(bundles["permuted"])
    if (
        p["paths"].shape != (624, 2, 18)
        or ix.shape != (386,)
        or p["held"].shape != (120,)
        or np.intersect1d(ix, p["held"]).size
        or e["times"].shape != (624, 36)
        or np.any(np.diff(e["times"], axis=1) <= 0)
    ):
        raise ValueError("Changed source/held identity or strict chronology")
    replay = np.array([J.counts(row[:, :17].ravel()) / 34 for row in p["paths"][ix]])
    if np.max(abs(replay - mass)) > 2e-12:
        raise ValueError("SOURCE cells are not the inherited author observation family")
    bank = np.r_[np.zeros(203, int), np.ones(183, int)]
    held = p["held"]
    held_times = e["times"][held].reshape(120, 2, 18)
    held_units = e["original_units"][held].reshape(120, 2, 18)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {name: R.digest(ROOT / name) for name in SCIENCE},
            "inputs": {name: R.digest(ROOT / name) for name in files},
            "original_scope_complete": False,
            "cohort_reuse": "historical representation/cohorts,not an untouched lockbox",
        },
    )
    R.archive(
        out / "observations.npz",
        source_paths=p["paths"][ix],
        targets=np.stack([y, shuffled]),
        bank=bank,
        source_indices=ix,
        held=held,
        held_paths=p["paths"][held],
        held_times=held_times,
        held_units=held_units,
    )
    with (out / "cold_inputs.jsonl").open("x") as f:
        for u, author in enumerate(held):
            for w in range(2):
                record = {
                    "author_key": f"held/{int(author)}/window/{w}",
                    "events": [
                        {
                            "event_id": f"event/{int(author)}/{int(held_units[u, w, t])}",
                            "timestamp": float(held_times[u, w, t]),
                            "cell": int(p["paths"][author, w, t]),
                        }
                        for t in range(18)
                    ],
                }
                f.write(json.dumps(record, allow_nan=False) + "\n")
    quality = {
        "source_authors": 386,
        "bank_sizes": np.bincount(bank).tolist(),
        "source_triples": 386 * 2 * 16,
        "held_authors": 120,
        "held_triples": 120 * 2 * 16,
        "source_held_overlap": 0,
        "triples_share_events": True,
        "SOURCE_assignments": "old sealed SOURCE-only binary Y",
        "old_permutation_reused": True,
        "source_mass_replay_error": float(np.max(abs(replay - mass))),
        "new_neural_API_private_training_application_label_calls": 0,
        "sampled_publication_not_conversation": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_inputs.jsonl", "input_quality.json"),
    )
    print(json.dumps(quality), flush=True)


def compute(out):
    """Construct the registered laws once,keeping every target and signed account."""
    cfg = R.check(out)["config"]
    z = R.prepared(out)
    start = time.perf_counter()
    mass = np.array(
        [[M.triple_mass(path) for path in row] for row in z["source_paths"]]
    )
    reference = mass.mean((0, 1))
    field = np.empty((2, 2, 2, 4, 5, 8, 8, 8))
    tables = np.zeros((2, 2, 2, 4, 32, 8, 8, 8))
    readout, receipts, rows = [], [], []
    full_profiles = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(int)
    for law in range(2):
        for bank in range(2):
            subset = z["bank"] == bank
            for window in range(2):
                if time.perf_counter() - start > cfg["compute_seconds_cap"]:
                    raise TimeoutError(
                        "Finite compute cap exhausted;no retuned continuation"
                    )
                y, p = z["targets"][law, subset], mass[subset, window]
                profiles, laws, receipt = M.construct(
                    p, y, cfg["ipf_tolerance"], cfg["ipf_max_sweeps"]
                )
                codes = profiles @ (1 << np.arange(5))
                for k, code in enumerate(codes):
                    tables[law, bank, window, :, code] = laws[:, k]
                field[law, bank, window] = M.fields(profiles, laws)
                receipts.append(
                    {
                        "law": cfg["laws"][law],
                        "bank": bank,
                        "window": window,
                        "profiles": len(profiles),
                        "IPF": receipt,
                    }
                )
                errors = M.margin_errors(laws)
                for model in range(3):
                    rows.append(
                        {
                            "law": cfg["laws"][law],
                            "bank": bank,
                            "window": window,
                            "reference": M.MODELS[model],
                            "KL_observed_reference_nats": M.divergence(
                                laws[3], laws[model]
                            ),
                            "adjacent_profile_max_error": errors[
                                "adjacent_profile_max"
                            ],
                            "common_pooled_triple_max_error": errors[
                                "common_pooled_triple_max"
                            ],
                        }
                    )
        profiles, pooled, receipt = M.construct(
            mass.mean(1), z["targets"][law], cfg["ipf_tolerance"], cfg["ipf_max_sweeps"]
        )
        readout.append(M.fields(profiles, pooled))
        receipts.append(
            {
                "law": cfg["laws"][law],
                "bank": "pooled",
                "window": "pooled",
                "profiles": len(profiles),
                "IPF": receipt,
            }
        )
    delta = field[:, :, :, 3] - field[:, :, :, 2]
    comparisons, primary = [], []
    pairs = (
        ((0, 0), (1, 1), "cross_author_cross_window"),
        ((0, 1), (1, 0), "cross_author_cross_window"),
        ((0, 0), (1, 0), "cross_author_same_window"),
        ((0, 1), (1, 1), "cross_author_same_window"),
        ((0, 0), (0, 1), "same_author_cross_window"),
        ((1, 0), (1, 1), "same_author_cross_window"),
    )
    for law in range(2):
        this_primary = []
        for left, right, kind in pairs:
            inner, cosine, coverage, cells = M.recurrence(
                delta[law][left], delta[law][right], reference
            )
            if kind == "cross_author_cross_window":
                this_primary.append(inner)
            for q in range(5):
                comparisons.append(
                    {
                        "law": cfg["laws"][law],
                        "comparison": kind,
                        "left_bank": left[0],
                        "left_window": left[1],
                        "right_bank": right[0],
                        "right_window": right[1],
                        "target": cfg["targets"][q],
                        "signed_inner_probability_squared": float(inner[q]),
                        "field_cosine": None
                        if not np.isfinite(cosine[q])
                        else float(cosine[q]),
                        "reference_mass_coverage": coverage,
                        "joint_supported_cells": cells,
                    }
                )
        primary.append(np.mean(this_primary, axis=0))
    R.archive(
        out / "computed.npz",
        author_mass=mass,
        reference=reference,
        tables=tables,
        full_profiles=full_profiles,
        fields=field,
        delta=delta,
        primary=np.array(primary),
    )
    R.archive(out / "cold_readout.npz", readout=np.array(readout))
    R.table(out / "law_accounts.csv", rows)
    R.table(out / "target_assignment_recurrence.csv", comparisons)
    R.write(out / "solver_receipts.json", receipts)
    summary = {
        "primary_signed_cross_author_cross_window_ENACO": clean(primary),
        "primary_units": "probability-field product,NOTpredictionr/personality fraction/pvalue",
        "actual_positive_targets": [
            cfg["targets"][q] for q in range(5) if primary[0][q] > 0
        ],
        "old_permutation_positive_targets": [
            cfg["targets"][q] for q in range(5) if primary[1][q] > 0
        ],
        "max_margin_errors": {
            key: max(row[key] for row in rows)
            for key in ("adjacent_profile_max_error", "common_pooled_triple_max_error")
        },
        "compute_seconds": time.perf_counter() - start,
        "source_triples": 12352,
        "independent_source_authors": 386,
        "new_model_API_training_calls": 0,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "computed.npz",
            "cold_readout.npz",
            "law_accounts.csv",
            "target_assignment_recurrence.csv",
            "solver_receipts.json",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


def cold(out):
    """Use only the compiled reading and observed application events in this process."""
    R.check(out, ("computed_seal.json",))
    readout = S.selected(out / "cold_readout.npz", ("readout",))["readout"]
    values = []
    with (
        (out / "cold_inputs.jsonl").open() as f,
        (out / "cold_outputs.jsonl").open("x") as sink,
    ):
        for line in f:
            value = M.cold_record(readout, json.loads(line))
            sink.write(json.dumps(value, allow_nan=False) + "\n")
            values.append(value)
    support = np.array([v["supported"] for v in values])
    receipt = {
        "cold_windows": len(values),
        "authors": len(values) // 2,
        "fully_supported_counts_law_model_target": support.sum(0).tolist(),
        "reads_source_authors_or_targets": False,
        "application_labels_read": False,
        "new_neural_API_train_private_calls": 0,
        "missing_not_imputed": True,
    }
    R.write(out / "cold_receipt.json", receipt)
    R.seal(out, "cold_seal.json", ("cold_outputs.jsonl", "cold_receipt.json"))
    print(json.dumps(receipt), flush=True)


def verify(out):
    """Replay counts,Markov products and cold values with different arithmetic."""
    R.check(out, STAGES[:2])
    z, a = (
        R.prepared(out),
        S.selected(
            out / "computed.npz",
            ("author_mass", "tables", "fields", "delta", "full_profiles"),
        ),
    )
    count_error, markov_error, marginal_error, field_error, cold_error = (0.0,) * 5
    for u in range(386):
        for w in range(2):
            counts = np.zeros((8, 8, 8))
            for triple in zip(
                z["source_paths"][u, w, :-2],
                z["source_paths"][u, w, 1:-1],
                z["source_paths"][u, w, 2:],
                strict=True,
            ):
                counts[triple] += 1
            count_error = max(
                count_error, float(np.max(abs(counts / 16 - a["author_mass"][u, w])))
            )
    for laws in a["tables"].reshape(-1, 4, 32, 8, 8, 8):
        f = laws[3]
        direct = np.zeros_like(f)
        for g in range(32):
            for b in range(8):
                t = f[g, :, b, :]
                if t.sum() > 0:
                    direct[g, :, b, :] = np.outer(t.sum(1), t.sum(0)) / t.sum()
        markov_error = max(markov_error, float(np.max(abs(direct - laws[0]))))
        marginal_error = max(marginal_error, *M.margin_errors(laws).values())
    for i in np.ndindex(2, 2, 2):
        t = a["tables"][i]
        total = t[3].sum(0)
        differences = []
        for q in range(5):
            num = sum(
                (t[3, g] - t[2, g] for g in range(32) if a["full_profiles"][g, q]),
                np.zeros((8, 8, 8)),
            )
            # Keep the finite denominator mismatch,not an assumed exact margin.
            num -= a["fields"][i][2, q] * (t[3].sum(0) - t[2].sum(0))
            differences.append(
                np.divide(num, total, out=np.full_like(total, np.nan), where=total > 0)
            )
        delta = np.asarray(differences)
        known = np.isfinite(delta) & np.isfinite(a["delta"][i])
        field_error = max(
            field_error,
            float(np.max(abs(delta[known] - a["delta"][i][known])))
            if known.any()
            else 0,
        )
    readout = S.selected(out / "cold_readout.npz", ("readout",))["readout"]
    with (
        (out / "cold_inputs.jsonl").open() as f,
        (out / "cold_outputs.jsonl").open() as g,
    ):
        for line, response in zip(f, g, strict=True):
            record, actual = json.loads(line), json.loads(response)
            cells = [v["cell"] for v in record["events"]]
            each = [
                readout[..., a, b, c]
                for a, b, c in zip(cells[:-2], cells[1:-1], cells[2:], strict=True)
            ]
            expected = sum(each) / len(each)
            reported = np.asarray(actual["scores"], float)
            if not np.array_equal(np.isfinite(expected), np.isfinite(reported)):
                raise ValueError("Cold missing-cell status changed")
            known = np.isfinite(expected)
            cold_error = max(
                cold_error,
                float(np.max(abs(expected[known] - reported[known])))
                if known.any()
                else 0,
            )
    passed = (
        max(count_error, markov_error, field_error, cold_error) < 1e-10
        and marginal_error <= 4e-10
    )
    value = {
        "PASS": passed,
        "count_error": count_error,
        "markov_error": markov_error,
        "margin_error": marginal_error,
        "target_field_error": field_error,
        "cold_error": cold_error,
        "same_agent_alternate_arithmetic_not_blind": True,
        "source_triples_recounted": 12352,
        "cold_windows": 240,
    }
    R.write(out / "verification.json", value)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(value), flush=True)
    if not passed:
        raise ValueError("Finite arithmetic/identity verification failed")


def main():
    """Run only the requested once-only stage;never resume or overwrite science."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compute", "cold", "verify", "check")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.output, STAGES)
        print("HM235 all input/code/output seals PASS")
    else:
        try:
            globals()[args.stage](args.output)
        except Exception as exc:
            if args.output.is_dir():
                failure = args.output / f"{args.stage}_failure.json"
                if not failure.exists():
                    R.write(
                        failure,
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "no_auto_retry": True,
                        },
                    )
            raise


if __name__ == "__main__":
    main()
