#!/usr/bin/env python3
"""HM54 alternate source/patch/null arithmetic; no neural model calls."""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_evidence_response_operation import (
    IDENTITY,
    OUT,
    check,
    write,
)
from scripts.verify_grain_math_answer_recoding_mechanism import mapping
from scripts.verify_grain_math_evaluative_routing import native_score
from scripts.verify_grain_math_relational_cut import csv_rows
from scripts.verify_grain_math_relational_operation_specificity import compare, odds


def scalar(native):
    """Compute expectation, coverage and odds with scalar candidate sums."""
    score, mass = native_score(native)
    return {"score": score, "log_odds": odds(native), "candidate_mass": mass}


def null_scalar(logits, offset):
    """Reconstruct high-group output offsets using scalar arithmetic."""
    z = [x + (offset if j >= 5 else 0) for j, x in enumerate(logits)]
    top = max(z)
    denominator = math.fsum(math.exp(x - top) for x in z)
    q = [math.exp(x - top) / denominator for x in z]
    return {
        "score": math.fsum(p * j / 9 for j, p in enumerate(q)),
        "log_odds": odds({"candidate_logits": z}),
        "probabilities": q,
    }


def pair_scalar(n0, n1, p0, p1):
    """Compute interactions as differences of actions, not differences of contrasts."""
    result = {}
    for metric in ("score", "log_odds"):
        a0 = p0[metric] - n0[metric]
        a1 = p1[metric] - n1[metric]
        result.update(
            {
                f"native_{metric}_contrast": n1[metric] - n0[metric],
                f"patched_{metric}_contrast": p1[metric] - p0[metric],
                f"action_{metric}_0": a0,
                f"action_{metric}_1": a1,
                f"mixed_{metric}": a1 - a0,
            }
        )
    return result


def summary_scalar(rows):
    """Recompute signed mean/RMS by scalar sums."""
    result = {"pairs": len(rows)}
    for metric in ("score", "log_odds"):
        for prefix in ("native", "patched", "mixed"):
            key = prefix + "_" + metric + ("_contrast" if prefix != "mixed" else "")
            v = [r[key] for r in rows]
            result[key + "_mean"] = math.fsum(v) / len(v)
            result[key + "_rms"] = math.sqrt(math.fsum(x * x for x in v) / len(v))
    result["native_patched_log_direction_matches"] = sum(
        r["native_log_odds_contrast"] * r["patched_log_odds_contrast"] > 0 for r in rows
    )
    return result


def verify(out):
    """Verify sources, all actual patches and paired/null tables without forwards."""
    m, status = check(out), read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 672
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or status["model_seconds"] > 600
    ):
        raise ValueError("Incomplete finite acquisition")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / name) != sha:
            raise ValueError("Changed actual output " + name)
    seal = read_json(out / "00/source_seal.json")
    if (
        not seal["before_all_new_calls"]
        or seal["new_source_calls"] != 0
        or seal["fit_or_application_answer_selection"]
        or seal["source_capture_count"] != 24
        or seal["source_operator_count"] != 12
    ):
        raise ValueError("Source barrier changed")
    rows = {r["unit"]: r for r in read_json(out / "00/rows.json")}
    design = read_json(out / "00/design.json")
    if (
        len(rows) != 96
        or len(design) != 672
        or len({r["body"] for r in rows.values()}) != 38
    ):
        raise ValueError("Declared input scope missing")
    src = read_json(out / "00/source_rows.json")
    si = {
        (r["row"]["content"], r["row"]["person"], r["row"]["direction"]): r for r in src
    }
    source, source_readouts, source_states = {}, {}, {}
    errors = []
    for r in src:
        row = r["row"]
        meta, a = load_arrays(ROOT / r["capture"])
        if (
            row["codec"] != 1
            or row["role"] != "source"
            or not row["reused"]
            or meta["unit"] != row["standard_reference_uid"]
            or len(a["full"]) != len(row["input_ids"])
        ):
            raise ValueError("Original source identity drift")
        key = (row["content"], row["person"], row["direction"])
        source_states[key] = a["full"]
        source_readouts[key] = scalar(meta["native"])
        compare(r["readout"], source_readouts[key], errors)
    offsets = read_json(out / "00/source_offsets.json")
    with np.load(out / "00/source_vectors.npz", allow_pickle=False) as archive:
        for target, person in itertools.product(m["spec"]["targets"], ("A", "B")):
            a, b = (si[target, person, d]["row"] for d in (0, 1))
            mapping(a, b)
            delta = np.subtract(
                source_states[target, person, 1][b["partition"]["suffix"]].astype(
                    float
                ),
                source_states[target, person, 0][a["partition"]["suffix"]].astype(
                    float
                ),
            )
            key = f"{target}__{person}"
            if not np.array_equal(delta, archive[key]):
                raise ValueError(
                    "Source operator not reconstructed from original captures"
                )
            source[key] = delta
            errors.append(
                abs(
                    offsets[key]
                    - (
                        source_readouts[target, person, 1]["log_odds"]
                        - source_readouts[target, person, 0]["log_odds"]
                    )
                )
            )
    captures, natives, meta_native = {}, {}, {}
    for e in design[:96]:
        r = rows[e["receiver"]]
        if e["kind"] != "capture" or e["arm"] != "native" or e["unit"] != r["unit"]:
            raise ValueError("Native design drift")
        if mapping(si[r["content"], r["person"], 0]["row"], r) != r["source_mapping"]:
            raise ValueError("Source token map changed")
        other = si[r["content"], "B" if r["person"] == "A" else "A", 0]["row"]
        if mapping(other, r, allow_actor=True) != r["wrong_source_mapping"]:
            raise ValueError("Other-actor slot map changed")
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        if (
            any(meta[k] != e[k] for k in IDENTITY)
            or meta["finished_epoch"] <= seal["created_epoch"]
        ):
            raise ValueError("Native identity/source barrier drift")
        if len(a["full"]) != len(r["input_ids"]):
            raise ValueError("Full token/state count drift")
        captures[r["unit"]] = a["full"]
        meta_native[r["unit"]] = meta["native"]
        v = scalar(meta["native"])
        errors.extend(
            (
                abs(v["score"] - meta["native"]["conditional_score"]),
                abs(v["candidate_mass"] - meta["native"]["candidate_mass"]),
            )
        )
        natives[r["unit"]] = (
            {
                k: r[k]
                for k in ("unit", "content", "class", "person", "direction", "pair_id")
            }
            | v
            | {"tokens": len(r["input_ids"])}
        )
    effects, patched = {}, {}
    pre, assignment, noop, random_error = 0.0, 0.0, 0.0, 0.0
    for e in design[96:]:
        r, d = rows[e["receiver"]], rows[e["donor"]]
        proof = mapping(d, r)
        if (
            d["direction"] != 1 - r["direction"]
            or d["class"] != r["class"]
            or d["pair_id"] != r["pair_id"]
            or e["donor_positions"] != proof["source_positions"]
        ):
            raise ValueError("Paired donor drift")
        p = r["partition"]["suffix"]
        current = captures[r["unit"]]
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        if (
            any(meta[k] != e[k] for k in IDENTITY)
            or meta["finished_epoch"] <= seal["created_epoch"]
        ):
            raise ValueError("Patch identity/source barrier drift")
        arm = e["arm"]
        if arm == "exact_suffix":
            value = captures[d["unit"]][e["donor_positions"]]
        elif arm == "noop":
            value = current[p]
        else:
            if arm not in ("source_plus", "source_minus", "wrong_person", "random"):
                raise ValueError("Unknown actual arm")
            person = (
                ("B" if r["person"] == "A" else "A")
                if arm == "wrong_person"
                else r["person"]
            )
            delta = source[f"{r['content']}__{person}"]
            norms = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            if arm == "random":
                noise = np.random.default_rng(
                    m["spec"]["random_seed"] + r["pair_id"]
                ).normal(size=delta.shape)
                delta = (
                    noise
                    * (norms / np.sqrt(np.einsum("ij,ij->i", noise, noise)))[:, None]
                )
            sign = -1 if arm == "source_minus" else 1
            value = (current[p].astype(float) + sign * delta).astype(current.dtype)
            if arm == "random":
                displacement = a["patch_values"].astype(float) - current[p].astype(
                    float
                )
                lengths = np.sqrt(np.einsum("ij,ij->i", displacement, displacement))
                random_error = max(
                    random_error,
                    float(np.max(abs(lengths - norms) / np.maximum(norms, 1e-20))),
                )
        if list(a["patch_positions"]) != p:
            raise ValueError("Actual patch positions changed")
        assignment = max(
            assignment,
            float(np.max(abs(value.astype(float) - a["patch_values"].astype(float)))),
        )
        pre = max(
            pre, float(np.max(abs(a["before"].astype(float) - current.astype(float))))
        )
        unselected = np.ones(len(current), bool)
        unselected[p] = False
        if not np.array_equal(
            a["before"][unselected], a["full"][unselected]
        ) or not np.array_equal(a["full"][p], a["patch_values"]):
            raise ValueError("Actual assignment/untouched token changed")
        v = scalar(meta["native"])
        n = natives[r["unit"]]
        if arm == "noop":
            noop = max(
                noop, abs(v["score"] - n["score"]), abs(v["log_odds"] - n["log_odds"])
            )
        patched[r["unit"], arm] = v
        effects[e["unit"]] = {k: e[k] for k in IDENTITY} | {
            **{
                prefix + "_" + k: obj[k]
                for prefix, obj in (
                    ("native", n),
                    ("donor", natives[d["unit"]]),
                    ("patched", v),
                )
                for k in ("score", "log_odds")
            },
            "score_action": v["score"] - n["score"],
            "log_odds_action": v["log_odds"] - n["log_odds"],
            "candidate_mass": v["candidate_mass"],
        }
    for name, expected in (("native.csv", natives), ("effects.csv", effects)):
        table = csv_rows(out / "03" / name)
        if len(table) != len(expected):
            raise ValueError("Missing native/effect rows")
        for row in table:
            compare(row, expected[int(row["unit"])], errors)
    ri = {
        (r["content"], r["class"], r["person"], r["direction"]): r
        for r in rows.values()
    }
    pairs = {}
    for row in csv_rows(out / "03/evidence_pair_actions.csv"):
        key = tuple(row[k] for k in ("content", "class", "person", "arm"))
        if key in pairs:
            raise ValueError("Duplicate pair")
        t, c, p, arm = key
        a, b = (ri[t, c, p, d] for d in (0, 1))
        v = pair_scalar(
            natives[a["unit"]],
            natives[b["unit"]],
            patched[a["unit"], arm],
            patched[b["unit"], arm],
        )
        compare(row, v | {"pair_id": a["pair_id"]}, errors)
        pairs[key] = (
            dict(zip(("content", "class", "person", "arm"), key, strict=True)) | v
        )
    if len(pairs) != 288:
        raise ValueError("Missing288paired actions")
    nulls = {}
    for row in csv_rows(out / "03/output_offset_null.csv"):
        unit, sign = int(row["unit"]), int(row["source_sign"])
        r = rows[unit]
        key = (unit, sign)
        if key in nulls or sign not in (-1, 1):
            raise ValueError("Invalid null row")
        b = sign * offsets[f"{r['content']}__{r['person']}"]
        v = null_scalar(meta_native[unit]["candidate_logits"], b)
        compare(
            row,
            {
                k: r[k]
                for k in ("unit", "content", "class", "person", "direction", "pair_id")
            }
            | {
                "source_sign": sign,
                "offset": b,
                "score": v["score"],
                "log_odds": v["log_odds"],
                "score_action": v["score"] - natives[unit]["score"],
                "log_odds_action": v["log_odds"] - natives[unit]["log_odds"],
                **{f"p{k}": x for k, x in enumerate(v["probabilities"])},
            },
            errors,
        )
        nulls[key] = v
    if len(nulls) != 192:
        raise ValueError("Missing192nullrows")
    null_pair_keys = set()
    null_max = 0.0
    for row in csv_rows(out / "03/output_offset_null_pairs.csv"):
        t, c, p, sign = (
            row["content"],
            row["class"],
            row["person"],
            int(row["source_sign"]),
        )
        key = (t, c, p, sign)
        if key in null_pair_keys:
            raise ValueError("Duplicate null pair")
        null_pair_keys.add(key)
        a, b = (ri[t, c, p, d] for d in (0, 1))
        v = pair_scalar(
            natives[a["unit"]],
            natives[b["unit"]],
            nulls[a["unit"], sign],
            nulls[b["unit"], sign],
        )
        compare(row, v | {"pair_id": a["pair_id"]}, errors)
        null_max = max(null_max, abs(v["mixed_log_odds"]))
    if len(null_pair_keys) != 96:
        raise ValueError("Missing96nullpairs")
    for filename, keys in (
        ("class_summary.csv", ("class", "arm")),
        ("target_class_summary.csv", ("content", "class", "arm")),
    ):
        groups = defaultdict(list)
        for r in pairs.values():
            groups[tuple(r[k] for k in keys)].append(r)
        actual = csv_rows(out / "03" / filename)
        if len(actual) != len(groups):
            raise ValueError("Missing summary groups")
        for row in actual:
            compare(row, summary_scalar(groups[tuple(row[k] for k in keys)]), errors)
    summary = read_json(out / "03/summary.json")
    required = {
        "packet": "HM54",
        "receivers": 96,
        "effects": 576,
        "paired_actions": 288,
        "output_null_rows": 192,
        "output_null_pairs": 96,
        "new_calls": 672,
        "source_captures_reused": 24,
        "unique_body_count": 38,
        "no_accuracy_selection": True,
        "nonzero_interaction_not_unique_reading_program": True,
        "original_scope_complete": False,
    }
    if any(summary[k] != v for k, v in required.items()):
        raise ValueError("Summary scope/interpretation drift")
    class_groups = defaultdict(list)
    for row in pairs.values():
        class_groups[row["class"], row["arm"]].append(row)
    summary_keys = [(r["class"], r["arm"]) for r in summary["groups"]]
    if len(summary_keys) != len(class_groups) or set(summary_keys) != set(class_groups):
        raise ValueError("Summary class/arm coverage drift")
    for row, key in zip(summary["groups"], summary_keys, strict=True):
        compare(row, summary_scalar(class_groups[key]), errors)
    errors.extend(
        (
            abs(summary["maximum_null_log_interaction"] - null_max),
            abs(
                summary["minimum_candidate_mass"]
                - min(
                    r["candidate_mass"] for r in [*natives.values(), *effects.values()]
                )
            ),
        )
    )
    if (
        max(errors) > m["spec"]["arithmetic_tolerance"]
        or null_max > 1e-10
        or pre > 1e-6
        or assignment > 1e-6
        or noop > 1e-6
        or random_error > 1e-5
    ):
        raise ValueError("Arithmetic/assignment tolerance exceeded")
    write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM54_NOT_GLOBAL",
            "new_calls_checked": 672,
            "source_captures_checked": 24,
            "source_deltas_checked": 12,
            "native_effect_pair_null_nullpair_checks": [96, 576, 288, 192, 96],
            "maximum_scalar_error": max(errors),
            "maximum_assignment_error": assignment,
            "maximum_prepatch_error": pre,
            "maximum_noop_error": noop,
            "maximum_random_norm_relative_error": random_error,
            "maximum_null_log_interaction": null_max,
            "new_neural_calls": 0,
            "same_agent_alternate_arithmetic_not_neural_replication": True,
        },
    )
    print(
        f"VERIFIED:672actual/576patches/288pairs/192nulls,maxerror{max(errors):.3e}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)
