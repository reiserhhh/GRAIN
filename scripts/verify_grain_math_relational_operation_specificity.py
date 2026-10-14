#!/usr/bin/env python3
"""HM52 alternate source, token assignment and context-action arithmetic audit."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_operation_specificity import OUT, check, write
from scripts.verify_grain_math_evaluative_routing import native_score
from scripts.verify_grain_math_relational_cut import csv_rows


def odds(native):
    """Recompute log-odds with scalar sums, independently of runner vector arithmetic."""
    z = native["candidate_logits"]
    values = []
    for v in (z[:5], z[5:]):
        offset = max(v)
        values.append(offset + math.log(math.fsum(math.exp(x - offset) for x in v)))
    return values[1] - values[0]


def group_summary(rows):
    """Rebuild all class metrics from raw audited observations."""
    n = len(rows)
    f = [r["full_change"] for r in rows]
    d = [r["patch_change"] for r in rows]
    l = [r["patch_log_odds"] - r["native_log_odds"] for r in rows]
    eta = [(1 - 2 * r["direction"]) * (-1 if r["person"] == "A" else 1) for r in rows]
    sq = math.fsum(x * x for x in f)
    return {
        "rows": n,
        "full_rms": math.sqrt(sq / n),
        "patch_rms": math.sqrt(math.fsum(x * x for x in d) / n),
        "signed_projection": math.fsum(x * y for x, y in zip(f, d, strict=True)) / sq
        if sq
        else None,
        "native_contrast_tiny": sq < 1e-12,
        "remaining_rmse": math.sqrt(
            math.fsum((x - y) ** 2 for x, y in zip(f, d, strict=True)) / n
        ),
        "direction_count": sum(x * y > 0 for x, y in zip(f, d, strict=True)),
        "oriented_score_mean": math.fsum(s * x for s, x in zip(eta, d, strict=True))
        / n,
        "log_odds_action_rms": math.sqrt(math.fsum(x * x for x in l) / n),
        "oriented_log_odds_mean": math.fsum(s * x for s, x in zip(eta, l, strict=True))
        / n,
    }


def compare(actual, expected, errors):
    """Check fields without converting unobserved projection into a false numeric zero."""
    for k, value in expected.items():
        got = actual[k]
        if value is None:
            if got not in (None, ""):
                raise ValueError("Unexpected numeric value " + k)
        elif isinstance(value, bool):
            if str(got) != str(value):
                raise ValueError("Boolean mismatch " + k)
        elif isinstance(value, str):
            if got != value:
                raise ValueError("Identity mismatch " + k)
        else:
            errors.append(abs(float(got) - float(value)))


def verify(out):
    """Audit actual672calls and all declared contrasts; never run a new model call."""
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
            raise ValueError("Actual output changed " + name)
    seal = read_json(out / "00/source_seal.json")
    if not seal["before_all_new_calls"] or seal["new_source_forwards"] != 0:
        raise ValueError("Source barrier failed")
    parent = ROOT / m["spec"]["parent"]
    prior = read_json(parent / "00/manifest.json")
    hm50 = ROOT / prior["spec"]["parent"]
    source_caps = {
        (e["content"], e["person"], e["direction"]): e
        for e in read_json(hm50 / "00/design.json")
        if e["kind"] == "capture" and e["role"] == "source"
    }
    old_rows = {r["unit"]: r for r in read_json(hm50 / "00/receivers.json")}
    source, source_ids = {}, {}
    with np.load(out / "00/source_vectors.npz", allow_pickle=False) as archive:
        for target in m["spec"]["targets"]:
            for actor in ("A", "B"):
                zero, one = [source_caps[target, actor, b] for b in (0, 1)]
                _, a = load_arrays(hm50 / f"01/units/unit_{zero['unit']:04d}.json")
                _, b = load_arrays(hm50 / f"01/units/unit_{one['unit']:04d}.json")
                p, q = zero["partition"]["suffix"], one["partition"]["suffix"]
                source[target, actor] = np.subtract(
                    b["full"][q].astype(float), a["full"][p].astype(float)
                )
                source_ids[target, actor] = (
                    [old_rows[zero["receiver"]]["input_ids"][i] for i in p],
                    p,
                )
            source[target, "shared"] = 0.5 * (source[target, "A"] + source[target, "B"])
            source[target, "bound"] = 0.5 * (source[target, "A"] - source[target, "B"])
            for key in ("A", "B", "shared", "bound"):
                if not np.array_equal(source[target, key], archive[f"{target}__{key}"]):
                    raise ValueError("Source operator construction drift")
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    design = read_json(out / "00/design.json")
    if len(rows) != 96 or len(design) != 672:
        raise ValueError("Missing receiver or arm")
    captures, scores, logs, errors = {}, {}, {}, []
    for e in design[:96]:
        r = rows[e["receiver"]]
        if e["kind"] != "capture" or e["unit"] != e["receiver"]:
            raise ValueError("All native captures must precede every patch")
        p = r["partition"]["suffix"]
        ids, sp = source_ids[r["content"], r["person"]]
        if ids != [r["input_ids"][i] for i in p]:
            raise ValueError("Source/new query token mismatch")
        proof = {
            "source_positions": sp,
            "application_positions": p,
            "suffix_ids": ids,
            "relative_slots": list(range(len(ids))),
        }
        if proof != r["source_mapping"]:
            raise ValueError("Source token mapping drift")
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        for k in (
            "unit",
            "pair",
            "receiver",
            "donor",
            "kind",
            "arm",
            "content",
            "class",
            "person",
            "direction",
        ):
            if meta[k] != e[k]:
                raise ValueError("Capture identity drift")
        y, mass = native_score(meta["native"])
        errors.extend(
            (
                abs(y - meta["native"]["conditional_score"]),
                abs(mass - meta["native"]["candidate_mass"]),
            )
        )
        captures[e["receiver"]], scores[e["receiver"]], logs[e["receiver"]] = (
            a["full"],
            y,
            odds(meta["native"]),
        )
    effects, native_rows, max_pre, max_noop, max_random = {}, {}, 0.0, 0.0, 0.0
    min_mass = 1.0
    for e in design:
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        for k in (
            "unit",
            "pair",
            "receiver",
            "donor",
            "kind",
            "arm",
            "content",
            "class",
            "person",
            "direction",
        ):
            if meta[k] != e[k]:
                raise ValueError("Call identity drift")
        y, mass = native_score(meta["native"])
        errors.extend(
            (
                abs(y - meta["native"]["conditional_score"]),
                abs(mass - meta["native"]["candidate_mass"]),
            )
        )
        min_mass = min(min_mass, mass)
        fields = {
            k: e[k]
            for k in ("unit", "pair", "content", "class", "person", "direction", "arm")
        }
        if e["kind"] == "capture":
            native_rows[e["unit"]] = fields | {
                "native": y,
                "candidate_mass": mass,
                "native_log_odds": odds(meta["native"]),
                "tokens": a["full"].shape[0],
            }
            continue
        current, donor = captures[e["receiver"]], captures[e["donor"]]
        p, q = e["partition"]["suffix"], e["mapping"]["donor_positions"]
        r, d = rows[e["receiver"]], rows[e["donor"]]
        if (
            d["direction"] != 1 - r["direction"]
            or any(d[k] != r[k] for k in ("content", "class", "person", "query"))
            or [r["input_ids"][i] for i in p] != [d["input_ids"][i] for i in q]
            or list(a["patch_positions"]) != p
        ):
            raise ValueError("Wrong matched donor/suffix")
        arm, t, actor = e["arm"], e["content"], e["person"]
        if arm == "exact_suffix":
            value = donor[q]
        elif arm == "noop":
            value = current[p]
        else:
            if arm == "source_full":
                delta = source[t, actor]
            elif arm == "source_bound":
                delta = (1 if actor == "A" else -1) * source[t, "bound"]
            elif arm == "wrong_person":
                delta = source[t, "B" if actor == "A" else "A"]
            elif arm == "random":
                delta = np.random.default_rng(
                    m["spec"]["random_seed"] + e["pair"] // 2
                ).normal(size=source[t, actor].shape)
                lengths = np.sqrt(np.sum(source[t, actor] ** 2, axis=1))
                delta *= (lengths / np.sqrt(np.sum(delta**2, axis=1)))[:, None]
            else:
                raise ValueError("Unexpected arm")
            value = (
                current[p].astype(float) + (1 - 2 * e["direction"]) * delta
            ).astype(current.dtype)
            if arm == "random":
                length = np.sqrt(
                    np.sum(
                        (value.astype(float) - current[p].astype(float)) ** 2, axis=1
                    )
                )
                max_random = max(
                    max_random,
                    float(np.max(abs(length - lengths) / np.maximum(lengths, 1e-20))),
                )
        if not np.array_equal(value, a["patch_values"]):
            raise ValueError("Wrong source patch")
        expected = a["before"].copy()
        expected[p] = value
        if not np.array_equal(expected, a["full"]):
            raise ValueError("Unexpected unselected-position change")
        max_pre = max(
            max_pre,
            float(np.max(abs(a["before"].astype(float) - current.astype(float)))),
        )
        if arm == "noop":
            max_noop = max(max_noop, abs(y - scores[e["receiver"]]))
        effects[e["unit"]] = fields | {
            "native": scores[e["receiver"]],
            "donor": scores[e["donor"]],
            "patched": y,
            "candidate_mass": mass,
            "native_log_odds": logs[e["receiver"]],
            "donor_log_odds": logs[e["donor"]],
            "patch_log_odds": odds(meta["native"]),
            "full_change": scores[e["donor"]] - scores[e["receiver"]],
            "patch_change": y - scores[e["receiver"]],
            "remaining": scores[e["donor"]] - y,
        }
    for name, index in (("effects.csv", effects), ("native.csv", native_rows)):
        actual = csv_rows(out / "03" / name)
        if len(actual) != len(index):
            raise ValueError("Missing actual rows")
        for row in actual:
            compare(row, index[int(row["unit"])], errors)
    for filename, keys in (
        ("class_summary.csv", ("class", "arm")),
        ("target_summary.csv", ("content", "class", "arm")),
        ("person_summary.csv", ("person", "class", "arm")),
    ):
        groups = defaultdict(list)
        for r in effects.values():
            groups[tuple(r[k] for k in keys)].append(r)
        actual = csv_rows(out / "03" / filename)
        if len(actual) != len(groups):
            raise ValueError("Missing summary groups")
        for row in actual:
            compare(row, group_summary(groups[tuple(row[k] for k in keys)]), errors)
    index = {
        (r["content"], r["class"], r["person"], r["direction"], r["arm"]): r
        for r in effects.values()
    }
    contrasts, mixed = {}, defaultdict(list)
    for row in csv_rows(out / "03/class_contrasts.csv"):
        t, actor, b, arm, control = (
            row["content"],
            row["person"],
            int(row["direction"]),
            row["arm"],
            row["control"],
        )
        own, other = index[t, "own", actor, b, arm], index[t, control, actor, b, arm]
        score = own["patch_change"] - other["patch_change"]
        log = (
            own["patch_log_odds"]
            - own["native_log_odds"]
            - other["patch_log_odds"]
            + other["native_log_odds"]
        )
        eta = (1 - 2 * b) * (-1 if actor == "A" else 1)
        v = {
            "score_action_difference": score,
            "log_odds_action_difference": log,
            "oriented_score_difference": eta * score,
            "oriented_log_odds_difference": eta * log,
        }
        compare(row, v, errors)
        key = (t, actor, b, arm, control)
        if key in contrasts:
            raise ValueError("Duplicate contrast")
        contrasts[key] = v
        mixed[control, arm].append(v)
    if len(contrasts) != 432:
        raise ValueError("Missing432 predetermined class contrasts")
    contrast_summary = csv_rows(out / "03/class_contrast_summary.csv")
    if len(contrast_summary) != 18:
        raise ValueError("Missing class contrast groups")
    for row in contrast_summary:
        v = mixed[row["control"], row["arm"]]
        n = len(v)
        compare(
            row,
            {
                "rows": n,
                "oriented_score_difference": math.fsum(
                    r["oriented_score_difference"] for r in v
                )
                / n,
                "oriented_log_odds_difference": math.fsum(
                    r["oriented_log_odds_difference"] for r in v
                )
                / n,
                "score_difference_rms": math.sqrt(
                    math.fsum(r["score_action_difference"] ** 2 for r in v) / n
                ),
                "log_odds_difference_rms": math.sqrt(
                    math.fsum(r["log_odds_action_difference"] ** 2 for r in v) / n
                ),
            },
            errors,
        )
    stored = read_json(out / "03/summary.json")
    for row in stored["groups"]:
        selected = [
            r
            for r in effects.values()
            if (r["class"], r["arm"]) == (row["class"], row["arm"])
        ]
        compare(row, group_summary(selected), errors)
    errors.extend(
        (
            abs(stored["maximum_prepatch_difference"] - max_pre),
            abs(stored["maximum_noop_error"] - max_noop),
            abs(stored["minimum_candidate_mass"] - min_mass),
        )
    )
    maximum = max(errors, default=0.0)
    if maximum > 1e-10 or max_pre > 1e-6 or max_noop > 1e-6 or max_random > 1e-5:
        raise ValueError(
            f"Arithmetic/assignment tolerance failed:{maximum, max_pre, max_noop, max_random}"
        )
    result = {
        "status": "VERIFIED_FINITE_HM52_NOT_GLOBAL",
        "actual_forwards": 672,
        "new_source_forwards": 0,
        "source_captures_recomputed": 24,
        "native_captures_checked": 96,
        "all_token_patches_checked": 576,
        "class_contrasts_checked": 432,
        "maximum_alternate_scalar_error": maximum,
        "maximum_prepatch_error": max_pre,
        "maximum_noop_error": max_noop,
        "random_norm_max_relative_error": max_random,
        "same_agent_alternate_arithmetic_not_neural_replication": True,
        "verifier_model_calls": 0,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result, flush=True)


def main():
    """Verify only; never start or retry a collector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.output)


if __name__ == "__main__":
    main()
