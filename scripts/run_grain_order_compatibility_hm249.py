#!/usr/bin/env python3
"""Check SOURCE/reader order feasibility on all frozen HM248 inputs, no fits."""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_order_compatibility as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_order_compatibility_hm249.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_reader_order_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ORDER_COMPATIBILITY_PLAN_HM249.md",
    "grain_research/construction/math_order_compatibility.py",
    "scripts/run_grain_order_compatibility_hm249.py",
    "tests/test_grain_order_compatibility.py",
    "scripts/run_grain_selector_matrix_hm248.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
)
KEYS = (
    "source",
    "native",
    "removal_indices",
    "panel_indices",
    "retained_conditions",
    "retained_lengths",
)


def prepare(out):
    """Bind numerical observations, aliases and addresses without reading labels."""
    if out.exists():
        raise FileExistsError("No replacement or automatic rerun")
    cfg = R.read(ROOT / CONFIG)
    p, source_parent = ROOT / cfg["selector_parent"], ROOT / cfg["source_parent"]
    pm = R.check(
        p, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    R.check(source_parent, ("collected_seal.json", "analyzed_seal.json"))
    if cfg["targets"] != pm["config"]["targets"] or cfg["source_accounts"] != [
        "F",
        "B",
        "D",
    ]:
        raise ValueError("Changed source or reader definition")
    parent = H.selected(
        p / "observations.npz", ("selections", "panel_indices", "source_actions")
    )
    original = H.selected(
        source_parent / "observations.npz", ("accounts", "conditions", "lengths")
    )
    y = H.selected(p / "analysis.npz", ("native_readings",))["native_readings"]
    if y.shape != (24, 5, 2, 5) or parent["selections"].shape != (24, 5, 2):
        raise ValueError("Full historical author/selector/arm family required")
    source = np.stack(
        [original["accounts"][u, parent["selections"][u]] for u in range(24)]
    )
    if not np.array_equal(source[:, :, 1] - source[:, :, 0], parent["source_actions"]):
        raise ValueError("Source endpoint/action binding changed")
    roles, choices = R.read(p / "roles.json"), R.read(p / "choices.json")
    role_map = {(r["author"], r["selector"], r["arm"]): r for r in roles}
    if len(role_map) != 240:
        raise ValueError("Complete unique role addresses required")
    addresses, retained_conditions, retained_lengths = [], [], []
    for u in range(24):
        counts, lengths = [], []
        for r in range(5):
            choice = choices[u * 5 + r]
            if (choice["author"], choice["selector"]) != (u, r):
                raise ValueError("Changed selector/author ownership")
            for arm, name in enumerate(("support", "counter")):
                role = role_map[u, r, arm]
                keep = [j for j in range(12) if j not in choice[name]]
                if len(keep) != 8:
                    raise ValueError("Changed equal-size observation contract")
                counts.append(int(original["conditions"][u, keep].sum()))
                lengths.append(int(original["lengths"][u, keep].sum()))
                addresses.append(
                    {
                        "author": u,
                        "original_author_index": int(parent["panel_indices"][u]),
                        "point": 2 * r + arm,
                        "selector": cfg["targets"][r],
                        "arm": "remove_" + name,
                        "token_sha256": role["token_sha256"],
                        "removed_event_ids": [
                            choice["event_ids"][j] for j in choice[name]
                        ],
                        "retained_event_ids": [choice["event_ids"][j] for j in keep],
                        "handle": role["handle"],
                    }
                )
        retained_conditions.append(counts)
        retained_lengths.append(lengths)
    verification = R.read(p / "verification.json")
    if verification.get("PASS") is not True:
        raise ValueError("HM248 arithmetic verification required")
    inputs = [
        p / name
        for name in (
            "manifest.json",
            "observations.npz",
            "analysis.npz",
            "choices.json",
            "roles.json",
            "verification.json",
            "analyzed_seal.json",
            "verification_seal.json",
        )
    ] + [source_parent / "observations.npz", source_parent / "analyzed_seal.json"]
    out.mkdir(parents=True, exist_ok=False)
    R.archive(
        out / "observations.npz",
        source=source.reshape(24, 10, 3, 5),
        native=y.reshape(24, 10, 5),
        removal_indices=parent["selections"].reshape(24, 10),
        panel_indices=parent["panel_indices"],
        retained_conditions=np.array(retained_conditions),
        retained_lengths=np.array(retained_lengths),
    )
    R.write(out / "addresses.json", addresses)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {f: R.digest(ROOT / f) for f in SCIENCE},
            "inputs": {str(f.relative_to(ROOT)): R.digest(f) for f in inputs},
            "reader_error": 2 * verification["native_head_max_error"],
            "reader_error_scope": "observed head reconstruction only, not all model/scientific error",
            "post_outcome_diagnostic": True,
            "new_model_calls": 0,
            "original_scope_complete": False,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "addresses.json"))
    print("HM249 registered: existing endpoints, no calibration fitting", flush=True)


def observations(out):
    """Resolve unique inputs per author; token copies never increase sample count."""
    z = H.selected(out / "observations.npz", KEYS)
    addresses = R.read(out / "addresses.json")
    selected, alias_rows = [], []
    for u in range(24):
        rows = addresses[u * 10 : (u + 1) * 10]
        if [r["point"] for r in rows] != list(range(10)) or any(
            r["author"] != u for r in rows
        ):
            raise ValueError("Changed addressed observation family")
        keep, aliases = M.unique_observations(
            [r["token_sha256"] for r in rows],
            z["source"][u],
            z["native"][u],
            z["removal_indices"][u],
        )
        selected.append(keep)
        alias_rows.extend({"author": u, "points": a} for a in aliases)
    return z, addresses, selected, alias_rows


def witness(u, field, target, orientation, result, keep, addresses, s, y):
    """Emit a concrete addressed contradiction instead of a fitted remedy."""
    pair = result["guarded"]["witness"]
    if pair is None or result["guarded"]["bound"] == 0:
        return None
    i, j = pair
    return {
        "author": u,
        "source_account": field,
        "target": target,
        "orientation": orientation,
        "lower_point": addresses[u * 10 + int(keep[i])],
        "upper_point": addresses[u * 10 + int(keep[j])],
        "source_gap": np.atleast_1d(s[j] - s[i]).tolist(),
        "reader_gap_opposed_to_order": float(y[i] - y[j]),
        "empirical_adjusted_half_gap": result["guarded"]["bound"],
    }


def analyze(out):
    """Compute fixed scalar and shared-profile order constraints, without a grid."""
    started = time.monotonic()
    manifest = R.check(out)
    cfg = manifest["config"]
    z, addresses, kept, aliases = observations(out)
    errors = {
        "source_error": cfg["source_error"],
        "reader_error": manifest["reader_error"],
    }
    v = np.array(cfg["evaluation_direction"])
    scalar_rows, shared_rows, pairs, witnesses = [], [], [], []
    for u, keep in enumerate(kept):
        source, reader = z["source"][u, keep], z["native"][u, keep]
        for a, field in enumerate(cfg["source_accounts"]):
            for q, target in enumerate(cfg["targets"]):
                s, y = source[:, a, q], reader[:, q]
                result = M.scalar_orientations(s, y, **errors)
                row = {
                    "author": u,
                    "source_account": field,
                    "target": target,
                    "unique_inputs": len(keep),
                }
                for orientation in ("increasing", "decreasing"):
                    r = result[orientation]
                    for arm in ("exact", "guarded"):
                        row.update(
                            {
                                orientation + "_" + arm + "_" + k: r[arm][k]
                                for k in ("constraints", "violations", "bound")
                            }
                        )
                    w = witness(
                        u,
                        field,
                        target,
                        orientation,
                        r,
                        keep,
                        addresses,
                        s if orientation == "increasing" else -s,
                        y,
                    )
                    if w is not None:
                        witnesses.append(w)
                row.update(
                    {
                        "exact_either_bound": result["exact_either_bound"],
                        "guarded_either_bound": result["guarded_either_bound"],
                    }
                )
                scalar_rows.append(row)
                for i in range(len(keep)):
                    for j in range(i + 1, len(keep)):
                        ds, dy = float(s[j] - s[i]), float(y[j] - y[i])
                        guarded = abs(ds) > 2 * cfg["source_error"]
                        pairs.append(
                            {
                                "author": u,
                                "source_account": field,
                                "target": target,
                                "point_i": int(keep[i]),
                                "point_j": int(keep[j]),
                                "source_delta": ds,
                                "reader_delta": dy,
                                "source_order_guarded": guarded,
                                "increasing_conflict": guarded
                                and ds * dy < 0
                                and abs(dy) > 2 * manifest["reader_error"],
                                "retained_condition_difference": int(
                                    z["retained_conditions"][u, keep[j]]
                                    - z["retained_conditions"][u, keep[i]]
                                ),
                                "retained_token_difference": int(
                                    z["retained_lengths"][u, keep[j]]
                                    - z["retained_lengths"][u, keep[i]]
                                ),
                            }
                        )
            s, y = source[:, a, :] * v, reader @ v / 5
            r = M.order_bound(s, y, **errors)
            row = {
                "author": u,
                "source_account": field,
                "unique_inputs": len(keep),
                "possible_unordered_pairs": len(keep) * (len(keep) - 1) // 2,
            }
            for arm in ("exact", "guarded"):
                row.update(
                    {
                        arm + "_" + k: r[arm][k]
                        for k in ("constraints", "violations", "bound")
                    }
                )
            shared_rows.append(row)
            w = witness(
                u,
                field,
                "fixed_shared_v",
                "coordinatewise_increasing",
                r,
                keep,
                addresses,
                s,
                y,
            )
            if w is not None:
                witnesses.append(w)
    summaries = []
    for field in cfg["source_accounts"]:
        for q in cfg["targets"]:
            rows = [
                r
                for r in scalar_rows
                if r["source_account"] == field and r["target"] == q
            ]
            bounds = [r["guarded_either_bound"] for r in rows]
            observed = [b for b in bounds if b is not None]
            bank_means = []
            for lo in (0, 12):
                bank = [
                    r["guarded_either_bound"]
                    for r in rows
                    if lo <= r["author"] < lo + 12
                    and r["guarded_either_bound"] is not None
                ]
                bank_means.append(float(np.mean(bank)) if bank else None)
            summaries.append(
                {
                    "source_account": field,
                    "target": q,
                    "observed_authors": len(observed),
                    "unobserved_authors": 24 - len(observed),
                    "both_orientations_conflict_authors": sum(
                        b is not None and b > 0 for b in bounds
                    ),
                    "increasing_conflict_authors": sum(
                        r["increasing_guarded_bound"] is not None
                        and r["increasing_guarded_bound"] > 0
                        for r in rows
                    ),
                    "either_bound_mean": float(np.mean(observed)) if observed else None,
                    "either_bound_median": float(np.median(observed))
                    if observed
                    else None,
                    "either_bound_bank_means": bank_means,
                }
            )
    shared_summary = []
    for field in cfg["source_accounts"]:
        rows = [r for r in shared_rows if r["source_account"] == field]
        comparable = [r for r in rows if r["guarded_bound"] is not None]
        shared_summary.append(
            {
                "source_account": field,
                "observed_authors": len(comparable),
                "unobserved_authors": 24 - len(comparable),
                "conflict_authors": sum(r["guarded_bound"] > 0 for r in comparable),
                "guarded_constraints": sum(r["guarded_constraints"] for r in rows),
                "possible_unordered_pairs": sum(
                    r["possible_unordered_pairs"] for r in rows
                ),
                "conditional_bound_mean": float(
                    np.mean([r["guarded_bound"] for r in comparable])
                )
                if comparable
                else None,
            }
        )
    for name, rows in (
        ("scalar_bounds.csv", scalar_rows),
        ("shared_bounds.csv", shared_rows),
        ("pair_accounts.csv", pairs),
    ):
        R.table(out / name, rows)
    R.write(out / "witnesses.json", witnesses)
    R.write(
        out / "summary.json",
        {
            "packet": "HM249",
            "post_outcome_diagnostic": True,
            "scalar": summaries,
            "shared_profile": shared_summary,
            "unique_inputs": sum(len(k) for k in kept),
            "aliases": aliases,
            "pair_accounts": len(pairs),
            "addressed_witnesses": len(witnesses),
            "reader_error": manifest["reader_error"],
            "compute_seconds": time.monotonic() - started,
            "new_model_API_private_training_calls": 0,
            "fit_predictions": 0,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "scalar_bounds.csv",
            "shared_bounds.csv",
            "pair_accounts.csv",
            "witnesses.json",
            "summary.json",
        ),
    )
    print(
        "HM249 analyzed: "
        + str(len(pairs))
        + " dependent pair accounts, no calibration fit",
        flush=True,
    )


def literal_bound(source, reader, source_error, reader_error):
    """Independent scalar-loop enumeration of exact and guarded order constraints."""
    n, gap, guarded_gap, count, guarded_count, violations, guarded_violations = (
        len(reader),
        0.0,
        0.0,
        0,
        0,
        0,
        0,
    )
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = [float(source[j, k] - source[i, k]) for k in range(source.shape[1])]
            dy = float(reader[i] - reader[j])
            if all(x >= 0 for x in d):
                count += 1
                violations += dy > 0
                gap = max(gap, dy / 2)
            if all(x > 2 * source_error for x in d):
                guarded_count += 1
                guarded_violations += dy > 2 * reader_error
                guarded_gap = max(guarded_gap, dy / 2 - reader_error)
    return {
        "exact_constraints": count,
        "guarded_constraints": guarded_count,
        "exact_violations": violations,
        "guarded_violations": guarded_violations,
        "exact_bound": gap if count else None,
        "guarded_bound": guarded_gap if guarded_count else None,
    }


def verify(out):
    """Replay every bound and original source/native endpoint in alternate arithmetic."""
    import csv

    from scripts import run_grain_selector_matrix_hm248 as S

    manifest = R.check(out, ("analyzed_seal.json",))
    cfg, z = manifest["config"], H.selected(out / "observations.npz", KEYS)
    _, _, kept, aliases = observations(out)
    scalar = list(csv.DictReader((out / "scalar_bounds.csv").open()))
    shared = list(csv.DictReader((out / "shared_bounds.csv").open()))
    errors, checked, count_errors = [], 0, 0
    for rows, is_shared in ((scalar, False), (shared, True)):
        for row in rows:
            u, a = (
                int(row["author"]),
                cfg["source_accounts"].index(row["source_account"]),
            )
            keep = kept[u]
            if is_shared:
                v = np.array(cfg["evaluation_direction"])
                x = z["source"][u, keep, a] * v
                y = z["native"][u, keep] @ v / 5
                checks = [
                    (
                        "",
                        literal_bound(
                            x, y, cfg["source_error"], manifest["reader_error"]
                        ),
                    )
                ]
            else:
                q = cfg["targets"].index(row["target"])
                x, y = z["source"][u, keep, a, q, None], z["native"][u, keep, q]
                checks = [
                    (
                        o + "_",
                        literal_bound(
                            sign * x, y, cfg["source_error"], manifest["reader_error"]
                        ),
                    )
                    for o, sign in (("increasing", 1), ("decreasing", -1))
                ]
            for prefix, values in checks:
                for key, value in values.items():
                    saved = row[prefix + key]
                    checked += 1
                    if value is None:
                        count_errors += saved != ""
                    elif "bound" in key:
                        errors.append(abs(float(saved) - value))
                    else:
                        count_errors += int(saved) != value
    p, source_parent = ROOT / cfg["selector_parent"], ROOT / cfg["source_parent"]
    old = H.selected(p / "observations.npz", ("selections",))
    source = H.selected(source_parent / "observations.npz", ("accounts",))["accounts"]
    source_error = max(
        float(np.max(abs(z["source"][u] - source[u, old["selections"][u].reshape(10)])))
        for u in range(24)
    )
    logits = S.native(p)["logits"].reshape(24, 10, 5, 10)
    odds = np.empty((24, 10, 5))
    for u, j, q in np.ndindex(odds.shape):
        values = logits[u, j, q]
        hi, lo = max(values[5:]), max(values[:5])
        odds[u, j, q] = (
            hi
            + math.log(math.fsum(math.exp(v - hi) for v in values[5:]))
            - lo
            - math.log(math.fsum(math.exp(v - lo) for v in values[:5]))
        )
    native_error = float(np.max(abs(odds - z["native"])))
    pair_rows = list(csv.DictReader((out / "pair_accounts.csv").open()))
    pair_errors, pair_checks, witness_errors = [], 0, []
    for row in pair_rows:
        u, i, j = (int(row[k]) for k in ("author", "point_i", "point_j"))
        a, q = (
            cfg["source_accounts"].index(row["source_account"]),
            cfg["targets"].index(row["target"]),
        )
        ds = float(z["source"][u, j, a, q] - z["source"][u, i, a, q])
        dy = float(odds[u, j, q] - odds[u, i, q])
        guarded = abs(ds) > 2 * cfg["source_error"]
        conflict = guarded and ds * dy < 0 and abs(dy) > 2 * manifest["reader_error"]
        pair_errors.extend(
            (abs(float(row["source_delta"]) - ds), abs(float(row["reader_delta"]) - dy))
        )
        count_errors += row["source_order_guarded"] != str(guarded)
        count_errors += row["increasing_conflict"] != str(conflict)
        count_errors += int(row["retained_condition_difference"]) != int(
            z["retained_conditions"][u, j] - z["retained_conditions"][u, i]
        )
        count_errors += int(row["retained_token_difference"]) != int(
            z["retained_lengths"][u, j] - z["retained_lengths"][u, i]
        )
        pair_checks += 1
    addresses = R.read(out / "addresses.json")
    for row in R.read(out / "witnesses.json"):
        u, a = row["author"], cfg["source_accounts"].index(row["source_account"])
        lower, upper = row["lower_point"], row["upper_point"]
        i, j = lower["point"], upper["point"]
        if lower != addresses[u * 10 + i] or upper != addresses[u * 10 + j]:
            raise ValueError("Witness addresses changed")
        if row["target"] == "fixed_shared_v":
            v = np.array(cfg["evaluation_direction"])
            ds = (z["source"][u, j, a] - z["source"][u, i, a]) * v
            dy = float((odds[u, i] - odds[u, j]) @ v / 5)
        else:
            q = cfg["targets"].index(row["target"])
            sign = 1 if row["orientation"] == "increasing" else -1
            ds = np.atleast_1d(
                sign * (z["source"][u, j, a, q] - z["source"][u, i, a, q])
            )
            dy = float(odds[u, i, q] - odds[u, j, q])
        if (
            not np.all(ds > 2 * cfg["source_error"])
            or dy <= 2 * manifest["reader_error"]
        ):
            raise ValueError("No guarded contradiction in saved witness")
        witness_errors.extend(
            (
                float(np.max(abs(ds - row["source_gap"]))),
                abs(dy - row["reader_gap_opposed_to_order"]),
                abs(
                    dy / 2
                    - manifest["reader_error"]
                    - row["empirical_adjusted_half_gap"]
                ),
            )
        )
    if (
        count_errors
        or max(errors + pair_errors + witness_errors, default=0) > 1e-12
        or source_error != 0
        or native_error > 1e-12
    ):
        raise ValueError(
            "Order or original endpoint replay mismatch; no tolerance rescue"
        )
    result = {
        "PASS": True,
        "literal_order_accounts": checked,
        "count_or_missing_errors": count_errors,
        "bound_max_error": max(errors, default=0),
        "original_source_error": source_error,
        "original_native_odds_error": native_error,
        "original_endpoint_replays": 1200,
        "literal_pair_accounts": pair_checks,
        "pair_max_error": max(pair_errors, default=0),
        "witness_max_error": max(witness_errors, default=0),
        "alias_pairs": len(aliases),
        "same_agent_alternate_not_blind": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(result, flush=True)


def main():
    """Execute one immutable finite stage, never an autonomous precision search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "analyze", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("analyzed_seal.json", "verification_seal.json"))
        print("HM249 all code/input/result seals PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
