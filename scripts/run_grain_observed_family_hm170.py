#!/usr/bin/env python3
"""Construct one finite author-linked source-observation family, without grids."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observed_family as F
from scripts import run_grain_configuration_kernel_hm144 as K

CONFIG = "configs/grain_observed_family_hm170.json"
BUNDLE = "results/grain_construction_v1/C4/transport_r1/anonymous_bundle.json"
INPUTS = K.INPUTS | {"anonymous_bundle": BUNDLE}
SCIENCE = (
    CONFIG,
    "docs/GRAIN_OBSERVED_FAMILY_PLAN_HM170.md",
    "grain_research/construction/math_observed_family.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/score_grain_observed_family_cold.py",
    "scripts/verify_grain_observed_family_hm170.py",
    "tests/test_grain_observed_family.py",
    "scripts/run_grain_configuration_kernel_hm144.py",
)
DEFAULT = ROOT / "results/grain_math_response_v1/observed_author_family_r1"


def read(path):
    """Read a scientific JSON without raw material or credential access."""
    return json.loads(Path(path).read_text())


def digest(path):
    """Stream an input/artifact identity, never interpret unauthorized NPZ keys."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Write an immutable registered result once."""
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def table(path, rows):
    """Write full finite result rows without selecting favorable cells."""
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive(path, **arrays):
    """Save only locally retained numerical derivatives, once."""
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def seal(out, name, files):
    """Bind completed stages without modifying previous results."""
    write(out / name, {p: digest(out / p) for p in files})


def check(out, stages=()):
    """Verify method, input, event and completed-output identities."""
    manifest = read(out / "manifest.json")
    for category in ("code", "inputs"):
        for p, h in manifest[category].items():
            if digest(ROOT / p) != h:
                raise ValueError("Changed registered " + p)
    for stage in ("prepared_seal.json",) + tuple(stages):
        for p, h in read(out / stage).items():
            if digest(out / p) != h:
                raise ValueError("Changed sealed result " + p)
    return manifest


def prepared(out):
    """Load only this packet's numerical event observations."""
    with np.load(out / "observations.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Register scope and chronology before source function construction."""
    spec = read(ROOT / CONFIG)
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "config": spec,
            "code": {p: digest(ROOT / p) for p in SCIENCE},
            "inputs": {p: digest(ROOT / p) for p in INPUTS.values()},
            "cohort_reuse": "historical application cohort;exploratory not lockbox",
            "original_scope_complete": False,
        },
    )
    vectors, direction, venue, times, lengths, roles = K.load_numeric()
    old = read(ROOT / BUNDLE)
    if not np.array_equal(direction, np.asarray(old["directions"])):
        raise ValueError("Frozen C3/C4 anonymous directions disagree")
    if {k: len(v) for k, v in roles.items()} != spec["roles"]:
        raise ValueError("Changed author roles")
    coordinates = vectors @ direction
    order = np.array([np.lexsort((np.arange(36), row)) for row in times])
    ordered = lambda a: np.take_along_axis(a, order, axis=1)
    coordinates = np.take_along_axis(coordinates, order[:, :, None], axis=1)
    venue = ordered(venue)
    times, lengths = ordered(times), ordered(lengths)
    names = sorted(set(venue.ravel()))
    codes = {s: i for i, s in enumerate(names)}
    venue_codes = np.array([[codes[v] for v in row] for row in venue], int)
    conditions = np.isin(venue, old["context_rule"]["top"]).astype(int)
    pairs, ties = [], 0
    for u in range(624):
        for window, (start, stop) in enumerate(spec["windows"]):
            for t in range(start, stop - 1):
                if times[u, t + 1] <= times[u, t]:
                    ties += 1
                    continue
                pairs.append((u, window, t, t + 1))
    pairs = np.array(pairs, int)
    counts = np.bincount(pairs[:, 0], minlength=624)
    if np.any(counts == 0):
        raise ValueError(
            "No paired observations for an author; scope cannot shrink silently"
        )
    archive(
        out / "observations.npz",
        coordinates=coordinates,
        conditions=conditions,
        times=times,
        lengths=lengths,
        venue_codes=venue_codes,
        original_units=order,
        pairs=pairs,
        **roles,
    )
    quality = {
        "authors": 624,
        "records": 624 * 36,
        "dimensions": 15,
        "role_counts": {k: len(v) for k, v in roles.items()},
        "strictly_ordered_pairs": len(pairs),
        "timestamp_ties_not_scored": ties,
        "cross_window_pairs": 0,
        "source_directions_replayed": True,
        "observed_chronological_span_median_seconds": float(
            np.median(times[:, -1] - times[:, 0])
        ),
        "next_means": "next cached record;real intervening records/opportunities unknown",
        "raw_text_or_scale_labels_read": False,
        "new_neural_forwards": 0,
    }
    write(out / "input_quality.json", quality)
    with (out / "data_schema.md").open("x") as f:
        f.write(
            "# HM170 numerical data schema\n\n"
            + "624 authors,36 cached comment vectors/author,source403/validation101/held120.\n"
            + "E5 normalized comment vectors -> frozen C3 anonymous15 coordinates.\n"
            + "Author/event/unit,coarse recorded venue,timestamp,span metadata only; no texts/scales.\n"
            + f"{len(pairs)} eligible within-window pairs;{ties} timestamp ties excluded.\n"
            + "Next cached record is not next actual human action. All aliases remain anonymous.\n"
        )
    seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "input_quality.json", "data_schema.md"),
    )
    print(json.dumps(quality))


def paired_arrays(z, pairs, thresholds):
    """Bind current coordinates to three observations on one later event."""
    u, _, t, nxt = pairs.T
    x = z["coordinates"][u, t]
    later = z["coordinates"][u, nxt][:, F.INDICES] > thresholds
    y = np.column_stack([z["venue_codes"][u, t] == z["venue_codes"][u, nxt], later])
    return x, z["conditions"][u, t], y.astype(float), u


def compile_source(out):
    """Fit only source author observations and freeze the probability units."""
    check(out)
    z = prepared(out)
    p = z["pairs"][np.isin(z["pairs"][:, 0], z["source_train"])]
    cut = np.median(z["coordinates"][p[:, 0], p[:, 2]][:, F.INDICES], axis=0)
    x, c, y, u = paired_arrays(z, p, cut)
    law = F.compile_law(x, c, y, u)
    if not np.array_equal(cut, law["thresholds"]):
        raise ValueError("Endpoint and source partition units disagree")
    F.verify_law(law)
    write(out / "source_law.json", law)
    # The stronger comparator uses exact recorded venue, without changing the score.
    counts = np.bincount(u, minlength=624)
    weight = 1 / (len(z["source_train"]) * counts[u])
    vc = z["venue_codes"][u, p[:, 2]]
    table_by_venue = {}
    for v in sorted(set(vc)):
        ix = vc == v
        table_by_venue[str(v)] = (weight[ix] @ y[ix] / weight[ix].sum()).tolist()
    write(
        out / "source_comparators.json",
        {
            "exact_venue": table_by_venue,
            "fallback": "source condition mean;flagged,not unseen-venue evidence",
        },
    )
    source_early = np.array([z["conditions"][a, :18].mean() for a in z["source_train"]])
    source_span = np.array(
        [z["times"][a, 17] - z["times"][a, 0] for a in z["source_train"]]
    )
    write(
        out / "source_strata.json",
        {
            "choice_cut": float(np.median(source_early)),
            "span_cut_seconds": float(np.median(source_span)),
            "definition": "2*(choice>=cut)+(span>=cut)",
        },
    )
    seal(
        out,
        "compiled_seal.json",
        ("source_law.json", "source_comparators.json", "source_strata.json"),
    )
    print(
        json.dumps(
            {
                "source_authors": law["source_authors"],
                "source_events": law["source_events"],
                "reference": law["reference"],
                "cell_support": law["support"],
                "neural_forwards": 0,
            }
        )
    )


def cold(out):
    """Export current events only and score in a separate numeric process."""
    check(out, ("compiled_seal.json",))
    z = prepared(out)
    with (out / "cold_inputs.jsonl").open("x") as f:
        for u in range(624):
            for window in range(2):
                p = z["pairs"][(z["pairs"][:, 0] == u) & (z["pairs"][:, 1] == window)]
                events = [
                    {
                        "event_id": f"event_{u}_{int(z['original_units'][u, t])}",
                        "condition": int(z["conditions"][u, t]),
                        "coordinates": z["coordinates"][u, t].tolist(),
                    }
                    for t in p[:, 2]
                ]
                f.write(
                    json.dumps({"author_key": f"window_{u}_{window}", "events": events})
                    + "\n"
                )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_observed_family_cold.py"),
            "--bundle",
            str(out / "source_law.json"),
            "--sha256",
            digest(out / "source_law.json"),
            "--input",
            str(out / "cold_inputs.jsonl"),
            "--output",
            str(out / "cold_scores.jsonl"),
        ],
        check=True,
    )
    seal(out, "cold_seal.json", ("cold_inputs.jsonl", "cold_scores.jsonl"))


def correlation(x, y):
    """Report an actual observed association, preserving undefined constants."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def centered_by_group(x, g):
    """Center a diagnostic for matched permutations, not measurement outputs."""
    result = np.array(x, float)
    for k in sorted(set(g)):
        mask = g == k
        result[mask] -= result[mask].mean(0)
    return result


def analyze(out):
    """Examine every role/target and matched author links; never optimize them."""
    manifest = check(out, ("compiled_seal.json", "cold_seal.json"))
    started = time.perf_counter()
    z, law = prepared(out), read(out / "source_law.json")
    comps, cuts = (
        read(out / "source_comparators.json"),
        read(out / "source_strata.json"),
    )
    cold_scores = [
        json.loads(line)
        for line in (out / "cold_scores.jsonl").read_text().splitlines()
    ]
    if len(cold_scores) != 1248 or any(
        r["status"] != "OBSERVED_SOURCE_FUNCTION" for r in cold_scores
    ):
        write(
            out / "unsupported_cold_status.json",
            {"statuses": [r["status"] for r in cold_scores]},
        )
        raise ValueError("Scope lacks complete source cells; no rescue")
    arrays = {
        k: np.array([r[k] for r in cold_scores]).reshape(624, 2, -1)
        for k in ("mean", "relative", "choice", "within", "empirical_mass")
    }
    observed = np.full((624, 2, 3), np.nan)
    event_rows, brier = [], np.zeros((624, 2, 3, 3))
    fallback_count = np.zeros((624, 2), int)
    p_all = z["pairs"]
    x, c, y, u = paired_arrays(z, p_all, np.asarray(law["thresholds"]))
    prob = F.event_probabilities(law, x, c)
    mu = np.asarray(law["condition_reference"])
    exact = np.array(
        [
            comps["exact_venue"].get(str(v), mu[ci].tolist())
            for v, ci in zip(z["venue_codes"][u, p_all[:, 2]], c, strict=True)
        ]
    )
    fallback = np.array(
        [str(v) not in comps["exact_venue"] for v in z["venue_codes"][u, p_all[:, 2]]]
    )
    for author in range(624):
        for win in range(2):
            ix = (u == author) & (p_all[:, 1] == win)
            observed[author, win] = y[ix].mean(0)
            brier[author, win] = np.stack(
                [((v[ix] - y[ix]) ** 2).mean(0) for v in (prob, mu[c], exact)], axis=-1
            )
            fallback_count[author, win] = fallback[ix].sum()
            for j, target in enumerate(F.TARGETS):
                event_rows.append(
                    {
                        "author_index": author,
                        "window": win,
                        "target": target,
                        "events": int(ix.sum()),
                        "score": float(arrays["relative"][author, win, j]),
                        "choice": float(arrays["choice"][author, win, j]),
                        "within": float(arrays["within"][author, win, j]),
                        "observed_rate": float(observed[author, win, j]),
                        "brier_full": float(brier[author, win, j, 0]),
                        "brier_condition": float(brier[author, win, j, 1]),
                        "brier_exact_venue": float(brier[author, win, j, 2]),
                        "unseen_venue_fallbacks": int(fallback_count[author, win]),
                    }
                )
    share = z["conditions"][:, :18].mean(1)
    span = z["times"][:, 17] - z["times"][:, 0]
    strata = 2 * (share >= cuts["choice_cut"]).astype(int) + (
        span >= cuts["span_cut_seconds"]
    ).astype(int)
    rng = np.random.default_rng(manifest["config"]["permutation_seed"])
    metric_rows, perm_arrays, summary_roles = [], {}, {}
    for role in ("source_train", "source_validation", "held"):
        ids = z[role]
        group = strata[ids]
        permutations = np.tile(np.arange(len(ids)), (199, 1))
        for p in permutations:
            for g in sorted(set(group)):
                ix = np.flatnonzero(group == g)
                p[ix] = rng.permutation(ix)
        xx = centered_by_group(arrays["relative"][ids, 0], group)
        yy = centered_by_group(observed[ids, 1], group)
        statistic = np.mean(xx * yy, axis=0)
        null = np.array([np.mean(xx * yy[p], axis=0) for p in permutations])
        tails = (1 + np.sum(np.abs(null) >= np.abs(statistic)[None], axis=0)) / 200
        perm_arrays[role + "_indices"] = permutations
        perm_arrays[role + "_statistics"] = null
        perm_arrays[role + "_observed"] = statistic
        summary_roles[role] = {
            "authors": len(ids),
            "strata_counts": np.bincount(group, minlength=4).tolist(),
            "early_later_observed_correlations": [
                correlation(arrays["relative"][ids, 0, j], observed[ids, 1, j])
                for j in range(3)
            ],
            "early_later_score_correlations": [
                correlation(
                    arrays["relative"][ids, 0, j], arrays["relative"][ids, 1, j]
                )
                for j in range(3)
            ],
            "matched_covariance": statistic.tolist(),
            "matched_two_sided_tail_ranks": tails.tolist(),
        }
        for win in range(2):
            for j, target in enumerate(F.TARGETS):
                metric_rows.append(
                    {
                        "role": role,
                        "window": win,
                        "target": target,
                        "authors": len(ids),
                        "score_sd": float(arrays["relative"][ids, win, j].std()),
                        "choice_sd": float(arrays["choice"][ids, win, j].std()),
                        "within_sd": float(arrays["within"][ids, win, j].std()),
                        "mean_observed": float(observed[ids, win, j].mean()),
                        "mean_reading": float(arrays["mean"][ids, win, j].mean()),
                        "brier_full": float(brier[ids, win, j, 0].mean()),
                        "brier_condition": float(brier[ids, win, j, 1].mean()),
                        "brier_exact_venue": float(brier[ids, win, j, 2].mean()),
                        "early_later_observed_r": summary_roles[role][
                            "early_later_observed_correlations"
                        ][j],
                        "early_later_score_r": summary_roles[role][
                            "early_later_score_correlations"
                        ][j],
                        "matched_tail_rank": float(tails[j]),
                    }
                )
    account_error = float(
        np.max(np.abs(arrays["relative"] - arrays["choice"] - arrays["within"]))
    )
    if account_error > 1e-12:
        raise ValueError("Choice/conditional account broken")
    table(out / "author_window_accounts.csv", event_rows)
    table(out / "metrics_by_role_target.csv", metric_rows)
    archive(out / "permutation_accounts.npz", **perm_arrays)
    archive(
        out / "author_accounts.npz",
        **arrays,
        observed=observed,
        brier=brier,
        strata=strata,
        fallback_count=fallback_count,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "source_reference": law["reference"],
        "source_cells": law["support"],
        "role_results": summary_roles,
        "author_windows": 1248,
        "targets": list(F.TARGETS),
        "account_error": account_error,
        "seconds": elapsed,
        "new_neural_forwards": 0,
        "predictor_hyperparameter_search": False,
        "source_calibration_estimated": True,
        "same_event_outcomes_are_not_independent_trials": True,
        "human_personality_validity": False,
        "original_scope_complete": False,
    }
    write(out / "summary.json", summary)
    seal(
        out,
        "analysis_seal.json",
        (
            "author_window_accounts.csv",
            "metrics_by_role_target.csv",
            "permutation_accounts.npz",
            "author_accounts.npz",
            "summary.json",
        ),
    )
    if elapsed > manifest["config"]["analysis_seconds_cap"]:
        raise RuntimeError("Analysis cap exceeded;results kept,do not retry")
    print(json.dumps(summary))


def main():
    """Run one registered finite stage; never resume an old queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compile", "cold", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compile": compile_source, "cold": cold, "analyze": analyze}[
        args.stage
    ](args.out)


if __name__ == "__main__":
    main()
