#!/usr/bin/env python3
"""Connect baseline-equivalent author evidence to actual native intervention tests."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_response_observability as M
from grain_research.construction import math_joint_reading as J
from grain_research.construction import math_native_readout_location as N
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_joint_reading_hm106 as G
from scripts import run_grain_native_readout_location_hm108 as V

SPEC = ROOT / "configs/grain_author_response_observability_hm109.json"
CODE = (
    "configs/grain_author_response_observability_hm109.json",
    "docs/GRAIN_AUTHOR_RESPONSE_OBSERVABILITY_PLAN_HM109.md",
    "grain_research/construction/math_author_response_observability.py",
    "scripts/run_grain_author_response_observability_hm109.py",
    "scripts/score_grain_author_response_cold_hm109.py",
    "tests/test_grain_math_author_response_observability.py",
)
FIELDS = {"native": (0, 12), "local": (12, 18), "additive": (18, 30), "lift": (30, 42)}


def save(path, **arrays):
    """Save numeric research assets exclusively, preserving previous failures."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def write_table(path, rows):
    """Keep missing scientific values blank, never substitute a false zero."""
    columns = sorted({k for r in rows for k in r})
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def authenticate(spec):
    """Check the sealed actual sources without collecting additional model calls."""
    native, local = ROOT / spec["native_source"], ROOT / spec["local_source"]
    V.check(native)
    H.verify_hashes(native, H.load(native / "output_seal.json"))
    G.check(local)
    H.verify_hashes(local, H.load(local / "output_seal.json"))
    capture = ROOT / spec["capture"]
    status = H.load(capture / "status.json")
    if status["state"] != "BUDGET_STOP" or status["pending"] is not None:
        raise ValueError("Do not reuse a live or uncertain intervention collector")
    return native, local, capture


def prepare(out):
    """Freeze the finite comparison before source construction or application."""
    if out.exists():
        raise FileExistsError("Preserve existing packet and checkpoint")
    spec = H.load(SPEC)
    if any(
        spec[k] != 0
        for k in ("new_forwards", "predictor_fits", "api_calls", "private_reads")
    ):
        raise ValueError("No inference, fitting, API or private-read budget")
    native, local, _ = authenticate(spec)
    out.mkdir(parents=True)
    H.write(
        out / "manifest.json",
        {
            "spec": spec,
            "code": {k: H.sha(ROOT / k) for k in CODE},
            "native": H.seals(native, native),
            "local": H.seals(local, local),
        },
    )
    print(
        "HM109 PREPARED: qualified source accounts, no new model or fitting", flush=True
    )


def check(out):
    """Require unchanged code and source outputs for every phase."""
    manifest = H.load(out / "manifest.json")
    H.verify_hashes(ROOT, manifest["code"])
    native, local, capture = authenticate(manifest["spec"])
    H.verify_hashes(native, manifest["native"])
    H.verify_hashes(local, manifest["local"])
    return manifest["spec"], native, local, capture


def catalog(native, capture):
    """Bind group metadata and eligible Person-A corners, not model-made eligibility."""
    groups = H.load(capture / "00/groups.json")
    lookup = {tuple(g["coordinates"]): g for g in groups}
    with (native / "coverage.csv").open() as stream:
        coverage = list(csv.DictReader(stream))
    matched = {int(r["group"]) for r in coverage if r["matched"] == "True"}
    if len(lookup) != 72 or len(matched) != 43:
        raise ValueError("Actual HM107 coverage changed")
    for g in groups:
        for corner in (0, 1):
            if (
                g["rows"][corner]["owner"] != 2
                or g["rows"][corner]["level"] != (2, 0)[corner]
            ):
                raise ValueError("Target-A history qualification or polarity changed")
    strata = list(itertools.product(range(2), range(2), range(6)))
    eligible = [i for i, c in enumerate(strata) if lookup[(1, *c)]["group"] in matched]
    return (
        lookup,
        matched,
        strata,
        {"source_complete": list(range(24)), "source_known_matched": eligible},
    )


def bank_arrays(bank, strata, lookup, matched, native, local):
    """Read baseline N/G/L/R and only actually complete causal observations."""
    with np.load(local / "01/laws.npz", allow_pickle=False) as archive:
        labels, coefficients = archive["labels"], archive["coefficients"]
    row_lookup = {tuple(r): i for i, r in enumerate(labels)}
    observations, directed, random, complete = [], [], [], []
    with np.load(native / "logits.npz", allow_pickle=False) as archive:
        for c in strata:
            g = lookup[(bank, *c)]["group"]
            base, additive, lift = [], [], []
            for target in range(6):
                field = N.decompose(archive[f"g{g}__baseline__q{target}__logits"])
                base.append(field["corners"]["native"][:2])
                additive.append(field["corners"]["additive_readout"][:2])
                lift.append(field["corners"]["joint_field_lift"][:2])
            base, additive, lift = (
                np.stack(a, axis=1).reshape(2, 12) for a in (base, additive, lift)
            )
            local_values = J.reconstruct(coefficients[row_lookup[(bank, *c)], 0])[:2]
            observations.append(np.column_stack((base, local_values, additive, lift)))
            complete.append(g in matched)
            for arm, result in (("joint_removed", directed), ("random", random)):
                if g not in matched:
                    result.append(np.full((2, 12), np.nan))
                else:
                    values = [
                        N.readout(archive[f"g{g}__{arm}__q{t}__logits"][:2])
                        for t in range(6)
                    ]
                    result.append(base - np.stack(values, axis=1).reshape(2, 12))
    return {
        "observations": np.concatenate(observations),
        "groups": np.repeat(np.arange(24), 2),
        "directed": np.concatenate(directed),
        "random": np.concatenate(random),
        "complete": np.array(complete),
    }


def qr_hidden(b, d, rank):
    """Independent pivoted-QR realization of the same finite observation space."""
    norms = np.linalg.norm(b, axis=0)
    q, _, _ = qr(b / np.where(norms > 0, norms, 1), mode="economic", pivoting=True)
    q = q[:, :rank]
    return d - q @ (q.T @ d)


def numeric_accounts(source, selected, fraction):
    """Compile every target witness from source only and account for all observers."""
    ix = np.array([[2 * g, 2 * g + 1] for g in selected]).ravel()
    x, groups = source["observations"][ix], source["groups"][ix]
    directed, random = source["directed"][ix], source["random"][ix]
    if not np.isfinite(directed).all() or not np.isfinite(random).all():
        raise ValueError(
            "Source construction cannot consume missing intervention values"
        )
    observers = M.source_observers(x[:, :12], x[:, 12:18], x[:, 18:30], groups)
    rows, checks = (
        [],
        {"projection_QR": 0.0, "choice": 0.0, "baseline": 0.0, "closure": 0.0},
    )
    hidden_base = None
    for name, b in observers.items():
        for arm, d in (("directed", directed), ("random", random)):
            visible, hidden, meta = M.response_account(b, d)
            checks["projection_QR"] = max(
                checks["projection_QR"],
                float(np.max(abs(hidden - qr_hidden(b, d, meta["rank"])))),
            )
            checks["closure"] = max(
                checks["closure"], float(np.max(abs(d - visible - hidden)))
            )
            if name == "native" and arm == "directed":
                hidden_base = hidden
            for target, unit in itertools.product(range(6), range(2)):
                j = target * 2 + unit
                total = float(np.linalg.norm(d[:, j]))
                remaining = float(np.linalg.norm(hidden[:, j]))
                rows.append(
                    {
                        "observer": name,
                        "arm": arm,
                        "target_index": target,
                        "unit_index": unit,
                        "atoms": len(x),
                        "rank": meta["rank"],
                        "rank_tolerance": meta["tolerance"],
                        "unobserved_dimension": len(x) - meta["rank"],
                        "response_RMS": total / np.sqrt(len(x)),
                        "unobserved_response_RMS": remaining / np.sqrt(len(x)),
                        "unobserved_norm_ratio": remaining / total if total else None,
                    }
                )
    laws, active = M.positive_witnesses(hidden_base, fraction)
    delta = (laws[0] - laws[1]).T
    checks["choice"] = float(np.max(abs(delta @ M.choice_columns(groups))))
    checks["baseline"] = float(np.max(abs(delta @ x[:, :12])))
    if np.any(laws <= 0) or np.max(abs(laws.sum(axis=1) - 1)) > 1e-10:
        raise ValueError("Source witnesses are not interior normalized evidence laws")
    return (
        {
            "indices": ix,
            "groups": groups,
            "laws": laws,
            "active": active,
            "source_observations": x,
            "directed": directed,
            "random": random,
        },
        rows,
        checks,
    )


def build_sources(out, spec, native, local, lookup, matched, strata, cohorts):
    """Seal source-derived laws before application exports; no application fitting."""
    source = bank_arrays(0, strata, lookup, matched, native, local)
    compiled, accounts, checks = {}, [], {}
    arrays = {}
    for cohort, selected in cohorts.items():
        data, rows, error = numeric_accounts(source, selected, spec["profile_fraction"])
        compiled[cohort], checks[cohort] = data, error
        accounts.extend(dict(cohort=cohort, **r) for r in rows)
        arrays.update({f"{cohort}__{k}": v for k, v in data.items()})
    save(out / "source_bundle.npz", **arrays)
    write_table(out / "response_accounts.csv", accounts)
    H.write(
        out / "source_seal.json",
        {
            name: H.sha(out / name)
            for name in ("source_bundle.npz", "response_accounts.csv")
        },
    )
    return compiled, accounts, checks, source


def histories(laws, spec):
    """Generate new event histories outside the cold scorer, preserving law provenance."""
    addresses, meta = [], []
    for seed in spec["seeds"]:
        rng = np.random.default_rng(seed)
        for direction, sign in itertools.product(range(12), range(2)):
            addresses.append(
                rng.choice(
                    laws.shape[1],
                    spec["records_per_profile"],
                    p=laws[sign, :, direction],
                )
            )
            meta.append((seed, direction, sign))
    return np.asarray(addresses), np.asarray(meta)


def analyze(out):
    """Execute source compilation, new histories and truly separate cold scoring."""
    started = time.monotonic()
    spec, native, local, capture = check(out)
    lookup, matched, strata, cohorts = catalog(native, capture)
    compiled, accounts, checks, source = build_sources(
        out, spec, native, local, lookup, matched, strata, cohorts
    )
    banks = {0: source}
    for bank in (1, 2):
        banks[bank] = bank_arrays(bank, strata, lookup, matched, native, local)
    profiles, samples, coverage, commands = [], [], [], []
    cold_checks = {"cold_exact": 0.0, "paired_identity": 0.0}
    for cohort, data in compiled.items():
        addresses, meta = histories(data["laws"], spec)
        history_path = out / f"{cohort}__histories.npz"
        save(history_path, addresses=addresses, metadata=meta)
        delta = (data["laws"][0] - data["laws"][1]).T
        for bank, bank_data in banks.items():
            ix = data["indices"]
            x, groups = bank_data["observations"][ix], bank_data["groups"][ix]
            complete = bool(np.all(bank_data["complete"][cohorts[cohort]]))
            observation_path = out / f"{cohort}__bank{bank}__observations.npz"
            cold_path = out / f"{cohort}__bank{bank}__cold.npz"
            save(observation_path, observations=x, groups=groups)
            command = [
                sys.executable,
                str(ROOT / "scripts/score_grain_author_response_cold_hm109.py"),
                "--observations",
                str(observation_path),
                "--histories",
                str(history_path),
                "--output",
                str(cold_path),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            commands.append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "allowed_inputs": [str(observation_path), str(history_path)],
                    "truth_or_law_or_intervention_input": False,
                }
            )
            with np.load(cold_path, allow_pickle=False) as child:
                mean = child["mean"]
                direct = np.array([np.mean(x[row], axis=0) for row in addresses])
                cold_checks["cold_exact"] = max(
                    cold_checks["cold_exact"], float(np.max(abs(mean - direct)))
                )
                cold_checks["paired_identity"] = max(
                    cold_checks["paired_identity"],
                    float(np.max(abs(mean[:, :12] - mean[:, 18:30] - mean[:, 30:42]))),
                )
            contrast = delta @ x
            actual = {
                arm: delta @ bank_data[arm][ix] if complete else None
                for arm in ("directed", "random")
            }
            coverage.append(
                {
                    "cohort": cohort,
                    "bank": bank,
                    "groups": len(cohorts[cohort]),
                    "atoms": len(x),
                    "complete_intervention": complete,
                    "reason": "complete_actual_support"
                    if complete
                    else "missing_capture_not_zero",
                }
            )
            for direction, j in itertools.product(range(12), range(12)):
                profiles.append(
                    {
                        "cohort": cohort,
                        "bank": bank,
                        "direction": direction,
                        "active_source_witness": bool(data["active"][direction]),
                        "target_index": j // 2,
                        "unit_index": j % 2,
                        "native_difference": float(contrast[direction, j]),
                        "additive_difference": float(contrast[direction, 18 + j]),
                        "lift_difference": float(contrast[direction, 30 + j]),
                        "directed_response_difference": float(
                            actual["directed"][direction, j]
                        )
                        if complete
                        else None,
                        "random_response_difference": float(
                            actual["random"][direction, j]
                        )
                        if complete
                        else None,
                    }
                )
            for direction, target in itertools.product(range(12), range(6)):
                profiles.append(
                    {
                        "cohort": cohort,
                        "bank": bank,
                        "direction": direction,
                        "active_source_witness": bool(data["active"][direction]),
                        "target_index": target,
                        "unit_index": "local",
                        "local_difference": float(contrast[direction, 12 + target]),
                    }
                )
            sampled = mean.reshape(len(spec["seeds"]), 12, 2, 42)
            for s, direction, j in itertools.product(
                range(len(spec["seeds"])), range(12), range(12)
            ):
                samples.append(
                    {
                        "cohort": cohort,
                        "bank": bank,
                        "seed": spec["seeds"][s],
                        "direction": direction,
                        "target_index": j // 2,
                        "unit_index": j % 2,
                        "sample_native_difference": float(
                            sampled[s, direction, 0, j] - sampled[s, direction, 1, j]
                        ),
                        "population_native_difference": float(contrast[direction, j]),
                    }
                )
    for label, rows in (
        ("profile_accounts", profiles),
        ("sample_accounts", samples),
        ("coverage", coverage),
    ):
        write_table(out / f"{label}.csv", rows)
    H.write(out / "cold_commands.json", commands)
    if (
        max(v for row in checks.values() for v in row.values()) > spec["tolerance"]
        or max(cold_checks.values()) > spec["tolerance"]
    ):
        raise ValueError("Fixed numeric/interface checks failed; retain partial packet")
    summary = summarize(accounts, profiles, coverage)
    summary.update(
        {
            "status": "FINITE_HM109_EVALUATED_NOT_GLOBAL",
            "classification": spec["classification"],
            "checks": checks | {"cold": cold_checks},
            "seconds": time.monotonic() - started,
            "new_forwards": 0,
            "predictor_fits": 0,
            "original_scope_complete": False,
            "new_simulated_histories": len(cohorts) * len(spec["seeds"]) * 24,
            "new_simulated_event_records": len(cohorts)
            * len(spec["seeds"])
            * 24
            * spec["records_per_profile"],
            "cold_processes": len(commands),
            "fresh_intervention_missing": True,
            "fields": FIELDS,
            "targets": spec["targets"],
            "units": spec["units"],
        }
    )
    H.write(out / "summary.json", summary)
    H.write(out / "output_seal.json", H.seals(out, out))
    print(
        "HM109 EVALUATED",
        {
            "seconds": summary["seconds"],
            "checks": summary["checks"],
            "histories": summary["new_simulated_histories"],
        },
        flush=True,
    )


def summarize(accounts, profiles, coverage):
    """Summarize every observer/cohort/channel and retain absent response support."""
    result = {"cohorts": {}, "coverage": coverage}
    for cohort in sorted({r["cohort"] for r in accounts}):
        observers = {}
        for name in ("native", "native_local", "native_paired", "condition_native"):
            chosen = [
                r
                for r in accounts
                if r["cohort"] == cohort
                and r["observer"] == name
                and r["arm"] == "directed"
            ]
            observers[name] = {
                "rank": chosen[0]["rank"],
                "unobserved_dimension": chosen[0]["unobserved_dimension"],
                "mean_unobserved_norm_ratio": float(
                    np.mean([r["unobserved_norm_ratio"] for r in chosen])
                ),
                "ratios_all_12": [r["unobserved_norm_ratio"] for r in chosen],
            }
        banks = {}
        for bank in range(3):
            rows = [
                r
                for r in profiles
                if r["cohort"] == cohort
                and r["bank"] == bank
                and r["unit_index"] != "local"
            ]
            local = [
                r["local_difference"]
                for r in profiles
                if r["cohort"] == cohort
                and r["bank"] == bank
                and r["unit_index"] == "local"
            ]

            def rms(key, rows=rows):
                values = [r[key] for r in rows if r.get(key) is not None]
                return float(np.sqrt(np.mean(np.square(values)))) if values else None

            banks[str(bank)] = {
                k + "_RMS": rms(k)
                for k in (
                    "native_difference",
                    "additive_difference",
                    "lift_difference",
                    "directed_response_difference",
                    "random_response_difference",
                )
            }
            banks[str(bank)]["local_difference_RMS"] = float(
                np.sqrt(np.mean(np.square(local)))
            )
        result["cohorts"][cohort] = {"observers": observers, "profile_contrasts": banks}
    return result


def verify(out):
    """Check raw provenance, alternate projections, exact laws and actual cold outputs."""
    started = time.monotonic()
    spec, native, local, capture = check(out)
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    H.verify_hashes(out, H.load(out / "source_seal.json"))
    lookup, matched, strata, cohorts = catalog(native, capture)
    replay_accounts = []
    with np.load(out / "source_bundle.npz", allow_pickle=False) as source:
        for cohort, selected in cohorts.items():
            expected, rows, _ = numeric_accounts(
                bank_arrays(0, strata, lookup, matched, native, local),
                selected,
                spec["profile_fraction"],
            )
            for name, value in expected.items():
                if not np.array_equal(source[f"{cohort}__{name}"], value):
                    raise ValueError("Source construction replay changed: " + name)
            replay_accounts.extend(dict(cohort=cohort, **row) for row in rows)
    with (out / "response_accounts.csv").open() as stream:
        actual_accounts = list(csv.DictReader(stream))
    columns = sorted({k for row in replay_accounts for k in row})
    rendered = [
        {k: "" if row.get(k) is None else str(row.get(k, "")) for k in columns}
        for row in replay_accounts
    ]
    if actual_accounts != rendered:
        raise ValueError("Full response account table did not replay")
    replay_profiles = []
    error = 0.0
    with (out / "profile_accounts.csv").open() as stream:
        replay_profiles = list(csv.DictReader(stream))
    with (out / "sample_accounts.csv").open() as stream:
        replay_samples = list(csv.DictReader(stream))
    replay_coverage, typed_profiles = [], []
    for cohort in cohorts:
        with np.load(out / "source_bundle.npz", allow_pickle=False) as data:
            laws = data[f"{cohort}__laws"]
        for bank in range(3):
            with np.load(
                out / f"{cohort}__bank{bank}__observations.npz", allow_pickle=False
            ) as obs:
                x, groups = obs["observations"], obs["groups"]
            actual_bank = bank_arrays(bank, strata, lookup, matched, native, local)
            ix = np.array([[2 * g, 2 * g + 1] for g in cohorts[cohort]]).ravel()
            if not np.array_equal(
                x, actual_bank["observations"][ix]
            ) or not np.array_equal(groups, actual_bank["groups"][ix]):
                raise ValueError("Export no longer matches sealed baseline material")
            complete = bool(np.all(actual_bank["complete"][cohorts[cohort]]))
            replay_coverage.append(
                {
                    "cohort": cohort,
                    "bank": bank,
                    "groups": len(cohorts[cohort]),
                    "atoms": len(x),
                    "complete_intervention": complete,
                    "reason": "complete_actual_support"
                    if complete
                    else "missing_capture_not_zero",
                }
            )
            with (
                np.load(
                    out / f"{cohort}__histories.npz", allow_pickle=False
                ) as saved_histories,
                np.load(
                    out / f"{cohort}__bank{bank}__cold.npz", allow_pickle=False
                ) as cold,
            ):
                ids = saved_histories["addresses"]
                expected_ids, expected_meta = histories(laws, spec)
                if not np.array_equal(ids, expected_ids) or not np.array_equal(
                    saved_histories["metadata"], expected_meta
                ):
                    raise ValueError("Seeded histories changed")
                for i, row in enumerate(ids):
                    value = sum((x[e] for e in row), start=np.zeros(42)) / len(row)
                    error = max(error, float(np.max(abs(value - cold["mean"][i]))))
                    for j, group in enumerate(np.unique(groups)):
                        picked = [x[e] for e in row if groups[e] == group]
                        if int(cold["counts"][i, j]) != len(picked):
                            raise ValueError("Conditional sample counts changed")
                        if picked:
                            error = max(
                                error,
                                float(
                                    np.max(
                                        abs(
                                            np.mean(picked, axis=0)
                                            - cold["conditional_mean"][i, j]
                                        )
                                    )
                                ),
                            )
                        elif not np.isnan(cold["conditional_mean"][i, j]).all():
                            raise ValueError("Missing condition mean is not unobserved")
                    if not np.array_equal(
                        np.bincount(row, minlength=len(x)), cold["event_counts"][i]
                    ):
                        raise ValueError("Atom event count changed")
                sample_means = cold["mean"].reshape(len(spec["seeds"]), 12, 2, 42)
                for row in replay_samples:
                    if row["cohort"] != cohort or int(row["bank"]) != bank:
                        continue
                    direction = int(row["direction"])
                    j = 2 * int(row["target_index"]) + int(row["unit_index"])
                    s = spec["seeds"].index(int(row["seed"]))
                    expected = (
                        sample_means[s, direction, 0, j]
                        - sample_means[s, direction, 1, j]
                    )
                    error = max(
                        error, abs(expected - float(row["sample_native_difference"]))
                    )
                    contrast = laws[0, :, direction] - laws[1, :, direction]
                    error = max(
                        error,
                        abs(
                            float(contrast @ x[:, j])
                            - float(row["population_native_difference"])
                        ),
                    )
            for row in replay_profiles:
                if row["cohort"] != cohort or int(row["bank"]) != bank:
                    continue
                d = (
                    laws[0, :, int(row["direction"])]
                    - laws[1, :, int(row["direction"])]
                )
                numeric = dict(row)
                for key in ("bank", "direction", "target_index"):
                    numeric[key] = int(row[key])
                numeric["active_source_witness"] = (
                    row["active_source_witness"] == "True"
                )
                if row["unit_index"] == "local":
                    column = 12 + int(row["target_index"])
                    key = "local_difference"
                    numeric[key] = float(row[key])
                else:
                    column = 2 * int(row["target_index"]) + int(row["unit_index"])
                    key = "native_difference"
                    numeric["unit_index"] = int(row["unit_index"])
                    for label, offset in (
                        ("native_difference", 0),
                        ("additive_difference", 18),
                        ("lift_difference", 30),
                    ):
                        value = sum(
                            float(d[e]) * float(x[e, offset + column])
                            for e in range(len(d))
                        )
                        error = max(error, abs(value - float(row[label])))
                        numeric[label] = float(row[label])
                    for arm in ("directed", "random"):
                        label = arm + "_response_difference"
                        if complete:
                            value = sum(
                                float(d[e]) * float(actual_bank[arm][ix[e], column])
                                for e in range(len(d))
                            )
                            error = max(error, abs(value - float(row[label])))
                            numeric[label] = float(row[label])
                        elif row[label] != "":
                            raise ValueError("Missing response must remain blank")
                        else:
                            numeric[label] = None
                value = sum(float(d[e]) * float(x[e, column]) for e in range(len(d)))
                error = max(error, abs(value - float(row[key])))
                typed_profiles.append(numeric)
    with (out / "coverage.csv").open() as stream:
        actual = list(csv.DictReader(stream))
    columns = sorted({k for row in replay_coverage for k in row})
    if actual != [{k: str(row[k]) for k in columns} for row in replay_coverage]:
        raise ValueError("Actual causal support table changed")
    summary = H.load(out / "summary.json")
    recomputed = summarize(replay_accounts, typed_profiles, replay_coverage)
    if any(summary[key] != recomputed[key] for key in recomputed):
        raise ValueError("Scientific summary did not replay from full tables")
    if error > spec["tolerance"]:
        raise ValueError("Alternate direct author replay failed")
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM109_NOT_GLOBAL",
            "alternate_error": error,
            "seconds": time.monotonic() - started,
            "same_agent_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    print("HM109 VERIFIED", {"alternate_error": error}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "analyze", "verify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/grain_math_response_v1/author_response_observability_r1",
    )
    args = parser.parse_args()
    globals()[args.phase](args.output.resolve())
