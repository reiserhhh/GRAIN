#!/usr/bin/env python3
"""Construct and audit relative-author evidence without native teacher targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import binom
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_relative_choice as M

SPEC = ROOT / "configs/grain_relative_choice_hm140.json"
CODE = (
    "configs/grain_relative_choice_hm140.json",
    "docs/GRAIN_RELATIVE_CHOICE_PLAN_HM140.md",
    "grain_research/construction/math_relative_choice.py",
    "scripts/run_grain_relative_choice_hm140.py",
    "tests/test_grain_relative_choice.py",
)


def read(path):
    """Read an explicit local JSON dependency."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Create a new numerical artifact without replacing a historical file."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    """Hash a dependency without loading its native arrays into the constructor."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(path, rows):
    """Create a complete table, including nonidentifying and alias cases."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    """Freeze text-vector menus and the finite source-only inference dictionary."""
    if out.exists():
        raise FileExistsError("Preserve existing or failed run; no overwrite")
    spec = read(SPEC)
    source = ROOT / spec["source_fields"]
    application = ROOT / spec["application_encoder"]
    for name, path in (("source_seal", source), ("application_seal", application)):
        seal = read(ROOT / spec[name])
        if seal[str(path)] != sha(path):
            raise ValueError("Changed frozen vector input: " + str(path))
    with np.load(source, allow_pickle=False) as archive:
        source_g = archive["GRAIN_all15"].copy()
    with np.load(application, allow_pickle=False) as archive:
        application_g = archive["g"].reshape(2, 2, 6, 2, 2, 15).copy()
    if source_g.shape != (2, 2, 6, 4, 15):
        raise ValueError("Full source15/corner geometry required")
    addresses = []
    for role, key, corpus_key in (
        ("source", "source_records", "source"),
        ("new", "application_records", "corpus"),
    ):
        records = read(ROOT / spec[key])
        for bank, family in itertools.product(range(2), range(6)):
            pair = [
                next(
                    r
                    for r in records
                    if r[corpus_key] == 0
                    and r["bank"] == bank
                    and r["family"] == family
                    and r["b"] == b
                    and r["c"] == 1
                )
                for b in (-1, 1)
            ]
            addresses.append(
                {
                    "role": role,
                    "menu": bank * 6 + family,
                    "record_ids": [r["id"] for r in pair],
                    "qualifier": "voluntary_actual_action",
                    "offer_status": "hypothetically_assembled_not_historical_offer",
                }
            )
    raw_source = (source_g[0, :, :, 3] - source_g[0, :, :, 1]).reshape(12, 15)
    raw_application = (
        application_g[0, :, :, 1, 1] - application_g[0, :, :, 0, 1]
    ).reshape(12, 15)
    scale = float(np.sqrt(np.mean(np.sum(raw_source**2, axis=1))))
    if not np.isfinite(scale) or not scale > 0:
        raise ValueError("Nonzero source-only scalar menu unit required")
    d, new_d = raw_source / scale, raw_application / scale
    geometric = M.geometry(d, spec["rank_relative_tolerance"])
    axes = geometric["basis"][:, : spec["illustrative_dimensions"]].copy()
    for j in range(axes.shape[1]):
        if axes[np.argmax(abs(axes[:, j])), j] < 0:
            axes[:, j] *= -1
    grid = np.array(list(itertools.product(spec["prior_grid"], repeat=axes.shape[1])))
    out.mkdir(parents=True)
    np.savez(out / "dictionary.npz", menus=d, new_menus=new_d, axes=axes, grid=grid)
    write(out / "menu_addresses.json", addresses)
    write(
        out / "manifest.json",
        {
            "spec": spec,
            "code": {str(ROOT / p): sha(ROOT / p) for p in CODE},
            "inputs": {
                str(p): sha(p)
                for p in (
                    source,
                    application,
                    ROOT / spec["source_seal"],
                    ROOT / spec["application_seal"],
                    ROOT / spec["source_records"],
                    ROOT / spec["application_records"],
                )
            },
            "dictionary_sha256": sha(out / "dictionary.npz"),
            "source_only_menu_unit": scale,
            "source_menu_rank": geometric["rank"],
            "native_reader_arrays_loaded": False,
            "menu_origin": "hypothetical pairs of distinct voluntary actual-action records",
            "axes_scope": "2D declared-world demonstration;other13 not estimated as zero",
            "common_scale_assumption": "fixed logit utility scale;not empirically calibrated",
        },
    )


def check(out):
    """Verify constructor/source provenance without replaying neural work."""
    manifest = read(out / "manifest.json")
    for path, digest in (manifest["code"] | manifest["inputs"]).items():
        if sha(path) != digest:
            raise ValueError("Changed HM140 dependency: " + path)
    if sha(out / "dictionary.npz") != manifest["dictionary_sha256"]:
        raise ValueError("Changed frozen numerical dictionary")
    return manifest


def probes(d, new_d, axes):
    """Enumerate invariances and source-equivalent actions, with no fitted labels."""
    rows = []
    for menu, vector in enumerate(d @ axes):
        contrast = float(vector @ np.array([0.6, -0.9]))
        support, log_mass, _, variance = M.conditional_law(7, 9, 8, contrast)
        for bias, baseline in itertools.product((-3.0, 0.0, 2.0), (-1.0, 0.0, 1.0)):
            joint = binom.pmf(support, 7, expit(bias + baseline + contrast))
            joint *= binom.pmf(8 - support, 9, expit(bias + baseline))
            direct = joint / joint.sum()
            rows.append(
                {
                    "menu": menu,
                    "bias": bias,
                    "reference_level": baseline,
                    "contrast": contrast,
                    "conditional_variance": variance,
                    "enumeration_max_error": float(
                        np.max(abs(direct - np.exp(log_mass)))
                    ),
                }
            )
    geo = M.geometry(d)
    missing = new_d @ geo["null"]
    if missing.shape[1]:
        j = int(np.argmax(np.linalg.norm(missing, axis=1)))
        length = np.linalg.norm(missing[j])
        witness = geo["null"] @ (missing[j] / length) if length else geo["null"][:, 0]
    else:
        j, witness = 0, np.zeros(d.shape[1])
    reference = np.linspace(-0.4, 0.6, d.shape[1])
    shift = np.cos(np.arange(d.shape[1]))
    bias = np.sin(np.arange(len(d)))
    original = expit(bias + d @ reference)
    changed = expit((bias - d @ shift) + d @ (reference + shift))
    personal = axes @ np.array([0.9, -0.6])
    alias = expit(bias + d @ np.zeros(d.shape[1]) + d @ personal)
    declared = expit(bias + d @ personal)
    utility = bias + d @ reference
    temperature = 1.7
    temperature_alias = expit((2 * temperature) * (utility / 2))
    return rows, {
        "full_dimensions": d.shape[1],
        "source_rank": geo["rank"],
        "unidentified_dimensions": geo["null"].shape[1],
        "source_null_error": geo["null_error"],
        "new_union_rank": M.geometry(np.vstack([d, new_d]))["rank"],
        "source_equivalent_witness_error": float(np.max(abs(d @ witness))),
        "new_witness_logodds_change": float(np.max(abs(new_d @ witness))),
        "witness_new_menu": j,
        "new_menu_unseen_energy": float(np.sum(missing**2) / np.sum(new_d**2)),
        "common_shift_gauge_probability_error": float(np.max(abs(original - changed))),
        "personal_context_alias_probability_error": float(
            np.max(abs(alias - declared))
        ),
        "utility_temperature_alias_probability_error": float(
            np.max(abs(expit(temperature * utility) - temperature_alias))
        ),
        "conditional_enumeration_error": max(
            row["enumeration_max_error"] for row in rows
        ),
        "scope": "Fixed conditional-choice law,not a theorem identifying human traits",
    }


def make_counts(d, spec):
    """Generate paired declared worlds; keep truth separate from cold count inputs."""
    inputs, diagnostics = [], []
    patterns = list(itertools.product(spec["author_type_grid"], repeat=2))
    bias = 1.8 * np.sin(np.arange(len(d))) + np.linspace(-0.6, 0.6, len(d))
    for seed in spec["seeds"]:
        reference = np.random.default_rng(seed + 1000).binomial(
            spec["reference_trials_per_menu"], expit(bias)
        )
        for author, (pattern, repeat) in enumerate(
            itertools.product(patterns, range(spec["repetitions_per_type"]))
        ):
            theta = np.asarray(pattern)
            for world in spec["worlds"]:
                rng = np.random.default_rng(seed * 10000 + author)
                n = np.full(len(d), spec["free_trials_per_menu"])
                if world == "unequal_exposure":
                    n = np.rint(
                        6 + 36 * expit(theta[0] * np.linspace(-2, 2, len(d)))
                    ).astype(int)
                probability = expit(bias + d @ theta)
                x = rng.binomial(n, probability)
                rows = [
                    {
                        "menu": j,
                        "n": int(n[j]),
                        "success": int(x[j]),
                        "ref_n": spec["reference_trials_per_menu"],
                        "ref_success": int(reference[j]),
                        "qualification": "free",
                    }
                    for j in range(len(d))
                ]
                forced = rng.binomial(
                    spec["forced_trials_per_menu"], expit(np.linspace(-2, 2, len(d)))
                )
                rows += [
                    {
                        "menu": len(d) + j,
                        "n": spec["forced_trials_per_menu"],
                        "success": int(forced[j]),
                        "ref_n": 0,
                        "ref_success": 0,
                        "qualification": "forced",
                    }
                    for j in range(len(d))
                ]
                inputs.append(rows)
                diagnostics.append(
                    {
                        "seed": seed,
                        "author": author,
                        "world": world,
                        "repeat": repeat,
                        "expressive_contrast": theta.tolist(),
                        "asserted_stable_parameter": [0.0, 0.0]
                        if world == "personal_context_alias"
                        else theta.tolist(),
                        "personal_context_alias": world == "personal_context_alias",
                        "free_exposures": n.tolist(),
                    }
                )
    return inputs, diagnostics


def evaluate(inputs, diagnostics, d, grid):
    """Cold score every configuration; no reference refit or best-world selection."""
    rows, posterior, information = [], [], []
    dictionary = np.vstack([d, d])
    for sample, (records, meta) in enumerate(zip(inputs, diagnostics, strict=True)):
        score = M.cold_score(records, dictionary, grid)
        posterior.append(score["posterior"])
        information.append(score["information"])
        rows.append(
            {
                "sample": sample,
                "seed": meta["seed"],
                "author": meta["author"],
                "world": meta["world"],
                "status": score["status"],
                "U01_relative_mean": float(score["mean"][0]),
                "U02_relative_mean": float(score["mean"][1]),
                "U01_variance": float(score["covariance"][0, 0]),
                "U02_variance": float(score["covariance"][1, 1]),
                "posterior_cross_covariance": float(score["covariance"][0, 1]),
                "informative_menus": score["informative_menus"],
                "forced_trials_excluded": score["forced_trials_excluded"],
                "free_trials": sum(
                    r["n"] for r in records if r["qualification"] == "free"
                ),
                "reference_is_fixed": True,
            }
        )
    return rows, np.array(posterior), np.array(information)


def run(out):
    """Execute one finite CPU construction and preserve its complete counterexamples."""
    manifest = check(out)
    if (out / "output_seal.json").exists():
        raise FileExistsError("Completed run; use verify rather than rerun")
    spec = manifest["spec"]
    t0 = time.perf_counter()
    with np.load(out / "dictionary.npz", allow_pickle=False) as archive:
        d, new_d, axes, grid = (
            archive[k] for k in ("menus", "new_menus", "axes", "grid")
        )
    probe_rows, result = probes(d, new_d, axes)
    inputs, diagnostics = make_counts(d @ axes, spec)
    write(out / "cold_inputs.json", inputs)
    write(out / "diagnostics_only.json", diagnostics)
    write(
        out / "cold_inputs_seal.json",
        {
            "dictionary": sha(out / "dictionary.npz"),
            "count_inputs": sha(out / "cold_inputs.json"),
        },
    )
    rows, posterior, information = evaluate(inputs, diagnostics, d @ axes, grid)
    paired_errors = []
    authors_per_seed = len(spec["author_type_grid"]) ** 2 * spec["repetitions_per_type"]
    for seed in spec["seeds"]:
        for author in range(authors_per_seed):
            indices = [
                i
                for i, m in enumerate(diagnostics)
                if m["seed"] == seed and m["author"] == author
            ]
            shared = next(
                i for i in indices if diagnostics[i]["world"] == "shared_context"
            )
            alias = next(
                i
                for i in indices
                if diagnostics[i]["world"] == "personal_context_alias"
            )
            paired_errors.append(
                float(np.max(abs(posterior[shared] - posterior[alias])))
            )
    first = inputs[0]
    orthogonal = np.array([[0.6, -0.8], [0.8, 0.6]])
    rotation = M.cold_score(
        first, np.vstack([d @ axes, d @ axes]) @ orthogonal, grid @ orthogonal
    )
    status = M.cold_score([], d @ axes, grid)
    result.update(
        {
            "status": "FINITE_RELATIVE_AUTHOR_CONSTRUCTOR_WITH_EXACT_CONTEXT_ALIAS",
            "illustrative_dimensions": len(axes.T),
            "posterior_atoms": len(grid),
            "paired_author_parameter_configurations": authors_per_seed
            * len(spec["seeds"]),
            "world_configurations": len(rows),
            "menu_count_records": sum(map(len, inputs)),
            "independent_human_participants": 0,
            "new_neural_calls": 0,
            "native_target_training": False,
            "personal_context_alias_posterior_error": max(paired_errors),
            "coordinate_rotation_posterior_error": float(
                np.max(abs(rotation["posterior"] - posterior[0]))
            ),
            "empty_evidence_status": status["status"],
            "profile_status_counts": {
                s: sum(row["status"] == s for row in rows)
                for s in sorted({row["status"] for row in rows})
            },
            "cpu_seconds": time.perf_counter() - t0,
            "original_scope_complete": False,
            "decision": "Keep conditional expressive-choice comparison as one channel;shared nuisance cancels but personal context is observationally confounded.Neither more fitting nor low numerical error supplies psychological meaning",
        }
    )
    if (
        max(
            result[k]
            for k in (
                "conditional_enumeration_error",
                "common_shift_gauge_probability_error",
                "personal_context_alias_probability_error",
                "utility_temperature_alias_probability_error",
                "personal_context_alias_posterior_error",
                "coordinate_rotation_posterior_error",
            )
        )
        > spec["exact_tolerance"]
    ):
        raise ValueError("Construction invariant failed; preserve diagnostic artifacts")
    table(out / "conditional_invariance.csv", probe_rows)
    table(out / "author_profiles.csv", rows)
    np.savez(out / "profiles.npz", posterior=posterior, information=information)
    write(out / "result.json", result)
    outputs = [p for p in out.iterdir() if p.is_file() and not p.name.startswith("._")]
    write(out / "output_seal.json", {str(p): sha(p) for p in outputs})
    print(json.dumps(result, indent=2))


def verify(out):
    """Replay all profiles and independently enumerated laws from sealed inputs."""
    check(out)
    for path, digest in read(out / "output_seal.json").items():
        if sha(path) != digest:
            raise ValueError("Changed output: " + path)
    with np.load(out / "dictionary.npz", allow_pickle=False) as archive:
        d, new_d, axes, grid = (
            archive[k] for k in ("menus", "new_menus", "axes", "grid")
        )
    inputs, diagnostics = (
        read(out / "cold_inputs.json"),
        read(out / "diagnostics_only.json"),
    )
    rows, posterior, information = evaluate(inputs, diagnostics, d @ axes, grid)
    with np.load(out / "profiles.npz", allow_pickle=False) as archive:
        error = max(
            float(np.max(abs(posterior - archive["posterior"]))),
            float(np.max(abs(information - archive["information"]))),
        )
    with (out / "author_profiles.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(rows) or any(
        any(str(v) != old[k] for k, v in row.items())
        for old, row in zip(saved, rows, strict=True)
    ):
        raise ValueError("Profile table replay differs")
    probe_rows, result = probes(d, new_d, axes)
    with (out / "conditional_invariance.csv").open(newline="") as stream:
        saved_probes = list(csv.DictReader(stream))
    if len(saved_probes) != len(probe_rows) or any(
        any(str(v) != old[k] for k, v in row.items())
        for old, row in zip(saved_probes, probe_rows, strict=True)
    ):
        raise ValueError("Independent-law table replay differs")
    if error > 1e-12:
        raise ValueError("Cold numerical profile replay differs")
    print(
        "READONLY VERIFIED HM140",
        {
            "profiles": len(rows),
            "conditional_laws": len(probe_rows),
            "replay_error": error,
            **result,
        },
    )


def main():
    """Run fixed prepare/run/verify stages; never start a model or old queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / read(SPEC)["out"]
    with threadpool_limits(limits=1):
        {"prepare": prepare, "run": run, "verify": verify}[args.stage](out)


if __name__ == "__main__":
    main()
