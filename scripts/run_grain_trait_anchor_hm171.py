#!/usr/bin/env python3
"""Orient the unchanged HM170 author family using existing source-only labels."""

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
from grain_research.construction import math_trait_anchor as A
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_trait_anchor_hm171.json"
DEFAULT = ROOT / "results/grain_math_response_v1/trait_observation_anchor_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TRAIT_OBSERVATION_ANCHOR_PLAN_HM171.md",
    "grain_research/construction/math_trait_anchor.py",
    "scripts/run_grain_trait_anchor_hm171.py",
    "scripts/score_grain_trait_anchor_cold.py",
    "scripts/verify_grain_trait_anchor_hm171.py",
    "tests/test_grain_trait_anchor.py",
    "grain_research/vector_mechanisms/external_heads.py",
)
STAGES = ("compiled_seal.json", "cold_seal.json", "analysis_seal.json")


def labels(spec):
    """Read only authorized OOF author identifiers and existing Big5 outcomes."""
    with np.load(ROOT / spec["labels"], allow_pickle=False) as z:
        ids, y = z["author_ids"], z["y"]
    with np.load(ROOT / spec["material"], allow_pickle=False) as z:
        same_ids, same_y = z["author_ids"], z["y"]
    if (
        ids.shape != (624,)
        or y.shape != (624, 5)
        or len(set(ids.tolist())) != 624
        or not np.array_equal(ids, same_ids)
        or not np.array_equal(y, same_y)
        or not np.isfinite(y).all()
        or np.any((y < 0) | (y > 1))
        or tuple(spec["traits"]) != A.TRAITS
    ):
        raise ValueError("Existing source-only label identity/order/value mismatch")
    return y


def parent_check(spec):
    """Authenticate the complete old observation packet without rerunning it."""
    parent = ROOT / spec["parent"]
    R.check(parent, STAGES)
    return parent


def check(out, stages=()):
    """Verify all frozen scientific code, input bytes and completed stages."""
    manifest = R.read(out / "manifest.json")
    for category in ("code", "inputs"):
        for name, h in manifest[category].items():
            if R.digest(ROOT / name) != h:
                raise ValueError("Registered input/code changed: " + name)
    for stage in ("prepared_seal.json",) + tuple(stages):
        for name, h in R.read(out / stage).items():
            if R.digest(out / name) != h:
                raise ValueError("Sealed output changed: " + name)
    parent_check(manifest["config"])
    return manifest


def observations(out):
    """Load author distributions and roles, with no external outcomes."""
    with np.load(out / "observations.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def full_mass(z):
    """Give each author equal total mass, weighting windows by event counts."""
    return (z["mass"] * z["counts"][:, :, None]).sum(1) / z["counts"].sum(1)[:, None]


def prepare(out):
    """Seal the unchanged family and label join before trait associations."""
    spec = R.read(ROOT / CONFIG)
    parent = parent_check(spec)
    labels(spec)  # Alignment only; do not evaluate an application association.
    old, z = R.read(parent / "source_law.json"), R.prepared(parent)
    roles = ("source_train", "source_validation", "held")
    expected = (
        spec["source_authors"],
        spec["validation_authors"],
        spec["held_authors"],
    )
    if tuple(len(z[k]) for k in roles) != expected:
        raise ValueError("Changed frozen author roles")
    joined = np.concatenate([z[k] for k in roles])
    if not np.array_equal(np.sort(joined), np.arange(624)):
        raise ValueError("Author roles overlap or omit authors")
    state = R.F.states(z["coordinates"].reshape(-1, 15), old["thresholds"]).reshape(
        624, 36
    )
    pairs = z["pairs"]
    u, win, t, _ = pairs.T
    cell = 4 * z["conditions"][u, t] + state[u, t]
    count = np.zeros((624, 2, 8), int)
    np.add.at(count, (u, win, cell), 1)
    totals = count.sum(2)
    if not np.all(totals == 17):
        raise ValueError("Frozen two17-current-event windows changed")
    mass = count / totals[:, :, None]
    with np.load(parent / "author_accounts.npz", allow_pickle=False) as p:
        if not np.allclose(mass, p["empirical_mass"], rtol=0, atol=2e-15):
            raise ValueError("Direct current-event cells disagree with old cold mass")
        strata = p["strata"]
    out.mkdir(parents=True, exist_ok=False)
    parent_files = (
        "manifest.json",
        "prepared_seal.json",
        *STAGES,
        "observations.npz",
        "source_law.json",
        "source_strata.json",
        "cold_inputs.jsonl",
        "cold_scores.jsonl",
        "author_accounts.npz",
        "summary.json",
    )
    inputs = [spec["labels"], spec["material"]] + [
        str((parent / p).relative_to(ROOT)) for p in parent_files
    ]
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {p: R.digest(ROOT / p) for p in inputs},
            "label_permission": "existing Big5-only OOF source labels;no raw questionnaire/bridge/MEPS",
            "cohort_reuse": "exploratory historical cohort;not a new lockbox",
            "application_association_examined_before_seal": False,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        mass=mass,
        counts=totals,
        strata=strata,
        **{k: z[k] for k in roles},
    )
    quality = {
        "authors": 624,
        "windows": 1248,
        "current_events": int(totals.sum()),
        "all_events_per_window": 17,
        "labels_exactly_aligned": True,
        "trait_order": list(A.TRAITS),
        "label_missing": 0,
        "unchanged_parent_mass": True,
        "new_neural_forwards": 0,
        "new_private_questionnaire_or_raw_text_reads": 0,
        "old_author_scale_profiles_are_not_independent_comment_outcomes": True,
    }
    R.write(out / "input_quality.json", quality)
    with (out / "data_schema.md").open("x") as f:
        f.write(
            "# HM171 authorized numerical schema\n\n"
            "624 Big5-only authors;403 source/101 validation/120 historical held.\n"
            "Exactly HM170 current-event windows,8 cells,15 anonymous coordinates,0/5 cuts and TOP20 conditions.\n"
            "Old OOF author_ids/y match material exactly;trait order E,N,A,C,O;all finite0..1.\n"
            "No raw comments,new questionnaire,bridge,MEPS,teacher output or new embedding.\n"
            "Cold inputs contain current event IDs,condition and coordinates only.\n"
        )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "input_quality.json", "data_schema.md"),
    )
    print(json.dumps(quality))


def field_alignment(laws, reference):
    """Compare source-half directions in one shared, outcome-free geometry."""
    fields = []
    for b in laws:
        w = np.asarray(b["probabilities"], float) - b["prior"]
        if not np.isfinite(w).all():
            raise ValueError("Half-source cells unsupported;no smoothing rescue")
        fields.append(w - reference @ w)
    inner = (reference[:, None] * fields[0] * fields[1]).sum(0)
    norms = [np.sqrt((reference[:, None] * w**2).sum(0)) for w in fields]
    cos = np.divide(
        inner,
        norms[0] * norms[1],
        out=np.full(5, np.nan),
        where=(norms[0] * norms[1]) > 0,
    )
    return {
        "weighted_cross_inner_product": inner.tolist(),
        "weighted_norms": [v.tolist() for v in norms],
        "weighted_cosine": [None if not np.isfinite(v) else float(v) for v in cos],
        "geometry": "full-source8-cell mass;each half centered in this same geometry",
    }


def compile_source(out):
    """Calibrate all five fixed targets using only source author profiles."""
    manifest = check(out)
    spec, z = manifest["config"], observations(out)
    src = z["source_train"]
    source_y = labels(spec)[src]
    cuts = np.median(source_y, axis=0)
    binary = (source_y > cuts).astype(float)
    coords = R.read(ROOT / spec["parent"] / "source_law.json")["thresholds"]
    p = full_mass(z)[src]
    law = A.compile_law(p, binary, coords, cuts)
    A.verify_law(law)
    R.write(out / "source_law.json", law)
    halves = [np.arange(len(src))[k::2] for k in range(2)]
    half_laws = [A.compile_law(p[ix], binary[ix], coords, cuts) for ix in halves]
    for b in half_laws:
        A.verify_law(b)
    permutation = np.random.default_rng(spec["seed"] + 1).permutation(len(src))
    null_law = A.compile_law(p, binary[permutation], coords, cuts)
    A.verify_law(null_law)
    R.write(out / "source_permuted_law.json", null_law)
    R.write(
        out / "source_half_laws.json",
        {"source_positions": [h.tolist() for h in halves], "laws": half_laws},
    )
    ref = np.asarray(law["reference_mass"])
    w = np.asarray(law["probabilities"]) - law["prior"]
    diagnostics = field_alignment(half_laws, ref) | {
        "source_profile_permutation": permutation.tolist(),
        "source_prior": law["prior"],
        "label_cutoffs": cuts.tolist(),
        "all8_source_cells_supported": bool(np.all(ref > 0)),
        "target_field_gram": (w.T @ (ref[:, None] * w)).tolist(),
        "target_field_character_coefficients": (A.characters().T @ w / 8).tolist(),
        "half_source_laws_not_selected_or_averaged_for_application": True,
    }
    R.write(out / "source_diagnostics.json", diagnostics)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_law.json",
            "source_permuted_law.json",
            "source_half_laws.json",
            "source_diagnostics.json",
        ),
    )
    print(
        json.dumps(
            {
                "source_authors": len(src),
                "source_half_cosines": diagnostics["weighted_cosine"],
                "source_prior": law["prior"],
                "new_neural_forwards": 0,
            }
        )
    )


def cold(out):
    """Run a separate outcome-free reader on the exact old1248 input windows."""
    manifest = check(out, STAGES[:1])
    parent = ROOT / manifest["config"]["parent"]
    for name, output in (
        ("source_law.json", "cold_scores.jsonl"),
        ("source_permuted_law.json", "permuted_source_scores.jsonl"),
    ):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_trait_anchor_cold.py"),
                "--bundle",
                str(out / name),
                "--sha256",
                R.digest(out / name),
                "--input",
                str(parent / "cold_inputs.jsonl"),
                "--output",
                str(out / output),
            ],
            check=True,
        )
    R.seal(out, "cold_seal.json", ("cold_scores.jsonl", "permuted_source_scores.jsonl"))


def score_arrays(path):
    """Read all cold windows; unsupported statuses cannot silently shrink scope."""
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    if len(rows) != 1248 or any(r["status"] != "OBSERVED_TRAIT_FUNCTION" for r in rows):
        raise ValueError("Incomplete cold trait family;keep failure,do not rescue")
    for i, row in enumerate(rows):
        u, w = divmod(i, 2)
        if (
            row["author_key"] != f"window_{u}_{w}"
            or row["person_posterior"] is not False
        ):
            raise ValueError(
                "Changed outcome-free author address or score interpretation"
            )
    return {
        k: np.asarray([r[k] for r in rows]).reshape(624, 2, 8, 5)
        if k == "walsh_contributions"
        else np.asarray([r[k] for r in rows]).reshape(624, 2, -1)
        for k in ("mean", "relative", "choice", "within", "mass", "walsh_contributions")
    }


def covariance(x, y):
    """Keep signed additive contributions; correlations themselves do not add."""
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def fisher_interval(r, n):
    """Provide a descriptive Fisher-z interval across authors, not comments."""
    if r is None or n <= 3:
        return [None, None]
    if abs(r) >= 1:
        return [r, r]
    z, radius = np.arctanh(r), 1.959963984540054 / np.sqrt(n - 3)
    return np.tanh([z - radius, z + radius]).tolist()


def analyze(out):
    """Examine every target/component once; no application fitting or promotion."""
    manifest = check(out, STAGES[:2])
    started = time.perf_counter()
    spec, z = manifest["config"], observations(out)
    y = labels(spec)
    real = score_arrays(out / "cold_scores.jsonl")
    null = score_arrays(out / "permuted_source_scores.jsonl")
    if not np.allclose(real["mass"], z["mass"], rtol=0, atol=2e-15):
        raise ValueError("Trait anchor changed the author observation family")
    old_rows = [
        json.loads(s)
        for s in (ROOT / spec["parent"] / "cold_scores.jsonl").read_text().splitlines()
    ]
    old_scores = np.asarray([r["relative"] for r in old_rows]).reshape(624, 2, 3)
    walsh = real["walsh_contributions"]
    groups = {
        "condition_only": walsh[:, :, 4],
        "coordinate_only": walsh[:, :, [1, 2, 3]].sum(2),
        "condition_coordinate": walsh[:, :, [5, 6, 7]].sum(2),
    }
    error = max(
        float(np.max(np.abs(real["relative"] - real["choice"] - real["within"]))),
        float(np.max(np.abs(real["relative"] - walsh.sum(2)))),
    )
    if error > 1e-12:
        raise ValueError("Full score/component account disagrees")
    rng = np.random.default_rng(spec["seed"])
    rows, old_links, role_results, perms = [], [], {}, {}
    for role in ("source_train", "source_validation", "held"):
        ids = z[role]
        strata = z["strata"][ids]
        pp = np.tile(np.arange(len(ids)), (spec["permutations"], 1))
        for p in pp:
            for g in sorted(set(strata)):
                ix = np.flatnonzero(strata == g)
                p[ix] = rng.permutation(ix)
        xx = R.centered_by_group(real["relative"][ids], strata)
        yy = R.centered_by_group(y[ids], strata)
        stat = np.mean(xx * yy[:, None, :], axis=0)
        perm_stats = np.array([np.mean(xx * yy[p, None, :], axis=0) for p in pp])
        tails = (1 + (np.abs(perm_stats) >= np.abs(stat)).sum(0)) / (len(pp) + 1)
        (
            perms[role + "_indices"],
            perms[role + "_statistics"],
            perms[role + "_observed"],
        ) = pp, perm_stats, stat
        main_r, null_r = [], []
        for window in range(2):
            wr, nr = [], []
            for j, trait in enumerate(A.TRAITS):
                target = y[ids, j]
                score = real["relative"][ids, window, j]
                r = R.correlation(score, target)
                ni = R.correlation(null["relative"][ids, window, j], target)
                ci = fisher_interval(r, len(ids))
                cr, wi = real["choice"][ids, window, j], real["within"][ids, window, j]
                row = {
                    "role": role,
                    "window": window,
                    "trait": trait,
                    "authors": len(ids),
                    "score_trait_r": r,
                    "fisher_low": ci[0],
                    "fisher_high": ci[1],
                    "score_sd": float(score.std()),
                    "choice_trait_r": R.correlation(cr, target),
                    "within_trait_r": R.correlation(wi, target),
                    "full_trait_cov": covariance(score, target),
                    "choice_trait_cov": covariance(cr, target),
                    "within_trait_cov": covariance(wi, target),
                    "matched_covariance": float(stat[window, j]),
                    "matched_two_sided_tail_rank": float(tails[window, j]),
                    "one_source_profile_permutation_r": ni,
                }
                for name, values in groups.items():
                    row[name + "_trait_cov"] = covariance(
                        values[ids, window, j], target
                    )
                rows.append(row)
                wr.append(r)
                nr.append(ni)
                for k, old_target in enumerate(R.F.TARGETS):
                    old_links.append(
                        {
                            "role": role,
                            "window": window,
                            "endpoint": old_target,
                            "trait": trait,
                            "authors": len(ids),
                            "old_endpoint_trait_r": R.correlation(
                                old_scores[ids, window, k], target
                            ),
                        }
                    )
            main_r.append(wr)
            null_r.append(nr)
        role_results[role] = {
            "authors": len(ids),
            "strata_counts": np.bincount(strata, minlength=4).tolist(),
            "trait_r_by_window": main_r,
            "source_permutation_r_by_window": null_r,
            "score_early_later_r": [
                R.correlation(real["relative"][ids, 0, j], real["relative"][ids, 1, j])
                for j in range(5)
            ],
            "matched_two_sided_tail_ranks_by_window": tails.tolist(),
        }
    account_rows = []
    for u in range(624):
        for window in range(2):
            for j, trait in enumerate(A.TRAITS):
                account_rows.append(
                    {
                        "author_index": u,
                        "window": window,
                        "trait": trait,
                        "score": float(real["relative"][u, window, j]),
                        "choice": float(real["choice"][u, window, j]),
                        "within": float(real["within"][u, window, j]),
                        **{
                            name: float(values[u, window, j])
                            for name, values in groups.items()
                        },
                    }
                )
    R.table(out / "metrics_by_role_window_trait.csv", rows)
    R.table(out / "old_endpoint_trait_links.csv", old_links)
    R.table(out / "author_window_accounts.csv", account_rows)
    R.archive(
        out / "author_accounts.npz",
        **real,
        permuted_source_relative=null["relative"],
        old_scores=old_scores,
        strata=z["strata"],
    )
    R.archive(out / "permutation_accounts.npz", **perms)
    elapsed = time.perf_counter() - started
    summary = {
        "traits": list(A.TRAITS),
        "source_half_alignment": R.read(out / "source_diagnostics.json"),
        "roles": role_results,
        "author_windows": 1248,
        "main_rows": len(rows),
        "old_endpoint_link_rows": len(old_links),
        "account_error": error,
        "seconds": elapsed,
        "new_neural_forwards": 0,
        "application_fitting_or_target_selection": False,
        "predictor_accuracy_promotion": False,
        "raw_external_scale_links": True,
        "causal_personality_or_clinical_claim": False,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "metrics_by_role_window_trait.csv",
            "old_endpoint_trait_links.csv",
            "author_window_accounts.csv",
            "author_accounts.npz",
            "permutation_accounts.npz",
            "summary.json",
        ),
    )
    if elapsed > spec["max_seconds"]:
        raise RuntimeError("Finite analysis cap exceeded;keep results,no retry")
    print(json.dumps(summary))


def main():
    """Run exactly one immutable finite stage, never an old research queue."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compile", "cold", "analyze"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    {"prepare": prepare, "compile": compile_source, "cold": cold, "analyze": analyze}[
        args.stage
    ](args.out)


if __name__ == "__main__":
    main()
