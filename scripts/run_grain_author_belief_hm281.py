#!/usr/bin/env python3
"""One finite whole-path author-belief construction, not prediction tuning."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_belief as M
from scripts import run_grain_observed_family_hm170 as R
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_author_belief_hm281.json"
DEFAULT = ROOT / "results/grain_math_response_v1/author_belief_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_AUTHOR_BELIEF_PLAN_HM281.md",
    "grain_research/construction/math_author_belief.py",
    "scripts/run_grain_author_belief_hm281.py",
    "scripts/score_grain_author_belief_cold.py",
    "tests/test_grain_author_belief.py",
)
TRACE_KEYS = (
    "predictions",
    "logp",
    "profiles",
    "marginals",
    "effective_sources",
    "probability_shares_source_prior_application",
)


def prepare(out):
    """Freeze existing public numeric trajectories and complete source-only anchors."""
    if out.exists():
        raise FileExistsError("No overwrite or retry of a scientific packet")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, ("prepared_seal.json",))
    z = load_bundle(parent / "observations.npz")
    if (
        z["source_paths"].shape != (386, 2, 18)
        or z["held_paths"].shape != (120, 2, 18)
        or z["targets"].shape != (2, 386, 5)
        or np.intersect1d(z["source_indices"], z["held"]).size
        or np.bincount(z["bank"]).tolist() != [203, 183]
    ):
        raise ValueError("Changed source/held sampling scope")
    records = []
    for u, author in enumerate(z["held"]):
        record = {
            "author_key": f"held/{int(author)}",
            "episodes": [
                {
                    "events": [
                        {
                            "event_id": f"event/{int(author)}/{int(z['held_units'][u, w, t])}",
                            "timestamp": float(z["held_times"][u, w, t]),
                            "cell": int(z["held_paths"][u, w, t]),
                        }
                        for t in range(18)
                    ]
                }
                for w in range(2)
            ],
        }
        M.validate_record(record, 8)
        records.append(record)
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {f: R.digest(ROOT / f) for f in SCIENCE},
            "inputs": {
                str((parent / f).relative_to(ROOT)): R.digest(parent / f)
                for f in ("manifest.json", "prepared_seal.json", "observations.npz")
            },
            "original_scope_complete": False,
        },
    )
    donors = M.start_matched_donors(z["held_paths"][:, 0])
    R.archive(out / "observations.npz", **z, donors=donors)
    R.write(out / "records.json", records)
    R.write(out / "example_record.json", records[0])
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "held_authors": 120,
            "states": 8,
            "source_held_overlap": 0,
            "no_boundary_edge": True,
            "held_traits_not_read": True,
            "new_neural_API_private_training": 0,
            "historical_not_untouched_validation": True,
            "prior_support_not_observed_support": True,
            "first_state_matched_stranger_controls": int((donors >= 0).sum()),
            "stranger_initialization_is_counterfactual_not_same_author_record": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "records.json",
            "example_record.json",
            "input_quality.json",
        ),
    )
    R.write(out / "runtime.json", {"state": "PREPARED", "pid": None})
    print("HM281 prepared:386 SOURCE/120 held, zero neural calls")


def expired(_signum, _frame):
    """Terminate at the original finite CPU cap without model/parameter rescue."""
    raise TimeoutError("Original180 CPU-second cap exhausted")


def compute(out):
    """Compile priors once and predict BEFORE observing each held state."""
    import os

    cfg = R.check(out, ("prepared_seal.json",))["config"]
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("Do not repeat or resume this finite scientific computation")
    z = load_bundle(out / "observations.npz")
    started = time.monotonic()
    R.write(out / "runtime.json", {"state": "RUNNING", "pid": os.getpid()})
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["compute_seconds_cap"])
    try:
        result, paths = {}, z["held_paths"]
        for r, reference in enumerate(cfg["references"]):
            mask = z["bank"] == r if r < 2 else np.ones(386, bool)
            bundle = M.compile_source(
                z["source_paths"][mask],
                z["targets"][:, mask],
                z["source_indices"][mask],
                cfg["prior_row_mass"],
            )
            R.archive(out / f"source_{reference}.npz", **bundle)
            fresh, carry, strangers = ({k: [] for k in TRACE_KEYS} for _ in range(3))
            for author in range(120):
                fresh_author, carry_author, stranger_author = (
                    {k: [] for k in TRACE_KEYS} for _ in range(3)
                )
                for program in M.PROGRAMS:
                    rows = [M.observe(bundle, path, *program) for path in paths[author]]
                    for k in TRACE_KEYS:
                        fresh_author[k].append([row[k] for row in rows])
                    if program[0]:
                        late = M.observe(
                            bundle, paths[author, 1], *program, carry=rows[0]["carry"]
                        )
                        for k in TRACE_KEYS:
                            carry_author[k].append(late[k])
                        donor = int(z["donors"][author])
                        if donor >= 0:
                            foreign = M.observe(bundle, paths[donor, 0], *program)
                            stranger = M.observe(
                                bundle,
                                paths[author, 1],
                                *program,
                                carry=foreign["carry"],
                            )
                            for k in TRACE_KEYS:
                                stranger_author[k].append(stranger[k])
                        else:
                            for k in TRACE_KEYS:
                                stranger_author[k].append(np.full_like(late[k], np.nan))
                for k in TRACE_KEYS:
                    fresh[k].append(fresh_author[k])
                    carry[k].append(carry_author[k])
                    strangers[k].append(stranger_author[k])
            for k in TRACE_KEYS:
                result[f"{reference}/fresh/{k}"] = np.array(fresh[k])
                result[f"{reference}/carry/{k}"] = np.array(carry[k])
                result[f"{reference}/stranger/{k}"] = np.array(strangers[k])
        R.archive(out / "trajectories.npz", **result)
        runtime = {
            "state": "COMPUTED",
            "pid": None,
            "seconds": time.monotonic() - started,
            "new_neural_API_private_training": 0,
            "fresh_program_paths": 3 * 4 * 120 * 2,
            "carry_program_paths": 3 * 2 * 120,
            "matched_stranger_paths": 3 * 2 * int((z["donors"] >= 0).sum()),
            "original_scope_complete": False,
        }
        R.write(out / "runtime.json", runtime)
        R.seal(
            out,
            "computed_seal.json",
            ("trajectories.npz", "runtime.json")
            + tuple(f"source_{s}.npz" for s in cfg["references"]),
        )
        print(json.dumps(runtime))
    except BaseException as error:
        R.write(
            out / "runtime.json",
            {
                "state": "FAILED",
                "pid": None,
                "error": repr(error),
                "seconds": time.monotonic() - started,
                "no_retry": True,
            },
        )
        raise
    finally:
        signal.alarm(0)


def analyze(out):
    """Report signed mechanism contrasts; never select a program or tune it."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    a = load_bundle(out / "trajectories.npz")
    observations = load_bundle(out / "observations.npz")
    rng = np.random.default_rng(cfg["seed"])
    bootstrap = rng.integers(120, size=(cfg["bootstrap_authors"], 120))
    rows, author_rows, summaries = [], [], {}
    for reference in cfg["references"]:
        logp = a[f"{reference}/fresh/logp"]
        scores = logp[..., 1:].mean(-1)
        mean = scores.mean(2)
        carried = a[f"{reference}/carry/logp"][..., 1:].mean(-1)
        stranger = a[f"{reference}/stranger/logp"][..., 1:].mean(-1)
        contrasts = {
            "author_update_fixed": mean[:, 1] - mean[:, 0],
            "kernel_update_reset": mean[:, 2] - mean[:, 0],
            "kernel_update_shared": mean[:, 3] - mean[:, 1],
            "author_update_kernel": mean[:, 3] - mean[:, 2],
            "interaction": mean[:, 3] - mean[:, 2] - mean[:, 1] + mean[:, 0],
            "carry_fixed": carried[:, 0] - scores[:, 1, 1],
            "carry_kernel": carried[:, 1] - scores[:, 3, 1],
            "self_minus_stranger_fixed": carried[:, 0] - stranger[:, 0],
            "self_minus_stranger_kernel": carried[:, 1] - stranger[:, 1],
            "stranger_minus_fresh_fixed": stranger[:, 0] - scores[:, 1, 1],
            "stranger_minus_fresh_kernel": stranger[:, 1] - scores[:, 3, 1],
        }
        for name, delta in contrasts.items():
            active = np.isfinite(delta)
            d = delta[active]
            if not len(d):
                raise ValueError(
                    "No stranger support: report construction unavailable, not zero"
                )
            sample = (
                bootstrap
                if len(d) == 120
                else rng.integers(len(d), size=(cfg["bootstrap_authors"], len(d)))
            )
            lo, hi = np.quantile(d[sample].mean(1), [0.025, 0.975])
            rows.append(
                {
                    "reference": reference,
                    "contrast": name,
                    "mean_nats_per_transition": float(d.mean()),
                    "author_CI_low": float(lo),
                    "author_CI_high": float(hi),
                    "positive_authors": int((d > 0).sum()),
                    "authors": len(d),
                }
            )
            author_rows.extend(
                {
                    "reference": reference,
                    "contrast": name,
                    "author_index": u,
                    "signed_delta": float(v) if np.isfinite(v) else None,
                }
                for u, v in enumerate(delta)
            )
        first = a[f"{reference}/fresh/predictions"][:, :, :, 1]
        profiles = a[f"{reference}/fresh/profiles"]
        prior = load_bundle(out / f"source_{reference}.npz")
        bits = (prior["codes"][:, :, None] >> np.arange(5)) & 1
        reference_mean = bits.mean(1)
        # Full shared-kernel endpoint; preserve both assignments and both windows.
        target = a[f"{reference}/fresh/marginals"][:, 3, :, :, -1]
        summaries[reference] = {
            "fresh_logscore_programs_EARLY_LATE": scores.mean(0).tolist(),
            "first_transition_max_program_difference": float(
                abs(first - first[:, 0, None]).max()
            ),
            "posterior_profile_sum_max_error": float(abs(profiles.sum(-1) - 1).max()),
            "shared_kernel_final_effective_sources_mean": float(
                a[f"{reference}/fresh/effective_sources"][:, 3, :, -1].mean()
            ),
            "shared_kernel_transition_probability_shares_source_prior_application": a[
                f"{reference}/fresh/probability_shares_source_prior_application"
            ][:, 3, :, 1:]
            .mean((0, 1, 2))
            .tolist(),
            "shared_kernel_final_target_departure_RMS_assignments_windows_targets": np.sqrt(
                np.mean((target - reference_mean[None, None]) ** 2, axis=0)
            )
            .transpose(1, 0, 2)
            .tolist(),
            "target_reference_assignments": reference_mean.tolist(),
            "all_global_source_edges_observed": bool(np.all(prior["count"].sum(0) > 0)),
            "positive_new_path_support_supplied_by_declared_prior_not_empirical_proof": True,
            "first_state_matched_stranger_coverage": int(
                (observations["donors"] >= 0).sum()
            ),
            "stranger_is_counterfactual_initialization_not_full_context_match": True,
        }
    R.table(out / "contrasts.csv", rows)
    R.table(out / "author_contrasts.csv", author_rows)
    R.write(
        out / "summary.json",
        {
            "references": summaries,
            "runtime": R.read(out / "runtime.json"),
            "no_program_selection_or_accuracy_fit": True,
            "anchor_posteriors_not_human_trait_posteriors": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        ("contrasts.csv", "author_contrasts.csv", "summary.json"),
    )
    print(json.dumps(R.read(out / "summary.json")))


def literal_likelihood(bundle, episodes, kernel):
    """Independent rising-product likelihood without gamma functions or observe()."""
    n0, n = M.counts(episodes, bundle["alpha"].shape[-1])
    result = np.zeros(len(bundle["alpha0"]))
    for alpha, observed in [(bundle["alpha0"], n0)] + [
        (bundle["alpha"][:, a], n[a]) for a in range(len(n))
    ]:
        for b, count in enumerate(observed):
            for k in range(int(count)):
                result += np.log(alpha[:, b] + (k if kernel else 0))
        for k in range(int(observed.sum())):
            result -= np.log(alpha.sum(1) + (k if kernel else 0))
    return result


def verify(out):
    """Check ALL coherent shared-author endpoints via literal mixture products."""
    cfg = R.check(out, ("computed_seal.json", "analyzed_seal.json"))["config"]
    z, a = load_bundle(out / "observations.npz"), load_bundle(out / "trajectories.npz")
    likelihood_error, profile_error, count = 0.0, 0.0, 0
    for reference in cfg["references"]:
        b = load_bundle(out / f"source_{reference}.npz")
        for u in range(120):
            for mode, kernel in ((1, False), (3, True)):
                for w in range(2):
                    ell = literal_likelihood(b, [z["held_paths"][u, w]], kernel)
                    total = logsumexp(ell) - np.log(len(ell))
                    likelihood_error = max(
                        likelihood_error,
                        abs(total - a[f"{reference}/fresh/logp"][u, mode, w].sum()),
                    )
                    expected = M.profile(np.exp(ell - logsumexp(ell)), b["codes"])
                    profile_error = max(
                        profile_error,
                        float(
                            abs(
                                expected
                                - a[f"{reference}/fresh/profiles"][u, mode, w, :, -1]
                            ).max()
                        ),
                    )
                    count += 1
                ell0 = literal_likelihood(b, [z["held_paths"][u, 0]], kernel)
                both = literal_likelihood(b, z["held_paths"][u], kernel)
                total = logsumexp(both) - logsumexp(ell0)
                k = int(kernel)
                likelihood_error = max(
                    likelihood_error,
                    abs(total - a[f"{reference}/carry/logp"][u, k].sum()),
                )
                expected = M.profile(np.exp(both - logsumexp(both)), b["codes"])
                profile_error = max(
                    profile_error,
                    float(
                        abs(
                            expected - a[f"{reference}/carry/profiles"][u, k, :, -1]
                        ).max()
                    ),
                )
                count += 1
                donor = int(z["donors"][u])
                if donor >= 0:
                    early = [z["held_paths"][donor, 0]]
                    mixed = early + [z["held_paths"][u, 1]]
                    e0 = literal_likelihood(b, early, kernel)
                    em = literal_likelihood(b, mixed, kernel)
                    total = logsumexp(em) - logsumexp(e0)
                    likelihood_error = max(
                        likelihood_error,
                        abs(total - a[f"{reference}/stranger/logp"][u, k].sum()),
                    )
                    expected = M.profile(np.exp(em - logsumexp(em)), b["codes"])
                    profile_error = max(
                        profile_error,
                        float(
                            abs(
                                expected
                                - a[f"{reference}/stranger/profiles"][u, k, :, -1]
                            ).max()
                        ),
                    )
                    count += 1
    keys = [k for k in a if k.endswith("predictions")]
    for key, values in a.items():
        if "/stranger/" in key:
            active = z["donors"] >= 0
            if (
                not np.isfinite(values[active]).all()
                or not np.isnan(values[~active]).all()
            ):
                raise ArithmeticError("Stranger support/missing mask changed")
        elif not np.isfinite(values).all():
            raise ArithmeticError("Nonfinite supported trajectory")
    normal_error = max(float(np.nanmax(abs(a[k].sum(-1) - 1))) for k in keys)
    share_error = max(
        float(np.nanmax(abs(v.sum(-1) - 1)))
        for k, v in a.items()
        if k.endswith("probability_shares_source_prior_application")
    )
    report = {
        "PASS": max(likelihood_error, profile_error, normal_error, share_error) < 1e-10,
        "shared_path_endpoints": count,
        "literal_likelihood_error": float(likelihood_error),
        "literal_profile_error": profile_error,
        "prediction_normalization_error": normal_error,
        "probability_share_error": share_error,
        "same_agent_not_blind": True,
        "no_new_neural_calls": True,
        "matched_stranger_authors": int((z["donors"] >= 0).sum()),
    }
    if not report["PASS"]:
        raise ArithmeticError(json.dumps(report))
    cold = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_author_belief_cold.py"),
            "--bundle",
            str(out / "source_pooled.npz"),
            "--input",
            str(out / "example_record.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(cold.stdout)
    expected = a["pooled/carry/marginals"][0, 1, :, -1]
    report["cold_two_episode_marginal_error"] = float(
        abs(np.array(output["episodes"][1]["marginals"])[:, -1] - expected).max()
    )
    if report["cold_two_episode_marginal_error"] > 1e-12:
        raise ArithmeticError("Cold full-record marginal mismatch")
    R.write(out / "cold_readback.json", output)
    R.write(out / "verification.json", report)
    R.seal(out, "verification_seal.json", ("verification.json", "cold_readback.json"))
    print(json.dumps(report))


def main():
    """Execute declared finite stages; read-only check does not repeat an experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check")
    )
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    if a.stage == "check":
        R.check(
            a.output,
            (
                "prepared_seal.json",
                "computed_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        )
        print("HM281 seals PASS")
    else:
        globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
