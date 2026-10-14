#!/usr/bin/env python3
"""Qualify source-anchor/native action contacts once, without new model calls."""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_profile_reader as P
from grain_research.construction import math_same_event_multitarget as N
from grain_research.construction import math_source_anchor_exchange as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_same_event_multitarget_hm199 as H

CONFIG = "configs/grain_source_anchor_exchange_hm212.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_anchor_exchange_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SOURCE_ANCHOR_EXCHANGE_PLAN_HM212.md",
    "grain_research/construction/math_source_anchor_exchange.py",
    "scripts/run_grain_source_anchor_exchange_hm212.py",
    "tests/test_grain_source_anchor_exchange.py",
    "grain_research/construction/math_profile_reader.py",
    "grain_research/construction/math_same_event_multitarget.py",
    "scripts/run_grain_observed_family_hm170.py",
)
OPERATIONS = ("deletion", "isolated")
ACCOUNTS = ("raw", "common", "specific")
UNITS = ("score", "log_odds", "geometric_head", "candidate_mass")
VIEWS = ("view0", "view1", "mean")


def arrays(path):
    """Read only local numeric derivatives, with no pickle or private text."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Authenticate same-event inputs and seal a new immutable finite packet."""
    cfg = R.read(ROOT / CONFIG)
    vp, np_ = (ROOT / cfg[k] for k in ("vector_parent", "native_parent"))
    R.check(vp, ("compiled_seal.json", "scored_seal.json", "verification_seal.json"))
    H.check(np_)
    H.H.check_hashes(np_, R.read(np_ / "analyzed_seal.json"))
    H.H.check_hashes(np_, R.read(np_ / "verification_seal.json"))
    v, n, a = (
        arrays(vp / "observations.npz"),
        arrays(np_ / "00/observations.npz"),
        arrays(np_ / "analysis.npz"),
    )
    if set(v["source_indices"]) & set(v["panel_indices"]):
        raise ValueError("Source/panel overlap")
    if v["source_vectors"].shape != (386, 2, 12, 1024) or n["panel_vectors"].shape != (
        24,
        2,
        12,
        1024,
    ):
        raise ValueError("Changed registered event support")
    if not np.array_equal(v["panel_vectors"][:, 0], n["panel_vectors"]) or any(
        not np.array_equal(v[k], n[k]) for k in ("selected", "control")
    ):
        raise ValueError("Different native evidence addresses or vectors")
    source = v["source_vectors"].mean(2)
    fields = np.stack(
        [P.construct(source[:, b], v["labels"])["fields"][:, 1] for b in range(2)]
    )
    baseline_error = float(np.max(abs(fields - n["fields"][:, :, 1, 0])))
    if baseline_error > 2e-12:
        raise ValueError("Changed old matched full-space fields")
    native = a["native_actions"].copy()
    geometry = a["common_logit_action"] + a["specific_logit_action"]
    geometry_error = float(np.max(abs(geometry - native[..., 2])))
    if geometry_error > 1e-4:
        raise ValueError("Old geometric-head/logit control discrepancy changed")
    native[..., 2] = geometry
    common, specific = N.target_account(native)
    common[..., 2], specific[..., 2] = (
        a["common_logit_action"],
        a["specific_logit_action"],
    )
    actions = np.stack((native, common, specific), axis=3)
    dx = np.stack(
        [
            [
                N.vector_actions(
                    n["panel_vectors"][u, b], n["selected"][u, b], n["control"][u, b]
                )
                for b in range(2)
            ]
            for u in range(24)
        ]
    )
    files = (
        vp / "observations.npz",
        vp / "manifest.json",
        vp / "compiled_seal.json",
        np_ / "analysis.npz",
        np_ / "00/observations.npz",
        np_ / "00/manifest.json",
        np_ / "verification.json",
    )
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "historical_exploratory_not_confirmatory": True,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        source_means=source,
        labels=v["labels"].astype(int),
        source_indices=v["source_indices"],
        panel_indices=v["panel_indices"],
        vector_actions=dx,
        native_accounts=actions,
        fields=fields,
    )
    R.write(
        out / "quality.json",
        {
            "source_authors": len(source),
            "panel_authors": len(dx),
            "source_events": 9264,
            "panel_events": 576,
            "source_panel_overlap": 0,
            "old_field_max_error": baseline_error,
            "stored_head_vs_digit_geometry_max_error": geometry_error,
            "all_stratum_counts": [
                M.design(v["labels"], q)[1].tolist() for q in range(5)
            ],
            "new_neural_API_private_train_raw_text_calls": 0,
        },
    )
    R.seal(
        out, "prepared_seal.json", ("manifest.json", "observations.npz", "quality.json")
    )
    print(
        "PREPARED HM212: same-event, five source-anchor assignment orbits", flush=True
    )


def contact_projections(z):
    """Project source vectors into fixed native-action covariance directions."""
    dx, response = z["vector_actions"], z["native_accounts"]
    a = np.empty((2, 3, 4, 3, 5, 1024))
    for op, account, unit, view in np.ndindex(2, 3, 4, 3):
        x = dx[:, view, op] if view < 2 else dx[:, :, op].mean(1)
        y = (
            response[:, view, op, account, :, unit]
            if view < 2
            else response[:, :, op, account, :, unit].mean(1)
        )
        a[op, account, unit, view] = N.covariances(x, y).T
    return np.einsum("ibd,ocuvqd->ibocuvq", z["source_means"], a)


def compute(out):
    """Compute one fixed orbit family, retaining all negative and cross-target contacts."""
    start = time.monotonic()
    cfg = R.check(out)["config"]
    z = arrays(out / "observations.npz")
    t = contact_projections(z)
    flat = t.reshape(len(t), -1)
    observed, sampled, variances, indices, joint, primary = [], [], [], [], [], []
    for q in range(5):
        ix = M.assignment_indices(
            z["labels"], q, cfg["reference_draws"], cfg["reference_seed"]
        )
        obs = M.statistic(flat, z["labels"], q).reshape(t.shape[1:])
        refs = M.references(flat, z["labels"], q, ix).reshape((len(ix),) + t.shape[1:])
        var = M.exact_moments(flat, z["labels"], q)[1].reshape(t.shape[1:])
        pair = t[:, :, 0, 2, 2, 2, :].reshape(len(t), 10)
        jc = M.exact_moments(pair, z["labels"], q, joint=True)[1]
        primary_t = t[:, :, 0, 2, 2, 2, q].mean(1)
        value = float(obs[:, 0, 2, 2, 2, q].mean())
        reference = refs[:, :, 0, 2, 2, 2, q].mean(1)
        variance = float(M.exact_moments(primary_t[:, None], z["labels"], q)[1][0])
        primary.append(
            {
                "target": cfg["targets"][q],
                "covariance": value,
                **M.rank_account(value, reference, variance),
            }
        )
        observed.append(obs)
        sampled.append(refs)
        variances.append(var)
        indices.append(ix)
        joint.append(jc)
    adjusted = M.holm([p["two_sided_rank"] for p in primary])
    for p, h in zip(primary, adjusted, strict=True):
        p["holm_two_sided_rank"] = h
    elapsed = time.monotonic() - start
    if elapsed > cfg["max_compute_seconds"]:
        raise TimeoutError("Fixed compute budget exceeded; no automatic retry")
    observed, sampled, variances = (
        np.array(observed),
        np.array(sampled),
        np.array(variances),
    )
    R.archive(
        out / "analysis.npz",
        observed=observed,
        sampled=sampled,
        variances=variances,
        indices=np.array(indices),
        primary_joint_covariance=np.array(joint),
    )
    rows = []
    for q, b, op, ac, un, vi, reader in np.ndindex(observed.shape):
        ix = (q, b, op, ac, un, vi, reader)
        ref = sampled[(q, slice(None), b, op, ac, un, vi, reader)]
        rows.append(
            {
                "source_target": cfg["targets"][q],
                "source_view": b,
                "operation": OPERATIONS[op],
                "account": ACCOUNTS[ac],
                "unit": UNITS[un],
                "panel_view": VIEWS[vi],
                "reader_target": cfg["targets"][reader],
                "covariance": float(observed[ix]),
                **M.rank_account(float(observed[ix]), ref, float(variances[ix])),
            }
        )
    R.table(out / "contacts.csv", rows)
    R.table(out / "primary.csv", primary)
    R.write(
        out / "summary.json",
        {
            "packet": "HM212",
            "primary": primary,
            "reference_draws_per_target": cfg["reference_draws"],
            "source_authors": 386,
            "panel_authors": 24,
            "reference_assignment_count": 5 * cfg["reference_draws"],
            "all_contact_accounts": len(rows),
            "orbit_mean": "exactly_zero_by_finite_assignment_law",
            "shared_source_view_permutations": True,
            "new_neural_API_private_train_raw_text_calls": 0,
            "parameter_search": False,
            "recorded_compute_seconds": elapsed,
            "historical_exploratory_not_confirmatory": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "computed_seal.json",
        ("analysis.npz", "contacts.csv", "primary.csv", "summary.json"),
    )
    print("COMPUTED HM212", primary, "seconds", elapsed, flush=True)


def verify(out):
    """Replay all assignments via literal means and a second variance calculation."""
    start = time.monotonic()
    cfg = R.check(out, ("computed_seal.json",))["config"]
    z, saved = arrays(out / "observations.npz"), arrays(out / "analysis.npz")
    t = contact_projections(z)
    y = z["labels"]
    profile = y @ (2 ** np.arange(5))
    original_counts = np.bincount(profile, minlength=32)
    max_error, count = 0.0, 0
    for q in range(5):
        b, counts, omega, _ = M.design(y, q)
        for j, row in enumerate(saved["indices"][q]):
            yh = y.copy()
            yh[:, q] = y[row, q]
            if not np.array_equal(
                np.delete(yh, q, 1), np.delete(y, q, 1)
            ) or not np.array_equal(
                np.bincount(yh @ (2 ** np.arange(5)), minlength=32), original_counts
            ):
                raise ValueError("Changed other-four rows or full32 profile table")
            direct = np.zeros(t.shape[1:])
            for c in range(16):
                direct += omega[c] * (
                    t[(b == c) & (yh[:, q] == 1)].mean(0)
                    - t[(b == c) & (yh[:, q] == 0)].mean(0)
                )
            max_error = max(
                max_error, float(np.max(abs(direct - saved["sampled"][q, j])))
            )
            count += direct.size
        variance = np.zeros(t.shape[1:])
        pair_cov = np.zeros((10, 10))
        for c in range(16):
            rows = t[b == c]
            centered = rows - rows.mean(0)
            coefficient_var = omega[c] ** 2 / counts[:, c].prod()
            variance += (
                coefficient_var * len(rows) / (len(rows) - 1) * (centered**2).sum(0)
            )
            pp = centered[:, :, 0, 2, 2, 2, :].reshape(len(rows), 10)
            pair_cov += coefficient_var * len(rows) / (len(rows) - 1) * (pp.T @ pp)
        max_error = max(
            max_error,
            float(np.max(abs(variance - saved["variances"][q]))),
            float(np.max(abs(pair_cov - saved["primary_joint_covariance"][q]))),
        )
    # Literal vector field -> panel response covariance, not the projected path.
    for q, b, op, ac, un, vi, reader in itertools.product(
        range(5), range(2), range(2), range(3), range(4), range(3), range(5)
    ):
        field = z["fields"][b, q]
        x = (
            z["vector_actions"][:, vi, op]
            if vi < 2
            else z["vector_actions"][:, :, op].mean(1)
        )
        r = (
            z["native_accounts"][:, vi, op, ac, reader, un]
            if vi < 2
            else z["native_accounts"][:, :, op, ac, reader, un].mean(1)
        )
        values = x @ field
        cv = sum((values - values.mean()) * (r - r.mean())) / 23
        max_error = max(
            max_error, float(abs(cv - saved["observed"][q, b, op, ac, un, vi, reader]))
        )
    if max_error > 2e-12:
        raise ValueError("Literal orbit/contact replay failed")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "max_error": max_error,
            "literal_reference_values": count,
            "literal_observed_contacts": saved["observed"].size,
            "complete32_profile_counts_preserved": True,
            "other_four_rows_preserved": True,
            "all_source_views_share_assignment": True,
            "same_agent_not_independent": True,
            "recorded_verify_seconds": time.monotonic() - start,
            "reference_specification": cfg["primary"],
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM212", max_error, count, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("ALL SEALS PASS HM212", flush=True)
    else:
        {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)
