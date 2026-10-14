#!/usr/bin/env python3
"""Trace source-qualified contacts through cached summary interventions once."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_anchor_carrier_contrasts as C
from grain_research.construction import math_source_anchor_exchange as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_anchor_exchange_hm212 as A
from scripts import run_grain_target_selector_exchange_hm200 as X

CONFIG = "configs/grain_anchor_carrier_contrasts_hm213.json"
DEFAULT = ROOT / "results/grain_math_response_v1/anchor_carrier_contrasts_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ANCHOR_CARRIER_CONTRASTS_PLAN_HM213.md",
    "grain_research/construction/math_anchor_carrier_contrasts.py",
    "scripts/run_grain_anchor_carrier_contrasts_hm213.py",
    "tests/test_grain_anchor_carrier_contrasts.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "grain_research/construction/math_same_event_multitarget.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_source_anchor_exchange_hm212.py",
)


def load_states(parent, modes, units):
    """Read only cached numeric final states, never tokens or neural weights."""
    states = np.empty((24, 2, 2, len(modes), 5, 1536))
    paths = []
    for batch, unit in enumerate(units):
        for mode_index, mode in enumerate(modes):
            path = parent / f"01/b{batch:03d}_{mode}.npz"
            with np.load(path, allow_pickle=False) as z:
                values = z["states"].astype(float)
            if values.shape != (5, 1536) or not np.isfinite(values).all():
                raise ValueError("Malformed cached endpoint")
            states[unit["author"], unit["view"], unit["arm"], mode_index] = values
            paths.append(path)
    return states[:, :, 0] - states[:, :, 1], paths


def prepare(out):
    """Authenticate same-event frames and freeze paired contrasts without fitting."""
    cfg = R.read(ROOT / CONFIG)
    ap, wp, ip = (
        ROOT / cfg[k] for k in ("anchor_parent", "whole_parent", "increment_parent")
    )
    R.check(ap, ("computed_seal.json", "verification_seal.json"))
    for parent in (wp, ip):
        X.check(parent)
        for name in (
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        ):
            X.H.check_hashes(parent, R.read(parent / name))
        if R.read(parent / "status.json")["state"] != "COLLECTED":
            raise ValueError("Parent collection not terminal and complete")
    X.H.check_hashes(ip, R.read(ip / "recovery_accounting_seal.json"))
    receipt = R.read(ip / "recovery_accounting.json")
    if not receipt["original_remains_FAILED"] or receipt["formula_change"]:
        raise ValueError("Original failure or scientific formula overwritten")
    units, other_units = (R.read(p / "00/units.json") for p in (wp, ip))
    addresses = [(u["author"], u["view"], u["arm"]) for u in units]
    expected = set(np.ndindex(24, 2, 2))
    if len(units) != 96 or set(addresses) != expected or units != other_units:
        raise ValueError("Different or duplicate SAME-event addresses")
    wz, iz = (A.arrays(p / "00/observations.npz") for p in (wp, ip))
    if any(not np.array_equal(wz[k], iz[k]) for k in wz):
        raise ValueError("Source-action/address frame changed")
    z, old = A.arrays(ap / "observations.npz"), A.arrays(ap / "analysis.npz")
    if set(z["source_indices"]) & set(z["panel_indices"]):
        raise ValueError("Source/panel overlap")
    frame = np.einsum("uvd,bqd->uvbq", z["vector_actions"][:, :, 0], z["fields"])
    source_frame_error = float(np.max(abs(frame - wz["source_actions"][..., 1, 0])))
    if source_frame_error > 2e-12:
        raise ValueError("Parent actions do not use the SAME matched source fields")
    for q in range(5):
        M.validate_assignments(z["labels"], q, old["indices"][q])
    whole, paths = load_states(wp, ("native", "common", "foreign"), units)
    increment, extra_paths = load_states(ip, ("zero", "common", "foreign"), units)
    paths += extra_paths
    wh, ih = (A.arrays(p / "01/head.npz") for p in (wp, ip))
    if any(not np.array_equal(wh[k], ih[k]) for k in wh):
        raise ValueError("Output head changed across interventions")
    weights = wh["weights"].astype(float)
    direction = weights[5:].mean(0) - weights[:5].mean(0)
    wa, ia = (A.arrays(p / "analysis.npz")["actions"] for p in (wp, ip))
    if not np.array_equal(wa[:, :, 0], ia[:, :, 0]):
        raise ValueError("Native responses changed across parents")
    mode_actions = np.concatenate((wa[:, :, :1], ia[:, :, 1:], wa[:, :, 1:]), axis=2)
    state_actions = np.concatenate(
        (whole[:, :, :1], increment, whole[:, :, 1:]), axis=2
    )
    head_values = state_actions @ direction
    head_digit_error = float(np.max(abs(head_values - mode_actions[..., 2])))
    if head_digit_error > 1e-4:
        raise ValueError("Historical head/digit control discrepancy exceeded")
    mode_actions[..., 2] = head_values
    response = C.accounts(mode_actions)
    native_error = float(np.max(abs(response[:, :, 0] - z["native_accounts"][:, :, 0])))
    files = [ap / "observations.npz", ap / "analysis.npz", ap / "summary.json"]
    for parent in (wp, ip):
        files += [
            parent / name
            for name in (
                "00/units.json",
                "00/observations.npz",
                "00/manifest.json",
                "analysis.npz",
                "verification.json",
                "01/head.npz",
                "status.json",
                "collected_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            )
        ]
    files += [
        ip / "recovery_accounting.json",
        ip / "recovery_accounting_seal.json",
    ] + paths
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
        source_means=z["source_means"],
        labels=z["labels"],
        fields=z["fields"],
        source_indices=z["source_indices"],
        panel_indices=z["panel_indices"],
        vector_actions=z["vector_actions"][:, :, 0],
        mode_actions=mode_actions,
        response=response,
        assignment_indices=old["indices"],
        previous_native_observed=old["observed"][:, :, 0],
        previous_native_sampled=old["sampled"][:, :, :, 0],
    )
    R.write(
        out / "quality.json",
        {
            "source_authors": 386,
            "panel_authors": 24,
            "numeric_endpoint_state_files": len(paths),
            "same_native_action_max_error": native_error,
            "old_native_cache_offset_max_by_account_unit": np.max(
                abs(response[:, :, 0] - z["native_accounts"][:, :, 0]), axis=(0, 1, 3)
            ).tolist(),
            "HM202_HM203_native_actions_bit_equal": True,
            "same_source_action_max_error": source_frame_error,
            "stored_head_vs_digit_geometry_max_error": head_digit_error,
            "raw_foreign_C_self_error": float(
                np.max(abs(mode_actions[:, :, 3, 3] - mode_actions[:, :, 5, 3]))
            ),
            "paired_assignment_shape": list(old["indices"].shape),
            "original_HM203_failure_retained": True,
            "new_neural_API_private_train_raw_text_calls": 0,
        },
    )
    R.seal(
        out, "prepared_seal.json", ("manifest.json", "observations.npz", "quality.json")
    )
    print("PREPARED HM213: source-anchor x2 paired mechanism contrasts", flush=True)


def compute(out):
    """Compute one fixed paired-orbit family; retain all signed secondary cells."""
    cfg = R.check(out)["config"]
    start = time.monotonic()
    z = A.arrays(out / "observations.npz")
    t = C.projections(z["source_means"], z["vector_actions"], z["response"])
    flat = t.reshape(len(t), -1)
    observed, sampled, variances, joint, primary, modes = [], [], [], [], [], []
    for q in range(5):
        ix = z["assignment_indices"][q]
        obs = M.statistic(flat, z["labels"], q).reshape(t.shape[1:])
        refs = M.references(flat, z["labels"], q, ix).reshape((len(ix),) + t.shape[1:])
        var = M.exact_moments(flat, z["labels"], q)[1].reshape(t.shape[1:])
        p = C.primary_columns(t, q)
        pv, pr = M.statistic(p, z["labels"], q), M.references(p, z["labels"], q, ix)
        pc = M.exact_moments(p, z["labels"], q, joint=True)[1]
        joint.append(pc)
        for j, name in enumerate(cfg["primary_contrasts"]):
            primary.append(
                {
                    "target": cfg["targets"][q],
                    "contrast": name,
                    "covariance": float(pv[j]),
                    **M.rank_account(float(pv[j]), pr[:, j], float(pc[j, j])),
                }
            )
        for j, name in enumerate(C.MODES):
            mp = t[:, :, j, 2, 2, 2, q].mean(1)[:, None]
            mv, mr = (
                M.statistic(mp, z["labels"], q)[0],
                M.references(mp, z["labels"], q, ix)[:, 0],
            )
            variance = M.exact_moments(mp, z["labels"], q)[1][0]
            modes.append(
                {
                    "target": cfg["targets"][q],
                    "mode": name,
                    "covariance": float(mv),
                    **M.rank_account(float(mv), mr, float(variance)),
                    "secondary_not_new_confirmatory_family": True,
                }
            )
        observed.append(obs)
        sampled.append(refs)
        variances.append(var)
    for row, adjusted in zip(
        primary, M.holm([r["two_sided_rank"] for r in primary]), strict=True
    ):
        row["holm10_two_sided_rank"] = adjusted
    observed, sampled, variances = map(np.array, (observed, sampled, variances))
    native_error = float(np.max(abs(observed[:, :, 0] - z["previous_native_observed"])))
    native_ref_error = float(
        np.max(abs(sampled[:, :, :, 0] - z["previous_native_sampled"]))
    )
    algebra_error = C.telescoping_error(z["response"])
    if algebra_error > 2e-12:
        raise ValueError("Paired signed-algebra control failed")
    cache_observed = observed[:, :, 0] - z["previous_native_observed"]
    cache_sampled = sampled[:, :, :, 0] - z["previous_native_sampled"]
    elapsed = time.monotonic() - start
    if elapsed > cfg["max_compute_seconds"]:
        raise TimeoutError("Finite computation budget exceeded; no retry")
    R.archive(
        out / "analysis.npz",
        observed=observed,
        sampled=sampled,
        variances=variances,
        primary_joint_covariance=np.array(joint),
        cache_offset_observed=cache_observed,
        cache_offset_sampled=cache_sampled,
    )
    rows = []
    for q, b, channel, account, unit, view, reader in np.ndindex(observed.shape):
        index = (q, b, channel, account, unit, view, reader)
        ref = sampled[q, :, b, channel, account, unit, view, reader]
        rows.append(
            {
                "source_target": cfg["targets"][q],
                "source_view": b,
                "channel": C.CHANNELS[channel],
                "account": A.ACCOUNTS[account],
                "unit": A.UNITS[unit],
                "panel_view": A.VIEWS[view],
                "reader_target": cfg["targets"][reader],
                "covariance": float(observed[index]),
                **M.rank_account(float(observed[index]), ref, float(variances[index])),
            }
        )
    R.table(out / "contacts.csv", rows)
    R.table(out / "primary.csv", primary)
    R.table(out / "mode_diagonals.csv", modes)
    R.write(
        out / "summary.json",
        {
            "packet": "HM213",
            "primary": primary,
            "mode_diagonals": modes,
            "all_contact_accounts": len(rows),
            "reference_assignments_reused": 995,
            "new_reference_draws": 0,
            "primary_family_size": 10,
            "same_native_contact_max_error": native_error,
            "same_native_references_max_error": native_ref_error,
            "old_native_cache_offset_retained_not_corrected": True,
            "signed_telescope_max_error": algebra_error,
            "exact_joint_orbit_covariance_not_independent_mode_SD": True,
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
        (
            "analysis.npz",
            "contacts.csv",
            "primary.csv",
            "mode_diagonals.csv",
            "summary.json",
        ),
    )
    print("COMPUTED HM213", primary, "seconds", elapsed, flush=True)


def verify(out):
    """Replay literal SOURCE group means, paired variance and direct contacts."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    start = time.monotonic()
    z, saved = A.arrays(out / "observations.npz"), A.arrays(out / "analysis.npz")
    t = C.projections(z["source_means"], z["vector_actions"], z["response"])
    error, count = 0.0, 0
    y = z["labels"]
    original_counts = np.bincount(y @ (2 ** np.arange(5)), minlength=32)
    for q in range(5):
        b, counts, omega, _ = M.design(y, q)
        for j, indices in enumerate(z["assignment_indices"][q]):
            yh = y.copy()
            yh[:, q] = y[indices, q]
            if not np.array_equal(
                np.delete(yh, q, 1), np.delete(y, q, 1)
            ) or not np.array_equal(
                np.bincount(yh @ (2 ** np.arange(5)), minlength=32), original_counts
            ):
                raise ValueError("Source assignment broke fixed anchor structure")
            direct = np.zeros(t.shape[1:])
            for c in range(16):
                direct += omega[c] * (
                    t[(b == c) & (yh[:, q] == 1)].mean(0)
                    - t[(b == c) & (yh[:, q] == 0)].mean(0)
                )
            error = max(error, float(np.max(abs(direct - saved["sampled"][q, j]))))
            count += direct.size
        variance, joint = np.zeros(t.shape[1:]), np.zeros((2, 2))
        p = C.primary_columns(t, q)
        for c in range(16):
            a = t[b == c]
            centered = a - a.mean(0)
            coefficient = omega[c] ** 2 * len(a) / ((len(a) - 1) * counts[:, c].prod())
            variance += coefficient * (centered**2).sum(0)
            pp = p[b == c] - p[b == c].mean(0)
            joint += coefficient * (pp.T @ pp)
        error = max(
            error,
            float(np.max(abs(variance - saved["variances"][q]))),
            float(np.max(abs(joint - saved["primary_joint_covariance"][q]))),
        )
        for source_view, panel_view in np.ndindex(2, 3):
            x = (
                z["vector_actions"][:, panel_view]
                if panel_view < 2
                else z["vector_actions"].mean(1)
            )
            r = (
                z["response"][:, panel_view]
                if panel_view < 2
                else z["response"].mean(1)
            )
            values = x @ z["fields"][source_view, q]
            direct = (
                np.einsum("u,umcqt->mcqt", values - values.mean(), r - r.mean(0)) / 23
            )
            expected = saved["observed"][q, source_view, :, :, :, panel_view, :]
            error = max(
                error, float(np.max(abs(direct.transpose(0, 1, 3, 2) - expected)))
            )
    if error > 2e-12:
        raise ValueError("Literal source/paired mechanism replay failed")
    elapsed = time.monotonic() - start
    if elapsed > cfg["max_verify_seconds"]:
        raise TimeoutError("Finite audit budget exceeded; preserve outputs, no retry")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "max_error": error,
            "literal_reference_values": count,
            "literal_observed_contacts": saved["observed"].size,
            "paired_primary_covariances": 5,
            "all32_profile_counts_and_other4_rows_preserved": True,
            "source_views_and_modes_share_assignments": True,
            "same_agent_not_independent": True,
            "recorded_verify_seconds": elapsed,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM213", error, count, "seconds", elapsed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("ALL SEALS PASS HM213", flush=True)
    else:
        {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)
