#!/usr/bin/env python3
"""One finite source-allocation mechanism test; no predictor or precision search."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_source_catalogue_response as M
from grain_research.construction import math_source_reference_action as S
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_source_reference_action_hm215 import banks, select

CONFIG = "configs/grain_source_catalogue_response_hm216.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_catalogue_response_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SOURCE_CATALOGUE_RESPONSE_PLAN_HM216.md",
    "grain_research/construction/math_source_catalogue_response.py",
    "scripts/run_grain_source_catalogue_response_hm216.py",
    "tests/test_grain_source_catalogue_response.py",
    "grain_research/construction/math_source_reference_action.py",
    "grain_research/construction/math_diagnostic_action_bridge.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/run_grain_source_reference_action_hm215.py",
    "scripts/run_grain_observed_family_hm170.py",
)
KEYS = (
    "source_action",
    "destination_action",
    "calibration_action",
    "response_defect",
    "conditional_cross_ratio",
)


def prepare(out):
    """Freeze paired author exchange without reading any application outcome."""
    cfg = R.read(ROOT / CONFIG)
    parent, sp = ROOT / cfg["parent"], ROOT / cfg["source_parent"]
    R.check(parent, ("computed_seal.json", "verification_seal.json", "cold_seal.json"))
    R.check(sp, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    points = select(parent / "observations.npz", ("points",))["points"]
    allowed = (
        "source_means",
        "source_indices",
        "codes",
        "folds",
        "source_proportions",
        "variance",
    )
    z = select(sp / "observations.npz", allowed)
    if z["source_means"].shape != (386, 2, 1024) or points.shape != (24, 2, 2, 2, 1024):
        raise ValueError("Different source and addressed panel scope")
    if float(z["variance"]) != cfg["variance"]:
        raise ValueError("Changed inherited scale")
    old = banks(sp / "source_banks.npz")
    for b in (0, 1):
        ix = z["folds"] == b
        if not np.array_equal(
            old[b]["prototypes"], z["source_means"][ix]
        ) or not np.array_equal(old[b]["codes"], z["codes"][ix]):
            raise ValueError("Another source catalogue")
    nc = np.rint(12 * z["source_proportions"]).astype(int)
    if np.max(abs(nc / 12 - z["source_proportions"])) > 1e-14:
        raise ValueError("Condition counts not from these12-event windows")
    groups = [M.strata(z["codes"]), M.strata(z["codes"], nc)]
    supports = [M.support(z["codes"], z["folds"], c) for c in (None, nc)]
    assignments = np.empty((2, cfg["references"] + 1, 386), int)
    for f, family_groups in enumerate(groups):
        rng = np.random.default_rng(cfg["seeds"][f])
        assignments[f, 0] = z["folds"]
        for j in range(1, cfg["references"] + 1):
            assignments[f, j] = M.exchange(z["folds"], family_groups, rng)
    k = M.kernels(points, z["source_means"], z["variance"])
    inputs = [
        parent / "observations.npz",
        parent / "analysis.npz",
        parent / "manifest.json",
        sp / "observations.npz",
        sp / "source_banks.npz",
        sp / "manifest.json",
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
            "source_banks_are_hash_profile_splits_not_independent_populations": True,
            "application_labels_raw_native_outcomes_read": False,
        },
    )
    R.archive(
        out / "observations.npz",
        **z,
        condition_counts=nc,
        points=points,
        kernels=k,
        assignments=assignments,
    )
    R.write(
        out / "support.json",
        {name: s for name, s in zip(cfg["families"], supports, strict=True)},
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "support.json"))
    print(
        "PREPARED HM216",
        {
            name: s["movable_authors"]
            for name, s in zip(cfg["families"], supports, strict=True)
        },
        flush=True,
    )


def rms(x):
    """Descriptive RMS in log-response units, not accuracy or information share."""
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def compute(out):
    """Compare the one old split to two fixed whole-author allocation orbits."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    start = time.monotonic()
    arrays = {k: [] for k in KEYS}
    stats = {
        k: []
        for k in ("cross_ratio_RMS", "response_defect_RMS", "all5_cross_ratio_RMS")
    }
    control_error = telescope_error = identity_error = 0.0
    for f in range(2):
        aa, ss = {k: [] for k in arrays}, {k: [] for k in stats}
        for j, allocation in enumerate(z["assignments"][f]):
            result = M.response(M.laws(z["kernels"], z["codes"], allocation))
            for k, values in aa.items():
                values.append(result[k])
            for k, value in M.statistics(result).items():
                ss[k].append(value)
            control_error = max(control_error, result["control_profile_error"])
            telescope_error = max(telescope_error, result["telescope_error"])
            identity_error = max(
                identity_error,
                float(
                    np.max(abs(result["response_defect"] - result["defect_identity"]))
                ),
            )
            if j % 50 == 0:
                print(f"HM216 family{f} allocation{j}/{cfg['references']}", flush=True)
            if time.monotonic() - start > cfg["max_compute_seconds"]:
                R.write(
                    out / "compute_failure.json", {"finite_cap": True, "no_retry": True}
                )
                raise RuntimeError("Finite cap reached; preserve partial state")
        for k, values in arrays.items():
            values.append(np.asarray(aa[k]))
        for k, values in stats.items():
            values.append(np.asarray(ss[k]))
    arrays = {k: np.asarray(a) for k, a in arrays.items()}
    stats = {k: np.asarray(a) for k, a in stats.items()}
    old = select(ROOT / cfg["parent"] / "analysis.npz", KEYS)
    replay_error = max(
        float(
            np.max(
                abs(
                    a[0, 0]
                    - old[k][0].transpose(
                        (1, 2, 0, 3, 4)
                        if k == "conditional_cross_ratio"
                        else (1, 2, 0, 3)
                    )
                )
            )
        )
        for k, a in arrays.items()
    )
    summaries, rows, primary, mean_density = [], [], [], []
    for f, family in enumerate(cfg["families"]):
        groups = M.strata(z["codes"], z["condition_counts"] if f else None)
        density = M.orbit_mean(z["kernels"], z["codes"], z["folds"], groups)
        mean_density.append(density)
        mean_stats = M.statistics(M.response(density))
        for law, name in enumerate(cfg["laws"]):
            glob = stats["all5_cross_ratio_RMS"][f, :, law]
            prow = {
                "family": family,
                "law": name,
                "observed_all5_cross_ratio_RMS": float(glob[0]),
                "reference_median": float(np.median(glob[1:])),
                "reference_q025": float(np.quantile(glob[1:], 0.025)),
                "reference_q975": float(np.quantile(glob[1:], 0.975)),
                "upper_plus_one_rank": float(
                    (1 + np.sum(glob[1:] >= glob[0])) / len(glob)
                ),
                "orbit_mean_density_cross_ratio_RMS": float(
                    mean_stats["all5_cross_ratio_RMS"][law]
                ),
                "mean_density_not_mean_response": True,
                "not_unconditional_psychological_p": True,
            }
            if law == 0:
                primary.append(prow)
            for q, target in enumerate(cfg["targets"]):
                values = stats["cross_ratio_RMS"][f, :, law, q]
                summaries.append(
                    {
                        **prow,
                        "target": target,
                        "observed_target_cross_ratio_RMS": float(values[0]),
                        "target_reference_median": float(np.median(values[1:])),
                        "target_upper_plus_one_rank": float(
                            (1 + np.sum(values[1:] >= values[0])) / len(values)
                        ),
                        "observed_response_defect_RMS": float(
                            stats["response_defect_RMS"][f, 0, law, q]
                        ),
                    }
                )
                for j in range(cfg["references"] + 1):
                    cal = arrays["calibration_action"][f, j, :, :, law, q].mean(1)
                    defect = arrays["response_defect"][f, j, :, :, law, q].mean(1)
                    rows.append(
                        {
                            "family": family,
                            "allocation": j,
                            "law": name,
                            "target": target,
                            "cross_ratio_RMS": float(values[j]),
                            "response_defect_RMS": rms(defect),
                            "calibration_RMS": rms(cal),
                            "signed_cross_account": float(2 * np.mean(cal * defect)),
                            "full_action_gap_RMS": rms(cal + defect),
                        }
                    )
    error = max(control_error, telescope_error, identity_error, replay_error)
    if not np.isfinite(error) or error > cfg["comparison_tolerance"]:
        R.write(out / "compute_failure.json", {"max_error": error, "no_retry": True})
        raise ValueError("Same-event replay identity failed")
    R.archive(
        out / "analysis.npz",
        **arrays,
        **stats,
        orbit_mean_densities=np.asarray(mean_density),
    )
    R.table(out / "allocation_accounts.csv", rows)
    R.table(out / "response_summaries.csv", summaries)
    summary = {
        "packet": "HM216",
        "primary": primary,
        "control_error": control_error,
        "telescope_error": telescope_error,
        "identity_error": identity_error,
        "old_response_replay_error": replay_error,
        "source_count": 386,
        "panel_count": 24,
        "new_group_draws": 2 * cfg["references"],
        "unique_reference_allocations": [
            len({row.tobytes() for row in a[1:]}) for a in z["assignments"]
        ],
        "all30_target_summaries_retained": True,
        "same_panel_not_400_independent_replicates": True,
        "new_neural_API_private_training_fit_calls": 0,
        "original_scope_complete": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "analysis.npz",
            "allocation_accounts.csv",
            "response_summaries.csv",
            "summary.json",
        ),
    )
    print("COMPUTED HM216", json.dumps(summary), flush=True)


def literal_orbit(k, codes, folds, groups):
    """Independent pair-inclusion enumeration, rather than covariance formula."""
    out = np.empty((2, len(k), 2, 3, 2, 32))
    for bank, code in np.ndindex(2, 32):
        ix = np.flatnonzero(codes == code)
        n = int(np.sum(folds[ix] == bank))
        address = {int(i): h for h, group in enumerate(groups) for i in group}
        sizes = {h: len(group) for h, group in enumerate(groups)}
        nb = {h: int(np.sum(folds[group] == bank)) for h, group in enumerate(groups)}
        mean = np.zeros(k.shape[:-1])
        joint, product = np.zeros(k.shape[:-2]), np.zeros(k.shape[:-2])
        for i in ix:
            h = address[int(i)]
            inclusion = nb[h] / sizes[h]
            mean += inclusion * k[..., i] / n
            joint += inclusion * k[..., i].prod(-1) / n
            for j in ix:
                hh = address[int(j)]
                if i == j:
                    probability = inclusion
                elif h == hh:
                    probability = nb[h] * (nb[h] - 1) / (sizes[h] * (sizes[h] - 1))
                else:
                    probability = inclusion * nb[hh] / sizes[hh]
                product += probability * k[..., 0, i] * k[..., 1, j] / n**2
        out[bank, ..., code] = np.stack(
            (np.stack((mean[:, 0, :, 0], mean[:, 1, :, 1]), 1), joint, product), 2
        )
    return out


def verify(out):
    """Audit addresses, literal Gaussian laws and exact orbit means, not fit outcomes."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    z, a, start = (
        R.prepared(out),
        select(out / "analysis.npz", KEYS + ("orbit_mean_densities",)),
        time.monotonic(),
    )
    error, values, allocation_checks = 0.0, 0, 0
    literal_k = np.empty_like(z["kernels"])
    for u, v, arm, window in np.ndindex(24, 2, 2, 2):
        literal_k[u, v, arm, window] = [
            np.exp(
                -np.sum((z["points"][u, v, arm, window] - zz[window]) ** 2)
                / (2 * cfg["variance"])
            )
            for zz in z["source_means"]
        ]
    error = max(error, float(np.max(abs(literal_k - z["kernels"]))))
    values += literal_k.size
    for f in (0, 1):
        groups = M.strata(z["codes"], z["condition_counts"] if f else None)
        for allocation in z["assignments"][f]:
            for ix in groups:
                if not np.array_equal(np.sort(allocation[ix]), np.sort(z["folds"][ix])):
                    raise ValueError("Allocation changed a fixed stratum")
                allocation_checks += 1
        expected = literal_orbit(literal_k, z["codes"], z["folds"], groups)
        error = max(error, float(np.max(abs(expected - a["orbit_mean_densities"][f]))))
        values += expected.size
        for j in (0, 1, cfg["references"]):
            ff = z["assignments"][f, j]
            bs = [
                {
                    "prototypes": z["source_means"][ff == b],
                    "codes": z["codes"][ff == b],
                    "variance": z["variance"],
                }
                for b in (0, 1)
            ]
            for u, v in np.ndindex(24, 2):
                laws = [S.literal_laws(b, z["points"][u, v], v) for b in bs]
                lit = S.literal_certificate(*laws)
                for key in KEYS:
                    value = lit.get(key)
                    if key == "calibration_action":
                        value = lit["transport_action"] - lit["source_action"]
                    error = max(error, float(np.max(abs(value - a[key][f, j, u, v]))))
                    values += value.size
    if not np.isfinite(error) or error > cfg["comparison_tolerance"]:
        R.write(
            out / "verification_failure.json", {"max_error": error, "no_retry": True}
        )
        raise ValueError("Catalogue audit failed; retain result")
    receipt = {
        "PASS": True,
        "max_error": error,
        "literal_values": values,
        "allocation_stratum_checks": allocation_checks,
        "literal_reference_allocations": 6,
        "same_agent_not_blind": True,
        "elapsed_seconds": time.monotonic() - start,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM216", receipt, flush=True)


def main():
    """Run each immutable stage once; check only reads sealed artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("HM216 ALL SEALS PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
