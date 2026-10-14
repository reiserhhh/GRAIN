#!/usr/bin/env python3
"""Compare independent and shared references without changing author laws."""

import argparse
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_shared_reference as M
from grain_research.construction import math_source_target_law as S
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_shared_reference_hm231.json"
DEFAULT = ROOT / "results/grain_math_response_v1/shared_reference_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SHARED_REFERENCE_PLAN_HM231.md",
    "grain_research/construction/math_shared_reference.py",
    "scripts/run_grain_shared_reference_hm231.py",
    "scripts/score_grain_shared_reference_cold.py",
    "tests/test_grain_shared_reference.py",
    "grain_research/construction/math_source_target_law.py",
    "scripts/run_grain_observed_family_hm170.py",
)
STAGES = ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")


def load(path):
    """Read only this finite packet's explicit numeric arrays."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Bind old joint laws and graph without any new source or outcome fitting."""
    if out.exists():
        raise FileExistsError("Existing packet cannot be replaced")
    cfg = R.read(ROOT / CONFIG)
    parent, native = (ROOT / cfg[k] for k in ("parent", "comparison_parent"))
    R.check(parent, STAGES)
    R.check(
        native,
        (
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "ordinal_seal.json",
        ),
    )
    with np.load(parent / "laws.npz", allow_pickle=False) as z:
        w = z["weights"]
    with np.load(parent / "observations.npz", allow_pickle=False) as z:
        anchors, edges = z["anchors"], z["edges"]
    if w.shape != (24, 3, 386) or anchors.shape != (386, 5) or edges.shape != (32, 2):
        raise ValueError("SOURCE law material changed")
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "laws.npz",
            "observations.npz",
            "event_selections.json",
            *STAGES,
        )
    ] + [native / "analysis.npz", native / "ordinal_seal.json"]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "application_y_or_native_answers_in_cold_input": False,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        weights=w,
        anchors=anchors,
        authors=np.arange(24),
        edges=edges,
    )
    R.write(
        out / "input_quality.json",
        {
            "SOURCE_authors": 386,
            "application_authors": 24,
            "unique_events": 288,
            "arms": 3,
            "targets": 5,
            "new_model_API_private_train_calls": 0,
            "new_texts_or_people": 0,
            "historically_reused_numeric_material": True,
            "SOURCE_joint_rows_preserved": True,
            "pair_law_is_declared_not_natural_causal_truth": True,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print("PREPARED HM231: unchanged72 SOURCE laws; new shared-reference mechanism")


def compute(out):
    """Run cold calculations once with no access to old native comparison answers."""
    start = time.monotonic()
    cfg = R.check(out)["config"]
    command = [
        sys.executable,
        str(ROOT / "scripts/score_grain_shared_reference_cold.py"),
        "--inputs",
        str(out / "observations.npz"),
        "--output",
        str(out / "pair_laws.npz"),
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=cfg["max_cpu_seconds"],
    )
    R.write(
        out / "cold_receipt.json",
        {
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        },
    )
    process.check_returncode()
    seconds = time.monotonic() - start
    R.write(
        out / "runtime.json",
        {
            "seconds": seconds,
            "new_neural_API_private_train_calls": 0,
            "CPU_cap_pass": seconds <= cfg["max_cpu_seconds"],
        },
    )
    if seconds > cfg["max_cpu_seconds"]:
        raise RuntimeError("CPU cap exceeded; preserve output without retry")
    R.seal(out, STAGES[0], ("pair_laws.npz", "cold_receipt.json", "runtime.json"))
    print(f"COMPUTED HM231:900 joint laws in{seconds:.3f}s")


def cosine(a, b):
    """Keep undefined geometric contact missing, not a psychological accuracy."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else None


def analyze(out):
    """Retain all target/arm effects before interpreting native contacts."""
    cfg = R.check(out, STAGES[:1])["config"]
    z, a = load(out / "observations.npz"), load(out / "pair_laws.npz")
    with np.load(
        ROOT / cfg["comparison_parent"] / "analysis.npz", allow_pickle=False
    ) as old:
        native = old["odd"]
    if native.shape != (32, 5):
        raise ValueError("Complete old native edge readings required")
    edges = z["edges"]
    all_edges = np.array([(i, j) for i in range(24) for j in range(i + 1, 24)])
    rows, pair_rows, cycles, geometry = [], [], [], {}
    for arm, name in enumerate(cfg["arms"]):
        shared = 2 * a["shared_probability"][arm, edges[:, 0], edges[:, 1]] - 1
        independent = (
            2 * a["independent_probability"][arm, edges[:, 0], edges[:, 1]] - 1
        )
        shared_all = (
            2 * a["shared_probability"][arm, all_edges[:, 0], all_edges[:, 1]] - 1
        )
        same = a["same_source_probability"][arm, edges[:, 0], edges[:, 1]]
        changed = a["changed_source_probability"][arm, edges[:, 0], edges[:, 1]]
        for q, target in enumerate(cfg["targets"]):
            rows.append(
                {
                    "arm": name,
                    "target": target,
                    "same32_direction_changes": int(
                        np.sum(shared[:, q] * independent[:, q] < -1e-24)
                    ),
                    "all276_direction_changes": int(
                        np.sum(
                            shared_all[:, q]
                            * (
                                2
                                * a["independent_probability"][
                                    arm, all_edges[:, 0], all_edges[:, 1], q
                                ]
                                - 1
                            )
                            < -1e-24
                        )
                    ),
                    "shared_vs_independent_cosine": cosine(
                        shared[:, q], independent[:, q]
                    ),
                    "shared_native_cosine": cosine(shared[:, q], native[:, q]),
                    "independent_native_cosine": cosine(
                        independent[:, q], native[:, q]
                    ),
                    "comparison_change_RMS": float(
                        np.sqrt(np.mean((shared[:, q] - independent[:, q]) ** 2))
                    ),
                    "same_SOURCE_probability_min": float(same.min()),
                    "same_SOURCE_probability_max": float(same.max()),
                    "changed_SOURCE_probability_mean": float(changed.mean()),
                }
            )
            cycles.append(
                {
                    "arm": name,
                    "target": target,
                    "same32_shared_cycle": S.directed_cycle(edges, shared[:, q], 24),
                    "all276_shared_cycle": S.directed_cycle(
                        all_edges, shared_all[:, q], 24
                    ),
                }
            )
            for k, (i, j) in enumerate(edges):
                pair_rows.append(
                    {
                        "arm": name,
                        "target": target,
                        "left": int(i),
                        "right": int(j),
                        "independent_PI": float(
                            a["independent_probability"][arm, i, j, q]
                        ),
                        "shared_PI": float(a["shared_probability"][arm, i, j, q]),
                        "shared_PI_given_changed_SOURCE": float(
                            a["conditional_probability_given_changed_source"][
                                arm, i, j, q
                            ]
                        ),
                        "same_SOURCE_probability": float(same[k]),
                        "old_full_native_odd_nats": float(native[k, q]),
                    }
                )
        geometry[name] = [
            [cosine(shared[:, q], shared[:, r]) for r in range(5)] for q in range(5)
        ]
    cycle_nodes = [(15, 16), (16, 17), (17, 15)]
    cycle_values = {
        target: [float(a["shared_probability"][0, i, j, q]) for i, j in cycle_nodes]
        for q, target in enumerate(cfg["targets"])
    }
    R.table(out / "mechanism_comparison.csv", rows)
    R.table(out / "same_graph_accounts.csv", pair_rows)
    R.write(out / "cycle_witnesses.json", cycles)
    R.write(
        out / "summary.json",
        {
            "mechanism": "shared SOURCE exponential race, not accuracy optimization",
            "all_target_results": rows,
            "target_edge_cosines": geometry,
            "old_native_cycle_shared_PI_all_targets": cycle_values,
            "cycles": cycles,
            "marginal_audit_max": a["marginal_audit"][:, 3:].max(0).tolist(),
            "original_scope_complete": False,
            "pair_laws_identified_from_author_marginals": False,
            "conditional_comparison_not_psychological_probability": True,
            "native_contacts_full_material_only_other_arms_descriptive": True,
        },
    )
    R.seal(
        out,
        STAGES[1],
        (
            "mechanism_comparison.csv",
            "same_graph_accounts.csv",
            "cycle_witnesses.json",
            "summary.json",
        ),
    )
    print("ANALYZED HM231: all5targets/all3arms/all32edges retained")


def alternate(p, q):
    """Integrate each race interval separately rather than cumulative primitives."""
    ratio = p / q
    cuts = np.unique(ratio)
    joint = np.zeros((len(p), len(p)))
    for lo, hi in pairwise(cuts):
        sl, sh = np.maximum(p, lo * q).sum(), np.maximum(p, hi * q).sum()
        outer = np.outer(p * (ratio >= hi), q * (ratio <= lo))
        joint += outer * ((hi - lo) / (sl * sh))
    for i in range(len(p)):
        joint[i, i] = 1 / np.maximum(p / p[i], q / q[i]).sum()
    return joint


def verify(out):
    """Check the pair-law accounts and direct race integrals, without refitting."""
    R.check(out, STAGES[:2])
    z, a = load(out / "observations.npz"), load(out / "pair_laws.npz")
    w, y = z["weights"], z["anchors"]
    audit = a["marginal_audit"]
    if audit.shape != (900, 6):
        raise ValueError("Missing complete joint marginal audit")
    alt_error, quad_error, swap_error = 0.0, 0.0, 0.0
    replay_error = 0.0
    for arm, i, j in [(0, 15, 16), (0, 16, 17), (0, 17, 15), (1, 0, 23), (2, 1, 12)]:
        p, q = w[i, arm], w[j, arm]
        joint, alt = M.coupling(p, q), alternate(p, q)
        alt_error = max(alt_error, float(abs(joint - alt).max()))
        swap_error = max(swap_error, float(abs(M.coupling(q, p) - joint.T).max()))
        k = M.kernels(y)
        replay_error = max(
            replay_error,
            float(
                abs(
                    np.sum(alt[:, :, None] * k, axis=(0, 1))
                    - a["shared_probability"][arm, i, j]
                ).max()
            ),
        )
        ratio = p / q
        order = np.argsort(ratio)
        for left, right in [(order[-1], order[0]), (order[-2], order[1])]:
            lo, hi = ratio[right], ratio[left]
            integral = quad(
                lambda t, p=p, q=q: 1 / np.maximum(p, t * q).sum() ** 2,
                lo,
                hi,
                points=np.unique(ratio[(ratio > lo) & (ratio < hi)]),
                limit=500,
                epsabs=1e-11,
                epsrel=1e-11,
            )[0]
            quad_error = max(
                quad_error, abs(p[left] * q[right] * integral - joint[left, right])
            )
    old = load(
        ROOT
        / R.read(out / "manifest.json")["config"]["parent"]
        / "distribution_records.npz"
    )
    p, same, changed = (
        a["shared_probability"],
        a["same_source_probability"],
        a["changed_source_probability"],
    )
    conditional = a["conditional_probability_given_changed_source"]
    mask = changed > 0
    checks = {
        "all900_row_marginals": float(audit[:, 3].max()),
        "all900_column_marginals": float(audit[:, 4].max()),
        "all900_invariant_mean_differences": float(audit[:, 5].max()),
        "five_fixed_pairs_alternate_joint": alt_error,
        "ten_direct_integrals": float(quad_error),
        "alternate_readout_replay": replay_error,
        "five_fixed_pairs_swap": swap_error,
        "independent_old_law_replay": float(
            abs(a["independent_probability"] - old["pair_probability"]).max()
        ),
        "all_self_comparison": float(
            abs(p[:, np.arange(24), np.arange(24)] - 0.5).max()
        ),
        "shared_mass_partition": float(abs(same + changed - 1).max()),
        "conditional_unconditional_identity": float(
            abs(
                p[mask]
                - 0.5 * same[mask, None]
                - changed[mask, None] * conditional[mask]
            ).max()
        ),
    }
    common_null = M.coupling(w[0, 0], w[0, 0])
    checks["identical_author_law_null"] = float(
        abs(common_null - np.diag(w[0, 0])).max()
    )
    passed = bool(all(v <= 1e-10 for v in checks.values()))
    R.write(
        out / "verification.json",
        {
            "PASS": passed,
            "checks": checks,
            "same_agent_not_blind": True,
            "new_neural_API_private_train_calls": 0,
        },
    )
    if not passed:
        raise ValueError("Verification failed; preserve all outputs")
    R.seal(out, STAGES[2], ("verification.json",))
    print("VERIFIED HM231: joint marginals/mean invariance/alternate integration")


def main():
    """Execute explicit finite stages; do not queue another experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, STAGES)
        print("SEALED HM231 unchanged")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
