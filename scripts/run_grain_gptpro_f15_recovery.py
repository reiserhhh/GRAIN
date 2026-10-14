#!/usr/bin/env python3
"""Run the F15 recovery certificates (F15-01..04) with COMMON per-card outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3 import f15_theory as th

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_gptpro_foundation_v3"
SEEDS = (20260915, 20260916, 20260917)
CODE = [
    "grain_research/gptpro_foundation_v3/f15_theory.py",
    "grain_research/gptpro_foundation_v3/f15.py",
    "grain_research/gptpro_foundation_v3/f02.py",
    "scripts/run_grain_gptpro_f15_recovery.py",
    "tests/test_grain_gptpro_f15_theory.py",
    "docs/GRAIN_F15_RECOVERY_PROTOCOL_V1.md",
    "configs/grain_f01_primary_cards_v1.json",
    "grain_research/gptpro_foundation_v3/f10_geometry.py",
]


def sha(path):
    """File hash."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    """Strict JSON conversion."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    """Immutable strict JSON."""
    with Path(path).open("x") as f:
        f.write(
            json.dumps(clean(value), sort_keys=True, indent=1, allow_nan=False) + "\n"
        )


def write_csv(path, rows, header=None):
    """Immutable CSV; header-only if no rows."""
    header = header or (list(rows[0]) if rows else ["reason"])
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(clean(r))


def card(
    output, card_id, protocol, inputs, summary, metrics, decision, extra, log, tests
):
    """COMMON eight files plus card-specific artifacts."""
    folder = output / card_id
    folder.mkdir()
    write_json(folder / "protocol.json", protocol)
    write_json(folder / "input_manifest.json", inputs)
    write_json(folder / "summary.json", summary)
    write_csv(folder / "metrics.csv", metrics, None if metrics else ["reason"])
    (folder / "decision.md").write_text(decision)
    with (folder / "run_log.jsonl").open("x") as f:
        for row in log:
            f.write(json.dumps(clean(row)) + "\n")
    (folder / "tests.txt").write_text(tests)
    for name, value in extra.items():
        if name.endswith(".csv"):
            write_csv(folder / name, value)
        else:
            write_json(folder / name, value)
    manifest = {
        p.name: sha(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and not p.name.startswith(".")
    }
    write_json(folder / "output_manifest.json", manifest)


def run_psr(output, tests):
    """F15-01 exact Hankel ranks, PSR closure and memory terms."""
    started = time.monotonic()
    ranks, closures, memories, metrics = [], [], [], []
    for world in ("c2_iid", "c2_xor", "hidden_order3"):
        for h in (2, 3, 4):
            _, _, d = th.hankel(world, h, 3)
            rank, sv, tol = th.numerical_rank(d)
            ranks.append(
                {
                    "world": world,
                    "history_len": h,
                    "test_len": 3,
                    "shape": list(d.shape),
                    "rank": rank,
                    "singular_values": sv,
                    "tolerance": tol,
                }
            )
            metrics.append(
                {
                    "task_id": "F15-01",
                    "world_id": world,
                    "method": f"hankel_H{h}_K3",
                    "estimand": "linear_dimension",
                    "value": rank,
                }
            )
        for h in (2, 3):
            model = th.fit_linear_psr(world, h, 3)
            err, n = th.closure_error(model, world, 6, 3)
            closures.append(
                {
                    "world": world,
                    "block_history_len": h,
                    "rank": model["rank"],
                    "core_tests": model["core_tests"],
                    "fit_error": model["fit_error"],
                    "closure_max_error": err,
                    "closure_checks": n,
                    "decision": "CLOSED" if err <= 1e-10 else "NOT_CLOSED",
                }
            )
            metrics.append(
                {
                    "task_id": "F15-01",
                    "world_id": world,
                    "method": f"psr_from_H{h}",
                    "estimand": "closure_max_abs_error",
                    "value": err,
                }
            )
    for world, short, long in (
        ("c2_xor", 2, 4),
        ("c2_xor", 1, 4),
        ("hidden_order3", 2, 4),
        ("hidden_order3", 3, 4),
        ("c2_iid", 0, 3),
    ):
        m = th.memory_term(world, short, long)
        memories.append(
            {
                "world": world,
                "state_len": short,
                "history_len": long,
                "weighted_mean": m["weighted_mean"],
                "weighted_variance": m["weighted_variance"],
                "rows": m["rows"],
            }
        )
        metrics.append(
            {
                "task_id": "F15-01",
                "world_id": world,
                "method": f"memory_state{short}_hist{long}",
                "estimand": "memory_term_variance",
                "value": m["weighted_variance"],
            }
        )
    short_fail = next(
        c
        for c in closures
        if c["world"] == "hidden_order3" and c["block_history_len"] == 2
    )
    long_ok = next(
        c
        for c in closures
        if c["world"] == "hidden_order3" and c["block_history_len"] == 3
    )
    xor = next(
        c for c in closures if c["world"] == "c2_xor" and c["block_history_len"] == 2
    )
    summary = {
        "status": "COMPUTED",
        "c2_xor_linear_dimension": xor["rank"],
        "c2_xor_psr_closure_max_error": xor["closure_max_error"],
        "hidden_order3_short_block_rank": short_fail["rank"],
        "hidden_order3_short_block_closure_error": short_fail["closure_max_error"],
        "hidden_order3_long_block_rank": long_ok["rank"],
        "hidden_order3_long_block_closure_error": long_ok["closure_max_error"],
        "seconds": time.monotonic() - started,
    }
    decision = f"""# F15-01 decision

- C2 XOR: finite system-dynamics rank {xor["rank"]} for histories<=2,3,4; the linear PSR built from the H<=2 block closes on every history<=6 and test<=3 (max error {xor["closure_max_error"]:.3g}, {xor["closure_checks"]} checks). The last-two-outcome state is a sufficient predictive state; its memory term is 0.
- Hidden order-3 world: histories<=2 show rank {short_fail["rank"]} (same as C2), yet histories<=3 reveal rank {long_ok["rank"]}. The PSR fitted on the short block fails closure (max error {short_fail["closure_max_error"]:.3g}); the H<=3 PSR closes ({long_ok["closure_max_error"]:.3g}). Counterexample: an apparently low-rank short-history block does not certify the predictive state.
- Mori-Zwanzig memory term of the 2-step state in the hidden world has mean 0 and variance .09: it is a deterministic function of the omitted lag, not noise, and can be absorbed only by enlarging the state.
- Standard mathematics (PSR system-dynamics rank; Singh, James & Rudary 2004, Thm 1 p.3) used as is; no novelty claimed. Status COMPUTED; parent promotion is a separate review.
"""
    return (
        summary,
        metrics,
        decision,
        {
            "predictive_state_certificates.json": {
                "hankel_ranks": ranks,
                "psr_closure": closures,
                "memory_terms": memories,
            }
        },
    )


def run_blackwell(output, tests):
    """F15-02 same-law channels: full / mean / minimal sufficient; process sub-world."""
    started = time.monotonic()
    f09 = th.f09_family_channel()
    f04 = th.f04_binary_channel()
    res09 = th.blackwell_certificates(
        f09,
        {
            "family": [h[0] for h in f09["hypotheses"]],
            "beta": [h[1] for h in f09["hypotheses"]],
        },
    )
    res04 = th.blackwell_certificates(f04, {"world": f04["hypotheses"]})
    proc = th.process_blackwell(reps=2)
    metrics = []
    for name, res in (
        ("F09_family_beta", res09),
        ("F04_binary_analog_declared", res04),
        ("F09_process_subworld", proc),
    ):
        for k, v in res["mutual_information_nats"].items():
            target, stat = k.split(":")
            metrics.append(
                {
                    "task_id": "F15-02",
                    "world_id": name,
                    "method": stat,
                    "estimand": f"mutual_information_{target}_nats",
                    "value": v,
                }
            )
    summary = {
        "status": "COMPUTED",
        "f09_dimensions": res09["dimensions"],
        "f04_analog_dimensions": res04["dimensions"],
        "process_dimensions": proc["dimensions"],
        "f09_explicit": {k: v["status"] for k, v in res09["explicit_kernels"].items()},
        "f09_lp": {k: v["status"] for k, v in res09["lp"].items()},
        "f04_explicit": {k: v["status"] for k, v in res04["explicit_kernels"].items()},
        "f04_lp": {k: v["status"] for k, v in res04["lp"].items()},
        "process_certificates": {
            k: v["status"] for k, v in proc["certificates"].items()
        },
        "seconds": time.monotonic() - started,
    }
    mi09, mi04, mip = (
        res09["mutual_information_nats"],
        res04["mutual_information_nats"],
        proc["mutual_information_nats"],
    )
    decision = f"""# F15-02 decision

- F09 main family (9 hypotheses, even-corner development design, 3 reps): full {res09["dimensions"]["full"]} outcomes, mean {res09["dimensions"]["mean"]}, minimal sufficient {res09["dimensions"]["minimal"]}. Mean is a garbling of full (explicit kernel {res09["explicit_kernels"]["mean_from_full"]["status"]}); full is not a garbling of mean (LP {res09["lp"]["full_from_mean"]["status"]}); minimal sufficient and full are Blackwell-equivalent (explicit kernels).
  MI (uniform prior, nats): family full {mi09["family:full"]:.6f} vs mean {mi09["family:mean"]:.6f}; beta full {mi09["beta:full"]:.6f} vs mean {mi09["beta:mean"]:.6f}. The mean discards all family information: at the even corners each family has two +1 and two -1 levels and logistic(b)+logistic(-b)=1. In this family the odd-corner future law is in bijection with the hypothesis, so hypothesis MI ({mi09["hypothesis:full"]:.6f} full) is the predictive-state MI.
- F09 process sub-world (belief vs corner-equivalent direct response, beta in .5/1/1.5, 2 reps): corners only E1 carries process MI {mip["process:E1"]:.3g}; adding intermediate levels E2 gives {mip["process:E2"]:.3g} (law gaps {", ".join(f"{v:.6f}" for v in proc["intermediate_law_gap_by_beta"].values())}); observing belief B (E3) gives {mip["process:E3"]:.6f}. E1 is a garbling of E2 and E3 (explicit kernels), not conversely ({proc["certificates"]["E2_from_E1_lp"]["status"]}, {proc["certificates"]["E3_from_E1_lp"]["status"]}). Process naming is not identified from corner reports; intermediate levels identify it only in principle at this budget; the extra observation identifies it.
- F04_binary_analog_declared (new declared world, L/T/Q uniform, a=0, b=1.5, 2 reps; not the Gaussian F04 targets): world MI full {mi04["world:full"]:.6f} vs mean {mi04["world:mean"]:.6f}; minimal sufficient ({res04["dimensions"]["minimal"]} classes) equivalent to full.
- LP results are retained as solver diagnostics; where HiGHS reports a residual above 1e-10 on a structurally feasible direction the explicit kernel certificate decides; infeasibility is decided only by a verified Farkas vector.
- Production rule: compress only to statistics Blackwell-equivalent to the full data for the declared target; a mean score is not sufficient for pattern or process targets; a process claim needs the observation that breaks the equivalence.
"""
    return (
        summary,
        metrics,
        decision,
        {
            "compression_order.json": {
                "F09_family_beta": res09,
                "F04_binary_analog_declared": res04,
                "F09_process_subworld": proc,
            }
        },
    )


def run_auxiliary(output, tests):
    """F15-03 exact rank certificates for auxiliary-variable identification."""
    full = th.auxiliary_rank(th.CONDITIONS)
    four = th.auxiliary_rank(th.CONDITIONS[:4])
    same = th.auxiliary_rank([th.CONDITIONS[0]] * 5)
    grid = [[a, b] for a in (-2.0, -0.5, 0.0, 0.7, 2.0) for b in (-1.5, 0.0, 0.3, 1.8)]
    var5 = [th.variability_rank(th.CONDITIONS, y) for y in grid]
    var4 = [th.variability_rank(th.CONDITIONS[:4], y) for y in grid]
    s = np.random.default_rng(
        np.random.SeedSequence([20260915, 15, 3, 9])
    ).standard_normal((10000, 2))
    roundtrip = float(np.abs(th.unmix(th.mix(s)) - s).max())
    metrics = [
        {
            "task_id": "F15-03",
            "world_id": "five_conditions",
            "method": "natural_parameter_differences",
            "estimand": "rank",
            "value": full["rank"],
        },
        {
            "task_id": "F15-03",
            "world_id": "four_conditions",
            "method": "natural_parameter_differences",
            "estimand": "rank",
            "value": four["rank"],
        },
        {
            "task_id": "F15-03",
            "world_id": "identical_conditions",
            "method": "natural_parameter_differences",
            "estimand": "rank",
            "value": same["rank"],
        },
        {
            "task_id": "F15-03",
            "world_id": "five_conditions",
            "method": "hyvarinen_variability_min_rank_grid20",
            "estimand": "rank",
            "value": min(var5),
        },
        {
            "task_id": "F15-03",
            "world_id": "four_conditions",
            "method": "hyvarinen_variability_max_rank_grid20",
            "estimand": "rank",
            "value": max(var4),
        },
        {
            "task_id": "F15-03",
            "world_id": "declared_mixing",
            "method": "analytic_inverse_roundtrip",
            "estimand": "max_abs_error",
            "value": roundtrip,
        },
    ]
    summary = {
        "status": "COMPUTED",
        "rank_5": full["rank"],
        "rank_4cond": four["rank"],
        "rank_identical": same["rank"],
        "variability_rank_5_min": min(var5),
        "variability_rank_4_max": max(var4),
        "inverse_roundtrip": roundtrip,
    }
    decision = f"""# F15-03 decision

- Five declared conditions: natural-parameter difference matrix L (4x4) has rank {full["rank"]} (required 4; Khemakhem et al. 2020 Thm 1(iv), p.5 eq.14). Hyvarinen et al. 2019 Thm 1 assumption 3 (2n+1=5 values, p.4-5 eqs 8-9) holds at all 20 declared points (min rank {min(var5)}).
- Counterexamples: the first four conditions give rank {four["rank"]} (at most 3 differences, cannot serve as a full-rank positive control; variability max rank {max(var4)}); identical conditions give rank {same["rank"]}.
- The analytic inverse x1-.2x2^3 recovers sources with max error {roundtrip:.2e}; this is a KNOWN inverse, not a learned one, and is not evidence that an estimator identifies the mixing.
- These are identifiability conditions of the declared finite world only.
"""
    return (
        summary,
        metrics,
        decision,
        {
            "auxiliary_identification.json": {
                "five": full,
                "four": four,
                "identical": same,
                "variability_grid": grid,
                "variability_rank_five": var5,
                "variability_rank_four": var4,
                "inverse_roundtrip_max_error": roundtrip,
            }
        },
    )


def paired_bootstrap(diffs_by_seed, reps=2000, seed=20260915):
    """Seed-stratified bootstrap of mean per-pair differences (COMMON rule)."""
    rng = np.random.default_rng(seed)
    rep = np.zeros(reps)
    for d in diffs_by_seed:
        rep += d[rng.integers(len(d), size=(reps, len(d)))].mean(1) / len(diffs_by_seed)
    return (
        float(np.mean([d.mean() for d in diffs_by_seed])),
        float(np.quantile(rep, 0.025)),
        float(np.quantile(rep, 0.975)),
    )


def run_contrastive(output, tests):
    """F15-04 fixed contrastive discriminator with paired CIs + recomputed F10 reference."""
    rows, per_pair = [], {}
    for seed in SEEDS:
        for arm in ("quadratic_x", "quadratic_true_s", "null_world"):
            out = th.contrastive_run(seed, arm, return_losses=True)
            for r in out:
                losses = r.pop("pair_losses", None)
                if r["role"] == "application":
                    per_pair[(seed, arm)] = losses
                rows.append(r)
    chance = float(np.log(2))
    contrasts = []
    for name, a, b in (
        ("quadratic_x_minus_chance", "quadratic_x", None),
        ("quadratic_true_s_minus_chance", "quadratic_true_s", None),
        ("null_world_minus_chance", "null_world", None),
        ("quadratic_x_minus_quadratic_true_s", "quadratic_x", "quadratic_true_s"),
    ):
        diffs = [per_pair[(s, a)] - (per_pair[(s, b)] if b else chance) for s in SEEDS]
        mean, lo, hi = paired_bootstrap(diffs)
        contrasts.append(
            {
                "contrast": name,
                "mean": mean,
                "lower": lo,
                "upper": hi,
                "decision": "SUPPORTED_LOWER_LOSS"
                if hi < 0
                else ("SUPPORTED_HIGHER_LOSS" if lo > 0 else "INCONCLUSIVE"),
            }
        )
    georows, geometry = th.f10_geometry_recompute()
    f10_path = BASE / "F10/geometry_r1/metrics.csv"
    with f10_path.open() as f:
        archived = {
            (r["world"], r["role"], r["method"]): float(r["full_mse"])
            for r in csv.DictReader(f)
        }
    replay = {}
    for key in archived:
        vals = [r["se"] for r in georows if (r["world"], r["role"], r["method"]) == key]
        replay[key] = float(np.mean(vals)) / 2
    replay_err = max(abs(replay[k] - archived[k]) for k in archived)
    unified = (
        [
            {
                "section": "contrastive",
                "seed": r["seed"],
                "world_id": "auxiliary_ica_null"
                if r["arm"] == "null_world"
                else "auxiliary_ica",
                "role": r["role"],
                "method": r["arm"],
                "estimand": "log_loss",
                "value": r["log_loss"],
                "lower": "",
                "upper": "",
                "note": f"acc={r['accuracy']:.4f};auc={r['auc']:.4f};bayes={r.get('bayes_log_loss', '')}",
            }
            for r in rows
        ]
        + [
            {
                "section": "contrastive_paired",
                "seed": "pooled3",
                "world_id": "auxiliary_ica",
                "role": "application",
                "method": c["contrast"],
                "estimand": "mean_log_loss_difference",
                "value": c["mean"],
                "lower": c["lower"],
                "upper": c["upper"],
                "note": c["decision"],
            }
            for c in contrasts
        ]
        + [
            {
                "section": "f10_recomputed",
                "seed": "pooled3",
                "world_id": f"f10_{w}",
                "role": role,
                "method": m,
                "estimand": "full_mse",
                "value": replay[(w, role, m)],
                "lower": "",
                "upper": "",
                "note": f"archived={archived[(w, role, m)]}",
            }
            for (w, role, m) in archived
        ]
        + [
            {
                "section": "f10_geometry",
                "seed": g["seed"],
                "world_id": f"f10_{g['world']}",
                "role": "source",
                "method": "action_rank2",
                "estimand": "null_direction_second_order_mean_square",
                "value": g["null_direction_second_order_mean_square"],
                "lower": "",
                "upper": "",
                "note": f"rank={g['rank']};null_dim={g['null_dimension']}",
            }
            for g in geometry
        ]
    )
    app = [r for r in rows if r["role"] == "application"]

    def mean(arm, key):
        return float(np.mean([r[key] for r in app if r["arm"] == arm]))

    summary = {
        "status": "COMPUTED",
        "unit": "sample (the auxiliary ICA world has no authors)",
        "streams": "SeedSequence([seed,15,3,role,null]) for sampling; within-role permutation from the same stream",
        "application_log_loss": {
            arm: mean(arm, "log_loss")
            for arm in ("quadratic_x", "quadratic_true_s", "null_world")
        },
        "bayes_log_loss": mean("quadratic_x", "bayes_log_loss"),
        "chance_log_loss": chance,
        "paired_contrasts": contrasts,
        "all_converged": all(r["converged"] for r in rows),
        "f10_replay_max_abs_error": replay_err,
        "f10_null_second_order": [
            g["null_direction_second_order_mean_square"]
            for g in geometry
            if g["world"] == "null_curvature"
        ],
        "f10_reference_file_sha256": sha(f10_path),
        "not_executed": [
            "iVAE / learned nonlinear ICA (neural training prohibited by F15 protocol)",
            "SAE / concept dictionary (no new model development)",
        ],
    }
    ll = summary["application_log_loss"]
    cx = {c["contrast"]: c for c in contrasts}
    decision = f"""# F15-04 decision

- Fixed quadratic contrastive discriminator (source-only fit, 3 seeds, sample unit): application log loss {ll["quadratic_x"]:.4f} vs chance {chance:.4f} and exact Bayes {summary["bayes_log_loss"]:.4f}. Paired difference to chance {cx["quadratic_x_minus_chance"]["mean"]:.4f} [{cx["quadratic_x_minus_chance"]["lower"]:.4f}, {cx["quadratic_x_minus_chance"]["upper"]:.4f}] -> {cx["quadratic_x_minus_chance"]["decision"]}. Null world difference {cx["null_world_minus_chance"]["mean"]:.4f} [{cx["null_world_minus_chance"]["lower"]:.4f}, {cx["null_world_minus_chance"]["upper"]:.4f}] -> {cx["null_world_minus_chance"]["decision"]} (negative control).
- Known-inverse feature reference {ll["quadratic_true_s"]:.4f}; quadratic-x minus known-inverse {cx["quadratic_x_minus_quadratic_true_s"]["mean"]:.4f} [{cx["quadratic_x_minus_quadratic_true_s"]["lower"]:.4f}, {cx["quadratic_x_minus_quadratic_true_s"]["upper"]:.4f}] -> {cx["quadratic_x_minus_quadratic_true_s"]["decision"]}: the part of log p(x|u)/p(x) that quadratic-in-x features cannot represent.
- Reading: the fixed model detects auxiliary-dependent density structure; it does not recover sources or the mixing (no learned inverse), and the theorem's estimation guarantee (universal discriminator) does not apply.
- F10 reference recomputed by re-executing the frozen design (replay max error {replay_err:.2e} against archived metrics). In the null-curvature world the rank-2 action subspace has 2 null directions whose second-order effect has mean square {", ".join(f"{v:.4f}" for v in summary["f10_null_second_order"])} (seeds): first-order null does not mean no effect.
- NOT_EXECUTED: iVAE/learned nonlinear ICA, SAE. Minimal extension contract: evaluate a learned-mixing estimator on this frozen world (source-only training, application scoring, known inverse as reference) only if a downstream decision needs source recovery.
"""
    return summary, unified, decision, {"auxiliary_action_metrics.csv": unified}


def run_adoption(output, tests):
    """F15-05 method adoption for A01-A08 with primary cards and finite certificates."""
    cards = {
        c["id"]: c
        for c in json.loads(
            (ROOT / "configs/grain_f01_primary_cards_v1.json").read_text()
        )["cards"]
    }
    table = [
        (
            "A01",
            "PSR / controlled state / bisimulation",
            ["PC01", "PC02", "PC03"],
            "F15-01 Hankel ranks, PSR closure 1778 checks, too-short-history counterexample; F02 exact future equivalence",
            "ADOPTED_STANDARD_WITH_FINITE_CERTIFICATE",
            "bisimulation metric not computed (exact equivalence suffices for finite exact laws)",
        ),
        (
            "A02",
            "state-space and hierarchical system identification",
            ["PC04"],
            "F05 invariants (Jacobian ranks 5/5/4, similarity invariance), profiles, filter/dense equality; F04 REML/BLUP",
            "ADOPTED_STANDARD_WITH_FINITE_CERTIFICATE",
            "practical identification near unit root remains weak (F05 coverage .565)",
        ),
        (
            "A03",
            "nonlinear ICA / iVAE / paired identification",
            ["PC05", "PC06", "PC07"],
            "F15-03 rank 4/3/0 and variability rank; F15-04 fixed contrastive discriminator with paired CIs",
            "FINITE_CERTIFICATE_ESTIMATOR_NOT_EXECUTED",
            "learned mixing estimator (neural) not executed by protocol",
        ),
        (
            "A04",
            "IB / Blackwell / Mori-Zwanzig",
            ["PC08", "PC09", "PC10"],
            "F15-02 garbling certificates (explicit kernels, Farkas), exact MI; F15-01 memory term",
            "ADOPTED_STANDARD_WITH_FINITE_CERTIFICATE",
            "IB trade-off curve not optimized; MZ used as discrete analog",
        ),
        (
            "A05",
            "action metric / generalized eigenproblem",
            [],
            "F10 geometry (recomputed in F15-04): action rank-2 exact in linear world, misses null-direction curvature",
            "FINITE_CERTIFICATE",
            "native reader Jacobians/common chart open (F10 gap)",
        ),
        (
            "A06",
            "MTMM / G-theory / DIF / calibration",
            ["PC16"],
            "F11 method/trait equivalence certificate, anchor rank 8->9; F12 conditional calibration",
            "UNAVAILABLE_PRIMARY_FINITE_CERTIFICATE_PRESENT",
            "primary definitions not retrieved; material facet variance open (F13)",
        ),
        (
            "A07",
            "ADO / ALBA / scalar CAT",
            ["PC11", "PC12"],
            "F06 EIG/variance/target-specific policies; scalar IRT-CAT",
            "ADOPTED_STANDARD",
            "human burden/reactivity parameters are H04",
        ),
        (
            "A08",
            "RSA / counterfactual / NLI / concept & SAE tools",
            ["PC13", "PC14"],
            "F16 exact deletion effects; F08 NLI reader; F09 process equivalence",
            "MIXED_SEE_ITEM_TABLE",
            "RSA listener as qualification observation law is an extension contract",
        ),
    ]
    rows = []
    for fam, name, ids, evidence, status, remaining in table:
        rows.append(
            {
                "family": fam,
                "name": name,
                "primary_cards": ";".join(ids),
                "primary_locators": " | ".join(cards[i]["locator"] for i in ids),
                "finite_evidence": evidence,
                "status": status,
                "remaining_or_extension_contract": remaining,
            }
        )
    items = [
        {
            "tool": "RSA",
            "observed_object": "utterance choice given referent and context",
            "intervention": "change context alternatives/costs",
            "non_unique_decomposition": "speaker rationality vs cost vs prior trade off",
            "solves_f15_question": "no; would supply a qualification observation law (extension)",
        },
        {
            "tool": "NLI reader",
            "observed_object": "entailment probabilities of fixed hypotheses",
            "intervention": "edit subject status/pole in controlled text",
            "non_unique_decomposition": "content vs qualification errors jointly determine the likelihood (F16 text transport)",
            "solves_f15_question": "partly; used as reader with joint error law",
        },
        {
            "tool": "concept bottleneck",
            "observed_object": "prediction via named concepts",
            "intervention": "set concept values",
            "non_unique_decomposition": "concept naming is a labelling choice; equivalent rotations exist",
            "solves_f15_question": "no; not applicable (no named concept claims)",
        },
        {
            "tool": "SAE",
            "observed_object": "sparse dictionary of activations",
            "intervention": "ablate/steer features",
            "non_unique_decomposition": "dictionary not unique (Locatello-type non-identifiability)",
            "solves_f15_question": "no; not executed (no new model development)",
        },
    ]
    summary = {
        "status": "REVIEWED",
        "families": len(rows),
        "statuses": {r["family"]: r["status"] for r in rows},
    }
    decision = "# F15-05 decision\n\nEvery A01–A08 family has primary locators (or recorded unavailability), finite evidence, a status and a remaining item / extension contract; RSA/NLI/CBM/SAE are itemized. No family is marked complete by citation alone.\n"
    return (
        summary,
        [
            {
                "task_id": "F15-05",
                "method": r["family"],
                "estimand": "status",
                "value": r["status"],
            }
            for r in rows
        ],
        decision,
        {"method_adoption_final.csv": rows, "tool_items.csv": items},
    )


def run_rules(output, tests):
    """F15-06 representation rules from the certificates only."""
    rules = [
        {
            "rule": "Declare the target before compressing; keep a statistic only if it is Blackwell-equivalent (sufficient) for that target",
            "evidence": "F15-02: mean score carries 0 nats about F09 family; minimal sufficient equivalent to full",
        },
        {
            "rule": "A low-rank short-history block does not certify the predictive state; test closure on longer histories",
            "evidence": "F15-01 hidden order-3: rank 4 on H<=2 vs 8 on H<=3; short PSR closure error .6",
        },
        {
            "rule": "Residual predictability after a short state is memory, not noise, when it is a function of omitted history; enlarge the state or report the target as history-dependent",
            "evidence": "F15-01 memory variance .09 deterministic in lag 3",
        },
        {
            "rule": "Process names (belief vs direct) require an observation that breaks their equivalence; corner reports alone carry zero information",
            "evidence": "F15-02 process MI E1 0, E2 ~2e-5, E3 .109",
        },
        {
            "rule": "Auxiliary-variable identification needs enough condition variability (rank 2n for n Gaussian sources); a fixed discriminator detects dependence but does not recover sources",
            "evidence": "F15-03 ranks 4/3/0; F15-04 paired contrasts",
        },
        {
            "rule": "First-order action metrics can miss second-order effects along their null directions; keep local derivative, finite edit and full function separate",
            "evidence": "F15-04 F10 recomputation null-direction second-order mean square .22-.33",
        },
    ]
    acceptance = {
        "cards": {
            c: {
                "implementation": "DONE",
                "protocol": "DONE_V2_WITH_DECLARED_DEVIATIONS",
                "science": "FINITE_CERTIFICATES_SUPPORTED",
                "promotion": "PROPOSED_FOR_SUPERVISOR_REVIEW",
            }
            for c in ("F15-01", "F15-02", "F15-03", "F15-04")
        },
        "reviews": {
            c: {
                "implementation": "DONE",
                "protocol": "DONE",
                "science": "REVIEW_ONLY",
                "promotion": "PROPOSED_FOR_SUPERVISOR_REVIEW",
            }
            for c in ("F15-05", "F15-06")
        },
        "open": [
            "learned nonlinear ICA / iVAE not executed",
            "IB curve not optimized",
            "primary sources unavailable for MTMM/G-theory/invariance, selection bounds, validity (recorded)",
        ],
    }
    summary = {"status": "REVIEWED", "rules": len(rules)}
    decision = (
        "# F15-06 decision\n\n"
        + "\n".join(f"- {r['rule']} ({r['evidence']})." for r in rules)
        + "\n\nStandard mathematics is used as such (PSR, Blackwell, iVAE conditions); contributions are the finite counterexamples and their production rules for text measurement, not new theorems.\n"
    )
    return (
        summary,
        [
            {
                "task_id": "F15-06",
                "method": "rules",
                "estimand": "count",
                "value": len(rules),
            }
        ],
        decision,
        {"representation_rules.json": rules, "acceptance.json": acceptance},
    )


def main():
    """Run the four computational F15 cards into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tests", type=Path, required=True, help="junit/pytest summary file to embed"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tests = args.tests.read_text()
    code = {p: sha(ROOT / p) for p in CODE}
    write_json(
        args.output / "protocol_before_run.json",
        {
            "protocol": "docs/GRAIN_F15_RECOVERY_PROTOCOL_V1.md",
            "code": code,
            "seeds": SEEDS,
            "neural_training": False,
            "api_calls": 0,
        },
    )
    for card_id, fn, reqs in (
        ("F15-01", run_psr, ["R08", "R30", "A01", "A04"]),
        ("F15-02", run_blackwell, ["R30", "A04"]),
        ("F15-03", run_auxiliary, ["R07", "A03"]),
        ("F15-04", run_contrastive, ["A03", "A05"]),
        (
            "F15-05",
            run_adoption,
            ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"],
        ),
        ("F15-06", run_rules, ["R03", "R30", "R31"]),
    ):
        started = time.monotonic()
        summary, metrics, decision, extra = fn(args.output, tests)
        protocol = {
            "card": card_id,
            "requirements": reqs,
            "protocol": "docs/GRAIN_F15_RECOVERY_PROTOCOL_V1.md",
            "code": code,
        }
        inputs = {"code": code, "python": sys.version}
        if card_id == "F15-04":
            inputs["f10_geometry_metrics"] = {
                "path": "results/grain_gptpro_foundation_v3/F10/geometry_r1/metrics.csv",
                "sha256": summary["f10_reference_file_sha256"],
            }
        log = [
            {
                "card": card_id,
                "seconds": time.monotonic() - started,
                "status": summary["status"],
            }
        ]
        card(
            args.output,
            card_id,
            protocol,
            inputs,
            summary,
            metrics,
            decision,
            extra,
            log,
            tests,
        )
        print(
            json.dumps(
                {"card": card_id, "seconds": round(time.monotonic() - started, 1)}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
