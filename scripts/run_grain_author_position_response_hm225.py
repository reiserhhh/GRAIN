#!/usr/bin/env python3
"""One finite author-level/observer-response anchor join, not precision tuning."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_position_response as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_author_position_response_hm225.json"
DEFAULT = ROOT / "results/grain_math_response_v1/author_position_response_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_AUTHOR_POSITION_RESPONSE_PLAN_HM225.md",
    "grain_research/construction/math_author_position_response.py",
    "scripts/run_grain_author_position_response_hm225.py",
    "scripts/score_grain_author_position_response_cold.py",
    "tests/test_grain_author_position_response.py",
    "grain_research/construction/math_prospective_evidence.py",
)
NAMES = (
    "source_level_full",
    "source_level_association",
    "source_level_departure",
    "source_response_full",
    "source_response_association",
    "source_response_departure",
    "native_level",
    "native_response",
    "native_level_relative",
    "native_response_relative",
)
STAGES = ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")


def selected(path, keys):
    """Read an explicit array allowlist without pickle or other dataset keys."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def number(value):
    """Keep undefined statistics missing rather than encoding them as zero."""
    return float(value) if np.isfinite(value) else None


def interval(values):
    """Use only defined whole-author resamples and record the valid count."""
    finite = np.asarray(values)[np.isfinite(values)]
    return (
        np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None]
    ), len(finite)


def prepare(out):
    """Seal identical numerical evidence before interpreting human outcomes."""
    if out.exists():
        raise FileExistsError("Existing science is not replaced or retried")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    z = selected(
        parent / "observations.npz",
        (
            "source_means",
            "codes",
            "variance",
            "source_indices",
            "panel_indices",
            "vectors",
            "full_accounts",
        ),
    )
    n = selected(parent / "analysis.npz", ("native_readings", "source_actions"))
    selections = R.read(parent / "selections.json")
    ids = selected(ROOT / cfg["labels"], ("author_ids",))["author_ids"]
    same = selected(ROOT / cfg["identity_material"], ("author_ids",))["author_ids"]
    if (
        ids.shape != (624,)
        or not np.array_equal(ids, same)
        or len(set(ids.tolist())) != 624
    ):
        raise ValueError("Existing public identities/order mismatch")
    if (
        z["vectors"].shape != (24, 12, 1024)
        or z["source_means"].shape != (386, 2, 1024)
        or n["native_readings"].shape != (24, 3, 5)
    ):
        raise ValueError("Full declared same-author scope required")
    if (
        set(z["panel_indices"]) & set(z["source_indices"])
        or len(set(z["panel_indices"].tolist())) != 24
    ):
        raise ValueError("Source/application roles overlap or repeat")
    if len(selections) != 24 or any(
        r["author"] != u or r["original_author_index"] != int(z["panel_indices"][u])
        for u, r in enumerate(selections)
    ):
        raise ValueError("Different event-to-author binding")
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "prepared_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "observations.npz",
            "analysis.npz",
            "selections.json",
        )
    ]
    inputs += [ROOT / cfg["labels"], ROOT / cfg["identity_material"]]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "application_outcome_values_read_before_seal": False,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz", **z, **n, panel_author_ids=ids[z["panel_indices"]]
    )
    R.write(out / "event_selections.json", selections)
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "source_authors": 386,
            "distinct_events": 288,
            "source_panel_overlap": 0,
            "public_numeric_ID_order_verified": True,
            "application_y_values_read": False,
            "historical_exploratory_not_new_lockbox": True,
            "new_neural_private_API_train_calls": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "event_selections.json", "input_quality.json"),
    )
    print(
        "PREPARED HM225;PUBLIC numeric identity verified;application y not interpreted"
    )


def compute(out):
    """Build the fixed-origin record and actually invoke a separate cold scorer."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    start = time.monotonic()
    bank = {k: z[k] for k in ("source_means", "codes", "variance")}
    origin = M.reference(bank)
    bundle = {
        "bank": {k: a.tolist() for k, a in bank.items()},
        "origin": origin.tolist(),
        "reference": cfg["reference"],
        "targets": cfg["targets"],
    }
    rows = [
        {
            "author": u,
            "event_ids": [str(e) for e in r["event_ids"]],
            "vectors": z["vectors"][u].tolist(),
            "support": r["support"],
            "counter": r["counter"],
        }
        for u, r in enumerate(R.read(out / "event_selections.json"))
    ]
    R.write(out / "source_bundle.json", bundle)
    R.write(out / "cold_inputs.json", rows)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_author_position_response_cold.py"),
            "--bundle",
            str(out / "source_bundle.json"),
            "--inputs",
            str(out / "cold_inputs.json"),
            "--output",
            str(out / "cold_scores.json"),
        ],
        capture_output=True,
        text=True,
        timeout=cfg["max_cpu_seconds"],
        check=False,
    )
    R.write(
        out / "cold_receipt.json",
        {
            "command": process.args,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "no_model_outcome_calls": True,
        },
    )
    process.check_returncode()
    results = R.read(out / "cold_scores.json")
    if [r["author"] for r in results] != list(range(24)):
        raise ValueError("Cold output author binding mismatch")
    arrays = {
        k: np.asarray([r[k] for r in results])
        for k in ("absolute", "positions", "response")
    }
    a = M.objects(arrays["positions"], arrays["response"], z["native_readings"])
    R.archive(out / "score_records.npz", **arrays, origin=origin, objects=a)
    elapsed = time.monotonic() - start
    if elapsed > cfg["max_cpu_seconds"]:
        raise RuntimeError("CPU cap exceeded;preserve output and do not rerun")
    R.write(
        out / "compute_runtime.json",
        {
            "seconds": elapsed,
            "authors": 24,
            "new_neural_calls": 0,
            "outcome_values_read": False,
            "no_refit": True,
        },
    )
    R.seal(
        out,
        "computed_seal.json",
        (
            "source_bundle.json",
            "cold_inputs.json",
            "cold_scores.json",
            "cold_receipt.json",
            "score_records.npz",
            "compute_runtime.json",
        ),
    )
    print("COMPUTED HM225 cold source-only record", elapsed, flush=True)


def analyze(out):
    """Join the existing authorized public outcomes once, without score fitting."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    start = time.monotonic()
    z, a = (
        R.prepared(out),
        selected(out / "score_records.npz", ("objects", "positions", "response")),
    )
    lhs = selected(ROOT / cfg["labels"], ("author_ids", "y"))
    rhs = selected(ROOT / cfg["identity_material"], ("author_ids", "y"))
    y = M.join(
        lhs["author_ids"],
        rhs["author_ids"],
        lhs["y"],
        rhs["y"],
        z["panel_indices"],
        z["source_indices"],
    )
    result = M.associations(
        a["objects"], y, cfg["bootstrap_draws"], cfg["permutations"], cfg["seed"]
    )
    rows = []
    for i, obj in enumerate(NAMES):
        for q, target in enumerate(cfg["targets"]):
            for t, anchor in enumerate(cfg["targets"]):
                ci, count = interval(result["bootstrap"][:, i, q, t])
                rows.append(
                    {
                        "object": obj,
                        "target": target,
                        "human_anchor": anchor,
                        "pearson_r": number(result["pearson"][i, q, t]),
                        "spearman_rho": number(result["spearman"][i, q, t]),
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                        "bootstrap_valid": count,
                        "pointwise_permutation_tail": number(
                            result["point_tail"][i, q, t]
                        ),
                        "family250_permutation_tail": number(
                            result["family_tail"][i, q, t]
                        ),
                        "authors": len(y),
                    }
                )
    R.table(out / "all_contacts.csv", rows)
    delta_rows = []
    for level, response in ((0, 3), (2, 5), (6, 7), (8, 9)):
        for q, target in enumerate(cfg["targets"]):
            delta = result["pearson"][level, q, q] - result["pearson"][response, q, q]
            boot = (
                result["bootstrap"][:, level, q, q]
                - result["bootstrap"][:, response, q, q]
            )
            ci, count = interval(boot)
            delta_rows.append(
                {
                    "level": NAMES[level],
                    "response": NAMES[response],
                    "anchor": target,
                    "delta_r_level_minus_response": number(delta),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "bootstrap_valid": count,
                }
            )
    R.table(out / "paired_differences.csv", delta_rows)
    mechanism = []
    for i in (3, 4, 5):
        matrix = M.correlations(a["objects"][:, i], a["objects"][:, 7])
        for q, target in enumerate(cfg["targets"]):
            for t, native in enumerate(cfg["targets"]):
                mechanism.append(
                    {
                        "source_response": NAMES[i],
                        "target": target,
                        "native_target": native,
                        "response_response_r": number(matrix[q, t]),
                        "human_outcomes_used": False,
                    }
                )
    R.table(out / "mechanism_contacts.csv", mechanism)
    covariance_rows, errors = [], []
    for name, full, first, second in (
        ("source_level", a["objects"][:, 0], a["objects"][:, 1], a["objects"][:, 2]),
        ("source_response", a["objects"][:, 3], a["objects"][:, 4], a["objects"][:, 5]),
        (
            "native_level",
            a["objects"][:, 6],
            np.repeat(a["objects"][:, 6].mean(1, keepdims=True), 5, axis=1),
            a["objects"][:, 8],
        ),
        (
            "native_response",
            a["objects"][:, 7],
            np.repeat(a["objects"][:, 7].mean(1, keepdims=True), 5, axis=1),
            a["objects"][:, 9],
        ),
    ):
        total, part1, part2 = (M.covariance(b, y) for b in (full, first, second))
        errors.append(float(np.max(abs(total - part1 - part2))))
        for q, target in enumerate(cfg["targets"]):
            for t, anchor in enumerate(cfg["targets"]):
                covariance_rows.append(
                    {
                        "object": name,
                        "target": target,
                        "human_anchor": anchor,
                        "full_covariance": float(total[q, t]),
                        "association_or_common": float(part1[q, t]),
                        "departure_or_relative": float(part2[q, t]),
                        "identity_error": float(
                            total[q, t] - part1[q, t] - part2[q, t]
                        ),
                    }
                )
    R.table(out / "covariance_accounts.csv", covariance_rows)
    R.archive(out / "analysis.npz", public_outcomes=y, **result)
    primary = [r for r in rows if r["target"] == r["human_anchor"] == "A"]
    summary = {
        "packet": "HM225",
        "primary_A": primary,
        "all_diagonals": [r for r in rows if r["target"] == r["human_anchor"]],
        "primary_A_paired_differences": [r for r in delta_rows if r["anchor"] == "A"],
        "primary_A_mechanism": [
            r for r in mechanism if r["target"] == r["native_target"] == "A"
        ],
        "covariance_identity_max": max(errors),
        "objects": list(NAMES),
        "all_contact_count": len(rows),
        "authors": len(y),
        "public_human_anchor_joined_after_frozen_score": True,
        "no_prediction_promotion_or_fit": True,
        "original_scope_complete": False,
        "seconds": time.monotonic() - start,
    }
    if summary["seconds"] > cfg["max_cpu_seconds"]:
        raise RuntimeError("CPU analysis cap exceeded;no restart")
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "all_contacts.csv",
            "paired_differences.csv",
            "mechanism_contacts.csv",
            "covariance_accounts.csv",
            "analysis.npz",
            "summary.json",
        ),
    )
    print(
        json.dumps(
            {
                "A_diagonals": {r["object"]: r["pearson_r"] for r in primary},
                "A_mechanism": summary["primary_A_mechanism"],
            }
        ),
        flush=True,
    )


def literal(bank, points):
    """Independent distance-based likelihood arithmetic, without matrix logits."""
    source, codes, variance = (
        np.asarray(bank["source_means"]),
        np.asarray(bank["codes"]),
        float(bank["variance"]),
    )
    output = []
    for x in points:
        k = np.exp(-np.sum((source - x) ** 2, -1) / (2 * variance)).mean(1)
        law = np.array([k[codes == g].sum() / len(k) for g in range(32)])
        accounts = []
        for q in range(5):
            high = ((np.arange(32) >> q) & 1).astype(bool)
            other = codes & ~(1 << q)
            base = np.array(
                [
                    k[other == (g & ~(1 << q))].mean()
                    * np.count_nonzero(codes == g)
                    / len(k)
                    for g in range(32)
                ]
            )
            f, b = (
                np.log(law[high].sum() / law[~high].sum()),
                np.log(base[high].sum() / base[~high].sum()),
            )
            accounts.append([f, b, f - b])
        output.append(np.asarray(accounts).T)
    return np.asarray(output)


def verify(out):
    """Verify every new cold account and scalar anchor association once."""
    R.check(out, STAGES[:2])
    z, a = (
        R.prepared(out),
        selected(
            out / "score_records.npz",
            ("absolute", "positions", "response", "origin", "objects"),
        ),
    )
    bundle = R.read(out / "source_bundle.json")
    origin = literal(bundle["bank"], z["source_means"].reshape(-1, 1024)).mean(0)
    kernel_error, cold_error = float(np.max(abs(origin - a["origin"]))), 0.0
    for u, row in enumerate(R.read(out / "cold_inputs.json")):
        score = M.score(
            bundle["bank"],
            row["vectors"],
            row["event_ids"],
            row["support"],
            row["counter"],
            bundle["origin"],
        )
        cold_error = max(
            cold_error,
            *(
                float(np.max(abs(score[k] - a[k][u])))
                for k in ("absolute", "positions", "response")
            ),
        )
        kernel_error = max(
            kernel_error,
            float(
                np.max(abs(literal(bundle["bank"], score["points"]) - a["absolute"][u]))
            ),
        )
    replay = max(
        float(np.max(abs(a["absolute"][:, 0] - z["full_accounts"]))),
        float(np.max(abs(a["response"] - z["source_actions"]))),
    )
    analysis = selected(out / "analysis.npz", ("public_outcomes", "pearson"))
    r_error = 0.0
    for i in range(10):
        for q in range(5):
            for t in range(5):
                x, y = a["objects"][:, i, q], analysis["public_outcomes"][:, t]
                dx, dy = x - sum(x) / len(x), y - sum(y) / len(y)
                denominator = np.sqrt(sum(dx * dx) * sum(dy * dy))
                ref = sum(dx * dy) / denominator if denominator > 0 else np.nan
                got = analysis["pearson"][i, q, t]
                if np.isfinite(ref) != np.isfinite(got):
                    raise ValueError("Undefined correlation was not preserved")
                if np.isfinite(ref):
                    r_error = max(r_error, abs(float(ref - got)))
    cov_error = R.read(out / "summary.json")["covariance_identity_max"]
    receipt = {
        "literal_kernel_and_origin_max": kernel_error,
        "cold_direct_max": cold_error,
        "frozen_HM218_score_replay_max": replay,
        "all250_scalar_r_max": r_error,
        "covariance_identity_max": cov_error,
        "new_neural_calls": 0,
        "same_agent_not_blind": True,
        "PASS": max(kernel_error, cold_error, replay, r_error, cov_error) < 2e-11,
    }
    R.write(out / "verification.json", receipt)
    if not receipt["PASS"]:
        raise ValueError("Verification failed;do not rerun or change sealed science")
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt), flush=True)


def main():
    """Run one declared stage, preserving failures and all older artifacts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check")
    )
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    if a.stage == "check":
        R.check(a.output, STAGES)
        print("HM225 ALL SEALS PASS")
    else:
        {"prepare": prepare, "compute": compute, "analyze": analyze, "verify": verify}[
            a.stage
        ](a.output)


if __name__ == "__main__":
    main()
