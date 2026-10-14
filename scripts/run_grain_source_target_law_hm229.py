#!/usr/bin/env python3
"""Construct and inspect one full joint SOURCE law without accuracy tuning."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_comparison_geometry as G
from grain_research.construction import math_prospective_evidence as P
from grain_research.construction import math_source_target_law as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_source_target_law_hm229.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_target_law_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SOURCE_TARGET_LAW_PLAN_HM229.md",
    "grain_research/construction/math_source_target_law.py",
    "scripts/run_grain_source_target_law_hm229.py",
    "scripts/score_grain_source_target_law_cold.py",
    "tests/test_grain_source_target_law.py",
    "grain_research/construction/math_prospective_evidence.py",
    "grain_research/construction/math_author_comparison_geometry.py",
    "scripts/run_grain_observed_family_hm170.py",
)
STAGES = ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")


def selected(path, keys):
    """Read only authorized numeric keys, with pickle disabled."""
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in keys}


def prepare(out):
    """Seal full existing scope before SOURCE-law computation or comparisons."""
    if out.exists():
        raise FileExistsError("Existing science cannot be replaced or retried")
    cfg = R.read(ROOT / CONFIG)
    parent, comparison, position = (
        ROOT / cfg[k] for k in ("parent", "comparison_parent", "position_parent")
    )
    R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    R.check(
        comparison,
        (
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "ordinal_seal.json",
        ),
    )
    R.check(position, STAGES)
    z = selected(
        parent / "observations.npz",
        (
            "source_means",
            "codes",
            "variance",
            "source_indices",
            "panel_indices",
            "vectors",
        ),
    )
    graph = selected(comparison / "observations.npz", ("edges", "panel_indices"))
    selections = R.read(parent / "selections.json")
    allowed = selected(ROOT / cfg["labels"], ("author_ids", "y"))
    same_ids = selected(ROOT / cfg["identity_material"], ("author_ids",))["author_ids"]
    ids, anchors = allowed["author_ids"], allowed["y"][z["source_indices"]]
    if (
        ids.shape != (624,)
        or not np.array_equal(ids, same_ids)
        or len(set(ids.tolist())) != 624
        or not np.isfinite(anchors).all()
        or anchors.shape != (386, 5)
        or np.any((anchors < 0) | (anchors > 1))
    ):
        raise ValueError(
            "Existing public numeric SOURCE identity/anchor contract changed"
        )
    if (
        z["source_means"].shape != (386, 2, 1024)
        or z["vectors"].shape != (24, 12, 1024)
        or z["source_indices"].shape != (386,)
        or z["panel_indices"].shape != (24,)
        or len(set(z["source_indices"].tolist())) != 386
        or len(set(z["panel_indices"].tolist())) != 24
        or set(z["source_indices"]) & set(z["panel_indices"])
        or float(z["variance"]) != cfg["variance"]
        or not np.array_equal(graph["panel_indices"], z["panel_indices"])
        or not np.array_equal(graph["edges"], G.graph())
    ):
        raise ValueError("Changed frozen vector, role or comparison scope")
    if len(selections) != 24 or any(
        r["author"] != u or r["original_author_index"] != int(z["panel_indices"][u])
        for u, r in enumerate(selections)
    ):
        raise ValueError("Wrong event-to-author binding")
    for u, row in enumerate(selections):
        M.points(
            z["vectors"][u],
            [str(e) for e in row["event_ids"]],
            row["support"],
            row["counter"],
        )
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "prepared_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "observations.npz",
            "selections.json",
        )
    ]
    inputs += [
        comparison / p
        for p in (
            "manifest.json",
            "observations.npz",
            "analysis.npz",
            "analyzed_seal.json",
            "verification_seal.json",
            "ordinal_seal.json",
        )
    ]
    inputs += [
        position / "score_records.npz",
        ROOT / cfg["labels"],
        ROOT / cfg["identity_material"],
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "public_y_block_decompressed_before_SOURCE_selection": True,
            "application_y_values_used": False,
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **z, anchors=anchors, edges=graph["edges"])
    R.write(out / "event_selections.json", selections)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "application_authors": 24,
            "application_events": 288,
            "source_panel_overlap": 0,
            "SOURCE_anchor_min": anchors.min(0).tolist(),
            "SOURCE_anchor_max": anchors.max(0).tolist(),
            "application_y_used": False,
            "embedding": "existing neural E5 full1024,not a new LLM-free encoder",
            "historical_exploratory_not_lockbox": True,
            "new_neural_API_private_train_calls": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "event_selections.json", "input_quality.json"),
    )
    print("PREPARED HM229:386SOURCE/24authors/288events/5targets", flush=True)


def compute(out):
    """Run an independent numeric cold process and retain all seventy-two laws."""
    start = time.monotonic()
    cfg, z = R.check(out)["config"], R.prepared(out)
    bank = {k: z[k] for k in ("source_means", "codes", "variance", "anchors")}
    M.validate(bank)
    R.archive(out / "source_bundle.npz", **bank)
    entries = [
        {
            "author": u,
            "event_ids": [str(e) for e in r["event_ids"]],
            "vectors": z["vectors"][u].tolist(),
            "support": r["support"],
            "counter": r["counter"],
        }
        for u, r in enumerate(R.read(out / "event_selections.json"))
    ]
    R.write(out / "cold_inputs.json", entries)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_source_target_law_cold.py"),
            "--bundle",
            str(out / "source_bundle.npz"),
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
        },
    )
    process.check_returncode()
    results = R.read(out / "cold_scores.json")
    if [r["author"] for r in results] != list(range(24)):
        raise ValueError("Cold author addresses changed")
    data = {k: np.asarray([r[k] for r in results]) for k in results[0] if k != "author"}
    R.archive(out / "laws.npz", **data)
    seconds = time.monotonic() - start
    R.write(
        out / "runtime.json",
        {
            "seconds": seconds,
            "laws": 72,
            "new_neural_calls": 0,
            "no_fit_or_grid": True,
            "CPU_cap_pass": seconds <= cfg["max_cpu_seconds"],
        },
    )
    if seconds > cfg["max_cpu_seconds"]:
        raise RuntimeError("CPU cap exceeded;preserve output without retry")
    R.seal(
        out,
        "computed_seal.json",
        (
            "source_bundle.npz",
            "cold_inputs.json",
            "cold_scores.json",
            "cold_receipt.json",
            "laws.npz",
            "runtime.json",
        ),
    )
    print(f"COMPUTED HM229 cold laws in {seconds:.3f}s", flush=True)


def cosine(a, b):
    """Keep undefined contact missing; this is geometry, not human Pearson r."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else None


def analyze(out):
    """Describe full law shapes and old observer contacts without tuning to them."""
    start = time.monotonic()
    cfg = R.check(out, STAGES[:1])["config"]
    z, a = (
        R.prepared(out),
        selected(out / "laws.npz", ("weights", "mean", "high_probability")),
    )
    w, y, edges = a["weights"], z["anchors"], z["edges"]
    native = selected(ROOT / cfg["comparison_parent"] / "analysis.npz", ("odd",))["odd"]
    if native.shape != (32, 5):
        raise ValueError("Missing complete observed comparison graph")
    pi = np.stack([M.superiority(w[:, arm], y) for arm in range(3)])
    ix = np.random.default_rng(cfg["null_seed"]).permutation(len(y))
    null = M.superiority(w[:, 0], y[ix])
    cdfs, supports = [], []
    for q in range(5):
        support = np.unique(y[:, q])
        supports.append(support)
        cdfs.append(
            M.cdf(w.reshape(72, 386), y, q, support).reshape(24, 3, len(support))
        )
    arrays = {
        "pair_probability": pi,
        "permuted_pair_probability": null,
        "permutation": ix,
    }
    arrays.update({f"support_{q}": v for q, v in enumerate(supports)})
    arrays.update({f"cdf_{q}": v for q, v in enumerate(cdfs)})
    R.archive(out / "distribution_records.npz", **arrays)
    rows, cycles, pair_rows = [], [], []
    all_edges = np.array([(i, j) for i in range(24) for j in range(i + 1, 24)])
    for arm, name in enumerate(cfg["arms"]):
        for q, target in enumerate(cfg["targets"]):
            d = 2 * pi[arm, edges[:, 0], edges[:, 1], q] - 1
            md = a["mean"][edges[:, 0], arm, q] - a["mean"][edges[:, 1], arm, q]
            pd = (
                a["high_probability"][edges[:, 0], arm, q]
                - a["high_probability"][edges[:, 1], arm, q]
            )
            crossing = []
            for k, (i, j) in enumerate(edges):
                diff = cdfs[q][i, arm] - cdfs[q][j, arm]
                is_crossing = bool(np.any(diff > 1e-12) and np.any(diff < -1e-12))
                crossing.append(is_crossing)
                pair_rows.append(
                    {
                        "arm": name,
                        "target": target,
                        "left": int(i),
                        "right": int(j),
                        "Pi": float(pi[arm, i, j, q]),
                        "signed_PI": float(d[k]),
                        "mean_difference": float(md[k]),
                        "binary_probability_difference": float(pd[k]),
                        "CDF_crossing": is_crossing,
                        "old_native_odd_nats": float(native[k, q]),
                    }
                )
            full_d = 2 * pi[arm, all_edges[:, 0], all_edges[:, 1], q] - 1
            cycles.append(
                {
                    "arm": name,
                    "target": target,
                    "same32_edge_cycle": M.directed_cycle(edges, d, 24),
                    "mathematical_full_pair_cycle": M.directed_cycle(
                        all_edges, full_d, 24
                    ),
                }
            )
            rows.append(
                {
                    "arm": name,
                    "target": target,
                    "same_graph_edges": 32,
                    "CDF_crossings": int(sum(crossing)),
                    "mean_vs_PI_sign_mismatches": int(np.sum(md * d < -1e-24)),
                    "binary_vs_PI_sign_mismatches": int(np.sum(pd * d < -1e-24)),
                    "PI_vs_native_cosine": cosine(d, native[:, q]),
                    "permuted_PI_vs_native_cosine": cosine(
                        2 * null[edges[:, 0], edges[:, 1], q] - 1, native[:, q]
                    ),
                    "max_abs_PI_departure_from_half": float(abs(d).max() / 2),
                }
            )
    R.table(out / "same_graph_contacts.csv", rows)
    R.table(out / "all_pair_accounts.csv", pair_rows)
    R.write(out / "cycle_witnesses.json", cycles)
    grams = {}
    for arm, name in enumerate(cfg["arms"]):
        d = 2 * pi[arm, edges[:, 0], edges[:, 1]] - 1
        grams[name] = [[cosine(d[:, q], d[:, r]) for r in range(5)] for q in range(5)]
    R.write(
        out / "summary.json",
        {
            "mechanism": "one joint SOURCE law before functionals;no neural response or anchor fit",
            "same_graph_contacts": rows,
            "cycles": cycles,
            "target_edge_cosines": grams,
            "application_y_used": False,
            "original_scope_complete": False,
            "permutation_control": "one fixed joint SOURCE assignment shuffle;not a p-value",
            "distribution_spread": "SOURCE-law spread,not confidence interval/trait-state variance",
            "seconds": time.monotonic() - start,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "distribution_records.npz",
            "same_graph_contacts.csv",
            "all_pair_accounts.csv",
            "cycle_witnesses.json",
            "summary.json",
        ),
    )
    print("ANALYZED HM229:all5targets/3arms/32oldedges retained", flush=True)


def verify(out):
    """Recompute all probability accounts with sorted-support arithmetic."""
    cfg = R.check(out, STAGES[:2])["config"]
    z = R.prepared(out)
    a = selected(out / "laws.npz", ("weights", "mean", "covariance", "high_log_odds"))
    records = selected(out / "distribution_records.npz", ("pair_probability",))
    w, y = a["weights"], z["anchors"]
    replay = np.empty_like(records["pair_probability"])
    cdf_error = 0.0
    for arm in range(3):
        for q in range(5):
            levels, inverse = np.unique(y[:, q], return_inverse=True)
            for right in range(24):
                mass = np.bincount(
                    inverse, weights=w[right, arm], minlength=len(levels)
                )
                below = np.r_[0, np.cumsum(mass)[:-1]]
                witness = (below + mass / 2)[inverse]
                replay[arm, :, right, q] = np.sum(w[:, arm] * witness, axis=1)
            stored = selected(out / "distribution_records.npz", (f"cdf_{q}",))[
                f"cdf_{q}"
            ]
            for t, level in enumerate(levels):
                literal = sum(w[:, arm, i] for i in range(386) if y[i, q] <= level)
                cdf_error = max(
                    cdf_error, float(abs(literal - stored[:, arm, t]).max())
                )
    selections = R.read(out / "event_selections.json")
    point_rows = np.stack(
        [
            M.points(
                z["vectors"][u],
                [str(e) for e in r["event_ids"]],
                r["support"],
                r["counter"],
            )
            for u, r in enumerate(selections)
        ]
    )
    bank = {k: z[k] for k in ("source_means", "codes", "variance", "anchors")}
    inherited = P.accounts(bank, point_rows.reshape(72, 1024))[:, 0].reshape(24, 3, 5)
    old = selected(ROOT / cfg["position_parent"] / "score_records.npz", ("absolute",))[
        "absolute"
    ][:, :, 0]
    rows = [M.score(bank, r) for r in R.read(out / "cold_inputs.json")]
    cold_error = max(
        float(abs(np.stack([r[key] for r in rows]) - a[key]).max())
        for key in ("weights", "mean", "covariance", "high_log_odds")
    )
    common = np.broadcast_to(w[0, 0], (24, 386))
    checks = {
        "sorted_support_pair_replay_max": float(
            abs(replay - records["pair_probability"]).max()
        ),
        "literal_all_CDF_max": cdf_error,
        "cold_parent_process_replay_max": cold_error,
        "old_binary_formula_max": float(abs(inherited - a["high_log_odds"]).max()),
        "old_HM225_binary_record_max": float(abs(old - a["high_log_odds"]).max()),
        "normalization_max": float(abs(w.sum(-1) - 1).max()),
        "swap_max": float(
            abs(
                records["pair_probability"]
                + records["pair_probability"].swapaxes(1, 2)
                - 1
            ).max()
        ),
        "self_max": float(
            abs(
                records["pair_probability"][:, np.arange(24), np.arange(24)] - 0.5
            ).max()
        ),
        "common_law_max": float(abs(M.superiority(common, y) - 0.5).max()),
    }
    faces = np.tile(np.array([1, 5, 9, 3, 4, 8, 2, 6, 7])[:, None], (1, 5))
    dice = np.zeros((3, 9))
    for u in range(3):
        dice[u, u * 3 : (u + 1) * 3] = 1 / 3
    pi = M.superiority(dice, faces)
    checks["cited_dice_cycle_max"] = float(
        abs(pi[[0, 1, 2], [1, 2, 0], 0] - 5 / 9).max()
    )
    passed = max(checks.values()) <= 2e-11
    R.write(
        out / "verification.json",
        {
            "PASS": passed,
            "checks": checks,
            "pair_accounts": 8640,
            "law_records": 72,
            "audit": "same-agent alternate arithmetic,not independent blind replication",
            "no_precision_optimization": True,
            "cited_equal_mean_dice": {
                "mean": (dice @ faces)[:, 0].tolist(),
                "cyclic_win_probability": pi[[0, 1, 2], [1, 2, 0], 0].tolist(),
            },
        },
    )
    if not passed:
        raise ValueError("Finite account failed;preserve rather than refit/retry")
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM229:full pair/CDF/binary/cold accounts", flush=True)


def main():
    """Execute one explicit immutable stage; never automatically expand scope."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check")
    )
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    if a.stage == "check":
        R.check(a.out, STAGES)
        print("HM229 scientific and output seals PASS")
    else:
        globals()[a.stage](a.out)


if __name__ == "__main__":
    main()
