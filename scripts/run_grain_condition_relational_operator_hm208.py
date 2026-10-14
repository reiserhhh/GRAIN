#!/usr/bin/env python3
"""One same-event conditional relation experiment, with no prediction search."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_condition_relational_operator as M
from grain_research.construction import math_joint_target_atlas as A
from grain_research.construction import math_same_event_multitarget as N
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_relational_target_operator_hm207 as H

CONFIG = "configs/grain_condition_relational_operator_hm208.json"
DEFAULT = ROOT / "results/grain_math_response_v1/condition_relational_operator_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CONDITION_RELATIONAL_OPERATOR_PLAN_HM208.md",
    "grain_research/construction/math_condition_relational_operator.py",
    "scripts/run_grain_condition_relational_operator_hm208.py",
    "tests/test_grain_condition_relational_operator.py",
    "grain_research/construction/math_relational_target_operator.py",
    "grain_research/construction/math_joint_target_atlas.py",
    "grain_research/construction/math_same_event_multitarget.py",
    "scripts/run_grain_relational_target_operator_hm207.py",
    "scripts/run_grain_observed_family_hm170.py",
)
NAMES = (*M.ACCOUNTS, "all")


def prepare(out):
    """Authenticate SAME numeric events and pack source-only template summaries."""
    cfg = R.read(ROOT / CONFIG)
    rp, cp, ap = (
        ROOT / cfg[k]
        for k in ("relation_parent", "condition_parent", "application_parent")
    )
    R.check(rp, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    R.check(cp, ("compiled_seal.json", "scored_seal.json", "verification_seal.json"))
    R.check(ap, ("compiled_seal.json", "analysis_seal.json"))
    h, c, app = H.arrays(rp / "observations.npz"), R.prepared(cp), R.prepared(ap)
    errors = {
        "source_means": float(np.max(np.abs(c["vectors"].mean(2) - h["source_means"]))),
        "panel_coordinates": float(
            np.max(np.abs(h["panel_vectors"] @ h["mapping"] - app["coordinates"][:, 0]))
        ),
    }
    if c["vectors"].shape != (386, 2, 12, 1024) or app["conditions"].shape != (
        24,
        2,
        12,
    ):
        raise ValueError("Changed registered condition-event scope")
    if any(not np.array_equal(c[k], h[k]) for k in ("source_indices", "folds")):
        raise ValueError("Source author/fold linkage changed")
    if not np.array_equal(A.profile_codes(c["labels"]), h["codes"]):
        raise ValueError("Source target profiles changed")
    if any(
        not np.array_equal(app[k], h[k]) for k in ("author_ids", "selected", "control")
    ):
        raise ValueError("Application identity or edit addresses changed")
    if not np.array_equal(app["source_indices"], h["source_indices"]):
        raise ValueError("Application reference source changed")
    if max(errors.values()) > cfg["comparison_tolerance"]:
        raise ValueError("Condition is attached to another vector event")
    tau = np.array(
        [
            M.templates(c["vectors"][c["folds"] == b], c["conditions"][c["folds"] == b])
            for b in (0, 1)
        ]
    )
    files = [
        rp / "observations.npz",
        rp / "operator_bundle.npz",
        rp / "analysis.npz",
        rp / "manifest.json",
        cp / "observations.npz",
        cp / "manifest.json",
        ap / "observations.npz",
        ap / "manifest.json",
        ROOT / cfg["native_parent"] / "analysis.npz",
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        **h,
        source_proportions=c["conditions"].mean(2),
        templates_by_fold=tau,
        panel_conditions=app["conditions"],
    )
    counts = [
        [
            [int(np.sum(c["conditions"][c["folds"] == b, v] == z)) for z in (0, 1)]
            for v in (0, 1)
        ]
        for b in (0, 1)
    ]
    quality = {
        "source_authors": 386,
        "source_events": 9264,
        "panel_authors": 24,
        "panel_events": 576,
        "input_join_errors": errors,
        "template_event_counts_by_fold_window_condition": counts,
        "single_condition_source_windows_retained": int(
            np.sum((c["conditions"].mean(2) == 0) | (c["conditions"].mean(2) == 1))
        ),
        "templates_label_free_source_only": True,
        "historical_encoder_overlap_N_selection_retained": True,
        "new_neural_API_private_train_application_labels": 0,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(quality))


def build(out):
    """Compile both declared references before inspecting old native answers."""
    R.check(out)
    z, start = R.prepared(out), time.perf_counter()
    bs = []
    for reference in (0, 1):
        row = []
        for bank in (0, 1):
            ix = z["folds"] == bank
            row.append(
                M.compile_bank(
                    z["source_means"][ix],
                    z["source_proportions"][ix],
                    z["codes"][ix],
                    z["templates_by_fold"][bank ^ reference],
                )
            )
        bs.append(row)
    R.archive(
        out / "operator_bundle.npz",
        **{
            f"b{r}{b}_{k}": v
            for r in (0, 1)
            for b in (0, 1)
            for k, v in bs[r][b].items()
        },
    )
    receipt = {
        "references": 2,
        "source_accounts": list(M.ACCOUNTS),
        "characters": 31,
        "source_fit_only": True,
        "native_answers_read": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "build_summary.json", receipt)
    R.seal(out, "built_seal.json", ("operator_bundle.npz", "build_summary.json"))
    print(json.dumps(receipt))


def banks(out):
    """Load frozen dual components without a refit."""
    z = H.arrays(out / "operator_bundle.npz")
    return [
        [{k[4:]: v for k, v in z.items() if k.startswith(f"b{r}{b}_")} for b in (0, 1)]
        for r in (0, 1)
    ]


def panel_parts(z, tau):
    """Recompute BOTH choice and expression when actual edit slots change."""
    x = H.panel_frames(z)
    p = np.empty((24, 2, 5))
    for u, v in np.ndindex(24, 2):
        s, c, ix = z["selected"][u, v], z["control"][u, v], np.arange(12)
        groups = (ix, ix[~np.isin(ix, s)], ix[~np.isin(ix, c)], s, c)
        p[u, v] = [z["panel_conditions"][u, v, group].mean() for group in groups]
    parts = M.parts(x.transpose(0, 2, 1, 3), p.transpose(0, 2, 1), tau)
    return parts.transpose(0, 2, 1, 3, 4)


def aggregate(raw):
    """Append full sums while retaining every oriented component."""
    a = np.concatenate((raw, raw.sum(axis=-3, keepdims=True)), axis=-3)
    return np.concatenate((a, a.sum(axis=-2, keepdims=True)), axis=-2)


def analyze(out):
    """Discriminate composition, expression and mixed relations, not select r."""
    cfg = R.check(out, ("built_seal.json",))["config"]
    start, z, bs = time.perf_counter(), R.prepared(out), banks(out)
    parent = H.arrays(ROOT / cfg["relation_parent"] / "analysis.npz")
    dots = np.array([M.contractions(*bs[r]) for r in (0, 1)])
    refs = np.empty((2, 199, 4, 4, 31))
    for r in (0, 1):
        left, right = bs[r]
        ga = np.array(
            [[left["a"][:, i] @ right["a"][:, j].T for j in (0, 1)] for i in (0, 1)]
        )
        gb = np.array(
            [[left["b"][:, i] @ right["b"][:, j].T for j in (0, 1)] for i in (0, 1)]
        )
        for k, (p0, p1) in enumerate(
            zip(parent["permutation0"], parent["permutation1"], strict=True)
        ):
            if time.perf_counter() - start > cfg["max_analyze_seconds"]:
                R.write(
                    out / "analysis_failure.json",
                    {
                        "reason": "fixed_CPU_cap",
                        "reference": r,
                        "completed_pairings": k,
                    },
                )
                raise RuntimeError("Finite cap reached;no automatic rerun")
            refs[r, k] = M.gram_contractions(
                ga, gb[..., p0, :][..., p1], left["w"], right["w"]
            )
    source_rows = [
        {
            "template_reference": cfg["template_references"][r],
            "account0": M.ACCOUNTS[i],
            "account1": M.ACCOUNTS[j],
            "character": m + 1,
            "signed_contraction": float(dots[r, i, j, m]),
        }
        for r, i, j, m in np.ndindex(2, 4, 4, 31)
    ]
    R.table(out / "source_accounts.csv", source_rows)
    rows, ranks = [], np.empty((2, 5, 31))
    for r, i, m in np.ndindex(2, 5, 31):
        value = dots[r, i, i, m] if i < 4 else dots[r, :, :, m].sum()
        reference = refs[r, :, i, i, m] if i < 4 else refs[r, :, :, :, m].sum((1, 2))
        ranks[r, i, m] = (1 + np.sum(reference >= value)) / 200
        rows.append(
            {
                "template_reference": cfg["template_references"][r],
                "account": NAMES[i],
                "character": m + 1,
                "signed_contraction": float(value),
                "descriptive_upper_rank": float(ranks[r, i, m]),
            }
        )
    R.table(out / "component_references.csv", rows)
    scores = np.empty((2, 2, 2, 24, 2, 5, 4, 4, 31))
    stranger = np.roll(np.arange(24), -1)
    for r, bank, companion in np.ndindex(2, 2, 2):
        pp = panel_parts(z, bs[r][bank]["templates"])
        for view in (0, 1):
            own = pp[:, view]
            other = pp[np.arange(24) if companion == 0 else stranger, 1 - view, 0]
            other = np.broadcast_to(other[:, None], own.shape)
            args = (own, other) if view == 0 else (other, own)
            scores[r, bank, companion, :, view] = M.read(bs[r][bank], *args)
    actions = np.stack(
        (
            scores[:, :, :, :, :, 1] - scores[:, :, :, :, :, 2],
            scores[:, :, :, :, :, 3] - scores[:, :, :, :, :, 4],
        ),
        axis=5,
    )
    aa = aggregate(actions)
    native = H.arrays(ROOT / cfg["native_parent"] / "analysis.npz")["native_actions"]
    contacts = []
    for r, bank, companion, op, unit, i, j in np.ndindex(2, 2, 2, 2, 3, 5, 5):
        x = aa[r, bank, companion, :, :, op, i, j][:, :, M.P.TARGET_MODES].mean(1)
        y = native[:, :, op, :, unit].mean(1)
        cov, corr = N.covariances(x, y), N.correlations(x, y)
        for target, reader in np.ndindex(5, 5):
            contacts.append(
                {
                    "template_reference": cfg["template_references"][r],
                    "bank": bank,
                    "source_account": NAMES[i],
                    "observation_account": NAMES[j],
                    "companion": "own" if companion == 0 else "cyclic_stranger",
                    "operation": "deletion" if op == 0 else "isolated",
                    "unit": ("score", "logodds", "geometric")[unit],
                    "source_target": A.TARGETS[target],
                    "reader_target": A.TARGETS[reader],
                    "covariance": float(cov[target, reader]),
                    "r": H.number(corr[target, reader]),
                }
            )
    R.table(out / "native_contacts.csv", contacts)
    errors = {
        "parent_source_total": float(
            np.max(np.abs(dots.sum((1, 2)) - parent["contractions"][2, 2, 0]))
        ),
        "parent_reference_total": float(
            np.max(np.abs(refs.sum((2, 3)) - parent["references"]))
        ),
        "parent_cold_total": float(
            np.max(np.abs(scores.sum((6, 7)) - parent["scores"][:, 2, 0]))
        ),
        "parent_action_total": float(
            np.max(np.abs(actions.sum((6, 7)) - parent["actions"][:, 2, 0]))
        ),
    }
    R.archive(
        out / "analysis.npz",
        contractions=dots,
        references=refs,
        scores=scores,
        actions=actions,
        ranks=ranks,
    )
    diag = [
        row
        for row in contacts
        if row["companion"] == "own"
        and row["operation"] == "deletion"
        and row["unit"] == "score"
        and row["observation_account"] == "all"
        and row["source_target"] == row["reader_target"]
    ]
    summary = {
        "primary_all5_source_accounts": M.target_accounts(dots).tolist(),
        "per_target_diagonal_contractions": [
            {
                name: dots[r, :, :, mode].diagonal().tolist()
                for name, mode in zip(A.TARGETS, M.P.TARGET_MODES, strict=True)
            }
            for r in (0, 1)
        ],
        "per_target_diagonal_upper_ranks": ranks[:, :4][..., M.P.TARGET_MODES].tolist(),
        "original_pair_total_all5": dots.sum((1, 2))[:, M.P.TARGET_MODES]
        .sum(1)
        .tolist(),
        "inherited_total_upper_rank": R.read(
            ROOT / cfg["relation_parent"] / "summary.json"
        )["primary_pairing_reference_upper_rank"],
        "exact_preservation_errors": errors,
        "native_own_source_account_diagonals": diag,
        "template_difference_norms": [
            [
                float(np.linalg.norm(bank["templates"][v, 1] - bank["templates"][v, 0]))
                for v in (0, 1)
            ]
            for bank in bs[0]
        ],
        "elapsed_seconds": time.perf_counter() - start,
        "new_neural_API_private_training": 0,
        "psychological_prediction_accuracy": False,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "source_accounts.csv",
            "component_references.csv",
            "native_contacts.csv",
            "summary.json",
        ),
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k != "native_own_source_account_diagonals"
            }
        )
    )


def verify(out):
    """Replay template/event and primal/dual accounts once, without refitting."""
    cfg = R.check(out, ("built_seal.json", "analyzed_seal.json"))["config"]
    start, z, bs = time.perf_counter(), R.prepared(out), banks(out)
    results = H.arrays(out / "analysis.npz")
    raw = R.prepared(ROOT / cfg["condition_parent"])
    error, literal_checks, field_checks = 0.0, 0, 0
    for bank, view, condition in np.ndindex(2, 2, 2):
        ix = raw["folds"] == bank
        events = raw["vectors"][ix, view][raw["conditions"][ix, view] == condition]
        expected = sum(events) / len(events)
        error = max(
            error,
            float(
                np.max(np.abs(expected - z["templates_by_fold"][bank, view, condition]))
            ),
        )
        literal_checks += 1
    columns = [0, 17, 91, 200]
    for r in (0, 1):
        ts = []
        for bank in bs[r]:
            t = np.array(
                [
                    [
                        bank["a"][:, a][:, columns].T
                        @ (w[:, None] * bank["b"][:, b][:, columns])
                        for w in bank["w"]
                    ]
                    for a, b in M.LEGS
                ]
            )
            scalar = bank["w"] @ np.prod(bank["proportion_residuals"], axis=1)
            delta = (bank["templates"][:, 1] - bank["templates"][:, 0])[:, columns]
            expected = scalar[:, None, None] * np.outer(*delta)
            error = max(error, float(np.max(np.abs(expected - t[0]))))
            ts.append(t)
            literal_checks += 31
        literal = np.einsum("amij,bmij->abm", *ts)
        tiny = [
            {**bank, "a": bank["a"][:, :, columns], "b": bank["b"][:, :, columns]}
            for bank in bs[r]
        ]
        error = max(error, float(np.max(np.abs(literal - M.contractions(*tiny)))))
        literal_checks += 16 * 31
        for bank in (0, 1):
            pp = panel_parts(z, bs[r][bank]["templates"])
            for view in (0, 1):
                f = M.field(bs[r][bank], pp[0, 1 - view, 0], view)
                for op, (first, second) in enumerate(((1, 2), (3, 4))):
                    d = pp[0, view, first] - pp[0, view, second]
                    expected = np.array(
                        [
                            f[:, j] @ d[c if view == 0 else e]
                            for j, (c, e) in enumerate(M.LEGS)
                        ]
                    ).transpose(1, 0, 2)
                    error = max(
                        error,
                        float(
                            np.max(
                                np.abs(
                                    expected
                                    - results["actions"][r, bank, 0, 0, view, op]
                                )
                            )
                        ),
                    )
                    field_checks += 16 * 31
    record = {
        "frame_key": "actual-conditional-cold-panel-0",
        "windows": [
            [
                {
                    "event_id": f"{z['author_ids'][0]}:{v}:{j}",
                    "vector": z["panel_vectors"][0, v, j].tolist(),
                    "condition": int(z["panel_conditions"][0, v, j]),
                }
                for j in range(12)
            ]
            for v in (0, 1)
        ],
    }
    for r in (0, 1):
        cold = M.cold(bs[r][0], record)
        expected = results["scores"][r, 0, 0, 0, 0, 0]
        error = max(
            error,
            float(
                np.max(
                    np.abs(
                        np.asarray(cold["source_by_observation_readings"]) - expected
                    )
                )
            ),
        )
        literal_checks += 16 * 31
    errors = R.read(out / "summary.json")["exact_preservation_errors"]
    if max(error, *errors.values()) > cfg["comparison_tolerance"]:
        raise ValueError("Conditional relation accounting failed;no precision rescue")
    R.write(out / "cold_example.json", record)
    receipt = {
        "PASS": True,
        "alternate_max_error": error,
        "literal_checks": literal_checks,
        "finite_field_checks": field_checks,
        "parent_preservation_errors": errors,
        "same_agent_not_independent": True,
        "science_rerun": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verified_seal.json", ("cold_example.json", "verification.json"))
    print(json.dumps(receipt))


def main():
    """Execute one write-once stage, or an actual independent-process cold read."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "build", "analyze", "verify", "cold"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "cold":
        R.check(
            args.out, ("built_seal.json", "analyzed_seal.json", "verified_seal.json")
        )
        bs = banks(args.out)
        record = R.read(args.out / "cold_example.json")
        print(
            json.dumps(
                {
                    "frame_key": record["frame_key"],
                    "references": [
                        {
                            "reference": r,
                            "five_target_pair_readings": np.asarray(
                                M.cold(bs[r][0], record)["total_pair_readings"]
                            )[M.P.TARGET_MODES].tolist(),
                        }
                        for r in (0, 1)
                    ],
                    "input_labels_or_native_answers": False,
                }
            )
        )
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
