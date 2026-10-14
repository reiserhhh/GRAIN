#!/usr/bin/env python3
"""Build one continuous event-link witness, not a prediction or kernel grid."""

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
from grain_research.construction import math_event_link_witness as M
from grain_research.construction import math_joint_target_atlas as A
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_source_meaning_bridge_hm233 import parent_seal, selected

CONFIG = "configs/grain_event_link_witness_hm247.json"
DEFAULT = ROOT / "results/grain_math_response_v1/event_link_witness_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVENT_LINK_WITNESS_PLAN_HM247.md",
    "grain_research/construction/math_event_link_witness.py",
    "grain_research/construction/math_joint_target_atlas.py",
    "scripts/run_grain_event_link_witness_hm247.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
    "tests/test_grain_event_link_witness.py",
)


def prepare(out):
    """Authenticate vectors, authors and publication addresses before outcomes."""
    cfg = R.read(ROOT / CONFIG)
    parents = {
        k: ROOT / cfg[k]
        for k in ("source_parent", "vector_parent", "address_parent", "event_parent")
    }
    for p in parents.values():
        parent_seal(p, "prepared_seal.json", ("observations.npz",))
    s = selected(
        parents["source_parent"] / "observations.npz",
        ("vectors", "conditions", "folds", "source_indices"),
    )
    v = selected(
        parents["vector_parent"] / "observations.npz",
        ("source_vectors", "panel_vectors", "source_indices", "panel_indices"),
    )
    a = selected(
        parents["address_parent"] / "observations.npz",
        (
            "source_means",
            "panel_vectors",
            "codes",
            "folds",
            "source_indices",
            "author_ids",
        ),
    )
    e = selected(
        parents["event_parent"] / "observations.npz",
        ("times", "original_units", "conditions"),
    )
    identity = selected(ROOT / cfg["identity_material"], ("author_ids",))
    panel_raw = v["panel_vectors"][:, 1]
    if (
        s["vectors"].shape != (386, 2, 12, 1024)
        or v["panel_vectors"].shape != (24, 2, 2, 12, 1024)
        or not np.array_equal(s["vectors"], v["source_vectors"])
        or not np.array_equal(s["vectors"].mean(2), a["source_means"])
        or not np.array_equal(a["panel_vectors"], v["panel_vectors"][:, 0])
        or not np.array_equal(
            identity["author_ids"][v["panel_indices"]], a["author_ids"]
        )
        or any(not np.array_equal(s[k], a[k]) for k in ("folds", "source_indices"))
        or not np.array_equal(s["source_indices"], v["source_indices"])
        or not np.array_equal(np.bincount(s["folds"]), [203, 183])
        or np.intersect1d(v["panel_indices"], s["source_indices"]).size
        or len(set(s["source_indices"].tolist())) != 386
        or len(set(v["panel_indices"].tolist())) != 24
        or len(set(a["author_ids"].tolist())) != 24
    ):
        raise ValueError(
            "Frozen source/application author, view or vector identity changed"
        )
    order = np.argsort(e["original_units"], axis=1)
    ordered_times = np.take_along_axis(e["times"], order, axis=1)
    ordered_conditions = np.take_along_axis(e["conditions"], order, axis=1)
    positions = np.take_along_axis(e["original_units"], order, axis=1)
    ids, panel_ids = s["source_indices"], v["panel_indices"]
    source_times = ordered_times[ids, :24].reshape(386, 2, 12)
    panel_times = ordered_times[panel_ids, :24].reshape(24, 2, 12)
    panel_conditions = ordered_conditions[panel_ids, :24].reshape(24, 2, 12)
    if (
        not np.array_equal(
            ordered_conditions[ids, :24].reshape(386, 2, 12), s["conditions"]
        )
        or np.any(np.diff(source_times.reshape(386, 24), axis=1) <= 0)
        or np.any(np.diff(panel_times.reshape(24, 24), axis=1) <= 0)
        or np.any(positions[np.r_[ids, panel_ids], :24] != np.arange(24))
        or not np.allclose(np.linalg.norm(s["vectors"], axis=-1), 1, rtol=0, atol=2e-12)
        or not np.allclose(np.linalg.norm(panel_raw, axis=-1), 1, rtol=0, atol=2e-12)
    ):
        raise ValueError(
            "Raw-vector units, original event conditions or strict chronology disagree"
        )
    for bank in (0, 1):
        M.target_weights(a["codes"][s["folds"] == bank])
    out.mkdir(parents=True, exist_ok=False)
    inputs = [
        p / name
        for p in parents.values()
        for name in ("observations.npz", "prepared_seal.json")
    ]
    inputs.append(ROOT / cfg["identity_material"])
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
            "source_targets": "inherited historical five-bit source labels; no application labels read",
            "old_panel_reuse": "24 historical authors; not fresh representative holdout; N-enriched panel",
            "previous_turn": "PROGRESS_DECISION:instrument optimization closed unless error prevents mechanism decision",
        },
    )
    R.archive(
        out / "observations.npz",
        source_vectors=s["vectors"],
        source_conditions=s["conditions"],
        codes=a["codes"],
        folds=s["folds"],
        source_indices=ids,
        source_times=source_times,
        panel_vectors=panel_raw,
        panel_conditions=panel_conditions,
        panel_times=panel_times,
        panel_indices=panel_ids,
        author_ids=a["author_ids"],
    )
    with (out / "cold_input.jsonl").open("x") as f:
        for u in range(24):
            author = str(a["author_ids"][u])
            for view in (0, 1):
                record = {
                    "record_id": f"{author}/{view}",
                    "author": author,
                    "regime": "raw_pandora_publication",
                    "events": [
                        {
                            "event_id": f"{author}:{12 * view + t}",
                            "owner": author,
                            "position": 12 * view + t,
                            "timestamp": float(panel_times[u, view, t]),
                            "vector": panel_raw[u, view, t].tolist(),
                            "condition": int(panel_conditions[u, view, t]),
                        }
                        for t in range(12)
                    ],
                }
                M.canonical_window(record, 1024)
                f.write(json.dumps(record, allow_nan=False) + "\n")
    quality = {
        "source_authors": 386,
        "source_banks": [203, 183],
        "source_events": 9264,
        "source_adjacent_edges": 8492,
        "application_authors": 24,
        "application_events": 576,
        "application_adjacent_edges": 528,
        "cold_records": 48,
        "dimensions": 1024,
        "source_application_overlap": 0,
        "source_vector_replay_error": 0,
        "address_parent_variant": "cue-masked, separately authenticated; raw variant used here",
        "cue_raw_RMS": float(np.sqrt(np.mean((a["panel_vectors"] - panel_raw) ** 2))),
        "strict_publication_order": True,
        "edge_independence_assumed": False,
        "raw_prose_application_labels_native_answers_new_encoder_API_private_train": 0,
        "pre_registration_failures": [
            "prepare guard rejected cue/raw mismatch before mkdir/manifest/outcomes; corrected explicit raw binding"
        ],
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_input.jsonl", "input_quality.json"),
    )
    print(json.dumps(quality), flush=True)


def compute(out):
    """Construct all fixed link accounts and inherited full-profile exchanges."""
    cfg = R.check(out)["config"]
    z = R.prepared(out)
    start = time.monotonic()

    def guard():
        if time.monotonic() - start > cfg["max_compute_seconds"]:
            raise TimeoutError("Finite computation cap; no automatic restart")

    g = np.empty((2, 2, 2, 4, 386, 386))
    for v, w in ((0, 0), (0, 1), (1, 1)):
        g[v, w] = M.grams(
            z["source_vectors"][:, v],
            z["source_conditions"][:, v],
            z["source_vectors"][:, w],
            z["source_conditions"][:, w],
            block=cfg["blocks"],
            guard=guard,
        )
        print(
            json.dumps(
                {"completed_views": [v, w], "seconds": time.monotonic() - start}
            ),
            flush=True,
        )
    g[1, 0] = g[0, 1][:, [0, 2, 1, 3]].transpose(0, 1, 3, 2)
    idx = [np.flatnonzero(z["folds"] == b) for b in (0, 1)]
    weights = [M.target_weights(z["codes"][i]) for i in idx]
    cross = np.empty((2, 2, 2, 4, 5, 5))
    rows = []
    for v in (0, 1):
        for w in (0, 1):
            for k, kernel in enumerate(M.KERNELS):
                for component, name in enumerate(M.COMPONENTS):
                    h = g[v, w, k, component][np.ix_(*idx)]
                    cross[v, w, k, component] = weights[0] @ h @ weights[1].T
                    for q in range(5):
                        for r in range(5):
                            rows.append(
                                {
                                    "kernel": kernel,
                                    "source_view": v,
                                    "other_bank_view": w,
                                    "component": name,
                                    "target": cfg["targets"][q],
                                    "other_bank_target": cfg["targets"][r],
                                    "product": float(cross[v, w, k, component, q, r]),
                                }
                            )
    primary = (
        np.diagonal(cross[0, 1, :, 3], axis1=-2, axis2=-1)
        + np.diagonal(cross[1, 0, :, 3], axis1=-2, axis2=-1)
    ) / 2
    h = (g[0, 1, :, 3] + g[1, 0, :, 3])[:, idx[0]][:, :, idx[1]] / 2
    rng = np.random.default_rng(cfg["null_seed"])
    nulls = np.empty((cfg["null_replicates"], 2, 5))
    allocations = []
    for trial in range(cfg["null_replicates"]):
        guard()
        codes = A.permute_profiles(z["codes"], z["folds"], rng)
        allocations.append(codes)
        wa, wb = [M.target_weights(codes[i]) for i in idx]
        nulls[trial] = np.einsum("qi,kij,qj->kq", wa, h, wb, optimize=True)
    common = np.diagonal(g[0, 1, :, 3], axis1=-2, axis2=-1).mean(-1) - g[
        0, 1, :, 3
    ].mean((-2, -1))
    summaries = []
    for k, kernel in enumerate(M.KERNELS):
        values, references = primary[k], nulls[:, k]
        summaries.append(
            {
                "kernel": kernel,
                "same_target_products_ENACO": values.tolist(),
                "positive_targets": int(np.sum(values > 0)),
                "primary_all_five_sum": float(values.sum()),
                "exploratory_upper_rank_all_five": float(
                    (1 + np.sum(references.sum(-1) >= values.sum())) / 200
                ),
                "exploratory_upper_ranks_ENACO": (
                    (1 + (references >= values).sum(0)) / 200
                ).tolist(),
                "null_sum_mean": float(references.sum(-1).mean()),
                "null_sum_q025_q975": np.quantile(
                    references.sum(-1), [0.025, 0.975]
                ).tolist(),
                "common_same_author_minus_product_recurrence": float(common[k]),
                "rank_is_confirmatory_p": False,
            }
        )
    R.archive(
        out / "source_accounts.npz",
        grams=g,
        cross_products=cross,
        primary=primary,
        nulls=nulls,
        allocated_codes=np.asarray(allocations),
        common=common,
    )
    R.archive(
        out / "cold_bundle.npz",
        source_vectors=z["source_vectors"],
        source_conditions=z["source_conditions"],
        codes=z["codes"],
        folds=z["folds"],
    )
    R.table(out / "source_cross_accounts.csv", rows)
    result = {
        "kernel_results": summaries,
        "seconds": time.monotonic() - start,
        "new_model_API_private_training_application_label_answer_fit": 0,
        "fixed_observations_not_method_selection": True,
        "original_scope_complete": False,
    }
    R.write(out / "source_summary.json", result)
    R.seal(
        out,
        "computed_seal.json",
        (
            "source_accounts.npz",
            "cold_bundle.npz",
            "source_cross_accounts.csv",
            "source_summary.json",
        ),
    )
    print(json.dumps(result), flush=True)


def cold_runtime(bundle, inputs, output):
    """Read a frozen source and one addressed author window at a time."""
    z = selected(bundle, ("source_vectors", "source_conditions", "codes", "folds"))
    indices = [np.flatnonzero(z["folds"] == b) for b in (0, 1)]
    weights = [M.target_weights(z["codes"][i]) for i in indices]
    with inputs.open() as f, output.open("x") as out:
        for line in f:
            record = json.loads(line)
            M.canonical_window(record, 1024)
            first = min(e["position"] for e in record["events"])
            if first not in (0, 12):
                raise ValueError("Unregistered source/application view")
            view = first // 12
            banks = [
                M.cold_read(
                    z["source_vectors"][i, view],
                    z["source_conditions"][i, view],
                    w,
                    record,
                )
                for i, w in zip(indices, weights, strict=True)
            ]
            out.write(
                json.dumps(
                    {
                        "record_id": record["record_id"],
                        "source_view": view,
                        "banks": banks,
                    },
                    allow_nan=False,
                )
                + "\n"
            )


def cold(out):
    """Execute the cold scorer separately, without application labels or answers."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    start = time.monotonic()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "cold-runtime",
        "--bundle",
        str(out / "cold_bundle.npz"),
        "--inputs",
        str(out / "cold_input.jsonl"),
        "--output",
        str(out / "cold_output.jsonl"),
    ]
    p = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=cfg["max_cold_seconds"],
    )
    values, rows, changes = [], [], []
    for line in (out / "cold_output.jsonl").read_text().splitlines():
        record = json.loads(line)
        changes.append(record["banks"][0]["changed_after_slots"])
        for bank, result in enumerate(record["banks"]):
            a = np.asarray(result["attested_accounts"])
            b = np.asarray(result["relinked_audit_accounts"])
            values.append(np.stack((a, b)))
            for arm, data in (("attested", a), ("relinked_audit", b)):
                for k, kernel in enumerate(M.KERNELS):
                    for c, name in enumerate(M.COMPONENTS):
                        for q, target in enumerate(cfg["targets"]):
                            rows.append(
                                {
                                    "record_id": record["record_id"],
                                    "source_bank": bank,
                                    "kernel": kernel,
                                    "component": name,
                                    "target": target,
                                    "arm": arm,
                                    "reading": float(data[k, c, q]),
                                }
                            )
    values = np.asarray(values).reshape(48, 2, 2, 2, 4, 5)
    delta = values[:, :, 1] - values[:, :, 0]
    result = {
        "records": 48,
        "accounts": len(rows),
        "seconds": time.monotonic() - start,
        "changed_after_slots": changes,
        "PP_relink_error": float(abs(delta[:, :, :, 0]).max()),
        "relink_RR_delta_RMS_by_kernel_ENACO": np.sqrt(
            np.mean(delta[:, :, :, 3] ** 2, axis=(0, 1))
        ).tolist(),
        "relink_all_account_delta_RMS_by_kernel_ENACO": np.sqrt(
            np.mean(delta.sum(3) ** 2, axis=(0, 1))
        ).tolist(),
        "new_human_text_or_application_accuracy_claim": False,
    }
    R.table(out / "cold_accounts.csv", rows)
    R.archive(out / "cold_accounts.npz", accounts=values)
    R.write(
        out / "cold_receipt.json",
        {
            "command": command,
            "exit_code": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "summary": result,
        },
    )
    R.seal(
        out,
        "cold_seal.json",
        (
            "cold_output.jsonl",
            "cold_accounts.csv",
            "cold_accounts.npz",
            "cold_receipt.json",
        ),
    )
    print(json.dumps(result), flush=True)


def literal_accounts(x, c, y, d):
    """Independently evaluate endpoint product coefficients and Euclidean norms."""
    zx, zy = np.c_[x, np.eye(2)[c]], np.c_[y, np.eye(2)[d]]
    linear = zx @ zy.T
    energy = (
        np.linalg.norm(zx, axis=-1)[:, None]
        + np.linalg.norm(zy, axis=-1)[None]
        - np.linalg.norm(zx[:, None] - zy[None], axis=-1)
    ) / 2
    m, n = len(x) - 1, len(y) - 1
    px, py = np.ones((m, m)) / m**2, np.ones((n, n)) / n**2
    rx, ry = np.eye(m) / m - px, np.eye(n) / n - py
    values = []
    for k in (linear, energy):
        left, right = k[:-1, :-1], k[1:, 1:]
        values.append(
            [
                np.einsum("ij,kl,ik,jl->", a, b, left, right)
                for a, b in ((px, py), (px, ry), (rx, py), (rx, ry))
            ]
        )
    return np.asarray(values)


def verify(out):
    """Replay finite literal accounts, profile exchanges and actual cold addresses."""
    cfg = R.check(out, ("computed_seal.json", "cold_seal.json"))["config"]
    z, a = (
        R.prepared(out),
        selected(
            out / "source_accounts.npz",
            ("grams", "primary", "nulls", "allocated_codes"),
        ),
    )
    errors = np.zeros(2)
    checked = 0
    for u, v, i, j in (
        (0, 0, 0, 0),
        (0, 1, 0, 1),
        (1, 0, 203, 4),
        (1, 1, 10, 10),
        (0, 1, 12, 205),
        (0, 0, 202, 385),
        (1, 0, 385, 203),
        (1, 1, 383, 385),
    ):
        direct = literal_accounts(
            z["source_vectors"][i, u],
            z["source_conditions"][i, u],
            z["source_vectors"][j, v],
            z["source_conditions"][j, v],
        )
        errors = np.maximum(
            errors, np.max(abs(direct - a["grams"][u, v, :, :, i, j]), axis=1)
        )
        checked += direct.size
    idx = [np.flatnonzero(z["folds"] == b) for b in (0, 1)]
    h = (a["grams"][0, 1, :, 3] + a["grams"][1, 0, :, 3])[:, idx[0]][:, :, idx[1]] / 2
    weights = [M.target_weights(z["codes"][i]) for i in idx]
    primary = np.asarray(
        [
            [
                sum(
                    weights[0][q, i] * h[k, i, j] * weights[1][q, j]
                    for i in range(len(idx[0]))
                    for j in range(len(idx[1]))
                )
                for q in range(5)
            ]
            for k in range(2)
        ]
    )
    primary_error = float(abs(primary - a["primary"]).max())
    rng = np.random.default_rng(cfg["null_seed"])
    null_error = 0.0
    for t in range(cfg["null_replicates"]):
        codes = A.permute_profiles(z["codes"], z["folds"], rng)
        if not np.array_equal(codes, a["allocated_codes"][t]):
            raise ValueError("Changed inherited whole-profile exchange")
        w0, w1 = [M.target_weights(codes[i]) for i in idx]
        null_error = max(
            null_error,
            float(
                abs(
                    np.einsum("qi,kij,qj->kq", w0, h, w1, optimize=True) - a["nulls"][t]
                ).max()
            ),
        )
    reverse_error = float(
        abs(
            a["grams"][1, 0] - a["grams"][0, 1][:, [0, 2, 1, 3]].transpose(0, 1, 3, 2)
        ).max()
    )
    cold_values = selected(out / "cold_accounts.npz", ("accounts",))["accounts"]
    pp_error = float(abs(cold_values[:, :, 1, :, 0] - cold_values[:, :, 0, :, 0]).max())
    address_checks = 0
    for n, line in enumerate((out / "cold_input.jsonl").read_text().splitlines()):
        record = json.loads(line)
        x, c, ids = M.canonical_window(record, 1024)
        u, v = divmod(n, 2)
        if not np.array_equal(x, z["panel_vectors"][u, v]) or not np.array_equal(
            c, z["panel_conditions"][u, v]
        ):
            raise ValueError("Cold record differs from registered vector/condition")
        p = M.relink(c[None])[0]
        if not np.array_equal(np.sort(p), np.arange(1, 12)) or not np.array_equal(
            2 * c[:-1] + c[p], 2 * c[:-1] + c[1:]
        ):
            raise ValueError(
                "Audit changed endpoint multisets or condition-pair counts"
            )
        if len(set(ids)) != 12:
            raise ValueError("Duplicate event mistaken for new evidence")
        address_checks += 12
    if (
        errors[0] > cfg["algebra_tolerance"]
        or errors[1] > cfg["literal_distance_tolerance"]
        or max(primary_error, null_error, reverse_error, pp_error)
        > cfg["algebra_tolerance"]
    ):
        raise ArithmeticError("Fixed finite-account verification failed")
    result = {
        "status": "PASS",
        "literal_actual_source_accounts": checked,
        "literal_linear_error": float(errors[0]),
        "literal_distance_error": float(errors[1]),
        "all_primary_products_literal_error": primary_error,
        "null_exchange_replay_error": null_error,
        "reverse_orientation_error": reverse_error,
        "cold_PP_invariance_error": pp_error,
        "cold_event_address_checks": address_checks,
        "same_agent_alternate_not_blind": True,
        "all_cold_accounts_independently_recomputed": False,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)


def main():
    """Run exactly one registered stage, preserving partial/failed primaries."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "stage",
        choices=("prepare", "compute", "cold", "verify", "check", "cold-runtime"),
    )
    p.add_argument("--out", type=Path, default=DEFAULT)
    for flag in ("bundle", "inputs", "output"):
        p.add_argument("--" + flag, type=Path)
    a = p.parse_args()
    if a.stage == "cold-runtime":
        if not all((a.bundle, a.inputs, a.output)):
            p.error("cold-runtime requires bundle/inputs/output")
        cold_runtime(a.bundle, a.inputs, a.output)
    elif a.stage == "check":
        R.check(
            a.out, ("computed_seal.json", "cold_seal.json", "verification_seal.json")
        )
        print("PASS: all scientific seals; original scope OPEN")
    else:
        try:
            {"prepare": prepare, "compute": compute, "cold": cold, "verify": verify}[
                a.stage
            ](a.out)
        except Exception as e:
            if a.out.exists():
                R.write(
                    a.out / f"{a.stage}_failure.json",
                    {"error": repr(e), "automatic_restart": False},
                )
            raise


if __name__ == "__main__":
    main()
