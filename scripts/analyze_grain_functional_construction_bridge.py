#!/usr/bin/env python3
"""Freeze and analyze S1 natural-response connections with zero neural calls."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.construction_bridge import (
    author_split,
    bootstrap_quality,
    candidate_predictions,
    correlation,
    covariance_account,
    covariance_matrix,
    fit_reference,
    response_account,
    two_order_allocation,
    validate_arrays,
)

BASE = ROOT / "results/grain_vector_mechanisms"
QUERIES = ("TF_cont", "neuroticism", "confirmation_seeking")
SALT = "grain-functional-construction-bridge-v1-20260908"


def read(path):
    """Read a declared local JSON artifact, not a dataset search."""
    return json.loads(path.read_text())


def write(path, value):
    """Write strict JSON so nonfinite results cannot silently enter reports."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def sha(path):
    """Digest a file incrementally without printing any source contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(args):
    """Freeze membership before statistics and preserve complete scalar/vector inputs."""
    from scripts.analyze_grain_multiquery_sensitivity import load_panel, response_rows

    source, out = args.source.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()) or not source.is_relative_to(
        BASE.resolve()
    ):
        raise ValueError("All private data must remain inside the ignored result root")
    status = read(source / "run_status.json")
    if status["status"] != "COMPLETE" or status["completed_authors"] != 24:
        raise ValueError("Complete original 24-author S1 panel required")
    manifest, records = load_panel(source)
    authors = sorted({r[0]["author_id"] for r in records})
    if (
        len(authors) != 24
        or len(records) != 48
        or tuple(manifest["queries"]) != QUERIES
    ):
        raise ValueError("Do not silently drop authors or change query identities")
    ref = author_split(authors, SALT)
    out.mkdir(parents=True, exist_ok=False)
    split = {
        "salt": SALT,
        "authors": authors,
        "reference_mask": ref.tolist(),
        "reference_count": int(ref.sum()),
        "evaluation_count": int((~ref).sum()),
        "scope": "fixed within-development-panel author holdout; not a new lockbox",
    }
    write(out / "split_manifest.json", split)
    by_key = {(r[0]["author_id"], r[0]["view"]): r for r in records}
    if len(by_key) != 48 or set(by_key) != {(u, v) for u in authors for v in (0, 1)}:
        raise ValueError("Exactly one original packet per author/view required")
    gradients = np.empty((24, 2, 3, 1536))
    displacement = np.empty((24, 2, 2, 3, 1536))
    natural, patch = np.empty((24, 2, 2, 3)), np.empty((24, 2, 2, 3))
    joint_d, joint_y, joint_ids = [], [], []
    groups, alphabet = [], {}
    for a, uid in enumerate(authors):
        groups.append(by_key[(uid, 0)][0]["source_group"])
        for v in range(2):
            row, packet, vectors = by_key[(uid, v)]
            if set(row["e1"]) & set(row["e2"]):
                raise ValueError("S1 public edits must be disjoint")
            jd, jy = [], []
            for q, query in enumerate(QUERIES):
                p = packet["queries"][query]
                if query in alphabet and alphabet[query] != p["alphabet"]:
                    raise ValueError("Candidate direction changed between packets")
                alphabet[query] = p["alphabet"]
                scores = {
                    name: value["conditional_score"]
                    for name, value in p["readouts"].items()
                }
                h = np.asarray(vectors[f"{query}__full"], float)
                gradients[a, v, q] = vectors[f"{query}__gradient"]
                for e, edit in enumerate(("e1", "e2")):
                    displacement[a, v, e, q] = vectors[f"{query}__{edit}"] - h
                    natural[a, v, e, q] = scores[edit] - scores["full"]
                    patch[a, v, e, q] = scores["patch_" + edit] - scores["full"]
                if row["joint"]:
                    jd.append(np.asarray(vectors[f"{query}__joint"], float) - h)
                    jy.append(scores["joint"] - scores["full"])
            if row["joint"]:
                joint_d.append(jd)
                joint_y.append(jy)
                joint_ids.append((a, v))
    validate_arrays(gradients, displacement, natural, patch)
    source_rows, _joint, _attribution = response_rows(records)
    source_rows = source_rows.set_index(["author_id", "view", "edit", "query"])
    replay = 0.0
    for a, uid in enumerate(authors):
        for v, e, q in itertools.product(range(2), range(2), range(3)):
            original = source_rows.loc[(uid, v, f"e{e + 1}", QUERIES[q])]
            replay = max(
                replay,
                abs(natural[a, v, e, q] - original.natural),
                abs(patch[a, v, e, q] - original.patch),
            )
    if replay > 1e-12:
        raise ValueError("Original S1 scalar reconstruction failed")
    np.savez_compressed(
        out / "private_inputs.npz",
        gradients=gradients,
        displacement=displacement,
        natural=natural,
        patch=patch,
        joint_d=np.asarray(joint_d),
        joint_y=np.asarray(joint_y),
        joint_ids=np.asarray(joint_ids, int),
    )
    write(
        out / "private_input_metadata.json",
        {"source_groups": groups, "query_alphabets": alphabet},
    )
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/construction_bridge.py",
        ROOT / "docs/GRAIN_FUNCTIONAL_CONSTRUCTION_BRIDGE_PLAN_V1.md",
    ]
    inputs = [
        source / "manifest.json",
        source / "private_material.json",
        source / "run_status.json",
    ]
    inputs += sorted((source / "private_packets").glob("panel_*.json"))
    inputs += sorted((source / "private_packets").glob("panel_*.npz"))
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_FUNCTIONAL_CONSTRUCTION_BRIDGE_V1",
            "source": str(source),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
            "split_sha256": sha(out / "split_manifest.json"),
            "input_sha256": sha(out / "private_inputs.npz"),
            "queries": list(QUERIES),
            "gradient_kind": "S1 endpoint, not integrated",
            "site": "block16 final query residual",
            "bootstrap_draws": args.bootstrap,
            "bootstrap_seed": 20260908,
            "max_seconds": 1200,
            "neural_calls": 0,
            "new_gradients": 0,
            "label_joins": 0,
            "source_scalar_replay_error": replay,
            "who_status": "deferred: S1 encoder geometry uses the same edited text; no independent geometry bound",
            "s21_status": "separate historical pairing/site comparison; never concatenate with S1",
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "authors": 24, "neural_calls": 0},
    )
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "authors": 24,
                "reference": 12,
                "evaluation": 12,
                "joint": len(joint_ids),
            }
        )
    )


def interval(values):
    """Return a finite percentile interval over author bootstrap replicates."""
    return np.quantile(values, [0.025, 0.975], axis=0)


def run(args):
    """Analyze frozen inputs; all fitting is reference-only and all results are retained."""
    out = args.output_dir.resolve()
    m, split = read(out / "manifest.json"), read(out / "split_manifest.json")
    if read(out / "run_status.json")["status"] != "PREPARED":
        raise ValueError(
            "Use a new directory rather than overwrite completed observations"
        )
    if (
        sha(out / "split_manifest.json") != m["split_sha256"]
        or sha(out / "private_inputs.npz") != m["input_sha256"]
    ):
        raise ValueError("Frozen input or membership changed")
    for name, digest in m["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen implementation changed: {name}")
    start = time.monotonic()
    with np.load(out / "private_inputs.npz", allow_pickle=False) as z:
        g, d, y, patch = (
            z[k] for k in ("gradients", "displacement", "natural", "patch")
        )
        jd, jy, ji = (z[k] for k in ("joint_d", "joint_y", "joint_ids"))
    ref = np.asarray(split["reference_mask"], bool)
    held = ~ref
    authors = np.asarray(split["authors"])
    rg, scale, mean = fit_reference(g, y, ref)
    np.savez_compressed(
        out / "private_reference.npz", gradient=rg, scale=scale, cell_mean=mean
    )
    local = np.einsum("avqd,aveqd->aveq", g, d)
    (
        metrics,
        predictions,
        allocations,
        components,
        covariance_rows,
        covariance_errors,
    ) = [], [], [], [], [], []
    own_raw = candidate_predictions(rg, d, mean)[0]["private_evidence_private_reader"]
    account = response_account(y, patch, local, own_raw)
    max_response_error = float(np.max(np.abs(sum(account.values()) - y)))
    max_allocation_error, max_covariance_error = 0.0, 0.0
    draws = m["bootstrap_draws"]
    rng = np.random.default_rng(m["bootstrap_seed"])
    bootstrap = rng.integers(12, size=(draws, 12))
    for units, unit in (("raw", np.ones(3)), ("reference_sd", scale)):
        models, crossed = candidate_predictions(rg, d, mean, unit)
        target, pm, il = y / unit, patch / unit, local / unit
        for model, pred in models.items():
            for a, v, e, q in itertools.product(
                np.flatnonzero(held), range(2), range(2), range(3)
            ):
                predictions.append(
                    {
                        "author_id": authors[a],
                        "view": v,
                        "edit": f"e{e + 1}",
                        "query": QUERIES[q],
                        "units": units,
                        "model": model,
                        "prediction": pred[a, v, e, q],
                    }
                )
            for target_name, actual in (("natural", target), ("patch", pm)):
                for q, query in enumerate(QUERIES):
                    stats = bootstrap_quality(
                        actual[held, ..., q], pred[held, ..., q], draws
                    )
                    metrics.append(
                        {
                            "units": units,
                            "model": model,
                            "target": target_name,
                            "query": query,
                            **stats,
                        }
                    )
        for model, pred in (("observed_patch", pm), ("own_endpoint_linear", il)):
            for q, query in enumerate(QUERIES):
                metrics.append(
                    {
                        "units": units,
                        "model": model,
                        "target": "natural",
                        "query": query,
                        **bootstrap_quality(
                            target[held, ..., q], pred[held, ..., q], draws
                        ),
                    }
                )
        for q, r in itertools.combinations(range(3), 2):
            allocated = two_order_allocation(crossed, q, r)
            max_allocation_error = max(
                max_allocation_error,
                float(
                    np.max(
                        np.abs(
                            allocated["readout"]
                            + allocated["evidence"]
                            - allocated["difference"]
                        )
                    )
                ),
            )
            for a, v, e in itertools.product(np.flatnonzero(held), range(2), range(2)):
                allocations.append(
                    {
                        "author_id": authors[a],
                        "view": v,
                        "edit": f"e{e + 1}",
                        "left": QUERIES[q],
                        "right": QUERIES[r],
                        "units": units,
                        **{
                            name: float(value[a, v, e])
                            for name, value in allocated.items()
                        },
                    }
                )
        a = models["private_evidence_private_reader"]
        for centered in (True, False):
            mode = "within_edit_view" if centered else "pooled"
            actual_account = covariance_account(target[held], a[held], centered)
            reconstructed = sum(
                value for name, value in actual_account.items() if name != "observed"
            )
            max_covariance_error = max(
                max_covariance_error,
                float(np.max(np.abs(reconstructed - actual_account["observed"]))),
            )
            for name, value in actual_account.items():
                samples = np.asarray(
                    [
                        covariance_account(target[held][take], a[held][take], centered)[
                            name
                        ]
                        for take in bootstrap
                    ]
                )
                lo, hi = interval(samples)
                for q, r in itertools.product(range(3), repeat=2):
                    covariance_rows.append(
                        {
                            "units": units,
                            "centering": mode,
                            "term": name,
                            "left": QUERIES[q],
                            "right": QUERIES[r],
                            "value": value[q, r],
                            "lower": lo[q, r],
                            "upper": hi[q, r],
                            "role": "observed accounting, not fitted prediction",
                        }
                    )
            observed = actual_account["observed"]
            for model, pred in models.items():
                for strength in (
                    "conditional_on_held_input",
                    "frozen_reference_distribution",
                ):
                    ids = held if strength == "conditional_on_held_input" else ref
                    predicted_cov = covariance_matrix(
                        pred[ids], centered_within_cell=centered
                    )
                    for q, r in itertools.product(range(3), repeat=2):
                        covariance_rows.append(
                            {
                                "units": units,
                                "centering": mode,
                                "term": model,
                                "left": QUERIES[q],
                                "right": QUERIES[r],
                                "value": predicted_cov[q, r],
                                "lower": None,
                                "upper": None,
                                "role": strength,
                            }
                        )
                    mask = ~np.eye(3, dtype=bool)
                    numerator = float(np.linalg.norm(predicted_cov - observed))
                    denominator = float(np.linalg.norm(observed))
                    samples = []
                    for take in bootstrap:
                        obs = covariance_matrix(
                            target[held][take], centered_within_cell=centered
                        )
                        pc = (
                            covariance_matrix(
                                pred[held][take], centered_within_cell=centered
                            )
                            if strength == "conditional_on_held_input"
                            else predicted_cov
                        )
                        samples.append(np.linalg.norm(pc - obs))
                    lo, hi = interval(samples)
                    covariance_errors.append(
                        {
                            "units": units,
                            "centering": mode,
                            "model": model,
                            "strength": strength,
                            "frobenius_error": numerator,
                            "error_lower": float(lo),
                            "error_upper": float(hi),
                            "relative_error": numerator / denominator
                            if denominator > 1e-18
                            else None,
                            "off_diagonal_mse": float(
                                np.mean((predicted_cov[mask] - observed[mask]) ** 2)
                            ),
                        }
                    )
        if time.monotonic() - start > m["max_seconds"]:
            raise TimeoutError("Bounded cache analysis exceeded the frozen time budget")
    for a, v, e, q in itertools.product(range(24), range(2), range(2), range(3)):
        components.append(
            {
                "author_id": authors[a],
                "role": "reference" if ref[a] else "evaluation",
                "view": v,
                "edit": f"e{e + 1}",
                "query": QUERIES[q],
                "natural": y[a, v, e, q],
                "patch": patch[a, v, e, q],
                "individual_linear": local[a, v, e, q],
                **{name: value[a, v, e, q] for name, value in account.items()},
            }
        )
    energy = []
    for q, query in enumerate(QUERIES):
        for name, value in account.items():
            v = value[held, ..., q]
            energy.append(
                {
                    "query": query,
                    "component": name,
                    "mean": float(v.mean()),
                    "rms": float(np.sqrt(np.mean(v**2))),
                    "normalized_rms": float(np.sqrt(np.mean(v**2)) / scale[q]),
                    "r_with_natural": correlation(v, y[held, ..., q]),
                }
            )
    joint = []
    for j, (a, v) in enumerate(ji):
        for q, query in enumerate(QUERIES):
            natural_i = jy[j, q] - y[a, v, 0, q] - y[a, v, 1, q]
            pred_i = rg[q] @ (jd[j, q] - d[a, v, 0, q] - d[a, v, 1, q])
            joint.append(
                {
                    "author_id": authors[a],
                    "role": "reference" if ref[a] else "evaluation",
                    "query": query,
                    "natural_interaction": natural_i,
                    "predicted_interaction": pred_i,
                    "actual_joint_patch": None,
                }
            )
    tables = {
        "metrics.csv": metrics,
        "cross_query_predictions.csv": predictions,
        "cross_query_allocations.csv": allocations,
        "response_accounting.csv": components,
        "covariance_accounting.csv": covariance_rows,
        "covariance_prediction_errors.csv": covariance_errors,
        "component_summary.csv": energy,
        "joint_edit_accounting.csv": joint,
    }
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    pd.DataFrame({"query": QUERIES, "reference_natural_sd": scale}).to_csv(
        out / "reference_units.csv", index=False
    )
    write(
        out / "mechanism_decision.json",
        {
            "status": "COMPLETE_PENDING_INTERPRETATION",
            "new_neural_calls": 0,
            "interpretation_rule": "Compare natural vs patch connections and query-shared/private models; no accuracy winner selects a new queue",
            "who_status": m["who_status"],
            "s21_status": m["s21_status"],
            "new_source_or_lockbox_labels": False,
        },
    )
    summary = {
        "status": "COMPLETE",
        "authors": 24,
        "reference_authors": 12,
        "evaluation_authors": 12,
        "queries": list(QUERIES),
        "evaluation_cells_per_query": 48,
        "joint_authors": len(ji),
        "joint_evaluation_authors": int(sum(not ref[a] for a, _ in ji)),
        "seconds": time.monotonic() - start,
        "new_neural_calls": 0,
        "new_gradients": 0,
        "label_joins": 0,
        "max_response_account_error": max_response_error,
        "max_allocation_error": max_allocation_error,
        "max_covariance_error": max_covariance_error,
        "reference_scale": scale.tolist(),
        "who_status": m["who_status"],
        "manifest_sha256": sha(out / "manifest.json"),
        "output_hashes": {name: sha(out / name) for name in tables},
    }
    write(out / "summary.json", summary)
    write(out / "run_status.json", summary)
    lines = [
        "# Functional construction bridge: completed numerical packet",
        "",
        "Within-development-panel 12/12 author split; no labels or neural calls.",
        "Natural edits, actual single-site patch and frozen functional transport are distinct.",
        "Full interpretation is written separately after inspecting all arms.",
        "",
        "## Frozen-reader response fidelity",
        "",
    ]
    selected = pd.DataFrame(metrics)
    selected = selected[
        (selected.units == "raw")
        & (selected.model == "private_evidence_private_reader")
    ]
    lines += [
        selected[["query", "target", "r", "mse_over_zero"]].to_string(index=False),
        "",
        "## Source and checks",
        json.dumps(summary, indent=2),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    """Run only one declared phase; do not load model weights or auto-expand data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument(
        "--source", type=Path, default=BASE / "multiquery_sensitivity_s1_r1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "functional_construction_bridge_v1"
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least100 fixed author bootstrap draws")
    with threadpool_limits(limits=2):
        if args.phase == "prepare":
            prepare(args)
        else:
            try:
                run(args)
            except Exception as exc:
                if args.output_dir.exists():
                    write(
                        args.output_dir / "error_log.json",
                        {"type": type(exc).__name__, "message": str(exc)},
                    )
                raise


if __name__ == "__main__":
    main()
