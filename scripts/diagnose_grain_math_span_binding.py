#!/usr/bin/env python3
"""Zero-forward nonlinear-interaction and independent span-packet checks."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_response import write_csv

DEFAULT = ROOT / "results/grain_math_response_v1/span_binding_r1"
OUTPUT = ROOT / "results/grain_math_response_v1/span_binding_diagnostic_r1"


def mixed_difference(full, projection, complement):
    """Compute the finite cross-effect from scalar changes, not vector energies."""
    return np.asarray(full) - np.asarray(projection) - np.asarray(complement)


def analyze(source, out):
    """Reconstruct frame/readout invariants and every available complete arm set."""
    if out.exists():
        raise FileExistsError("preserve prior diagnostic; use a separate output")
    manifest = frozen_files(source)
    spec = manifest["spec"]
    design = read_json(source / "00/design.json")
    patches = read_json(source / "00/patch_design.json")
    pairs = read_json(source / "00/pairs.json")
    basis_meta = read_json(source / "02/basis_manifest.json")
    if (
        digest(source / "02/basis.npz") != basis_meta["array_sha256"]
        or basis_meta["patch_calls_before_freeze"] != 0
    ):
        raise ValueError("frame bytes or chronology changed")
    for name, value in basis_meta["capture_hashes"].items():
        if digest(source / name) != value:
            raise ValueError("source capture changed")
    with np.load(source / "02/basis.npz", allow_pickle=False) as a:
        frames = {k: a[k] for k in a.files}
    captures, bodies, score_error = {}, {}, 0.0
    body_error, body_queries = 0.0, 0
    for row in design:
        path = source / "01/units" / f"unit_{row['unit']:04d}.json"
        meta = read_json(path)
        if (
            meta["address"] != row
            or digest(path.with_suffix(".npz")) != meta["array_sha256"]
        ):
            raise ValueError("capture address/array changed")
        native = meta["native"]
        local = np.asarray(native["candidate_logits"], float)
        probability = np.exp(local - local.max())
        probability /= probability.sum()
        scalar = float(probability @ np.asarray(manifest["alphabet"]["values"]))
        score_error = max(score_error, abs(scalar - native["conditional_score"]))
        if abs(scalar - meta["old_native"]) > spec["baseline_readout_tolerance"]:
            raise ValueError("old baseline replay failed")
        captures[row["doc_id"], row["query"]] = scalar
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            if row["query"] == spec["queries"][0]:
                bodies[row["doc_id"]] = {
                    k: a[k].astype(float) for k in a.files if not k.endswith("_Q")
                }
        if row["query"] != spec["queries"][0]:
            body_queries += 1
            body_error = max(body_error, meta["body_query_error"])
            if (
                meta["body_query_error"]
                > spec["body_query_absolute_tolerance"]
                + spec["body_query_relative_tolerance"] * meta["body_query_scale"]
            ):
                raise ValueError("query causality replay failed")
    source_pairs = [p for p in pairs if p["role"] == "source"]
    frame_rows = []
    for layer in spec["layers"]:
        means = []
        for axis in ("A", "T"):
            key = f"L{layer}_{axis}_property"
            deltas = [
                (bodies[p["donor_doc"]][key] - bodies[p["receiver_doc"]][key]).sum(
                    axis=0
                )
                / len(bodies[p["donor_doc"]][key])
                for p in source_pairs
                if p["axis"] == axis
            ]
            if len(deltas) != 32:
                raise ValueError("source pair count changed")
            means.append(np.add.reduce(deltas) / 32)
        d = np.stack(means, axis=1)
        u, random = frames[f"L{layer}_U"], frames[f"L{layer}_R"]
        if u.shape[1] == 2:
            alternate = np.linalg.qr(d, mode="reduced")[0]
            projector_error = float(
                np.linalg.norm(u @ u.T - alternate @ alternate.T, ord="fro")
            )
        else:
            alternate = np.linalg.pinv(d)
            projector_error = float(np.linalg.norm(u @ u.T - d @ alternate, ord="fro"))
        frame_rows.append(
            {
                "layer": layer,
                "rank": u.shape[1],
                "source_mean_error": float(np.max(np.abs(d - frames[f"L{layer}_D"]))),
                "alternate_projector_error": projector_error,
                "orthogonality_error": float(
                    np.max(np.abs(u.T @ u - np.eye(u.shape[1])))
                ),
                "random_orthogonality_error": float(
                    np.max(np.abs(random.T @ random - np.eye(random.shape[1])))
                ),
            }
        )
    cells, visited = {}, []
    split_error, arithmetic_error, random_norm_error, noop_error = 0.0, 0.0, 0.0, 0.0
    norm_bounds = []
    for row in patches:
        path = source / "03/units" / f"unit_{row['unit']:05d}.json"
        if not path.exists():
            continue
        record = read_json(path)
        if record["address"] != row:
            raise ValueError("patch address changed")
        if (
            abs(record["receiver_native"] - captures[row["receiver_doc"], row["query"]])
            > 1e-12
            or abs(record["donor_native"] - captures[row["donor_doc"], row["query"]])
            > 1e-12
        ):
            raise ValueError("patch baseline unit changed")
        local = np.asarray(record["native"]["candidate_logits"], float)
        probability = np.exp(local - local.max())
        probability /= probability.sum()
        scalar = float(probability @ np.asarray(manifest["alphabet"]["values"]))
        delta = scalar - record["receiver_native"]
        arithmetic_error = max(arithmetic_error, abs(delta - record["patch_delta"]))
        split_error = max(split_error, record["split_error"])
        if row["arm"] == "random_matched":
            random_norm_error = max(
                random_norm_error,
                abs(record["intervention_norm"] - record["projection_norm"]),
            )
        if row["arm"] == "noop":
            noop_error = max(noop_error, abs(delta))
        else:
            cells[
                row["family"],
                row["axis"],
                row["polarity"],
                row["order"],
                row["layer"],
                row["query"],
                row["arm"],
            ] = delta
        name = f"L{row['layer']}_{row['axis']}_{'neutral' if row['arm'] == 'neutral' else 'property'}"
        d = bodies[row["donor_doc"]][name] - bodies[row["receiver_doc"]][name]
        u = frames[f"L{row['layer']}_U"]
        expected = np.linalg.norm((d @ u) @ u.T)
        allowance = record["live_body_error"] * np.sqrt(d.size) + 1e-9
        error = abs(expected - record["projection_norm"])
        if error > allowance:
            raise ValueError(
                "actual intervention norm exceeds live-prefix numeric bound"
            )
        norm_bounds.append(error)
        visited.append(str(path.relative_to(source)))
    if (
        arithmetic_error > 1e-12
        or random_norm_error > 1e-8
        or noop_error > spec["baseline_readout_tolerance"]
        or split_error > 1e-8
    ):
        raise ValueError("independent intervention checks failed")
    roles = {p["family"]: p["role"] for p in pairs}
    interaction_rows, binding_rows = [], []
    grouped = defaultdict(dict)
    for key, value in cells.items():
        grouped[key[:-1]][key[-1]] = value
    for key, arms in grouped.items():
        if not {"full_property", "projection", "complement"}.issubset(arms):
            continue
        interaction_rows.append(
            {
                "family": key[0],
                "role": roles[key[0]],
                "axis": key[1],
                "polarity": key[2],
                "order": key[3],
                "layer": key[4],
                "query": key[5],
                "full_delta": arms["full_property"],
                "projection_delta": arms["projection"],
                "complement_delta": arms["complement"],
                "mixed_difference": float(
                    mixed_difference(
                        arms["full_property"], arms["projection"], arms["complement"]
                    )
                ),
            }
        )
    polarities = defaultdict(dict)
    for row in interaction_rows:
        polarities[
            row["family"], row["axis"], row["order"], row["layer"], row["query"]
        ][row["polarity"]] = row
    for key, rows in polarities.items():
        if set(rows) != {0, 1}:
            continue
        high, low = rows[1], rows[0]
        binding_rows.append(
            {
                "family": key[0],
                "role": high["role"],
                "axis": key[1],
                "order": key[2],
                "layer": key[3],
                "query": key[4],
                "full_binding": high["full_delta"] - low["full_delta"],
                "projection_binding": high["projection_delta"]
                - low["projection_delta"],
                "complement_binding": high["complement_delta"]
                - low["complement_delta"],
                "mixed_binding": high["mixed_difference"] - low["mixed_difference"],
            }
        )
    table = defaultdict(list)
    for row in binding_rows:
        table[
            row["role"], row["axis"], row["order"], row["layer"], row["query"]
        ].append(row)
    means = [
        {
            "role": key[0],
            "axis": key[1],
            "order": key[2],
            "layer": key[3],
            "query": key[4],
            "families": len(rows),
            **{
                f"mean_{field}": float(np.mean([r[field] for r in rows]))
                for field in (
                    "full_binding",
                    "projection_binding",
                    "complement_binding",
                    "mixed_binding",
                )
            },
        }
        for key, rows in table.items()
    ]
    with (source / "04/binding_action_by_family.csv").open() as stream:
        primary = list(csv.DictReader(stream))
    primary_index = {
        (
            int(r["family"]),
            r["axis"],
            int(r["order"]),
            int(r["layer"]),
            r["query"],
            r["arm"],
        ): float(r["binding_action"])
        for r in primary
    }
    summary_error = 0.0
    for row in binding_rows:
        prefix = row["family"], row["axis"], row["order"], row["layer"], row["query"]
        for arm, field in (
            ("full_property", "full_binding"),
            ("projection", "projection_binding"),
            ("complement", "complement_binding"),
        ):
            summary_error = max(
                summary_error, abs(row[field] - primary_index[*prefix, arm])
            )
    if summary_error > 1e-12:
        raise ValueError("primary binding summary changed")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "frame_checks.csv", frame_rows)
    write_csv(out / "finite_interactions.csv", interaction_rows)
    write_csv(out / "binding_decomposition_by_family.csv", binding_rows)
    write_csv(out / "binding_decomposition_summary.csv", means)
    hashes = {
        name: digest(source / name)
        for name in (
            *visited,
            "00/manifest.json",
            "02/basis_manifest.json",
            "02/basis.npz",
            "04/binding_action_by_family.csv",
        )
    }
    write_json(out / "checked_source_hashes.json", hashes)
    result = {
        "status": "COMPLETE_CHECKED"
        if len(visited) == len(patches)
        else "PARTIAL_CHECKED_NO_PROMOTION",
        "new_model_calls": 0,
        "source": str(source),
        "captures_checked": len(design),
        "interventions_checked": len(visited),
        "expected_interventions": len(patches),
        "body_queries_checked": body_queries,
        "body_query_max_error": body_error,
        "native_score_max_error": score_error,
        "intervention_arithmetic_max_error": arithmetic_error,
        "random_norm_max_error": random_norm_error,
        "noop_max_error": noop_error,
        "split_max_error": split_error,
        "primary_binding_max_error": summary_error,
        "projection_norm_numeric_max_error": max(norm_bounds, default=None),
        "complete_finite_interactions": len(interaction_rows),
        "complete_binding_decompositions": len(binding_rows),
        "frame_checks": frame_rows,
        "diagnostic_code_sha256": digest(Path(__file__)),
        "same_agent_alternative_formula_check_not_independent_researcher": True,
        "not_a_preregistered_primary_effect": True,
    }
    write_json(out / "verification.json", result)
    print(result, flush=True)


def main():
    """Preserve diagnostic versions; never collect new observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    analyze(args.source, args.out)


if __name__ == "__main__":
    main()
