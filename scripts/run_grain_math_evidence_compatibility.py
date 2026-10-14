#!/usr/bin/env python3
"""HM67: fixed-direction compatibility, using frozen observations and new worlds."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_evidence_compatibility as M
from grain_research.construction import math_heterogeneous_evidence_cube as H
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_heterogeneous_evidence_cube import check, read_tables

SOURCE = ROOT / "results/grain_math_response_v1/heterogeneous_evidence_cube_r1"
OUT = ROOT / "results/grain_math_response_v1/contextual_evidence_compatibility_r1"
CODE = (
    "docs/GRAIN_CONTEXTUAL_EVIDENCE_COMPATIBILITY_PLAN_V1.md",
    "grain_research/construction/math_evidence_compatibility.py",
    "scripts/run_grain_math_evidence_compatibility.py",
    "tests/test_grain_math_evidence_compatibility.py",
)
STAGES = (
    "score",
    "log_odds",
    "centered_logits",
    "body_endpoint",
    "score_link_control",
    "log_odds_link_control",
)
WORLDS = ("additive", "common_scalar_nonlinear", "selective_interaction", "link_only")
SEEDS = (20261009, 20261010, 20261011)


def write(path, data):
    """Save a local finite checkpoint or result without serializing NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    """Write complete scalar results with explicit blank unidentifiable fields."""
    if not rows:
        raise ValueError("Expected declared nonempty output table")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    """Bind the new protocol and all actual raw historical inputs before analysis."""
    if out.exists():
        raise FileExistsError(
            "Preserve existing packets;no overwrite or automatic retry"
        )
    check(SOURCE)
    if (
        read_json(SOURCE / "verification.json")["status"]
        != "VERIFIED_FINITE_HM61_NOT_GLOBAL"
    ):
        raise ValueError("Verified complete HM61 required")
    names = [
        "00/manifest.json",
        "00/rows.json",
        "01/source_barrier.json",
        "03/coefficients.npz",
        "03/manifest.json",
        "verification.json",
        "run_status.json",
    ]
    names += [
        f"01/units/unit_{i:04d}.{ext}" for i in range(768) for ext in ("json", "npz")
    ]
    sources = {name: digest(SOURCE / name) for name in names}
    (out / "code").mkdir(parents=True)
    code = {}
    for name in CODE:
        destination = out / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        code[name] = digest(destination)
    write(
        out / "manifest.json",
        {
            "packet": "HM67",
            "source": str(SOURCE.relative_to(ROOT)),
            "sources": sources,
            "code": code,
            "stages": STAGES,
            "worlds": WORLDS,
            "seeds": SEEDS,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_neural_calls_training_API_private_labels": [0, 0, 0, 0, 0],
            "historical_result_after_not_blind_replication": True,
            "original_scope_complete": False,
        },
    )
    write(out / "checkpoint.json", {"status": "PREPARED_HM67", "active_processes": []})
    print("HM67 PREPARED:complete historical cube,new direction/compatibility question")


def bound_check(out):
    """Reject any drift in the finite protocol or original raw observations."""
    manifest = read_json(out / "manifest.json")
    if manifest["packet"] != "HM67" or tuple(manifest["stages"]) != STAGES:
        raise ValueError("Packet identity drift")
    for name, sha in manifest["sources"].items():
        if digest(SOURCE / name) != sha:
            raise ValueError("Frozen HM61 changed:" + name)
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "code" / name) != sha:
            raise ValueError("Frozen HM67 code drift:" + name)
    return manifest


def observations():
    """Reconstruct observation stages from raw captures, not just summary tables."""
    grouped, _, prefix = read_tables(SOURCE, read_json(SOURCE / "00/rows.json"))
    if prefix != 0:
        raise ValueError("Question-dependent evidence prefix")
    values = {}
    for family, group in grouped.items():
        z = group["logits"]
        centered = z - z.mean(-1, keepdims=True)
        decoded = H.decode(z)
        controls = H.decode(H.evaluate(H.coefficients(centered), order=1))
        values[family] = {
            **decoded,
            "centered_logits": centered,
            "body_endpoint": group["body"],
            **{unit + "_link_control": value for unit, value in controls.items()},
        }
    return values


def analyze(values, directions, identity, n=6):
    """Measure all directional components and bounds without selecting a predictor."""
    edges, low = M.conditional_edges(values, n)
    edge_rows, scalar_rows = [], []
    total, normal = 0.0, 0.0
    identified = 0
    for j, edge in enumerate(edges):
        split = M.line_split(edge, directions[j])
        row = {
            **identity,
            "evidence": j,
            "identified": split["identified"],
            "total_energy": split["total_energy"],
        }
        if split["identified"]:
            identified += 1
            total += split["total_energy"]
            normal += split["normal_energy"]
            row.update(
                {
                    k: split[k]
                    for k in (
                        "parallel_energy",
                        "normal_energy",
                        "normal_fraction",
                        "negative_gain_count",
                    )
                }
            )
        edge_rows.append(row)
        if edge.shape[1] == 6:
            for context, vector in enumerate(edge):
                for query, value in enumerate(vector):
                    scalar_rows.append(
                        {
                            **identity,
                            "evidence": j,
                            "low_corner": int(low[j, context]),
                            "query": query,
                            "contrast": float(value),
                        }
                    )
    squares = M.square_rows(values, directions, n)
    square_scalar = []
    for row in squares:
        square_scalar.append(
            {
                **identity,
                **{k: v for k, v in row.items() if k not in ("interaction", "corners")},
                "corners": json.dumps(row["corners"]),
            }
        )
    valid = [r for r in squares if r["identified"]]
    summary = {
        **identity,
        "identified_evidence_lines": identified,
        "normal_energy": normal,
        "identified_edge_energy": total,
        "normal_fraction": normal / total if total > 0 else None,
        "interaction_energy": sum(r["interaction_energy"] for r in squares),
        "projection_global_bound": sum(r["projection_bound"] for r in valid) / (n - 1),
        "angle_global_bound": sum(r["angle_bound"] for r in valid) / (n - 1),
        "maximum_compatibility_error": max(r["compatibility_error"] for r in squares),
        "maximum_bound_violation": max(
            [0.0]
            + [
                max(
                    r["projection_bound"] - r["four_edge_normal_energy"],
                    r["angle_bound"] - r["projection_bound"],
                )
                for r in valid
            ]
        ),
    }
    return edge_rows, square_scalar, scalar_rows, summary


def run(out):
    """Execute the full fixed finite analysis once, retaining failures and arrays."""
    bound_check(out)
    if read_json(out / "checkpoint.json")["status"] != "PREPARED_HM67":
        raise RuntimeError("Existing execution is not automatically repeated")
    start = time.monotonic()
    write(out / "checkpoint.json", {"status": "RUNNING_HM67"})
    try:
        tables = observations()
        source_lines = {
            stage: M.conditional_edges(tables[0][stage], 6)[0].mean(1)
            for stage in STAGES
        }
        np.savez_compressed(out / "source_lines.npz", **source_lines)
        write(
            out / "source_freeze.json",
            {
                "file_sha256": digest(out / "source_lines.npz"),
                "historical_application_already_existed": True,
                "new_application_diagnostics_not_used_to_select_lines": True,
            },
        )
        arrays, edge_rows, square_rows, contrasts, summaries = {}, [], [], [], []
        for family, stage in itertools.product((0, 1), STAGES):
            value = tables[family][stage]
            key = f"f{family}_{stage}"
            arrays[key] = value
            own = M.conditional_edges(value, 6)[0].mean(1)
            arrays[key + "_own_lines"] = own
            references = {"own_family": own}
            if family == 1:
                references["source_family0"] = source_lines[stage]
            for reference, directions in references.items():
                identity = {"family": family, "stage": stage, "reference": reference}
                er, sr, cr, summary = analyze(value, directions, identity)
                edge_rows.extend(er)
                square_rows.extend(sr)
                contrasts.extend(cr)
                summaries.append(summary)
        toy_arrays, toy_rows = {}, []
        for world, seed in itertools.product(WORLDS, SEEDS):
            for stage, value in M.synthetic_world(world, seed).items():
                key = f"{world}_{seed}_{stage}"
                toy_arrays[key] = value
                directions = M.conditional_edges(value, 6)[0].mean(1)
                summary = analyze(
                    value, directions, {"world": world, "seed": seed, "stage": stage}
                )[-1]
                toy_rows.append(summary)
        np.savez_compressed(out / "observation_tables.npz", **arrays)
        np.savez_compressed(out / "synthetic_material.npz", **toy_arrays)
        csv_write(out / "evidence_directions.csv", edge_rows)
        csv_write(out / "square_compatibility.csv", square_rows)
        csv_write(out / "conditional_target_contrasts.csv", contrasts)
        csv_write(out / "stage_summary.csv", summaries)
        csv_write(out / "synthetic_mechanisms.csv", toy_rows)
        result = {
            "status": "FINITE_HM67_COMPLETE_NOT_GLOBAL",
            "historical_bodies_readings": [128, 768],
            "families_stages_evidence_pairs_contexts": [2, 6, 6, 15, 16],
            "evidence_rows_square_rows": [len(edge_rows), len(square_rows)],
            "synthetic_worlds_seeds_tables": [4, 3, len(toy_arrays)],
            "new_neural_calls_training_API_private_labels": [0, 0, 0, 0, 0],
            "source_mean_direction_calibrations": 6,
            "CPU_wall_seconds": time.monotonic() - start,
            "stage_summaries": summaries,
            "not_personality_information_or_accuracy": True,
            "original_scope_complete": False,
        }
        write(out / "summary.json", result)
        files = [
            "source_lines.npz",
            "source_freeze.json",
            "observation_tables.npz",
            "synthetic_material.npz",
            "evidence_directions.csv",
            "square_compatibility.csv",
            "conditional_target_contrasts.csv",
            "stage_summary.csv",
            "synthetic_mechanisms.csv",
            "summary.json",
        ]
        write(
            out / "output_seal.json",
            {"files": {name: digest(out / name) for name in files}},
        )
        write(
            out / "checkpoint.json",
            {"status": "COMPLETED_HM67", "active_processes": []},
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "stage_summaries"}, indent=2
            )
        )
        print(
            json.dumps(
                [r for r in summaries if r["reference"] == "own_family"], indent=2
            )
        )
    except Exception as error:
        write(out / "checkpoint.json", {"status": "FAILED_HM67", "error": repr(error)})
        raise


def verify(out):
    """Independently replay raw links, scalar projections and direct square bounds."""
    bound_check(out)
    for name, sha in read_json(out / "output_seal.json")["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Output seal drift:" + name)
    with np.load(out / "observation_tables.npz", allow_pickle=False) as stream:
        arrays = {k: stream[k].copy() for k in stream.files}
    with np.load(out / "source_lines.npz", allow_pickle=False) as stream:
        refs = {k: stream[k].copy() for k in stream.files}
    grouped, _, _ = read_tables(SOURCE, read_json(SOURCE / "00/rows.json"))
    maximum, count = 0.0, 0

    def compare(a, b):
        """Count actual finite comparisons and reject substantive arithmetic error."""
        nonlocal maximum, count
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Alternate shape/finiteness mismatch")
        maximum = max(maximum, float(np.max(np.abs(a - b), initial=0)))
        count += a.size

    for family, group in grouped.items():
        z = group["logits"]
        centered = z - z.mean(-1, keepdims=True)
        x = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
        za = np.tile(centered.mean(0), (64, 1, 1))
        for j in range(6):
            slope = np.mean(centered * x[:, j, None, None], axis=0)
            za += x[:, j, None, None] * slope
        alt = {
            "score": softmax(z, axis=-1) @ (np.arange(10) / 9),
            "log_odds": logsumexp(z[:, :, 5:], axis=-1)
            - logsumexp(z[:, :, :5], axis=-1),
            "centered_logits": centered,
            "body_endpoint": group["body"],
            "score_link_control": softmax(za, axis=-1) @ (np.arange(10) / 9),
            "log_odds_link_control": logsumexp(za[:, :, 5:], axis=-1)
            - logsumexp(za[:, :, :5], axis=-1),
        }
        for stage, value in alt.items():
            compare(arrays[f"f{family}_{stage}"], value)
    with (out / "evidence_directions.csv").open(newline="") as stream:
        erows = list(csv.DictReader(stream))
    with (out / "square_compatibility.csv").open(newline="") as stream:
        srows = list(csv.DictReader(stream))
    if len(erows) != 108 or len(srows) != 4320:
        raise ValueError("Incomplete declared stages/references")
    for row in erows:
        family, stage, j = int(row["family"]), row["stage"], int(row["evidence"])
        values = arrays[f"f{family}_{stage}"].reshape(64, -1)
        direction = (
            refs[stage][j]
            if row["reference"] == "source_family0"
            else arrays[f"f{family}_{stage}_own_lines"][j]
        )
        indices = [i for i in range(64) if not i & (1 << (5 - j))]
        edges = values[np.array(indices) + (1 << (5 - j))] - values[indices]
        compare(float(row["total_energy"]), math.fsum(v * v for v in edges.ravel()))
        if row["identified"] == "True":
            u = direction / math.sqrt(math.fsum(v * v for v in direction))
            residual = [
                v - math.fsum(a * b for a, b in zip(v, u, strict=True)) * u
                for v in edges
            ]
            compare(
                float(row["normal_energy"]),
                math.fsum(v * v for v in np.array(residual).ravel()),
            )
    for row in srows:
        family, stage = int(row["family"]), row["stage"]
        values = arrays[f"f{family}_{stage}"].reshape(64, -1)
        mm, pm, mp, pp = json.loads(row["corners"])
        interaction = values[pp] + values[mm] - values[pm] - values[mp]
        energy = float(interaction @ interaction)
        compare(float(row["interaction_energy"]), energy)
        if row["identified"] == "True":
            lines = (
                refs[stage]
                if row["reference"] == "source_family0"
                else arrays[f"f{family}_{stage}_own_lines"]
            )
            j, k = int(row["j"]), int(row["k"])
            u, v = (
                lines[j] / np.linalg.norm(lines[j]),
                lines[k] / np.linalg.norm(lines[k]),
            )
            pj, pk = (
                interaction - (interaction @ u) * u,
                interaction - (interaction @ v) * v,
            )
            compare(float(row["projection_bound"]), (pj @ pj + pk @ pk) / 2)
            compare(
                float(row["angle_bound"]),
                (1 - abs(float(np.clip(u @ v, -1, 1)))) * energy / 2,
            )
            if float(row["four_edge_normal_energy"]) + 1e-9 < float(
                row["projection_bound"]
            ):
                raise ValueError("Four-edge inequality failed")
    with np.load(out / "synthetic_material.npz", allow_pickle=False) as stream:
        if len(stream.files) != 24:
            raise ValueError("Missing actual synthetic material")
        for key in stream.files:
            value = stream[key].reshape(64, -1)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite synthetic inputs")
    if maximum > 1e-8:
        raise ValueError("Alternate arithmetic discrepancy")
    result = {
        "status": "VERIFIED_FINITE_HM67_NOT_GLOBAL",
        "actual_comparisons": count,
        "maximum_alternate_error": maximum,
        "square_rows": len(srows),
        "new_neural_calls": 0,
        "original_scope_complete": False,
        "same_agent_alternate_arithmetic_not_independent_replication": True,
    }
    write(out / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "verify": verify}[args.phase](args.out)
