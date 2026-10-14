#!/usr/bin/env python3
"""Evaluate and verify HM93 with direct sums and closed-form graph projection."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_cold_evidence as C
from scripts.run_grain_hierarchical_query_hm88 import load, table, unit, verify, write
from scripts.run_grain_query_evidence_hm86 import seal


def alternate_read(raw, row, bundle):
    """Direct past-root exponential weights and explicit current head sums."""
    ix = row["body_indices"]
    k, v = raw["k"][ix].astype(float), raw["v"][ix].astype(float)
    qb = raw["q_body"]
    group = row["action_groups"][1]
    labels = row["root_labels"]
    endpoint = max(ix[i] for i in group)
    fields = np.zeros((3, 6))
    for qi in range(6):
        query = C.rotate(
            bundle["raw_queries"][qi],
            endpoint + bundle["gaps"][qi],
            bundle["frequencies"],
            bundle["scaling"],
        )
        for h in range(12):
            kh = h // (12 // k.shape[1])
            vh = h // (12 // v.shape[1])
            logit = np.array([np.dot(query[h], k[i, kh]) for i in group])
            weight = np.exp(logit - logit.max())
            weight /= weight.sum()
            for index, i in enumerate(group):
                c = np.dot(v[i, vh], bundle["directions"][qi][h])
                if labels[i] >= 0:
                    gate = float(labels[i] == 0)
                elif i == 0:
                    gate = 0.0
                else:
                    history = k[:i, kh] @ qb[i, h]
                    p = np.exp(history - history.max())
                    gate = p[np.asarray(labels[:i]) == 0].sum() / p.sum()
                fields[0, qi] += c / len(group)
                fields[1, qi] += weight[index] * c
                fields[2, qi] += weight[index] * c * gate
    carrier = np.sum(v[group], axis=0).ravel() / len(group)
    return fields, carrier


def alternate_account(joint, kernel):
    """Use the complete-graph Laplacian solution, not the primary least squares."""
    s, a = (joint + joint.T) / 2, (joint - joint.T) / 2
    even, odd = (
        (kernel + kernel.swapaxes(0, 1)) / 2,
        (kernel - kernel.swapaxes(0, 1)) / 2,
    )
    potential = a.sum(1) / len(a)
    gradient = potential[:, None] - potential[None]
    cycle = a - gradient
    return {
        "value": sum(
            joint[i, j] * kernel[i, j] for i, j in itertools.product(range(3), repeat=2)
        ),
        "traffic": sum(
            s[i, j] * even[i, j] for i, j in itertools.product(range(3), repeat=2)
        ),
        "endpoint": sum(
            gradient[i, j] * odd[i, j] for i, j in itertools.product(range(3), repeat=2)
        ),
        "cycle": sum(
            cycle[i, j] * odd[i, j] for i, j in itertools.product(range(3), repeat=2)
        ),
    }


def run(out):
    """Inspect every actual capture and cold ledger before reporting finite status."""
    manifest = load(out / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(out, manifest["inputs"])
    for stage in ("01/raw_seal.json", "02/output_seal.json", "03/output_seal.json"):
        verify(out, load(out / stage))
    rows, bundle = load(out / "00/rows.json"), load(out / "00/bundle.json")
    spec = manifest["spec"]
    with np.load(out / "02/kernels.npz", allow_pickle=False) as data:
        f, reset, raw_value, raw_reset = (
            data[k].copy() for k in ("kernel", "reset", "raw_value", "raw_reset")
        )
    error = {
        "capture_readout": 0.0,
        "raw_mean_roundoff": 0.0,
        "cold_account": 0.0,
        "relabeling": 0.0,
        "path_truth_counts": 0.0,
        "reversal_pairing": 0.0,
        "reset_cycle": 0.0,
        "account_identity": 0.0,
    }
    checks = 0
    for row in rows:
        alt, carrier = alternate_read(unit(out, row["unit"]), row, bundle)
        key = (
            row["lexicon"],
            spec["frames"].index(row["frame"]),
            spec["targets"].index(row["family"]),
        )
        if row["kind"] == "pair":
            key += (row["previous"], row["current"])
            expected, expected_carrier = f[key], raw_value[key]
        else:
            key += (row["current"],)
            expected, expected_carrier = reset[key], raw_reset[key]
        error["capture_readout"] = max(
            error["capture_readout"], float(np.abs(alt - expected).max())
        )
        error["raw_mean_roundoff"] = max(
            error["raw_mean_roundoff"], float(np.abs(carrier - expected_carrier).max())
        )
        checks += alt.size + carrier.size
    paths = {
        p["author"]: np.asarray(p["states"], int)
        for p in load(out / "00/application_paths.json")
    }
    truth = {p["author"]: p for p in load(out / "00/evaluation_truth.json")}
    accounts = {}
    for author, path in paths.items():
        for mode in ("toy_ring", "open_path"):
            points = np.r_[path, path[0]] if mode == "toy_ring" else path
            counts = np.zeros((3, 3), int)
            for i, j in zip(points[:-1], points[1:], strict=True):
                counts[i, j] += 1
            j = counts / counts.sum()
            if mode == "toy_ring":
                error["path_truth_counts"] = max(
                    error["path_truth_counts"],
                    float(np.max(np.abs(counts - truth[author]["counts"]))),
                )
            for lex, frame, family in itertools.product(range(2), range(2), range(6)):
                kernel = f[lex, frame, family]
                alt = alternate_account(j, kernel)
                accounts[author, mode, lex, frame, family] = alt
                error["account_identity"] = max(
                    error["account_identity"],
                    float(
                        np.abs(
                            alt["value"]
                            - alt["traffic"]
                            - alt["endpoint"]
                            - alt["cycle"]
                        ).max()
                    ),
                )
                checks += 72
                if mode == "toy_ring":
                    reverse = alternate_account(j.T, kernel)
                    error["reversal_pairing"] = max(
                        error["reversal_pairing"],
                        float(np.abs(alt["cycle"] + reverse["cycle"]).max()),
                    )
                    for p in itertools.permutations(range(3)):
                        perm = np.ix_(p, p)
                        relabel = alternate_account(j[perm], kernel[perm])
                        error["relabeling"] = max(
                            error["relabeling"],
                            float(np.abs(relabel["value"] - alt["value"]).max()),
                        )
                        checks += 18
                    reset_kernel = np.broadcast_to(
                        reset[lex, frame, family][None], kernel.shape
                    )
                    reset_cycle = alternate_account(j, reset_kernel)["cycle"]
                    error["reset_cycle"] = max(
                        error["reset_cycle"], float(np.abs(reset_cycle).max())
                    )
    with (out / "03/author_scores.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (
                row["author"],
                row["mode"],
                spec["lexicons"].index(row["lexicon"]),
                spec["frames"].index(row["frame"]),
                spec["targets"].index(row["family"]),
            )
            field = ("uniform_value", "local_query", "local_bound_query").index(
                row["field"]
            )
            qi = spec["targets"].index(row["readout"])
            for term in ("value", "traffic", "endpoint", "cycle"):
                error["cold_account"] = max(
                    error["cold_account"],
                    abs(float(row[term]) - accounts[key][term][field, qi]),
                )
                checks += 1
    plus, minus = (
        np.array(truth["toy_024"]["counts"]) / 45,
        np.array(truth["toy_000"]["counts"]) / 45,
    )
    if truth["toy_024"]["direction"] != 1 or truth["toy_000"]["direction"] != -1:
        raise ValueError("Declared truth order changed")
    delta = np.einsum("ij,lfcijbq->lfcbq", plus - minus, f)
    transport, diagonal = [], []
    for frame, field in itertools.product(range(2), range(3)):
        first, second = delta[:, frame, :, field]
        norm = float(np.linalg.norm(first) * np.linalg.norm(second))
        transport.append(
            {
                "frame": spec["frames"][frame],
                "field": manifest["fields"][field],
                "all36_cosine": float(np.sum(first * second) / norm) if norm else None,
                "all36_same_sign_count": int(np.count_nonzero(first * second > 0)),
                "source_rms": float(np.sqrt(np.mean(first**2))),
                "new_wording_rms": float(np.sqrt(np.mean(second**2))),
            }
        )
        for family in range(6):
            diagonal.append(
                {
                    "frame": spec["frames"][frame],
                    "field": manifest["fields"][field],
                    "family": spec["targets"][family],
                    "source_contrast": float(first[family, family]),
                    "new_wording_contrast": float(second[family, family]),
                }
            )
    main_errors = {k: v for k, v in error.items() if k != "raw_mean_roundoff"}
    if (
        max(main_errors.values()) > spec["arithmetic_atol"]
        or error["raw_mean_roundoff"] > 1e-5
    ):
        raise ValueError("Numerical mismatch; preserve artifacts instead of tuning")
    dest = out / "04"
    dest.mkdir()
    table(dest / "wording_transport.csv", transport)
    table(dest / "diagonal_contrasts.csv", diagonal)
    endpoint_values = np.array(
        [a["endpoint"] for k, a in accounts.items() if k[1] == "open_path"]
    )
    write(
        dest / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM93_NOT_GLOBAL",
            "numerical_errors": error,
            "scalar_checks": checks,
            "actual_forwards": len(rows),
            "actual_model_seconds": load(out / "01/status.json")["model_seconds"],
            "toy_profiles": len(paths),
            "independent_humans": 0,
            "kernel_contrast_abs_min_max": [
                float(np.abs(delta).min()),
                float(np.abs(delta).max()),
            ],
            "open_endpoint_abs_max": float(np.abs(endpoint_values).max()),
            "wording_transport": transport,
            "no_accuracy_selection": True,
            "HM87_failure_retained": manifest["HM87_failed_prefix_gate_retained"],
        },
    )
    write(dest / "output_seal.json", seal(out, dest))
    write(out / "verification.json", load(dest / "summary.json"))
    print("VERIFIED HM93", checks, "checks", error, flush=True)


if __name__ == "__main__":
    run(ROOT / "results/grain_math_response_v1/author_current_r1")
