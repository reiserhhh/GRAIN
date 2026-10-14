#!/usr/bin/env python3
"""Finite paired-proxy measurement bridges, without latent or teacher inputs.

Moment identification and semantic qualification are separate. A zero equation
residual cannot certify proxy independence or stable measurement channels.
"""

from __future__ import annotations

import numpy as np

VERSION = "HM252_paired_proxy_measurement_v1"


def decode(vectors, alphabet):
    """Decode a registered finite vector alphabet, never arbitrary new text."""
    a, v = np.asarray(alphabet, float), np.asarray(vectors, float)
    if (
        a.ndim != 2
        or len(a) < 2
        or len(np.unique(a, axis=0)) != len(a)
        or v.ndim != 2
        or v.shape[1] != a.shape[1]
        or not np.isfinite(a).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("Distinct finite alphabet and aligned vectors required")
    distances = ((v[:, None] - a[None]) ** 2).sum(axis=-1)
    address = distances.argmin(axis=1)
    if np.any(distances[np.arange(len(v)), address] > 1e-20):
        raise ValueError("Out-of-alphabet vector; no natural-text calibration")
    return address


def observed_joint(
    conditions, z_vectors, w_vectors, outcomes, weights, units, alphabet
):
    """Aggregate qualified weighted paired records; reject same-event views.

    Units are observation-ledger IDs, not claims that distinct events have
    conditionally independent errors. Weights represent an exact finite census.
    """
    c, y, weight, ids = (
        np.asarray(conditions),
        np.asarray(outcomes),
        np.asarray(weights, float),
        np.asarray(units),
    )
    z, w = decode(z_vectors, alphabet), decode(w_vectors, alphabet)
    n = len(w)
    if (
        c.shape != (n,)
        or y.shape != (n, 2)
        or weight.shape != (n,)
        or ids.shape != (n, 3)
        or len(z) != n
        or not np.isin(c, [0, 1]).all()
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(weight).all()
        or np.any(weight <= 0)
        or len(set(ids.flatten().tolist())) != ids.size
    ):
        raise ValueError(
            "Qualified distinct events and positive paired census required"
        )
    joint = np.zeros((2, len(alphabet), len(alphabet), 2, 2))
    np.add.at(joint, (c, z, w, y[:, 0], y[:, 1]), weight)
    if joint.sum() <= 0 or np.any(joint.sum(axis=(1, 2, 3, 4)) == 0):
        raise ValueError("Both condition supports required")
    return joint / joint.sum()


def bridge_solution(kernel, means, rank_tolerance=1e-10, equation_tolerance=1e-9):
    """Solve observed moments, retaining nonexistence and all null directions."""
    k, m = np.asarray(kernel, float), np.asarray(means, float)
    if (
        k.ndim != 2
        or m.ndim != 2
        or len(m) != len(k)
        or not np.isfinite(k).all()
        or not np.isfinite(m).all()
        or np.any(k < 0)
        or not np.allclose(k.sum(axis=1), 1)
        or not 0 < rank_tolerance < equation_tolerance
    ):
        raise ValueError("Positive conditional operator and aligned moments required")
    u, singular, vh = np.linalg.svd(k, full_matrices=True)
    rank = int(np.count_nonzero(singular > rank_tolerance))
    h = vh[:rank].T @ ((u[:, :rank].T @ m) / singular[:rank, None])
    residual = np.max(np.abs(k @ h - m), axis=0)
    return {
        "kernel": k.tolist(),
        "means": m.tolist(),
        "representative": h.tolist(),
        "null_basis": vh[rank:].T.tolist(),
        "rank": rank,
        "residual": residual.tolist(),
        "status": [
            "MOMENT_BRIDGE_EXISTS" if r <= equation_tolerance else "NO_MOMENT_BRIDGE"
            for r in residual
        ],
    }


def compile_bundle(joint, alphabet, target_ids, rank_tolerance=1e-10):
    """Compile only observed census moments; no hidden classes or application Y."""
    j, a = np.asarray(joint, float), np.asarray(alphabet, float)
    decode(a, a)
    if (
        j.shape != (2, len(a), len(a), 2, 2)
        or not np.isfinite(j).all()
        or np.any(j < 0)
        or not np.isclose(j.sum(), 1)
        or len(target_ids) != 2
        or len(set(target_ids)) != 2
    ):
        raise ValueError(
            "Two conditions, targets and complete observed joint law required"
        )
    programs = []
    for cell in j:
        zw = cell.sum(axis=(2, 3))
        z_mass = zw.sum(axis=1)
        if np.any(z_mass <= 0):
            raise ValueError("Missing Z support; no pseudocount calibration")
        kernel = zw / z_mass[:, None]
        means = (
            np.column_stack(
                (cell[:, :, 1, :].sum(axis=(1, 2)), cell[:, :, :, 1].sum(axis=(1, 2)))
            )
            / z_mass[:, None]
        )
        program = bridge_solution(kernel, means, rank_tolerance)
        w_mass = zw.sum(axis=0)
        if np.any(w_mass <= 0):
            raise ValueError("Missing W support")
        program["ordinary_regression"] = (
            np.column_stack(
                (cell[:, :, 1, :].sum(axis=(0, 2)), cell[:, :, :, 1].sum(axis=(0, 2)))
            )
            / w_mass[:, None]
        ).tolist()
        programs.append(program)
    reference = [float(j[:, :, :, 1, :].sum()), float(j[:, :, :, :, 1].sum())]
    return {
        "version": VERSION,
        "target_ids": list(target_ids),
        "alphabet": a.tolist(),
        "conditions": programs,
        "reference": reference,
        "contract": "Moment equation only; proxy qualification/channel transport are separate",
    }


def author_law(conditions, vectors, weights, alphabet):
    """Count an author's condition choice and within-condition vector laws."""
    c, weight = np.asarray(conditions), np.asarray(weights, float)
    w = decode(vectors, alphabet)
    if (
        c.shape != w.shape
        or weight.shape != w.shape
        or not np.isin(c, [0, 1]).all()
        or not np.isfinite(weight).all()
        or np.any(weight < 0)
        or weight.sum() <= 0
    ):
        raise ValueError("Aligned positive author observations required")
    counts = np.zeros((2, len(alphabet)))
    np.add.at(counts, (c, w), weight)
    mass = counts.sum(axis=1)
    laws = np.divide(
        counts, mass[:, None], out=np.zeros_like(counts), where=mass[:, None] > 0
    )
    return mass / mass.sum(), laws


def functional_reading(program, probability, tolerance=1e-9):
    """Read a law only when the functional is constant across all bridges.

    A returned witness is two feasible representatives, not a bounded interval.
    Representative per-event weights need not be probabilities; never clip them.
    """
    p = np.asarray(probability, float)
    k = np.asarray(program["kernel"], float)
    h = np.asarray(program["representative"], float)
    null = np.asarray(program["null_basis"], float)
    if (
        p.shape != (k.shape[1],)
        or np.any(p < 0)
        or not np.isfinite(p).all()
        or not np.isclose(p.sum(), 1)
    ):
        raise ValueError("Positive within-condition vector law required")
    defect = p @ null
    dual, _, _, _ = np.linalg.lstsq(k.T, p, rcond=1e-10)
    adjoint_error = float(np.max(np.abs(k.T @ dual - p)))
    outputs = []
    for target, status in enumerate(program["status"]):
        if status != "MOMENT_BRIDGE_EXISTS":
            outputs.append({"status": "NO_MOMENT_BRIDGE", "value": None})
        elif np.any(np.abs(defect) > tolerance):
            index = int(np.argmax(np.abs(defect)))
            d = null[:, index]
            outputs.append(
                {
                    "status": "UNIDENTIFIED_FUNCTIONAL",
                    "value": None,
                    "feasible_witness_values": [
                        float(p @ (h[:, target] - d)),
                        float(p @ (h[:, target] + d)),
                    ],
                }
            )
        else:
            outputs.append(
                {
                    "status": "IDENTIFIED_MOMENT_FUNCTIONAL",
                    "value": float(p @ h[:, target]),
                    "adjoint_value": float(
                        dual @ np.asarray(program["means"])[:, target]
                    ),
                    "adjoint_residual": adjoint_error,
                }
            )
    return outputs


def cold_profile(bundle, conditions, vectors, weights):
    """Cold score with vectors/choice only; no application criterion outcomes."""
    fields = {
        "version",
        "target_ids",
        "alphabet",
        "conditions",
        "reference",
        "contract",
    }
    if set(bundle) != fields or bundle["version"] != VERSION:
        raise ValueError(
            "Cold bundle schema excludes latent and application outcome fields"
        )
    choice, laws = author_law(conditions, vectors, weights, bundle["alphabet"])
    conditional = [
        functional_reading(p, law) if mass > 0 else None
        for p, law, mass in zip(bundle["conditions"], laws, choice, strict=True)
    ]
    targets = []
    for q, name in enumerate(bundle["target_ids"]):
        used = [conditional[c][q] for c in range(2) if choice[c] > 0]
        identified = all(x["status"] == "IDENTIFIED_MOMENT_FUNCTIONAL" for x in used)
        value = (
            float(
                sum(
                    choice[c] * conditional[c][q]["value"]
                    for c in range(2)
                    if choice[c] > 0
                )
            )
            if identified
            else None
        )
        ordinary = float(
            sum(
                choice[c]
                * (
                    laws[c]
                    @ np.asarray(bundle["conditions"][c]["ordinary_regression"])[:, q]
                )
                for c in range(2)
            )
        )
        targets.append(
            {
                "target": name,
                "status": "IDENTIFIED_MOMENT_FUNCTIONAL"
                if identified
                else "UNIDENTIFIED_OR_NO_BRIDGE",
                "absolute": value,
                "source_relative": value - bundle["reference"][q]
                if identified
                else None,
                "ordinary": ordinary,
            }
        )
    return {"choice": choice.tolist(), "conditional": conditional, "targets": targets}
