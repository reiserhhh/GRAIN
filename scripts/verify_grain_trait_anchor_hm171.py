#!/usr/bin/env python3
"""Readonly alternate count and density audit of all five fixed trait anchors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_trait_anchor_hm171 as S


def main():
    """Recount events and source author labels; do not invoke the cold formula."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=S.DEFAULT)
    args = p.parse_args()
    out = args.out
    manifest = S.check(out, S.STAGES)
    spec = manifest["config"]
    parent = ROOT / spec["parent"]
    z = S.observations(out)
    y = S.labels(spec)
    law = S.R.read(out / "source_law.json")
    inputs = [
        json.loads(s) for s in (parent / "cold_inputs.jsonl").read_text().splitlines()
    ]
    cut = law["coordinate_cutoffs"]
    mass = np.zeros((624, 2, 8))
    for i, rec in enumerate(inputs):
        u, win = divmod(i, 2)
        if rec["author_key"] != f"window_{u}_{win}":
            raise ValueError("Changed old outcome-free window address")
        for e in rec["events"]:
            cell = (
                4 * e["condition"]
                + int(e["coordinates"][0] > cut[0])
                + 2 * int(e["coordinates"][5] > cut[1])
            )
            mass[u, win, cell] += 1 / len(rec["events"])
    full = np.array(
        [
            sum(mass[u, w] * z["counts"][u, w] for w in range(2)) / sum(z["counts"][u])
            for u in range(624)
        ]
    )
    max_error = float(np.max(np.abs(mass - z["mass"])))
    src = z["source_train"]
    source_cut = np.median(y[src], axis=0)
    binary = (y[src] > source_cut).astype(float)
    diag = S.R.read(out / "source_diagnostics.json")
    perm = np.array(diag["source_profile_permutation"])
    if not np.array_equal(np.sort(perm), np.arange(len(src))):
        raise ValueError("Source negative control did not permute whole profiles")
    half = S.R.read(out / "source_half_laws.json")
    if half["source_positions"] != [list(range(k, len(src), 2)) for k in range(2)]:
        raise ValueError("Half-source assignment changed")
    candidates = [
        (src, binary, law),
        (src, binary[perm], S.R.read(out / "source_permuted_law.json")),
    ]
    candidates += [
        (src[h], binary[h], b)
        for h, b in zip(half["source_positions"], half["laws"], strict=True)
    ]
    fields = []
    for ids, outcomes, bundle in candidates:
        r, joint, pi = np.zeros(8), np.zeros((8, 5)), np.zeros(5)
        for u, outcome in zip(ids, outcomes, strict=True):
            r += full[u] / len(ids)
            joint += np.outer(full[u], outcome) / len(ids)
            pi += outcome / len(ids)
        psi = joint / r[:, None]
        rho1, rho0 = joint / pi, (r[:, None] - joint) / (1 - pi)
        for actual, name in (
            (r, "reference_mass"),
            (joint, "joint_positive_mass"),
            (pi, "prior"),
            (psi, "probabilities"),
            (rho1, "rho1"),
            (rho0, "rho0"),
        ):
            max_error = max(max_error, float(np.max(np.abs(actual - bundle[name]))))
        max_error = max(
            max_error,
            float(
                np.max(np.abs(psi - pi - pi * (1 - pi) * (rho1 - rho0) / r[:, None]))
            ),
        )
        fields.append((r, pi, psi))
    max_error = max(max_error, float(np.max(np.abs(source_cut - law["label_cutoffs"]))))
    r, pi, psi = fields[0]
    half_centered = [b[2] - r @ b[2] for b in fields[2:]]
    norms = [np.sqrt(np.sum(r[:, None] * v * v, axis=0)) for v in half_centered]
    inner = np.sum(r[:, None] * half_centered[0] * half_centered[1], axis=0)
    max_error = max(
        max_error,
        float(np.max(np.abs(inner / (norms[0] * norms[1]) - diag["weighted_cosine"]))),
    )
    chi = np.array(
        [
            [
                np.prod(
                    [1 if cell & (1 << b) else -1 for b in range(3) if mask & (1 << b)]
                )
                for mask in range(8)
            ]
            for cell in range(8)
        ]
    )
    all_scores, checks = [], 0
    for name, (rr, pp, ff) in zip(
        ("cold_scores.jsonl", "permuted_source_scores.jsonl"), fields[:2], strict=True
    ):
        rows = [json.loads(s) for s in (out / name).read_text().splitlines()]
        got_scores = np.zeros((624, 2, 5))
        mu = np.array(
            [
                np.sum(rr[4 * c : 4 * c + 4, None] * ff[4 * c : 4 * c + 4], axis=0)
                / sum(rr[4 * c : 4 * c + 4])
                for c in range(2)
            ]
        )
        for i, got in enumerate(rows):
            u, win = divmod(i, 2)
            p_u = mass[u, win]
            means = sum(p_u[a] * ff[a] for a in range(8))
            choice = sum(
                sum(p_u[4 * c : 4 * c + 4] - rr[4 * c : 4 * c + 4]) * mu[c]
                for c in range(2)
            )
            within = sum(p_u[a] * (ff[a] - mu[a // 4]) for a in range(8))
            coeff = sum(chi[a, :, None] * (ff[a] - pp) for a in range(8)) / 8
            moments = sum((p_u[a] - rr[a]) * chi[a] for a in range(8))
            parts = moments[:, None] * coeff
            if (
                got["author_key"] != f"window_{u}_{win}"
                or got["events"] != z["counts"][u, win]
                or got["status"] != "OBSERVED_TRAIT_FUNCTION"
                or got["person_posterior"]
            ):
                raise ValueError("Changed cold address/status/interpretation")
            for field, actual in (
                ("mean", means),
                ("relative", means - pp),
                ("choice", choice),
                ("within", within),
                ("mass", p_u),
                ("walsh_contributions", parts),
            ):
                max_error = max(max_error, float(np.max(np.abs(actual - got[field]))))
                checks += np.size(actual)
            got_scores[u, win] = means - pp
        all_scores.append(got_scores)
    perm_checks = 0
    with np.load(out / "permutation_accounts.npz", allow_pickle=False) as f:
        summary = S.R.read(out / "summary.json")
        for role in ("source_train", "source_validation", "held"):
            ids, groups = z[role], z["strata"][z[role]]
            xx, yy = all_scores[0][ids].copy(), y[ids].copy()
            for g in sorted(set(groups)):
                ix = np.flatnonzero(groups == g)
                xx[ix] -= np.sum(xx[ix], axis=0) / len(ix)
                yy[ix] -= np.sum(yy[ix], axis=0) / len(ix)
            observed = np.sum(xx * yy[:, None, :], axis=0) / len(ids)
            max_error = max(
                max_error, float(np.max(np.abs(observed - f[role + "_observed"])))
            )
            for k, index in enumerate(f[role + "_indices"]):
                if not np.array_equal(
                    np.sort(index), np.arange(len(ids))
                ) or not np.array_equal(groups, groups[index]):
                    raise ValueError("Matched permutation broke whole-author profile")
                value = np.sum(xx * yy[index, None, :], axis=0) / len(ids)
                max_error = max(
                    max_error, float(np.max(np.abs(value - f[role + "_statistics"][k])))
                )
                perm_checks += value.size
            tail = (
                1 + (np.abs(f[role + "_statistics"]) >= np.abs(observed)).sum(0)
            ) / (spec["permutations"] + 1)
            max_error = max(
                max_error,
                float(
                    np.max(
                        np.abs(
                            tail
                            - summary["roles"][role][
                                "matched_two_sided_tail_ranks_by_window"
                            ]
                        )
                    )
                ),
            )
    with (out / "metrics_by_role_window_trait.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 30:
        raise ValueError("Five traits x two windows x three roles not retained")
    for row in rows:
        ids, win, j = z[row["role"]], int(row["window"]), S.A.TRAITS.index(row["trait"])
        got = np.corrcoef(all_scores[0][ids, win, j], y[ids, j])[0, 1]
        max_error = max(max_error, float(abs(got - float(row["score_trait_r"]))))
        cov = sum(float(row[k]) for k in ("choice_trait_cov", "within_trait_cov"))
        wcov = sum(
            float(row[k])
            for k in (
                "condition_only_trait_cov",
                "coordinate_only_trait_cov",
                "condition_coordinate_trait_cov",
            )
        )
        max_error = max(
            max_error,
            abs(cov - float(row["full_trait_cov"])),
            abs(wcov - float(row["full_trait_cov"])),
        )
    if max_error > 1e-12:
        raise ValueError(
            "Readonly alternate trait-account audit failed: " + str(max_error)
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "max_error": max_error,
                "cold_numeric_values": checks,
                "permutation_values": perm_checks,
                "source_laws": len(fields),
                "main_trait_rows": len(rows),
                "new_neural_forwards": 0,
                "same_agent_not_blind": True,
                "old_and_new_seals_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
