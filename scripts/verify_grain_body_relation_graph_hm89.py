#!/usr/bin/env python3
"""Rebuild HM89 with linear systems and direct sums, not primary graph routines."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_cold_evidence import rotate
from scripts.run_grain_hierarchical_query_hm88 import load, sha, unit, verify, write

OUT = ROOT / "results/grain_math_response_v1/body_relation_graph_r1"


def table(path):
    """Read all declared primary rows for independent arithmetic comparison."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def probabilities(scores):
    """Normalize stable positive weights independently of scipy softmax."""
    p = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def solve(transition, labels):
    """Solve the absorbing boundary equations as a separate matrix problem."""
    n = len(labels)
    b = np.zeros((n, 3))
    boundary, interior = np.flatnonzero(labels >= 0), np.flatnonzero(labels < 0)
    b[boundary, labels[boundary]] = 1
    rhs = transition[np.ix_(interior, boundary)] @ b[boundary]
    rhs[:, 2] += 1 - transition[interior].sum(1)
    b[interior] = np.linalg.solve(
        np.eye(len(interior)) - transition[np.ix_(interior, interior)], rhs
    )
    return b


def run():
    """Recompute every arm, role record, probability account and author contrast."""
    manifest = load(OUT / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(OUT, manifest["inputs"])
    verify(OUT, load(OUT / "01/raw_seal.json"))
    verify(OUT, load(OUT / "02/output_seal.json"))
    rows, bundle = load(OUT / "00/rows.json"), load(OUT / "00/bundle.json")
    points = list(itertools.product((-1, 1), repeat=4))
    with np.load(OUT / "02/projections.npz", allow_pickle=False) as f:
        original = f["fields"].copy()
        pa, pb = f["pplus"].copy(), f["pminus"].copy()
    fields = np.zeros_like(original)
    account = {
        (int(r["unit"]), int(r["query"]), r["arm"]): r
        for r in table(OUT / "02/probability_accounts.csv")
    }
    role = {
        (int(r["unit"]), r["method"], int(r["event"])): r
        for r in table(OUT / "02/token_role_transport.csv")
    }
    errors = {
        "fields": 0.0,
        "accounts": 0.0,
        "role": 0.0,
        "probability_sum": 0.0,
        "author": 0.0,
    }
    checks = 0
    with threadpool_limits(limits=1):
        for row in rows:
            raw = unit(OUT, row["unit"])
            ix = row["body_indices"]
            k, v, q = (
                raw["k"][ix].astype(float),
                raw["v"][ix].astype(float),
                raw["q_body"].astype(float),
            )
            n, labels = len(k), np.array(row["root_labels"])
            ids = np.array(
                [
                    row["event_order"][
                        next(j for j, group in enumerate(row["groups"]) if pos in group)
                    ]
                    for pos in ix
                ]
            )
            full = []
            for h in range(12):
                score = q[:, h] @ k[:, h // (12 // k.shape[1])].T
                score[np.triu_indices(n, 1)] = -np.inf
                full.append(probabilities(score))
            a = np.mean(full, axis=0)
            t = np.tril(a, -1)
            local = t * (ids[:, None] == ids[None, :])
            pos = np.tril(np.ones_like(t), -1)
            for mat in (t, local, pos):
                mass = mat.sum(1)
                mat[:] = np.divide(
                    mat, mass[:, None], out=np.zeros_like(mat), where=mass[:, None] > 0
                )
            direct = np.zeros((n, 3))
            for i in range(n):
                if labels[i] >= 0:
                    direct[i, labels[i]] = 1
                else:
                    direct[i, 0] = t[i, labels == 0].sum()
                    direct[i, 1] = t[i, labels == 1].sum()
                    direct[i, 2] = 1 - direct[i, :2].sum()
            laws = {
                "direct": direct,
                "harmonic": solve(t, labels),
                "event_harmonic": solve(local, labels),
                "position": solve(pos, labels),
            }
            for name, law in laws.items():
                errors["probability_sum"] = max(
                    errors["probability_sum"], float(np.max(np.abs(law.sum(1) - 1)))
                )
                for event, span in enumerate(row["action_groups"]):
                    r = role[row["unit"], name, event]
                    sign = float(np.mean(law[span, 0] - law[span, 1]))
                    errors["role"] = max(
                        errors["role"],
                        abs(sign - float(r["signed_allocation"])),
                        abs(
                            sign * row["coordinates"][event + 2]
                            - float(r["actor_alignment"])
                        ),
                        abs(float(np.mean(law[span, 2])) - float(r["unresolved"])),
                    )
                    checks += 3
            actors = np.array([row["coordinates"][2], row["coordinates"][3], 0])
            gates = {
                "cold": np.ones(n),
                **{name: b[:, 0] for name, b in laws.items()},
                "actor_oracle": np.where(
                    ids == 2, 0.5, (actors[ids] > 0).astype(float)
                ),
                "reporter_oracle": np.where(
                    ids == 2, 0.5, (actors[ids] < 0).astype(float)
                ),
            }
            fi = manifest["spec"]["frames"].index(row["frame"])
            ti = bundle["targets"].index(row["target"])
            ci = points.index(tuple(row["coordinates"]))
            for qi in range(6):
                seed = rotate(
                    bundle["raw_queries"][qi],
                    ix[-1] + bundle["gaps"][qi],
                    bundle["frequencies"],
                    bundle["scaling"],
                )
                score = np.stack(
                    [k[:, h // (12 // k.shape[1])] @ seed[h] for h in range(12)]
                )
                p0 = probabilities(score)
                for ai, name in enumerate(manifest["arms"]):
                    mass = (p0 * gates[name][None]).sum(1)
                    p = p0 * gates[name][None] / mass[:, None]
                    value = 0.0
                    for h in range(12):
                        value += p[h] @ (
                            v[:, h // (12 // k.shape[1])]
                            @ np.array(bundle["directions"])[qi, h]
                        )
                    fields[fi, row["order"], ti, ci, ai, qi] = value
                    r = account[row["unit"], qi, name]
                    candidates = [("retained_mass", float(mass.mean()))] + [
                        (f"event{g}", float(p[:, ids == g].sum(1).mean()))
                        for g in range(3)
                    ]
                    candidates.append(
                        (
                            "unresolved",
                            float((p0 * laws[name][:, 2][None]).sum(1).mean())
                            if name in laws
                            else 0,
                        )
                    )
                    for label, actual in candidates:
                        errors["accounts"] = max(
                            errors["accounts"], abs(actual - float(r[label]))
                        )
                    checks += len(candidates) + 1
            if (row["unit"] + 1) % 96 == 0:
                print("verified units", row["unit"] + 1, flush=True)
    errors["fields"] = float(np.max(np.abs(fields - original)))
    for r in table(OUT / "02/author_law_contrasts.csv"):
        fi, ti, ri = (
            manifest["spec"]["frames"].index(r["frame"]),
            bundle["targets"].index(r["target"]),
            bundle["targets"].index(r["readout"]),
        )
        ai = manifest["arms"].index(r["arm"])
        delta = float((pa - pb) @ fields[fi, int(r["order"]), ti, :, ai, ri])
        errors["author"] = max(
            errors["author"], abs(delta - float(r["author_law_contrast"]))
        )
        checks += 1
    summary_roles = table(OUT / "02/role_summary.csv")
    for r in summary_roles:
        subset = [
            z
            for z in role.values()
            if (z["frame"], z["order"], z["method"])
            == (r["frame"], r["order"], r["method"])
        ]
        if (
            abs(
                np.mean([float(z["actor_alignment"]) for z in subset])
                - float(r["actor_alignment"])
            )
            > 1e-12
        ):
            raise ValueError("Role summary mismatch")
    if max(errors.values()) > manifest["spec"]["tolerance"]:
        raise ValueError("Independent numeric disagreement: " + json.dumps(errors))
    result = {
        "status": "VERIFIED_FINITE_HM89_NOT_GLOBAL",
        "errors": errors,
        "checks": checks,
        "same_agent_alternate_formula_not_third_party": True,
        "no_new_model_calls": True,
        "verification_code_sha256": sha(Path(__file__)),
        "original_scope_complete": False,
    }
    write(OUT / "verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
