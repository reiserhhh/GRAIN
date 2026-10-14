#!/usr/bin/env python3
"""Qualified episode joins and finite relational identification, not raw NLP."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linprog

SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=int)
FIELDS = (
    "atom_id",
    "author_id",
    "actor",
    "qualification",
    "episode_id",
    "condition",
    "state_id",
    "triple",
    "slot",
    "level",
)


def design_graph(bank, triple_index, triple, law):
    """Supply40 episode hyperedges over the same240 atom IDs in either law."""
    if bank not in (0, 1) or law not in (-1, 1) or len(set(triple)) != 3:
        raise ValueError("Declared bank, sign law and three distinct slots required")
    counters, rows, episode = Counter(), [], 0
    for signs in SIGNS:
        count = 5 + 2 * law * int(np.prod(signs))
        levels = [1] * 6
        for slot, sign in zip(triple, signs, strict=True):
            levels[slot] = int(sign + 1)
        for _ in range(count):
            for slot, level in enumerate(levels):
                key = slot, level
                ordinal = counters[key]
                counters[key] += 1
                rows.append(
                    {
                        "atom_id": f"s{slot}:v{level}:n{ordinal}",
                        "author_id": "A",
                        "actor": "A",
                        "qualification": "own_actual",
                        "episode_id": f"e{episode:02d}",
                        "condition": bank,
                        "state_id": "fixed_current",
                        "triple": triple_index,
                        "slot": slot,
                        "level": level,
                    }
                )
            episode += 1
    return rows


def package(rows, kind, seed):
    """Change report delivery only; never change supplied semantic identities."""
    ordered = list(rows)
    if kind == "reverse":
        ordered.reverse()
    elif kind in ("slot_shuffle", "duplicates"):
        ordered = []
        for slot in range(6):
            block = [r for r in rows if r["slot"] == slot]
            indices = np.random.default_rng(seed + slot).permutation(len(block))
            ordered.extend(block[i] for i in indices)
        if kind == "duplicates":
            ordered.extend(ordered[::7])
    elif kind not in ("episode_order", "packet_13"):
        raise ValueError("Unknown report packaging")
    width = 13 if kind == "packet_13" else 6
    return [
        {**r, "report_id": f"{kind}:r{i}", "packet_id": f"p{i // width}"}
        for i, r in enumerate(ordered)
    ]


def unique_atoms(rows):
    """Deduplicate exact supplied atom identities, rejecting conflicting copies."""
    seen, aliases = {}, 0
    for row in rows:
        if any(field not in row for field in FIELDS):
            raise ValueError(
                "Explicit semantic fields, including unknown nulls, required"
            )
        if row["slot"] not in range(6) or row["level"] not in (0, 1, 2):
            raise ValueError("Declared six-slot levels required")
        atom = {name: row[name] for name in FIELDS}
        key = row["atom_id"]
        if key in seen:
            if seen[key] != atom:
                raise ValueError("Conflicting reports of the same supplied atom")
            aliases += 1
        else:
            seen[key] = atom
    if len({r["author_id"] for r in seen.values()}) > 1:
        raise ValueError("One author ledger per observation")
    return list(seen.values()), aliases


def digest(rows, atomic_only=False):
    """Hash supplied relations independent of delivery order and exact aliases."""
    atoms, _ = unique_atoms(rows)
    if atomic_only:
        atoms = [{k: v for k, v in r.items() if k != "episode_id"} for r in atoms]
    data = json.dumps(sorted(atoms, key=lambda r: r["atom_id"]), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def routed_episodes(rows):
    """Join supplied complete episodes; unknowns remain unavailable, not zero."""
    atoms, aliases = unique_atoms(rows)
    groups, excluded, unresolved = defaultdict(list), [], []
    for row in atoms:
        if row["actor"] is None or row["qualification"] is None:
            unresolved.append({**row, "reason": "unknown_qualification"})
        elif row["actor"] != row["author_id"] or row["qualification"] != "own_actual":
            excluded.append({**row, "reason": "not_own_actual"})
        elif (
            row["episode_id"] is None
            or row["condition"] is None
            or row["state_id"] is None
        ):
            unresolved.append({**row, "reason": "unknown_episode_condition_state"})
        else:
            groups[row["episode_id"]].append(row)
    episodes = []
    for episode, block in sorted(groups.items()):
        if len({(r["condition"], r["state_id"], r["triple"]) for r in block}) != 1:
            raise ValueError("Supplied episode has conflicting condition/state/support")
        slots = [r["slot"] for r in block]
        if len(slots) != len(set(slots)):
            raise ValueError("Multiple distinct atoms claim the same episode slot")
        if set(slots) != set(range(6)):
            unresolved.extend(
                {**r, "reason": "incomplete_six_slot_unit"} for r in block
            )
            continue
        block.sort(key=lambda r: r["slot"])
        episodes.append(
            {
                "episode_id": episode,
                "condition": block[0]["condition"],
                "state_id": block[0]["state_id"],
                "triple": block[0]["triple"],
                "levels": [r["level"] for r in block],
                "atoms": [r["atom_id"] for r in block],
            }
        )
    return {
        "episodes": episodes,
        "unresolved": unresolved,
        "excluded": excluded,
        "aliases": aliases,
        "unique_atoms": len(atoms),
    }


def observation(rows, kernels):
    """Integrate complete qualified episode kernels; retain conditional choice."""
    routed = routed_episodes(rows)
    sums, counts = {}, Counter()
    for row in routed["episodes"]:
        key = row["condition"], tuple(row["levels"])
        if key not in kernels:
            raise ValueError("Episode lies outside the declared observer support")
        condition = row["condition"]
        sums.setdefault(condition, np.zeros_like(kernels[key], dtype=float))
        sums[condition] += kernels[key]
        counts[condition] += 1
    n = sum(counts.values())
    return {
        **routed,
        "score": sum(sums.values()) / n if n else None,
        "conditional": {c: sums[c] / counts[c] for c in counts},
        "choice": {c: counts[c] / n for c in counts},
        "counts": dict(counts),
    }


def order_zip_observation(rows, kernels):
    """Deliberately discard episode links, retaining values/actor/condition only."""
    atoms, _ = unique_atoms(rows)
    groups = defaultdict(lambda: defaultdict(list))
    for row in atoms:
        if row["actor"] == row["author_id"] and row["qualification"] == "own_actual":
            groups[row["condition"]][row["slot"]].append(row["level"])
    values = []
    for condition, slots in groups.items():
        if set(slots) != set(range(6)) or len({len(v) for v in slots.values()}) != 1:
            raise ValueError("Comparator requires balanced complete slot marginals")
        for levels in zip(*(slots[j] for j in range(6)), strict=True):
            values.append(kernels[(condition, tuple(levels))])
    return np.mean(values, axis=0) if values else None


def character_matrix(degree):
    """Return constant and subset-character rows on the fixed binary support."""
    if degree not in (1, 2, 3):
        raise ValueError("Degree must be1,2 or3")
    subsets = [()] + [
        s for d in range(1, degree + 1) for s in itertools.combinations(range(3), d)
    ]
    return np.stack(
        [np.prod(SIGNS[:, s], axis=1) if s else np.ones(8) for s in subsets]
    ).astype(float)


def identified_bounds(kernel, degree, third=None):
    """Compute sharp finite bounds and attainable LP/dual witnesses, not CIs."""
    k = np.asarray(kernel, dtype=float)
    if k.shape != (8,) or not np.isfinite(k).all():
        raise ValueError("Eight finite joint readouts required")
    a = character_matrix(degree)
    b = np.zeros(len(a))
    b[0] = 1
    if degree == 3:
        if third is None or not -1 <= third <= 1:
            raise ValueError("Explicit aggregate third moment required")
        b[-1] = third
    elif third is not None:
        raise ValueError("Third moment needs the full character family")
    witnesses = []
    for direction in (1, -1):
        c = direction * k
        sol = linprog(
            c,
            A_eq=a,
            b_eq=b,
            bounds=(0, None),
            method="highs",
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
            },
        )
        if not sol.success:
            raise ValueError("Inconsistent observation constraints: " + sol.message)
        dual = sol.eqlin.marginals
        slack = c - a.T @ dual
        witnesses.append(
            {
                "value": float(k @ sol.x),
                "p": sol.x.tolist(),
                "dual": dual.tolist(),
                "slack": slack.tolist(),
                "primal_residual": float(np.max(abs(a @ sol.x - b))),
                "duality_gap": float(abs(c @ sol.x - b @ dual)),
            }
        )
    return {
        "lower": witnesses[0]["value"],
        "upper": witnesses[1]["value"],
        "witnesses": witnesses,
        "a": a.tolist(),
        "b": b.tolist(),
        "rank": int(np.linalg.matrix_rank(a)),
        "not_sampling_CI": True,
    }
