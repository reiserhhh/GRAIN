#!/usr/bin/env python3
"""HM51 alternate source construction, all-token intervention and signed-action audit."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_source_relational_transport import (
    OUT,
    check,
    references,
    write,
)
from scripts.verify_grain_math_evaluative_routing import native_score
from scripts.verify_grain_math_relational_cut import csv_rows


def group_summary(values):
    """Recompute finite signed statistics with scalar summation, not runner functions."""
    n = len(values)
    square = math.fsum(f * f for f, d in values)
    return {
        "rows": n,
        "patch_rms": math.sqrt(math.fsum(d * d for f, d in values) / n),
        "full_rms": math.sqrt(square / n),
        "signed_projection": math.fsum(f * d for f, d in values) / square
        if square
        else None,
        "remaining_rmse": math.sqrt(math.fsum((f - d) ** 2 for f, d in values) / n),
        "direction_count": sum(f * d > 0 for f, d in values),
        "max_abs_change": max(abs(d) for f, d in values),
    }


def verify(out):
    """Verify actual168calls and source-only construction without fresh model calls."""
    m, status = check(out), read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 168
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or status["model_seconds"] > 600
    ):
        raise ValueError("Require complete bounded168 actual calls")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / name) != sha:
            raise ValueError("Changed actual output " + name)
    seal = read_json(out / "00/source_seal.json")
    if (
        seal["new_source_forwards"] != 0
        or seal["reused_source_captures"] != 24
        or not seal["before_all_new_application_calls"]
    ):
        raise ValueError("Source-sealed-before-application contract changed")
    states, _, stored_vectors = references(m, out)
    parent = ROOT / m["spec"]["parent"]
    caps = [e for e in read_json(parent / "00/design.json") if e["kind"] == "capture"]
    index = {(e["role"], e["content"], e["person"], e["direction"]): e for e in caps}
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    vectors, scores, errors = {}, {}, []
    for e in caps:
        meta = read_json(parent / f"01/units/unit_{e['unit']:04d}.json")
        y, mass = native_score(meta["native"])
        errors.extend(
            (
                abs(y - meta["native"]["conditional_score"]),
                abs(mass - meta["native"]["candidate_mass"]),
            )
        )
        scores[e["receiver"]] = y
    mappings = read_json(out / "00/mappings.json")
    for target in m["spec"]["targets"]:
        ds, actor_ids = {}, {}
        for actor in ("A", "B"):
            zero, one = [index["source", target, actor, d] for d in (0, 1)]
            p = zero["partition"]["suffix"]
            q = one["partition"]["suffix"]
            ids = [rows[zero["receiver"]]["input_ids"][i] for i in p]
            if ids != [rows[one["receiver"]]["input_ids"][i] for i in q]:
                raise ValueError("Changed source query tokens")
            ds[actor] = np.subtract(
                states[one["receiver"]][q].astype(np.float64),
                states[zero["receiver"]][p].astype(np.float64),
            )
            actor_ids[actor] = ids
        ds["shared"], ds["bound"] = (ds["A"] + ds["B"]) * 0.5, (ds["A"] - ds["B"]) * 0.5
        for key, v in ds.items():
            stored = stored_vectors[f"{target}__{key}"]
            if not np.array_equal(stored, v):
                raise ValueError(
                    "Source construction does not reproduce frozen sector " + key
                )
            vectors[target, key] = v
        changed = [
            i
            for i in range(len(actor_ids["A"]))
            if actor_ids["A"][i] != actor_ids["B"][i]
        ]
        proof = mappings["actor_slot"][target]
        if (
            len(changed) != 1
            or changed != [proof["intentional_actor_slot"]]
            or [actor_ids[a][changed[0]] for a in ("A", "B")]
            != proof["actor_token_ids"]
        ):
            raise ValueError("Actor-slot changed more than the intended query name")
    design, effects, by_pair = read_json(out / "00/design.json"), {}, defaultdict(dict)
    groups, targets, people = defaultdict(list), defaultdict(list), defaultdict(list)
    max_pre, max_random, max_exact, max_noop = 0.0, 0.0, 0.0, 0.0
    for e in design:
        if e["role"] != "application" or e["direction"] != e["pair"] % 2:
            raise ValueError("Application-only bidirectional design changed")
        src = index["source", e["content"], e["person"], e["direction"]]
        app = index["application", e["content"], e["person"], e["direction"]]
        proof = mappings["same_person"][str(e["pair"])]
        p, sp = e["partition"]["suffix"], src["partition"]["suffix"]
        ids = [rows[e["receiver"]]["input_ids"][i] for i in p]
        if (
            ids != [rows[src["receiver"]]["input_ids"][i] for i in sp]
            or proof
            != {
                "source_positions": sp,
                "application_positions": p,
                "suffix_ids": ids,
                "relative_slots": list(range(len(p))),
            }
            or app["unit"] != e["parent_capture"]
        ):
            raise ValueError("Invalid exact same-person source/application mapping")
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        if any(
            meta[k] != e[k]
            for k in ("unit", "pair", "role", "kind", "arm", "receiver", "donor")
        ):
            raise ValueError("Actual call identity mismatch")
        current, donor = states[e["receiver"]], states[e["donor"]]
        if list(a["patch_positions"]) != p:
            raise ValueError("Incomplete suffix replacement")
        arm, actor, t = e["arm"], e["person"], e["content"]
        if arm == "exact_suffix":
            value = donor[p]
        elif arm == "noop":
            value = current[p]
        else:
            if arm == "source_full":
                delta = vectors[t, actor]
            elif arm == "source_shared":
                delta = vectors[t, "shared"]
            elif arm == "source_bound":
                delta = (1 if actor == "A" else -1) * vectors[t, "bound"]
            elif arm == "wrong_person":
                delta = vectors[t, "B" if actor == "A" else "A"]
            elif arm == "random":
                delta = np.random.default_rng(
                    m["spec"]["random_seed"] + e["pair"] // 2
                ).normal(size=vectors[t, actor].shape)
                lengths = np.sqrt(np.sum(vectors[t, actor] ** 2, axis=1))
                delta = delta * (lengths / np.sqrt(np.sum(delta**2, axis=1)))[:, None]
            else:
                raise ValueError("Unexpected arm")
            value = (
                current[p].astype(float) + (1 - 2 * e["direction"]) * delta
            ).astype(current.dtype)
            if arm == "random":
                actual = np.sqrt(
                    np.sum(
                        (value.astype(float) - current[p].astype(float)) ** 2, axis=1
                    )
                )
                max_random = max(
                    max_random,
                    float(np.max(abs(actual - lengths) / np.maximum(lengths, 1e-20))),
                )
        if not np.array_equal(value, a["patch_values"]):
            raise ValueError("Wrong source-only replacement")
        expected = a["before"].copy()
        expected[p] = value
        if not np.array_equal(expected, a["full"]):
            raise ValueError("Changed unselected token or wrong actual assignment")
        max_pre = max(
            max_pre,
            float(np.max(abs(a["before"].astype(float) - current.astype(float)))),
        )
        y, mass = native_score(meta["native"])
        native, target = scores[e["receiver"]], scores[e["donor"]]
        f, d = target - native, y - native
        effects[e["unit"]] = {
            "native": native,
            "donor": target,
            "patched": y,
            "candidate_mass": mass,
            "full_change": f,
            "patch_change": d,
            "remaining": target - y,
        }
        errors.extend(
            (
                abs(y - meta["native"]["conditional_score"]),
                abs(mass - meta["native"]["candidate_mass"]),
            )
        )
        if arm == "exact_suffix":
            old, _ = native_score(
                read_json(
                    parent / f"01/units/unit_{e['parent_exact_suffix']:04d}.json"
                )["native"]
            )
            max_exact = max(max_exact, abs(y - old))
        if arm == "noop":
            max_noop = max(max_noop, abs(y - native))
        by_pair[e["pair"]][arm] = y
        groups[arm].append((f, d))
        targets[t, arm].append((f, d))
        people[actor, arm].append((f, d))
    table = csv_rows(out / "03/effects.csv")
    if len(table) != 168 or len(effects) != 168 or len(by_pair) != 24:
        raise ValueError("Incomplete real effects")
    for r in table:
        for k, v in effects[int(r["unit"])].items():
            errors.append(abs(float(r[k]) - v))
    cell_terms = []
    for r in csv_rows(out / "03/four_cells.csv"):
        pair, v = int(r["pair"]), by_pair[int(r["pair"])]
        e = next(e for e in design if e["pair"] == pair)
        y0 = scores[e["receiver"]]
        a, b = v["source_shared"] - y0, v["source_bound"] - y0
        i = math.fsum((v["source_full"], y0, -v["source_shared"], -v["source_bound"]))
        joint = v["source_full"] - y0
        expected = {
            "y00": y0,
            "y10": v["source_shared"],
            "y01": v["source_bound"],
            "y11": v["source_full"],
            "shared_action": a,
            "bound_action": b,
            "interaction": i,
            "joint_action": joint,
            "reconstruction_error": math.fsum((joint, -a, -b, -i)),
        }
        for k, z in expected.items():
            errors.append(abs(float(r[k]) - z))
        cell_terms.append((a, b, i, joint))
    if len(cell_terms) != 24:
        raise ValueError("Incomplete24 source-sector interactions")
    x, n = cell_terms, len(cell_terms)
    energy = {
        "shared_square": math.fsum(v[0] ** 2 for v in x) / n,
        "bound_square": math.fsum(v[1] ** 2 for v in x) / n,
        "interaction_square": math.fsum(v[2] ** 2 for v in x) / n,
        "shared_bound_cross": math.fsum(2 * v[0] * v[1] for v in x) / n,
        "shared_interaction_cross": math.fsum(2 * v[0] * v[2] for v in x) / n,
        "bound_interaction_cross": math.fsum(2 * v[1] * v[2] for v in x) / n,
    }
    energy["reconstructed_square"], energy["joint_square"] = (
        math.fsum(energy.values()),
        math.fsum(v[3] ** 2 for v in x) / n,
    )
    for k, v in energy.items():
        errors.append(abs(float(csv_rows(out / "03/signed_energy.csv")[0][k]) - v))
    for r in csv_rows(out / "03/source_geometry.csv"):
        a, b, shared, bound = [
            vectors[r["content"], k].ravel() for k in ("A", "B", "shared", "bound")
        ]
        expected = {
            "full_square": math.fsum(float(v) ** 2 for v in a)
            + math.fsum(float(v) ** 2 for v in b),
            "shared_square": 2 * math.fsum(float(v) ** 2 for v in shared),
            "bound_square": 2 * math.fsum(float(v) ** 2 for v in bound),
            "person_difference_cosine": float(
                a @ b / (np.sqrt(a @ a) * np.sqrt(b @ b))
            ),
        }
        for k, v in expected.items():
            # Large state energies use relative roundoff, unlike bounded scalar scores.
            errors.append(abs(float(r[k]) - v) / max(1.0, abs(v)))
    summary = read_json(out / "03/summary.json")
    for file, lookup in (
        ("arm_summary.csv", groups),
        ("content_summary.csv", targets),
        ("person_summary.csv", people),
    ):
        table = csv_rows(out / "03" / file)
        if len(table) != len(lookup):
            raise ValueError("Missing grouped target/person results")
        for r in table:
            key = (
                r["arm"]
                if file.startswith("arm")
                else (
                    r["content"] if file.startswith("content") else r["person"],
                    r["arm"],
                )
            )
            expected = group_summary(lookup[key])
            for k, v in expected.items():
                if v is None:
                    if r[k] != "":
                        raise ValueError("Invented zero-denominator projection")
                else:
                    errors.append(abs(float(r[k]) - v))
            if file.startswith("arm"):
                for k, v in expected.items():
                    if v is not None:
                        errors.append(abs(summary["groups"][key][k] - v))
    for k, v in (
        ("effects", 168),
        ("four_cells", 24),
        ("new_source_calls", 0),
        ("max_fresh_prepatch_difference", max_pre),
        ("exact_suffix_replay_max_error", max_exact),
        ("noop_max_error", max_noop),
        ("interaction_rms", math.sqrt(math.fsum(v[2] ** 2 for v in x) / n)),
    ):
        errors.append(abs(summary[k] - v))
    if (
        max(errors) > 1e-10
        or max_random > m["spec"]["random_norm_relative_tolerance"]
        or max(max_exact, max_noop) > m["spec"]["score_replay_tolerance"]
    ):
        raise ValueError(
            "Source-only assignment/replay/signed arithmetic verification failed"
        )
    result = {
        "status": "VERIFIED_FINITE_HM51_NOT_GLOBAL",
        "actual_forwards": 168,
        "new_source_forwards": 0,
        "source_captures_recomputed": 24,
        "patches_all_token_positions_checked": 168,
        "four_cells_checked": 24,
        "maximum_alternate_scalar_or_relative_energy_error": max(errors),
        "maximum_prepatch_difference": max_pre,
        "random_norm_max_relative_error": max_random,
        "exact_suffix_replay_max_error": max_exact,
        "noop_max_error": max_noop,
        "same_agent_alternate_arithmetic_not_neural_replication": True,
        "verifier_new_model_calls": 0,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result, flush=True)


def main():
    """Check one actual finite result root, with no automatic experimental rerun."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
