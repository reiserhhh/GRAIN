#!/usr/bin/env python3
"""Freeze, execute and audit HM143 source-only conditional class bridges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_conditional_bridge as B
from grain_research.construction import math_multiview_law as M

SPEC = ROOT / "configs/grain_conditional_bridge_hm143.json"
CODE = (
    "configs/grain_conditional_bridge_hm143.json",
    "docs/GRAIN_CONDITIONAL_BRIDGE_PLAN_HM143.md",
    "grain_research/construction/math_conditional_bridge.py",
    "grain_research/construction/math_multiview_law.py",
    "scripts/run_grain_conditional_bridge_hm143.py",
    "tests/test_grain_conditional_bridge.py",
)
EDGES = ((0, 1), (1, 2), (0, 2))


def read(path):
    """Read a local structured artifact without evaluating content."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash every artifact byte, not selected metadata."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Preserve failed/previous attempts by exclusively creating artifacts."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def archive(path, **arrays):
    """Create a named numerical archive exclusively."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def table(path, rows):
    """Export every predeclared outcome, including ambiguous and failing cases."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    """Verify immutable HM142 and freeze this packet's code and alphabet."""
    if out.exists():
        raise FileExistsError("Existing run retained; no overwrite/restart")
    source = ROOT / read(SPEC)["immutable_source"]
    old = read(source / "manifest.json")
    for path, digest in (old["code"] | old["inputs"]).items():
        if sha(path) != digest:
            raise ValueError("Changed immutable HM142 dependency: " + path)
    for name, digest in read(source / "output_seal.json").items():
        if sha(source / name) != digest:
            raise ValueError("Changed immutable HM142 output: " + name)
    out.mkdir(parents=True)
    with np.load(source / "alphabet.npz", allow_pickle=False) as a:
        archive(out / "alphabet.npz", alphabet=a["alphabet"])
    write(
        out / "manifest.json",
        {
            "packet": "HM143",
            "code": {str(ROOT / p): sha(ROOT / p) for p in CODE},
            "inputs": {
                str(source / p): sha(source / p)
                for p in ("manifest.json", "output_seal.json", "alphabet.npz")
            },
            "alphabet_sha256": sha(out / "alphabet.npz"),
            "scope": "Conditional anonymous class coordinates; supplied identities, finite vector simulation, not human personality factors",
        },
    )


def check(out):
    """Check the entire fixed executable/data provenance before run or replay."""
    m = read(out / "manifest.json")
    for path, digest in (m["code"] | m["inputs"]).items():
        if sha(path) != digest:
            raise ValueError("Changed frozen dependency: " + path)
    if sha(out / "alphabet.npz") != m["alphabet_sha256"]:
        raise ValueError("Changed frozen alphabet")


def laws(spec):
    """Declare generator-only conditional emission tables, not estimator inputs."""
    base = np.asarray(spec["emissions"])
    return tuple(
        tuple(base[v][order] for v, order in enumerate(c))
        for c in spec["condition_category_permutations"]
    )


def world_joint(name):
    """Construct fixed latent worlds with identical uniform local marginals."""
    if name == "independent_conditional_classes":
        return np.ones((3, 3, 3)) / 27
    t = np.zeros((3, 3, 3))
    for a in range(3):
        if name == "shared_author_class":
            t[a, a, a] = 1 / 3
        elif name == "soft_frustrated_response":
            t[a, a, a] += 1 / 9
            t[(a + 1) % 3, a, a] += 1 / 9
            t[(a + 1) % 3, (a + 1) % 3, a] += 1 / 9
        else:
            raise ValueError("Unknown declared world")
    return t


def pairs(t):
    """Return ordered AB, BC and AC marginal tables."""
    return {(0, 1): t.sum(axis=2), (1, 2): t.sum(axis=0), (0, 2): t.sum(axis=1)}


def sample(rng, labels, emissions):
    """Generate independent event categories conditional on one local class."""
    return np.stack(
        [
            (rng.random(len(labels))[:, None] > np.cumsum(m[:, labels].T, axis=1)).sum(
                axis=1
            )
            for m in emissions
        ],
        axis=1,
    ).astype(np.int8)


def states(rng, law, n):
    """Draw states for generation; addresses are kept evaluation-only."""
    flat = rng.choice(law.size, n, p=law.ravel())
    return np.stack(np.unravel_index(flat, law.shape), axis=1).astype(np.int8)


def first_decode(categories, alphabet):
    """Decode actual finite first-view vectors; no arbitrary embedding inference."""
    vectors = alphabet[0, categories]
    return (
        np.linalg.norm(vectors[:, None] - alphabet[0][None], axis=2)
        .argmin(axis=1)
        .astype(np.int8)
    )


def pack_models(models):
    """Archive only anonymous source-law estimates, never hidden state IDs."""
    return {
        "weights": np.stack([m["weights"] for m in models]),
        "emissions": np.stack([m["emissions"] for m in models]),
        "eigenvalues": np.stack([m["eigenvalues"] for m in models]),
    }


def coupled(models, observed):
    """Construct all signed anonymous couplings from observed pair frequencies."""
    return {
        e: B.deconvolve_pair(
            observed[e], models[e[0]]["emissions"][0], models[e[1]]["emissions"][0]
        )["coupling"]
        for e in EDGES
    }


def description(result):
    """Serialize all optimal frames and preferred-edge diagnostics."""
    return {
        "optimum_count": result["optimum_count"],
        "objective": result["objective"],
        "unconstrained_objective": result["unconstrained_objective"],
        "forced_consistency_cost": result["forced_consistency_cost"],
        "second_best_gap": result["second_best_gap"],
        "cycle_fixed_points": result["cycle_fixed_points"],
        "cycle": None if result["cycle"] is None else result["cycle"].tolist(),
        "optimal_frames": [
            [s.argmax(axis=0).tolist() for s in frame] for frame in result["frames"]
        ],
        "edge_optimum_counts": [result["edges"][e]["optimum_count"] for e in EDGES],
        "edge_match_masses_or_signed_objectives": [
            result["edges"][e]["match_mass_or_signed_objective"] for e in EDGES
        ],
    }


def route_difference(profiles, result):
    """Compare A-to-C direct versus A-to-B-to-C transport without collapsing it."""
    if result["cycle"] is None:
        return None
    r10, r21, r20 = (result["edges"][e]["matrices"][0] for e in EDGES)
    direct = profiles[:, 0] @ r20.T
    via = profiles[:, 0] @ r10.T @ r21.T
    return float(np.max(abs(direct - via)))


def alignment(model, matrices):
    """Evaluation-only correspondence to generator labels; never choose frames."""
    return min(
        itertools.permutations(range(3)),
        key=lambda p: sum(
            np.sum((a[:, p] - b) ** 2)
            for a, b in zip(model["emissions"], matrices, strict=True)
        ),
    )


def population(spec):
    """Identify exact observation laws and four distinct bridge possibilities."""
    emissions, pi = laws(spec), np.asarray(spec["weights"])
    observed_local = [M.joint_law(pi, e) for e in emissions]
    models = [M.fit_law(t, 3, spec["contraction"]) for t in observed_local]
    orders = [alignment(m, e) for m, e in zip(models, emissions, strict=True)]
    local_recovery = max(
        float(np.max(abs(a[:, p] - b)))
        for m, e, p in zip(models, emissions, orders, strict=True)
        for a, b in zip(m["emissions"], e, strict=True)
    )
    grid = np.array(list(itertools.product(range(3), repeat=3)))
    profiles = np.stack(
        [M.posterior(grid, m["weights"], m["emissions"]) for m in models], axis=1
    )
    rows, arrays, gauges = [], {"observed_local_tensors": np.stack(observed_local)}, {}
    worlds = {name: pairs(world_joint(name)) for name in spec["sample_worlds"]}
    worlds["incompatible_pair_cohorts"] = {
        (0, 1): np.eye(3) / 3,
        (1, 2): np.eye(3) / 3,
        (0, 2): np.eye(3)[:, [1, 2, 0]].T / 3,
    }
    for name, j in worlds.items():
        observed = {e: emissions[e[0]][0] @ j[e] @ emissions[e[1]][0].T for e in EDGES}
        learned = coupled(models, observed)
        result = B.synchronize(learned)
        compatibility = B.triangle_compatibility(learned)
        bound = B.cycle_mismatch_bound(learned, result)
        arrays[name + "_observed_pairs"] = np.stack([observed[e] for e in EDGES])
        arrays[name + "_couplings"] = np.stack([learned[e] for e in EDGES])
        if compatibility["feasible"]:
            arrays[name + "_joint_witness"] = compatibility["joint_witness"]
        gauges[name] = B.gauge_audit(learned, profiles)
        error = max(
            float(np.max(abs(learned[e][np.ix_(orders[e[0]], orders[e[1]])] - j[e])))
            for e in EDGES
        )
        rows.append(
            {
                "world": name,
                "coupling_recovery_error": error,
                "global_joint_feasible": compatibility["feasible"],
                "joint_witness_error": compatibility["witness_error"],
                "cycle_fixed_points": result["cycle_fixed_points"],
                "global_frame_count": result["optimum_count"],
                "forced_consistency_cost": result["forced_consistency_cost"],
                "preferred_edge_mass": result["edges"][(0, 1)][
                    "match_mass_or_signed_objective"
                ],
                "cycle_bound_slack": None if bound is None else bound["slack"],
                "route_difference_max": route_difference(profiles, result),
                "classification": description(result),
            }
        )
    independent, parity = np.ones((3, 3, 3)) / 27, np.zeros((3, 3, 3))
    for a, b in itertools.product(range(3), repeat=2):
        parity[a, b, (-a - b) % 3] = 1 / 9
    first, second = (
        np.einsum("abc,ia,jb,kc->ijk", t, *[e[0] for e in emissions])
        for t in (independent, parity)
    )
    arrays.update(
        independent_latent_triple=independent,
        parity_latent_triple=parity,
        independent_observed_triple=first,
        parity_observed_triple=second,
    )
    witness = {
        "local_law_recovery_error": local_recovery,
        "all_world_local_observation_alias_error": max(
            float(
                np.max(
                    abs(
                        M.joint_law(j.sum(axis=1 - side), emissions[c])
                        - observed_local[c]
                    )
                )
            )
            for world in worlds.values()
            for edge, j in world.items()
            for side, c in enumerate(edge)
        ),
        "uniform_prevalence": pi.tolist(),
        "pair_alias_error": max(
            float(np.max(abs(pairs(independent)[e] - pairs(parity)[e]))) for e in EDGES
        ),
        "latent_triple_alias_TV": float(abs(independent - parity).sum() / 2),
        "observable_triple_alias_TV": float(abs(first - second).sum() / 2),
        "scope": "Perfect local laws/pairwise fitting do not identify global persistence, bijection, or higher-order coupling; populations simulated",
    }
    return rows, arrays, gauges, witness


def metrics(seed, name, models, emissions, observed, learned, cold, scores, sync):
    """Evaluate only after source models/cold profiles are sealed, no selection."""
    orders = [alignment(m, e) for m, e in zip(models, emissions, strict=True)]
    truth = pairs(world_joint(name))
    square_errors = [
        learned[e][np.ix_(orders[e[0]], orders[e[1]])] - truth[e] for e in EDGES
    ]
    errors = [
        models[e[0]]["emissions"][0] @ learned[e] @ models[e[1]]["emissions"][0].T
        - observed[e]
        for e in EDGES
    ]
    return {
        "seed": seed,
        "world": name,
        "source_coupling_RMSE": float(np.sqrt(np.mean(np.stack(square_errors) ** 2))),
        "signed_inverse_minimum": float(min(j.min() for j in learned.values())),
        "observation_reconstruction_error_automatic_for_square_channels": float(
            max(np.max(abs(e)) for e in errors)
        ),
        "candidate_frame_count": sync["optimum_count"],
        "candidate_second_best_gap_not_a_significance_test": sync["second_best_gap"],
        "candidate_cycle_fixed_points": sync["cycle_fixed_points"],
        "forced_consistency_cost": sync["forced_consistency_cost"],
        "cold_profile_variance": float(np.var(scores, axis=0).mean()),
        "cold_normalization_error": float(np.max(abs(scores.sum(axis=2) - 1))),
        "conditional_units_kept": int(np.prod(scores.shape)),
        "cold_route_difference_max": route_difference(scores, sync),
        "cold_authors": len(cold),
        "status": "DESCRIPTIVE_CANDIDATE_NOT_COMMON_TRAIT_ACCEPTANCE",
    }


def run(out):
    """Run finite local laws/bridges with all cold outputs preceding evaluation."""
    check(out)
    write(
        out / "execution_started.json",
        {"status": "STARTED", "replay": "Never restart an existing attempt"},
    )
    start, spec = time.perf_counter(), read(SPEC)
    emissions, pi = laws(spec), np.asarray(spec["weights"])
    with np.load(out / "alphabet.npz", allow_pickle=False) as a:
        alphabet = a["alphabet"].copy()
    population_rows, population_arrays, gauges, witness = population(spec)
    write(out / "population.json", population_rows)
    archive(out / "population.npz", **population_arrays)
    write(out / "gauge_audit.json", gauges)
    write(out / "identification_witness.json", witness)
    all_metrics, failures = [], []
    for seed in spec["seeds"]:
        if time.perf_counter() - start > spec["max_CPU_wall_seconds"]:
            raise TimeoutError("Original CPU cap reached; retain checkpoints")
        root = out / str(seed)
        root.mkdir()
        rng = np.random.default_rng(seed)
        local, hidden = [], []
        for e in emissions:
            h = rng.choice(3, spec["local_source_authors_per_condition_seed"], p=pi)
            raw = sample(rng, h, e)
            vectors = np.stack([alphabet[v, raw[:, v]] for v in range(3)], axis=1)
            local.append(M.decode_vectors(vectors, alphabet))
            hidden.append(h)
        archive(root / "source_observations.npz", categories=np.stack(local))
        archive(
            root / "oracle_source_generation_only.npz", hidden_classes=np.stack(hidden)
        )
        tensors = [M.empirical_tensor(x, (3, 3, 3), spec["event_views"]) for x in local]
        archive(root / "observed_tensors.npz", tensors=np.stack(tensors))
        try:
            models = [M.fit_law(t, 3, spec["contraction"]) for t in tensors]
        except M.IdentificationError as exc:
            failure = {
                "seed": seed,
                "status": "FIXED_LOCAL_LAW_FAILURE",
                "reason": str(exc),
            }
            write(root / "failure.json", failure)
            failures.append(failure)
            continue
        archive(root / "source_models.npz", **pack_models(models))
        for name in spec["sample_worlds"]:
            if time.perf_counter() - start > spec["max_CPU_wall_seconds"]:
                raise TimeoutError("Original CPU cap reached; retain completed worlds")
            directory = root / name
            directory.mkdir()
            bridge, observed = {}, {}
            for edge, j in pairs(world_joint(name)).items():
                labels = states(
                    rng, j, spec["paired_bridge_authors_per_edge_world_seed"]
                )
                x = np.stack(
                    [
                        first_decode(
                            sample(rng, labels[:, side], [emissions[c][0]])[:, 0],
                            alphabet,
                        )
                        for side, c in enumerate(edge)
                    ],
                    axis=1,
                )
                bridge["edge" + str(edge[0]) + str(edge[1])] = x
                observed[edge] = B.empirical_pair(x)
            archive(directory / "bridge_observations.npz", **bridge)
            learned = coupled(models, observed)
            sync = B.synchronize(learned)
            archive(
                directory / "source_couplings.npz",
                couplings=np.stack([learned[e] for e in EDGES]),
                observed_pairs=np.stack([observed[e] for e in EDGES]),
                frames=np.stack(sync["frames"]),
            )
            write(directory / "source_frame_description.json", description(sync))
            labels = states(rng, world_joint(name), spec["cold_authors_per_world_seed"])
            cold = np.stack(
                [
                    M.decode_vectors(
                        np.stack([alphabet[v, raw[:, v]] for v in range(3)], axis=1),
                        alphabet,
                    )
                    for c, e in enumerate(emissions)
                    for raw in [sample(rng, labels[:, c], e)]
                ],
                axis=1,
            )
            archive(directory / "cold_observations.npz", categories=cold)
            archive(
                directory / "oracle_cold_generation_only.npz", hidden_classes=labels
            )
            scores = np.stack(
                [
                    M.posterior(cold[:, c], m["weights"], m["emissions"])
                    for c, m in enumerate(models)
                ],
                axis=1,
            )
            candidates = np.stack(
                [B.stitch_profiles(scores, frame) for frame in sync["frames"]]
            )
            archive(
                directory / "cold_profiles.npz",
                local=scores,
                reference_candidates=candidates,
            )
            write(
                directory / "cold_seal.json",
                {p.name: sha(p) for p in directory.iterdir() if p.is_file()},
            )
            row = metrics(
                seed, name, models, emissions, observed, learned, cold, scores, sync
            )
            write(directory / "metrics.json", row)
            all_metrics.append(row)
    table(out / "metrics.csv", all_metrics)
    write(
        out / "result.json",
        {
            "packet": "HM143",
            "status": "FINITE_CONDITIONAL_BRIDGE_CONSTRUCTOR_EXECUTED",
            "population": population_rows,
            "witness": witness,
            "sample_metrics": all_metrics,
            "failures": failures,
            "source_calibration_author_instances": spec[
                "local_source_authors_per_condition_seed"
            ]
            * 3
            * len(spec["seeds"]),
            "paired_bridge_author_instances": spec[
                "paired_bridge_authors_per_edge_world_seed"
            ]
            * 3
            * len(spec["sample_worlds"])
            * len(spec["seeds"]),
            "cold_author_instances": spec["cold_authors_per_world_seed"]
            * len(spec["sample_worlds"])
            * len(spec["seeds"]),
            "CPU_wall_seconds": time.perf_counter() - start,
            "new_neural_API_private_training_calls": 0,
            "original_scope_complete": False,
            "scope": "Unknown conditional coordinate bridges; all populations simulated, identity supplied; no human personality promotion",
        },
    )
    write(
        out / "output_seal.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )


def verify(out):
    """Read-only full artifact replay and alternative solves/counts/coordinates."""
    check(out)
    for name, digest in read(out / "output_seal.json").items():
        if sha(out / name) != digest:
            raise ValueError("Changed execution artifact: " + name)
    spec = read(SPEC)
    rows, arrays, gauges, witness = population(spec)
    if (
        read(out / "population.json") != rows
        or read(out / "gauge_audit.json") != gauges
        or read(out / "identification_witness.json") != witness
    ):
        raise ValueError("Exact population replay changed")
    with np.load(out / "population.npz", allow_pickle=False) as saved:
        if any(
            np.max(abs(saved[key] - a)) > spec["exact_tolerance"]
            for key, a in arrays.items()
        ):
            raise ValueError("Population numerical archive changed")
    errors = {
        "tensor_count": 0.0,
        "models": 0.0,
        "pair_count": 0.0,
        "alternative_inverse": 0.0,
        "direct_Bayes": 0.0,
        "stitch": 0.0,
        "metrics": 0.0,
        "full_triple_enumeration": 0.0,
    }
    values_checked, pairs_checked = 0, 0
    output_rows, emissions = [], laws(spec)
    for seed in spec["seeds"]:
        root = out / str(seed)
        with np.load(root / "source_observations.npz", allow_pickle=False) as a:
            x = a["categories"].copy()
        with np.load(root / "observed_tensors.npz", allow_pickle=False) as a:
            tensors = a["tensors"].copy()
        for c in range(3):
            count = np.zeros((3, 3, 3))
            np.add.at(count, tuple(x[c].T), 1)
            errors["tensor_count"] = max(
                errors["tensor_count"],
                float(np.max(abs(count / x.shape[1] - tensors[c]))),
            )
        if (root / "failure.json").exists():
            continue
        models = [M.fit_law(t, 3, spec["contraction"]) for t in tensors]
        with np.load(root / "source_models.npz", allow_pickle=False) as saved:
            errors["models"] = max(
                errors["models"],
                max(
                    float(np.max(abs(saved[k] - v)))
                    for k, v in pack_models(models).items()
                ),
            )
        for name in spec["sample_worlds"]:
            directory = root / name
            for file, digest in read(directory / "cold_seal.json").items():
                if sha(directory / file) != digest:
                    raise ValueError("Cold model/profile seal changed")
            observed = {}
            with np.load(
                directory / "bridge_observations.npz", allow_pickle=False
            ) as a:
                for e in EDGES:
                    pair = a["edge" + str(e[0]) + str(e[1])]
                    count = np.zeros((3, 3))
                    np.add.at(count, tuple(pair.T), 1)
                    observed[e] = count / len(pair)
                    pairs_checked += len(pair)
            learned = coupled(models, observed)
            sync = B.synchronize(learned)
            if description(sync) != read(directory / "source_frame_description.json"):
                raise ValueError("Source frame description changed")
            with np.load(directory / "source_couplings.npz", allow_pickle=False) as a:
                for edge_index, e in enumerate(EDGES):
                    errors["pair_count"] = max(
                        errors["pair_count"],
                        float(
                            np.max(abs(a["observed_pairs"][edge_index] - observed[e]))
                        ),
                    )
                    first, second = (
                        models[e[0]]["emissions"][0],
                        models[e[1]]["emissions"][0],
                    )
                    alternative = np.linalg.solve(
                        second, np.linalg.solve(first, observed[e]).T
                    ).T
                    errors["alternative_inverse"] = max(
                        errors["alternative_inverse"],
                        float(np.max(abs(alternative - a["couplings"][edge_index]))),
                    )
                if not np.array_equal(a["frames"], np.stack(sync["frames"])):
                    raise ValueError("Source optimum frame set changed")
            with np.load(directory / "cold_observations.npz", allow_pickle=False) as a:
                cold = a["categories"].copy()
            scores = np.stack(
                [
                    M.posterior(cold[:, c], m["weights"], m["emissions"])
                    for c, m in enumerate(models)
                ],
                axis=1,
            )
            with np.load(directory / "cold_profiles.npz", allow_pickle=False) as a:
                if np.max(abs(a["local"] - scores)) > spec["exact_tolerance"]:
                    raise ValueError("Cold profiles changed")
                for c, m in enumerate(models):
                    direct = np.tile(m["weights"], (len(cold), 1))
                    for v, emission in enumerate(m["emissions"]):
                        direct *= emission[cold[:, c, v]]
                    direct /= direct.sum(axis=1, keepdims=True)
                    errors["direct_Bayes"] = max(
                        errors["direct_Bayes"],
                        float(np.max(abs(direct - a["local"][:, c]))),
                    )
                for index, frame in enumerate(sync["frames"]):
                    direct = np.stack(
                        [scores[:, c, s.argmax(axis=0)] for c, s in enumerate(frame)],
                        axis=1,
                    )
                    errors["stitch"] = max(
                        errors["stitch"],
                        float(np.max(abs(direct - a["reference_candidates"][index]))),
                    )
                values_checked += scores.size + a["reference_candidates"].size
            row = metrics(
                seed, name, models, emissions, observed, learned, cold, scores, sync
            )
            saved = read(directory / "metrics.json")
            for key, val in row.items():
                if isinstance(val, (float, int)) and saved[key] is not None:
                    errors["metrics"] = max(errors["metrics"], abs(val - saved[key]))
                elif saved[key] != val:
                    raise ValueError("Metric replay changed: " + key)
            output_rows.append(row)
    result = read(out / "result.json")
    if result["sample_metrics"] != output_rows or result["population"] != rows:
        raise ValueError("Result-to-world reconciliation failed")
    # Independent six-index arithmetic verifies higher-order observation alias.
    for name, latent in (
        ("independent", arrays["independent_latent_triple"]),
        ("parity", arrays["parity_latent_triple"]),
    ):
        enumerated = np.zeros((3, 3, 3))
        for i, j, k, a, b, c in itertools.product(range(3), repeat=6):
            enumerated[i, j, k] += (
                latent[a, b, c]
                * emissions[0][0][i, a]
                * emissions[1][0][j, b]
                * emissions[2][0][k, c]
            )
        errors["full_triple_enumeration"] = max(
            errors["full_triple_enumeration"],
            float(np.max(abs(enumerated - arrays[name + "_observed_triple"]))),
        )
    if max(errors.values()) > spec["exact_tolerance"]:
        raise AssertionError(errors)
    print(
        json.dumps(
            {
                "status": "READONLY_REPLAY_VERIFIED",
                "errors": errors,
                "cold_values_checked": values_checked,
                "paired_bridge_rows_checked": pairs_checked,
                "scope": "Same-agent alternate arithmetic, not independent blind replication",
            },
            indent=2,
        )
    )


def main():
    """Select an exclusive preparation/run or a strictly read-only verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / read(SPEC)["out"]
    with threadpool_limits(limits=1):
        {"prepare": prepare, "run": run, "verify": verify}[args.action](out)


if __name__ == "__main__":
    main()
