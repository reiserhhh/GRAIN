#!/usr/bin/env python3
"""Execute fixed event-graph measurements using sealed HM112 numeric kernels."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_grain_author_evidence_merge_hm102 as H

from grain_research.construction import math_event_graph_observation as M

SPEC = ROOT / "configs/grain_event_graph_hm113.json"
CODE = (
    "configs/grain_event_graph_hm113.json",
    "docs/GRAIN_EVENT_GRAPH_OBSERVATION_PLAN_HM113.md",
    "grain_research/construction/math_event_graph_observation.py",
    "scripts/run_grain_event_graph_hm113.py",
    "tests/test_grain_math_event_graph_observation.py",
)


def check(out):
    """Authenticate the exact frozen plan, inputs and admitted source outputs."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out, spec):
    """Export all declared joint kernels and new graph designs without fitting."""
    if out.exists():
        raise FileExistsError("Preserve existing or failed experiment")
    old = ROOT / spec["previous"]
    manifest = H.load(old / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(old, manifest["inputs"])
    H.verify_hashes(old, H.load(old / "04/output_seal.json"))
    if (
        H.load(old / "verification.json")["status"]
        != "VERIFIED_FINITE_HM112_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM112 joint construction required")
    cases = H.load(old / "00/cases.json")
    with np.load(old / "04/fields.npz", allow_pickle=False) as z:
        compiled = z["compiled_scores"][0]
        reference = z["reference_scores"]
    indices = np.empty((2, 4, 8), dtype=int)
    for b, t in itertools.product(range(2), range(4)):
        matched = [
            r
            for r in cases
            if r["kind"] == "triple"
            and r["bank"] == b
            and r["triple_index"] == t
            and r["owner"] == 1
        ]
        mapping = {tuple(r["signs"]): r["case"] for r in matched}
        if set(mapping) != set(map(tuple, M.SIGNS)):
            raise ValueError("All eight qualified corners required")
        indices[b, t] = [mapping[tuple(s)] for s in M.SIGNS]
    folder = out / "00"
    folder.mkdir(parents=True)
    np.savez(
        folder / "kernels.npz", compiled=compiled[indices], reference=reference[indices]
    )
    graphs = [
        {
            "id": f"b{b}:t{t}:law{law}",
            "bank": b,
            "triple": t,
            "law": law,
            "records": M.design_graph(b, t, triple, law),
        }
        for b, (t, triple), law in itertools.product(
            range(2), enumerate(spec["triples"]), (-1, 1)
        )
    ]
    H.write(folder / "graphs.json", graphs)
    H.write(folder / "source_indices.json", indices.tolist())
    H.write(folder / "targets.json", manifest["spec"]["targets"])
    admitted = (
        "00/manifest.json",
        "00/cases.json",
        "04/fields.npz",
        "04/summary.json",
        "04/output_seal.json",
        "verification.json",
    )
    source = {str((old / n).relative_to(ROOT)): H.sha(old / n) for n in admitted}
    H.write(
        folder / "manifest.json",
        {
            "spec": spec,
            "code": {n: H.sha(ROOT / n) for n in CODE},
            "source": source,
            "inputs": H.seals(out, folder),
            "already_opened_numeric_materials": True,
            "new_graph_designs_not_new_neural_texts": True,
            "source_pretrained_dependency_inherited": True,
        },
    )
    print("PREPARED16 qualified graph laws; no new observer fitting")


def kernel_map(kernels, triples):
    """Index every supplied binary triple by its actual six-slot configuration."""
    result = {}
    for b, (t, triple), (s, signs) in itertools.product(
        range(2), enumerate(triples), enumerate(M.SIGNS)
    ):
        levels = [1] * 6
        for slot, sign in zip(triple, signs, strict=True):
            levels[slot] = int(sign + 1)
        result[b, tuple(levels)] = kernels[b, t, s]
    return result


def clones(rows, prefix):
    """Declare distinct numeric episodes, not duplicate reports of one episode."""
    return [
        {**r, "atom_id": prefix + r["atom_id"], "episode_id": prefix + r["episode_id"]}
        for r in rows
    ]


def run(out):
    """Construct report-invariant measurements and missing-relation intervals."""
    start = time.monotonic()
    manifest = check(out)
    spec = manifest["spec"]
    graphs, targets = H.load(out / "00/graphs.json"), H.load(out / "00/targets.json")
    with np.load(out / "00/kernels.npz", allow_pickle=False) as z:
        compiled, reference = z["compiled"], z["reference"]
    kc, kr = (kernel_map(k, spec["triples"]) for k in (compiled, reference))
    folder = out / "01"
    folder.mkdir()
    (folder / "reports").mkdir()
    packaged, accounts, qualification = [], {}, []
    for gi, graph in enumerate(graphs):
        records = graph["records"]
        base = M.observation(records, kc)["score"]
        reference_base = M.observation(records, kr)["score"]
        accounts[graph["id"]] = (base, reference_base)
        for pi, name in enumerate(spec["packages"]):
            reports = M.package(records, name, spec["seed"])
            H.write(folder / f"reports/g{gi:02d}_p{pi}.json", reports)
            obs, ref = M.observation(reports, kc), M.observation(reports, kr)
            naive = M.order_zip_observation(reports, kc)
            for q, target in enumerate(targets):
                packaged.append(
                    {
                        "graph": graph["id"],
                        "graph_index": gi,
                        "package_index": pi,
                        "package": name,
                        "target": target,
                        "compiled": float(obs["score"][q]),
                        "reference": float(ref["score"][q]),
                        "order_zip": float(naive[q]),
                        "packaging_delta": float(obs["score"][q] - base[q]),
                        "naive_minus_graph": float(naive[q] - base[q]),
                        "episodes": len(obs["episodes"]),
                        "aliases": obs["aliases"],
                        "relation_digest": M.digest(reports),
                        "atomic_digest": M.digest(reports, atomic_only=True),
                    }
                )
        erased = [{**r, "episode_id": None} for r in records]
        unknown = [{**r, "actor": None} for r in records]
        other = [
            {
                **r,
                "actor": "B",
                "qualification": "other_report",
                "atom_id": "B:" + r["atom_id"],
                "episode_id": "B:" + r["episode_id"],
            }
            for r in records[:6]
        ]
        incomplete = M.observation(records[1:], kc)
        qualification.append(
            {
                "graph": graph["id"],
                "erased_episode_score": M.observation(erased, kc)["score"],
                "unknown_actor_score": M.observation(unknown, kc)["score"],
                "other_actor_added_delta": float(
                    np.max(abs(M.observation(records + other, kc)["score"] - base))
                ),
                "other_actor_atoms_excluded": len(
                    M.observation(records + other, kc)["excluded"]
                ),
                "incomplete_complete_episodes": len(incomplete["episodes"]),
                "incomplete_unresolved_atoms": len(incomplete["unresolved"]),
            }
        )
    H.table(folder / "packaging.csv", packaged)
    H.write(folder / "qualification.json", qualification)
    relations = []
    for b, t, (q, target) in itertools.product(range(2), range(4), enumerate(targets)):
        plus, actual_plus = accounts[f"b{b}:t{t}:law1"]
        minus, actual_minus = accounts[f"b{b}:t{t}:law-1"]
        matching = [g for g in graphs if g["bank"] == b and g["triple"] == t]
        same_atoms = M.digest(matching[0]["records"], True) == M.digest(
            matching[1]["records"], True
        )
        relations.append(
            {
                "bank": b,
                "triple": t,
                "target": target,
                "same_atomic_multiset": same_atoms,
                "different_episode_relations": M.digest(matching[0]["records"])
                != M.digest(matching[1]["records"]),
                "compiled_difference": float(plus[q] - minus[q]),
                "reference_difference": float(actual_plus[q] - actual_minus[q]),
            }
        )
    H.table(folder / "relations.csv", relations)
    bounds, certificates = [], []
    chi = M.SIGNS.prod(1)
    for b, t, (q, target) in itertools.product(range(2), range(4), enumerate(targets)):
        k = compiled[b, t, :, q]
        for scheme, degree, third in (
            ("unary", 1, None),
            ("pair", 2, None),
            ("third_minus", 3, -0.4),
            ("third_plus", 3, 0.4),
        ):
            if time.monotonic() - start > spec["max_seconds"]:
                raise TimeoutError("Original finite CPU cap reached; no reset")
            result = M.identified_bounds(k, degree, third)
            certificates.append(
                {"bank": b, "triple": t, "target": target, "scheme": scheme, **result}
            )
            bounds.append(
                {
                    "bank": b,
                    "triple": t,
                    "target": target,
                    "scheme": scheme,
                    "lower": result["lower"],
                    "upper": result["upper"],
                    "width": result["upper"] - result["lower"],
                    "rank": result["rank"],
                }
            )
        additive = k - chi * np.mean(chi * k)
        null = M.identified_bounds(additive, 2)
        certificates.append(
            {
                "bank": b,
                "triple": t,
                "target": target,
                "scheme": "third_removed_null",
                **null,
            }
        )
    H.table(folder / "identified_bounds.csv", bounds)
    H.write(folder / "lp_certificates.json", certificates)
    choice = []
    for t, law, reps in itertools.product(range(4), (-1, 1), ((3, 1), (1, 3))):
        rows = []
        for b, count in enumerate(reps):
            graph = next(
                g
                for g in graphs
                if g["bank"] == b and g["triple"] == t and g["law"] == law
            )
            for n in range(count):
                rows.extend(clones(graph["records"], f"c{b}:n{n}:"))
        obs = M.observation(rows, kc)
        for q, target in enumerate(targets):
            choice.append(
                {
                    "triple": t,
                    "law": law,
                    "target": target,
                    "positive_bank_choice": obs["choice"][1],
                    "bank0": float(obs["conditional"][0][q]),
                    "bank1": float(obs["conditional"][1][q]),
                    "score": float(obs["score"][q]),
                    "numeric_episodes": len(obs["episodes"]),
                }
            )
    H.table(folder / "choice.csv", choice)
    summary = {
        "designed_graph_laws": len(graphs),
        "reports_delivered": len(graphs) * len(spec["packages"]),
        "packaging_target_rows": len(packaged),
        "relation_target_rows": len(relations),
        "bound_rows": len(bounds),
        "LP_calls": 2 * len(certificates),
        "same_graph_max_packaging_delta": max(
            abs(r["packaging_delta"]) for r in packaged
        ),
        "arbitrary_zip_max_delta": max(abs(r["naive_minus_graph"]) for r in packaged),
        "graph_relation_RMS_compiled": [
            float(
                np.sqrt(
                    np.mean(
                        [
                            r["compiled_difference"] ** 2
                            for r in relations
                            if r["bank"] == b
                        ]
                    )
                )
            )
            for b in range(2)
        ],
        "graph_relation_RMS_reference": [
            float(
                np.sqrt(
                    np.mean(
                        [
                            r["reference_difference"] ** 2
                            for r in relations
                            if r["bank"] == b
                        ]
                    )
                )
            )
            for b in range(2)
        ],
        "pair_identification_width_RMS": float(
            np.sqrt(np.mean([r["width"] ** 2 for r in bounds if r["scheme"] == "pair"]))
        ),
        "third_completed_max_width": max(
            abs(r["width"]) for r in bounds if r["scheme"].startswith("third")
        ),
        "third_removed_null_max_width": max(
            abs(r["upper"] - r["lower"])
            for r in certificates
            if r["scheme"] == "third_removed_null"
        ),
        "new_neural_calls": 0,
        "fits": 0,
        "not_personality_accuracy": True,
        "supplied_event_identity_not_discovered": True,
        "original_scope_complete": False,
        "CPU_seconds": time.monotonic() - start,
    }
    H.write(folder / "summary.json", summary)
    H.write(folder / "output_seal.json", H.seals(out, folder))
    print(summary)


def verify(out):
    """Independently replay explicit counts, relation moments and LP witnesses."""
    spec = check(out)["spec"]
    H.verify_hashes(out, H.load(out / "01/output_seal.json"))
    targets, graphs = H.load(out / "00/targets.json"), H.load(out / "00/graphs.json")
    with np.load(out / "00/kernels.npz", allow_pickle=False) as z:
        kernels, refs = z["compiled"], z["reference"]
    lookup = kernel_map(kernels, spec["triples"])
    reference_lookup = kernel_map(refs, spec["triples"])
    maximum = 0.0
    with (out / "01/packaging.csv").open() as stream:
        package_rows = list(csv.DictReader(stream))
    for row in package_rows:
        reports = H.load(
            out
            / f"01/reports/g{int(row['graph_index']):02d}_p{row['package_index']}.json"
        )
        atoms = {r["atom_id"]: r for r in reports}
        groups = defaultdict(dict)
        for r in atoms.values():
            groups[r["episode_id"]][r["slot"]] = r
        values = [
            lookup[(group[0]["condition"], tuple(group[j]["level"] for j in range(6)))]
            for group in groups.values()
        ]
        q = targets.index(row["target"])
        maximum = max(
            maximum,
            abs(sum(v[q] for v in values) / len(values) - float(row["compiled"])),
        )
        reference_values = [
            reference_lookup[
                (group[0]["condition"], tuple(group[j]["level"] for j in range(6)))
            ]
            for group in groups.values()
        ]
        streams = defaultdict(list)
        for atom in atoms.values():
            streams[atom["slot"]].append(atom["level"])
        synthetic = [
            lookup[(next(iter(atoms.values()))["condition"], tuple(levels))][q]
            for levels in zip(*(streams[j] for j in range(6)), strict=True)
        ]
        maximum = max(
            maximum,
            abs(
                sum(v[q] for v in reference_values) / len(reference_values)
                - float(row["reference"])
            ),
            abs(sum(synthetic) / len(synthetic) - float(row["order_zip"])),
        )
    chi = np.array([(-1) ** (3 - int(np.sum(s == 1))) for s in M.SIGNS])
    with (out / "01/relations.csv").open() as stream:
        for row in csv.DictReader(stream):
            b, t, q = int(row["bank"]), int(row["triple"]), targets.index(row["target"])
            if (
                row["same_atomic_multiset"] != "True"
                or row["different_episode_relations"] != "True"
            ):
                raise ValueError("Matched atom and changed relation checks failed")
            maximum = max(
                maximum,
                abs(
                    0.8 * sum(chi * kernels[b, t, :, q]) / 8
                    - float(row["compiled_difference"])
                ),
                abs(
                    0.8 * sum(chi * refs[b, t, :, q]) / 8
                    - float(row["reference_difference"])
                ),
            )
    for graph in graphs:
        records = graph["records"]
        joined = defaultdict(dict)
        for r in records:
            joined[r["episode_id"]][r["slot"]] = r["level"]
        signs = np.array(
            [
                [g[j] - 1 for j in spec["triples"][graph["triple"]]]
                for g in joined.values()
            ]
        )
        if not np.all(signs.mean(0) == 0) or not np.all(
            (signs.T @ signs / 40 - np.eye(3)) == 0
        ):
            raise ValueError("Unary/pair moment equality failed")
        maximum = max(
            maximum, abs(float(np.prod(signs, axis=1).mean()) - 0.4 * graph["law"])
        )
    certificates = H.load(out / "01/lp_certificates.json")
    for cert in certificates:
        b, t, q = cert["bank"], cert["triple"], targets.index(cert["target"])
        k = kernels[b, t, :, q]
        if cert["scheme"] == "third_removed_null":
            k = k - chi * np.dot(chi, k) / 8
        a, rhs = np.array(cert["a"]), np.array(cert["b"])
        for direction, witness in zip((1, -1), cert["witnesses"], strict=True):
            p, dual = np.array(witness["p"]), np.array(witness["dual"])
            slack = direction * k - a.T @ dual
            if min(p) < -spec["identity_atol"] or min(slack) < -spec["identity_atol"]:
                raise ValueError("LP primal/dual positivity failed")
            maximum = max(
                maximum,
                float(np.max(abs(a @ p - rhs))),
                abs(float(k @ p) - witness["value"]),
                abs(float(direction * k @ p - rhs @ dual)),
            )
        if cert["scheme"] == "pair":
            mean, effect = sum(k) / 8, abs(sum(chi * k) / 8)
            maximum = max(
                maximum,
                abs(cert["lower"] - mean + effect),
                abs(cert["upper"] - mean - effect),
            )
        elif (
            cert["scheme"].startswith("third_")
            and cert["scheme"] != "third_removed_null"
        ):
            law = -1 if cert["scheme"] == "third_minus" else 1
            explicit = (
                sum(
                    float(value) * (5 + 2 * law * int(c))
                    for value, c in zip(k, chi, strict=True)
                )
                / 40
            )
            maximum = max(
                maximum, abs(cert["lower"] - explicit), abs(cert["upper"] - explicit)
            )
    with (out / "01/choice.csv").open() as stream:
        for row in csv.DictReader(stream):
            mass = float(row["positive_bank_choice"])
            maximum = max(
                maximum,
                abs(
                    float(row["score"])
                    - (1 - mass) * float(row["bank0"])
                    - mass * float(row["bank1"])
                ),
            )
    summary = H.load(out / "01/summary.json")
    qual = H.load(out / "01/qualification.json")
    if any(
        r["erased_episode_score"] is not None
        or r["unknown_actor_score"] is not None
        or r["other_actor_added_delta"] > spec["identity_atol"]
        for r in qual
    ):
        raise ValueError("Unknown/other-actor routing failed")
    if (
        maximum > spec["identity_atol"]
        or summary["same_graph_max_packaging_delta"] > spec["identity_atol"]
    ):
        raise ValueError("Explicit integration/certificate mismatch")
    if np.max(abs(kernels - refs)) > spec["reference_atol"]:
        raise ValueError("Inherited kernel construction gate failed")
    result = {
        "status": "VERIFIED_FINITE_HM113_NOT_GLOBAL",
        "alternate_max_error": maximum,
        "packaged_target_rows_replayed": len(package_rows),
        "LP_witnesses_checked": 2 * len(certificates),
        "unknown_link_not_zero": True,
        "original_scope_complete": False,
        "same_agent_alternate_formula_not_third_party": True,
    }
    path = out / "verification.json"
    if path.exists():
        if H.load(path) != result:
            raise ValueError("Existing verification differs")
    else:
        H.write(path, result)
    print(result)


def main():
    """Require an explicit finite phase and non-overwriting output address."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "verify"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if args.phase == "prepare":
        prepare(out, H.load(SPEC))
    elif args.phase == "run":
        run(out)
    else:
        verify(out)


if __name__ == "__main__":
    main()
