#!/usr/bin/env python3
"""HM-05: cold cross-target role transfer with frozen relevance and no refits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_local_roles import value_profile, vectors
from run_grain_math_response import ROOT, sha, write_csv, write_json

from grain_research.construction import math_binding as B
from grain_research.construction import math_local_roles as R
from grain_research.construction import math_role_sharing as S

SPEC = ROOT / "configs/grain_math_role_sharing_v1.json"
DEFAULT = ROOT / "results/grain_math_response_v1/role_sharing_r1"


def freeze(out):
    """Bind the finite sharing experiment to immutable programs and observations."""
    spec = json.loads(SPEC.read_text())
    source, application = (
        ROOT / spec["source_packet"],
        ROOT / spec["application_packet"],
    )
    signature = {}
    hash_count = 0
    for root in (source, application):
        path = root / "04/artifact_hashes.json"
        signature[str(root.relative_to(ROOT))] = sha(path)
        for name, digest in json.loads(path.read_text()).items():
            if sha(root / name) != digest:
                raise ValueError(f"frozen input changed: {root / name}")
            hash_count += 1
    code = [
        Path(__file__),
        SPEC,
        ROOT / "grain_research/construction/math_role_sharing.py",
        ROOT / "grain_research/construction/math_local_roles.py",
        ROOT / "grain_research/construction/math_binding.py",
        ROOT / "grain_research/construction/math_local.py",
        ROOT / "scripts/run_grain_math_local_roles.py",
        ROOT / "scripts/run_grain_math_response.py",
        ROOT / "scripts/run_grain_math_binding.py",
    ]
    signature.update({str(p.relative_to(ROOT)): sha(p) for p in code})
    protocol = out / "00/protocol_before_run.json"
    if protocol.exists():
        if json.loads(protocol.read_text())["signature"] != signature:
            raise ValueError("sharing protocol changed; retain prior output")
    else:
        for p in code:
            dest = out / "00/code" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        write_json(
            protocol,
            {"spec": spec, "signature": signature, "input_hashes_verified": hash_count},
        )
    return spec, source, application


def evaluate(out, spec, source, application):
    """Borrow roles on the same event vectors; retain each target's relevance."""
    import csv

    bank = read_bank(application / "00/documents.jsonl")
    refs = json.loads((application / "01/observation_indices.json").read_text())
    targets = sorted({r["target"] for r in bank})
    if len(targets) != 2 or spec["shared_weight"] != 0.5:
        raise ValueError(
            "this finite two-target protocol has no tunable mixing parameter"
        )
    with (application / "02/predictions.csv").open() as stream:
        replay = {
            (r["reader"], r["material_id"]): r
            for r in csv.DictReader(stream)
            if r["arm"] == "automatic_local"
        }
    results, role_rows, errors, disagreements = [], [], [], []
    for reader in ("e5", "bge"):
        models = {t: load_model(source / f"02/{reader}_{t}.npz") for t in targets}
        common, difference = S.split_common_roles(
            *(models[t]["whole_factors"] for t in targets)
        )
        np.savez_compressed(
            out / f"00/{reader}_shared_roles.npz",
            **{
                f"{g}__{k}": v
                for g, p in (("common", common), ("difference", difference))
                for k, v in p.items()
            },
        )
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(application / f"01B/embeddings/{reader}.npy").astype(float)
        flat = [e for events in refs["automatic_events"] for e in events]
        x = vectors([e["readout"] for e in flat], old, new)
        ax, bx = (vectors([e[k] for e in flat], old, new) for k in ("action", "object"))
        roles = {
            t: S.score_roles(S.affine_roles(models[t]["whole_factors"]), x)
            for t in targets
        }
        shared, delta = S.score_roles(common, x), S.score_roles(difference, x)
        errors.extend(
            [
                float(np.max(abs(shared + delta - roles[targets[0]]))),
                float(np.max(abs(shared - delta - roles[targets[1]]))),
            ]
        )
        kernels = {t: B.score_kernel(models[t]["auto_tensor"], ax, bx) for t in targets}
        cursor = 0
        for row, events in zip(bank, refs["automatic_events"], strict=True):
            indices = np.arange(cursor, cursor + len(events))
            cursor += len(events)
            target = row["target"]
            other = next(t for t in targets if t != target)
            valid = np.array(
                [e["status"] == "ok" and e["event_id"] is not None for e in events]
            )
            ids = [
                {"event_id": e["event_id"]}
                for e, yes in zip(events, valid, strict=True)
                if yes
            ]
            unknown = R.missing_records(events, len(row["events"]))
            truth = R.true_profile(row)
            readouts = {
                "native_roles": roles[target][indices],
                "cross_target_roles": roles[other][indices],
                "equal_shared_roles": shared[indices],
            }
            for arm, role_values in readouts.items():
                z = np.column_stack((kernels[target][indices], role_values))
                observed, _ = value_profile(ids, z[valid], unresolved=unknown)
                results.append(
                    {
                        "reader": reader,
                        "target": target,
                        "style": row["style"],
                        "kind": row["kind"],
                        "family": row["family"],
                        "material_id": row["material_id"],
                        "arm": arm,
                        "ambiguous": row["ambiguous"],
                        "complete": observed["complete"],
                        "unit": observed["unit"],
                        "prediction": observed["score"],
                        "present": bool(observed["observed_events"])
                        if observed["complete"]
                        else None,
                        "expected_unit": truth["unit"],
                        "expected_present": bool(truth["observed_events"]),
                        "expected_score": truth["score"],
                    }
                )
                if arm == "native_roles":
                    previous = replay[reader, row["material_id"]]
                    if bool(previous["unit"]) != (observed["unit"] is not None):
                        raise AssertionError(
                            "native replay observation coverage changed"
                        )
                    if observed["unit"] is not None:
                        errors.append(abs(observed["unit"] - float(previous["unit"])))
                if not row["ambiguous"]:
                    for j, e in enumerate(events):
                        if not valid[j]:
                            continue
                        matched = [
                            r for r in row["events"] if r["event_id"] == e["event_id"]
                        ]
                        if not matched:
                            continue
                        expected = matched[0]["bits"][1:]
                        for k, name in enumerate(
                            ("polarity", "ownership", "actuality")
                        ):
                            role_rows.append(
                                {
                                    "reader": reader,
                                    "target": target,
                                    "kind": row["kind"],
                                    "arm": arm,
                                    "role": name,
                                    "material_id": row["material_id"],
                                    "value": float(role_values[j, k]),
                                    "expected": expected[k],
                                }
                            )
            if not row["ambiguous"]:
                disagreements.append(
                    {
                        "reader": reader,
                        "target": target,
                        "material_id": row["material_id"],
                        "kind": row["kind"],
                        "role_rms_disagreement": float(
                            np.sqrt(
                                np.mean(
                                    (roles[target][indices] - roles[other][indices])
                                    ** 2
                                )
                            )
                        ),
                    }
                )
    if max(errors) > 1e-10:
        raise AssertionError(max(errors))
    write_csv(out / "01/predictions.csv", results)
    write_csv(out / "01/role_readouts.csv", role_rows)
    write_csv(out / "01/disagreements.csv", disagreements)
    write_json(
        out / "02/algebra_replay.json",
        {
            "status": "PASS",
            "max_error": max(errors),
            "comparisons": len(errors),
            "scope": "compiled common plus difference and original HM-04 native units",
        },
    )
    return bank, results, role_rows


def summarize(out, bank, predictions, roles):
    """Report transport and its loss; no best-arm selection is performed."""
    groups, metrics, role_metrics = defaultdict(list), [], []
    for p in predictions:
        if not p["ambiguous"]:
            groups[p["reader"], p["target"], p["style"], p["kind"], p["arm"]].append(p)
    for key, rows in groups.items():
        complete = [r for r in rows if r["complete"]]
        eligible = [
            r
            for r in complete
            if r["prediction"] is not None and r["expected_score"] is not None
        ]
        metrics.append(
            {
                **dict(
                    zip(("reader", "target", "style", "kind", "arm"), key, strict=True)
                ),
                "n": len(rows),
                "complete": len(complete),
                "unit_rmse": float(
                    np.sqrt(
                        np.mean(
                            [(r["unit"] - r["expected_unit"]) ** 2 for r in complete]
                        )
                    )
                )
                if complete
                else None,
                "tp": sum(r["present"] and r["expected_present"] for r in complete),
                "fp": sum(r["present"] and not r["expected_present"] for r in complete),
                "fn": sum(not r["present"] and r["expected_present"] for r in complete),
                "tn": sum(
                    not r["present"] and not r["expected_present"] for r in complete
                ),
                "profile_rmse_joint_observed": float(
                    np.sqrt(
                        np.mean(
                            [
                                (r["prediction"] - r["expected_score"]) ** 2
                                for r in eligible
                            ]
                        )
                    )
                )
                if eligible
                else None,
            }
        )
    groups.clear()
    for r in roles:
        if r["kind"] == "mixture":
            groups[r["reader"], r["target"], r["role"], r["arm"]].append(r)
    for key, rows in groups.items():
        role_metrics.append(
            {
                **dict(zip(("reader", "target", "role", "arm"), key, strict=True)),
                "n": len(rows),
                "rmse": float(
                    np.sqrt(np.mean([(r["value"] - r["expected"]) ** 2 for r in rows]))
                ),
                "direction_correct": sum(
                    (r["value"] > 0) == (r["expected"] > 0) for r in rows
                ),
            }
        )
    lookup = {(r["reader"], r["material_id"], r["arm"]): r for r in predictions}
    comparisons, rng = [], np.random.default_rng(5108)
    for reader in ("e5", "bge"):
        for target in sorted({r["target"] for r in bank}):
            chosen = [
                r for r in bank if r["target"] == target and r["kind"] == "mixture"
            ]
            for arm in ("cross_target_roles", "equal_shared_roles"):
                diffs, families = [], []
                for row in chosen:
                    a, b = (
                        lookup[reader, row["material_id"], "native_roles"],
                        lookup[reader, row["material_id"], arm],
                    )
                    if not a["complete"] or not b["complete"]:
                        continue
                    diffs.append(
                        (b["unit"] - b["expected_unit"]) ** 2
                        - (a["unit"] - a["expected_unit"]) ** 2
                    )
                    families.append(row["family"])
                d, f = np.array(diffs), np.array(families)
                samples = [
                    float(
                        np.concatenate(
                            [
                                d[f == v]
                                for v in rng.choice(np.unique(f), len(np.unique(f)))
                            ]
                        ).mean()
                    )
                    for _ in range(1000)
                ]
                comparisons.append(
                    {
                        "reader": reader,
                        "target": target,
                        "arm": arm,
                        "n": len(d),
                        "mse_cost_vs_native": float(d.mean()),
                        "fixed_family_ci": np.quantile(
                            samples, [0.025, 0.975]
                        ).tolist(),
                    }
                )
    write_csv(out / "02/metrics.csv", metrics)
    write_csv(out / "02/role_metrics.csv", role_metrics)
    write_csv(out / "02/comparisons.csv", comparisons)
    write_json(
        out / "02/summary.json",
        {
            "metrics": metrics,
            "role_metrics": role_metrics,
            "comparisons": comparisons,
            "scope": "two frozen targets in one reader basis; transport success is not universality or personality validity",
        },
    )


def main():
    """Execute this three-arm parameter transport, without re-encoding or fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    spec, source, application = freeze(args.out)
    bank, predictions, roles = evaluate(args.out, spec, source, application)
    summarize(args.out, bank, predictions, roles)
    write_json(
        args.out / "execution_progress.json",
        {
            "packet": "HM-05",
            "status": "finite_packet_verified",
            "goal_complete": False,
            "new_encodings": 0,
            "new_fits": 0,
            "private_reads": 0,
            "api_calls": 0,
        },
    )
    write_json(
        args.out / "02/artifact_hashes.json",
        {
            str(p.relative_to(args.out)): sha(p)
            for p in args.out.rglob("*")
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    print(
        "HM-05 finite cross-target role transport finished; zero new fits/encodings",
        flush=True,
    )


if __name__ == "__main__":
    main()
