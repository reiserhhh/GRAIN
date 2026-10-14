#!/usr/bin/env python3
"""Execute a frozen-source group/evidence-locality research packet offline."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json

sys.path.insert(0, str(ROOT))
from grain_research.construction import math_local as L
from grain_research.construction.math_response import bootstrap_interval

DEFAULT_OUT = ROOT / "results/grain_math_response_v1/local_r1"
TARGETS = ("coordination_action", "verification_action")


def material_text(core, target, bits):
    """Render a declared four-factor evidence event plus a register operation."""
    k, p, o, a, s = bits
    base = core[2 + 2 * target + (k == -1)]
    action = (
        ("did " if p == 1 else "did not ")
        if a == 1
        else ("would " if p == 1 else "would not ")
    )
    premise = "In the completed event, " if a == 1 else "If this happened next week, "
    event = premise + "I " + action + base + "."
    account = (
        "My own account is: " + event
        if o == 1
        else "My colleague's account is: '" + event + "' My own action is not reported."
    )
    return (
        ("Record of the event. " if s == 1 else "Here is the story. ")
        + core[1]
        + " "
        + account
        + (" End of the record." if s == 1 else " That is what was described.")
    )


def prepare(out):
    """Freeze public materials and source roles before any vector readout."""
    cores_path = ROOT / "configs/grain_math_local_families_v1.json"
    cores = json.loads(cores_path.read_text())
    order = np.random.default_rng(108).permutation(len(cores))
    roles = {
        int(f): "source" if i < 10 else "validation" if i < 16 else "test"
        for i, f in enumerate(order)
    }
    bank, index, mixtures = [], {}, []
    for target, name in enumerate(TARGETS):
        for f, core in enumerate(cores):
            for bits in product((-1, 1), repeat=5):
                index[(target, f, bits)] = len(bank)
                bank.append(
                    {
                        "material_id": f"t{target}_f{f:02d}_"
                        + "".join(str((b + 1) // 2) for b in bits),
                        "target": name,
                        "family": f,
                        "role": roles[f],
                        "bits": list(bits),
                        "text": material_text(core, target, bits),
                        "kind": "single",
                        "event_id": f"t{target}_f{f:02d}",
                    }
                )
    for target, name in enumerate(TARGETS):
        for f in sorted(k for k, v in roles.items() if v == "test"):
            for p in (-1, 1):
                base = index[(target, f, (1, p, 1, 1, 1))]
                additions = {
                    "other": index[(target, f, (1, -p, -1, 1, 1))],
                    "irrelevant": index[(target, f, (-1, -p, 1, 1, 1))],
                    "hypothetical": index[(target, f, (1, -p, 1, -1, 1))],
                    "alias": base,
                    "new_own_event": index[(target, f, (1, -p, 1, 1, 1))],
                }
                for kind, added in additions.items():
                    row = {
                        "material_id": f"t{target}_f{f:02d}_{p}_{kind}",
                        "target": name,
                        "family": f,
                        "role": "composition",
                        "bits": [0] * 5,
                        "kind": kind,
                        "text": bank[base]["text"]
                        + "\nAnother report follows.\n"
                        + bank[added]["text"],
                    }
                    mix = {
                        "row": len(bank),
                        "base": base,
                        "added": added,
                        "same_event": kind == "alias",
                        "kind": kind,
                        "target": name,
                        "family": f,
                        "expected_delta": -p if kind == "new_own_event" else 0,
                    }
                    bank.append(row)
                    mixtures.append(mix)
    source_files = [
        Path(__file__),
        ROOT / "grain_research/construction/math_local.py",
        cores_path,
    ]
    signature = {
        "code": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        "prior_reader_registry": sha(
            ROOT / "results/grain_math_response_v1/r1/00/reader_registry.json"
        ),
        "scope": "HM-01",
        "split_seed": 108,
    }
    sig_path = out / "00/protocol_before_run.json"
    if sig_path.exists():
        old = json.loads(sig_path.read_text())
        if old["signature"] != signature:
            raise ValueError(
                "changed protocol/code: preserve old output and use a new directory"
            )
        if sha(out / "00/material_bank.jsonl") != old["bank_sha256"]:
            raise ValueError("frozen bank hash mismatch")
        return
    dest = out / "00"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "material_bank.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in bank)
    )
    write_json(dest / "mixtures.json", mixtures)
    write_json(
        dest / "reader_registry.json",
        json.loads(
            (
                ROOT / "results/grain_math_response_v1/r1/00/reader_registry.json"
            ).read_text()
        ),
    )
    for p in source_files:
        snap = dest / "code_snapshot" / p.relative_to(ROOT)
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(p.read_bytes())
    write_json(
        sig_path,
        {
            "signature": signature,
            "bank_sha256": sha(dest / "material_bank.jsonl"),
            "single_rows": 1536,
            "composition_rows": len(mixtures),
            "families_per_target": 24,
            "split": [10, 6, 8],
            "qualification": "own, actual, relevant; unobserved is null",
            "fit": "source only, validation hyperparameters, source LOFO calibration",
            "arms": [
                "joint",
                "additive",
                "declared_product",
                "direct_ridge",
                "rbf",
                "permutation",
                "half_design_joint",
            ],
            "result_guided": True,
            "new_encodings_budget": 2 * len(bank),
            "api_calls": 0,
            "neural_training": 0,
        },
    )
    print(
        f"prepared {len(bank)} public texts, source/validation/test=10/6/8 per target",
        flush=True,
    )


def save_program(path, program):
    """Save all cold coefficients as non-object NumPy arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            group + "__" + k: v
            for group, model in program.items()
            for k, v in model.items()
        },
    )


def load_program(path):
    """Load a frozen cold program without material metadata or evaluation labels."""
    result = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            group, name = key.split("__", 1)
            result.setdefault(group, {})[name] = data[key]
    return result


def fit_all(out, bank):
    """Freeze every model before any held-family metric is calculated."""
    manifest = out / "02/model_freeze.json"
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text())["bundles"].items():
            if sha(out / "02/bundles" / name) != digest:
                raise ValueError("bundle integrity failure")
        return
    info = {}
    for reader in MODELS:
        vectors = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in TARGETS:
            mask = np.array(
                [r["target"] == target and r["kind"] == "single" for r in bank]
            )
            rows = [r for r, m in zip(bank, mask, strict=True) if m]
            bits = np.array([r["bits"] for r in rows])
            family = np.array([r["family"] for r in rows])
            role = np.array([r["role"] for r in rows])
            for half in (False, True):
                key = f"{reader}_{target}" + ("_half" if half else "")
                program, info[key] = L.fit_programs(
                    vectors[mask], bits, family, role, half
                )
                save_program(out / f"02/bundles/{key}.npz", program)
                print(f"fitted {key}: source={info[key]['source_rows']}", flush=True)
    write_json(out / "02/source_fits.json", info)
    write_json(
        manifest,
        {
            "source_only": True,
            "test_selection": False,
            "bundles": {
                p.name: sha(p)
                for p in (out / "02/bundles").glob("*.npz")
                if not p.name.startswith("._")
            },
        },
    )


def evaluate(out, bank):
    """Score frozen programs once; retain numerical failures and all controls."""
    if (out / "03/summary.json").exists():
        return
    rows, by_family, effects, group_rows, compound_rows = [], [], [], [], []
    transport = []
    mixtures = json.loads((out / "00/mixtures.json").read_text())
    for reader in MODELS:
        x = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in TARGETS:
            singles = np.array(
                [r["target"] == target and r["kind"] == "single" for r in bank]
            )
            rr = [r for r, m in zip(bank, singles, strict=True) if m]
            bits, family, role = (
                np.array([r[k] for r in rr]) for k in ("bits", "family", "role")
            )
            program = load_program(out / f"02/bundles/{reader}_{target}.npz")
            scores, z_all = L.score_programs(program, x)
            z = z_all[singles]
            held = role == "test"
            for arm, values in scores.items():
                y = values[singles]
                rows.append(
                    {
                        "reader": reader,
                        "target": target,
                        "arm": arm,
                        **L.metrics(y[held], z[held], bits[held]),
                    }
                )
                for f in np.unique(family[held]):
                    m = held & (family == f)
                    by_family.append(
                        {
                            "reader": reader,
                            "target": target,
                            "arm": arm,
                            "family": int(f),
                            "mse": float(
                                np.mean(
                                    (y[m] - L.ideal_contribution(bits[m])[:, 0]) ** 2
                                )
                            ),
                        }
                    )
            half = load_program(out / f"02/bundles/{reader}_{target}_half.npz")
            half_scores, half_z = L.score_programs(half, x[singles])
            rows.append(
                {
                    "reader": reader,
                    "target": target,
                    "arm": "half_design_joint",
                    **L.metrics(half_scores["joint"][held], half_z[held], bits[held]),
                }
            )
            for item in L.group_diagnostics(z, bits, family, role):
                group_rows.append({"reader": reader, "target": target, **item})
            for f in np.unique(family[held]):
                for arm in scores:
                    m = held & (family == f)
                    b, y = bits[m], scores[arm][singles][m]
                    coeff = (
                        np.column_stack([np.ones(len(b)), L.characters(b[:, :4])]).T
                        @ y
                        / len(b)
                    )
                    for character, value in enumerate(coeff):
                        effects.append(
                            {
                                "reader": reader,
                                "target": target,
                                "arm": arm,
                                "family": int(f),
                                "character_mask": character,
                                "coefficient": float(value),
                            }
                        )
            decisions = L.qualify(z_all)
            for m in (m for m in mixtures if m["target"] == target):
                ids = [m["base"], m["added"]]
                events = [
                    {"event_id": "base"},
                    {"event_id": "base" if m["same_event"] else "added"},
                ]
                local = L.local_profile(events, scores["joint"][ids], decisions[ids])
                base = L.local_profile(
                    events[:1], scores["joint"][ids[:1]], decisions[ids[:1]]
                )
                # Declared-role routing is a separate upper bound, not a learned parser.
                q = [
                    bool(L.ideal_contribution(np.array([bank[i]["bits"]]))[0, 0])
                    for i in ids
                ]
                oracle = L.local_profile(events, scores["joint"][ids], np.array(q))
                oracle_base = L.local_profile(
                    events[:1], scores["joint"][ids[:1]], np.array(q[:1])
                )
                compound_rows.append(
                    {
                        "reader": reader,
                        "target": target,
                        **m,
                        "global_delta": float(
                            scores["joint"][m["row"]] - scores["joint"][m["base"]]
                        ),
                        "local_delta": local["score"] - base["score"]
                        if local["score"] is not None and base["score"] is not None
                        else None,
                        "local_observed_events": local["observed_events"],
                        "declared_routing_delta": oracle["score"] - oracle_base["score"]
                        if oracle["score"] is not None
                        and oracle_base["score"] is not None
                        else None,
                    }
                )
            dest = out / "03"
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dest / f"cold_scores_{reader}_{target}.npz", factors=z_all, **scores
            )
            for condition in ("source", "validation", "test"):
                m = role == condition
                residual = z[m] - bits[m, :4]
                transport.append(
                    {
                        "reader": reader,
                        "target": target,
                        "domain": condition,
                        "coordinate_offset": residual.mean(axis=0).tolist(),
                        "coordinate_rmse": np.sqrt(
                            np.mean(residual**2, axis=0)
                        ).tolist(),
                    }
                )
    comparisons = []
    for reader, target in product(MODELS, TARGETS):
        rr = [r for r in by_family if r["reader"] == reader and r["target"] == target]
        joint = {r["family"]: r["mse"] for r in rr if r["arm"] == "joint"}
        for arm in ("additive", "direct_ridge", "rbf", "declared_product"):
            delta = np.array(
                [r["mse"] - joint[r["family"]] for r in rr if r["arm"] == arm]
            )
            comparisons.append(
                {
                    "reader": reader,
                    "target": target,
                    "comparator": arm,
                    "joint_mse_improvement": float(delta.mean()),
                    "family_ci": bootstrap_interval(delta),
                    "families": len(delta),
                }
            )
    write_csv(out / "03/metrics.csv", rows)
    write_csv(out / "03/by_family.csv", by_family)
    write_csv(out / "03/character_effects.csv", effects)
    write_csv(out / "03/compound_effects.csv", compound_rows)
    write_csv(out / "03/group_defects.csv", group_rows)
    write_csv(out / "03/observation_domains.csv", transport)
    write_csv(out / "03/comparisons.csv", comparisons)
    write_json(
        out / "03/summary.json",
        {
            "metrics": rows,
            "comparisons": comparisons,
            "combinatorial_witness": L.combinatorial_probe(),
        },
    )


def verify(out, bank):
    """Numerically reconstruct every frozen score with separate explicit formulas."""
    maximum = 0.0
    for reader, target in product(MODELS, TARGETS):
        x = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        program = load_program(out / f"02/bundles/{reader}_{target}.npz")
        with np.load(out / f"03/cold_scores_{reader}_{target}.npz") as expected:
            f = program["factors"]
            z = x @ f["w"] + (f["bias"] - f["mu"] @ f["w"])
            maximum = max(maximum, float(np.max(abs(z - expected["factors"]))))
            for arm in ("joint", "additive"):
                model = program[arm]
                phi = (
                    z
                    if arm == "additive"
                    else np.column_stack(
                        [
                            np.prod(z[:, [j for j in range(4) if m & (1 << j)]], axis=1)
                            for m in range(1, 16)
                        ]
                    )
                )
                y = phi @ model["w"] + (model["bias"] - model["mu"] @ model["w"])
                maximum = max(maximum, float(np.max(abs(y[:, 0] - expected[arm]))))
            for arm in ("direct_ridge", "permutation"):
                model = program[arm]
                y = x @ model["w"] + (model["bias"] - model["mu"] @ model["w"])
                maximum = max(maximum, float(np.max(abs(y[:, 0] - expected[arm]))))
            km = program["rbf"]
            # A direct difference tensor is an independent kernel expression.
            for start in range(0, len(x), 32):
                squared = np.sum(
                    (x[start : start + 32, None] - km["reference"][None]) ** 2, axis=-1
                )
                y = np.exp(-float(km["gamma"]) * squared) @ km["dual"] + km["center"]
                maximum = max(
                    maximum,
                    float(np.max(abs(y[:, 0] - expected["rbf"][start : start + 32]))),
                )
    for target in TARGETS:
        rr = [r for r in bank if r["target"] == target and r["kind"] == "single"]
        sets = {
            role: {r["family"] for r in rr if r["role"] == role}
            for role in ("source", "validation", "test")
        }
        if any(sets[a] & sets[b] for a, b in product(sets, repeat=2) if a != b):
            raise AssertionError("family overlap")
    if maximum > 1e-10:
        raise AssertionError(f"numerical replay defect {maximum}")
    write_json(
        out / "04/verification.json",
        {
            "status": "PASS",
            "max_replay_error": maximum,
            "independent_human_review": False,
            "replay": "same-agent alternate algebra, all four model bundles",
        },
    )
    paths = sorted(
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name not in ("artifact_hashes.json", "execution_progress.json")
    )
    write_json(
        out / "04/artifact_hashes.json",
        {str(p.relative_to(out)): sha(p) for p in paths},
    )
    write_json(
        out / "execution_progress.json",
        {
            "scope": "HM-01",
            "status": "FINITE_PACKET_VERIFIED",
            "goal_complete": False,
            "old_plans_closed": False,
            "evidence": "03/summary.json, 04/verification.json",
        },
    )
    print(f"verified all cold scores: max error {maximum:.3g}", flush=True)


def main():
    """Run a finite stage or a full restartable packet without outside services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "encode", "run", "verify", "all"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    prepare(args.out)
    bank = [
        json.loads(s)
        for s in (args.out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    if args.stage in ("encode", "all"):
        for reader in MODELS:
            encode(args.out, reader, args.device)
    if args.stage in ("run", "all"):
        fit_all(args.out, bank)
        evaluate(args.out, bank)
    if args.stage in ("verify", "all"):
        verify(args.out, bank)


if __name__ == "__main__":
    main()
