#!/usr/bin/env python3
"""HM74: finite character ablations at one complete contextualized query cut."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_causal_ownership_sectors as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_heterogeneous_evidence_cube import decode
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_ownership_composition import CODE as OLD_CODE
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/causal_ownership_sectors_r1"
SPEC = ROOT / "configs/grain_causal_ownership_sectors_hm74.json"
REFERENCE = ROOT / "results/grain_math_response_v1/ownership_composition_r1"
PARENT = ROOT / "results/grain_math_response_v1/equal_information_target_actions_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_causal_ownership_sectors_hm74.json",
            "docs/GRAIN_CAUSAL_OWNERSHIP_SECTORS_PLAN_HM74.md",
            "grain_research/construction/math_causal_ownership_sectors.py",
            "scripts/run_grain_causal_ownership_sectors.py",
            "scripts/verify_grain_causal_ownership_sectors.py",
            "tests/test_grain_causal_ownership_sectors.py",
            "grain_research/construction/math_target_bound_actions.py",
            *OLD_CODE,
        )
    )
)
STAGES = ("score", "log_odds", "centered_logits")


def unit_path(out, unit):
    """Address the same raw acquisition unit in every phase."""
    return out / f"01/units/unit_{unit:04d}.json"


def payload_path(out, cube, arm):
    """Address a pre-outcome sealed counterfactual assignment."""
    return out / f"02/payloads/cube_{cube:03d}_{arm}.json"


def prepare(out):
    """Freeze new material, actual token proofs, fixed arms and local dependencies."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve existing versions; no overwrite")
    spec = read_json(SPEC)
    if spec["arms"] != list(M.ARMS) or spec["layer"] != 16:
        raise ValueError("Fixed arms and layer required")
    if (
        read_json(PARENT / "verification.json")["status"]
        != "VERIFIED_FINITE_HM73_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM73 required")
    old = read_json(REFERENCE / "00/manifest.json")
    if (
        spec["system"]
        != read_json(ROOT / "configs/grain_math_passive_event_composition_v1.json")[
            "system"
        ]
    ):
        raise ValueError("Observer system must not be retuned")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    alphabet = old["alphabet"]
    if [tokenizer.encode(s, add_special_tokens=False) for s in alphabet["labels"]] != [
        [i] for i in alphabet["token_ids"]
    ]:
        raise ValueError("Candidate alphabet drift")
    rows, groups = [], {}
    for doc in M.documents(spec):
        queries = (
            (doc["focal"],)
            if doc["companion"] == "neutral"
            else (doc["focal"], doc["companion"])
        )
        prefix = None
        for query in queries:
            key = (doc["focal"], doc["companion"], query)
            cube = groups.setdefault(key, len(groups))
            text, ids, part, _, _ = render(
                doc["body"], query, "A", 1, spec["system"], tokenizer
            )
            current = ids[: part["body"][-1] + 1]
            if prefix is not None and prefix != current:
                raise ValueError("Evidence prefix depends on future query")
            prefix = current
            rows.append(
                {
                    **doc,
                    "receiver": len(rows),
                    "cube": cube,
                    "query": query,
                    "input_ids": ids,
                    "rendered": text,
                    "partition": part,
                }
            )
    cubes = []
    proofs = []
    for key, cube in groups.items():
        selected = sorted(
            (r for r in rows if r["cube"] == cube), key=lambda r: r["corner"]
        )
        suffix_ids = [
            [r["input_ids"][i] for i in r["partition"]["suffix"]] for r in selected
        ]
        if [r["corner"] for r in selected] != list(range(8)) or any(
            ids != suffix_ids[0] for ids in suffix_ids
        ):
            raise ValueError(
                "Complete cube and identical ordered suffix tokens required"
            )
        for corner in (0, 2, 4, 6):
            a, b = selected[corner : corner + 2]
            ia, ib = a["input_ids"], b["input_ids"]
            changed = [i for i, (x, y) in enumerate(zip(ia, ib, strict=True)) if x != y]
            if (
                len(changed) != 2
                or Counter(ia) != Counter(ib)
                or any(i not in a["partition"]["body"] for i in changed)
            ):
                raise ValueError("Ownership must swap exactly two body-name tokens")
            proofs.append(
                {
                    "cube": cube,
                    "receivers": [a["receiver"], b["receiver"]],
                    "changed_positions": changed,
                }
            )
        cubes.append(
            {
                "cube": cube,
                "focal": key[0],
                "companion": key[1],
                "query": key[2],
                "receivers": [r["receiver"] for r in selected],
            }
        )
    design = [
        {
            "unit": r["receiver"],
            "receiver": r["receiver"],
            "cube": r["cube"],
            "corner": r["corner"],
            "kind": "capture",
            "arm": "native",
        }
        for r in rows
    ]
    for cube in cubes:
        for corner, receiver in enumerate(cube["receivers"]):
            for arm in M.ARMS:
                design.append(
                    {
                        "unit": len(design),
                        "receiver": receiver,
                        "cube": cube["cube"],
                        "corner": corner,
                        "kind": "patch",
                        "arm": arm,
                    }
                )
    if (
        len(rows) != 528
        or len(cubes) != 66
        or len(design) != spec["max_forwards"]
        or len({r["body"] for r in rows}) != 288
    ):
        raise ValueError("Incomplete declared 528/66/4224/288 scope")
    for directory in ("00/code", "01/units", "02/payloads", "03"):
        (out / directory).mkdir(parents=True)
    assets = {
        "00/rows.json": rows,
        "00/cubes.json": cubes,
        "00/design.json": design,
        "00/token_proofs.json": proofs,
    }
    for name, value in assets.items():
        write(out / name, value)
    for name in CODE:
        dest = out / "00/code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    sources = (
        PARENT / "00/manifest.json",
        PARENT / "verification.json",
        REFERENCE / "00/manifest.json",
        REFERENCE / "01/observer.json",
    )
    write(
        out / "00/manifest.json",
        {
            "packet": "HM74",
            "spec": spec,
            "alphabet": alphabet,
            "code": {n: digest(ROOT / n) for n in CODE},
            "assets": {n: digest(out / n) for n in assets},
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "model_config_sha256": digest(MODEL / "config.json"),
            "tokenizer": {
                p.name: digest(p) for p in MODEL.glob("*token*") if p.is_file()
            },
            "model_weights_separately_hashed": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_bodies": 288,
            "native_captures": 528,
            "query_cubes": 66,
            "counterfactual_full_cube_not_standalone_reader": True,
            "original_scope_complete": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "attempted_forward_calls": 0,
            "completed_units": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print("HM74 PREPARED: 66 query cubes;528 native+3696 fixed patches", flush=True)


def check(out):
    """Reject drift before every phase without altering frozen predecessors."""
    m = read_json(out / "00/manifest.json")
    for n, sha in m["code"].items():
        if digest(ROOT / n) != sha or digest(out / "00/code" / n) != sha:
            raise ValueError("Frozen code drift: " + n)
    for n, sha in m["assets"].items():
        if digest(out / n) != sha:
            raise ValueError("Material drift: " + n)
    for n, sha in m["sources"].items():
        if digest(ROOT / n) != sha:
            raise ValueError("Predecessor drift: " + n)
    for n, sha in m["tokenizer"].items():
        if digest(MODEL / n) != sha:
            raise ValueError("Tokenizer drift")
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Model config drift")
    return m


def freeze_payloads(out, rows, cubes, spec):
    """Seal every subtraction before the first intervened output is observed."""
    barrier = out / "02/payload_barrier.json"
    if barrier.exists():
        for name, sha in read_json(barrier)["files"].items():
            if digest(out / name) != sha:
                raise ValueError("Payload barrier drift")
        return
    files = {}
    for group in cubes:
        states = []
        for receiver in group["receivers"]:
            path = unit_path(out, receiver)
            _, arr = load_arrays(path)
            states.append(arr["full"][rows[receiver]["partition"]["suffix"]])
            for p in (path, path.with_suffix(".npz")):
                files[str(p.relative_to(out))] = digest(p)
        cube = np.stack(states)
        for arm in M.ARMS:
            path = payload_path(out, group["cube"], arm)
            if path.exists():
                raise RuntimeError(
                    "Partial payload build requires inspection, not overwrite"
                )
            value = M.payload(cube, arm, spec["random_seed"] + group["cube"])
            save(
                path,
                {
                    "cube": group["cube"],
                    "arm": arm,
                    "values_sha256": M.array_hash(value),
                },
                {"values": value},
            )
            for p in (path, path.with_suffix(".npz")):
                files[str(p.relative_to(out))] = digest(p)
    write(
        barrier,
        {
            "native_calls_at_seal": 528,
            "patch_calls_at_seal": 0,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": files,
        },
    )
    print("HM74 all native states and462 payloads sealed before patches", flush=True)


def collect(out):
    """Execute each attempted acquisition once, persisting a resumable exact ledger."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("Stopped or uncertain attempts require inspection; no retry")
    if len(list((out / "01/units").glob("unit_*.json"))) != status["completed_units"]:
        raise ValueError("Acquisition ledger/file disagreement")
    rows, cubes, design = (
        read_json(out / n) for n in ("00/rows.json", "00/cubes.json", "00/design.json")
    )
    started, previous = time.monotonic(), status["model_seconds"]
    spec = m["spec"]
    group_cache, current_cube = {}, None
    try:
        observer = load_observer()
        actual, old = observer.manifest(), read_json(REFERENCE / "01/observer.json")
        for key in (
            "model",
            "torch",
            "transformers",
            "dtype",
            "device",
            "hidden_size",
            "layers",
            "max_tokens",
            "attention",
            "training",
            "template_sha256",
        ):
            if actual[key] != old[key]:
                raise ValueError("Observer drift: " + key)
        write(out / "01/observer.json", actual)
        for entry in design[status["completed_units"] :]:
            elapsed = previous + time.monotonic() - started
            if (
                elapsed >= spec["max_model_seconds"]
                or status["attempted_forward_calls"] >= spec["max_forwards"]
            ):
                raise TimeoutError(
                    "Fixed acquisition cap reached; preserve partial output"
                )
            row = rows[entry["receiver"]]
            patch = None
            if entry["kind"] == "patch":
                if current_cube != entry["cube"]:
                    freeze_payloads(
                        out, rows, cubes, spec
                    ) if current_cube is None else None
                    group_cache = {
                        arm: load_arrays(payload_path(out, entry["cube"], arm))[1][
                            "values"
                        ]
                        for arm in M.ARMS
                    }
                    group_cache["native"] = {
                        r: load_arrays(unit_path(out, r))[1]["full"]
                        for r in cubes[entry["cube"]]["receivers"]
                    }
                    current_cube = entry["cube"]
                patch = (
                    row["partition"]["suffix"],
                    group_cache[entry["arm"]][entry["corner"]],
                )
            status.update(
                status="COLLECTING",
                in_flight=entry["unit"],
                model_seconds=previous + time.monotonic() - started,
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, captured = observe_full(
                observer, row, m["alphabet"], spec["layer"], patch
            )
            metadata = {**entry, "native": native}
            if patch is None:
                arrays = {"full": captured["full"]}
                metadata["full_sha256"] = M.array_hash(arrays["full"])
            else:
                p = patch[0]
                before = captured["before"]
                unchanged = np.arange(p[0])
                original = group_cache["native"][entry["receiver"]]
                metadata.update(
                    before_sha256=M.array_hash(before),
                    unchanged_before_sha256=M.array_hash(before[unchanged]),
                    unchanged_after_sha256=M.array_hash(captured["full"][unchanged]),
                    native_before_max_error=float(
                        np.max(abs(before.astype(float) - original.astype(float)))
                    ),
                    actual_payload_max_error=float(
                        np.max(
                            abs(
                                captured["full"][p].astype(float)
                                - patch[1].astype(float)
                            )
                        )
                    ),
                )
                arrays = {"actual_suffix": captured["full"][p]}
                if (
                    metadata["unchanged_before_sha256"]
                    != metadata["unchanged_after_sha256"]
                    or metadata["before_sha256"] != M.array_hash(original)
                    or metadata["actual_payload_max_error"] != 0
                ):
                    raise ValueError(
                        "Intervention changed undeclared states or mismatched payload"
                    )
            save(unit_path(out, entry["unit"]), metadata, arrays)
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                forward_calls=status["forward_calls"] + 1,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 64 == 0 or status["completed_units"] == 528:
                print(
                    f"HM74 {entry['kind']}: {status['completed_units']}/4224; elapsed={status['model_seconds']:.1f}s",
                    flush=True,
                )
        if (
            any(
                status[k] != spec["max_forwards"]
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > spec["max_model_seconds"]
        ):
            raise ValueError("Incomplete or over-budget collection")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
        files = [
            p
            for directory in ("01", "02")
            for p in (out / directory).rglob("*")
            if p.is_file()
        ]
        write(
            out / "raw_seal.json", {str(p.relative_to(out)): digest(p) for p in files}
        )
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def analyze(out):
    """Describe all fixed causal arms, signed actions and author-composition effects."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Full actual collection required")
    if (out / "03/output_seal.json").exists():
        raise FileExistsError("Preserve analyzed result")
    for name, sha in read_json(out / "raw_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Raw capture drift: " + name)
    design, cubes = read_json(out / "00/design.json"), read_json(out / "00/cubes.json")
    data = defaultdict(dict)
    minimum_mass = 1.0
    noop_error = 0.0
    for entry in design:
        meta = read_json(unit_path(out, entry["unit"]))
        z = np.asarray(meta["native"]["candidate_logits"], float)
        values = decode(z)
        data[entry["cube"]].setdefault(entry["arm"], {})[entry["corner"]] = {
            **values,
            "centered_logits": z - z.mean(),
        }
        minimum_mass = min(minimum_mass, meta["native"]["candidate_mass"])
    coefficients, actions, composition, joint, query_pairs = [], [], [], [], []
    aggregate = defaultdict(list)
    tables = {}
    for group in cubes:
        cube = group["cube"]
        identity = {k: group[k] for k in ("cube", "focal", "companion", "query")}
        for stage in STAGES:
            values = {
                arm: np.asarray([data[cube][arm][i][stage] for i in range(8)])
                for arm in ("native", *M.ARMS)
            }
            native = values["native"]
            noop_error = max(noop_error, float(np.max(abs(values["noop"] - native))))
            table = np.stack([values[arm] for arm in ("native", *M.ARMS)])
            tables[f"cube_{cube:03d}_{stage}"] = table
            for arm, value in values.items():
                c = M.coefficients(value)
                for j, character in enumerate(M.CHARACTERS):
                    coefficients.append(
                        {
                            **identity,
                            "stage": stage,
                            "arm": arm,
                            "character": character,
                            "value": np.asarray(c[j]).tolist(),
                            "native_value": np.asarray(
                                M.coefficients(native)[j]
                            ).tolist(),
                        }
                    )
                act, orig = M.ownership_actions(value), M.ownership_actions(native)
                for j, context in enumerate(("marginal", "x_low", "x_high")):
                    delta = act[j] - orig[j]
                    actions.append(
                        {
                            **identity,
                            "stage": stage,
                            "arm": arm,
                            "context": context,
                            "native": orig[j].tolist(),
                            "patched": act[j].tolist(),
                            "change": delta.tolist(),
                            "loss": (-delta).tolist(),
                        }
                    )
                    if stage != "centered_logits":
                        category = (
                            "neutral"
                            if group["companion"] == "neutral"
                            else "matched"
                            if group["query"] == group["companion"]
                            else "focal"
                        )
                        aggregate[(stage, arm, context, category)].append(
                            float(delta[0])
                        )
                for character, account in M.composition_contrasts(
                    value, m["spec"]["author_profile_amplitude"]
                ).items():
                    composition.append(
                        {
                            **identity,
                            "stage": stage,
                            "arm": arm,
                            "character": character,
                            **{k: np.asarray(v).tolist() for k, v in account.items()},
                        }
                    )
            interaction = M.joint_ablation(
                native, values["remove_yg"], values["remove_xyg"], values["remove_both"]
            )
            joint.append(
                {
                    **identity,
                    "stage": stage,
                    "interaction": interaction.tolist(),
                    "rms": float(np.sqrt(np.mean(interaction**2))),
                    "loss_identity_error": float(
                        np.max(
                            abs(
                                (native - values["remove_both"])
                                - (
                                    native
                                    - values["remove_yg"]
                                    + native
                                    - values["remove_xyg"]
                                    - interaction
                                )
                            )
                        )
                    ),
                }
            )
    pair_lookup = {(g["focal"], g["companion"], g["query"]): g["cube"] for g in cubes}
    for group in cubes:
        if group["companion"] == "neutral" or group["query"] != group["companion"]:
            continue
        other = pair_lookup[group["focal"], group["companion"], group["focal"]]
        for stage in STAGES[:2]:
            a, b = (
                tables[f"cube_{group['cube']:03d}_{stage}"],
                tables[f"cube_{other:03d}_{stage}"],
            )
            native_a, native_b = M.ownership_actions(a[0]), M.ownership_actions(b[0])
            for j, arm in enumerate(("native", *M.ARMS)):
                aa, bb = M.ownership_actions(a[j]), M.ownership_actions(b[j])
                for k, context in enumerate(("marginal", "x_low", "x_high")):
                    query_pairs.append(
                        {
                            "focal": group["focal"],
                            "companion": group["companion"],
                            "stage": stage,
                            "arm": arm,
                            "context": context,
                            "matched_action": float(aa[k, 0]),
                            "focal_action": float(bb[k, 0]),
                            "matched_minus_focal": float(aa[k, 0] - bb[k, 0]),
                            "selectivity_change": float(
                                aa[k, 0] - bb[k, 0] - native_a[k, 0] + native_b[k, 0]
                            ),
                        }
                    )
    summary = {
        "packet": "HM74",
        "actual_forwards": 4224,
        "new_bodies": 288,
        "query_cubes": 66,
        "minimum_candidate_mass": minimum_mass,
        "noop_max_error": noop_error,
        "aggregate": [
            {
                "stage": key[0],
                "arm": key[1],
                "context": key[2],
                "category": key[3],
                "n": len(v),
                "action_change_mean": float(np.mean(v)),
                "action_change_rms": float(np.sqrt(np.mean(np.square(v)))),
                "negative_change_count": sum(x < 0 for x in v),
            }
            for key, v in aggregate.items()
        ],
        "two_queries_per_non_neutral_pair_not_global_selectivity": True,
        "common_reader_not_human_trait_or_individual_Bu": True,
        "original_scope_complete": False,
        "no_accuracy_search": True,
    }
    for name, value in (
        ("coefficients.csv", coefficients),
        ("actions.csv", actions),
        ("composition_consequences.csv", composition),
        ("joint_ablation.csv", joint),
        ("query_pair_contrasts.csv", query_pairs),
    ):
        write_csv(out / "03" / name, value)
    save(
        out / "03/tables.json",
        {"arm_order": ["native", *M.ARMS], "stage_order": STAGES},
        tables,
    )
    write(out / "03/summary.json", summary)
    write(
        out / "03/output_seal.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in (out / "03").iterdir()
            if p.is_file() and p.name != "output_seal.json"
        },
    )
    print(
        "HM74 complete analysis: causal actions, all coefficients and composition consequences",
        flush=True,
    )


def main():
    """Expose bounded preparation, collection and analysis as separate phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.phase == "collect":
        with collector_guard(args.out):
            collect(args.out)
    else:
        globals()[args.phase](args.out)


if __name__ == "__main__":
    main()
