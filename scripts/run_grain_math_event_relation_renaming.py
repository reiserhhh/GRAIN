#!/usr/bin/env python3
"""HM58: bounded passive C4 occasion-code orbits, not prediction optimization."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_event_probability_pooling as P
from grain_research.construction import math_event_relation_renaming as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_evidence_response_operation import readout
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_event_probability_pooling import CODE as PROB_CODE
from scripts.run_grain_math_event_probability_pooling import check as probability_check
from scripts.run_grain_math_event_probability_pooling import source_snapshot
from scripts.run_grain_math_event_readout_composition import CODE as SOURCE_CODE
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_event_relation_renaming_v1.json"
OUT = ROOT / "results/grain_math_response_v1/event_relation_renaming_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_event_relation_renaming_v1.json",
            "docs/GRAIN_EVENT_RELATION_RENAMING_PLAN_V1.md",
            "grain_research/construction/math_event_relation_renaming.py",
            "scripts/run_grain_math_event_relation_renaming.py",
            "scripts/verify_grain_math_event_relation_renaming.py",
            "tests/test_grain_math_event_relation_renaming.py",
            *PROB_CODE,
            *SOURCE_CODE,
        )
    )
)
FIELDS = (
    "unit",
    "base_unit",
    "source_unit",
    "parent_unit",
    "g",
    "role",
    "content",
    "query",
    "family",
    "polarity",
    "order",
    "regime",
)


def token_proof(original, row, tokenizer):
    """Prove that only actual occasion-digit token identities changed."""
    a, b = original["input_ids"], row["input_ids"]
    if len(a) != len(b) or original["partition"] != row["partition"]:
        raise ValueError("Renaming changed token lengths/partitions")
    encoded = tokenizer(
        original["rendered"], add_special_tokens=False, return_offsets_mapping=True
    )
    if encoded["input_ids"] != a:
        raise ValueError("Original tokenizer IDs drift")
    positions = []
    for match in M.CODE.finditer(original["rendered"]):
        start, stop = match.start() + 1, match.end()
        selected = [
            i
            for i, (x, y) in enumerate(encoded["offset_mapping"])
            if x < stop and y > start
        ]
        if len(selected) != 1 or selected[0] not in original["partition"]["body"]:
            raise ValueError("Occasion digit must be exactly one body token")
        positions.extend(selected)
    changed = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    if sorted(set(positions)) != changed:
        raise ValueError("Changes outside exact occasion digits")
    return {
        "unit": row["unit"],
        "base_unit": row["base_unit"],
        "g": row["g"],
        "positions": changed,
        "old_ids": [a[i] for i in changed],
        "new_ids": [b[i] for i in changed],
        "equal_full_length": len(a),
        "signature_unchanged": True,
    }


def prepare(out):
    """Seal full group rosters, raw historical bindings and code before forwards."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve previous runs")
    spec = read_json(SPEC)
    reference = ROOT / spec["probability_reference"]
    old_probability = probability_check(reference)
    if (
        read_json(reference / "verification.json")["status"]
        != "VERIFIED_FINITE_HM57_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM57 required")
    oldrows, _, paths, lineage, targets, _ = source_snapshot(old_probability["spec"])
    if len(oldrows) != spec["expected_source_readouts"]:
        raise ValueError("Complete source roster required")
    for name in (
        "00/manifest.json",
        "verification.json",
        "03/summary.json",
        "03/artifact_hashes.json",
    ):
        lineage[str((reference / name).relative_to(ROOT))] = digest(reference / name)
    for name, expected in read_json(reference / "03/artifact_hashes.json").items():
        if digest(reference / name) != expected:
            raise ValueError("Frozen HM57 result drift")
        lineage[str((reference / name).relative_to(ROOT))] = expected
    source = ROOT / spec["source"]
    old = read_json(source / "00/manifest.json")
    original_parent = ROOT / old["spec"]["parent"]
    inherited = read_json(original_parent / "00/manifest.json")
    if (
        spec["group_elements"] != list(M.GROUP)
        or spec["regimes"] != list(M.REGIMES)
        or spec["fit_patch_API_training_private"]
    ):
        raise ValueError("Declared finite non-training protocol required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    selected = [r for r in oldrows if r["query"] == r["content"]]
    rows, proofs, reused = [], [], {}
    for base_unit, original in enumerate(selected):
        for g in M.GROUP:
            body, history = M.transform(original["body"], original["history"], g)
            text, ids, part, cues, actor = render(
                body,
                original["content"],
                "A",
                1,
                inherited["spec"]["system"],
                tokenizer,
            )
            row = {
                **original,
                "unit": len(rows),
                "base_unit": base_unit,
                "source_unit": original["unit"],
                "g": g,
                "body": body,
                "history": history,
                "rendered": text,
                "input_ids": ids,
                "partition": part,
                "cue_slots": cues,
                "actor_slot": actor,
                "reused": g == 0,
            }
            if g == 0:
                if any(
                    row[k] != original[k]
                    for k in (
                        "body",
                        "history",
                        "rendered",
                        "input_ids",
                        "partition",
                        "cue_slots",
                        "actor_slot",
                    )
                ):
                    raise ValueError("Identity orbit differs from historical input")
                path = ROOT / paths[original["unit"]]
                raw = read_json(path)
                reused[str(row["unit"])] = {
                    "meta": str(path.relative_to(ROOT)),
                    "meta_sha256": digest(path),
                    "array_sha256": raw["array_sha256"],
                }
            else:
                proofs.append(token_proof(original, row, tokenizer))
            rows.append(row)
    pairs = M.pair_design(targets)
    if (
        len(selected),
        len(rows),
        len(reused),
        len(proofs),
        len(pairs),
        sum(r["role"] == "atom" for r in rows),
    ) != (
        spec["expected_base_documents"],
        spec["expected_readouts"],
        spec["expected_reused"],
        spec["expected_token_proofs"],
        spec["expected_pairs"],
        spec["expected_atom_readouts"],
    ):
        raise ValueError("Incomplete declared group design")
    for folder in ("00/code", "01/units", "03"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    write(out / "00/rows.json", rows)
    write(out / "00/token_proofs.json", proofs)
    write(out / "00/pairs.json", pairs)
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(path)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM58",
            "spec": spec,
            "targets": targets,
            "layer": old["layer"],
            "alphabet": old["alphabet"],
            "system": inherited["spec"]["system"],
            "code": code,
            "source_hashes": lineage,
            "reused": reused,
            "assets": {
                n: digest(out / n)
                for n in ("00/rows.json", "00/token_proofs.json", "00/pairs.json")
            },
            "model_config_sha256": digest(MODEL / "config.json"),
            "model_weights_not_separately_hashed": True,
            "observer_reference": str(
                (original_parent / "01/observer.json").relative_to(ROOT)
            ),
            "observer_reference_sha256": digest(original_parent / "01/observer.json"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "independent_human_units": 0,
            "no_fit_or_patch": True,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "attempted_forward_calls": 0,
            "completed_units": 0,
            "reused_units": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        "PREPARED HM58:864 readings/216 exact reuses/648 bounded new calls", flush=True
    )


def check(out):
    """Check immutable materials, all inherited evidence and exact historical reuses."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen material drift " + name)
    for name, sha in m["source_hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source drift " + name)
    if (
        digest(MODEL / "config.json") != m["model_config_sha256"]
        or digest(ROOT / m["observer_reference"]) != m["observer_reference_sha256"]
    ):
        raise ValueError("Observer reference drift")
    for item in m["reused"].values():
        path = ROOT / item["meta"]
        if (
            digest(path) != item["meta_sha256"]
            or digest(path.with_suffix(".npz")) != item["array_sha256"]
        ):
            raise ValueError("Historical capture drift")
    return m


def observation(out, row, manifest):
    """Bind native logits to exactly reused or newly saved full input identities."""
    if row["reused"]:
        meta = read_json(ROOT / manifest["reused"][str(row["unit"])]["meta"])
    else:
        path = out / f"01/units/unit_{row['unit']:04d}.json"
        meta = read_json(path)
        if (
            any(meta[k] != row[k] for k in (*FIELDS, "body", "input_ids"))
            or digest(path.with_suffix(".npz")) != meta["array_sha256"]
        ):
            raise ValueError("Fresh capture binding mismatch")
    raw = meta["native"]
    p = P.probabilities(raw["candidate_logits"])
    v = np.asarray(manifest["alphabet"]["values"])
    if (
        len(p) != 10
        or abs(float(p @ v) - raw["conditional_score"])
        > manifest["spec"]["arithmetic_tolerance"]
        or np.max(np.abs(p - raw["conditional_probabilities"]))
        > manifest["spec"]["arithmetic_tolerance"]
    ):
        raise ValueError("Raw candidate replay failed")
    return raw


def collect(out):
    """Run each new orbit once with a recoverable checkpoint and finite bound."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("No automatic retry of terminal or uncertain attempts")
    if (
        len(list((out / "01/units").glob("unit_*.json"))) != status["completed_units"]
        or status["completed_units"] != status["forward_calls"]
    ):
        raise ValueError("Saved capture/status mismatch")
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual, prior = observer.manifest(), read_json(ROOT / m["observer_reference"])
        for k in (
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
            if actual[k] != prior[k]:
                raise ValueError("Observer drift " + k)
        write(out / "01/observer.json", actual)
        for row in read_json(out / "00/rows.json"):
            if row["reused"]:
                continue
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                observation(out, row, m)
                continue
            elapsed = previous + time.monotonic() - started
            if not call_allowed(status, m["spec"], elapsed):
                raise TimeoutError("648-forward/900-second bound")
            status.update(
                status="COLLECTING", in_flight=row["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            raw, arrays = observe_full(observer, row, m["alphabet"], m["layer"])
            status["forward_calls"] += 1
            save(
                path,
                {k: row[k] for k in (*FIELDS, "body", "input_ids")}
                | {"native": raw, "finished_epoch": time.time()},
                {"endpoint": arrays["full"][-1]},
            )
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 108 == 0:
                print(f"ACTUAL:{status['completed_units']}/648", flush=True)
        if (
            any(
                status[k] != m["spec"]["max_forwards"]
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > m["spec"]["max_model_seconds"]
        ):
            raise ValueError("Incomplete finite acquisition")
        status.update(status="COLLECTED", reused_units=m["spec"]["expected_reused"])
        write(out / "run_status.json", status)
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def analyze(out):
    """Compare fixed orbits and actual-code pools; never select a best naming."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] != "COLLECTED":
        raise ValueError("Complete finite acquisition required")
    rows = read_json(out / "00/rows.json")
    v, tol = np.asarray(m["alphabet"]["values"]), m["spec"]["arithmetic_tolerance"]
    native, probabilities, raw_logits, atoms, accounts = [], {}, {}, {}, {}
    for r in rows:
        raw = observation(out, r, m)
        p = P.probabilities(raw["candidate_logits"])
        probabilities[r["unit"]], raw_logits[r["unit"]] = (
            p,
            np.asarray(raw["candidate_logits"]),
        )
        native.append(
            {k: r[k] for k in FIELDS} | readout(raw) | {"reused": r["reused"]}
        )
        if r["role"] == "atom":
            atoms[
                r["content"], r["family"], r["polarity"], r["history"][0]["event_id"]
            ] = raw_logits[r["unit"]]
        else:
            accounts[
                r["content"],
                r["family"],
                r["polarity"],
                r["order"],
                r["regime"],
                r["g"],
            ] = r
    pools, pool_rows = {}, []
    for r in rows:
        if r["role"] != "account":
            continue
        for basis in m["spec"]["bases"]:
            keys, weights, count = M.actual_atom_weights(r["history"], basis)
            x = np.asarray([atoms[(r["content"], *key)] for key in keys])
            mix, geom = P.pools(x, weights)
            for name, distribution in (("mixture", mix), ("geometric", geom)):
                pools[r["unit"], basis, name] = distribution
                for j, value in enumerate(distribution):
                    pool_rows.append(
                        {
                            "unit": r["unit"],
                            "g": r["g"],
                            "basis": basis,
                            "pool": name,
                            "records": count,
                            "candidate": j,
                            "probability": float(value),
                            "score": float(distribution @ v),
                        }
                    )
    paired, orbits, collisions = [], [], []
    for pair in read_json(out / "00/pairs.json"):
        base = {
            k: pair[k]
            for k in (
                "pair",
                "content",
                "family",
                "polarity",
                "order",
                "contrast",
                "orientation",
            )
        }
        selected, vectors = [], {}
        for g in M.GROUP:
            hi, lo = (accounts[(*pair[k], g)] for k in ("hi_key", "lo_key"))
            a, b = hi["unit"], lo["unit"]
            pn, pm = probabilities[a], probabilities[b]
            dp = pn - pm
            change = float(dp @ v)
            line = base | {
                "g": g,
                "hi": a,
                "lo": b,
                "native_change": change,
                "native_tv": float(np.abs(dp).sum() / 2),
            }
            vectors[g] = {"native_hi": pn, "native_lo": pm}
            for basis, name in itertools.product(
                m["spec"]["bases"], m["spec"]["pools"]
            ):
                x, y = pools[a, basis, name], pools[b, basis, name]
                key = basis + "_" + name
                line[key + "_change"] = float((x - y) @ v)
                vectors[g][key + "_hi"], vectors[g][key + "_lo"] = x, y
                collisions.append(
                    base
                    | {"g": g, "scope": "one_code", "basis": basis, "pool": name}
                    | M.collision(x, y, pn, pm, v, tol)
                )
            paired.append(line)
            selected.append(line)
        result = base | M.orbit_decomposition([r["native_change"] for r in selected])
        result["g0_change"] = selected[0]["native_change"]
        result["oriented_mean"] = (
            None
            if pair["orientation"] is None
            else pair["orientation"] * result["orbit_mean"]
        )
        pn, pm = (
            np.mean([vectors[g]["native_" + k] for g in M.GROUP], axis=0)
            for k in ("hi", "lo")
        )
        result["orbit_native_tv"] = float(np.abs(pn - pm).sum() / 2)
        for basis, name in itertools.product(m["spec"]["bases"], m["spec"]["pools"]):
            key = basis + "_" + name
            x, y = (
                np.mean([vectors[g][key + "_" + k] for g in M.GROUP], axis=0)
                for k in ("hi", "lo")
            )
            result[key + "_mean_change"] = float((x - y) @ v)
            result[key + "_orbit_residual"] = (
                result["orbit_mean"] - result[key + "_mean_change"]
            )
            collisions.append(
                base
                | {"g": -1, "scope": "full_C4_mean", "basis": basis, "pool": name}
                | M.collision(x, y, pn, pm, v, tol)
            )
        orbits.append(result)
    summaries = []
    for content in ("ALL", *m["targets"]):
        for contrast in sorted({r["contrast"] for r in orbits}):
            group = [
                r
                for r in orbits
                if r["contrast"] == contrast
                and (content == "ALL" or r["content"] == content)
            ]
            if not group:
                continue
            invariant = float(np.mean([r["invariant_energy"] for r in group]))
            sensitivity = float(np.mean([r["code_variance"] for r in group]))
            energy = invariant + sensitivity
            summaries.append(
                {
                    "content": content,
                    "contrast": contrast,
                    "pairs": len(group),
                    "mean_contrast": float(np.mean([r["orbit_mean"] for r in group])),
                    "invariant_RMS": float(np.sqrt(invariant)),
                    "code_SD_RMS": float(np.sqrt(sensitivity)),
                    "total_RMS": float(np.sqrt(energy)),
                    "invariant_energy_share": invariant / energy if energy else None,
                    "g0_RMS": float(
                        np.sqrt(np.mean([r["g0_change"] ** 2 for r in group]))
                    ),
                    "oriented_mean": float(np.mean([r["oriented_mean"] for r in group]))
                    if group[0]["orientation"] is not None
                    else None,
                    "all_codes_same_nonzero_sign": sum(
                        r["positive_codes"] == 4 or r["negative_codes"] == 4
                        for r in group
                    ),
                    "orbit_native_tv_mean": float(
                        np.mean([r["orbit_native_tv"] for r in group])
                    ),
                }
            )
    collision_summaries = []
    for scope, contrast, basis, name in itertools.product(
        ("one_code", "full_C4_mean"),
        sorted({r["contrast"] for r in orbits}),
        m["spec"]["bases"],
        m["spec"]["pools"],
    ):
        group = [
            r
            for r in collisions
            if (r["scope"], r["contrast"], r["basis"], r["pool"])
            == (scope, contrast, basis, name)
        ]
        same = [r for r in group if r["same_summary"]]
        collision_summaries.append(
            {
                "scope": scope,
                "contrast": contrast,
                "basis": basis,
                "pool": name,
                "pairs": len(group),
                "same_summary": len(same),
                "same_summary_changed_observation": sum(
                    r["same_summary_changed_observation"] for r in group
                ),
                "maximum_summary_difference": max(
                    r["summary_max_difference"] for r in group
                ),
                "paired_score_RMS_floor": float(
                    np.sqrt(np.mean([r["paired_score_RMS_floor"] ** 2 for r in same]))
                )
                if same
                else None,
            }
        )
    tables = {
        "native": native,
        "pool_coordinates": pool_rows,
        "paired_changes": paired,
        "orbits": orbits,
        "collisions": collisions,
        "orbit_summary": summaries,
        "collision_summary": collision_summaries,
    }
    for name, data in tables.items():
        write_csv(out / f"03/{name}.csv", data)
    summary = {
        "packet": "HM58",
        "status": "ANALYZED_FINITE_NOT_GLOBAL",
        "counts": {k: len(d) for k, d in tables.items()},
        "actual_new_calls": status["forward_calls"],
        "reused_calls": status["reused_units"],
        "model_seconds": status["model_seconds"],
        "maximum_orbit_identity_error": max(r["identity_error"] for r in orbits),
        "results": [r for r in summaries if r["content"] == "ALL"],
        "no_fit_or_patch": True,
        "is_personality_accuracy": False,
        "group_average_invariance_is_tautological": True,
        "group_is_C4_not_full_S4": True,
        "original_scope_complete": False,
    }
    write(out / "03/summary.json", summary)
    write(
        out / "03/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted((out / "03").iterdir())
            if p.name != "artifact_hashes.json" and not p.name.startswith("._")
        },
    )
    print(summary, flush=True)


def main():
    """Expose finite, checkpointed stages with no generation or automatic retry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.phase == "collect":
        with collector_guard(args.output):
            collect(args.output)
    else:
        {"prepare": prepare, "analyze": analyze}[args.phase](args.output)


if __name__ == "__main__":
    main()
