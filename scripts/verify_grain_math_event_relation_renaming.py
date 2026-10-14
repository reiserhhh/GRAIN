#!/usr/bin/env python3
"""HM58 raw-binding and alternate-formula audit, without observer calls or fits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_math_response_v1/event_relation_renaming_r1"
MAX_ERROR = 0.0


def read(path):
    """Read explicit JSON evidence without invoking a model or analysis module."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Independently bind local evidence bytes with streaming SHA256."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(actual, expected, tolerance):
    """Check every saved field, preserving strings, null and booleans."""
    global MAX_ERROR
    if expected is None:
        if actual not in (None, ""):
            raise ValueError("Null replaced with a value")
    elif isinstance(expected, (bool, np.bool_)):
        expected = bool(expected)
        if actual not in (expected, str(expected)):
            raise ValueError("Boolean mismatch")
    elif isinstance(expected, (str, list, dict)):
        if actual != expected:
            raise ValueError("Identity/structural mismatch")
    else:
        a, b = float(actual), float(expected)
        error = abs(a - b)
        if not np.isfinite(a) or error > tolerance:
            raise ValueError(f"Numerical mismatch:{a}/{b}")
        MAX_ERROR = max(MAX_ERROR, error)


def table_check(out, name, expected, keys, tolerance):
    """Require exact row count, unique keys and every recomputed field."""
    with (out / f"03/{name}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    key = lambda r: tuple(str(r[k]) for k in keys)
    found = {key(r): r for r in rows}
    want = {key(r): r for r in expected}
    if (
        len(rows) != len(expected)
        or len(found) != len(rows)
        or len(want) != len(expected)
        or set(found) != set(want)
    ):
        raise ValueError("Incomplete/duplicate table:" + name)
    for k, expected_row in want.items():
        if set(found[k]) != set(expected_row):
            raise ValueError("Table field mismatch:" + name)
        for field, value in expected_row.items():
            compare(found[k][field], value, tolerance)
    return len(rows)


def verify(out):
    """Independently rebuild all orbit/pool/collision outputs from actual logits."""
    from transformers import AutoTokenizer

    m, state = read(out / "00/manifest.json"), read(out / "run_status.json")
    spec, tol, v = m["spec"], m["spec"]["arithmetic_tolerance"], np.arange(10) / 9
    if (
        m["packet"] != "HM58"
        or state["status"] != "COLLECTED"
        or state["in_flight"] is not None
    ):
        raise ValueError("Terminal, complete HM58 required")
    for k in ("forward_calls", "attempted_forward_calls", "completed_units"):
        if state[k] != 648:
            raise ValueError("Wrong actual call accounting")
    if state["reused_units"] != 216 or state["model_seconds"] > 900:
        raise ValueError("Acquisition bound/reuse mismatch")
    for name, expected in m["code"].items():
        if sha(ROOT / name) != expected or sha(out / "00/code" / name) != expected:
            raise ValueError("Code drift:" + name)
    for mapping, root in ((m["assets"], out), (m["source_hashes"], ROOT)):
        for name, expected in mapping.items():
            if sha(root / name) != expected:
                raise ValueError("Evidence drift:" + name)
    for name, expected in read(out / "03/artifact_hashes.json").items():
        if sha(out / name) != expected:
            raise ValueError("Result drift:" + name)
    model = Path(read(ROOT / m["observer_reference"])["model"])
    if (
        sha(model / "config.json") != m["model_config_sha256"]
        or sha(ROOT / m["observer_reference"]) != m["observer_reference_sha256"]
    ):
        raise ValueError("Model reference drift")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    original = [
        r
        for r in read(ROOT / spec["source"] / "00/rows.json")
        if r["query"] == r["content"]
    ]
    rows, proofs = read(out / "00/rows.json"), read(out / "00/token_proofs.json")
    if (
        len(original) != 216
        or len(rows) != 864
        or len(proofs) != 648
        or [r["unit"] for r in rows] != list(range(864))
    ):
        raise ValueError("Incomplete canonical input roster")
    proof_index = {r["unit"]: r for r in proofs}
    if len(proof_index) != 648:
        raise ValueError("Duplicate token proofs")
    native, probs, logits, atoms, accounts = [], {}, {}, {}, {}
    fields = (
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
    minimum_mass = 1.0
    for r in rows:
        o, g = original[r["base_unit"]], r["g"]
        if (
            r["unit"] != 4 * r["base_unit"] + g
            or r["source_unit"] != o["unit"]
            or r["reused"] != (g == 0)
        ):
            raise ValueError("Orbit/source identity mismatch")
        mapping = {"K" + str(j): "K" + str(1 + (j - 1 + g) % 4) for j in (1, 2, 3, 4)}
        for field in (
            "content",
            "query",
            "family",
            "polarity",
            "order",
            "regime",
            "role",
            "parent_unit",
        ):
            compare(r[field], o[field], tol)
        expected_history = [
            {**x, "event_id": mapping[x["event_id"]]} for x in o["history"]
        ]
        compare(r["history"], expected_history, tol)
        if any(
            r[field]
            != re.sub(
                r"\bK[1-4]\b",
                lambda match, labels=mapping: labels[match.group()],
                o[field],
            )
            for field in ("body", "rendered")
        ):
            raise ValueError("Text differs beyond declared code action")
        encoded = tokenizer(
            r["rendered"], add_special_tokens=False, return_offsets_mapping=True
        )
        if (
            r["input_ids"] != encoded["input_ids"]
            or len(r["input_ids"]) != len(o["input_ids"])
            or r["partition"] != o["partition"]
        ):
            raise ValueError("Actual token/input length mismatch")
        if g:
            old_encoded = tokenizer(
                o["rendered"], add_special_tokens=False, return_offsets_mapping=True
            )
            expected_positions = []
            for match in re.finditer(r"\bK[1-4]\b", o["rendered"]):
                positions = [
                    i
                    for i, (a, b) in enumerate(old_encoded["offset_mapping"])
                    if a < match.end() and b > match.start() + 1
                ]
                if len(positions) != 1 or positions[0] not in o["partition"]["body"]:
                    raise ValueError("Invalid occasion-digit token")
                expected_positions += positions
            actual_positions = [
                i
                for i, (a, b) in enumerate(
                    zip(o["input_ids"], r["input_ids"], strict=True)
                )
                if a != b
            ]
            compare(
                proof_index[r["unit"]],
                {
                    "unit": r["unit"],
                    "base_unit": r["base_unit"],
                    "g": g,
                    "positions": actual_positions,
                    "old_ids": [o["input_ids"][i] for i in actual_positions],
                    "new_ids": [r["input_ids"][i] for i in actual_positions],
                    "equal_full_length": len(o["input_ids"]),
                    "signature_unchanged": True,
                },
                tol,
            )
            if actual_positions != sorted(set(expected_positions)):
                raise ValueError("Undeclared token changes")
            path = out / f"01/units/unit_{r['unit']:04d}.json"
            raw = read(path)
            for field in (*fields, "body", "input_ids"):
                compare(raw[field], r[field], tol)
        else:
            item = m["reused"][str(r["unit"])]
            path = ROOT / item["meta"]
            if sha(path) != item["meta_sha256"]:
                raise ValueError("Reused raw metadata drift")
            raw = read(path)
            if raw["unit"] != r["parent_unit"] or raw["target"] != r["content"]:
                raise ValueError("Historical content capture misbound")
        if sha(path.with_suffix(".npz")) != raw["array_sha256"]:
            raise ValueError("Raw endpoint array drift")
        if g:
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                if (
                    list(data.files) != ["endpoint"]
                    or data["endpoint"].shape != (1536,)
                    or not np.isfinite(data["endpoint"]).all()
                ):
                    raise ValueError("Invalid actual endpoint capture")
        z = np.asarray(raw["native"]["candidate_logits"], float)
        p = softmax(z)
        compare(float(p @ v), raw["native"]["conditional_score"], tol)
        for a, b in zip(p, raw["native"]["conditional_probabilities"], strict=True):
            compare(a, b, tol)
        probs[r["unit"]], logits[r["unit"]] = p, z
        mass = raw["native"]["candidate_mass"]
        minimum_mass = min(mass, minimum_mass)
        native.append(
            {k: r[k] for k in fields}
            | {
                "score": float(p @ v),
                "log_odds": float(logsumexp(z[5:]) - logsumexp(z[:5])),
                "candidate_mass": mass,
                "reused": r["reused"],
            }
        )
        if r["role"] == "atom":
            key = (
                r["content"],
                r["family"],
                r["polarity"],
                r["history"][0]["event_id"],
            )
            if key in atoms:
                raise ValueError("Duplicate isolated occasion")
            atoms[key] = p
        else:
            accounts[
                (r["content"], r["family"], r["polarity"], r["order"], r["regime"], g)
            ] = r
    if (len(atoms), len(accounts)) != (96, 768):
        raise ValueError("Incomplete actual-code atom/account crossing")
    pools, pool_rows = {}, []
    for r in rows:
        if r["role"] != "account":
            continue
        # Independent parser: this declared four-regime history has only new/alias.
        if any(x["kind"] not in ("new", "alias") for x in r["history"]):
            raise ValueError("Undeclared history operation")
        own = [x for x in r["history"] if x["actor"] == "A"]
        events = {}
        for x in own:
            if x["kind"] == "new":
                if x["event_id"] in events:
                    raise ValueError("Duplicate distinct event")
                events[x["event_id"]] = x
            elif x["parent_id"] != "r1" or x["event_id"] != own[0]["event_id"]:
                raise ValueError("Alias relation mismatch")
        for basis, records in (("event", list(events.values())), ("mention", own)):
            distributions = np.asarray(
                [
                    atoms[(r["content"], x["family"], x["polarity"], x["event_id"])]
                    for x in records
                ]
            )
            mix = distributions.mean(axis=0)
            product = np.prod(distributions ** (1 / len(records)), axis=0)
            geom = product / product.sum()
            for name, distribution in (("mixture", mix), ("geometric", geom)):
                pools[r["unit"], basis, name] = distribution
                for j, value in enumerate(distribution):
                    pool_rows.append(
                        {
                            "unit": r["unit"],
                            "g": r["g"],
                            "basis": basis,
                            "pool": name,
                            "records": len(records),
                            "candidate": j,
                            "probability": float(value),
                            "score": float(distribution @ v),
                        }
                    )
    declared_pairs = read(out / "00/pairs.json")
    pairs = []
    regimes = ("balanced", "alias_focus", "distinct_focus", "other_focus")
    for content, family, polarity, order in itertools.product(
        m["targets"], (0, 1), (0, 1), (0, 1)
    ):
        for label, hi, lo in (
            ("alias_minus_balanced", "alias_focus", "balanced"),
            ("distinct_minus_alias", "distinct_focus", "alias_focus"),
            ("other_minus_balanced", "other_focus", "balanced"),
        ):
            pairs.append(
                {
                    "pair": len(pairs),
                    "content": content,
                    "family": family,
                    "polarity": polarity,
                    "order": order,
                    "contrast": label,
                    "hi_key": [content, family, polarity, order, hi],
                    "lo_key": [content, family, polarity, order, lo],
                    "orientation": 2 * polarity - 1,
                }
            )
    for content, family, polarity, regime in itertools.product(
        m["targets"], (0, 1), (0, 1), regimes
    ):
        pairs.append(
            {
                "pair": len(pairs),
                "content": content,
                "family": family,
                "polarity": polarity,
                "order": -1,
                "contrast": "order_" + regime,
                "hi_key": [content, family, polarity, 1, regime],
                "lo_key": [content, family, polarity, 0, regime],
                "orientation": None,
            }
        )
    compare(declared_pairs, pairs, tol)
    if len(pairs) != 240:
        raise ValueError("Missing design contrast")
    paired, orbits, collisions = [], [], []

    def certificate(x, y, pn, pm):
        """Recompute common-summary floors from midpoint geometry."""
        delta = pn - pm
        same = bool(np.all(np.abs(x - y) <= tol))
        scalar = float(delta @ v)
        midpoint = (pn + pm) / 2
        return {
            "summary_max_difference": float(np.max(np.abs(x - y))),
            "same_summary": same,
            "native_score_change": scalar,
            "native_tv": float(np.linalg.norm(delta, ord=1) / 2),
            "same_summary_changed_observation": same
            and np.linalg.norm(delta, ord=1) / 2 > tol,
            "paired_score_RMS_floor": float(
                np.sqrt(
                    ((pn @ v - midpoint @ v) ** 2 + (pm @ v - midpoint @ v) ** 2) / 2
                )
            )
            if same
            else None,
            "paired_probability_L2_floor": float(
                np.sqrt(
                    (np.sum((pn - midpoint) ** 2) + np.sum((pm - midpoint) ** 2)) / 2
                )
            )
            if same
            else None,
        }

    for pair in pairs:
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
        vectors, selected = {}, []
        for g in (0, 1, 2, 3):
            a, b = (accounts[(*pair[k], g)]["unit"] for k in ("hi_key", "lo_key"))
            pn, pm = probs[a], probs[b]
            line = base | {
                "g": g,
                "hi": a,
                "lo": b,
                "native_change": float(pn @ v - pm @ v),
                "native_tv": float(np.linalg.norm(pn - pm, ord=1) / 2),
            }
            vectors[g] = {"native_hi": pn, "native_lo": pm}
            for basis, name in itertools.product(spec["bases"], spec["pools"]):
                x, y = pools[a, basis, name], pools[b, basis, name]
                key = basis + "_" + name
                line[key + "_change"] = float(x @ v - y @ v)
                vectors[g][key + "_hi"], vectors[g][key + "_lo"] = x, y
                collisions.append(
                    base
                    | {"g": g, "scope": "one_code", "basis": basis, "pool": name}
                    | certificate(x, y, pn, pm)
                )
            paired.append(line)
            selected.append(line)
        delta = np.asarray([r["native_change"] for r in selected])
        mean, variance = float(np.mean(delta)), float(np.var(delta, ddof=0))
        result = base | {
            "orbit_mean": mean,
            "code_variance": variance,
            "code_sd": float(np.std(delta, ddof=0)),
            "orbit_energy": float(np.dot(delta, delta) / 4),
            "invariant_energy": mean**2,
            "identity_error": abs(float(np.dot(delta, delta) / 4) - mean**2 - variance),
            "positive_codes": int(np.count_nonzero(delta > 0)),
            "negative_codes": int(np.count_nonzero(delta < 0)),
            "min": float(delta.min()),
            "max": float(delta.max()),
            "g0_change": selected[0]["native_change"],
            "oriented_mean": pair["orientation"] * mean
            if pair["orientation"] is not None
            else None,
        }
        pn, pm = (
            np.sum([vectors[g]["native_" + k] for g in (0, 1, 2, 3)], axis=0) / 4
            for k in ("hi", "lo")
        )
        result["orbit_native_tv"] = float(np.linalg.norm(pn - pm, ord=1) / 2)
        for basis, name in itertools.product(spec["bases"], spec["pools"]):
            key = basis + "_" + name
            x, y = (
                np.sum([vectors[g][key + "_" + k] for g in (0, 1, 2, 3)], axis=0) / 4
                for k in ("hi", "lo")
            )
            result[key + "_mean_change"] = float(x @ v - y @ v)
            result[key + "_orbit_residual"] = mean - result[key + "_mean_change"]
            collisions.append(
                base
                | {"g": -1, "scope": "full_C4_mean", "basis": basis, "pool": name}
                | certificate(x, y, pn, pm)
            )
        orbits.append(result)
    summaries = []
    for content, contrast in itertools.product(
        ("ALL", *m["targets"]), sorted({r["contrast"] for r in orbits})
    ):
        group = [
            r
            for r in orbits
            if r["contrast"] == contrast
            and (content == "ALL" or r["content"] == content)
        ]
        invariant = float(np.mean([r["orbit_mean"] ** 2 for r in group]))
        sensitivity = float(np.mean([r["code_variance"] for r in group]))
        energy = float(np.mean([r["orbit_energy"] for r in group]))
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
                "g0_RMS": float(np.sqrt(np.mean([r["g0_change"] ** 2 for r in group]))),
                "oriented_mean": float(np.mean([r["oriented_mean"] for r in group]))
                if group[0]["orientation"] is not None
                else None,
                "all_codes_same_nonzero_sign": sum(
                    r["positive_codes"] == 4 or r["negative_codes"] == 4 for r in group
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
        spec["bases"],
        spec["pools"],
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
    counts = {}
    for name, data, keys in (
        ("native", native, ("unit",)),
        ("pool_coordinates", pool_rows, ("unit", "basis", "pool", "candidate")),
        ("paired_changes", paired, ("pair", "g")),
        ("orbits", orbits, ("pair",)),
        ("collisions", collisions, ("pair", "g", "basis", "pool")),
        ("orbit_summary", summaries, ("content", "contrast")),
        (
            "collision_summary",
            collision_summaries,
            ("scope", "contrast", "basis", "pool"),
        ),
    ):
        counts[name] = table_check(out, name, data, keys, tol)
    summary = read(out / "03/summary.json")
    compare(summary["counts"], counts, tol)
    for a, b in zip(
        summary["results"], [r for r in summaries if r["content"] == "ALL"], strict=True
    ):
        for k in b:
            compare(a[k], b[k], tol)
    if (
        summary["actual_new_calls"] != 648
        or summary["reused_calls"] != 216
        or summary["original_scope_complete"]
        or not summary["no_fit_or_patch"]
    ):
        raise ValueError("Scientific status/accounting mismatch")
    result = {
        "status": "VERIFIED_FINITE_HM58_NOT_GLOBAL",
        "raw_readouts": len(rows),
        "token_proofs": len(proofs),
        "tables": counts,
        "maximum_alternate_error": MAX_ERROR,
        "minimum_candidate_mass": minimum_mass,
        "actual_new_calls": 648,
        "reused_calls": 216,
        "fit_calls": 0,
        "API_calls": 0,
        "training_calls": 0,
        "independent_method_replication": False,
        "original_scope_complete": False,
    }
    (out / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(result)


def main():
    """Audit one declared completed packet without restarting acquisition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.output)


if __name__ == "__main__":
    main()
