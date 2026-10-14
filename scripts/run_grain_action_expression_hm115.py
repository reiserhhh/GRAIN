#!/usr/bin/env python3
"""Freeze, run and verify one expression/actor mechanism battery, not a search."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_action_expression as M
from grain_research.construction import math_conditional_chart as C
from grain_research.construction import math_history_routing as R
from scripts import run_grain_author_evidence_merge_hm102 as H

SPEC = ROOT / "configs/grain_action_expression_hm115.json"
CODE = (
    "configs/grain_action_expression_hm115.json",
    "docs/GRAIN_ACTION_EXPRESSION_PLAN_HM115.md",
    "grain_research/construction/math_action_expression.py",
    "scripts/run_grain_action_expression_hm115.py",
    "tests/test_grain_math_action_expression.py",
    "grain_research/construction/math_conditional_chart.py",
    "grain_research/construction/math_history_routing.py",
    "grain_research/vector_mechanisms/prefix.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
)
OBJECTS = ("k", "v", "mean_v", "uniform", "weighted")


def check(out):
    """Authenticate scientific code, sources and prepared token material."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    return m


def prepare(out):
    """Seal all material and an unmodified source reader before model execution."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    if out.exists():
        raise FileExistsError("Do not replace previous or failed runs")
    spec = H.load(SPEC)
    source, predecessor = ROOT / spec["source"], ROOT / spec["predecessor"]
    old = H.load(source / "00/manifest.json")
    H.verify_hashes(ROOT, old["code"] | old["source"])
    H.verify_hashes(source, old["inputs"])
    if (
        H.load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM101_NOT_GLOBAL"
    ):
        raise ValueError("Frozen source required")
    if (
        H.load(predecessor / "verification.json")["status"]
        != "VERIFIED_FINITE_HM114_NOT_GLOBAL"
    ):
        raise ValueError("Completed predecessor required")
    bundle = H.load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Target selection is forbidden")
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system_path = ROOT / "configs/grain_math_passive_event_composition_v1.json"
    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": H.load(system_path)["system"]},
            {"role": "user", "content": "__BODY__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    prefix, current = encode(rendered.split("__BODY__")[0]), encode(spec["current"])
    designs = []
    for v, w, f, (c, b, o) in itertools.product(range(3), range(2), range(6), C.SIGNS):
        designs.append(
            {
                "id": len(designs),
                "variant": v,
                "wording": w,
                "family": f,
                "condition": int(c),
                "polarity": int(b),
                "owner": int(o),
                **M.material(spec, v, w, f, int(c), int(b), int(o)),
            }
        )
    contexts = [encode(r["condition_text"] + "\n") for r in designs]
    actions = [encode(r["action_text"] + "\n") for r in designs]
    cw, aw = max(map(len, contexts)), max(map(len, actions))
    rows, bags = [], {}
    for design, context, action in zip(designs, contexts, actions, strict=True):
        history = list(range(len(prefix), len(prefix) + cw))
        act = list(range(len(prefix) + cw, len(prefix) + cw + aw))
        now = list(range(len(prefix) + cw + aw, len(prefix) + cw + aw + len(current)))
        padding = history[len(context) :] + act[len(action) :]
        ids = (
            prefix
            + context
            + [tok.pad_token_id] * (cw - len(context))
            + action
            + [tok.pad_token_id] * (aw - len(action))
            + current
        )
        # Existing mask API requires ignored columns in history; no real edge changes.
        history += act[len(action) :]
        row = {
            **design,
            "input_ids": ids,
            "history": history,
            "current": now,
            "padding": padding,
            "relay": [p for p in range(len(ids)) if p not in set(history + now)],
        }
        R.route_mask(len(ids), history, row["relay"], now, padding, 1, 1)
        bag = Counter(t for p, t in enumerate(ids) if p not in padding)
        key = (
            design["variant"],
            design["wording"],
            design["family"],
            design["condition"],
            design["polarity"],
        )
        if key in bags and bags[key] != bag:
            raise ValueError("Owner swap changed actual token multiset")
        bags[key] = bag
        rows.append(row)
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete planned support")
    token_pairs = [
        [[encode(b) for b in family] for family in variant]
        for variant in spec["behaviours"]
    ]
    folder = out / "00"
    folder.mkdir(parents=True)
    H.write(folder / "rows.json", rows)
    H.write(folder / "bundle.json", bundle)
    H.write(folder / "behavior_tokens.json", token_pairs)
    sources = [
        source / "00/manifest.json",
        source / "00/bundle.json",
        source / "verification.json",
        predecessor / "verification.json",
        system_path,
    ]
    H.write(
        folder / "manifest.json",
        {
            "spec": spec,
            "model": old["model"],
            "code": {n: H.sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in sources},
            "inputs": H.seals(out, folder),
            "owner_token_bag_groups": len(bags),
            "current_ids": current,
            "current_positions": rows[0]["current"],
            "same_opportunity_for_both_actors": True,
            "original_scope_complete": False,
        },
    )
    print("PREPARED HM115", len(rows), "prefixes", len(bags), "owner bags", flush=True)


def raw_unit(out, index):
    """Read only fully committed prefix arrays with completed causal traces."""
    path = out / f"01/units/u{index:03d}"
    meta = H.load(path.with_suffix(".json"))
    if H.sha(path.with_suffix(".npz")) != meta["sha256"]:
        raise ValueError("Changed capture")
    if (
        meta["trace"]["completed_blocks"] != list(range(17))
        or meta["trace"]["forbidden_calls"]
    ):
        raise ValueError("Invalid prefix execution trace")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def collect(out):
    """Execute each fresh prefix once with a retained finite budget and checkpoint."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    m = check(out)
    H.verify_hashes(MODEL, m["model"])
    if (out / "01/raw_seal.json").exists():
        raise FileExistsError("Do not rerun a terminal capture")
    (out / "01/units").mkdir(parents=True, exist_ok=True)
    path = out / "01/status.json"
    status = (
        H.load(path)
        if path.exists()
        else {
            "state": "RUNNING",
            "completed": 0,
            "attempts": 0,
            "model_seconds": 0.0,
            "pending": None,
        }
    )
    with collector_guard(out):
        observer = load_observer()
        for row in H.load(out / "00/rows.json"):
            p = out / f"01/units/u{row['id']:03d}"
            if p.with_suffix(".json").exists():
                raw_unit(out, row["id"])
                continue
            if (
                status["pending"] is not None
                or status["model_seconds"] >= m["spec"]["max_model_seconds"]
            ):
                raise RuntimeError("Pending failure or finite cap; do not reset")
            status.update(pending=row["id"], attempts=status["attempts"] + 1)
            write_json(path, status)
            begin = time.monotonic()
            try:
                arrays, trace = R.capture_routed_prefix(observer, row, 1, 1)
            except Exception:
                status.update(
                    state="FAILED",
                    model_seconds=status["model_seconds"] + time.monotonic() - begin,
                )
                write_json(path, status)
                raise
            seconds = time.monotonic() - begin
            with p.with_suffix(".npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            H.write(
                p.with_suffix(".json"),
                {
                    "trace": trace,
                    "sha256": H.sha(p.with_suffix(".npz")),
                    "seconds": seconds,
                },
            )
            status.update(
                completed=status["completed"] + 1,
                model_seconds=status["model_seconds"] + seconds,
                pending=None,
            )
            write_json(path, status)
            if status["completed"] % 48 == 0:
                print("HM115", status, flush=True)
        status["state"] = (
            "COLLECTED"
            if status["model_seconds"] <= m["spec"]["max_model_seconds"]
            else "BUDGET_STOP"
        )
        write_json(path, status)
        if status["state"] != "COLLECTED":
            raise TimeoutError("Finite cap crossed; retain results without renewal")
        H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    print("COLLECTED HM115", status, flush=True)


def observations(out):
    """Extract fixed native coordinates and both inherited readout units."""
    m = check(out)
    bundle, objects = H.load(out / "00/bundle.json"), {}
    for row in H.load(out / "00/rows.json"):
        raw = raw_unit(out, row["id"])
        if row["current"] != m["current_positions"]:
            raise ValueError("Position alignment cannot be fitted")
        readings = R.read_current(raw["k"], raw["v"], max(row["current"]), bundle)
        values = {
            "k": raw["k"].ravel(),
            "v": raw["v"].ravel(),
            "mean_v": raw["v"].mean(0).ravel(),
            "uniform": readings[0],
            "weighted": readings[1],
        }
        index = (
            row["variant"],
            row["wording"],
            row["family"],
            (row["condition"] + 1) // 2,
            (row["polarity"] + 1) // 2,
            (row["owner"] + 1) // 2,
        )
        for name, value in values.items():
            if name not in objects:
                objects[name] = np.empty((3, 2, 6, 2, 2, 2, len(value)))
            objects[name][index] = value
    return objects


def comparisons():
    """Return all frozen control, paraphrase and new-instance comparison addresses."""
    return [("opportunity_only", (0, 0), (0, 1))] + [
        (f"{'rewording' if v == 1 else 'new_instance'}_w{w}", (0, w), (v, w))
        for v in (1, 2)
        for w in range(2)
    ]


def analyze(out):
    """Evaluate fixed fields without loading neural runtime or fitting any metric."""
    m = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    if H.load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Full finite support required")
    folder = out / "02"
    folder.mkdir()
    objects = observations(out)
    np.savez_compressed(folder / "fields.npz", **objects)
    matches, details, permutations, author_rows, scope_rows, characters = (
        [],
        [],
        {},
        [],
        [],
        {},
    )
    for name, values in objects.items():
        effects = M.actor_fields(values)
        for v, w, f in itertools.product(range(3), range(2), range(6)):
            if name not in characters:
                characters[name] = np.empty((3, 2, 6, 8, values.shape[-1]))
            characters[name][v, w, f] = C.decompose(values[v, w, f].reshape(8, -1))
        for v, w in itertools.product(range(3), range(2)):
            scope_rows.append(
                {
                    "object": name,
                    "variant": v,
                    "wording": w,
                    **{
                        s + "_RMS": float(np.sqrt(np.mean(effects[s][v, w] ** 2)))
                        for s in effects
                    },
                }
            )
        for comparison, source, target in comparisons():
            for scope in ("own", "even", "odd"):
                s, t = effects[scope][source], effects[scope][target]
                match = M.direction_match(
                    s,
                    t,
                    m["spec"]["direction_min_norm"],
                    m["spec"]["permutation_tie_atol"],
                )
                key = f"{comparison}:{name}:{scope}"
                values = match.pop("permutation_values")
                if values is not None:
                    permutations[key] = np.asarray(values)
                details.append(
                    {"comparison": comparison, "object": name, "scope": scope, **match}
                )
                matches.append(
                    {
                        "comparison": comparison,
                        "object": name,
                        "scope": scope,
                        "status": match["status"],
                        "matched_mean_cosine": match["matched_mean_cosine"],
                        "permutation_mean": match.get("permutation_mean"),
                        "matching_rank_with_ties": match.get("matching_rank_with_ties"),
                        "matched_minus_permutation_mean": match.get(
                            "matched_minus_permutation_mean"
                        ),
                    }
                )
                sc, tc = M.author_contrasts(s), M.author_contrasts(t)
                for f in range(6):
                    sn, tn = np.linalg.norm(sc[f]), np.linalg.norm(tc[f])
                    cosine = (
                        float(np.dot(sc[f], tc[f]) / (sn * tn))
                        if min(sn, tn) > m["spec"]["direction_min_norm"]
                        else None
                    )
                    author_rows.append(
                        {
                            "comparison": comparison,
                            "object": name,
                            "scope": scope,
                            "family": m["spec"]["targets"][f],
                            "source_contrast_norm": float(sn),
                            "target_contrast_norm": float(tn),
                            "contrast_cosine": cosine,
                            "difference_RMS": float(
                                np.sqrt(np.mean((tc[f] - sc[f]) ** 2))
                            ),
                        }
                    )
    lexical, vocabulary = M.lexical_fields(H.load(out / "00/behavior_tokens.json"))
    for comparison, source, target in comparisons():
        if source[0] == target[0]:
            continue
        match = M.direction_match(
            lexical[source[0]], lexical[target[0]], m["spec"]["direction_min_norm"]
        )
        permutations[f"{comparison}:lexical:role_blind"] = np.asarray(
            match.pop("permutation_values")
        )
        details.append(
            {
                "comparison": comparison,
                "object": "lexical",
                "scope": "role_blind",
                **match,
            }
        )
        matches.append(
            {
                "comparison": comparison,
                "object": "lexical",
                "scope": "role_blind",
                "status": match["status"],
                "matched_mean_cosine": match["matched_mean_cosine"],
                "permutation_mean": match.get("permutation_mean"),
                "matching_rank_with_ties": match.get("matching_rank_with_ties"),
                "matched_minus_permutation_mean": match.get(
                    "matched_minus_permutation_mean"
                ),
            }
        )
    np.savez_compressed(folder / "characters.npz", **characters)
    np.savez_compressed(folder / "permutations.npz", **permutations)
    H.table(folder / "matching_summaries.csv", matches)
    H.table(folder / "author_contrasts.csv", author_rows)
    H.table(folder / "actor_scope_RMS.csv", scope_rows)
    H.write(folder / "matching_details.json", details)
    H.write(
        folder / "summary.json",
        {
            "packet": "HM115",
            "actual_status": H.load(out / "01/status.json"),
            "matched_actor_opportunities": True,
            "comparison_cases": len(matches),
            "cases": matches,
            "scope_RMS": scope_rows,
            "lexical_vocabulary_size": len(vocabulary),
            "neural_runtime_loaded_in_analysis": any(
                n in sys.modules for n in ("torch", "transformers")
            ),
            "new_fits": 0,
            "human_personality_accuracy": False,
            "original_scope_complete": False,
            "interpretation": "Correspondence is supplied behavior design,not identical psychological traits;odd is actor-assignment interaction,not uniquely identified binding subspace",
        },
    )
    H.write(folder / "output_seal.json", H.seals(out, folder))
    print(
        "ANALYZED HM115",
        len(matches),
        "fixed comparisons",
        len(author_rows),
        "numeric author contrasts",
        flush=True,
    )


def verify(out):
    """Replay actual captures, independent factorial effects and numeric integrals."""
    m = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    actual = observations(out)
    errors = {
        "raw_replay": 0.0,
        "factorial_effect": 0.0,
        "author_integral": 0.0,
        "direction_dot": 0.0,
    }
    with np.load(out / "02/fields.npz", allow_pickle=False) as z:
        for name in OBJECTS:
            errors["raw_replay"] = max(
                errors["raw_replay"], float(np.max(abs(actual[name] - z[name])))
            )
    for name, x in actual.items():
        f = M.actor_fields(x)
        for v, w, family in itertools.product(range(3), range(2), range(6)):
            beta = C.decompose(x[v, w, family].reshape(8, -1))
            for c, sign in enumerate((-1, 1)):
                even, odd = beta[2] + sign * beta[3], beta[6] + sign * beta[7]
                errors["factorial_effect"] = max(
                    errors["factorial_effect"],
                    float(np.max(abs(even - f["even"][v, w, family, c]))),
                    float(np.max(abs(odd - f["odd"][v, w, family, c]))),
                )
            direct = sum(
                (
                    (0.3 * x[v, w, family, c, 0, 1] + 0.7 * x[v, w, family, c, 1, 1])
                    - (0.7 * x[v, w, family, c, 0, 1] + 0.3 * x[v, w, family, c, 1, 1])
                )
                / 12
                for c in range(2)
            )
            compiled = M.author_contrasts(f["own"][v, w])[family]
            errors["author_integral"] = max(
                errors["author_integral"], float(np.max(abs(direct - compiled)))
            )
    for row in H.load(out / "02/matching_details.json"):
        if row["status"] != "OBSERVED_DIRECTION" or row["object"] == "lexical":
            continue
        _, source, target = next(r for r in comparisons() if r[0] == row["comparison"])
        f = M.actor_fields(actual[row["object"]])[row["scope"]]
        s, t = f[source], f[target]
        direct = [
            [
                float(
                    np.dot(s[i, c], t[i, c])
                    / (np.linalg.norm(s[i, c]) * np.linalg.norm(t[i, c]))
                )
                for c in range(2)
            ]
            for i in range(6)
        ]
        errors["direction_dot"] = max(
            errors["direction_dot"],
            float(np.max(abs(np.asarray(direct) - row["matched_family_condition"]))),
        )
    if max(errors.values()) > m["spec"]["identity_atol"]:
        raise ValueError("Independent formula replay differs")
    if H.load(out / "02/summary.json")["neural_runtime_loaded_in_analysis"]:
        raise ValueError("Numerical analysis loaded a model runtime")
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM115_NOT_GLOBAL",
            "errors": errors,
            "captures_checked": len(H.load(out / "00/rows.json")),
            "same_agent_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    print("VERIFIED HM115", errors, flush=True)


def main():
    """Expose bounded stages with explicit fresh or existing artifact roots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
