#!/usr/bin/env python3
"""Bounded opportunity/actor experiment and independently runnable author charts."""

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
from grain_research.construction import math_conditional_chart as M
from grain_research.construction import math_history_routing as R
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_conditional_chart_hm114.json"
CODE = (
    "configs/grain_conditional_chart_hm114.json",
    "docs/GRAIN_CONDITIONAL_RELATION_CHART_PLAN_HM114.md",
    "grain_research/construction/math_conditional_chart.py",
    "scripts/run_grain_conditional_chart_hm114.py",
    "scripts/score_grain_conditional_chart_hm114.py",
    "tests/test_grain_math_conditional_chart.py",
    "grain_research/construction/math_history_routing.py",
    "grain_research/vector_mechanisms/prefix.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
)


def check(out):
    """Authenticate declared source, program and actual prepared token streams."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    return m


def prepare(out):
    """Seal all new public factorial material before any local model call."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve prior or failed experiments")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    old = H.load(source / "00/manifest.json")
    H.verify_hashes(ROOT, old["code"] | old["source"])
    H.verify_hashes(source, old["inputs"])
    if (
        H.load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM101_NOT_GLOBAL"
    ):
        raise ValueError("Verified frozen source reader required")
    bundle = H.load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Do not change target battery")
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
    prefix = encode(rendered.split("__BODY__")[0])
    designs = []
    for bank, family, (c, b, o) in itertools.product(range(2), range(6), M.SIGNS):
        text = M.material(spec, bank, family, int(c), int(b), int(o))
        designs.append(
            {
                "id": len(designs),
                "bank": bank,
                "family": family,
                "condition": int(c),
                "polarity": int(b),
                "owner": int(o),
                **text,
            }
        )
    contexts = [encode(r["condition_text"] + "\n") for r in designs]
    actions = [encode(r["action_text"] + "\n") for r in designs]
    cw, aw = max(map(len, contexts)), max(map(len, actions))
    current = encode(spec["current"])
    rows, bags = [], {}
    for r, context, action in zip(designs, contexts, actions, strict=True):
        h = list(range(len(prefix), len(prefix) + cw))
        a = list(range(len(prefix) + cw, len(prefix) + cw + aw))
        now = list(range(len(prefix) + cw + aw, len(prefix) + cw + aw + len(current)))
        padding = h[len(context) :] + a[len(action) :]
        ids = (
            prefix
            + context
            + [tok.pad_token_id] * (cw - len(context))
            + action
            + [tok.pad_token_id] * (aw - len(action))
            + current
        )
        # The existing frozen mask API requires every ignored column in H.
        # All routes stay open here, so placing action padding in H changes no real edge.
        h = h + a[len(action) :]
        row = {
            **r,
            "input_ids": ids,
            "history": h,
            "current": now,
            "padding": padding,
            "relay": [p for p in range(len(ids)) if p not in set(h + now)],
        }
        R.route_mask(len(ids), h, row["relay"], now, padding, 1, 1)
        bag = Counter(v for p, v in enumerate(ids) if p not in padding)
        key = r["bank"], r["family"], r["polarity"]
        if key in bags and bags[key] != bag:
            raise ValueError("Condition/actor swap changed actual token bag")
        bags[key] = bag
        rows.append(row)
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Unexpected finite support")
    folder = out / "00"
    folder.mkdir(parents=True)
    H.write(folder / "rows.json", rows)
    H.write(folder / "bundle.json", bundle)
    source_paths = [
        source / n for n in ("00/manifest.json", "00/bundle.json", "verification.json")
    ]
    source_paths.append(system_path)
    H.write(
        folder / "manifest.json",
        {
            "spec": spec,
            "model": old["model"],
            "code": {n: H.sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in source_paths},
            "inputs": H.seals(out, folder),
            "token_bag_groups": len(bags),
            "current_ids": current,
            "current_positions": rows[0]["current"],
            "padding_is_design_not_natural_text": True,
            "original_scope_complete": False,
        },
    )
    print(
        "PREPARED HM114",
        len(rows),
        "public prefixes",
        len(bags),
        "matched bags",
        flush=True,
    )


def raw_unit(out, i):
    """Read only a complete hash/trace-authenticated capture."""
    p = out / f"01/units/u{i:03d}"
    meta = H.load(p.with_suffix(".json"))
    if H.sha(p.with_suffix(".npz")) != meta["sha256"]:
        raise ValueError("Capture changed")
    if (
        meta["trace"]["completed_blocks"] != list(range(17))
        or meta["trace"]["forbidden_calls"]
    ):
        raise ValueError("Invalid causal prefix trace")
    with np.load(p.with_suffix(".npz"), allow_pickle=False) as z:
        return {n: z[n] for n in z.files}


def collect(out):
    """Capture once per new text; resume known commits without budget reset."""
    m = check(out)
    H.verify_hashes(MODEL, m["model"])
    if (out / "01/raw_seal.json").exists():
        raise FileExistsError("Completed collector cannot rerun")
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
                raise RuntimeError("Prior pending call or finite cap; do not restart")
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
            with p.with_suffix(".npz").open("xb") as f:
                np.savez_compressed(f, **arrays)
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
            if status["completed"] % 24 == 0:
                print("HM114", status, flush=True)
        status["state"] = (
            "COLLECTED"
            if status["model_seconds"] <= m["spec"]["max_model_seconds"]
            else "BUDGET_STOP"
        )
        write_json(path, status)
        if status["state"] != "COLLECTED":
            raise TimeoutError("Finite cap crossed; no budget extension")
        H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    print("COLLECTED", status, flush=True)


def build(out):
    """Freeze conditional instruments before exporting new numeric author ledgers."""
    m = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    if H.load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Complete new material required")
    spec, bundle = m["spec"], H.load(out / "00/bundle.json")
    fields = np.empty((2, 6, 2, 2, 2, 2, 6))
    for r in H.load(out / "00/rows.json"):
        raw = raw_unit(out, r["id"])
        ix = (
            r["bank"],
            r["family"],
            (r["condition"] + 1) // 2,
            (r["polarity"] + 1) // 2,
            (r["owner"] + 1) // 2,
        )
        fields[ix] = R.read_current(raw["k"], raw["v"], max(r["current"]), bundle)
    folder = out / "02"
    folder.mkdir()
    np.savez(folder / "fields.npz", fields=fields)
    charts, coefficient_rows = {}, []
    for bank, family, unit in itertools.product(range(2), range(6), range(2)):
        corners = fields[bank, family, :, :, :, unit].reshape(8, 6)
        beta = M.decompose(corners)
        for character, q in itertools.product(range(8), range(6)):
            coefficient_rows.append(
                {
                    "bank": bank,
                    "family": family,
                    "unit": unit,
                    "character": character,
                    "target": spec["targets"][q],
                    "coefficient": float(beta[character, q]),
                }
            )
        charts[f"{bank}:{family}:{unit}"] = M.compile_chart(
            fields[bank, family, :, :, 1, unit], spec["rank_rtol"]
        )
    H.table(folder / "factorial_coefficients.csv", coefficient_rows)
    H.write(folder / "calibration.json", charts)
    H.write(folder / "instrument_seal.json", H.seals(out, folder))
    observations, truth = [], []
    for bank, family, unit, a, b, pi in itertools.product(
        range(2),
        range(6),
        range(2),
        spec["levels"],
        spec["responses"],
        spec["positive_condition_shares"],
    ):
        counts = M.event_counts(a, b, pi, spec["events_per_profile"])
        means = M.integrate(fields[bank, family, :, :, 1, unit], counts)
        rid = len(observations)
        observations.append(
            {
                "id": rid,
                "bank": bank,
                "family": family,
                "unit": unit,
                "means": means.tolist(),
                "counts": counts.sum(1).tolist(),
                "qualification": "own_actual",
            }
        )
        truth.append(
            {
                "id": rid,
                "level": a,
                "response": b,
                "share": pi,
                "event_counts": counts.tolist(),
            }
        )
    special = [
        {
            "id": 648,
            "bank": 0,
            "family": 0,
            "unit": 0,
            "means": None,
            "counts": [200, 200],
            "qualification": "other_actual",
        },
        {
            "id": 649,
            "bank": 0,
            "family": 0,
            "unit": 0,
            "means": None,
            "counts": [400, 0],
            "qualification": "own_actual",
        },
    ]
    H.write(folder / "observations.json", observations + special)
    H.write(folder / "truth.json", truth)
    H.write(
        folder / "observation_seal.json",
        {
            str(p.relative_to(out)): H.sha(p)
            for p in (folder / "observations.json", folder / "truth.json")
        },
    )
    print(
        "BUILT24 fixed charts,648 numeric unit-profiles +2 null controls;not new humans",
        flush=True,
    )


def evaluate(out):
    """Assess construction, scoped opportunity effects and untuned transport."""
    m = check(out)
    H.verify_hashes(out, H.load(out / "02/instrument_seal.json"))
    H.verify_hashes(out, H.load(out / "02/observation_seal.json"))
    spec, charts = m["spec"], H.load(out / "02/calibration.json")
    observations, truths = (
        H.load(out / "02/observations.json"),
        H.load(out / "02/truth.json"),
    )
    truth = {r["id"]: r for r in truths}
    folder = out / "03"
    rows = []
    for mode in ("source", "local"):
        cold = H.load(folder / f"{mode}.json")
        if (
            cold["neural_runtime_loaded"]
            or cold["truth_input"]
            or len(cold["results"]) != len(observations)
        ):
            raise ValueError("Cold dependency/scope failure")
        for obs, score in zip(observations, cold["results"], strict=True):
            if score["id"] != obs["id"] or score["mode"] != mode:
                raise ValueError("Cold score bound to the wrong observation")
            if obs["id"] not in truth:
                if score["coordinates"] is not None:
                    raise ValueError("Missing scope was fabricated as observed")
                continue
            t = truth[obs["id"]]
            chart = charts[
                f"{0 if mode == 'source' else obs['bank']}:{obs['family']}:{obs['unit']}"
            ]
            y, offset, gain = (
                np.asarray(obs["means"]),
                np.asarray(chart["offset"]),
                np.asarray(chart["gain"]),
            )
            alternate = M.condition_projection(chart, y, spec["rank_rtol"])
            coordinates = np.asarray(score["coordinates"])
            if not np.allclose(coordinates, alternate, atol=spec["identity_atol"]):
                raise ValueError("Alternate two-condition recovery differs")
            explicit = [
                np.repeat(
                    np.array([offset[c] - gain[c], offset[c] + gain[c]]),
                    t["event_counts"][c],
                    axis=0,
                ).mean(0)
                for c in range(2)
            ]
            if mode == "local" and not np.allclose(
                y, explicit, atol=spec["identity_atol"]
            ):
                raise ValueError("Explicit event integration differs")
            rows.append(
                {
                    "id": obs["id"],
                    "mode": mode,
                    "bank": obs["bank"],
                    "family": obs["family"],
                    "unit": obs["unit"],
                    "level": t["level"],
                    "response": t["response"],
                    "share": t["share"],
                    "observed_level": float(coordinates[0]),
                    "observed_response": float(coordinates[1]),
                    "level_error": float(coordinates[0] - t["level"]),
                    "response_error": float(coordinates[1] - t["response"]),
                    "identified_coordinate_error": float(
                        np.max(
                            abs(
                                coordinates
                                - np.asarray(chart["identified_projector"])
                                @ [t["level"], t["response"]]
                            )
                        )
                    ),
                    "raw_condition_difference_RMS": float(
                        np.sqrt(np.mean((y[1] - y[0]) ** 2))
                    ),
                    "residual_RMS": float(
                        np.sqrt(np.mean(np.asarray(score["residual"]) ** 2))
                    ),
                    "rank": chart["rank"],
                    "choice_error": abs(score["choice"][1] - t["share"]),
                }
            )
    H.table(folder / "author_charts.csv", rows)
    with np.load(out / "02/fields.npz", allow_pickle=False) as z:
        fields = z["fields"]
    mechanisms = []
    for bank, unit in itertools.product(range(2), range(2)):
        scoped, foreign, interaction = [], [], []
        for family in range(6):
            k = fields[bank, family, :, :, :, unit]
            own = (k[:, 1, 1] - k[:, 0, 1]) / 2
            other = (k[:, 1, 0] - k[:, 0, 0]) / 2
            scoped.append(own)
            foreign.append(other)
            interaction.append((own[1] - own[0]) - (other[1] - other[0]))
        subset = [r for r in rows if r["bank"] == bank and r["unit"] == unit]
        const = [r for r in subset if r["mode"] == "local" and r["response"] == 0]
        source = [r for r in subset if r["mode"] == "source"]
        mechanisms.append(
            {
                "bank": bank,
                "unit": unit,
                "own_gain_RMS": float(np.sqrt(np.mean(np.array(scoped) ** 2))),
                "foreign_gain_RMS": float(np.sqrt(np.mean(np.array(foreign) ** 2))),
                "condition_by_action_scope_RMS": float(
                    np.sqrt(np.mean(np.array(interaction) ** 2))
                ),
                "raw_condition_difference_with_B0_RMS": float(
                    np.sqrt(
                        np.mean([r["raw_condition_difference_RMS"] ** 2 for r in const])
                    )
                ),
                "source_chart_response_error_RMS": float(
                    np.sqrt(np.mean([r["response_error"] ** 2 for r in source]))
                ),
                "source_chart_residual_RMS": float(
                    np.sqrt(np.mean([r["residual_RMS"] ** 2 for r in source]))
                ),
            }
        )
    H.table(folder / "mechanisms.csv", mechanisms)
    coupling = {}
    for bank, unit, owner in itertools.product(range(2), range(2), range(2)):
        gains = (
            fields[bank, :, :, 1, owner, unit] - fields[bank, :, :, 0, owner, unit]
        ) / 2
        coupling[f"{bank}:{unit}:{owner}"] = M.target_coupling(
            gains.transpose(1, 0, 2), spec["rank_rtol"]
        )
    H.write(folder / "target_coupling.json", coupling)
    local = [r for r in rows if r["mode"] == "local"]
    summary = {
        "packet": "HM114",
        "new_public_prefixes": len(H.load(out / "00/rows.json")),
        "actual_status": H.load(out / "01/status.json"),
        "numeric_unit_profiles": len(truths),
        "profiles_per_bank_family": 27,
        "two_units_not_independent_authors": True,
        "rank_distribution": dict(Counter(c["rank"] for c in charts.values())),
        "local_coordinate_max_error": max(
            max(abs(r["level_error"]), abs(r["response_error"])) for r in local
        ),
        "local_identified_coordinate_max_error": max(
            r["identified_coordinate_error"] for r in local
        ),
        "choice_max_error": max(r["choice_error"] for r in rows),
        "mechanisms": mechanisms,
        "mixed_evidence_ranks": {key: chart["rank"] for key, chart in coupling.items()},
        "new_predictor_fits": 0,
        "new_human_personality_accuracy": False,
        "local_exact_recovery_is_algebraic_control": True,
        "original_scope_complete": False,
    }
    H.write(folder / "summary.json", summary)
    H.write(folder / "output_seal.json", H.seals(out, folder))
    print("EVALUATED", summary, flush=True)


def verify(out):
    """Replay raw six-target reading and all declared cold-scorer equations."""
    m = check(out)
    for seal in (
        "01/raw_seal.json",
        "02/instrument_seal.json",
        "02/observation_seal.json",
        "03/output_seal.json",
    ):
        H.verify_hashes(out, H.load(out / seal))
    with np.load(out / "02/fields.npz", allow_pickle=False) as z:
        fields = z["fields"]
    bundle = H.load(out / "00/bundle.json")
    error = 0.0
    for r in H.load(out / "00/rows.json"):
        primitive = raw_unit(out, r["id"])
        actual = R.read_current(
            primitive["k"], primitive["v"], max(r["current"]), bundle
        )
        ix = (
            r["bank"],
            r["family"],
            (r["condition"] + 1) // 2,
            (r["polarity"] + 1) // 2,
            (r["owner"] + 1) // 2,
        )
        error = max(error, float(np.max(abs(actual - fields[ix]))))
    summary = H.load(out / "03/summary.json")
    charts = H.load(out / "02/calibration.json")
    observations = H.load(out / "02/observations.json")
    truths = {r["id"]: r for r in H.load(out / "02/truth.json")}
    cold_error, explicit_error, chart_error = 0.0, 0.0, 0.0
    for key, chart in charts.items():
        bank, family, unit = map(int, key.split(":"))
        actual_chart = M.compile_chart(
            fields[bank, family, :, :, 1, unit], m["spec"]["rank_rtol"]
        )
        if chart["rank"] != actual_chart["rank"]:
            raise ValueError("Stored calibration rank differs")
        for name in ("offset", "gain", "design", "inverse", "identified_projector"):
            chart_error = max(
                chart_error,
                float(np.max(abs(np.asarray(chart[name]) - actual_chart[name]))),
            )
    for mode in ("source", "local"):
        cold = H.load(out / f"03/{mode}.json")
        if (
            len(cold["results"]) != len(observations)
            or cold["truth_input"]
            or cold["neural_runtime_loaded"]
        ):
            raise ValueError("Invalid independent cold process")
        for obs, score in zip(observations, cold["results"], strict=True):
            if obs["id"] != score["id"] or score["mode"] != mode:
                raise ValueError("Cold score binding differs")
            if obs["id"] not in truths:
                status = (
                    "UNAVAILABLE_OWNERSHIP"
                    if obs["qualification"] != "own_actual"
                    else "UNAVAILABLE_CONDITION"
                )
                if score["coordinates"] is not None or score["status"] != status:
                    raise ValueError("Null control fabricated")
                continue
            chart = charts[
                f"{0 if mode == 'source' else obs['bank']}:{obs['family']}:{obs['unit']}"
            ]
            y = np.asarray(obs["means"])
            alternate = M.condition_projection(chart, y, m["spec"]["rank_rtol"])
            centered = (y - np.asarray(chart["offset"])).ravel()
            residual = centered - np.asarray(chart["design"]) @ alternate
            for name, value in (
                ("coordinates", alternate),
                ("residual", residual),
                ("choice", np.asarray(obs["counts"]) / sum(obs["counts"])),
                ("pooled_observation", np.average(y, axis=0, weights=obs["counts"])),
                ("raw_condition_difference", y[1] - y[0]),
                ("identified_projector", chart["identified_projector"]),
            ):
                cold_error = max(
                    cold_error, float(np.max(abs(np.asarray(score[name]) - value)))
                )
            truth = truths[obs["id"]]
            endpoints = fields[obs["bank"], obs["family"], :, :, 1, obs["unit"]]
            explicit = [
                np.repeat(endpoints[c], truth["event_counts"][c], axis=0).mean(0)
                for c in range(2)
            ]
            explicit_error = max(
                explicit_error, float(np.max(abs(np.asarray(explicit) - y)))
            )
            expected_status = (
                "IDENTIFIED" if chart["rank"] == 2 else "PARTIAL_IDENTIFICATION"
            )
            if score["status"] != expected_status:
                raise ValueError("Rank status differs")
    for key, stored in H.load(out / "03/target_coupling.json").items():
        bank, unit, owner = map(int, key.split(":"))
        gains = (
            fields[bank, :, :, 1, owner, unit] - fields[bank, :, :, 0, owner, unit]
        ) / 2
        actual = M.target_coupling(gains.transpose(1, 0, 2), m["spec"]["rank_rtol"])
        if stored != actual:
            raise ValueError("Target geometry replay differs")
    if (
        error > m["spec"]["identity_atol"]
        or summary["choice_max_error"] > m["spec"]["identity_atol"]
        or max(cold_error, explicit_error, chart_error) > m["spec"]["identity_atol"]
    ):
        raise ValueError("Raw replay or choice failure")
    if summary["local_identified_coordinate_max_error"] > m["spec"]["identity_atol"]:
        raise ValueError("Identified chart arithmetic failed")
    full_capacity = summary["rank_distribution"] == {"2": 24}
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM114_NOT_GLOBAL"
            if full_capacity
            else "VERIFIED_FINITE_HM114_PARTIAL_CAPACITY_NOT_GLOBAL",
            "raw_reader_replay_max_error": error,
            "new_prefix_commits_checked": 96,
            "numeric_cold_chart_rows_checked": 1296,
            "null_controls_checked": 4,
            "cold_alternate_projection_max_error": cold_error,
            "explicit_integer_integral_max_error": explicit_error,
            "calibration_replay_max_error": chart_error,
            "same_agent_alternate_formula_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    print(
        "VERIFIED HM114 finite scoped response chart,not psychological completion",
        flush=True,
    )


def main():
    """Select one explicit finite stage; never automatically restart a queue."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "phase", choices=("prepare", "collect", "build", "evaluate", "verify")
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    globals()[a.phase](a.output.resolve())


if __name__ == "__main__":
    main()
