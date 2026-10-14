#!/usr/bin/env python3
"""Versioned text evidence, source-learned dependence and conditional updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_history import (
    active_records,
    fit_source,
    restore_summary,
    score_records,
)
from grain_research.joint_conditional_observation import read_events
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_joint_conditional_observation import dump, read, sha

BASE = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1"
DEFAULT = BASE / "evidence_history_r1"
PREVIOUS = BASE / "joint_conditional_observation_r1"
NATURAL = BASE / "relational_evidence_r1/natural/private_relations.jsonl"
TARGETS = np.array([[1, 0], [0, 2], [1, 0.5]], float)


def source_schedule(cfg):
    """Observed block/event/repeated-report addresses, shared only as a design."""
    rows = []
    for e in range(cfg["blocks"] * cfg["events_per_block"]):
        block = e // cfg["events_per_block"]
        for r in range(cfg["reports_per_event"]):
            rows.append(
                {
                    "record_id": f"e{e}r{r}",
                    "event_id": f"e{e}",
                    "block_id": f"b{block}",
                    "condition": -1 if block % 2 == 0 else 1,
                    "kind": "new" if r == 0 else "repeat",
                    "value_index": len(rows),
                }
            )
    return rows


def stream_schedule(cfg):
    """Fixed explicit histories; operation identities never depend on outcomes."""
    source = source_schedule(cfg)
    first = [r for r in source if r["kind"] == "new"]
    history = []
    for e, row in enumerate(first):
        row = dict(row)
        if e == 0:
            row["value_index"] = len(source)
        history.append(row)
        if e == 5:
            history.append(
                {"record_id": "alias0", "kind": "alias", "parent_id": "e0r0"}
            )
        if e == 11:
            history.append(
                {
                    "record_id": "summary12",
                    "kind": "summary",
                    "parents": [
                        r["record_id"] for r in active_records(history)["records"]
                    ],
                }
            )
        if e == 17:
            history.append(
                dict(first[0], record_id="correct0", kind="revise", parent_id="e0r0")
            )
    history.append({"record_id": "retract23", "kind": "retract", "parent_id": "e23r0"})
    history.extend(r for r in source if r["kind"] == "repeat")
    return history


def begin(out, name):
    """Check frozen code/inputs and refuse to overwrite a completed stage."""
    for path, expected in read(out / "freeze.json").items():
        if sha(path) != expected:
            raise ValueError("Frozen input changed: " + path)
    folder = out / name
    folder.mkdir(exist_ok=False)
    return read(out / "config.json"), folder


def prepare(out):
    """Freeze source readers and the bounded new history design before any draw."""
    cfg_file = ROOT / "configs/grain_evidence_history.json"
    cfg = read(cfg_file)
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        cfg_file,
        ROOT / "docs/GRAIN_EVIDENCE_HISTORY_PLAN_V1.md",
        ROOT / "grain_research/evidence_history.py",
        ROOT / "grain_research/joint_conditional_observation.py",
        Path(__file__),
        PREVIOUS / "public_bank.csv",
        PREVIOUS / "config.json",
        PREVIOUS / "cold_source/observations.csv",
        PREVIOUS / "cold_application/observations.csv",
        PREVIOUS / "paired_followup/fit/meters.json",
        PREVIOUS / "paired_followup/fit/seal.json",
        NATURAL,
    ]
    dump(out / "freeze.json", {str(p): sha(p) for p in paths})
    dump(out / "config.json", cfg)
    dump(
        out / "schedule.json",
        {"source": source_schedule(cfg), "stream": stream_schedule(cfg)},
    )
    print("history design frozen", flush=True)


def observed_bank():
    """Reapply one frozen calibrated event reader to existing actual neural reads."""
    bank = pd.read_csv(PREVIOUS / "public_bank.csv").sort_values("case_id")
    frames = pd.concat(
        [
            pd.read_csv(PREVIOUS / c / "observations.csv")
            for c in ["cold_source", "cold_application"]
        ]
    )
    bank = bank.merge(frames, on="case_id", validate="one_to_one").sort_values(
        "case_id"
    )
    cfg = read(PREVIOUS / "config.json")
    mp = PREVIOUS / "paired_followup/fit/meters.json"
    assert sha(mp) == read(mp.parent / "seal.json")["meters_sha256"]
    meters = read(mp)
    y = bank[cfg["channels"]].to_numpy(float)
    y = np.c_[y, y[:, 8] * y[:, 9]]
    score = np.zeros(len(bank))
    for z in [-1, 1]:
        mask = bank.condition == z
        score[mask] = read_events(y[mask], meters[f"nli_product__{z}"])[:, 0]
    return bank.reset_index(drop=True), score


def draw_persons(cfg, rng, n):
    """Generate bounded private person parameters and correlated event outcomes."""
    beta = np.column_stack(
        [rng.uniform(*cfg[k], size=n) for k in ["level_range", "response_range"]]
    )
    signs = np.zeros((n, cfg["blocks"] * cfg["events_per_block"]), int)
    for b in range(cfg["blocks"]):
        z = -1 if b % 2 == 0 else 1
        m = beta[:, 0] + beta[:, 1] * z
        assert np.all(np.abs(m) < 1)
        probability = (1 + m) / 2
        common = rng.random(n) < probability
        share = rng.random(n) < cfg["shared_block_probability"]
        independent = rng.random((n, cfg["events_per_block"])) < probability[:, None]
        s = np.where(share[:, None], common[:, None], independent)
        signs[:, b * cfg["events_per_block"] : (b + 1) * cfg["events_per_block"]] = (
            2 * s - 1
        )
    confirmation = rng.choice([-1, 1], size=signs.shape)
    variants = rng.random((n, signs.shape[1], cfg["reports_per_event"]))
    return beta, signs, confirmation, variants


def cases(cfg, bank, signs, confirmation, draws, variants):
    """Address actual text variants for an event; not latent truth as score input."""
    lookup = {
        (int(r.condition), int(r.worry), int(r.reassurance), int(r.variant)): i
        for i, r in bank.iterrows()
        if r.condition in [-1, 1]
    }
    result = np.zeros((len(signs), signs.shape[1] * cfg["reports_per_event"]), int)
    for e in range(signs.shape[1]):
        z = -1 if (e // cfg["events_per_block"]) % 2 == 0 else 1
        for r in range(cfg["reports_per_event"]):
            vs = np.array(variants)[
                np.floor(draws[:, e, r] * len(variants)).astype(int)
            ]
            result[:, e * cfg["reports_per_event"] + r] = [
                lookup[(z, int(s), int(t), int(v))]
                for s, t, v in zip(signs[:, e], confirmation[:, e], vs)
            ]
    wrong = [
        lookup[(-1, -int(s), int(t), int(bank.iloc[i].variant))]
        for s, t, i in zip(signs[:, 0], confirmation[:, 0], result[:, 0])
    ]
    return np.c_[result, wrong]


def generate(out):
    """Save source and application observations separately from evaluation truth."""
    cfg, folder = begin(out, "generate")
    bank, scores = observed_bank()
    source, evidence, truth = {}, {}, {}
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        b, s, c, v = draw_persons(cfg, rng, cfg["source_authors"])
        ids = cases(cfg, bank, s, c, v, cfg["source_variants"])
        source[f"s{seed}_values"] = scores[ids[:, :-1]]
        source[f"s{seed}_cases"] = ids[:, :-1]
        b, s, c, v = draw_persons(cfg, rng, cfg["application_authors"])
        truth[f"s{seed}_beta"] = b
        truth[f"s{seed}_event_signs"] = s
        for style, variants in [
            ("source_style", cfg["source_variants"]),
            ("held_style", cfg["held_variants"]),
        ]:
            ids = cases(cfg, bank, s, c, v, variants)
            evidence[f"s{seed}_{style}_cases"] = ids
            evidence[f"s{seed}_{style}_values"] = scores[ids]
    np.savez_compressed(folder / "source.npz", **source)
    np.savez_compressed(folder / "evidence.npz", **evidence)
    np.savez_compressed(folder / "truth.npz", **truth)
    pd.DataFrame({"case_id": bank.case_id, "value": scores}).to_csv(
        folder / "bank_values.csv", index=False
    )
    dump(
        folder / "seal.json",
        {
            name: sha(folder / name)
            for name in ["source.npz", "evidence.npz", "truth.npz", "bank_values.csv"]
        },
    )
    print("source and application draws saved separately", flush=True)


def fit(out):
    """Freeze source-only covariance models, with no author truth or new evidence."""
    cfg, folder = begin(out, "fit")
    inputs = [out / "generate/source.npz", out / "schedule.json"]
    outputs = [folder / n for n in ["models.json", "guard.json", "seal.json"]]
    denied = install(
        inputs,
        outputs,
        [out / "generate/truth.npz", out / "generate/evidence.npz"],
        [ROOT],
    )
    records = read(inputs[1])["source"]
    models = {}
    with np.load(inputs[0], allow_pickle=False) as data:
        for seed in cfg["seeds"]:
            models[str(seed)] = fit_source(
                data[f"s{seed}_values"], records, floor=cfg["covariance_floor"]
            )
    dump(folder / "models.json", models)
    dump(folder / "guard.json", {"denied": denied})
    dump(folder / "seal.json", {"models_sha256": sha(folder / "models.json")})
    print("three source models frozen", flush=True)


def value_rows(history):
    """Append-only wrong control including obsolete reports and exact copies."""
    rows, seen = [], {}
    for r in history:
        if r["kind"] in {"new", "repeat", "revise"}:
            rows.append(r)
        elif r["kind"] == "alias":
            rows.append(dict(seen[r["parent_id"]], record_id=r["record_id"]))
        seen[r["record_id"]] = r
    return rows


def apply(out):
    """Compile each history prefix before applying it to new-person readouts."""
    cfg, folder = begin(out, "apply")
    mp = out / "fit/models.json"
    assert sha(mp) == read(mp.parent / "seal.json")["models_sha256"]
    inputs = [mp, out / "schedule.json", out / "generate/evidence.npz"]
    outputs = [
        folder / n
        for n in [
            "programs.json",
            "predictions.npz",
            "guard.json",
            "seal.json",
            "history_status.csv",
        ]
    ]
    denied = install(inputs, outputs, [out / "generate/truth.npz"], [ROOT])
    models, history = read(inputs[0]), read(inputs[1])["stream"]
    programs, predictions, status = {}, {}, []
    for seed in cfg["seeds"]:
        for step in range(1, len(history) + 1):
            part = history[:step]
            state = active_records(part)
            packed = {"format": "versioned_numeric_evidence_v1", "history": part}
            assert restore_summary(packed) == state
            for arm, mode in [
                ("versioned", "full"),
                ("event_only", "event_only"),
                ("all_rows", "independent"),
            ]:
                records = state["records"] if arm != "all_rows" else value_rows(part)
                f = score_records(
                    models[str(seed)], np.zeros((1, len(records))), records, mode
                )
                key = f"s{seed}_step{step}_{arm}"
                programs[key] = {
                    "indices": [r["value_index"] for r in records],
                    "record_ids": [r["record_id"] for r in records],
                    "weights": TARGETS @ f["weights"],
                    "offset": TARGETS @ f["offset"],
                    "covariance": TARGETS @ f["covariance"] @ TARGETS.T,
                    "kind": part[-1]["kind"],
                }
            if seed == cfg["seeds"][0]:
                status.append(
                    {
                        "step": step,
                        "kind": part[-1]["kind"],
                        "active_reports": len(state["records"]),
                        "active_events": state["active_events"],
                        "observed_events": state["observed_events"],
                        "stale_derived": "|".join(state["stale_derived"]),
                    }
                )
    dump(folder / "programs.json", programs)
    with np.load(inputs[2], allow_pickle=False) as data:
        for key, program in programs.items():
            seed, rest = key.split("_", 1)
            for style in ["source_style", "held_style"]:
                y = data[f"{seed}_{style}_values"][:, program["indices"]]
                predictions[f"{seed}_{style}_{rest}"] = (
                    y @ np.asarray(program["weights"]).T + program["offset"]
                )
    np.savez_compressed(folder / "predictions.npz", **predictions)
    pd.DataFrame(status).to_csv(folder / "history_status.csv", index=False)
    dump(
        folder / "guard.json",
        {
            "denied": denied,
            "summary_restoration_prefixes": len(history) * len(cfg["seeds"]),
        },
    )
    dump(
        folder / "seal.json",
        {n: sha(folder / n) for n in ["programs.json", "predictions.npz"]},
    )
    print(f"{len(programs)} history programs applied", flush=True)


def evaluate(out):
    """Report all sequential targets against hidden person parameters after sealing."""
    cfg, folder = begin(out, "evaluate")
    ap = out / "apply"
    for name, digest in read(ap / "seal.json").items():
        assert sha(ap / name) == digest
    programs = read(ap / "programs.json")
    rows = []
    with (
        np.load(ap / "predictions.npz", allow_pickle=False) as predictions,
        np.load(out / "generate/truth.npz", allow_pickle=False) as truth,
    ):
        for key, p in programs.items():
            seed, step, arm = key.split("_", 2)
            actual = truth[seed + "_beta"] @ TARGETS.T
            variance = np.diag(p["covariance"])
            for style in ["source_style", "held_style"]:
                estimate = predictions[f"{seed}_{style}_{step}_{arm}"]
                for j, target in enumerate(cfg["source_targets"]):
                    error = estimate[:, j] - actual[:, j]
                    rows.append(
                        {
                            "seed": int(seed[1:]),
                            "step": int(step[4:]),
                            "kind": p["kind"],
                            "style": style,
                            "arm": arm,
                            "target": target,
                            "mse": float(np.mean(error**2)),
                            "bias": float(error.mean()),
                            "correlation": float(
                                np.corrcoef(estimate[:, j], actual[:, j])[0, 1]
                            ),
                            "working_variance": float(variance[j]),
                            "coverage": float(
                                np.mean(
                                    np.abs(error)
                                    <= 1.959964 * np.sqrt(max(variance[j], 0))
                                )
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "metrics_by_seed.csv", index=False)
    comparison = (
        frame.groupby(["step", "kind", "style", "arm", "target"], sort=False)[
            ["mse", "bias", "correlation", "working_variance", "coverage"]
        ]
        .mean()
        .reset_index()
    )
    comparison.to_csv(folder / "comparison.csv", index=False)
    natural = [json.loads(line) for line in NATURAL.read_text().splitlines()]
    dump(
        folder / "natural_qualification.json",
        {
            "rows": len(natural),
            "rows_with_event_address": sum(bool(r.get("event_id")) for r in natural),
            "rows_with_explicit_revision": sum(
                bool(r.get("supersedes")) for r in natural
            ),
            "rows_with_verified_cross_record_same_event": sum(
                bool(r.get("verified_event_link")) for r in natural
            ),
            "new_natural_personality_scores": 0,
            "reason": "Local syntactic event addresses do not establish independent opportunities or cross-comment revisions",
            "meps_read": False,
        },
    )
    print(
        comparison[comparison.step == comparison.step.max()].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["prepare", "generate", "fit", "apply", "evaluate"]
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())
