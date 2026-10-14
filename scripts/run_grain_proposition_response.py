#!/usr/bin/env python3
"""Run raw dependency binding, condition estimation and individual score tests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.proposition_response import (
    bind_reply,
    condition_text,
    digest_text,
    frames,
    response_coordinates,
    response_profile,
)
from grain_research.source_event_graph import stanza_words
from scripts.fetch_grain_syntax_reference import MODELS, PACKAGES, PROCESSORS
from scripts.run_grain_evidence_calibration import infer_pairs
from scripts.run_grain_explanation_chain import sha, write
from scripts.run_grain_source_event_graph import verify_models

BASE = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1"
CONFIG = ROOT / "configs/grain_proposition_response.json"
PRIOR = BASE / "evidence_identity_r1"


def public_bank():
    """Freeze novel syntax crossings and challenges before observing parses."""
    rows = []
    for wording in [0, 1]:
        for condition in [0, 1]:
            prefix = [
                ["When the deadline approaches", "When you face a deadline"],
                ["When instructions are missing", "When you lack instructions"],
            ][condition][wording]
            for mode in ["past_action_report", "intention"]:
                for actor in ["respondent", "other"]:
                    person = "you" if actor == "respondent" else "your colleague"
                    if mode == "past_action_report":
                        question = (
                            f"did {person} "
                            + ["ask someone for help", "request assistance"][wording]
                        )
                    else:
                        aux = "do" if actor == "respondent" else "does"
                        question = (
                            f"{aux} {person} "
                            + [
                                "intend to ask someone for help",
                                "plan to request assistance",
                            ][wording]
                        )
                    for sign, reply in [(1, "Yes."), (-1, "No.")]:
                        rows.append(
                            {
                                "id": f"c{len(rows):02d}",
                                "part": "source" if wording == 0 else "held",
                                "wording": wording,
                                "condition": condition,
                                "mode": mode,
                                "actor": actor,
                                "sign": sign,
                                "partner": prefix + ", " + question + "?",
                                "reply": reply,
                            }
                        )
    challenges = [
        (
            "Did you ask someone for help yesterday?",
            "No. My colleague asked someone for help yesterday.",
            "explicit_other",
        ),
        (
            "Did your colleague ask someone for help yesterday?",
            "No. I asked someone for help yesterday.",
            "explicit_self",
        ),
        ("Didn't you ask someone for help yesterday?", "Yes.", "negative_question"),
        (
            "Did you ask for help? Did your colleague request assistance?",
            "Yes.",
            "multiple_questions",
        ),
        (
            "My colleague said, 'Did you ask someone for help?'",
            "Yes.",
            "quoted_question",
        ),
        (
            "Do you intend to ask for help?",
            "I intended to ask for help, but I never did.",
            "unrealized_intention",
        ),
        ("", "Please help me with this task.", "current_request"),
        ("", "Yes.", "missing_antecedent"),
    ]
    for i, (partner, reply, tag) in enumerate(challenges):
        rows.append(
            {
                "id": f"x{i}",
                "part": "challenge",
                "challenge": tag,
                "partner": partner,
                "reply": reply,
            }
        )
    return rows


def simulate(table, config, out):
    """Apply actual frozen text readings to paired simulated expression histories."""
    rng = np.random.default_rng(config["seed"])
    theta = rng.uniform(-0.8, 0.8, (config["simulated_people"], 2, 2))
    modes = ["past_action_report", "intention"]
    lookup = {
        (int(r.wording), int(r.condition), r.mode, r.actor, int(r.sign)): r
        for r in table.itertuples()
    }
    observations, results = [], []
    for person, parameters in enumerate(theta):
        histories = {
            w: {
                arm: []
                for arm in [
                    "oracle",
                    "graph_oracle_condition",
                    "graph_estimated_condition",
                    "content",
                ]
            }
            for w in [0, 1]
        }
        for condition in [0, 1]:
            for mi, mode in enumerate(modes):
                for event in range(config["events_per_cell"]):
                    actor = "respondent" if rng.random() < 0.7 else "other"
                    sign = (
                        1 if rng.random() < (parameters[condition, mi] + 1) / 2 else -1
                    )
                    oracle = float(
                        sign * (1 if mi == 0 else -1) * (actor == "respondent")
                    )
                    for wording in [0, 1]:
                        row = lookup[(wording, condition, mode, actor, sign)]
                        graph = (
                            float(row.graph_target)
                            if np.isfinite(row.graph_target)
                            else None
                        )
                        estimated_c = (
                            int(row.estimated_condition)
                            if np.isfinite(row.estimated_condition)
                            else None
                        )
                        for arm, c, score in [
                            ("oracle", condition, oracle),
                            ("graph_oracle_condition", condition, graph),
                            ("graph_estimated_condition", estimated_c, graph),
                            ("content", condition, float(row.content_target)),
                        ]:
                            histories[wording][arm].append(
                                {"condition": c, "score": score}
                            )
                        observations.append(
                            {
                                "person": person,
                                "condition": condition,
                                "mode": mode,
                                "event": event,
                                "wording": wording,
                                "case": row.id,
                                "oracle": oracle,
                            }
                        )
        for wording in [0, 1]:
            for arm, records in histories[wording].items():
                profile = response_profile(records, [0, 1])
                for c in [0, 1]:
                    results.append(
                        {
                            "person": person,
                            "wording": wording,
                            "arm": arm,
                            "condition": c,
                            "truth": 0.7 * (parameters[c, 0] - parameters[c, 1]) / 2,
                            **profile[str(c)],
                        }
                    )
    frame = pd.DataFrame(results)
    pd.DataFrame(observations).to_csv(out / "simulated_event_ledger.csv", index=False)
    frame.to_csv(out / "simulated_profiles.csv", index=False)
    np.save(out / "simulated_parameters.npy", theta)
    metrics = []
    for (wording, arm), group in frame.groupby(["wording", "arm"]):
        valid = group.dropna(subset=["score"])
        residual = valid.score.to_numpy() - valid.truth.to_numpy()
        r = (
            float(np.corrcoef(valid.score, valid.truth)[0, 1])
            if len(valid) > 1 and valid.score.std() > 1e-12
            else None
        )
        metrics.append(
            {
                "wording": int(wording),
                "arm": arm,
                "profiles": len(group),
                "scored": len(valid),
                "r": r,
                "mse": float(np.mean(residual**2)) if len(valid) else None,
                "observed_events": int(group.observed.sum()),
                "assigned_opportunities": int(group.opportunities.sum()),
            }
        )
    write(out / "simulation_metrics.json", metrics)
    return metrics


def run(out):
    """Parse once, freeze provenance and apply fixed binding rules to new input."""
    config = json.loads(CONFIG.read_text())
    public = public_bank()
    natural_path = PRIOR / "private_natural_bank.csv"
    prior = json.loads((PRIOR / "freeze.json").read_text())
    if sha(natural_path) != prior["inputs"][natural_path.name]:
        raise ValueError("Previously selected natural text changed")
    # The complete prior protocol stays immutable; reuse only its selected raw sentences.
    natural = pd.read_csv(natural_path).fillna("")
    if len(natural) != config["natural_records"]:
        raise ValueError("Natural selection changed")
    records = public + [
        {"id": r.record_id, "part": "natural", "partner": "", "reply": r.reply}
        for r in natural.itertuples()
    ]
    out.mkdir(parents=True, exist_ok=False)
    write(out / "public_bank.json", public)
    write(out / "private_inputs.json", records)
    write(out / "config.json", config)
    dependencies = [
        Path(__file__),
        ROOT / "grain_research/proposition_response.py",
        ROOT / "grain_research/source_event_graph.py",
        CONFIG,
        ROOT / "docs/GRAIN_PROPOSITION_RESPONSE_PLAN_V1.md",
        ROOT / "scripts/run_grain_evidence_calibration.py",
        ROOT / "scripts/run_grain_entailment_reference.py",
        MODELS / "grain_inventory.json",
        natural_path,
        out / "public_bank.json",
        out / "private_inputs.json",
    ]
    write(
        out / "freeze.json",
        {
            "files": {str(p.relative_to(ROOT)): sha(p) for p in dependencies},
            "prior_freeze_keys": list(prior),
            "content_ridge_alpha": 0.001,
        },
    )
    inventory = verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    torch.set_num_threads(2)
    t0 = time.monotonic()
    pipeline = stanza.Pipeline(
        "en",
        dir=str(MODELS),
        package=None,
        processors=PROCESSORS,
        use_gpu=False,
        download_method=None,
        verbose=False,
    )
    load_seconds = time.monotonic() - t0
    t0 = time.monotonic()
    cache, outputs, rows, pairs = {}, [], [], []
    for record in records:
        raw = [record["partner"], record["reply"]]
        parsed = []
        for text in raw:
            if text not in cache:
                if (
                    len(cache) >= config["max_parse_texts"]
                    or time.monotonic() - t0 > config["max_seconds"]
                ):
                    raise TimeoutError("Bounded parser budget exceeded")
                cache[text] = stanza_words(pipeline(text), text) if text else []
            parsed.append(cache[text])
        pf, rf = (
            frames(raw[0], parsed[0], "partner"),
            frames(raw[1], parsed[1], "respondent"),
        )
        binding = bind_reply(*raw, pf, rf)
        coordinates, supported = response_coordinates(binding)
        target = (
            coordinates["past_action_report"] - coordinates["intention"]
            if supported
            else None
        )
        cond = condition_text(binding)
        outputs.append(
            {
                "id": record["id"],
                "part": record["part"],
                "partner_frames": pf,
                "reply_frames": rf,
                "binding": binding,
                "condition_text": cond,
                "coordinates": coordinates,
                "graph_target": target,
            }
        )
        rows.append(
            {k: v for k, v in record.items() if k not in ["partner", "reply"]}
            | {
                "status": binding["status"],
                "condition_text": cond,
                "graph_target": target,
                "frames": len(pf) + len(rf),
                "supported": supported,
            }
        )
        if record["part"] in ["source", "held", "challenge"]:
            pairs.append(
                {
                    "audit_id": record["id"],
                    "sentence": "Partner: " + raw[0] + "\nRespondent: " + raw[1],
                    "item_text": "The conversation is about asking for help.",
                }
            )
    parse_seconds = time.monotonic() - t0
    write(
        out / "private_parses.json",
        [
            {"text": k, "sha256": digest_text(k), "sentences": v}
            for k, v in cache.items()
        ],
    )
    write(out / "private_graphs.json", outputs)
    pair_frame = pd.DataFrame(pairs)
    if len(pair_frame) > config["max_nli_pairs"]:
        raise ValueError("NLI budget exceeded")
    pair_frame.to_csv(out / "nli_requests.csv", index=False)
    t1 = time.monotonic()
    probabilities = infer_pairs(pair_frame)
    probabilities.to_csv(out / "nli_readings.csv", index=False)
    nli_seconds = time.monotonic() - t1
    table = pd.DataFrame(rows)
    main = table.loc[table.part.isin(["source", "held"])].copy()
    source = main.part.eq("source")
    available = source & main.condition_text.ne("")
    if not available.any() or main.loc[available, "condition"].nunique() != 2:
        raise ValueError("Missing source conditions; retain parse artifacts")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x = vectorizer.fit_transform(main.loc[available, "condition_text"])
    centroids = np.stack(
        [
            np.asarray(x[main.loc[available, "condition"].eq(c)].mean(axis=0)).ravel()
            for c in [0, 1]
        ]
    )
    all_x = vectorizer.transform(main.condition_text)
    sim = np.asarray(all_x @ centroids.T)
    main["estimated_condition"] = np.where(
        main.condition_text.ne("") & (sim.max(axis=1) > 0), sim.argmax(axis=1), np.nan
    )
    main["content"] = main.id.map(probabilities.set_index("audit_id").p_entailment)
    truth_event = (
        main.sign.to_numpy()
        * np.where(main["mode"].eq("past_action_report"), 1, -1)
        * main.actor.eq("respondent")
    )
    content_model = Ridge(alpha=0.001).fit(
        main.loc[source, ["content"]], truth_event[source]
    )
    main["content_target"] = content_model.predict(main[["content"]])
    main["oracle_target"] = truth_event
    main.to_csv(out / "controlled_readings.csv", index=False)
    table.to_csv(out / "private_interface_summary.csv", index=False)
    write(
        out / "source_calibration.json",
        {
            "vocabulary": vectorizer.vocabulary_,
            "idf": vectorizer.idf_.tolist(),
            "centroids": centroids.tolist(),
            "content_intercept": float(content_model.intercept_),
            "content_coefficient": content_model.coef_.tolist(),
            "alpha": 0.001,
            "source_ids": main.loc[source, "id"].tolist(),
            "stanza": inventory["stanza"],
        },
    )
    metrics = simulate(main, config, out)
    binding_checks = []
    for part, group in main.groupby("part"):
        valid = group.dropna(subset=["graph_target"])
        binding_checks.append(
            {
                "part": part,
                "rows": len(group),
                "scored": len(valid),
                "target_exact": int(
                    np.isclose(valid.graph_target, valid.oracle_target).sum()
                ),
                "condition_correct": int(
                    group.estimated_condition.eq(group.condition).sum()
                ),
                "unknown_conditions": int(group.estimated_condition.isna().sum()),
            }
        )
    write(
        out / "summary.json",
        {
            "status": "complete",
            "public_records": len(public),
            "natural_records": len(natural),
            "unique_parsed_texts": len(cache) - 1,
            "load_seconds": load_seconds,
            "parse_seconds": parse_seconds,
            "nli_pairs": len(pairs),
            "nli_seconds": nli_seconds,
            "binding_checks": binding_checks,
            "natural_status_counts": table.loc[table.part.eq("natural"), "status"]
            .value_counts()
            .to_dict(),
            "simulation": metrics,
            "new_human_validity": False,
        },
    )
    write(
        out / "manifest.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in out.iterdir()
            if p.is_file() and not p.name.startswith("._")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "proposition_response_r1")
    run(parser.parse_args().out)
