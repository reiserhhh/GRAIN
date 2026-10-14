#!/usr/bin/env python3
"""Pinned reader -> typed event -> author response, on known language mechanisms.

No actual participant labels or text are loaded. The sentence bank is finite and
authored, not an open-domain NLI benchmark. Semantic vectors and NLI scores are
computed by existing local models, not supplied from the simulated event truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.event_evidence import (
    aggregate_events,
    conditional_profile,
    paired_evidence,
    split_controlled_reported_speech,
)
from scripts.prepare_grain_predecessor_reference import BASE, digest, write
from scripts.run_grain_entailment_reference import MODEL as NLI_MODEL
from scripts.run_grain_entailment_reference import REVISION as NLI_REVISION
from scripts.run_grain_evidence_calibration import infer_pairs
from scripts.run_grain_paraphrase_reference import MODEL, REVISION, encode_batches


def sentence_bank(config):
    """Cross wording, response, condition and owner without a personality word."""
    rows = []
    for query in config["queries"]:
        for z in (-1, 1):
            for sign in (-1, 1):
                for variant in range(4):
                    own = query["positive" if sign == 1 else "negative"][variant]
                    other = query["negative" if sign == 1 else "positive"][variant]
                    for role in config["roles"]:
                        if role == "self":
                            text = (
                                "Yesterday, "
                                + query["conditions"][str(z)].lower()
                                + ", "
                                + own
                            )
                        elif role == "other":
                            text = (
                                "Yesterday my friend told me: '"
                                + own
                                + "' I have not described my own response."
                            )
                        else:
                            text = (
                                "Yesterday my friend told me: '"
                                + other
                                + "' My own experience was different. "
                                + own
                            )
                        rows.append(
                            {
                                "case_id": f"{query['id']}_{z}_{sign}_{variant}_{role}",
                                "query": query["id"],
                                "condition": z,
                                "sign": sign,
                                "variant": variant,
                                "role": role,
                                "sentence": text,
                                "expected_direction": sign if role != "other" else 0,
                                "partition": "source_wording"
                                if variant < 2
                                else "held_wording",
                            }
                        )
    return pd.DataFrame(rows)


def read_bank(config, bank, output, reader_cache=None):
    """Run all fixed reader variants before generating or scoring simulated authors."""
    queries = {q["id"]: q for q in config["queries"]}
    pairs = []
    for row in bank.itertuples():
        for scope in ("episode", "habitual"):
            for key in ("positive", "negative"):
                pairs.append(
                    {
                        "audit_id": f"{row.case_id}/{scope}/{key}",
                        "case_id": row.case_id,
                        "scope": scope,
                        "key": key,
                        "sentence": row.sentence,
                        "item_text": queries[row.query][scope + "_" + key],
                    }
                )
    pairs = pd.DataFrame(pairs)
    cache = output / "public_nli_pairs.csv"
    existing = cache if cache.exists() else reader_cache
    if existing is not None and existing.exists():
        nli = pd.read_csv(existing)
        if len(nli) != len(pairs):
            raise ValueError("NLI cache row count mismatch")
        for column in pairs:
            if not nli[column].equals(pairs[column]):
                raise ValueError("NLI cache input mismatch")
        provenance = json.loads((existing.parent / "contract.json").read_text())
        if (
            provenance["nli_model"] != NLI_MODEL
            or provenance["nli_revision"] != NLI_REVISION
        ):
            raise ValueError("NLI cache model mismatch")
    else:
        print(f"Computing {len(pairs)} fixed local NLI pairs", flush=True)
        nli = pairs.merge(infer_pairs(pairs), on="audit_id", validate="one_to_one")
    nli.to_csv(cache, index=False)
    scores = bank.copy()
    for scope in ("episode", "habitual"):
        wide = (
            nli[nli.scope == scope]
            .pivot(index="case_id", columns="key", values="p_entailment")
            .reindex(bank.case_id)
        )
        decision = paired_evidence(
            wide.positive.to_numpy(),
            wide.negative.to_numpy(),
            config["nli_threshold"],
            config["nli_margin"],
        )
        for key, values in decision.items():
            scores[f"{scope}_{key}"] = values
    routed = []
    for row in bank.itertuples():
        routed.append(
            {"case_id": row.case_id, **split_controlled_reported_speech(row.sentence)}
        )
    write(output / "public_routed_spans.json", routed)
    routed_text = {row["case_id"]: row["writer_text"] for row in routed}
    scoped = pairs[pairs.scope == "episode"].copy()
    scoped["sentence"] = scoped.case_id.map(routed_text)
    scoped["audit_id"] = "owner/" + scoped.audit_id
    scoped_cache = output / "public_scoped_nli_pairs.csv"
    if scoped_cache.exists():
        routed_nli = pd.read_csv(scoped_cache)
        if len(routed_nli) != len(scoped) or any(
            routed_nli[c].tolist() != scoped[c].tolist() for c in scoped
        ):
            raise ValueError("Scoped NLI cache mismatch")
    else:
        routed_nli = scoped.merge(
            infer_pairs(scoped), on="audit_id", validate="one_to_one"
        )
        routed_nli.to_csv(scoped_cache, index=False)
    wide = routed_nli.pivot(
        index="case_id", columns="key", values="p_entailment"
    ).reindex(bank.case_id)
    decision = paired_evidence(
        wide.positive.to_numpy(),
        wide.negative.to_numpy(),
        config["nli_threshold"],
        config["nli_margin"],
    )
    for key, values in decision.items():
        scores[f"owner_episode_{key}"] = values
    hypotheses = [
        q["episode_" + key]
        for q in config["queries"]
        for key in ("positive", "negative")
    ]
    encoded, _ = encode_batches(
        hypotheses + bank.sentence.tolist(), output / "encoder_cache", MODEL, REVISION
    )
    if json.loads((output / "encoder_cache/contract.json").read_text())[
        "truncated_texts"
    ]:
        raise ValueError("Designed sentences must not be silently truncated")
    h = encoded[len(hypotheses) :]
    sim = h @ encoded[: len(hypotheses)].T
    indexes = {q["id"]: i for i, q in enumerate(config["queries"])}
    margins, keys = [], []
    for row, values in zip(bank.itertuples(), sim, strict=True):
        v = values[2 * indexes[row.query] : 2 * indexes[row.query] + 2]
        margins.append(v[0] - v[1])
        keys.append(1 if v[0] >= v[1] else -1)
    scores["cosine_margin"] = margins
    scores["cosine_key"] = keys
    scores.to_csv(output / "public_reader_scores.csv", index=False)
    return scores


def literal_summary(scores):
    """Describe direction/abstention on designed cases, not psychometric validity."""
    rows = []
    for (partition, role, query), part in scores.groupby(
        ["partition", "role", "query"]
    ):
        for method in ("cosine_key", "episode", "habitual", "owner_episode"):
            pred = (
                part.cosine_key.to_numpy()
                if method == "cosine_key"
                else part[method + "_direction"].fillna(0).to_numpy()
            )
            expected = part.expected_direction.to_numpy()
            accepted = pred != 0
            rows.append(
                {
                    "partition": partition,
                    "role": role,
                    "query": query,
                    "method": method,
                    "n": len(part),
                    "direction_or_abstention_accuracy": float(
                        np.mean(pred == expected)
                    ),
                    "accepted_fraction": float(accepted.mean()),
                    "wrong_direction_fraction": float(
                        np.mean((expected != 0) & (pred == -expected))
                    ),
                    "unsupported_acceptance_fraction": float(
                        np.mean((expected == 0) & accepted)
                    ),
                }
            )
    return pd.DataFrame(rows)


def simulate(config, bank, seed, world):
    """Generate independent authors, binary responses and known design conditions.

    Y is the analytic expected response or contrast of the generating probabilities,
    not a questionnaire label. Event words do not encode author IDs or target values.
    """
    rng = np.random.default_rng(seed)
    n, count = config["authors"], config["events_per_author"]
    if count % 2 or config["source_authors"] >= n:
        raise ValueError("Even balanced event count and a held author set required")
    rows, truth = [], []
    for query in config["queries"]:
        a, b = rng.normal(size=(2, n))
        if world == "level":
            b[:] = 0
        elif world == "response":
            a[:] = 0
        else:
            raise ValueError("Unknown mechanism world")
        minus, plus = 2 * expit(a - b) - 1, 2 * expit(a + b) - 1
        for author in range(n):
            is_source = author < config["source_authors"]
            z = np.tile([-1, 1], count // 2)
            rng.shuffle(z)
            behavior = np.where(
                rng.random(count) < expit(a[author] + b[author] * z), 1, -1
            )
            roles = rng.choice(config["roles"], count, p=config["role_probabilities"])
            variants = rng.integers(0, 2, count) + (0 if is_source else 2)
            aid = f"synthetic_{author:03d}"
            truth.append(
                {
                    "author_id": aid,
                    "query": query["id"],
                    "source": is_source,
                    "level": float((plus[author] + minus[author]) / 2),
                    "response": float((plus[author] - minus[author]) / 2),
                }
            )
            for t, (c, sign, role, variant) in enumerate(
                zip(z, behavior, roles, variants, strict=True)
            ):
                # A friend's reported response is independent of this author's latent response.
                text_sign = int(rng.choice([-1, 1])) if role == "other" else int(sign)
                case = f"{query['id']}_{c}_{text_sign}_{variant}_{role}"
                rows.append(
                    {
                        "author_id": aid,
                        "query": query["id"],
                        "event_id": f"e{t}",
                        "case_id": case,
                        "condition": int(c),
                        "oracle_score": float(sign),
                        "oracle_accepted": role != "other",
                    }
                )
    events = pd.DataFrame(rows).merge(
        bank.drop(columns=["query", "condition"]), on="case_id", validate="many_to_one"
    )
    events["sentence_hash"] = events.sentence.map(
        lambda t: hashlib.sha256(t.encode()).hexdigest()
    )
    return events, pd.DataFrame(truth)


def assess_profiles(config, events, truth, seed, world):
    """Fit a fixed linear decoder on source authors, then test disjoint authors/words."""
    rows, predictions = [], []
    target = "level" if world == "level" else "response"
    readers = {
        "oracle": (events.oracle_score, events.oracle_accepted),
        "cosine_key": (events.cosine_key, np.ones(len(events), dtype=bool)),
        "cosine_margin": (events.cosine_margin, np.ones(len(events), dtype=bool)),
        "episode_soft": (events.episode_contrast, np.ones(len(events), dtype=bool)),
        "episode_gated": (events.episode_contrast, events.episode_accepted),
        "habitual_gated": (events.habitual_contrast, events.habitual_accepted),
        "owner_episode_gated": (
            events.owner_episode_contrast,
            events.owner_episode_accepted,
        ),
    }
    for reader, (score, accepted) in readers.items():
        frame = events.assign(score=score, accepted=accepted)
        aggregate = aggregate_events(frame)
        profile = conditional_profile(aggregate)
        for representation, features in (
            ("pooled", ["level"]),
            ("conditional", ["level", "response"]),
        ):
            joint = truth.merge(
                profile,
                on=["author_id", "query"],
                suffixes=("_true", "_read"),
                validate="one_to_one",
            )
            for query, group in joint.groupby("query"):
                x = group[[f + "_read" for f in features]].to_numpy()
                valid = np.isfinite(x).all(axis=1)
                train, test = (
                    group.source.to_numpy() & valid,
                    ~group.source.to_numpy() & valid,
                )
                if train.sum() < 10 or test.sum() < 10:
                    rows.append(
                        {
                            "seed": seed,
                            "world": world,
                            "query": query,
                            "reader": reader,
                            "representation": representation,
                            "status": "INSUFFICIENT_EVIDENCE",
                            "n_test": int(test.sum()),
                        }
                    )
                    continue
                y = group[target + "_true"].to_numpy()
                model = Ridge(alpha=config["ridge_alpha"]).fit(x[train], y[train])
                predicted = model.predict(x[test])
                r = (
                    float(pearsonr(y[test], predicted).statistic)
                    if predicted.std() > 1e-12
                    else None
                )
                rows.append(
                    {
                        "seed": seed,
                        "world": world,
                        "query": query,
                        "reader": reader,
                        "representation": representation,
                        "status": "COMPLETE",
                        "n_test": int(test.sum()),
                        "pearson_r": r,
                        "rmse": float(np.sqrt(np.mean((predicted - y[test]) ** 2))),
                        "n_train": int(train.sum()),
                        "event_acceptance": float(
                            frame.loc[frame["query"].eq(query), "accepted"].mean()
                        ),
                    }
                )
                for aid, actual, pred in zip(
                    group.loc[test, "author_id"], y[test], predicted, strict=True
                ):
                    predictions.append(
                        {
                            "seed": seed,
                            "world": world,
                            "query": query,
                            "reader": reader,
                            "representation": representation,
                            "author_id": aid,
                            "actual": actual,
                            "predicted": float(pred),
                        }
                    )
    return rows, predictions


def run(args):
    """Freeze the public design, compute readers once, then run the five fixed seeds."""
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {
        "config": config,
        "nli_model": NLI_MODEL,
        "nli_revision": NLI_REVISION,
        "encoder_model": MODEL,
        "encoder_revision": REVISION,
        "code_sha256": digest(Path(__file__)),
        "module_sha256": digest(ROOT / "grain_research/event_evidence.py"),
        "private_data_loaded": False,
        "conditions_from": "known_design_not_inferred",
        "model_selection": "none; fixed source-trained Ridge; no trait labels",
    }
    contract_path = args.output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Resume contract changed; use a new output directory")
    write(contract_path, contract)
    if (args.output / "summary.json").exists():
        print((args.output / "summary.json").read_text())
        return
    bank = sentence_bank(config)
    bank.to_csv(args.output / "public_sentence_bank.csv", index=False)
    scores = read_bank(config, bank, args.output, args.reader_cache)
    literal = literal_summary(scores)
    literal.to_csv(args.output / "literal_by_condition.csv", index=False)
    rows, predictions = [], []
    for seed in config["seeds"]:
        for world in config["worlds"]:
            events, truth = simulate(config, scores, seed, world)
            result, pred = assess_profiles(config, events, truth, seed, world)
            rows.extend(result)
            predictions.extend(pred)
            print(f"Finished seed={seed} world={world}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "metrics_by_seed.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        args.output / "public_synthetic_predictions.csv", index=False
    )
    summary = (
        frame.groupby(["world", "query", "reader", "representation"])
        .agg(
            r_mean=("pearson_r", "mean"),
            r_std=("pearson_r", "std"),
            rmse=("rmse", "mean"),
            n_test_min=("n_test", "min"),
            event_acceptance=("event_acceptance", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "profile_summary.csv", index=False)
    write(
        args.output / "summary.json",
        {
            "status": "COMPLETE",
            "unique_cases": len(bank),
            "nli_pairs": len(bank) * 6,
            "attribution_method": "explicit controlled reporting grammar, not general learned attribution",
            "seeds": config["seeds"],
            "profile_rows": len(frame),
            "scope": config["scope"],
            "source_held_wording_disjoint": True,
            "personality_prediction": False,
        },
    )
    print((args.output / "summary.json").read_text(), flush=True)


def main():
    """Run offline; a fresh directory freezes inputs and completed same-code runs resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_event_evidence.json"
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/event_r1"
    )
    parser.add_argument(
        "--reader-cache",
        type=Path,
        help="Exact input/model-bound raw NLI CSV from an earlier run",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
