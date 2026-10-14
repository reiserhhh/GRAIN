#!/usr/bin/env python3
"""Bounded M10 source -> blind design -> application -> truth evaluation stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.new_condition_design import (
    basis,
    choose,
    fit_expanded,
    fit_source,
    hidden_basis,
    moments,
    score_weights,
    true_risk,
)
from grain_research.vector_mechanisms.stage_file_guard import install

DEFAULT = (
    ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1/new_condition_design_r1"
)
INPUTS = [
    "configs/grain_new_condition_design.json",
    "docs/GRAIN_NEW_CONDITION_DESIGN_PLAN_V1.md",
    "grain_research/new_condition_design.py",
    "grain_research/pilot_transport.py",
    "grain_research/vector_mechanisms/stage_file_guard.py",
    "scripts/run_grain_new_condition_design.py",
    "scripts/run_grain_evidence_calibration.py",
    "scripts/run_grain_entailment_reference.py",
]


def dump(path, obj):
    """Save readable arrays/scalars without opaque pickle objects."""

    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    Path(path).write_text(
        json.dumps(obj, indent=2, default=convert, allow_nan=False) + "\n"
    )


def read(path):
    """Read a stage's declared JSON input."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash bytes for frozen build and score provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start(out, stage):
    """Refuse drift or overwriting a completed/partial stage."""
    for name, digest in read(out / "frozen.json").items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    folder = out / stage
    folder.mkdir(exist_ok=False)
    return read(out / "config.json"), folder


def key(seed, world):
    """Stable run key, independent of randomized Python hashes."""
    return f"s{seed}_{world}"


def prepare(out):
    """Freeze one public text bank, budgets and source/application metadata."""
    out.mkdir(parents=True, exist_ok=False)
    cfg = read(ROOT / INPUTS[0])
    dump(out / "config.json", cfg)
    dump(out / "frozen.json", {name: sha(ROOT / name) for name in INPUTS})
    rows = []
    for ci, z in enumerate(cfg["levels"]):
        minutes = int(25 + 20 * z)
        for sign, field in enumerate(["negative_phrases", "positive_phrases"]):
            for pi, phrase in enumerate(cfg[field]):
                sentence = f"The boring task was allotted {minutes} minutes. {phrase}"
                parsed = (
                    int(re.search(r"allotted (\d+) minutes", sentence)[1]) - 25
                ) / 20
                rows.append(
                    dict(
                        case_id=len(rows),
                        condition_index=ci,
                        condition=parsed,
                        sign_index=sign,
                        phrase_index=pi,
                        sentence=sentence,
                    )
                )
    pd.DataFrame(rows).to_csv(out / "public_bank.csv", index=False)
    z = cfg["application_schedule"]
    aliases = list(range(len(z))) + cfg["duplicate_events"]
    dump(
        out / "schedule.json",
        dict(condition=[z[j] for j in aliases], time=aliases, event_ids=aliases),
    )
    print("prepare: 40 public texts, 80 NLI pairs, frozen", flush=True)


def readers(out):
    """Actually score the public bank with the pinned LOCAL NLI model once."""
    from scripts.run_grain_evidence_calibration import infer_pairs
    from scripts.run_grain_entailment_reference import MODEL, REVISION

    cfg, folder = start(out, "readers")
    bank = pd.read_csv(out / "public_bank.csv")
    rows = []
    for item in bank.itertuples():
        for pole, hypothesis in enumerate(
            [cfg["negative_phrases"][0], cfg["positive_phrases"][0]]
        ):
            rows.append(
                dict(
                    audit_id=f"c{item.case_id}p{pole}",
                    case_id=item.case_id,
                    pole=pole,
                    sentence=item.sentence,
                    item_text=hypothesis,
                )
            )
    requests = pd.DataFrame(rows)
    if len(requests) != cfg["nli_pair_budget"]:
        raise ValueError("Changed reader budget")
    begun = time.monotonic()
    scored = requests.merge(infer_pairs(requests), on="audit_id", validate="one_to_one")
    scored.to_csv(folder / "nli_pairs.csv", index=False)
    margins = scored.pivot(index="case_id", columns="pole", values="p_entailment")
    bank["margin"] = (
        margins.loc[bank.case_id, 1].to_numpy()
        - margins.loc[bank.case_id, 0].to_numpy()
    )
    scores = np.empty((5, 2, 4))
    anchors = []
    for ci, z in enumerate(cfg["levels"]):
        f = bank[bank.condition_index == ci]
        minus, plus = (float(f[f.sign_index == s].margin.mean()) for s in [0, 1])
        offset, gain = (plus + minus) / 2, (plus - minus) / 2
        if gain < 0.01:
            raise ValueError("Fixed meter does not orient event evidence")
        scores[ci] = ((f.margin.to_numpy() - offset) / gain).reshape(2, 4)
        anchors.append(
            dict(
                condition=z,
                offset=offset,
                gain=gain,
                second_minus=float(np.mean(scores[ci, 0] ** 2)),
                second_plus=float(np.mean(scores[ci, 1] ** 2)),
            )
        )
    np.save(folder / "scores.npy", scores)
    bank["event_score"] = scores.reshape(-1)
    bank.to_csv(folder / "scored_public_bank.csv", index=False)
    dump(folder / "meter.json", anchors)
    dump(
        folder / "runtime.json",
        dict(
            model=MODEL,
            revision=REVISION,
            pairs=len(scored),
            texts=len(bank),
            seconds=time.monotonic() - begun,
            local_files_only=True,
            new_neural_training=False,
        ),
    )
    print("readers: actual NLI finished", flush=True)


def simulate(cfg, scores, seed, world, role):
    """Generate bounded people/state and bank-linked event records in separate arrays."""
    base_seed = (
        seed
        + (10000 if world == "sticky" else 0)
        + {"source": 0, "expanded": 20000, "application": 40000}[role]
    )
    rng = np.random.default_rng(base_seed)
    h = np.asarray(cfg["profile_halfwidths"])
    n = cfg[
        {
            "source": "source_authors",
            "expanded": "expanded_source_authors",
            "application": "application_authors",
        }[role]
    ]
    z = np.array(
        cfg["source_schedule"] if role == "source" else cfg["application_schedule"],
        float,
    )
    reps = 1 if role == "application" else cfg["source_replicates"]
    profile = np.zeros((n, 4))
    profile[:, :3] = rng.uniform(-h[:3], h[:3], size=(n, 3))
    if world == "hidden_mode":
        # A separate stream preserves exactly the same SOURCE evidence bytes.
        profile[:, 3] = np.random.default_rng(base_seed + 80000).uniform(
            -h[3], h[3], size=n
        )
    rho, amplitude = cfg["rho"][world], cfg["state_amplitude"]
    signs = np.empty((n, len(z)), int)
    signs[:, 0] = rng.choice([-1, 1], size=n)
    flips = rng.random((n, len(z) - 1)) > (1 + rho) / 2
    for j in range(1, len(z)):
        signs[:, j] = signs[:, j - 1] * np.where(flips[:, j - 1], -1, 1)
    states = signs * amplitude
    latent_mean = profile @ basis(z, "hidden").T + states
    if np.max(np.abs(latent_mean)) >= 1:
        raise ValueError("Bernoulli probability is not bounded; no clipping allowed")
    shape = (n, len(z), reps, cfg["events_per_report"])
    positive = rng.random(shape) < ((1 + latent_mean) / 2)[:, :, None, None]
    phrases = rng.integers(0, 4, size=shape)
    ci = np.array([cfg["levels"].index(float(c)) for c in z])[None, :, None, None]
    bank_indices = (ci * 8 + positive.astype(int) * 4 + phrases).astype(np.uint8)
    observations = scores.reshape(-1)[bank_indices].mean(axis=3)
    return dict(
        observations=observations,
        bank_indices=bank_indices,
        condition=z,
        time=np.arange(len(z), dtype=float),
    ), dict(
        profile=profile,
        state=states,
        latent_mean=latent_mean,
        max_abs_latent=float(np.abs(latent_mean).max()),
    )


def generate(out):
    """Store evidence and latent truth under separate stage directories."""
    cfg, folder = start(out, "generate")
    evidence, truth = {}, {}
    scores = np.load(out / "readers/scores.npy")
    ledger = []
    for seed in cfg["seeds"]:
        for world in cfg["worlds"]:
            for role in ["source", "expanded", "application"]:
                ev, tr = simulate(cfg, scores, seed, world, role)
                name = key(seed, world) + "_" + role
                for field, value in ev.items():
                    evidence[name + "_" + field] = value
                for field, value in tr.items():
                    truth[name + "_" + field] = value
                ledger.append(
                    dict(
                        key=name,
                        authors=ev["observations"].shape[0],
                        events=int(ev["bank_indices"].size),
                        max_abs_latent=tr["max_abs_latent"],
                    )
                )
    source = {k: v for k, v in evidence.items() if "_application_" not in k}
    application = {k: v for k, v in evidence.items() if "_application_" in k}
    np.savez_compressed(folder / "source_evidence.npz", **source)
    np.savez_compressed(folder / "application_evidence.npz", **application)
    np.savez_compressed(folder / "truth.npz", **truth)
    pd.DataFrame(ledger).to_csv(folder / "generation_ledger.csv", index=False)
    checks = []
    for seed in cfg["seeds"]:
        a, b = (key(seed, w) + "_source_" for w in ["smooth", "hidden_mode"])
        equal = all(
            np.array_equal(source[a + f], source[b + f])
            for f in ["observations", "bank_indices", "condition", "time"]
        )
        assert equal
        checks.append(
            dict(seed=seed, source_observations_and_text_bank_indices_identical=equal)
        )
    dump(folder / "indistinguishable_source.json", checks)
    print("generate: controlled event evidence and separate truth saved", flush=True)


def fit(out):
    """Read only source evidence; never application values or latent profiles."""
    cfg, folder = start(out, "fit")
    src = out / "generate/source_evidence.npz"
    output, log = folder / "models.json", folder / "guard.json"
    denials = install(
        [src],
        [output, log],
        [out / "generate/truth.npz", out / "generate/application_evidence.npz"],
        [ROOT],
    )
    begun = time.monotonic()
    models = {}
    with np.load(src, allow_pickle=False) as ev:
        for seed in cfg["seeds"]:
            for world in cfg["worlds"]:
                name = key(seed, world)

                def data(role):
                    prefix = name + "_" + role + "_"
                    return (
                        ev[prefix + "observations"],
                        ev[prefix + "condition"],
                        ev[prefix + "time"],
                    )

                if world == "hidden_mode":
                    original = models[key(seed, "smooth")]["source"]
                else:
                    original = fit_source(*data("source"), maxiter=cfg["maxiter"])
                expanded = fit_expanded(*data("expanded"), original)
                models[name] = dict(source=original, expanded=expanded)
                print(
                    f"fit {name}: success={original['success']}, iterations={original['iterations']}",
                    flush=True,
                )
    dump(output, models)
    dump(
        log,
        dict(
            stage="fit",
            denied=denials,
            seconds=time.monotonic() - begun,
            source_fit_calls=6,
            hidden_source_fit_reused=True,
        ),
    )


def designs(out):
    """Commit all non-oracle observations and score weights with no application values."""
    cfg, folder = start(out, "designs")
    mp, sp = out / "fit/models.json", out / "schedule.json"
    output, ledger, log = (
        folder / "programs.json",
        folder / "selection.csv",
        folder / "guard.json",
    )
    denials = install(
        [mp, sp],
        [output, ledger, log],
        [out / "generate/truth.npz", out / "generate/application_evidence.npz"],
        [ROOT],
    )
    models, schedule = read(mp), read(sp)
    z, t, ids = (schedule[k] for k in ["condition", "time", "event_ids"])
    programs, rows = {}, []
    for seed in cfg["seeds"]:
        for world in cfg["worlds"]:
            name = key(seed, world)
            for ti, target in enumerate(cfg["targets"]):
                for policy in cfg["blind_policies"]:
                    model = models[name][
                        "expanded" if policy == "expanded_pilot" else "source"
                    ]
                    m = moments(
                        model,
                        z,
                        t,
                        ids,
                        target,
                        naive_duplicates=policy == "source_repeat_naive",
                    )
                    selected, trace = choose(
                        m, z, t, ids, policy, cfg["budget"], seed + 100 * ti
                    )
                    pkey = f"{name}__{target}__{policy}"
                    prefixes = []
                    for step in range(cfg["budget"] + 1):
                        s = selected[:step]
                        w, offset, var = score_weights(m, s)
                        prefixes.append(
                            dict(selected=s, weights=w, offset=offset, variance=var)
                        )
                    programs[pkey] = dict(
                        seed=seed,
                        world=world,
                        target=target,
                        policy=policy,
                        moments=m,
                        prefixes=prefixes,
                    )
                    for row in trace:
                        rows.append(dict(program=pkey, **row))
    dump(output, programs)
    pd.DataFrame(rows).to_csv(ledger, index=False)
    dump(
        log,
        dict(
            stage="designs",
            denied=denials,
            programs=len(programs),
            application_values_read=False,
            target_truth_read=False,
        ),
    )
    print("designs: all non-oracle selection programs frozen", flush=True)


def application(out):
    """Apply frozen linear programs to the actual new-author text-bank readouts."""
    cfg, folder = start(out, "application")
    pp, ep = out / "designs/programs.json", out / "generate/application_evidence.npz"
    output, log, seal = (
        folder / "predictions.npz",
        folder / "guard.json",
        folder / "seal.json",
    )
    program_hash = sha(pp)
    denials = install(
        [pp, ep], [output, log, seal], [out / "generate/truth.npz"], [ROOT]
    )
    programs = read(pp)
    predictions = {}
    aliases = list(range(20)) + cfg["duplicate_events"]
    with np.load(ep, allow_pickle=False) as ev:
        for name, p in programs.items():
            y = ev[key(p["seed"], p["world"]) + "_application_observations"][:, :, 0][
                :, aliases
            ]
            predictions[name] = np.stack(
                [
                    f["offset"] + y[:, f["selected"]] @ np.asarray(f["weights"])
                    for f in p["prefixes"]
                ],
                axis=1,
            )
    np.savez_compressed(output, **predictions)
    dump(seal, dict(programs_sha256=program_hash, predictions_sha256=sha(output)))
    dump(
        log,
        dict(
            stage="application",
            denied=denials,
            truth_read=False,
            programs=len(programs),
        ),
    )
    print("application: blind scores sealed before evaluating truth", flush=True)


def oracle_model(cfg, meter, world):
    """TRUE second moments, explicitly evaluation-only including report noise."""
    variances = np.array(cfg["profile_halfwidths"]) ** 2 / 3
    if world != "hidden_mode":
        variances[3] = 0
    v = np.diag(variances)
    x = basis(cfg["levels"], "hidden")
    state = cfg["state_amplitude"] ** 2
    second = np.array([(m["second_plus"] + m["second_minus"]) / 2 for m in meter])
    noise = (second - np.einsum("ij,jk,ik->i", x, v, x) - state) / cfg[
        "events_per_report"
    ]
    assert np.all(noise > 0)
    return dict(
        kind="hidden",
        covariance=v,
        mean=np.zeros(4),
        stationary=state,
        rate=-np.log(cfg["rho"][world]),
        noise_levels=cfg["levels"],
        noise=noise,
    )


def evaluate(out):
    """Evaluate frozen decisions; oracle is a diagnostic greedy reference only."""
    cfg, folder = start(out, "evaluate")
    pp = out / "designs/programs.json"
    seal = read(out / "application/seal.json")
    assert seal["programs_sha256"] == sha(pp)
    assert seal["predictions_sha256"] == sha(out / "application/predictions.npz")
    programs = read(pp)
    schedule, meter = read(out / "schedule.json"), read(out / "readers/meter.json")
    z, t, ids = (np.array(schedule[k]) for k in ["condition", "time", "event_ids"])
    rows, surfaces, oracle_rows = [], [], []
    oracle_programs = {}
    with (
        np.load(out / "generate/truth.npz", allow_pickle=False) as truth,
        np.load(
            out / "generate/application_evidence.npz", allow_pickle=False
        ) as evidence,
        np.load(out / "application/predictions.npz", allow_pickle=False) as preds,
    ):
        for seed in cfg["seeds"]:
            for world in cfg["worlds"]:
                name = key(seed, world)
                model = oracle_model(cfg, meter, world)
                profile = truth[name + "_application_profile"]
                states = truth[name + "_application_state"]
                y = evidence[name + "_application_observations"][:, :, 0][:, ids]
                for ti, target in enumerate(cfg["targets"]):
                    tm = moments(model, z, t, ids, target)
                    q = (
                        basis([0.5], "hidden")[0]
                        if target != "new_condition_contrast"
                        else (basis([0.5], "hidden")[0] - basis([-0.5], "hidden")[0])
                        / 2
                    )
                    actual = profile @ q + (
                        states[:, -1] if target == "current_response" else 0
                    )
                    selected, trace = choose(
                        tm, z, t, ids, "oracle", cfg["budget"], seed + 100 * ti
                    )
                    oracle_prefix = []
                    for step in range(cfg["budget"] + 1):
                        s = selected[:step]
                        w, off, var = score_weights(tm, s)
                        oracle_prefix.append(
                            dict(selected=s, weights=w, offset=off, variance=var)
                        )
                    op = dict(
                        seed=seed,
                        world=world,
                        target=target,
                        policy="oracle",
                        moments=tm,
                        prefixes=oracle_prefix,
                    )
                    oracle_programs[f"{name}__{target}__oracle"] = op
                    oracle_prediction = np.stack(
                        [
                            f["offset"] + y[:, f["selected"]] @ f["weights"]
                            for f in oracle_prefix
                        ],
                        axis=1,
                    )
                    oracle_rows.extend(
                        dict(seed=seed, world=world, target=target, **r) for r in trace
                    )
                    for policy in cfg["blind_policies"] + ["oracle"]:
                        pkey = f"{name}__{target}__{policy}"
                        p = op if policy == "oracle" else programs[pkey]
                        prediction = (
                            oracle_prediction if policy == "oracle" else preds[pkey]
                        )
                        for step, f in enumerate(p["prefixes"]):
                            estimated = prediction[:, step]
                            risk = true_risk(
                                tm, f["selected"], f["weights"], f["offset"]
                            )
                            err = estimated - actual
                            corr = (
                                float(np.corrcoef(estimated, actual)[0, 1])
                                if np.std(estimated) > 1e-12
                                else None
                            )
                            distinct = len(np.unique(ids[f["selected"]]))
                            row = dict(
                                seed=seed,
                                world=world,
                                target=target,
                                policy=policy,
                                step=step,
                                predicted_variance=f["variance"],
                                analytic_risk=risk,
                                empirical_mse=float(np.mean(err**2)),
                                mean_error=float(err.mean()),
                                correlation=corr,
                                distinct_reports=distinct,
                                new_condition_reports=int(
                                    sum(abs(z[j]) == 0.5 for j in f["selected"])
                                ),
                                expanded_source_cost=64
                                if policy == "expanded_pilot"
                                else 0,
                            )
                            if step:
                                previous = p["prefixes"][step - 1]
                                risk0 = true_risk(
                                    tm,
                                    previous["selected"],
                                    previous["weights"],
                                    previous["offset"],
                                )
                                choices = [
                                    j
                                    for j in range(len(z))
                                    if j not in previous["selected"]
                                    and (
                                        policy == "source_repeat_naive"
                                        or ids[j] not in ids[previous["selected"]]
                                    )
                                ]
                                alternatives = []
                                for j in choices:
                                    ss = previous["selected"] + [j]
                                    ww, oo, vv = score_weights(p["moments"], ss)
                                    rr = true_risk(tm, ss, ww, oo)
                                    alternatives.append(rr)
                                    surfaces.append(
                                        dict(
                                            seed=seed,
                                            world=world,
                                            target=target,
                                            policy=policy,
                                            step=step,
                                            candidate=j,
                                            event_id=int(ids[j]),
                                            condition=float(z[j]),
                                            time=float(t[j]),
                                            predicted_gain=previous["variance"] - vv,
                                            actual_expected_gain=risk0 - rr,
                                            analytic_next_risk=rr,
                                            selected=j == f["selected"][-1],
                                        )
                                    )
                                row["one_step_regret_same_scoring_model"] = max(
                                    0, risk - min(alternatives)
                                )
                                row["predicted_gain"] = (
                                    previous["variance"] - f["variance"]
                                )
                                row["actual_expected_gain"] = risk0 - risk
                            rows.append(row)
    pd.DataFrame(rows).to_csv(folder / "metrics_by_step.csv", index=False)
    pd.DataFrame(surfaces).to_csv(
        folder / "next_observation_loss_surface.csv", index=False
    )
    pd.DataFrame(oracle_rows).to_csv(folder / "oracle_selection.csv", index=False)
    dump(folder / "oracle_programs.json", oracle_programs)
    frame = pd.DataFrame(rows)
    last = frame[frame.step == cfg["budget"]]
    last.to_csv(folder / "final_by_seed.csv", index=False)
    summary = (
        last.groupby(["world", "target", "policy"], sort=False)[
            [
                "predicted_variance",
                "analytic_risk",
                "empirical_mse",
                "correlation",
                "distinct_reports",
                "new_condition_reports",
                "one_step_regret_same_scoring_model",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(folder / "final_comparison.csv", index=False)
    oracle = summary[summary.policy == "oracle"][
        ["world", "target", "analytic_risk"]
    ].rename(columns={"analytic_risk": "oracle_greedy_risk"})
    summary = summary.merge(oracle, on=["world", "target"], validate="many_to_one")
    summary["gap_to_oracle_greedy"] = summary.analytic_risk - summary.oracle_greedy_risk
    summary.to_csv(folder / "final_comparison.csv", index=False)
    source_same = read(out / "generate/indistinguishable_source.json")
    dump(
        folder / "summary.json",
        dict(
            comparisons=len(summary),
            scores=len(last),
            source_identical_checks=source_same,
            no_test_selection=True,
            oracle_is_greedy_not_global=True,
            gaussian_working_model_only=True,
        ),
    )
    print(
        summary[
            [
                "world",
                "target",
                "policy",
                "analytic_risk",
                "empirical_mse",
                "distinct_reports",
            ]
        ].to_string(index=False),
        flush=True,
    )


def main():
    """Expose independently bounded stages; never auto-resume an output stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "readers",
            "generate",
            "fit",
            "designs",
            "application",
            "evaluate",
        ],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
