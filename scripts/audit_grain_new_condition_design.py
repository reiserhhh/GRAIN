#!/usr/bin/env python3
"""Independent equations for frozen M10 bank, scores, risks and design regret.

This does not import the construction/scoring module or constitute an external
researcher replication. It recomputes raw evidence and linear algebra separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    """Read an artifact without importing its generating program."""
    return json.loads(path.read_text())


def sha(path):
    """Hash the retained frozen bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(cfg, meter, world, schedule, target):
    """Rebuild true moments directly from bounded profile and state generators."""
    z, t, ids = (np.array(schedule[k]) for k in ["condition", "time", "event_ids"])
    wave = np.sin(np.pi * z)
    wave[np.isin(z, [-1, 0, 1])] = 0
    x = np.column_stack([np.ones(len(z)), z, z * z, wave])
    v = np.array(cfg["profile_halfwidths"]) ** 2 / 3
    if world != "hidden_mode":
        v[3] = 0
    q = (
        np.array([1, 0.5, 0.25, 1])
        if target != "new_condition_contrast"
        else np.array([0, 0.5, 0, 1])
    )
    state = cfg["state_amplitude"] ** 2
    report_noise = []
    for row in x:
        cell = meter[cfg["levels"].index(float(row[1]))]
        esquare = (cell["second_minus"] + cell["second_plus"]) / 2
        report_noise.append(
            (esquare - (row * row) @ v - state) / cfg["events_per_report"]
        )
    noise = np.array(report_noise)
    kernel = (x * v) @ x.T + state * cfg["rho"][world] ** np.abs(
        t[:, None] - t[None, :]
    )
    kernel += (ids[:, None] == ids[None, :]) * np.sqrt(noise[:, None] * noise[None, :])
    cross = (x * v) @ q
    target_var = float((q * q) @ v)
    if target == "current_response":
        cross += state * cfg["rho"][world] ** np.abs(t - 19)
        target_var += state
    return kernel, cross, target_var, q


def direct(m, selected):
    """Independent direct solve for the retained, non-redundant score matrix."""
    if not selected:
        return np.zeros(0), m["target_mean"], m["variance"]
    s = np.array(selected, int)
    kernel = np.array(m["K"])[np.ix_(s, s)]
    k = np.array(m["k"])[s]
    w = np.linalg.solve(kernel, k)
    return w, m["target_mean"] - np.array(m["obs_mean"])[s] @ w, m["variance"] - k @ w


def main(out):
    """Recompute every saved program, score, metric and candidate loss surface."""
    root = Path(__file__).resolve().parents[1]
    for name, digest in read(out / "frozen.json").items():
        assert sha(root / name) == digest, name
    seal = read(out / "application/seal.json")
    assert seal["programs_sha256"] == sha(out / "designs/programs.json")
    assert seal["predictions_sha256"] == sha(out / "application/predictions.npz")
    cfg, schedule, meter = (
        read(out / p) for p in ["config.json", "schedule.json", "readers/meter.json"]
    )
    z, t, ids = (np.array(schedule[k]) for k in ["condition", "time", "event_ids"])
    nli = pd.read_csv(out / "readers/nli_pairs.csv")
    bank = pd.read_csv(out / "readers/scored_public_bank.csv")
    p = nli.pivot(index="case_id", columns="pole", values="p_entailment")
    difference = p[1] - p[0]
    np.testing.assert_allclose(bank.margin, difference, atol=1e-14)
    scores = np.load(out / "readers/scores.npy").reshape(-1)
    for i, row in enumerate(bank.itertuples()):
        anchor = meter[row.condition_index]
        assert (
            abs(
                (difference.loc[row.case_id] - anchor["offset"]) / anchor["gain"]
                - scores[i]
            )
            < 1e-13
        )
    for ci in range(5):
        np.testing.assert_allclose(
            scores.reshape(5, 2, 4)[ci].mean(axis=1), [-1, 1], atol=1e-13
        )

    source = np.load(out / "generate/source_evidence.npz", allow_pickle=False)
    application = np.load(out / "generate/application_evidence.npz", allow_pickle=False)
    truth = np.load(out / "generate/truth.npz", allow_pickle=False)
    preds = np.load(out / "application/predictions.npz", allow_pickle=False)
    event_count = 0
    for collection in [source, application]:
        for field in collection.files:
            if field.endswith("bank_indices"):
                index = collection[field]
                observed = collection[field.replace("bank_indices", "observations")]
                np.testing.assert_array_equal(scores[index].mean(axis=3), observed)
                event_count += index.size
    models = read(out / "fit/models.json")
    for seed in cfg["seeds"]:
        a, b = (f"s{seed}_{world}" for world in ["smooth", "hidden_mode"])
        for field in ["observations", "bank_indices"]:
            np.testing.assert_array_equal(
                source[f"{a}_source_{field}"], source[f"{b}_source_{field}"]
            )
        assert models[a]["source"] == models[b]["source"]
    for stage, expected in [("fit", 2), ("designs", 2), ("application", 1)]:
        assert len(read(out / stage / "guard.json")["denied"]) == expected

    programs = read(out / "designs/programs.json")
    programs.update(read(out / "evaluate/oracle_programs.json"))
    metrics = pd.read_csv(out / "evaluate/metrics_by_step.csv")
    surface = pd.read_csv(out / "evaluate/next_observation_loss_surface.csv")
    metric_index = metrics.set_index(["seed", "world", "target", "policy", "step"])
    surface_index = {
        k: v
        for k, v in surface.groupby(
            ["seed", "world", "target", "policy", "step"], sort=False
        )
    }
    worst = {
        "weights": 0.0,
        "offset": 0.0,
        "variance": 0.0,
        "prediction": 0.0,
        "risk": 0.0,
        "empirical_mse": 0.0,
        "surface": 0.0,
        "regret": 0.0,
        "oracle_risk_identity": 0.0,
    }

    def check(which, a, b):
        error = (
            float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0
        )
        worst[which] = max(worst[which], error)
        assert error < 2e-9, (which, error)

    count = 0
    for name, program in programs.items():
        seed, world, target, policy = (
            program[k] for k in ["seed", "world", "target", "policy"]
        )
        trueK, truek, truevar, q = oracle(cfg, meter, world, schedule, target)
        run = f"s{seed}_{world}"
        y = application[run + "_application_observations"][:, :, 0][:, ids]
        actual = truth[run + "_application_profile"] @ q
        if target == "current_response":
            actual += truth[run + "_application_state"][:, -1]
        if policy == "oracle":
            check("variance", np.array(program["moments"]["K"]), trueK)
            check("variance", np.array(program["moments"]["k"]), truek)
        for step, frame in enumerate(program["prefixes"]):
            s = frame["selected"]
            w, offset, var = direct(program["moments"], s)
            check("weights", w, frame["weights"])
            check("offset", offset, frame["offset"])
            check("variance", var, frame["variance"])
            estimate = offset + y[:, s] @ w
            if policy != "oracle":
                check("prediction", estimate, preds[name][:, step])
            risk = truevar - 2 * w @ truek[s] + w @ trueK[np.ix_(s, s)] @ w + offset**2
            row = metric_index.loc[(seed, world, target, policy, step)]
            check("risk", risk, row.analytic_risk)
            check("empirical_mse", np.mean((estimate - actual) ** 2), row.empirical_mse)
            if policy == "oracle":
                check("oracle_risk_identity", risk, var)
            if policy != "source_repeat_naive":
                assert len(np.unique(ids[s])) == len(s)
            if step:
                prev = program["prefixes"][step - 1]
                s0 = prev["selected"]
                w0, off0, _ = direct(program["moments"], s0)
                r0 = (
                    truevar
                    - 2 * w0 @ truek[s0]
                    + w0 @ trueK[np.ix_(s0, s0)] @ w0
                    + off0**2
                )
                alternatives = []
                for f in surface_index[
                    (seed, world, target, policy, step)
                ].itertuples():
                    ss = s0 + [f.candidate]
                    ww, oo, vv = direct(program["moments"], ss)
                    rr = (
                        truevar
                        - 2 * ww @ truek[ss]
                        + ww @ trueK[np.ix_(ss, ss)] @ ww
                        + oo**2
                    )
                    check("surface", rr, f.analytic_next_risk)
                    check("surface", r0 - rr, f.actual_expected_gain)
                    check("surface", prev["variance"] - vv, f.predicted_gain)
                    alternatives.append(rr)
                check(
                    "regret",
                    max(0, risk - min(alternatives)),
                    row.one_step_regret_same_scoring_model,
                )
            count += 1
    hidden_rules = []
    for seed in cfg["seeds"]:
        aa = f"s{seed}_smooth__new_condition_contrast__source"
        bb = f"s{seed}_hidden_mode__new_condition_contrast__source"
        np.testing.assert_array_equal(preds[aa], preds[bb])
        a = metric_index.loc[(seed, "smooth", "new_condition_contrast", "source", 8)]
        b = metric_index.loc[
            (seed, "hidden_mode", "new_condition_contrast", "source", 8)
        ]
        missing_variance = cfg["profile_halfwidths"][3] ** 2 / 3
        assert abs(b.analytic_risk - a.analytic_risk - missing_variance) < 1e-12
        hidden_rules.append(
            {
                "seed": seed,
                "all_prefix_scores_identical": True,
                "added_risk": float(b.analytic_risk - a.analytic_risk),
                "unobserved_independent_mode_variance": missing_variance,
            }
        )
    (out / "evaluate/identifiability_identity.json").write_text(
        json.dumps(hidden_rules, indent=2) + "\n"
    )
    details = {
        "status": "PASS",
        "programs": len(programs),
        "prefixes": count,
        "candidate_surfaces": len(surface),
        "bank_linked_events": event_count,
        "source_world_identity_checks": 3,
        "hidden_mode_risk_identity_checks": 3,
        "worst_errors": worst,
        "independent_equations_not_independent_researcher": True,
    }
    (out / "evaluate/independent_audit.json").write_text(
        json.dumps(details, indent=2) + "\n"
    )
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/grain_vector_mechanisms/mechanism_todo_v1/new_condition_design_r1",
    )
    main(parser.parse_args().output)
