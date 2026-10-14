#!/usr/bin/env python3
"""Independent 2SLS, held accounting and fixed-reader artifact checks for M09."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_query_moment_operator import read, sha, write


def audit_math(out, cfg):
    """Independent projected least squares on fit prefix, and held error replay."""
    folder = out / "math"
    observations = pd.read_csv(folder / "observations.csv")
    fits = read(folder / "fits.json")
    held = pd.read_csv(folder / "held_errors.csv").set_index(
        ["world", "seed", "person", "arm", "target"]
    )
    coefficients = pd.read_csv(folder / "coefficient_predictions.csv")
    maximum, scores, rank_boundaries = 0.0, 0, 0
    for (world, seed, person), frame in observations.groupby(
        ["world", "seed", "person"]
    ):
        frame = frame.sort_values("occasion")
        fit = fits[f"{world}:{seed}:{person}"]
        part = frame[frame.partition == "fit"]
        future = frame[frame.partition == "held"]
        if (
            len(part) != cfg["fit_opportunities"]
            or len(future) != cfg["held_opportunities"]
            or part.occasion.max() >= future.occasion.min()
        ):
            raise AssertionError("Prefix/held boundary violated")
        z, u, y = (part[k].to_numpy() for k in ("condition", "partner", "response"))
        x = np.c_[np.ones(len(y)), z, u, z * u]
        if fit["status"] == "conditional_sum_only":
            if np.linalg.matrix_rank(x) != 2:
                raise AssertionError("Wrong fixed-partner boundary")
            rank_boundaries += 1
            continue
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        maximum = max(maximum, np.max(abs(beta - fit["ols"])))
        betas = {"ols": beta}
        if "iv" in fit:
            r = part.assignment.to_numpy()
            w = np.c_[np.ones(len(r)), z, r, z * r]
            projected_x = w @ np.linalg.lstsq(w, x, rcond=None)[0]
            iv = np.linalg.solve(projected_x.T @ projected_x, projected_x.T @ y)
            maximum = max(maximum, np.max(abs(iv - fit["iv"])))
            betas.update(iv=iv, control_function=iv)
        zz, uu, yy, ss = (
            future[k].to_numpy()
            for k in ("condition", "partner", "response", "truth_state")
        )
        xx = np.c_[np.ones(len(zz)), zz, uu, zz * uu]
        for arm, beta in betas.items():
            common = np.zeros(len(xx))
            if arm == "control_function":
                rr = future.assignment.to_numpy()
                ww = np.c_[np.ones(len(rr)), zz, rr, zz * rr]
                common = (xx[:, 2:] - ww @ fit["first_stage"]) @ np.asarray(fit["cf"])[
                    4:
                ]
            for target, predicted, actual in (
                ("current_expression", xx @ beta + common, yy),
                ("common_state_evidence", common, ss),
                ("remaining_state", yy - xx @ beta, ss),
            ):
                row = held.loc[(world, seed, person, arm, target)]
                maximum = max(
                    maximum,
                    abs(row.mse - np.mean((predicted - actual) ** 2)),
                    abs(row.bias - np.mean(predicted - actual)),
                )
                scores += 2
    if maximum > 1e-6:
        raise AssertionError(f"Independent replay discrepancy {maximum}")
    # Pair persons within a fixed world; preserve every weak-instrument extreme.
    rng = np.random.default_rng(91410)
    rows = []
    for (world, target), part in coefficients.groupby(["world", "target"]):
        a = part[part.arm == "ols"].set_index(["seed", "person"])
        b = part[part.arm == "iv"].set_index(["seed", "person"])
        if not len(b):
            continue
        b = b.loc[a.index]
        ea, eb = (
            (a.score - a.truth).to_numpy() ** 2,
            (b.score - b.truth).to_numpy() ** 2,
        )
        draws = []
        blocks = [
            np.flatnonzero(a.index.get_level_values("seed") == seed)
            for seed in cfg["seeds"]
        ]
        for _ in range(999):
            ix = np.concatenate([rng.choice(k, len(k), replace=True) for k in blocks])
            draws.append(np.sqrt(ea[ix].mean()) - np.sqrt(eb[ix].mean()))
        rows.append(
            {
                "world": world,
                "target": target,
                "delta_rmse": np.sqrt(ea.mean()) - np.sqrt(eb.mean()),
                "lower95": np.quantile(draws, 0.025),
                "upper95": np.quantile(draws, 0.975),
                "positive_seed_count": sum(ea[k].mean() > eb[k].mean() for k in blocks),
            }
        )
    pd.DataFrame(rows).to_csv(out / "paired_coefficient_comparisons.csv", index=False)
    return {
        "participants": len(fits),
        "held_values_checked": scores,
        "fixed_input_boundaries": rank_boundaries,
        "maximum_error": float(maximum),
        "negative_coefficient_variances": int((coefficients.variance < -1e-10).sum()),
    }


def audit_observers(out):
    """Check numeric probabilities, unchanged reply hashes and original turn roles.

    No model forward is repeated here; this is an artifact/scope audit, not an
    independent recreation of model activations or a human-gold coding study.
    """
    nli = pd.read_csv(out / "nli/public_nli.csv")
    pcols = ["p_contradiction", "p_neutral", "p_entailment"]
    if nli.audit_id.duplicated().any() or not np.isfinite(nli[pcols]).all().all():
        raise AssertionError("Invalid NLI outputs")
    np.testing.assert_allclose(nli[pcols].sum(axis=1), 1, atol=1e-6)
    natural = pd.read_csv(out / "natural/private_observer_scores.csv")
    letters = "ABCDEFG"
    logits = natural[[f"logit_{k}" for k in letters]].to_numpy()
    probabilities = natural[[f"p_{k}" for k in letters]].to_numpy()
    maximum = float(np.max(abs(softmax(logits, axis=1) - probabilities)))
    if maximum > 1e-6:
        raise AssertionError(
            "Saved natural logits do not reproduce label probabilities"
        )
    if not np.all(natural.reply_sequence == natural.partner_sequence + 1):
        raise AssertionError("Nonadjacent natural pair")
    if set(natural.scope) != {"pair", "reply"}:
        raise AssertionError("Missing natural scope")
    changes = []
    for code, group in natural.groupby("participant_code"):
        if len(group) != 2 or group.reply_sha256.nunique() != 1:
            raise AssertionError("Current reply changed between scopes")
        a, b = (group[group.scope == s].iloc[0] for s in ("pair", "reply"))
        pa, pb = (r[[f"p_{k}" for k in letters]].to_numpy(float) for r in (a, b))
        mean = (pa + pb) / 2
        js = 0.5 * (
            np.sum(pa * np.log((pa + 1e-30) / (mean + 1e-30)))
            + np.sum(pb * np.log((pb + 1e-30) / (mean + 1e-30)))
        )
        changes.append(
            {
                "participant_code": code,
                "changed_top_act": a.top_act != b.top_act,
                "jensen_shannon_nats": js,
                "pair_top": a.top_act,
                "reply_top": b.top_act,
            }
        )
    pd.DataFrame(changes).to_csv(out / "natural/private_scope_changes.csv", index=False)
    access = read(out / "natural/source_access.json")["access_policy"]
    if (
        access["mood_or_questionnaire_csv_opened"]
        or access["model_training"]
        or access["raw_text_or_source_ids_persisted"]
    ):
        raise AssertionError("MEPS scope violated")
    probes = pd.read_csv(out / "natural/public_probe_scores.csv")
    qualification = {
        "status": "NOT_QUALIFIED_FOR_ACT_INTERPRETATION",
        "basis": "Observed elementary public-probe errors; no post-hoc accuracy cutoff",
        "public_probes": {
            language: {
                "correct": int((group.expected == group.predicted).sum()),
                "total": len(group),
                "errors": group.loc[
                    group.expected != group.predicted, ["expected", "predicted"]
                ].to_dict("records"),
            }
            for language, group in probes.groupby("language")
        },
        "allowed_use": ["observer_failure_analysis", "output_context_dependence"],
        "disallowed_use": [
            "individual_or_group_act_profile",
            "personality_or_causal_interpretation",
            "training_targets",
        ],
        "human_gold_labels": False,
    }
    write(out / "natural/observer_qualification.json", qualification)
    return {
        "nli_pairs": len(nli),
        "natural_participants": len(changes),
        "natural_changed_labels": int(sum(r["changed_top_act"] for r in changes)),
        "natural_mean_js": float(np.mean([r["jensen_shannon_nats"] for r in changes])),
        "logit_replay_error": maximum,
        "model_forwards_repeated": False,
        "human_gold_labels": False,
        "natural_act_interpretation": qualification["status"],
    }


def main():
    """Save independent audit and inventory, excluding macOS metadata sidecars."""
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    c, cfg = read(a.output / "contract.json"), read(a.output / "config.json")
    for path, expected in c["files"].items():
        if sha(ROOT / path) != expected:
            raise AssertionError(f"Frozen source changed: {path}")
    with threadpool_limits(limits=1):
        numerical, observers = audit_math(a.output, cfg), audit_observers(a.output)
    result = {
        "status": "PASS_COMPUTATIONAL_AND_SCOPE",
        "math": numerical,
        "observers": observers,
        "scientific_counterexamples_retained": [
            "weak_assignment",
            "direct_assignment",
            "no_assignment",
            "constant_partner",
            "NLI_negative_question_scope",
            "Qwen_public_act_probe_failure",
        ],
        "natural_causal_claim": False,
    }
    write(a.output / "audit.json", result)
    files = sorted(
        v
        for v in a.output.rglob("*")
        if v.is_file()
        and not v.name.startswith("._")
        and v.name != "delivery_manifest.json"
    )
    write(
        a.output / "delivery_manifest.json",
        {str(v.relative_to(a.output)): sha(v) for v in files},
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
