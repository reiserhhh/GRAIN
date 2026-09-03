#!/usr/bin/env python
"""P0: GRAIN estimator correctness on synthetic ground truth (plan v2, Phase 1).

Generates bag-of-words corpora with planted person / condition / interaction
effects on the REAL anchor lexicon categories, pushes them through the REAL
v2 scoring + centering + aggregation code, and checks recovery. Null worlds
check false-positive behavior. Also benchmarks the method-of-moments variance
decomposition against MixedLM.

No real user data or labels are used. This doubles as a permanent regression
harness: rerun after any scoring-code change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.suica_v2_lib import (  # noqa: E402
    V2_CONSTRUCTS,
    base_from_slices,
    check_scorer_equivalence,
    fast_anchor_rates,
    mom_variance_components,
    noise_corrected_react_amplitude,
    react_signature,
    score_slices_v2,
    spearman_brown,
)
from scripts.run_suica_narrative_projective_anchor_validation_v2 import ANCHOR_LEXICONS  # noqa: E402
from scripts.build_suica_directive_growth_strict_repair_v2 import ADVERSITY_TERMS  # noqa: E402

OUT_DIR = ROOT / "results" / "suica_synthetic_ground_truth_v2"
REPORT = ROOT / "reports" / "suica_synthetic_ground_truth_v2.md"

CATEGORIES = {
    "novelty_play": sorted(ANCHOR_LEXICONS["novelty_play"]),
    "directive": sorted(ANCHOR_LEXICONS["directive"]),
    "second_person": sorted(ANCHOR_LEXICONS["second_person"]),
    "self_focus": sorted(ANCHOR_LEXICONS["self_focus"]),
    "negative_affect": sorted(ANCHOR_LEXICONS["negative_affect"]),
    "conflict_threat": sorted(ANCHOR_LEXICONS["conflict_threat"]),
    "uncertainty": sorted(ANCHOR_LEXICONS["uncertainty"]),
    "redemption_growth": sorted(ANCHOR_LEXICONS["redemption_growth"]),
    "adversity": sorted(ADVERSITY_TERMS),
}
DRIVER_OF = {
    "novelty_play": "novelty",
    "directive": "directive",
    "second_person": "directive",
    "self_focus": "selffocus",
    "negative_affect": "tension",
    "conflict_threat": "tension",
    "uncertainty": "tension",
    "redemption_growth": "adversity",
    "adversity": "adversity",
}
DRIVERS = ["novelty", "directive", "selffocus", "tension", "adversity"]
CONSTRUCT_DRIVER = {
    "novelty_play_v2": "novelty",
    "directive_action_v2": "directive",
    "adversity_recovery_v2": "adversity",
    "first_person_usage_v2": "selffocus",
    "tension_core_v2": "tension",
}
FILLER = [
    "table", "window", "walk", "walked", "music", "garden", "coffee", "street", "cloud",
    "paper", "stone", "river", "morning", "evening", "chair", "road", "tree", "light",
    "water", "house", "door", "hand", "eye", "food", "sound", "color", "line", "page",
    "city", "field", "air", "wall", "floor", "box", "cup", "glass", "shirt", "shoe",
]
BASE_W = {"filler": 0.70, "self_focus": 0.06, "second_person": 0.03, "directive": 0.03,
          "uncertainty": 0.03, "negative_affect": 0.025, "conflict_threat": 0.025,
          "novelty_play": 0.03, "redemption_growth": 0.025, "adversity": 0.025}
WORLDS = {
    "W1_person": (0.55, 0.20, 0.10),
    "W2_condition": (0.15, 0.65, 0.10),
    "W3_interaction": (0.25, 0.25, 0.50),
    "W0_null": (0.0, 0.0, 0.0),
}
N_USERS, N_COND_POOL, N_COND_PER_USER, SLICES_PER_CELL, TOKENS = 240, 10, 6, 6, 110
NULL_REPS = 20  # raised from 8 (referee R3-2: tighter FPR estimate)


def _check_filler() -> None:
    lex_words = set().union(*ANCHOR_LEXICONS.values()) | set(ADVERSITY_TERMS)
    bad = [w for w in FILLER if w in lex_words]
    if bad:
        raise AssertionError(f"filler words collide with lexicons: {bad}")


def gen_world(seed: int, s_b: float, s_c: float, s_g: float, *, n_users: int = N_USERS):
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    theta = {d: rng.normal(0, 1, n_users) for d in DRIVERS}
    delta = rng.normal(0, 1, (N_COND_POOL, len(cats)))
    g = {d: rng.normal(0, 1, (n_users, N_COND_POOL)) for d in DRIVERS}
    rows, truth_rows = [], []
    for u in range(n_users):
        conds = rng.choice(N_COND_POOL, size=N_COND_PER_USER, replace=False)
        for c in conds:
            w = np.array([BASE_W[k] * np.exp(
                s_b * theta[DRIVER_OF[k]][u] + s_c * delta[c, i] + s_g * g[DRIVER_OF[k]][u, c]
            ) for i, k in enumerate(cats)])
            w_full = np.concatenate([[BASE_W["filler"]], w])
            p = w_full / w_full.sum()
            truth_rows.append({"user_id": f"u{u:04d}", "condition": f"c{c:02d}",
                               **{f"p_{k}": p[i + 1] for i, k in enumerate(cats)}})
            for s in range(SLICES_PER_CELL):
                n_tok = max(60, rng.poisson(TOKENS))
                counts = rng.multinomial(n_tok, p)
                toks: list[str] = []
                if counts[0]:
                    toks.extend(rng.choice(FILLER, size=counts[0]).tolist())
                for i, k in enumerate(cats):
                    if counts[i + 1]:
                        toks.extend(rng.choice(CATEGORIES[k], size=counts[i + 1]).tolist())
                rng.shuffle(toks)
                rows.append({"user_id": f"u{u:04d}", "condition": f"c{c:02d}",
                             "slice_index": s, "slice_text": " ".join(toks)})
    return pd.DataFrame(rows), pd.DataFrame(truth_rows), theta


def truth_constructs(truth: pd.DataFrame) -> pd.DataFrame:
    t = truth.copy()
    t["novelty_play_v2"] = 100 * t["p_novelty_play"]
    t["directive_action_v2"] = np.sqrt(100 * t["p_directive"] * 100 * t["p_second_person"])
    t["first_person_usage_v2"] = 100 * t["p_self_focus"]
    proj = 100 * (t["p_negative_affect"] + t["p_conflict_threat"] + t["p_uncertainty"])
    t["tension_core_v2"] = 0.40 * proj + 0.35 * 100 * t["p_uncertainty"] + 0.25 * 100 * t["p_conflict_threat"]
    t["adversity_recovery_v2"] = 0.4 * 100 * t["p_redemption_growth"]  # strict part has no analytic form
    return t


def planted_shares(truth_c: pd.DataFrame, construct: str) -> dict[str, float]:
    m = truth_c.pivot_table(index="user_id", columns="condition", values=construct)
    grand = float(np.nanmean(m.to_numpy()))
    row = (m.sub(m.mean(axis=0), axis=1)).mean(axis=1)
    col = (m.sub(m.mean(axis=1), axis=0)).mean(axis=0)
    resid = m.sub(grand)
    for _ in range(3):
        resid = resid.sub(resid.mean(axis=1), axis=0)
        resid = resid.sub(resid.mean(axis=0), axis=1)
    vb, vc, vg = float(row.var(ddof=1)), float(col.var(ddof=1)), float(np.nanvar(resid.to_numpy(), ddof=1))
    total = vb + vc + vg
    return {"person": vb / total if total else np.nan,
            "condition": vc / total if total else np.nan,
            "interaction": vg / total if total else np.nan}


def disjoint_condition_retest(scored: pd.DataFrame, construct: str) -> float:
    halves = []
    for user_id, group in scored.groupby("user_id"):
        conds = sorted(group["condition"].unique())
        if len(conds) < 4:
            continue
        half_a = set(conds[0::2])
        a = group.loc[group["condition"].isin(half_a)]
        b = group.loc[~group["condition"].isin(half_a)]
        halves.append((user_id, a, b))
    if not halves:
        return float("nan")
    frame_a = pd.concat([a for _, a, _ in halves])
    frame_b = pd.concat([b for _, _, b in halves])
    base_a = base_from_slices(frame_a, construct, centered=True).set_index("user_id")["base_score"]
    base_b = base_from_slices(frame_b, construct, centered=True).set_index("user_id")["base_score"]
    joined = pd.concat([base_a, base_b], axis=1, keys=["a", "b"]).dropna()
    return float(joined["a"].corr(joined["b"]))


def mixedlm_shares(scored: pd.DataFrame, construct: str) -> dict[str, float]:
    import statsmodels.formula.api as smf
    data = scored[["user_id", "condition", construct]].dropna().copy()
    data["cell"] = data["user_id"] + ":" + data["condition"]
    data["g"] = 1
    md = smf.mixedlm(f"{construct} ~ 1", data, groups="g",
                     vc_formula={"u": "0 + C(user_id)", "c": "0 + C(condition)", "uc": "0 + C(cell)"})
    fit = md.fit(reml=True, method="lbfgs", maxiter=200)
    vcs = {k: float(v) for k, v in fit.vcomp_named.items()} if hasattr(fit, "vcomp_named") else {}
    if not vcs:
        names = list(md.exog_vc.names) if hasattr(md, "exog_vc") else ["u", "c", "uc"]
        vcs = dict(zip(names, [float(v) for v in fit.vcomp]))
    resid = float(fit.scale)
    total = sum(vcs.values()) + resid
    return {"person_share": vcs.get("u", np.nan) / total, "condition_share": vcs.get("c", np.nan) / total,
            "interaction_share": vcs.get("uc", np.nan) / total, "residual_share": resid / total}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _check_filler()
    rng = np.random.default_rng(7)
    sample_texts = [" ".join(rng.choice(FILLER + CATEGORIES["self_focus"] + CATEGORIES["uncertainty"],
                                        size=80).tolist()) for _ in range(200)]
    tier_u_path = ROOT / "data_sets/prepared/suica_tiers_v2/tier_u_comments.parquet"
    if tier_u_path.exists():
        real = pd.read_parquet(tier_u_path, columns=["body"]).head(300)["body"].astype(str).tolist()
        sample_texts.extend(real)
    equiv = check_scorer_equivalence(sample_texts)

    results: dict[str, dict] = {"scorer_equivalence_max_diff": equiv}
    world_seeds = {"W1_person": 1001, "W2_condition": 1002, "W3_interaction": 1003}
    world_rows = []
    for world, (s_b, s_c, s_g) in WORLDS.items():
        if world == "W0_null":
            continue
        slices, truth, theta = gen_world(world_seeds[world], s_b, s_c, s_g)
        scored = score_slices_v2(slices)
        truth_c = truth_constructs(truth)
        for construct in V2_CONSTRUCTS:
            driver = CONSTRUCT_DRIVER[construct]
            base = base_from_slices(scored, construct, centered=True)
            order = base["user_id"].str.slice(1).astype(int).to_numpy()
            rank_r = float(stats.spearmanr(theta[driver][order], base["base_score"]).statistic)
            est = mom_variance_components(scored, construct)
            planted = planted_shares(truth_c, construct)
            est_bcg = np.array([est["person_share"], est["condition_share"], est["interaction_share"]])
            est_bcg_n = est_bcg / est_bcg.sum() if np.nansum(est_bcg) > 0 else est_bcg
            pl = np.array([planted["person"], planted["condition"], planted["interaction"]])
            dom_match = bool(np.nanargmax(est_bcg_n) == np.nanargmax(pl))
            dom_err = float(abs(est_bcg_n[np.nanargmax(pl)] - pl[np.nanargmax(pl)]))
            world_rows.append({
                "world": world, "construct": construct, "rank_recovery_spearman": rank_r,
                "planted_person": pl[0], "planted_condition": pl[1], "planted_interaction": pl[2],
                "est_person": est_bcg_n[0], "est_condition": est_bcg_n[1], "est_interaction": est_bcg_n[2],
                "dominant_component_match": dom_match, "dominant_share_abs_error": dom_err,
                "est_residual_share": est["residual_share"],
            })
        if world == "W3_interaction":
            cell = react_signature(scored, "tension_core_v2")
            amp = noise_corrected_react_amplitude(cell)
            truth_react = truth_c.copy()
            truth_react["centered"] = truth_react.groupby("condition")["tension_core_v2"].transform(lambda s: s - s.mean())
            true_amp = truth_react.groupby("user_id")["centered"].std(ddof=1).rename("true_amp")
            joined = amp.set_index("user_id").join(true_amp).dropna()
            results["react_amplitude_recovery_r_W3"] = float(joined["react_amplitude_corrected"].corr(joined["true_amp"]))
        if world == "W1_person":
            try:
                sub_users = [f"u{idx:04d}" for idx in range(150)]
                mm = mixedlm_shares(scored.loc[scored["user_id"].isin(sub_users)], "tension_core_v2")
                mom = mom_variance_components(scored.loc[scored["user_id"].isin(sub_users)], "tension_core_v2")
                results["mixedlm_vs_mom_tension_W1"] = {
                    "mixedlm": mm,
                    "mom": {k: mom[k] for k in ("person_share", "condition_share", "interaction_share", "residual_share")},
                }
            except Exception as exc:  # pragma: no cover
                results["mixedlm_vs_mom_tension_W1"] = {"error": str(exc)}

    null_rows = []
    fp = 0
    fp_total = 0
    for rep in range(NULL_REPS):
        slices, _truth, _theta = gen_world(5000 + rep, 0.0, 0.0, 0.0, n_users=160)
        scored = score_slices_v2(slices)
        for construct in V2_CONSTRUCTS:
            r = disjoint_condition_retest(scored, construct)
            n_eff = scored["user_id"].nunique()
            z = np.arctanh(np.clip(r, -0.999, 0.999)) * np.sqrt(max(4, n_eff) - 3)
            significant = bool(abs(z) > 1.96)
            fp += int(significant)
            fp_total += 1
            est = mom_variance_components(scored, construct)
            null_rows.append({"rep": rep, "construct": construct, "null_retest_r": r,
                              "null_person_share": est["person_share"], "flagged_significant": significant})
    null_frame = pd.DataFrame(null_rows)
    world_frame = pd.DataFrame(world_rows)
    world_frame.to_csv(OUT_DIR / "world_recovery.csv", index=False)
    null_frame.to_csv(OUT_DIR / "null_worlds.csv", index=False)

    w1 = world_frame.loc[world_frame["world"].eq("W1_person")]
    criteria = {
        "scorer_equivalence_pass": bool(equiv <= 1e-9),
        # Criterion scope (referee R3-2): rank recovery is defined on the person-
        # dominant world W1 only; W2/W3 recovery is reported descriptively.
        "rank_recovery_pass": bool((w1["rank_recovery_spearman"] >= 0.90).sum() >= 4),
        "dominant_component_pass": bool(world_frame.groupby("world")["dominant_component_match"].sum().ge(4).all()),
        "dominant_share_error_pass": bool((world_frame["dominant_share_abs_error"] <= 0.10).mean() >= 0.8),
        "null_person_share_pass": bool(null_frame["null_person_share"].mean() <= 0.02),
        "null_false_positive_rate": fp / max(1, fp_total),
        "null_false_positive_pass": bool(fp / max(1, fp_total) <= 0.08),
    }
    hard_keys = [k for k in criteria if k.endswith("_pass")]
    criteria["P0_verdict"] = "pass" if all(criteria[k] for k in hard_keys) else "fail"
    results["criteria"] = criteria
    (OUT_DIR / "p0_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")

    lines = ["# GRAIN P0 Synthetic Ground Truth (v2)", "",
             f"Scorer equivalence max diff: {equiv:.2e}", "",
             "## Recovery by world", "", world_frame.round(4).to_markdown(index=False), "",
             "## Null worlds", "",
             f"mean null person share: {null_frame['null_person_share'].mean():.4f}; "
             f"false-positive rate: {fp}/{fp_total} = {fp/max(1,fp_total):.3f}", "",
             "## MixedLM vs MoM (tension, W1)", "",
             "```json", json.dumps(results.get("mixedlm_vs_mom_tension_W1", {}), indent=2, default=float), "```", "",
             f"react amplitude recovery r (W3, tension): {results.get('react_amplitude_recovery_r_W3', float('nan')):.4f}", "",
             "## Criteria", "", "```json", json.dumps(criteria, indent=2, default=float), "```"]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(criteria, indent=2, default=float))


if __name__ == "__main__":
    main()
