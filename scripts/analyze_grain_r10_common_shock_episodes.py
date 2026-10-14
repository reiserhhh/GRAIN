#!/usr/bin/env python
"""Analysis for R10 common-shock episodes (protocol V1 + A1). Episode is the sampling unit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F05/common_shock_episodes_r1"
METRICS = ["state_err2", "state_cover", "state_width", "f1_err2", "f1_cover", "f1_width", "f1_cond_err2",
           "f4_err2", "f4_cover", "f4_width", "f4_cond_err2"]


def load():
    eps = sorted(p for p in (OUT / "episodes").iterdir() if p.is_dir() and not p.name.startswith("._"))
    auth = pd.concat([pd.read_csv(p / "author_metrics.csv", float_precision="round_trip") for p in eps], ignore_index=True)
    info = pd.DataFrame([json.loads((p / "episode.json").read_text()) for p in eps])
    return auth, info


def icc_oneway(d: pd.Series, groups: pd.Series):
    g = pd.DataFrame({"d": d.values, "g": groups.values}).groupby("g")["d"]
    k, m = g.ngroups, g.size().mean()
    grand = d.mean()
    msb = (g.size() * (g.mean() - grand) ** 2).sum() / (k - 1)
    msw = ((d - groups.map(g.mean())) ** 2).sum() / (len(d) - k)
    return float((msb - msw) / (msb + (m - 1) * msw))


def main() -> int:
    auth, info = load()
    E = auth.groupby("world").seed.nunique().to_dict()
    rng = np.random.default_rng(20260926)
    seeds = sorted(auth.seed.unique())
    draws = rng.integers(0, len(seeds), size=(2000, len(seeds)))
    levels, contrasts = [], []
    for (world, budget, method), g in auth.groupby(["world", "budget", "method"]):
        ind = g[g.arm == "independent"].set_index(["seed", "author_id"]).sort_index()
        jnt = g[g.arm == "joint"].set_index(["seed", "author_id"]).sort_index()
        assert ind.index.equals(jnt.index)
        for arm, df in (("independent", ind), ("joint", jnt)):
            for met in METRICS:
                em = df[met].groupby(level=0).mean()
                levels.append({"world": world, "budget": budget, "method": method, "arm": arm, "metric": met, "mean": float(em.mean()),
                               "episode_se": float(em.std(ddof=1) / np.sqrt(len(em))), "episodes": len(em)})
        for met in METRICS:
            d = (jnt[met] - ind[met]).dropna()
            em = d.groupby(level=0).mean().reindex(seeds)
            n_ep = int(em.notna().sum())
            se_ep = float(em.std(ddof=1) / np.sqrt(n_ep))
            se_naive = float(d.std(ddof=1) / np.sqrt(len(d)))
            tq = stats.t.ppf(0.975, n_ep - 1)
            bs = np.nanmean(em.to_numpy()[draws], axis=1)
            icc = icc_oneway(d, pd.Series(d.index.get_level_values(0), index=d.index))
            contrasts.append({"world": world, "budget": budget, "method": method, "metric": met, "mean_diff": float(em.mean()),
                              "episode_se": se_ep, "t_low": float(em.mean() - tq * se_ep), "t_high": float(em.mean() + tq * se_ep),
                              "boot_low": float(np.percentile(bs, 2.5)), "boot_high": float(np.percentile(bs, 97.5)),
                              "naive_author_se": se_naive, "se_ratio_episode_over_naive": se_ep / se_naive if se_naive > 0 else np.nan,
                              "icc_within_episode": icc, "design_effect": 1 + (len(d) / n_ep - 1) * icc,
                              "sd_episode_means": float(em.std(ddof=1)), "episodes": n_ep, "author_units": len(d),
                              "episodes_joint_better": int((em < 0).sum()) if met.endswith("err2") else None})
    pd.DataFrame(levels).to_csv(OUT / "levels.csv", index=False)
    con = pd.DataFrame(contrasts)
    con.to_csv(OUT / "paired_contrasts.csv", index=False)
    # source component (source law varies across independent source episodes)
    src = info.assign(true_common=info.world.map({"core": 0.0, "shared_shock": 0.125}),
                      q_hat=info.moment_total_variance, r_hat=info.log_sigma.map(lambda v: float(np.exp(2 * v[1]))))
    src["common_err"] = src.moment_common_variance - src.true_common
    src["personal_fit_failures_total"] = src.personal_fit_failures.map(lambda d: sum(d.values()))
    src_cols = ["world", "seed", "moment_raw_common_variance", "moment_common_variance", "moment_clipped", "true_common", "common_err", "q_hat", "r_hat",
                "truth_side_realized_common_msq_application", "personal_fit_failures_total", "seconds_total"]
    src[src_cols].to_csv(OUT / "source_component.csv", index=False)
    src_summary = {}
    prim = auth[(auth.budget == 48) & (auth.method == "personal_MAP")]
    for world, s in src.groupby("world"):
        p = prim[prim.world == world]
        d = (p[p.arm == "joint"].set_index(["seed", "author_id"]).state_err2 - p[p.arm == "independent"].set_index(["seed", "author_id"]).state_err2)
        em = d.groupby(level=0).mean().reindex(s.seed.values)
        src_summary[world] = {
            "common_hat_mean": float(s.moment_common_variance.mean()), "common_hat_sd": float(s.moment_common_variance.std(ddof=1)),
            "common_hat_range": [float(s.moment_common_variance.min()), float(s.moment_common_variance.max())],
            "raw_negative_clipped": int(s.moment_clipped.sum()), "q_hat_mean": float(s.q_hat.mean()), "q_hat_sd": float(s.q_hat.std(ddof=1)),
            "r_hat_mean": float(s.r_hat.mean()), "r_hat_sd": float(s.r_hat.std(ddof=1)),
            "source_round_success_min": int(min(min(v) for v in s.source_successful_authors_by_round)),
            "source_noise_success_all": bool(all(all(v) for v in s.source_noise_success_by_round)),
            "personal_fit_failures_total": int(s.personal_fit_failures_total.sum()),
            "corr_primary_diff_vs_common_hat": float(np.corrcoef(em.values, s.moment_common_variance.values)[0, 1]),
            "corr_primary_diff_vs_realized_common_msq": float(np.corrcoef(em.values, s.truth_side_realized_common_msq_application.values)[0, 1]),
        }
    (OUT / "source_component.json").write_text(json.dumps(src_summary, indent=1) + "\n")
    # archived three episodes (old evidence; not new replication)
    ref = pd.read_csv(ROOT / "results/grain_gptpro_foundation_v3/F05/joint_full_r1/joint_state_metrics.csv", float_precision="round_trip")
    ref = ref[ref.world.isin(["core", "shared_shock"]) & (ref.method == "personal_MAP") & (ref.parameter_prefix == 48)]
    ra = ref.groupby(["world", "seed", "author_id", "arm"]).state_MSE.mean().unstack("arm")
    old = (ra.joint - ra.independent).groupby(level=[0, 1]).mean().reset_index(name="episode_mean_state_err2_diff")
    old.to_csv(OUT / "archived_three_episode_reference.csv", index=False)
    files = ["levels.csv", "paired_contrasts.csv", "source_component.csv", "source_component.json", "archived_three_episode_reference.csv"]
    ep_files = sorted(p for p in (OUT / "episodes").rglob("*") if p.is_file() and not p.name.startswith("._"))
    (OUT / "analysis.json").write_text(json.dumps({
        "episodes_per_world": E, "author_rows": len(auth),
        "output_sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files},
        "episode_files_sha256": {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ep_files},
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
