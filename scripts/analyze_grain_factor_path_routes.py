#!/usr/bin/env python3
"""Publish aggregate all-route results, corrected identity audit and CV diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import ID_SEED
from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.factor_path_routes import (
    fold_diagnostics,
    historical_overlap,
    nested_predictions,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.run_grain_factor_path_routes import FROZEN, MATERIAL, factor_names

REPORT = ROOT / "reports/grain_vector_mechanisms/FACTOR_PATH_ALL_ROUTES_REPORT.md"


def table(frame):
    """Render aggregate-only Markdown without an optional tabulate dependency."""

    def value(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "NA"
            return f"{v:.2e}" if 0 < abs(v) < 0.0001 else f"{v:.4f}"
        return str(v).replace("|", "/")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join(value(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def analyze(source, native, report):
    """Require complete route packets and emit no author identifiers publicly."""
    if (
        read(source / "status.json")["status"] != "CPU_COMPLETE"
        or read(native / "analysis_status.json")["status"] != "COMPLETE_GFP02"
    ):
        raise ValueError("All four routes must complete before combined reporting")
    old = pd.read_csv(FROZEN / "scores_native.csv", dtype={"user_id": str})
    all_source_ids = {d["author_id"] for d in read(MATERIAL)}
    original_overlap = historical_overlap(old.user_id, all_source_ids, seed=ID_SEED)
    data = pd.read_csv(source / "private_joined.csv", dtype={"author_id": str})
    final_ids = set(data.author_id)
    native_ids = {d["author_id"] for d in read(native / "private_design.json")}
    direction_source = ROOT / read(native / "manifest.json")["source"]
    direction_source_count = sum(
        r["authors"]
        for r in read(direction_source / "direction_freeze.json")["queries"]
    )
    final_overlap = historical_overlap(old.user_id, final_ids, seed=ID_SEED)
    native_overlap = historical_overlap(old.user_id, native_ids, seed=ID_SEED)
    if final_overlap or native_overlap or native_ids & all_source_ids:
        raise ValueError("Corrected author-disjoint analysis failed")
    audit = {
        "original_source_authors": len(all_source_ids),
        "original_source_reference_overlap": len(original_overlap),
        "final_source_authors": len(final_ids),
        "final_source_reference_overlap": len(final_overlap),
        "native_authors": len(native_ids),
        "native_source_overlap": 0,
        "native_reference_overlap": 0,
        "native_direction_fit_source_authors": direction_source_count,
        "correction": "source_r1 raw-ID overlap field superseded; use source_r2 for external links/CV",
        "identity_rule": "stable_key(raw_user_id, cohort.ID_SEED)[:24]",
    }
    write(source / "corrected_identity_audit.json", audit)
    predictions = pd.read_csv(
        source / "private_nested_predictions.csv", dtype={"author_id": str}
    )
    diagnostics, shared, private_shared = [], [], []
    for (q, arm), g in predictions.groupby(["query", "arm"]):
        for condition, y, pred in (
            ("real", g.y_true, g.prediction),
            ("permuted", g.permuted_y, g.permuted_prediction),
        ):
            diagnostics.append(
                {
                    "query": q,
                    "arm": arm,
                    "condition": condition,
                    **fold_diagnostics(y, pred, g.fold, binary=q.endswith("_cont")),
                }
            )
    pd.DataFrame(diagnostics).to_csv(
        source / "fold_metric_diagnostics.csv", index=False
    )
    # This is an explanatory post-hoc decomposition, not another label-tuned head.
    with threadpool_limits(limits=2):
        for q, g in data.groupby("query"):
            g = g.sort_values("author_id")
            fitted, folds, _ = nested_predictions(
                g[factor_names()], g.score, g.author_id
            )
            m, y = g.score.to_numpy(), g.y_true.to_numpy()
            residual = m - fitted
            denom = np.std(m, ddof=1) * np.std(y, ddof=1)
            linked = float(np.cov(fitted, y)[0, 1] / denom)
            remainder = float(np.cov(residual, y)[0, 1] / denom)
            total = association(m, y)["pearson_r"]
            shared.append(
                {
                    "query": q,
                    "n": len(g),
                    "grain_to_function_oof_r": association(fitted, m)["pearson_r"],
                    "total_function_criterion_r": total,
                    "grain_linked_covariance_contribution": linked,
                    "remaining_covariance_contribution": remainder,
                    "accounting_error": abs(total - linked - remainder),
                }
            )
            private_shared.extend(
                {
                    "author_id": a,
                    "query": q,
                    "fold": int(f),
                    "function": float(mm),
                    "grain_linked_function": float(gg),
                    "remaining_function": float(rr),
                }
                for a, f, mm, gg, rr in zip(
                    g.author_id, folds, m, fitted, residual, strict=True
                )
            )
    shared = pd.DataFrame(shared)
    shared.to_csv(source / "shared_external_covariance_accounting.csv", index=False)
    pd.DataFrame(private_shared).to_csv(
        source / "private_shared_function.csv", index=False
    )
    path = pd.read_csv(source / "factor_path_associations.csv")
    ext = pd.read_csv(source / "factor_criterion_associations.csv")
    metric = pd.read_csv(source / "prediction_metrics.csv")
    deltas = pd.read_csv(source / "paired_prediction_deltas.csv")
    effect = pd.read_csv(native / "native_factor_effect_summary.csv")
    consistency = pd.read_csv(source / "factor_view_consistency.csv")
    supports = pd.read_csv(source / "material_support.csv")
    primary = path[path.term == "score"]
    main_effects = int((effect.abs_gain_lower > 0).sum())
    groups = data.groupby("source_group").author_id.nunique().to_dict()
    summary = {
        "status": "COMPLETE_FOUR_BOUNDED_ROUTES",
        "routes": ["GFP-01", "GFP-02", "GFP-03", "GFP-04"],
        "source_groups": groups,
        "factor_path_cells": len(path),
        "primary_factor_path_cells": len(primary),
        "primary_path_bh05_links": int((primary.q_bh < 0.05).sum()),
        "external_cells": len(ext),
        "external_bh05_links": int((ext.q_bh < 0.05).sum()),
        "native_direction_query_cells": len(effect),
        "native_abs_gain_ci_positive_cells": main_effects,
        "native_calls": read(native / "status.json")["calls"],
        "identity_audit": audit,
        "source": str(source.relative_to(ROOT)),
        "native": str(native.relative_to(ROOT)),
        "cpu_seconds": read(source / "status.json")["seconds"],
        "native_seconds": read(native / "status.json")["seconds"],
        "max_shared_accounting_error": float(shared.accounting_error.max()),
        "posthoc_diagnostics": "CV fold offsets and cross-fitted function decomposition; no model promotion from these",
        "remaining_scope": "MEPS small traits, modern dynamic GRAIN factors, full causal abstraction and universal theory remain separate",
    }
    write(source / "all_routes_summary.json", summary)
    paths = primary.reindex(
        primary.pearson_r.abs().sort_values(ascending=False).index
    ).head(8)
    externals = ext.reindex(
        ext.pearson_r.abs().sort_values(ascending=False).index
    ).head(8)
    scores = (
        metric[metric.control == "real_labels"]
        .pivot(index="query", columns="arm", values="pearson_r")
        .reset_index()
    )
    auc = (
        metric[(metric.control == "real_labels") & metric.auc.notna()]
        .pivot(index="query", columns="arm", values="auc")
        .reset_index()
    )
    e_gain = deltas[(deltas["query"] == "extraversion") & (deltas.candidate == "joint")]
    e_metric = metric[
        (metric.control == "real_labels") & (metric["query"] == "extraversion")
    ].set_index("arm")
    candidate_effect = effect[
        (effect["query"] == "TF_cont") & (effect.factor == "native:F03")
    ].iloc[0]
    candidate_criterion = ext[
        (ext["query"] == "TF_cont") & (ext.factor == "native:F03")
    ].iloc[0]
    gain_path = e_gain[e_gain.reference == "path"].iloc[0]
    gain_grain = e_gain[e_gain.reference == "grain"].iloc[0]
    weaker = int((effect.real_mean_abs < effect.null_mean_abs).sum())
    nulls = (
        metric[metric.control == "permuted_labels"]
        .groupby("arm")
        .pearson_r.agg(["min", "mean", "max"])
        .reset_index()
    )
    diag = pd.DataFrame(diagnostics)
    diag = diag[
        (diag.condition == "real")
        & diag["query"].str.endswith("_cont")
        & diag.arm.isin(["path", "joint"])
    ]
    text = f"""# GRAIN factor / query-path all-route results

Date: 2026-09-10. **All four declared routes executed.**

## 1. Decision

The useful join is presently **factor interpretation through query paths**, not
a universal accuracy-enhancing concatenation. Actual pre-existing GRAIN factors
connect with model judgments; one of eight tested native-factor/query directions
exceeds this run's matched shuffled-direction control. Real questionnaire links
are smaller. Extraversion is a useful joint-prediction candidate, not an all-trait
promotion. GRAIN factors and query scores remain distinct objects.

| Route | Finding | Next use |
| --- | --- | --- |
| GFP01 factor/path association | {len(primary)} primary cells; {int((primary.q_bh < 0.05).sum())} BH q<.05 links | Anonymous factor-to-query interpretation map |
| GFP02 actual activation intervention | {main_effects}/8 cells have positive paired absolute-gain CI over one norm-matched shuffled null | Follow native:F03/T-F as a model-direction candidate |
| GFP03 real criteria | Strongest individual-factor absolute r={ext.pearson_r.abs().max():.3f}; {int((ext.q_bh < 0.05).sum())}/108 BH q<.05 links | Candidate associations, retain codes rather than assign personality names |
| GFP04 prediction | Extraversion GRAIN {e_metric.loc["grain", "pearson_r"]:.3f}, path {e_metric.loc["path", "pearson_r"]:.3f}, joint {e_metric.loc["joint", "pearson_r"]:.3f}; no uniform benefit | Trait-specific optional joint head, not replace GRAIN |

## 2. Materials and the actual system

Final association/CV set: {len(final_ids)} authors ({groups}), three upstream
12-comment views each. Four whole-document, four native-comment and four
fixed-128-cross factors are scored by frozen historical TF-IDF/SVD/factor-posterior
bundles. Their basis used101 discovery and29 calibration authors. No factors,
factor names or GRAIN reference norms are fitted to the present labels. Each
factor is namespaced by operator; equal numeric factor IDs do not imply the
same axis across operators. These are actual historical factor scores, **not**
the newer query mean/configuration/final scores renamed as GRAIN.

Identity correction: source_r1 initially compared raw historical usernames to
new pseudonyms. The exact cohort hash mapping reveals one overlap (MBTI-only).
It is excluded in final source_r2, leaving213 Big5-only and63 MBTI-only authors.
The original source_r1 results and code snapshot are retained; its raw-string
overlap claim is superseded. Native interventions use {direction_source_count} source authors for
label-free direction fitting; all12 evaluation authors are disjoint from both
that source and all180 historical factor-study authors. No MEPS or bridge fit,
lockbox opening, private-text cloud request, or long training was performed.

{table(supports.drop(columns="bundle_sha256"))}

The frozen factor support test checks text quantity, not a certificate of
universal psychometric transport. Original timestamps are absent from these
masked source-view caches, so view agreement is not cross-day reliability.
Factor-view median correlations range
{consistency.groupby("factor").pearson_r.median().min():.3f}–{consistency.groupby("factor").pearson_r.median().max():.3f}.
The old optional geometry gate was not widened. All source comments were retained
by minimum-length scoring; upstream masking stays unchanged. Term-flag rates are
reported in `factor_freeze.json`; a flag is not itself verified score leakage.

## 3. Association and interpretation

Largest factor / frozen-function associations (all108 primary tests retained):

{table(paths[["factor", "query", "n", "pearson_r", "r_lower", "r_upper", "q_bh"]])}

Largest real questionnaire associations:

{table(externals[["factor", "query", "n", "pearson_r", "r_lower", "r_upper", "q_bh"]])}

The former concerns the model's trait judgment; the latter concerns measured
human criteria. A factor may align strongly with a model's reading strategy but
weakly with a questionnaire. That distinction is an empirical outcome, not a
reason to relabel all shared text structure as personality.

For a source-only cross-fitted GRAIN predictor of the function value, m_hat_G,
the additional explanatory accounting is exact:

    corr(m,Y) = cov(m_hat_G,Y)/(sd(m)sd(Y))
                + cov(m-m_hat_G,Y)/(sd(m)sd(Y)).

No Y enters the fit or hyperparameter choice of m_hat_G. The terms are signed
covariance contributions, not percentages, orthogonal causal components, or
mediation estimates. This post-hoc analysis asks which part of the model's
external link is shared with this GRAIN factor family.

{table(shared.drop(columns="accounting_error"))}

## 4. Actual vector interventions

Four source-standardized native factors predict P/Q state directions jointly
with Ridge alpha100. Move +/-0.25 of each direction while keeping original
context and all other token states unchanged, then run the actual fixed model
suffix. Controls use source-author-permuted factor directions with matched norm.
One direction-null realization is not a permutation significance test.

{table(effect[["query", "factor", "real_mean_effect", "real_lower", "real_upper", "real_positive_fraction", "null_mean_abs", "abs_gain", "abs_gain_lower", "abs_gain_upper"]])}

Effects are half the plus-minus difference in the model's conditional score.
For native:F03/T-F, positive sign fraction is {candidate_effect.real_positive_fraction:.3f}; mean effect {candidate_effect.real_mean_effect:.6f} versus null
absolute effect {candidate_effect.null_mean_abs:.6f} (about {candidate_effect.real_mean_abs / candidate_effect.null_mean_abs:.2f}x). The effect is small in score units but is
an actual, repeatable finite model intervention. It is **not** a change to the
author's trait. On the final source set this factor's T-F criterion r is
{candidate_criterion.pearson_r:.4f}, so this example presently identifies a model-reading direction, not a
validated T-F construct. {weaker} of eight real directions are weaker than the
matched null in absolute response; generic state sensitivity is not enough.
No cross-query vector axis equivalence is assumed: each query gets its own
literal-token-compatible fitted direction and displacement units.

Cost: {summary["native_calls"]} native calls, {summary["native_seconds"]:.1f}s;
384 signed suffix edits,24 exact full-state replays, no model training.
Affine predicted effects and curvature are also saved; affine/native agreement
alone is not substituted for actual replay.

## 5. Prediction comparison and metric diagnosis

Nested author CV: 5 outer folds,3 inner folds; Ridge alpha1/10/100/1000 selected
only by inner MSE. Scalers are train-only. All four arms share the same splits;
GRAIN12/path3/joint15/interactions51. Frozen function/native are descriptive
unfitted references, not additional supervised-CV heads. Continuous scores are
not clipped or automatically calibrated probabilities.

Pooled out-of-fold Pearson r:

{table(scores)}

Pooled axis AUC:

{table(auc)}

Extraversion participant-paired bootstrap (1000 draws):

{table(e_gain[["candidate", "reference", "delta_r", "delta_r_lower", "delta_r_upper"]])}

Joint changes E over GRAIN alone by {gain_grain.delta_r:.3f} [{gain_grain.delta_r_lower:.3f},{gain_grain.delta_r_upper:.3f}]; over path-only by
{gain_path.delta_r:.3f} [{gain_path.delta_r_lower:.3f},{gain_path.delta_r_upper:.3f}]. Interaction terms do not improve E in this packet. Neither the largest
single gain nor these CIs establish all-trait benefit; intervals resample fixed
OOF predictions, not an entirely retrained pipeline each draw.

Important metric observation: fold-specific training means alone can correlate
negatively with held labels. Do not interpret the negative pooled MBTI r as a
negative personality relationship. Here is the preserved supplementary ranking
and calibration diagnosis, without changing any predictions:

{table(diag[["query", "arm", "within_fold_centered_r", "mean_within_fold_auc", "auc_valid_folds", "minority_count", "training_mean_pooled_r", "mse_gain_vs_training_mean"]])}

SN has only four minority authors and one single-class test fold. Its pooled
AUC is particularly unstable; this tiny source panel is not the official MBTI
benchmark. One label permutation per arm gives the following pooled-r range:

{table(nulls)}

These negative controls are not high positive leakage signals; negative null
correlations also expose the pooled-fold intercept issue. No signs or AUCs
were flipped after inspection. No threshold, alpha grid, factor or observation
operator was chosen using held labels. The 277-to276 identity correction changes
MBTI fold allocation as well as membership; it is not an accuracy improvement.

## 6. Reproduction and next decision

Plan: `docs/GRAIN_FACTOR_PATH_ALL_ROUTES_PLAN_V1.md`.
Private CPU: `{source.relative_to(ROOT)}`.
Private native: `{native.relative_to(ROOT)}`.

```bash
"../project persona/.venv/bin/python" scripts/run_grain_factor_path_routes.py --output-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_source
"../project persona/.venv/bin/python" scripts/run_grain_factor_path_interventions.py --source-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_source --output-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_native
"../project persona/.venv/bin/python" scripts/analyze_grain_factor_path_routes.py --source-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_source --native-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_native --report-path results/grain_vector_mechanisms/factor_path_routes_v1/new_report.md
```

Fresh commands use the corrected276-person direction source. To reproduce the
historical native_r1 exactly, use its original277-source directions and
source_r1/code_snapshot.zip; the report deliberately records that distinction.
Never overwrite completed result folders. Replaying `--stage analyze` requires
no neural calls. Quantitative tables, predictions, state directions and per-unit
packets stay local/gitignored; only aggregate findings and code are public.

Priority now: connect **the same factor** across all three matrices,
A=factor/model association, J=actual intervention sensitivity, B=factor/real
criterion association. Whole:F02/T-F has the stronger observational query link;
native:F03/T-F has the cleaner current intervention candidate; whole:F04/E has
the stronger external candidate. They are not interchangeable discoveries.
Choose the next bounded experiment to connect those identities rather than
expand a predictive grid. Keep the E joint head as an optional use case.

This pilot closes four executable routes on12 existing factors, not all GRAIN
versions or all possible psychological constructs. MEPS small traits, dynamic
response factors and population-specific construct naming remain future work.
Theoretical precedents: [TCAV](https://proceedings.mlr.press/v80/kim18d.html),
[concept bottlenecks](https://proceedings.mlr.press/v119/koh20a.html), and
[causal abstraction](https://www.jmlr.org/papers/v26/23-0058.html).
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    write(
        source / "report_manifest.json",
        {
            "report_sha256": sha(report),
            "script_sha256": sha(__file__),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "native_manifest_sha256": sha(native / "manifest.json"),
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=REPORT)
    args = parser.parse_args()
    if not args.source_dir.resolve().is_relative_to(
        BASE.resolve()
    ) or not args.native_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Read private ignored mechanism packets only")
    analyze(
        args.source_dir.resolve(), args.native_dir.resolve(), args.report_path.resolve()
    )
