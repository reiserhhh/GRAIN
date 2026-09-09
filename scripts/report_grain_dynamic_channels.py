#!/usr/bin/env python3
"""Build aggregate research documentation and a portable offline HTML report."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_factor_path_routes import table


def render():
    """Publish aggregate evidence only, keeping source authors and texts private."""
    root = BASE / "dynamic_path_channels_v1"
    within = root / "within_comment"
    native = root / "native_direction"
    if (
        read(root / "status.json")["status"]
        != "COMPLETE_BOUNDED_DYNAMIC_CHANNEL_ANALYSIS"
        or read(within / "status.json")["status"] != "COMPLETE_WITHIN_COMMENT_CHANNELS"
        or read(native / "status.json")["status"] != "COMPLETE"
    ):
        raise ValueError("All finite branches must finish before reporting")
    audit = read(root / "restoration_audit.json")
    support = read(within / "manifest.json")
    intervention = read(native / "summary.json")
    cochange = pd.read_csv(root / "cochange_links.csv")
    target = cochange.query(
        "query=='TF_cont' and factor=='whole:F02' and part=='score'"
    )
    trait = pd.read_csv(within / "criterion_links.csv")
    dynamics = pd.read_csv(within / "cochange_links.csv")
    case = trait.query("query=='TF_cont' and feature=='disp::whole:F02'").iloc[0]
    motion = dynamics.query("query=='TF_cont' and feature=='disp::whole:F02'").iloc[0]
    tables = []
    for path, name in ((root, "sampled-comment"), (within, "within-comment")):
        m = pd.read_csv(path / "prediction_metrics.csv")
        view = m[
            m.arm.isin(
                [
                    "subset_0",
                    "subset_1",
                    "subset_4",
                    "subset_7",
                    "dispersion_only_addition",
                    "shuffled_order_addition",
                ]
            )
        ].pivot(index="query", columns="arm", values="pearson_r")
        view = view.rename(
            columns={
                "subset_0": "baseline",
                "subset_1": "+query",
                "subset_4": "+dynamic",
                "subset_7": "+all",
                "dispersion_only_addition": "+dispersion",
                "shuffled_order_addition": "+shuffled",
            }
        )
        view = view[
            ["baseline", "+query", "+dispersion", "+dynamic", "+shuffled", "+all"]
        ].reset_index()
        tables.append(f"### {name} branch: author-held Pearson r\n\n" + table(view))
    accounting = pd.read_csv(root / "gap_accounting.csv")
    mse = accounting.pivot(
        index="query", columns="block", values="mse_gain"
    ).reset_index()
    extreme = max(
        pd.read_csv(p / "gap_accounting.csv").accounting_error.abs().max()
        for p in (root, within)
    )
    meps = pd.read_csv(root / "meps_channel_inventory.csv")
    free = pd.read_csv(root / "meps_participation_links.csv").query(
        "view_kind=='free_chat'"
    )
    free = free.sort_values("p").head(3)[
        ["feature", "scale", "n", "pearson_r", "p", "q"]
    ]
    secondary = trait.sort_values("p").head(6)[
        ["query", "feature", "n", "pearson_r", "r_lower", "r_upper", "p", "q"]
    ]
    scope = [
        {
            "question": "GRAIN changes connect to model queries?",
            "answer": "Yes, selected F02/T-F co-change plus actual held-author intervention",
        },
        {
            "question": "More model linkage guarantees more real-trait linkage?",
            "answer": "No monotonic pattern across all nine queries; level and dispersion differ",
        },
        {
            "question": "Within-text dynamics carry candidate real-criterion information?",
            "answer": "Yes, exploratory F02 dispersion/T-F and F04 drift/extraversion links",
        },
        {
            "question": "Simply add every dynamic feature?",
            "answer": "No universal held-author gain in these fixed feature families",
        },
        {
            "question": "Can the gap be allocated mathematically?",
            "answer": "Predictive MSE gain: yes; true trait/action/reporting-error separation: not with these channels",
        },
    ]
    report = f"""# GRAIN dynamics, personality-query paths and measurement channels

Date: 2026-09-10. Finite execution complete. Discovery-led mechanism experiment,
not a new full-PANDORA benchmark, newly validated psychological scale, or release
scoring change. **The useful new result is an operational dynamic-factor to model
judgment connection, alongside evidence that model-readable and criterion-linked
information need not be the same part of a factor.**

## 1. Question-by-question decisions

{table(pd.DataFrame(scope))}

The source has 276 reference-disjoint authors: 213 Big5-only and 63 MBTI-only.
The same nine recorded criteria are used throughout. Source labels were already
exposed in earlier work; all current supervised estimates are exploratory nested
CV. The native follow-up reuses 12 authors disjoint from source and historical
factor construction. They are not a newly recruited or newly blind cohort.

## 2. Two time scales, not one ambiguous movement axis

Restored {audit["matched_comments"]:,} exact masked comments and their original
timestamps/subreddits. Every author contributes three 12-comment views from
36 distinct discussions. Median author span is {audit["span_days_median"]:.1f}
days; median within-author comment gap is {audit["median_gap_days_median"]:.1f}
days. These are sampled comments, not adjacent conversational responses.

The sampled-comment branch scores four three-comment blocks per view with the
frozen whole/native bundles: 3,312 blocks and eight coordinates. The separate
within-comment branch retains {support["eligible_comments"]:,} of
{support["input_comments"]:,} comments ({support["eligible_comments"] / support["input_comments"]:.1%}),
four lossless contiguous lexical-token spans each: {support["slices"]:,} slices.
All 276 authors remain; at least {support["eligible_min_per_view"]} eligible comments
per view. These are GRAIN lexical/punctuation tokens, not Transformer tokens.
No old factor basis or reference norm is refitted.

Four points support short-path descriptors, not individual AR parameters or a
complete flow/gust process decomposition. Moving a frozen whole factor to short
spans is an explicitly new observation use; the within-comment CV baseline
therefore includes its four new slice means. Neither branch replaces the released
GRAIN dynamic measurement system. This is a precisely bounded interface test.

## 3. Dynamic GRAIN coordinates can actually affect the model

For each author, difference the existing three period-level GRAIN scores and
the same frozen query function f(P,Q). Resample or permute whole authors, never
pretend the two transitions are independent participants.

{table(target[["adjustment", "authors", "r", "lower", "upper", "p", "q"]])}

The whole:F02/T-F link survives the primary 108-cell BH family at the reported
Monte Carlo resolution. Context adjustment includes text length/format,
community dispersion/change, span and median temporal gap. Intermediate
function-part tests are secondary. Primary permutation count was increased from
499 to 4,999 for every primary cell because .002 was too coarse a p-value floor;
authors and tested directions were not changed. This is exploratory localization,
not a preregistered isolated test.

Then fit a source-only displacement direction from changes in whole:F02 to
changes in P/Q, without questionnaire labels. Fix alpha=100 and step=0.25 before
the new native run. Patch only query-region states and the disjoint final state;
keep the original text and model weights fixed. Actual native results:

| Quantity | Value |
| --- | ---: |
| Authors / new calls | 12 / 60 |
| Signed half-plus-minus effect on conditional p(F) | {intervention["mean_signed_real"]:.6f} |
| Authors moving in the same negative direction | {intervention["real_negative"]}/12 |
| Mean absolute real-direction effect | {intervention["mean_abs_real"]:.6f} |
| Mean absolute norm-matched shuffled-direction effect | {intervention["mean_abs_null"]:.6f} |
| Real/null absolute-effect ratio | {intervention["abs_ratio"]:.3f} |
| Paired absolute-effect excess | {intervention["paired_abs_gain"]:.6f} |
| 95% author-bootstrap interval for excess | [{intervention["gain_ci"][0]:.6f}, {intervention["gain_ci"][1]:.6f}] |
| Maximum cached-state replay discrepancy | {intervention["max_state_replay_error"]:.1e} |

The alphabet is T=0, F=1. The negative half-contrast means the full +step versus
-step contrast is about -1.28 percentage points in conditional p(F), not a
personality-accuracy improvement. One shuffled direction is a matched negative
control, not a full random-direction significance distribution. This establishes
a usable model-sensitive displacement; it does not assert that this vector edit
is a natural human personality intervention or a unique attention circuit.

## 4. Model-readable and real-criterion-linked information partly separate

**Later same-day logic update:** the two correlations below use different
statistical levels and cannot alone establish separation. The subsequent
[matched-level review](SYSTEM_LOGIC_REVIEW_V1.md) compares the same author means:
F02 dispersion/real T-F r=.375 versus dispersion/query-function r=.073.
That is the appropriate exploratory contrast. Original values and test families
below are preserved, not retrospectively replaced.

Within-comment whole:F02 dispersion has real T/F r={case["pearson_r"]:.3f},
95% CI [{case["r_lower"]:.3f}, {case["r_upper"]:.3f}], n=63. Its change-to-query
co-change is only r={motion["r"]:.3f}. By contrast, the period-level F02 change
has the intervention-backed model relation above. These are two statistics of
the same coordinate, not interchangeable measures.

Leading exploratory within-comment criterion links:

{table(secondary)}

None survives the full 162-cell criterion family, so retain them as localized
candidates rather than naming factors after them. In particular, the r=.375
dispersion link is not yet a validated T/F subscale. Across the twelve period
coordinates, ranks of path-link strength versus real-criterion strength vary by
query (rho range -0.343 to +0.441). There is no general monotonic relationship.
The within-comment scan likewise yields no BH-supported all-query motion link.

The same F04 drift has different meanings at different clocks: sampled-comment
drift/extraversion r=-.162, within-comment drift/extraversion r=+.184. Their signs
cannot be pooled as one universal "motion trait". Most dynamic descriptors have
low cross-period correlations, with median values near zero; that is a property
of these sampled long-span materials, not a test of same-session repeatability.

## 5. Prediction comparisons answer the adjustment question, not the mechanism

All arms use identical author folds within branch. Ridge alpha is selected by
three inner folds, with train-only scaling, inside five outer folds. The base
is GRAIN12 (sampled-comment) or GRAIN12 plus four slice means (within-comment).
Extra blocks are query contributions, observed context, and dynamic+dispersion.
The latter is compared with dispersion alone and one fixed point-order shuffle.
No human criterion is read at inference time.

{chr(10).join(tables)}

The pure query addition preserves a useful E route in the sampled-comment
branch (.113 -> .238) and an O route (-.008 -> .136). Adding all dynamics is not
a general improvement: E falls to .089 with all additions, versus .238 with
query alone. Within-comment order additions do not beat the order-shuffled
comparator with a positive paired-r interval on any of nine outcomes. Small
per-feature criterion associations do not guarantee multivariate prediction gain.

These are small mechanism cohorts, not the prior 8,678-user MBTI baseline.
MBTI minority counts are EI=12, SN=4, TF=26, JP=27. Pooled CV r/AUC can be
distorted by fold-specific intercepts; the saved metric files include centered
within-fold r, valid-fold AUC and training-mean MSE. For example, sampled-comment
SN baseline pooled r=-.234 coexists with within-fold r=-.028 and AUC=.487.
Do not reinterpret a negative pooled r as a reversed personality mechanism.
Two fixed label permutations per baseline/full arm are sanity checks, not a
calibrated significance test. No result is rescued by flipping predicted signs.

## 6. Exact predictive-gap accounting, and the missing behavior channel

For feature-block subset S, let M(S) be held-author MSE. Average each block's
marginal improvement over all six block-addition orders. The resulting three
Shapley allocations sum to M(empty)-M(all), with maximum accounting discrepancy
{extreme:.2e}. Negative terms indicate worse prediction; they are not deleted.

{table(mse)}

Units are squared criterion units; compare within a criterion, not across
Big5 and binary MBTI. This is allocation of held-out predictive gain, not
percentages of true personality, causal variance or label noise.

MEPS numeric channel inventory (no new model fitting or questionnaire tuning):

{table(meps)}

There are user/assistant utterances, participation counts, task answers and
already-authorized pre/post self-reports. Participation is observed conversational
behavior. It is not an independent everyday-action criterion. For example:

{table(free)}

These 46-person correlations are exploratory and none of 418 scanned
participation/criterion links survives BH. Assistant counts also reflect the AI,
and missing assistance records are not automatically coded as zero use when
opportunity has not been established. No raw survey, duplicate coding archive,
Study 1 join or analysis-master table was opened.

The reason the broader "saying versus doing" gap cannot be numerically isolated
is identification, not insufficient model complexity. With only observed text
T and questionnaire Y, a world with unobserved action B=Y and a world with
B independent of both produce the same observed (T,Y) distribution. They imply
different text-action relationships. More elaborate functions of (T,Y) do not
select between them. Big5/MBTI self-report must not be renamed real action.

## 7. Updated mathematical interface and next useful experiment

Keep four distinct objects for each factor: level, dispersion, ordered shape,
and model sensitivity. Let H denote the concatenated P/Q state and G the frozen
factor coordinates. Source fitting provides Delta H = D Delta G + R. For a
frozen affine reader f(H)=b+g^T H, the following identity is exact:

    Delta f = g^T D Delta G + g^T R.

For the actual nonlinear native reader F, the exact differentiable path identity
is Delta F = integral_0^1 grad F(H+t Delta H)^T Delta H dt. Our finite native
patch tests a local action along D, not the entire integral or a natural text edit.
Separately, D_time(G+c_author)=D_time(G), so dynamic matching alone cannot recover
reference-relative author offsets. Preserve the static anchor.

This round's decision is **a factor-specific, statistic-specific interface**, not
universal concatenation. The next sharply defined mechanism question is why F02
level/change is model-readable while its within-text dispersion is more linked
to the recorded T/F criterion in this sample. Compare level-aligned and
dispersion-aligned state directions on the same material, retaining the current
whole:F02 direction as a positive model control. Do not restart a broad accuracy
grid or discard GRAIN coordinates simply because a query does not read them.
An action-gap experiment requires separately declared same-author action outcomes
and opportunity/timing, possibly from independently coded existing MEPS answers;
new human recruitment is not automatically required, but outcome definition is.

## 8. Reproduction, literature and scope

Run from the independent GRAIN repository with the parent project's environment:

```bash
"../project persona/.venv/bin/python" scripts/run_grain_dynamic_path_channels.py
"../project persona/.venv/bin/python" scripts/run_grain_within_comment_channels.py
"../project persona/.venv/bin/python" scripts/run_grain_dynamic_direction_intervention.py
"../project persona/.venv/bin/python" scripts/report_grain_dynamic_channels.py
"../project persona/.venv/bin/python" -m pytest tests/test_grain_dynamic_channels.py tests/test_grain_factor_path_routes.py tests/test_grain_functional_profiles.py tests/test_grain_full_state_edit.py -q
```

Default outputs are immutable on fresh preparation: use new output locations for
CPU reruns, or the explicit analysis/resume options to recompute analyses of
existing caches. The one native script intentionally refuses an existing output
directory; preserve/relocate a prior private packet before an intentional rerun.
Do not repeat a completed native run merely to generate this report.

Public aggregate report: this Markdown source and its offline HTML rendering.
Private numerical artifacts: `results/grain_vector_mechanisms/dynamic_path_channels_v1/`
with `within_comment/` and `native_direction/` children. Individual predictions,
raw material references and model arrays remain gitignored. Exact result hashes
and current code snapshots are in `delivery_manifest.json` and `code_snapshot.zip`.
Tables are used instead of charts because this deliverable primarily requires
exact nine-outcome lookup and a distinction between incompatible units/clocks.

The method choices are aligned with, rather than inferred as established by:

- [Fleeson (2001)](https://pubmed.ncbi.nlm.nih.gov/11414368/): trait-related
  behavior distributions can have distinct location and variability properties.
- [Campbell and Fiske (1959)](https://doi.org/10.1037/h0046016): different methods
  and different constructs must be distinguished when interpreting covariance.
- [Jain and Wallace (2019)](https://aclanthology.org/N19-1357/): a displayed
  attention pattern is not itself evidence of explanatory importance; here the
  native follow-up tests an actual state intervention instead.

This completes the five bounded computational questions. It does not assert
that all latent behavioral mechanisms are identified, or that every historical
GRAIN dynamic-factor family was tested. Released scoring, B0/B4, existing
lockboxes and parked long-running queues are unchanged.
"""
    destination = (
        ROOT / "reports/grain_vector_mechanisms/DYNAMIC_PATH_CHANNEL_REPORT.md"
    )
    destination.write_text(report, encoding="utf-8")
    body = MarkdownIt("commonmark").enable("table").render(report)
    html = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>GRAIN dynamic paths and channels</title><style>
body{margin:0;background:#f3f1eb;color:#182b2d;font:17px/1.6 Georgia,serif}main{max-width:1120px;margin:auto;padding:44px 30px 90px}h1{font-size:42px;line-height:1.15;max-width:850px}h2{margin-top:56px;padding-top:18px;border-top:2px solid #406766}h3{margin-top:30px}a{color:#075c74}table{border-collapse:collapse;display:block;overflow:auto;width:100%;font:13px/1.45 ui-monospace,monospace;background:white;margin:22px 0}th,td{padding:9px 11px;border:1px solid #d4dcd6;text-align:left;white-space:nowrap}th{background:#dbe6df}pre{background:#e2e7e1;padding:18px;overflow:auto;font-size:13px}code{font-size:.87em}p,li{max-width:1000px}@media(max-width:700px){main{padding:22px 16px}h1{font-size:30px}body{font-size:16px}}@media print{body{background:white}main{padding:0}h2{break-after:avoid}table{font-size:10px}}</style><main>"""
        + body
        + "</main></html>"
    )
    destination.with_suffix(".html").write_text(html, encoding="utf-8")
    code_files = [
        ROOT / p
        for p in [
            "docs/GRAIN_DYNAMIC_PATH_CHANNEL_PLAN_V1.md",
            "grain_research/vector_mechanisms/dynamic_channels.py",
            "scripts/run_grain_dynamic_path_channels.py",
            "scripts/run_grain_within_comment_channels.py",
            "scripts/run_grain_dynamic_direction_intervention.py",
            "scripts/report_grain_dynamic_channels.py",
            "tests/test_grain_dynamic_channels.py",
        ]
    ]
    evidence = [
        p
        for folder in (root, within, native)
        for p in folder.iterdir()
        if p.is_file()
        and not p.name.startswith("._")
        and p.suffix in (".csv", ".json")
        and p.name != "delivery_manifest.json"
    ]
    with zipfile.ZipFile(root / "code_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in code_files:
            z.write(p, str(p.relative_to(ROOT)))
    write(
        root / "delivery_manifest.json",
        {
            "status": "COMPLETE_FIVE_BOUNDED_QUESTIONS",
            "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code_files},
            "evidence_sha256": {str(p.relative_to(root)): sha(p) for p in evidence},
            "new_native_calls": 60,
            "primary_permutations": 4999,
            "bootstrap_draws": 499,
            "native_gain_bootstrap_draws": 4999,
            "reports_sha256": {
                str(p.relative_to(ROOT)): sha(p)
                for p in [destination, destination.with_suffix(".html")]
            },
            "report_mode": "offline_html; markdown is the versioned research source",
            "no_raw_private_text_in_report": True,
        },
    )
    print(destination)


if __name__ == "__main__":
    render()
