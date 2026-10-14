# Two-Context Local Construction Reference

2026-09-13. C01/C17 tranche under the all-plan goal. Status:
**REFERENCE_CONSTRUCTION_COMPLETE; Study B remains partially delivered.**

## 1. What this adds

The same declared construction recipe now produces four frozen local scorers:
common factor analysis and matched PCA, separately in PANDORA and Essays.
Each has a source reference, individual scores, empirical percentiles, conditional
material intervals and exact word-to-score contributions. New authors can be
scored without model fitting, a neural model or the other authors in their batch.

One Essays coordinate is associated with the independent questionnaire label for
Openness: r=-.267400, Fisher 95% CI [-.378042,-.149226], BH q=.001214 over all
100 evaluated connections. Its post-hoc partial correlation controlling log
text length remains -.259342 (q=.002277). Thus this association is not reduced
to the measured length difference alone.

The corresponding PCA score has r=.990490 with that common-factor score. Their
two significant questionnaire links describe almost the same lexical variation,
not two independent discoveries. This is a useful standard reference against
which to evaluate what GRAIN's conditional/path construction adds.

This is not a new GRAIN algorithm, a new five-trait scale or full Study B
acceptance. It implements the predecessor structural-reference and scoring
requirements that the earlier prediction tables did not deliver together.

## 2. Theory and provenance

The predecessor is the open-vocabulary author-word approach in
[Kulkarni et al. (2018)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201703):
derive shared lexical variation across authors, distinguish common factors from
total-variance components, and examine independent personality associations.
This work is already in the project's predecessor register, not a newly claimed
discovery. Our data, vocabulary budget, rotation and scoring implementation are
local adaptations, not an exact replication of its Facebook experiment.

Native `psych::fa` MINRES/promax is used for the common-factor reference;
PCA with varimax is the matched total-variance reference. The common-factor
model is the standard R = Lambda Phi Lambda' + Psi, with regression factor-score
weights. Five dimensions and 200 terms were fixed as a bounded comparison budget,
not inferred as the true number of psychological traits. Source fitting and
rotation occur without criterion labels; none of the directions is renamed or
flipped after evaluation.

The frozen object cards and run rules are in
[the construction plan](../../docs/GRAIN_LOCAL_CONSTRUCTION_REFERENCE_PLAN_V1.md).
The all-plan ledger preserves the broader V3 B1-B7 and V4 Study B requirements.

## 3. Two different local objects

| Context | Source authors | Held authors | Material / criterion |
|---|---:|---:|---|
| PANDORA | 504 | 120 | 36 original comments per author; Big5-only subset; continuous 0-1 questionnaire labels |
| Essays | 2,213 | 254 | One prompted essay per author; original IDs; binary high/low questionnaire labels |

PANDORA reuses the exact source-fitted fold-0 reference objects, checking source
indices and original comment concatenation. Essays uses the existing fold-9
evaluation split and fits two new source models. Total: 3,091 unique within-domain
author rows, with 374 held authors; four methods-by-context outputs, not four
independent datasets. These datasets have prior exploratory exposure. This is
developmental reconstruction, not a newly blind lockbox result.

The source vocabulary is selected locally (English stopword exclusion,
min_df=.05, max_df=.95, at most 200 terms). The denominator includes **all**
alphabetic tokens of at least two characters, including unselected terms.
Comments, quotes and authored statements are not converted into a factuality
measure by this lexical reference. Query/event ownership work is a separate arm.
The whole document is the scoring unit; resampling units do not introduce
another learned slice model. There is no new text upload or MEPS/bridge fitting.

Identically numbered coordinates in the two contexts do not denote the same
construct. PANDORA source z units and Essays source z units are local references;
their scores must not be pooled as one cross-population norm.

## 4. Exact scoring and uncertainty

For author document x, let c_j(x) be the count of source-selected word j and
L(x) the total eligible token count. Set v_j(x)=c_j(x)/L(x).
With source feature mean m, SD s, fixed score weights W and offset o,

\[
q(x)=((v(x)-m)\oslash s)W+o,\qquad
z(x)=(q(x)-\bar q_0)\oslash s_{q,0}.
\]

All reference terms are estimated on source authors only. The implementation
also exports the algebraically identical affine map z(x)=v(x)B+b. Its numerical
explanation is

\[
z(x)=b+\sum_{t\in\text{dictionary hits}(x)} B_{\operatorname{word}(t),:}/L(x).
\]

Every hit retains its original character address, including case-expansion
handling. Unselected words still affect the denominator. This is an exact
decomposition at the observed text, not a causal claim about editing a word.
Empirical percentiles use source midranks, not a normal-distribution assumption.
No eligible tokens returns unscored rather than an average-person score.

Conditional material uncertainty uses 200 seeded resamples: whole comments for
PANDORA; consecutive pairs of approximate sentence spans for Essays. It measures
sensitivity to reweighting observed material under an exchangeable-block
approximation and a fixed scorer. It is not a latent-trait confidence interval,
cross-day reliability or uncertainty over vocabulary/rotation/model estimation.
Essays blocks remain within one occasion and can be dependent.

| Context / method | Median interval width, local z units | Median blocks | Missing factor intervals |
|---|---:|---:|---:|
| PANDORA common factor | 1.298181 | 36 | 0 |
| PANDORA PCA | 1.370979 | 36 | 0 |
| Essays common factor | 1.581745 | 22.5 | 25 |
| Essays PCA | 1.627951 | 22.5 | 25 |

Each Essays method has five authors with fewer than two blocks: 25 missing
author-factor rows, not 25 authors. These interval widths make the finite
material limitation visible rather than silently giving every point score the
same precision. They do not by themselves establish a minimum adequate budget.

## 5. All-criterion evaluation

Four source-frozen maps x five anonymous coordinates x five labels produce
100 associations. The full table includes Pearson, approximate Fisher intervals,
Spearman, two-sided p and BH adjustment over all 100. Essays additionally reports
AUC using the unchanged factor direction. Selected examples below are labelled
as the strongest connections, not a post-hoc choice of a new scoring rule.

| Context / method / coordinate | Criterion | r | Fisher 95% CI | BH q |
|---|---|---:|---|---:|
| Essays common factor F01 | Openness | -.267400 | [-.378042,-.149226] | .001214 |
| Essays PCA F02 | Openness | -.261544 | [-.372633,-.143065] | .001214 |
| PANDORA common factor F04 | Extraversion | -.253016 | [-.413505,-.077279] | .126146 |
| PANDORA PCA F02 | Extraversion | -.264193 | [-.423385,-.089174] | .118264 |

The PANDORA links are exploratory positive evidence of association but do not
pass this 100-connection adjustment. The negative signs are fixed factor
orientation, not evidence of a harmful model. Essays fixed-direction AUCs remain
.333708 and .334770; we did not reverse them into a retrospectively optimized
classifier. This experiment is anonymous structure-to-criterion association,
not supervised five-trait prediction. It cannot be compared as an improvement
over the earlier common-factor+Ridge mean prediction r=.029584.

The strongest Essays common-factor score correlates -.339215 with log1p length;
PCA F02 correlates -.351921. A declared **post-hoc descriptive** follow-up computes
all 100 partial associations, without changing any score or norm. Openness
partial r values are -.259342 and -.253402, both BH q=.002277. This rules out a
pure log-length explanation for these observed associations, not every possible
content or sampling explanation. The fixed 100-test families do not adjust for
all historical exploration throughout the project.

## 6. Reproduction checks and costs

- Four source transforms exported before opening criterion arrays; scorer hashes
  checked unchanged afterward. Source/held author sets checked disjoint.
- All 748 held method-author scores independently recalculated from native
  CountVectorizer rates and source matrices. Maximum cold replay difference
  3.34e-15; single/batch score difference exactly zero.
- Cold CLI replay explicitly forbids imports of torch, transformers, sklearn,
  pandas, joblib and scipy. Only the frozen JSON and NumPy are needed for scoring.
- All 3,740 material interval rows checked; maximum finite bound difference
  5.33e-15. All 100 raw correlations and BH values independently recomputed.
- Four preselected example explanations reconstruct their full scores. These
  are arithmetic explanations, not independent psychological coding.
- Both common-factor solutions have positive uniqueness, no Heywood cases and
  no emitted fit warnings. Minimum uniqueness: PANDORA .206501, Essays .680598.
  Off-diagonal residual RMS: .044767 and .027322. PCA rotations converged.
- 30 selected tests passed, including 13 local-reference cases; Ruff passes.
  A cold-audit wrapper initially retained an extra CLI positional argument and
  stopped before scoring. It was fixed; complete fresh cold invocations now pass.
- Construction took 8.124 wall seconds: two new source fits, two reused fits,
  zero neural calls. Audit replay does not train new models. No long queue.

## 7. What is closed, and the next useful comparison

| Study B requirement | This tranche | Still required for complete Study B |
|---|---|---|
| Object/condition card | Two explicit lexical reference objects | Target-defined GRAIN/semantic construction objects |
| Frozen individual scoring/reference | Four JSON scorers and source empirical norms | Same-material remaining comparison arms |
| Evidence/material explanation | Word addresses, full additive arithmetic, conditional resampling | Source-event/behavior meaning and independent interpretation |
| Held-author criterion map | Full 100 connections; PANDORA and Essays label types separated | MBTI four-axis map and target-directed arms under their own splits |
| Cross-context production | Same recipe, local coefficients/norms | Construct links where justified, not automatic numeric equivalence |
| Independent reconstruction | Cold CLI and independent arithmetic within this agent's work | Another executor from the construction manual (C26) |

Next use the **same material and held authors** to assemble existing strong
text/embedding, semantic whole/chunk, native/frozen, non-address combination and
GRAIN condition/choice/path arms. Reuse legitimate exact caches first; document
incompatible materials rather than comparing their headline r values. Deliver
an externally anchored target and an anonymous conditional/path target as
specified in V4. The decision is what each construction adds to measurement,
not whether more factors/rotations can optimize this result.

## 8. Files and execution

Code: `grain_research/local_measure.py`,
`scripts/run_grain_local_construction.py`, `scripts/score_grain_local_measure.py`,
`scripts/audit_grain_local_construction.py`, `tests/test_grain_local_measure.py`.

Private artifacts remain ignored under
`results/grain_vector_mechanisms/all_plan_closure_v1/local_construction_r1/`:
`contract.json`, `construction_freeze.json`, four `private_*_scorer.json`,
author manifest, held scores/percentiles, patterns, explanations,
`private_material_intervals.csv`, `criterion_associations.csv`,
`length_associations.csv`, `posthoc_length_adjusted_associations.csv`,
`same_domain_score_relationships.csv`, `summary.json`, `audit.json`, and
`audit_replay/`. Do not publish source IDs, raw explanations or private texts.

From the independent GRAIN repo:

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/run_grain_local_construction.py --output results/grain_vector_mechanisms/all_plan_closure_v1/local_construction_next
"$PY" scripts/audit_grain_local_construction.py --output results/grain_vector_mechanisms/all_plan_closure_v1/local_construction_next
"$PY" -m pytest -q tests/test_grain_local_measure.py tests/test_grain_common_factors.py tests/test_grain_predecessors.py tests/test_grain_native_reference.py
```

Reconstruction requires the recorded local source material, prior PANDORA fold-0
reference artifacts, and the existing R psych/GPArotation fit environment. An
unseen local JSONL input uses one `{"id":"local-id","text":"..."}` per line:

```bash
"$PY" scripts/score_grain_local_measure.py --bundle results/grain_vector_mechanisms/all_plan_closure_v1/local_construction_r1/private_essays_common_factor_scorer.json --input /path/to/authors.jsonl --output /path/to/new_scores.jsonl --explain
```

The output path must be new and its parent must exist. Only score material
appropriate to the declared local object; this English lexical reference is not
the project's multilingual vector pipeline. The overall all-plan goal remains
ACTIVE, with no automatic commit, push, lockbox opening or permissions change.
