# Conditional response objects with a fixed reference distribution

2026-09-13. ACTIVE all-plan C01/C12/C15/C17. Freeze before computing scores.
Implements V3 section4.2 and V4 section4 target-distribution requirements, not
another criterion-prediction sweep. Previous W/C/U work compared early/late
observed mixtures; this tranche constructs a fixed source-reference scoring tool.

## Inherited methods and local object

Gelman & Little (1997), [sections2.1-2.3](https://sites.stat.columbia.edu/gelman/research/published/poststrat3.pdf),
provide weighted cell-response aggregation and explain why sparse cells motivate
model-based pooling. Here cells are observed textual contexts, not demographic
survey strata. We implement direct aggregation, not their logistic mixed model.
No assertion that a person's missing context is missing-at-random.

Manski (2003), [introduction](https://www.cemmap.ac.uk/wp-content/legacy/forms/manskipaper.pdf),
distinguishes identification from finite-sample uncertainty. We use a declared
unit-ball completion class and derive its linear-image bounds directly below.
The available PDF is contents/introduction, not the complete book. These are
standard integration/convex-geometry tools adapted to GRAIN, not novel theorems.

Object: 30 fixed IPIP-keyed E5 affinity directions in original text units. Scores
are model semantic evidence functions, not true personality or verified writer
assertions. Keep their names as reference keys, not constructed psychological axes.

## Materials and reference construction

Reuse all original units and E5 vectors of `local_bilinear_r2`: PANDORA624
authors/36comments; Essays2467 authors/whole-document 128word chunks. No newly
encoded text, human labels, neural fit, MEPS/bridge, lockbox or external upload.
PANDORA source504/held120, officialfold0; Essays source2213/held254, fold9.

Conditions: exact recorded subreddit for PANDORA; a single `essay_protocol`
condition for Essays. Subreddit is not a verified fixed topic/stimulus. Essays
has no observed within-person condition variation; its one-cell result tests
the mathematical boundary, not cross-situation psychological equivalence.

Source-only reference: nu_c = mean_source_author(observed unit fraction in c).
The source cell profile m0_c is the equal-author mean of source authors' own
cell means, not dominated by prolific authors. The reference is frozen and exported.
Two predefined targets: all source-observed cells; and the top5 cells by number
of distinct source authors (lexical tie break), with nu renormalized on those cells.
Top5 is a deliberately narrower estimand, not a better universal score. In a
one-condition corpus both targets coincide. No selection from outcomes or scores.

## Scores and exact relationships

For unit h_t normalized to unit length, e_t=h_t D, where D is the frozen
1024x30 anchor matrix. For a person's observed condition means m_uc:

1. Preserve observed selection fractions pi_uc and unknown-reference conditions.
2. Actual mean = reference-context contribution + within-context departure +
   new-context actual contribution. Known cells contribute pi*m0 and pi*(m-m0);
   unseen source cells retain pi*m unchanged. This is accounting, not removal or
   a causal attribution to personality. Components need not be orthogonal.
3. For fixed target nu, retain a_u=sum_observed nu_c*m_uc and coverage kappa_u.
   A complete point score is available only at kappa=1. a/kappa is reported
   separately as an author-specific supported-target score, NOT the fixed target.
4. Let lambda=1-kappa. Treat observed finite-cell means as fixed. Unknown cell
   mean embeddings are allowed anywhere in the unit ball. The full target then
   lies in a_u + lambda*{vD: ||v||<=1}. For every contrast t:

   t'T lies in [t'a - lambda||Dt||, t'a + lambda||Dt||].

This is exact for the declared completion class, an outer envelope for actual
reachable encoder/text means. It is NOT a confidence interval for population
condition means or a psychological trait. A box of independent coordinate bounds
can be looser for composite targets because all coordinates share the same D.
Report coordinate30, domain-mean5 and domain-pair-difference10 contrast envelopes.
No covariance independence assumption and no psychological variance interpretation.

5. Hiding one already-observed target cell (highest target mass, metadata-only)
   gives a reversible data-reveal check. Restoring its FULL saved empirical cell
   mean reduces each envelope width by exactly 2*nu_c*||Dt||; restoring a single
   new comment is not claimed to reveal the true population response function.
   A completion budget ranks missing cells by nu_c per equal cell-collection cost.
   It targets structural coverage, not expected personality-information gain.

## Deliverables and stopping

Export NumPy/JSON frozen references, per-author score/selection/support objects,
all original unit addresses, three-component variance including cross terms,
two-domain coverage/completion-budget tables, joint-vs-box geometry and reveal
checks. Score held authors singly with source invariance. Synthetic tests include
one-cell reduction, hidden-cell reversal, unseen conditions, unknown scores,
contrasts with dependent directions, fixed-reference fits and vector bounds.

Independently recalculate real-cell means, all held envelopes and contribution
identities. No new hyperparameter/pooling search; no external criterion fishing.
If common-target coverage is low, deliver conditional profiles plus the explicit
observation requirements. Model-based extrapolation is a separate future object,
not a hidden zero or population-mean imputation in this one.

Initial output: `results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r1/`.
Public report: `reports/grain_measurement_foundation/CONDITIONAL_REFERENCE_REPORT.md`.
Full Study B, actual condition response mechanisms, ownership/psychological
interpretation and independent replication remain separately accountable.

Numerical maintenance addendum after r1: the batch path inherited float32 vector
normalization while the cold scorer used float64. A ~1e-10 cold-score discrepancy
triggered r2 with float64 normalization on BOTH paths. This changes no target,
source authors, conditions or selection rule. Preserve r1, record the difference
from prior float32 semantic scores, and deliver `conditional_reference_r2` after
cold scoring passes. Flatten the budget CSV columns for ordinary tool ingestion.
