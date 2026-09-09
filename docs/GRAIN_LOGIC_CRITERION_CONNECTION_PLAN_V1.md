# GRAIN logic NEXT06/07: frozen observations and author criteria

2026-09-10, staged after `a4526f3`. This is a bounded exploratory connection,
not a new personality-prediction contest or a new blind evaluation.

## Decision

Does the newly extractable query configuration contain actual author-criterion
information, and does the lightweight program preserve that particular part?
Native-response fidelity alone did not answer this question. Keep N's local
response success and T/F's earlier successes even if criterion links differ.

## Fixed inputs and scope

- Existing permitted PANDORA source criteria from
  `all_factor_source_m29_v1/private_external_links.csv`, already used by M29
  and the earlier same-level logic pass. No MEPS, raw questionnaires or lockbox.
- Nine-query source configuration/GRAIN/remainder observations from
  `logic_configuration_relations_v1/operator_aligned_r3/` (213/63 authors).
  These are full-source descriptive references, not OOF targets.
- Two-query four-program author-OOF outputs from
  `logic_configuration_extraction_v1/`. Each source author was excluded from
  both the input readout fit and its fold's GRAIN mapping. No new model fitting.
- Primary unit: three-view author mean. Report this explicitly, not mixed with
  single12comment native deletion statistics. Same-author labels once per author.
- All nine teacher/query observations analyzed; only two existing extraction
  targets are analyzed. No criterion-selected new head, sign, subset or scale.

## Finite calculations

1. For all nine queries, correlate actual criteria with c, m_G and R separately.
   Keep the full model function/native criterion comparisons as context only.
2. Use signed covariance decomposition, with a common denominator sd(c)sd(Y):

       corr(c,Y) = Cov(m_G,Y)/[sd(c)sd(Y)] + Cov(R,Y)/[sd(c)sd(Y)].

   This is exact association accounting, not psychological mediation or variance
   explained. Opposing signs and contributions outside[0,1] are legitimate.
3. For each frozen OOF input program, compare criterion relations of c_hat,
   R_direct, R_derived and the GRAIN transport gap. No criterion model is fit.
4. Record the input-compression identity for each target and the same cohort:

       Cov(c,Y) = Cov(c_hat,Y) + Cov(c-c_hat,Y).

   Likewise for the fold-specific R. A readout can approximate a teacher yet
   discard its criterion-relevant part, or retain less teacher variation but
   more criterion association. It is this information allocation we inspect.
5. Distinguish full-source R from fold-reference OOF R; do not silently claim
   they are the same score. All joins and numerical c definitions must agree.

## Reporting and validation

Report Pearson r, Spearman, and unflipped AUC for binary criteria. Analytical p
and BH are descriptive screening; bootstrap1,999 complete author pairs for
the nine c/R relationships and the two primary E5-mean OOF targets. These are
conditional on frozen outputs, not full refit uncertainty. No criterion-set
calibration, new hyperparameter search or accuracy-promoted threshold.

Save all joined private values under a new gitignored directory, publish only
aggregate tables and an answer-first report. Validate finite criteria, unique
joins, source/OOF agreement and signed covariance closure. Add synthetic tests
that include cancellation; no new neural/API calls.

Finish when the local criterion-information allocation is described and the
next mechanism question is updated. Do not try to force every query to work or
claim actual behavior measurement from a self-report criterion.
