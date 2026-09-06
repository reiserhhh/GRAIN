# Domain follow-up: content reuse and conditional geometry

2026-09-06. **Exploratory follow-up, not a preregistered confirmation.**
The previous frozen geometry/function packet has been inspected: associations
are positive, especially in X. This design is fixed before inspecting token
overlap values. No new model forwards, source labels, psychometric head fitting,
or cohort selection. Use all167 cases and the frozen matched donors.

## Questions and estimands

1. Does correct context help predict original text outside exactly reused token
   phrases? Use native Qwen content token IDs, excluding chat template tokens.
   For n=1 and n=4, mark every current token belonging to a contiguous n-gram
   found anywhere in that context. No token decoding or language-specific word
   segmentation. These masks measure exact lexical reuse, not semantic identity.
2. Retain the frozen primary tail (drop the first scored token). For n=4 report
   actual-minus-matched mean log probability in four declared partitions:
   all tail; neither context covers; both contexts cover; actual-only coverage.
   A case contributes to a partition only with at least four tail tokens.
   Report eligible cases and coverage, never treat tokens as independent subjects.
   Fixed-case bootstrap1,000, seed731, conditional on material and donor choices.
3. Revisit each geometry/function association with a prespecified descriptive
   conditioning design: intercept, exact stratum fixed effects, log current
   token length, log actual/matched context-length ratio, actual-minus-matched
   unigram and four-token coverage. Regress each of delta cosine and delta logp
   onto the same design, then correlate residuals. Report rank, residual degrees
   of freedom and1,000 joint case-bootstrap intervals with refit nuisance OLS.
   Undefined draws stay undefined. This is not held-out prediction, mediation,
   proof of semantic reasoning, or a proposal to delete lexical/context signals.

One model and small, selectively sampled accounts/documents remain limitations.
X cells are collector metadata, not experimentally assigned topics. Shared donors
and selected cells constrain bootstrap interpretation; no population-wide or
cross-language-equivalence claim follows. No choosing the most favorable subset,
feature family or encoder on these post-hoc results.

## Outputs and stop

Save parent hashes, source snapshot, private per-case token masks and overlap
statistics, partition summaries, conditional association tables and public
interpretation. Synthetic tests must check coverage, selective case counts,
and nuisance-only versus independent-residual examples. Preserve all previous
packets unchanged. Stop on parent/token mismatch or numerical failure, not a
negative scientific result. This bounded no-neural follow-up closes the immediate
content-continuity explanation; it cannot close personality-specific validity.
