# HM277: paired source structure to held-author criterion reading

Prepared before computation, 2026-10-14 JST. No parameter/precision search.

## Unknown and competing predictions

HM273 found that the SOURCE F-J target field recurs across two banks for N/A.
That is not a test of the target criterion of an author who supplied no
observations to the bank making their score. Test that missing link now, for
ALL five historical binary Big5 criteria and BOTH unchanged vector spaces.

If the paired SOURCE field is useful at the author level, its cross-fitted
author reading should associate with the held author's own criterion, beyond
the within-bank whole-profile relabeling null. A repeatable group field may
instead have no such author-level linkage. Stable author readings alone can
also be author fingerprint rather than target meaning. These alternatives
lead to different next work; none is rescued by tuning width or score sign.

## Fixed data and operator

Use only HM273 `source_pair_bank.npz`: 386 historical SOURCE authors in banks
203/183; each has 22 overlapping next-cached pairs from two 12-event windows.
Bank0 scores bank1 and conversely. Never use own-bank sources or native-panel
labels. The old E5 embeddings, GRAIN15 projection, binary criteria and variance
are frozen and previously explored: this is conditional-on-frozen-geometry
exploratory cross-fitting, not a new untouched benchmark or paper replication.

Read all four existing laws F/J/I/R (observed pair, same-author product,
same-profile product, condition-matched right relink). Average pair readings
within each window and across all22 pairs; do NOT multiply overlapping pairs
as independent observations. Preserve all32 profile cells and all five target
expectations. Retain F-J, J-I, F-I and F-R with their signs.

For F/J/R, let a^L_uv be the source-author normalized kernel mass for query
pair v of author u. Then the target reading is

    s^L_uq = mean_v sum_j a^L_uvj * b_jq
    d_uq = s^F_uq - s^J_uq.

Each row uses only opposite-bank j. The weight operator is label-independent;
target meaning comes from the source criterion b, NOT geometry-only naming.
I is retained under its original within-profile definition, not substituted
by a product of normalized probabilities. No new mathematical theorem.

## Outputs and evaluation, fixed in advance

- Primary: author-level Pearson correlation of all22-pair F-J with its same
  target. Center x/y WITHIN evaluation bank only in the association statistic
  to avoid cross-fit reference offsets; this does NOT remove text context or
  alter scores. Retain pooled and each-bank raw correlations too.
- All F/J/I/R and four signed components: Pearson, rank correlation, score
  RMS/mean, plus Brier/MAE/RMSE for probability-valued law readings only.
- Window0/window1 results and within-author versus coarse-condition-nearest
  same-bank stranger readout correlations. Match using condition fractions
  and transition frequencies only; this is NOT complete topic/length matching.
- 199 whole32-profile permutations independently within the historical banks.
  Recompile F-J from the SAME frozen label-independent weights, and evaluate
  against the corresponding permuted held labels. Preserve joint label
  relationships. Report nominal two-sided p and max-absolute correction over
  2 spaces x5 targets. No false iid pair count.
- 999 bank-stratified author bootstraps for the primary correlations. Intervals
  are conditional on fixed cross-fitted scores; they do not include uncertainty
  from fitting geometry/source laws. No stopping based on obtained results.
- Cold author scorer accepts numeric pairs and a SOURCE bundle, no author
  criterion/LLM answer. A held-author exemplar must reproduce arithmetic means.

No required high correlation threshold and no sign reversal for promotion.
Positive associations are candidates; null/negative values change the route
just as materially. A split-window correlation is not temporal trait validity.
Historical binary criteria are not independent paired behavioral outcomes.

## Execution and boundary

One CPU run, 240s finite cap, 4-author blocks for memory only; no new model,
API, raw/private text, training, Git, Claude, automation, lockbox or old queue.
Freeze config/code before prepare, retain failed output without overwriting.
Focused synthetic tests before prepare. Verify selected pair readings with
explicit positive kernel sums and primary covariances by another form; no
extra precision branch. Original18/100/44 remains OPEN.

Prior art: HM273's standard joint/product kernel measures; Sejdinovic et al.
2013 (https://arxiv.org/abs/1306.2281) studies interactions beyond marginals.
Zhang et al.2023 proposition2.3 (https://arxiv.org/html/2303.10134v4) separates
functionals from full latent identification; no causal theorem applied here.
OpenAI/math pinned foundations166-232 informs declaring observation laws and
conditional independence, not a transported tree reconstruction theorem.
