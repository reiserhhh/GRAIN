# VM05 strong sparse source controls: execution extension V1

2026-09-06. This executes the already required source controls for the actual
strong MBTI reference, rather than inheriting conclusions from the weaker
lexical-projection head. It does not replace that completed experiment.

## Fixed scientific comparison

Four independent official axes, three frozen raw/cue/placebo arms. For each of
the12 task/arm pairs, exclude the99 historical reference members and fit retain,
shared-thread purge and five exact-count random deletion controls (731..735).
Total84 conditions,6294 evaluation authors per condition. The original6393-user
strong OOF is restricted to the same6294 evaluation users for reference-exclusion
comparisons; merely restricting scores is not called a refit. Full-source heads
also apply the frozen bridge-thread rule using only bridge provenance, not labels.

Reuse the unchanged word(1,2)/char(2,5)10000+10000 TF-IDF, min_df2, sublinear TF,
Logistic C=[.01,.1,1,10], inner3-fold author CV seed731, official outer5-fold,
fixed .5 classification threshold. Vocabulary fits only the current training
fold. Purge/count selection is repeated inside inner CV; no test-label selection.
No extra PCA, balancing, prompt, embedding, model training or hyperparameter search.

All12 original strong OOF packets must be complete and verified before starting.
Use explicit baseline directories and reject duplicates, incomplete packets or
wrong source/feature/arm metadata. A partial old failed SN packet is not a parent.
Original code/head identities are retained even where an audited memory-lifetime
fix changed implementation without changing the statistical estimator.

## Execution and resources

`run_grain_strong_source_controls.py` calls the unchanged external-links runner.
Single Mac CPU child,4 native threads inherited from the fitter, at most8GiB RSS,
at most1h per condition and8h per batch. Start a new condition only with3660s
remaining. File-lock each batch, preserve failures and use explicit identical
resume only for complete verified conditions. Do not edit any active scorer or
the Windows aa51b06 observer snapshot. Do not run beside another heavy Mac fit.

Existing full-source fits take roughly27-33min each;84 new conditions can require
tens of CPU hours, although purged/count-matched fits may be smaller. This is not
a one-hour claim. Run bounded batches and report actual remaining work. No smaller
sample, weaker head or fewer random seeds may be silently called the full suite.

## Evidence and decisions

Use source-control membership reconstruction independently at all24 selection
points per condition (five outer+bridge-source fit, each with three inner folds).
Verify all purge selections have zero retained shared-thread authors and every
random deletion exactly matches its purge counts, including inner folds.
Replay all saved fold heads with unchanged inputs. Show AUC, PR-AUC, BA, Brier,
positive rates, mean-fold score, fit sizes and resource records.

Compare purge-retain, purge-average of five random-control scores (not an
ensemble), reference-excluded refit versus same-user-restricted original OOF,
and the three arm differences. Paired author bootstrap1000 seed731 is conditional
on fitted OOF heads; fold/seed counts are not independent human replicates.
Source selection is observational: neither purge advantage nor a retained
association identifies a personality-specific causal mechanism.

Complete nulls remain a distinct requirement using the actual head and inner
selection. The previous15 completed nulls are not evidence for these strong MBTI
heads. Do not open bridge values, select a winner or certify all source controls
on a partially completed axis/arm suite.

## Resource-only null preparation

The existing null implementation caches all inner/outer feature matrices in RAM.
That is unsafe for the full strong sparse source cohort. The opt-in
`--disk-fold-cache` writes the exact label-free matrices to a private directory,
then reads only one outer fold at a time. It does not reduce authors, vocabulary,
grid candidates, inner selection, permutations or precision. Dense and sparse
observed/permuted synthetic replay must agree with the in-memory estimator to
1e-12 and select identical hyperparameters. Real observed OOF must still replay
its frozen source parent to1e-10 before the first permutation.

Cache identity binds input values/order, official folds, estimator and helper
code, shape/dtype, parent identity and checksums of all40 saved matrices. Verify
checksums on start/resume and every normal exit boundary; preserve partial
or changed caches rather than overwriting them. The disk bound is10GB. Existing
closed15 null packets are historical outputs and must not be resumed under the
new runner identity. The20-draw alert and199-draw final test remain unchanged.
