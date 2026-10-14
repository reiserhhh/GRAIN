# Same-Material Bilinear Construction

2026-09-13, C10/C14/C17 continuation, fixed before target-head fitting.

## Question and inherited mathematics

Does pairing GRAIN coordinates and semantic contrasts at the same text unit add
useful author information beyond their marginal distributions? Study B requires
a non-address combination control, not just comparing unrelated headline scores.

Use the standard local outer-product pooling pattern of
[Lin, RoyChowdhury and Maji (2015), section 2](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf).
The predecessor operates on image locations; this is an adaptation to text
units, not its experiment or a claim that bilinear pooling is new.

For unit t of author u, frozen GRAIN g_ut in R24 and keyed semantic contrast
s_ut in R30 give B_u=mean_t(g_ut s_ut'). A non-address reference is
M_u=mean(g_u) mean(s_u)'; B_u-M_u is within-panel cross-covariance (divisor T).
Independent uniform permutations of the semantic unit indices preserve all
marginals and have E[B_u^perm | g,s]=M_u. This equality also holds for T=1,
where co-location adds no information. It is a finite permutation identity,
not a psychological or causal theorem.

## Material and scoring objects

PANDORA: exact existing624 Big5-only authors,36comments,source504/held120,
officialfold0 held, fold1 source-validation, remaining folds source-training.
Essays: original2467 unique authors,source2213/held254,fold9 held,fold8
source-validation. Consecutive128-whitespace-token chunks preserve all original
characters and all text; these units are not occasions or syntax events.
Current data are development/reanalysis, not a newly opened lockbox.

Use the existing externally discovered frozen TF-IDF/SVD24 runtime; verify
PANDORA exclusion and Essays dataset separation. This is GRAIN's unit coordinate
interface, NOT the support-gated G1 geometry profile or all multiview GRAIN.
Keep the latter on its own32-comment protocol and124-supported-author results.

Semantic units: pinned E5-large-v2, query prefix, masked mean/L2,cap512. Reuse
verified PANDORA36comment vectors and300IPIP vectors. Encode Essays locally with
the same model/revision, no downloads, no finetuning. Log pre-truncation lengths;
do not claim vectors see every character when cap truncates. GRAIN receives each
complete unit. These are declared reader differences on the same available
material, not identical internal token observations.

Keyed contrasts use all300IPIP items: mean cosine to positive items minus mean
cosine to negative items for each of30facets. No threshold, source/factuality
filter or post-hoc rekeying. They are semantic contrasts, not validated personality
events. Mean E5, GRAIN mean/std, contrast means and B/M are author features.

## Fixed comparison family

1. Word1-3 plus char2-5 TF-IDF,10000features each, Ridge.
2. E5 unit mean, source PCA64, Ridge.
3. Frozen GRAIN unit mean/std48, Ridge.
4. Keyed semantic means30, Ridge.
5. Additive E5 PCA64 + GRAIN48 + semantic30.
6. Additive plus vec(M), non-address nonlinear control.
7. Additive plus vec(B), same-unit pairing.
8-10. Same width as7 but within-author shuffled semantic units, seeds924/925/926.

All new PCA/scaling/TF-IDF/head fitting occurs on source-training for selecting
Ridge alpha in1/10/100/1000 by mean validation Pearson across five traits. Refit
chosen alpha on source only; evaluate held once. No selection of slice size,
semantic keys, factor count, shuffle seeds or representation family on held data.
PANDORA outcomes are continuous0-1; Essays binary0/1. Ridge remains continuous,
unclipped; allfive traits get r/rho/MAE/RMSE and Essays AUC.

Report paired author-bootstrap1000 CI for same-unit minus additive, non-address
outer and each shuffle, and additive minus E5. Intervals are conditional on fitted
heads, not refit uncertainty or a multiplicity-adjusted acceptance decision.
No universal fixed-r success gate: identify the scope/direction of added value.
Source and held results remain separate. A positive result supports this
co-location construction; a negative one keeps the identity but rejects upgrading
this fixed implementation. Neither settles all GRAIN/personality mechanisms.

## Delivery and bounded execution

Material/encoder hashes and original unit addresses; complete feature matrices;
source tuning and frozen models; all author predictions; full metric/delta tables;
bilinear covariance identity and marginal-preserving tests; cold numeric head
replay; report and total-ledger update. Do not read MEPS/bridge or start native
LLM generation to fill an absent arm. Native-generation/frozen-readout, whole-vs-
chunk semantic and independent psychological interpretation remain Study B gaps.

Essays E5 encoding has a900second soft batch budget and resumable verified cache.
If paused, inspect its specific process/status and resume rather than restart.
No other long queues. Source fitting is CPU with at most two BLAS threads.
No commit/push, text upload, extra dependencies, lockbox or human recruitment.
