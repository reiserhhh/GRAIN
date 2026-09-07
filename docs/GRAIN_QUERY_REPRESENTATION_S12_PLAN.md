# S12: source query-conditioned representation localization

2026-09-08. Bounded discovery experiment following S9-S11, not a new benchmark.

## Decision

Can the useful N configuration contrast C=S_new-M be decoded from a
query-conditioned or joint key-value representation when E5 means fail?
Use exactly the213 S9 N source authors, all three cue-mask views. Retain all
authors regardless of scores. T/F's full-source successful student remains frozen;
do not refit on its64-author diagnostic subset.

One native forward per view captures three1536-dimensional inputs:
- body_mean: equal mean of12 within-comment body-token means after block16.
  Causal masking keeps these prefix states independent of the subsequent question,
  though they still contain the existing system prompt and preceding comments.
- query_state: final prompt-position state after block16, already contextualized.
- kv_pairing: concatenate all12 layer17 within-comment contributions
  sum_G sum_(i in G) (p_i-P_G/|G|)v_i, using actual rotated K and scaled q.
  This is an attention-output component, not all joint information in the model.

Average the three vectors per author. Save all view vectors privately for replay.
Do not save the large position-resolved KV caches. Compare same observer origins
with S9 (max absolute difference2e-5); do not silently switch numerical conditions.

## Finite fit and outcome connection

Primary teacher target C; controls S_new and M. PCA64/Ridge and fixed PLS2/Ridge,
alpha1/10/100 selected with inner3 teacher-MSE. Same author-level outer5 as S4/S11;
reuse S11 inner salts for paired comparisons. All centering, scaling and projection
fit within train folds. Publish all18 arms, no post-outcome winner selection.
Persist equivalent affine maps and replay OOF values, then freeze all predictions
before joining only previously exposed source N outcomes.

Report target r, target MSE divided by train-mean OOF MSE, N r/FisherCI,
Cov(Y,prediction), paired bootstrap delta versus body_mean (same method/target),
S11 E5 arm, S9 teacher and frozen S4 student. Covariance accounting is additive,
not an orthogonal variance decomposition. Simple representation decodability does
not by itself establish unique causal mediation; S9/S10 provide intervention evidence.

The favorable event is recoverable C and retained positive human-label association.
Better complete-teacher fidelity alone does not answer the question. If ordinary
linear readout fails on all three, preserve the source effect and investigate
query-specific response geometry rather than repeat mean-vector hyperparameters.
If one succeeds, test its numerical transport and a finite lower-rank shortcut.

## Resource contract

639 native forwards,0 backwards; model Qwen2.5-1.5B-Instruct at the existing pinned
local checkpoint, MPS/FP32/eager/compact input, max3072. No generated reasoning.
Pilot24 authors/72 forwards for correctness and cost only, then full213 if worthwhile.
Cumulative45-minute collector cap; checkpoint per complete author; only one neural
collector. CPU fits capped10minutes. No API, new checkpoint, LoRA, bridge/MEPS
questionnaires, lockbox, or old queue. Code/protocol committed before collection.

## Mathematics and primary-source application

For fixed representations Z, the squared-loss ideal readout is m(Z)=E[C|Z].
For nested information fields A subset B:
E[(C-E[C|A])^2]-E[(C-E[C|B])^2]
=E[(E[C|B]-E[C|A])^2].
Our three vector summaries are generally NOT nested, so their empirical scores
test recoverability in declared finite learner classes, not that theorem's order.
A deterministic query representation creates no new input information; it can
make existing nonlinear relations more accessible to a small head.

Fukumizu, Bach & Jordan (2004), [JMLR5:73-99](https://jmlr.org/papers/v5/fukumizu04a.html),
motivates response-preserving subspaces rather than maximum-variance compression.
S12 uses existing PCA/PLS baselines, not their RKHS conditional-independence estimator.
Alain & Bengio (2016; revised2018), [linear probes](https://arxiv.org/abs/1610.01644),
motivates independently trained readers of frozen intermediate representations;
this is regression to intervention responses, not their image-classification setup.
The S10 exponential-family KV covariance identity supplies the third representation,
with exact native head reconstruction retained as an implementation check.

Outputs: results/grain_vector_mechanisms/query_representation_s12_r1/
Report: reports/grain_vector_mechanisms/S12_QUERY_REPRESENTATION_REPORT.md.

## Resource checkpoint and next-question routing

24 authors/72 views complete in105.36seconds. S9 native replay error0;
maximum native head reconstruction error2.83e-6. Full213 collection is worthwhile
because it connects representations to the same defined useful source contrast;
estimated total16minutes, within45-minute cap. No interim outcomes inspected.

Interpret the paired comparisons, not only the largest number:
- Query-state recovery beyond body mean supports a query-conditioned shortcut.
  Next use a frozen prefix plus affine readout; verify exact prefix feature replay
  and retained source association before claiming reduced computation.
- Body mean recovering C as well as query state favors earlier-layer contextual
  encoding over a uniquely query-created route. It differs from the failed E5
  means because the underlying encoder and text context differ.
- KV-pairing recovery ties the already specified relational vector to the source
  contrast and motivates the Fisher-metric direction in the new theory note.
- No recovery in these finite linear readers retains S9's positive intervention
  result and points to a bounded joint/nonlinear readout comparison, not an
  infinite ordinary-mean hyperparameter search or a claim of impossible recovery.

Across all routes, a complete-teacher fit alone does not establish recovered C.
Do not choose representation or sign on source Y. Report exposed-source evidence
as discovery; leave new-domain and small-trait coverage for later mainline steps.
