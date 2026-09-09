# NEXT08c: input versus paired conditional support

2026-09-10. Frozen before the new source reversal endpoints. Continues the active
whole-system logic goal after `0bc707d`; does not change a release or lockbox.

## Question and four cells

The ordinal arm can distinguish presentation order but has not learned its
effect. Compare two fixed inputs and two training supports, without a grid:

- E5: concatenate comment mean and ordinal moment of the same12 vectors.
- Early P/Q: ordered question slots plus final slot after four frozen blocks.
- Natural: source original-order endpoints only.
- Paired: the same source authors/material plus their complete-list reversal.

Both outputs are learned together: fixed upper function configuration c and
full f. Native model output is evaluation only. No human criterion is a target.
No edited GRAIN R is invented; this experiment does not rescore its factors.

## Fixed panel and resource boundary

-32 MBTI-only source authors for T/F and32 Big5-only source authors for N,
  chosen by SHA256(`NEXT08c:` + query + `:` + author_id), without outcomes.
- Only source view0,12 complete comments, exact old E5 input binding.
- The old12 disjoint natural-material authors remain the exposed transport
  panel for both queries. Their native/f endpoints are already known, but no
  fitting, scaling or model selection uses them.
-64 source pairs +24 transport pairs =176 full forwards, plus4 original-order
  repeats on the first two transport authors per query:180 calls,900 seconds.
  Capture block3 P/Q and block16 P/Q in the same full forward. No generation,
  gradients, API, E5 recomputation or full-model training.
- Every completed pair is checkpointed. Pause at the declared resource boundary
  or any identity/replay error; do not silently drop a difficult author.

Comment strings/vector multisets are exact. Natural prompt rendering may change
boundary tokens, as already documented in NEXT08b. Record token length and bag
differences. Reject silent truncation and changed question-token identity.
Recheck all24 old transport endpoints and64 old source configuration values.

## Fixed estimation and evaluation

For each input, center each of its two blocks and scale by its RMS row norm
using original-order training authors only. Concatenate and divide by sqrt(2).
Those moments are identical between natural and paired cells within each fold.
Fit Ridge with intercept and alpha=.01*n_fit_rows. Thus the average-loss
penalty is constant when pairing doubles the rows. No degree/rank/alpha search.
Four author folds (seed20260910), all endpoints from a held author excluded.
Fit all-source models once and freeze old-panel predictions before evaluation.

Report all four cells, both queries and both outputs:
- original level r and RMSE; paired-even level r; reverse-original response r,
  RMS amplitude, sign agreement and MSE relative to predicting zero change;
- native response correlation separately from fixed-target fidelity;
- paired-minus-natural effects within each input and their interaction;
- whole-author bootstrap (1999) conditional on frozen predictions, not a claim
  of a fresh replication or uncertainty over every possible training sample.

After all predictions freeze, inspect existing source self-report links for
original levels and C2-even averages. Compare on the exact same32 authors, not
to older63-author headlines. Log raw teacher original/even/odd links too.
These are exploratory diagnostics of whether better imitation preserves author
information, not a new selected personality score or scale validation.

## Decision routing

- Paired support helps both inputs: conditional support was a useful missing
  ingredient; retain the simpler useful representation, no need to add depth.
- Only early P/Q helps: investigate contextual operations as the next input
  mechanism, not merely a scalar calibration problem.
- Neither helps: preserve this localized negative result; inspect the common
  presentation offset versus author-varying response before expanding data.
- Teacher gain with criterion decline: retain a model-mechanism finding but do
  not promote it as author measurement. The two objectives remain separate.
- A small selected maximum is not promotion. Report the whole factorial and
  effect magnitudes; the next step must explain a mechanism, not enlarge a grid.

## Mathematical connection

Pairing supplies observed function values at a new input transformation; it
does not assume a derivative with respect to words. It is motivated by the
value/response distinction in [Sobolev training](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html)
and [Jacobian matching](https://proceedings.mlr.press/v80/srinivas18a.html).
This is not a reproduction of their differentiable loss. Original-only and
paired observations have different support; the paired orbit separates even
level and odd response losses, as in the current input-quotient theory.

## Artifacts

New implementation only, no modification of the sealed prior collector:
`grain_research/vector_mechanisms/paired_support.py`,
`scripts/run_grain_paired_support.py`, `tests/test_grain_paired_support.py`.
Private: `results/grain_vector_mechanisms/logic_paired_support_v1/r1/`.
Public: `reports/grain_vector_mechanisms/PAIRED_SUPPORT_LOGIC_REPORT_V1.md`
and aggregate tables. No texts, author IDs, individual criteria or weights
are copied into Git. Local commits, no push.
