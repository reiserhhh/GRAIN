# NEXT08b: the actual input quotient and an order-sensitive extension

2026-09-10, after b7163e2. Discovery question: does the deployed independent
comment-bag input identify the fixed query target on reordered real material?
Previous S6 changed block16 states, P1R used authored toy stories, VM06 changed
MEPS assistant tokens. None is a paired whole-comment reordering of the present
12-author material. Do not pool these incompatible interventions as one dataset.

## Frozen finite experiment

- Same12 source-disjoint full-view authors from complete_level_response_v1;
  all12 comments retained verbatim. T/F and N queries only, same model/function.
- Compare original and exact list reversal. The original excerpt renderer
  reassigns displayed ordinal numbers, so this is presentation order/placement,
  not isolated human temporal dynamics. Verify body multiset, full token multiset,
  prompt length, question tokens and baseline replay; never truncate. See the
  tokenizer-only amendment below for boundary-token changes.
- Reuse E5 per-comment vectors. Their mean, query-weighted pool and empirical
  distribution are unchanged under reordering. No new E5 calls.
- Acquire original+reverse for24 author/query units, plus original repeats for
  the first2 authors/2queries:52 forwards maximum,600s inference maximum.
  Fixed local Qwen observer, MPS FP32/eager, no sampling/generation/API.
- Retain P/Q, native score, fixed f and its mean/configuration/final parts.
  Every completed unit persisted; a resource stop preserves partial work.
  No questionnaire fields in acquisition; no source/held overlap in fitting.

## Mathematical witness, not another accuracy sweep

For reversal J, J²=I. Let t+=(t(T)+t(JT))/2 and t-=(t(T)-t(JT))/2.
For any invariant prediction p, mean over the pair obeys

    ((t(T)-p)²+(t(JT)-p)²)/2 = (t+-p)² + (t-)².

Nonzero t- is an exact error floor on this uniformly weighted two-element
orbit for ANY invariant reader, linear or nonlinear. It is not an error bound
on the unaltered source distribution or proof of human personality information.
Separate baseline mismatch, common response, author-varying response and the
changes of the three fixed-function components. No interpretation from a sum
identity alone; compare actual native changes as well.

## One predeclared development attempt

Preserve the preceding pooled E5 program as p_mean. Add exactly one ordinal
first-moment feature, v(T)=mean_j s_j h_j, s_j=linspace(-1,1,n).
Then v(JT)=-v(T). Train a no-intercept Ridge residual term a'v/s on SOURCE
teacher targets minus source-fitted p_mean, fixed alpha=.01*n_train_views.
The scalar s is the training RMS vector norm about zero, not a centered norm;
centering would break reversal oddness. No hyperparameter search.

The two-output c/R targets, author5fold seed20260910,276-source panel and
GRAIN per-fold reference are unchanged. Freeze source-OOF and held full/reverse
predictions before new neural endpoints or human labels. Report source teacher
fidelity and existing exploratory criterion links without tuning on either.
Held response comparison uses c and native; R reversal truth is not fabricated
because new GRAIN reversal scores are not part of this finite experiment.

Fit data includes only natural ordered source views, not paired reversal
targets. Thus this arm tests transport of an order-sensitive input, not the
benefit of new paired training. Failure of one ordinal moment does not erase
the existence of an order-sensitive target or exclude richer sequence operators.

## Papers and use

- Zaheer et al.(2017), [Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf):
  set-valued architectures impose permutation invariance. We use that structural
  property, not claim a fixed finite mean vector is a universally sufficient set
  statistic. E5 encodes word order within comments; only between-comment order
  is removed by current pooling.
- Murphy et al.(2019), [Janossy pooling](https://arxiv.org/abs/1811.01900):
  averaging order-sensitive functions produces an invariant function. Our
  two-element reversal average is exact for C2 only, not the full12! group.

Decision: if target moves beyond replay noise, specify the lost input distinction.
If the fixed ordinal extension follows the change, retain it as a useful local
interface. Otherwise distinguish positional information from a learned
conditional operation; do not escalate to a feature/alpha grid. Next check
source-conditioned versus contextual representations using already built tools.

Outputs: private packets/models/predictions in logic_input_order_v1; public
aggregate report, formulas, tests and source-only code. Local commit, no push;
release scorers, old lockboxes and unrelated worktree edits untouched.

## Tokenizer-only amendment, before any new neural endpoints

Initial preparation stopped because complete prompt token multisets are not
identical after natural rendering. All24 original prompts exactly replay;
all24 reversals keep total token count unchanged; token-count-vector L1 change
is0-4 (mean3). No new model calls or endpoint inspection occurred. Retain all
authors; require exact unchanged comment strings and question tokens, record
the prompt token multiset difference rather than pretend it is absent. This
tests full-comment presentation order including boundary retokenization, not
isolated positional encoding. The independent E5 bags remain EXACTLY identical
as multisets, so the stated invariant-input risk identity is unaffected.
The complete packet will use `logic_input_order_v1/paired_r2/`; the initial
empty preparation directory remains as a historical failed preflight.
