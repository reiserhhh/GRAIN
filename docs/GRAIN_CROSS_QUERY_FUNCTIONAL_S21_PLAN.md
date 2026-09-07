# S21: locate shared versus query-specific functional reading

2026-09-08. Planned next after S19/S20, not yet executed. The transported N
gradient has actual-effect r=.943 on24 new authors and source N r=.200 on the
primary189. Do not replace this result with another ordinary prediction search.

## Question that changes the mainline

Is shared response concentrated in a generic numeric-answer direction, with
psychological selection in query-conditioned KV pairing? Or does each trait
need its own final functional covector? Both answers develop the user's
shared-plus-specific path hypothesis; they imply different reusable algorithms.

For query q and author u retain two distinct mathematical objects:

E_uq=g_bar_uq^T d_uq, and R_u,q<-r=g_ref,r^T d_uq.

The first is the measured within-query path integral. The second crosses the
reference gradient from query r with the recipient query q's native evidence.
Record the full matrix, not only the diagonal or favorable query pairs.
The exact transport split is

E_uq-R_u,q<-r =
(g_bar_uq-g_ref,q)^T d_uq + (g_ref,q-g_ref,r)^T d_uq.

This separates author/context variation from a declared cross-query mismatch.
Small cosine differences need not imply small response differences.
Use integrated-gradient completeness and uncentered gradient second-moment SVD
as in S19, based on
[Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html) and
[Constantine & Gleich2015](https://arxiv.org/abs/1408.0545).
No new causal or psychological meaning is assigned to an SVD axis by its rank.

## Fixed design

- Reference: first12 S19 source authors, view0. Transfer: source indices24:36,
  view0. Same exact comment bodies for all queries, no labels.
- Four digit0-9 queries: unchanged native N; existing confirmation-seeking
  wording; a fixed numeric Thinking/Feeling preference wording; one non-trait
  literal-language-property query, such as average sentence length.
  Before any new call, inspect/reuse the earlier S1B numeric wording if available,
  document the exact control question, score units and code hashes. This
  preparation is mandatory, not permission to select among outcome variants.
- Every task uses the same system, body order, evidence template and native digit
  alphabet. Preserve native T/F binary positive-control results unchanged;
  numeric T/F is a separate mechanism condition, not its replacement.
- Reuse N gradients/packets from S19 and effects from S20. All other questions
  get native capture plus t=0,.5,1 head-gradient calls on reference12.
  Add .25/.75 once only when the S19 completeness tolerance fails.
- Transfer12 requires native/full-pairing-removal endpoints only. Freeze each
  query reference before any transfer effects are inspected.
- Maximum:216 new forwards/108 backwards at three nodes,288/180 with all
  adaptive additions; ten-minute collection cap after model load. Start with
  four reference authors as a numerical/resource checkpoint. One neural process.
  Preserve incomplete scope on budget stop; do not silently expand cohorts.

## Analysis and decision

Report each query's diagonal effect reconstruction; the4x4 cross-reader fidelity
matrix; gradient norms/cosines; shared-gradient spectrum; actual response
magnitude; and author variation. Separate raw-unit transport from directional
similarity. No held-transfer-gradient fitting, source Y join or prompt selection.

If the non-trait gradient transfers as well as trait gradients, keep the reusable
numeric suffix direction and investigate query-conditioned evidence d(q) as the
next location of psychological specificity. That is localization, not a failed
personality association. If own-query gradients consistently outperform crossed
ones, model a small query-conditioned operator family next. Mixed results call
for a shared-plus-query-specific decomposition, not forced universal rank1.

Avoid a larger factor search or calibration to exposed N labels. Only after
freezing the next operator should a later phase connect source outcomes again.
N's useful source relationship and native T/F control remain the anchors.

Proposed implementation: scripts/run_grain_cross_query_functional.py plus a
small query specification/helper and synthetic tests. Output:
results/grain_vector_mechanisms/cross_query_functional_s21_r1/.
Preparation must record exact prompts before running. No new LoRA, old queues,
Claude call, private text transmission, release changes or protected labels.
