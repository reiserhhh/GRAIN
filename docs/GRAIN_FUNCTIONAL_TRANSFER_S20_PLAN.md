# S20: transport a shared functional covector to other source authors

2026-09-08. S19's eight-author numerical pilot gives leave-author-out effect
reconstruction r=.9690 and median gradient cosine=.9578. Complete its fixed24
before preparing S20. Connect the mathematical result back to the source response,
rather than enlarging a gradient-only table.

## Frozen formula and population

From the24 S19 integrated gradients define g_ref=mean_u(g_bar_u). No centering,
normalization, learned intercept, calibration or outcome-based sign change.
For every source author's three cached S12 pairing vectors, compute

\[
z_{u,v}=g_{\rm ref}^{T}d_{u,v},\qquad z_u=\tfrac13\sum_v z_{u,v}.
\]

The primary transfer cohort is the189 authors outside S19; the discovery24 and
pooled213 are secondary. Freeze all scores and hashes before any new endpoint or
source-outcome analysis. The gradient authors used no human labels, but are still
excluded from the primary transport estimate.

The basis is the path-gradient identity used in
[Sundararajan, Taly & Yan2017](https://proceedings.mlr.press/v70/sundararajan17a.html)
and finite gradient geometry motivated by
[Constantine & Gleich2015](https://arxiv.org/abs/1408.0545).
For each author E_u=g_bar_u^T d_u, so shared-readout error is exactly
(g_ref-g_bar_u)^T d_u plus numerical integration error. Small angular variation
alone does not bound response error: displacement magnitude and its alignment
with the gradient difference also matter. Test actual response transport.

## Bounded experiment

1. Revalidate S19, S12 and S14 source hashes. Freeze g_ref and all213x3 cached
   scores. All inputs are already collected; no new embedding calls.
2. On the next24 authors in frozen source order (indices24:48), view0 only,
   replay native head/score then remove full pairing d at layer17.
   Compare E=F(h)-F(h-d) with the frozen z_view0. Two forwards per author,
   no backpropagation, no fit, at most48 forwards and five minutes after loading.
   Retain complete author packets on a resource stop; do not expand the panel.
3. After predictions freeze, join only the already exposed N source questionnaire.
   No bridge, MEPS questionnaire or lockbox. Report all signed correlations for
   z_mean and z_view0, S9 configuration contrast C and S12's statistical C reader,
   by primary189/discovery24/all213. Report Fisher95%CI, paired author bootstrap
   for z_mean versus statistical reader, and Cov(Y,z). Do not select on outcomes.
4. Report held-author E reconstruction r, MSE/E energy, sign agreement and bootstrap
   r interval. E is a pairing intervention; C is a body-collapse intervention.
   Never silently treat them as the same target.

## Decisions

If the shared direction transfers and connects to source N/C, use it as a fixed
functional reader in the next mechanism step. If it transfers internally but
not to C/N, keep that mathematical result and investigate the missing path from
body-collapse to readout, rather than optimize this response against human labels.
If transport itself fails, use the gradient/displacement error equation to choose
orientation versus context-gain modeling. No routine hyperparameter search,
LoRA, old queues, cloud text transmission or released-scorer change.

Files: grain_research/vector_mechanisms/functional_transfer.py,
scripts/run_grain_functional_transfer.py, tests/test_grain_functional_transfer.py.
Output: results/grain_vector_mechanisms/functional_transfer_s20_r1/.
Report: reports/grain_vector_mechanisms/S20_FUNCTIONAL_TRANSFER_REPORT.md.
