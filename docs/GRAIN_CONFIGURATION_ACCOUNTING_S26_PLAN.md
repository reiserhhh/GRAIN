# S26: configuration displacement contains the original pairing path

2026-09-08. Fixed while S25 collection runs, before S25 source outcome analysis.
No additional model calls. This is a prospective accounting question, not a
source-label-selected blend: how do the successful S20 pairing readout and the
larger S25 actual configuration-head contrast overlap?

## Exact retained-path identity

For one head, let p_i,V_i be the native final-query attention weights/values,
G the disjoint comment-body groups, and h0 the actual projected-body head.
The original S10/S12 pairing vector is

d_pair = sum_G sum_(i in G) (p_i - P_G/|G|) V_i,
P_G=sum_(i in G) p_i.

Then h1=sum_i p_i V_i and

Delta_h = h1-h0 = d_pair + r,
r = sum_G P_G mean_G(V) + sum_(i outside body) p_i V_i - h0.

This is an exact weighted within-group centering identity, not an assertion
that r is noise or statistically independent. Concatenate heads and apply the
same frozen S20 g_N, keeping all three source views:

z_config = z_pair + z_remaining.
Cov(Y,z_config) = Cov(Y,z_pair) + Cov(Y,z_remaining).
Var(z_config) = Var(z_pair)+Var(z_remaining)+2Cov(z_pair,z_remaining).

The inherited exponential-weight readout and within-group decomposition are
documented in theoryV4/S10-S17. Functional reading is grounded in the integrated
gradient identity of [Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html).
These elementary exact identities are used to locate a mechanism, not described
as a new theorem of personality. No orthogonality follows from the subtraction.

An identifiable candidate inside r is the order of token averaging and
RMS normalization: RMSNorm(mean H) is generally not mean RMSNorm(H). The
[RMSNorm equation, Zhang & Sennrich2019](https://papers.nips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf)
and the actual Qwen implementation make this concrete. r also contains changes
in attention mass/redistribution and value means; this cache experiment must
not claim that all r is normalization. Separate them later only if decision-useful.

## Execution and decision

Require complete S25 frozen scores. Read213x3 S25 actual head differences and
the matching S12 kv_pairing arrays through the same verified S20 source chain.
Check vector addition, S20/S25 score replays and unique author/view order.
Freeze all z_remaining before joining already exposed source N outcomes.
No protected labels, no new training or directions, no sign/gain choice.

Report primary189 outside the gradient reference, discovery24 and all213:
source r/CI for all three components, covariance/variance identities, signed
channel covariance, component spread and view-pair consistency. Preserve both
channels regardless of direction. Negative covariance would show cancellation;
positive covariance would show additional aligned signal. Neither makes r pure
noise or proves independent psychological components.

Also compare r versus d_pair in the native Euclidean norm and under the frozen
functional scalar. Label the former coordinate-dependent. In particular, large
vector size need not imply large human covariance. Do not select a source blend
or present the accounting identities themselves as a predictive improvement.

Budget: zero neural calls, two CPU threads,60seconds numerical analysis after
source verification. Proposed runner scripts/run_grain_configuration_accounting.py,
helper/tests configuration_accounting.py, private output
results/grain_vector_mechanisms/configuration_accounting_s26_r1/.

If the full configuration head score is weaker, source covariance accounting
decides whether a broad displacement mixes the useful pairing signal with an
opposing component. If stronger, it identifies an added source-linked component.
Either outcome changes the next representation/mechanism choice without more
prompt gradients or exhaustive defensive experiments.
