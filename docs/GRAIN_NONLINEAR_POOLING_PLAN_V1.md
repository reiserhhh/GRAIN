# NEXT69: content-dependent pooled interaction, same coefficient budget

2026-09-11. PLANNED, not executed. NEXT68 learned most of the small response
available to its affine-position field. Change the input dependence, not alpha.

## Fixed contrast

Keep the NEXT68 local1169 prefix, source U8/centering/scales, last-distance
geometry, four source views and four unit-coefficient loss terms. Replace only
the16-dimensional linear/position pooled summary with16 nonlinear token features.
For k=0..15, select original frozen landmark index4*k from the64 original Gaussian
landmarks. Require the64-landmark contract or stop before fitting. Use existing
bandwidth, descriptor (z, the original three positional features), and Gaussian
formula. Do not select centers/bandwidth using gradients, outcomes or new8.

    phi_k(z_t,p_t) = exp(-||[z_t,p_t]-landmark_(4k)||^2/(2*bandwidth^2))
    v = mean_t phi_t - mean_original_source(mean_t phi_t)
    F = F_local + .5*v.T*M_q*v

M remains symmetric/unconstrained,136 coefficients. Total1305 matches NEXT68
linear quadratic, not a larger feature-budget search. No author whitening,
extra variance normalization, learned centers, selection/temperature parameter,
alpha or rank grid. Fixed alpha=.01 remains; different feature coordinates also
change the regularized function family, so do not call this a penalty-invariant
comparison. Current H16 still requires the original encoder.

## Execution

1. New module/runner and scalar, gradient, finite cross-token tests; freeze code
   before outcomes. Keep primary NEXT68 implementation and artifacts unchanged.
2. Exactly two fits, one per query (N/confirmation seeking), maximum180CPU seconds.
   Same source4 x four views, same score/density/last-delta/B-delta targets,
   unchanged target energy units. No human labels, new native/API calls or MEPS.
3. Reuse NEXT68 matched-local and linear-quadratic predictions without refit.
   Freeze48 current input-only prediction packets before target joining.
   Application cannot open any native current gradients/scores or teacher caches.
4. Report source B coordinate response, full hidden field and all four GRAIN
   actions; report new8 original wholeL and full/allocated actions separately.
   Verify one scalar gives all outputs and cached replay is identical.

## Decision and stop

Primary development question: does content-dependent J_phi improve source B
cross-response SSE over .973321/.974318 and actually depart from the old
constant/position response shape? Report both queries and all directions.
The directional shape test is an account, not another native experiment.
New8 wholeL/actions determine whether any predictive substitution is useful;
source acquisition does not establish generalization or questionnaire accuracy.

After this single contrast, stop the pooled-quadratic family. If meaningful
content-dependent response is learned, retain it as an explicit component; if
not, use the observed bottleneck to choose normalized evidence selection or a
different aggregation mechanism. Do not automatically try16 more center sets,
penalties or wider features. Do not rerun old long queues to fill tables.

Expected implementation areas: `grain_research/vector_mechanisms/`, `scripts/`,
`tests/`; private outputs `results/grain_vector_mechanisms/logic_nonlinear_pooling_v1/`;
public aggregate report/theory under existing GRAIN namespaces. Preserve release,
scorer, lockbox, data permissions and unrelated edits. Overall goal ACTIVE.
