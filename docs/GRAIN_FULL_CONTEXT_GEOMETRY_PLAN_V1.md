# NEXT65: preserve full context geometry in the conditional scalar

2026-09-11. Planned, not executed. NEXT64 added real input interaction but did
not recover both native source changes or complete new-author prediction.
Its four-source full-rank whitening makes all source pairs equidistant and
drops89--91% of new8 last-state displacement energy from context modulation.
Test that specific input loss next, not another source-fidelity or alpha grid.

## One change, complete path comparison

Reuse the exact source4 original+NEXT63 shifted views, existing exposed new8,
two queries, old BODY U8/normalization/64 landmarks/Hermite features and alpha=.01.
Each author1/4, each view1/2; original score/density scales, not augmented scales.
No new text, neural/API call, humanY/MEPS use or fake source leave-one-out.

Replace ONLY the context kernel coordinate metric. For original full hidden
last states h_s and their source mean h_bar:

    sigma_h = median positive ||h_s-h_j|| over original source pairs
    kappa_m(h) = exp(-||h-h_owner(m)||^2/(2 sigma_h^2))
    gamma_m(h) = kappa_m(h)-kappa_m(h_bar)
    Lhat = L_additive + mean_t sum_m gamma_m(h) e_m^T phi_m(z_t,p_t).

No PCA, dimension-wise whitening or fitted context axes. One isotropic Gaussian
over the FULL frozen hidden last state, with source-only bandwidth. A global
translation/orthogonal rotation leaves it invariant. No claim of general
invariance to arbitrary anisotropic embedding changes. Compute four source
distances then repeat by landmark owner; do not allocate hidden-width identity
matrices or token-by-hidden-by-center tensors.

It remains1169 coefficients,593 additive prefix; setting e=0 gives the same
additive function. At source mean, added scalar/BODY-gradient vanish. The last
gradient uses -(h-h_s)*kappa_s/sigma_h^2 plus the original last linear term.
Never fit a separate derivative output or include native current gradients.

With source centers lying in affine span S, the full kernel factorizes exactly:
`kappa_s(h)=exp(-||P_Sperp(h-h_bar)||^2/(2 sigma_h^2))*kappa_s(P_S(h-h_bar))`,
where the second factor uses centered source coordinates in the unwhitened metric.
Thus this candidate restores original in-span geometry and a common off-span
radial attenuation, not arbitrary off-span orientation sensitivity. Report these
two factors in pure-input diagnostics. If angular off-span information is needed,
four source-centered radial kernels cannot identify it by pretending to be full
hidden-space coverage. This limitation follows from the formula before results.

## Comparators and resource bound

Two new conditional solves, one perquery; reuse NEXT64 augmented additive and
whitened-conditional programs, and NEXT62 frozen-original reference. Four models
displayed, only2newfits. In forming streamed normal equations, verify that the
additive top-left block/RHS matches the previous augmented additive system to
numerical tolerance; do not refit it or count repeated fits as independent evidence.

120sCPU across fits, one primary/replay application, evaluation and audit.
Estimate after firstquery with30s reserve. No smaller rescue candidate, reranking
of source donors, bandwidth/alpha/rank search or hidden application-only offsets.
Freeze primary implementation and prediction outputs before joining new8 targets.

## Report and decisions

Report source original/shifted score and density loss, finite action-change
recovery against native NEXT63 (all4directions), and new8 wholeL r/RMSE/bias/SD,
SSE/sourceconstant, fullBODY error, full/allocation GRAIN action r/RMS/signs/SSEzero,
factor cells and paired query contrast. Keep the prior reference results visible.
Source is in-fit and new8 is previously exposed development, not new human validity.

Compare with both whitened conditional and augmented additive. Fullpromotion
requires both queries beat augmented additive fullactionerror, fullactionSSEzero<1
and scoreSSE/sourceconstant<1. A useful local geometry effect is retained even if
this conjunction fails, with magnitude rather than a generic PASS label.

If raw geometry improves native-change recovery but not wholeL, inspect the
missing scalar/context contribution rather than erase the measured interaction.
If source changes remain poorly recovered, close this context-metric candidate:
separate paired-average and paired-difference teaching algebra before deciding
whether the objective suppresses the interaction or the BODY dictionary misses
its direction. That is a new question, not permission to tune until new8 passes.

## Required outputs

New full-context module/runner/tests; private `logic_full_context_geometry_v1/`;
public aggregate `FULL_CONTEXT_GEOMETRY_REPORT_V1.md`; source/context kernel
distance account, exact replay, scalar/BODY/last/mixed finite differences,
orthogonal-coordinate invariance and unchanged additive normal-equation tests.
Sync theory/atlas/goal/parent and local commit. Release/scorer/lockbox unchanged.
