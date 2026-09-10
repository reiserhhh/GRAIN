# NEXT62: derivative-enhanced scalar token dictionary

2026-09-11. Planned, not executed. NEXT61 finds only3.984%/2.805% of the normalized
source density target represented by the finite scalar-kernel derivative span.
One development question: do source-local directional kernel features improve
the compatible whole function, beyond the original radial scalar dictionary?

## Fixed inputs and one new candidate

Reuse NEXT60 pure H16/GRAIN/layout, native source L/H16 BODY gradients, all source
references, rank8 descriptors, centering/SD,16 centers per source author, bandwidth,
and original gradient/scalar target normalizations. Source4 excluded individually;
old exposed new8 uses source4. No new material, current teacher, neural/API/Y/MEPS.

Only extend kernel_then_mean. Its frozen NEXT60 model is the unchanged baseline;
do not refit it. Mean-first's zero-allocation failure is already understood and
need not consume another candidate. Ten new fits, one prediction/replay and
one source column-space capacity account;120s CPU cap. Estimate after first full
reference; preserve bounded partial outputs if remaining work exceeds the cap.

For source center l_m, v=[z,p] and sigma=old bandwidth, add the scalar features

    psi_mj(v)=sigma * d/d(l_mj) k(v,l_m)
             =(z_j-l_mj)/sigma * k(v,l_m), j=1..8.

The new scalar is

    Lhat=b+a·mean(z)+beta·z_last
         +mean_t sum_m [c_m k(v_t,l_m)+sum_j d_mj psi_mj(v_t)].

Use its exact analytic derivatives, including

    d psi_mj/d z_a=k/sigma * [delta_aj-(z_j-l_mj)(z_a-l_ma)/sigma^2].

The extra features permit a nonzero independently oriented local derivative at
a center, while every field remains the derivative of one scalar. No arbitrary
vector-output head or actual-current gradient correction. It is a nested finite
dictionary; zero extra coefficients exactly recover the old function. With64
centers the parameter count is593 vs81; with48,449 vs65. Explicitly report this
capacity increase, not claim parameter-count-matched architectural superiority.

Keep the original equal-author value+density objective andalpha=.01, no new
feature standardizer beyond the analytic sigma scaling. No search for bandwidth,
rank, center placements or loss weights. Do not place centers at held response peaks.
Accumulate normal equations in token/coordinate chunks; avoid large hidden-width
by-center tensors. Only source losses and timing may control bounded execution.

## Source and application evidence

Source: same normalizations, scalar/density losses and penalized objective; fixed
1e-10 SVD density projection shows how much the added span actually represents.
Keep effective singular spectrum and coefficient amplification visible. Nested
unregularized span should not lose capacity beyond the documented numerical cutoff;
joint objective should not exceed the old feasible coefficient solution. Individual
score/field losses need not each be monotonic in the joint optimum.

Freeze source models and all pure-input predictions BEFORE application targets.
Application: wholeL r/RMSE/bias/SD/SSE-sourceconstant; BODY field error; all4 GRAIN
actions and zero-mean allocation actions with r/RMS/SSE-zero/signs; query contrasts.
Report source4 exclusion and new8 separately. Reuse the exact NEXT60 baseline
outputs, not historical NEXT57 comparisons as matched evidence. All coefficients
remain source trained; no gains/recalibration on new8 and no human-Y accuracy claim.

Promotion keeps NEXT60's complete-function criteria: candidate actionerror below
the baseline, actionSSEzero<1, scoreSSE/sourceconstant<1 for BOTH queries. Useful
source or local configuration effects are retained without relabeling them whole
inference. No mandatory psychological r>.5 gate.

If source capacity rises but application does not, next inspect missing whole-
context/query-last conditioning rather than tunealpha. If capacity barely grows,
inspect which token/context component the finite dictionary still misses before
adding more of the same centers. Do not turn this into another source-only loop.

## Tests, artifacts and theory

Finite differences of BODY/last gradients with non-unit scales; Hessian/cross-
partial symmetry; scalar line integral; exact old-function recovery; equal-author
weights; reference exclusion; pure-input target denial; frozen prediction replay.

New `hermite_token_potential` module/script/tests. Private
`results/grain_vector_mechanisms/logic_hermite_token_potential_v1/`, public aggregate
`HERMITE_TOKEN_POTENTIAL_REPORT_V1.md`, mathematical note, source/program hashes,
ledger/parent sync/local commits. No release/scorer/lockbox/MEPS changes.

[GDML](https://arxiv.org/pdf/1611.04678) uses derivative/Hessian kernels to learn
compatible fields. Here the explicit finite Hermite-type scalar expansion uses
Gaussian kernels and fixed source landmarks; it is not that paper's molecular
Matérn estimator or a claim of psychological energy conservation.
[Hansen](https://www2.compute.dtu.dk/~pcha/Regutools/RTv4manual.pdf) motivates keeping
source singular structure and coefficient amplification explicit. The scientific
contribution sought is a useful author-text/GRAIN/query pathway, not inventing SVD
or declaring a generic finite kernel universally expressive.
