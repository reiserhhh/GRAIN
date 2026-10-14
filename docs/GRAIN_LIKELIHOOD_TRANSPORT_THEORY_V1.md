# Source-supported marginal and joint-configuration updates

2026-10-13 JST. HM240 complete, goal active, no live job. Canonical output:
`results/grain_math_response_v1/likelihood_transport_r1`.
The original 18 protocols / 100 cards / 44 questions remain OPEN.

## 1. Construction and scientific result

A numerical evidence update now retains a full positive 32-profile law,
five marginal readings, their finite changes, and 26 higher configuration
accounts. The cold CLI needs numeric endpoints and bound source/evidence
identities, not model answers, application questionnaire values or raw text.
It does not replace the upstream neural encoder or construct likelihoods
from arbitrary natural language without the existing SOURCE instrument.

The full-evidence relation-law comparison has mean joint score-energy ratio
0.809140 across 48 bank-author accounts. But the frozen joint-only tilt
changes the five marginal readings by only 0.006669--0.010632 of the full
RMS change. Large joint geometry does not imply large marginal-score change.

The joint residual has 26 dimensions, versus five marginal directions;
26/31=0.838710 is the dimensional expectation for an isotropic score under a
uniform reference. This is an explanatory benchmark, NOT a fitted null for
these data. The observed 0.809140 is NOT a personality-information fraction.

Post-outcome comparison of the two disjoint SOURCE banks keeps all 24
authors: five-mean update pooled cosine 0.383511 (20/24 positive), higher
configuration full update 0.031439 (17/24), joint contribution 0.038845
(18/24). These are instrument-direction comparisons, not Pearson r,
psychological reliability estimates, or independent replications. Joint
configuration is retained, not named as a reusable new personality factor.

## 2. Observation contract

Reuse authenticated HM209 `panel0_log_profile` / `panel1_log_profile` only,
each 216 x 2 x 32, and opaque author/frame addresses from its cold CSV.
SOURCE386 authors remain split into banks203/183, with frozen Gaussian
variance0.05525111588037168 and the old uniform32-profile prior. The source
likelihood is an instrument defined from existing E5 vectors and historical
binary target profiles; it is not an identified natural human generative law.

There are 432 paired-versus-product relation-law comparisons on nine frames,
and 768 full-to-eight-edit comparisons under two laws, for 1,200 dependent
updates. Primary full/full relation comparisons are 48 accounts on 24 authors.
They change the declared source observation law on identical evidence, not
the humans or their text. Edited frames inherit the old selected/control
operations; they are not newly randomized psychological manipulations.

## 3. Exact finite mathematics

For strictly positive normalized laws p0,p1 on g in {0,1}^5, let
ell(g)=log[p1(g)/p0(g)]. Declare the exponential bridge

    p_t(g) = p0(g) exp[t ell(g)] / Z(t),  0 <= t <= 1.

This is numerical interpolation, NOT recovered token/attention flow,
chronological time, or causal author change. With Y(g) the five bits,

    mu_t = E_t Y; C_t = Cov_t(Y); b_t = Cov_t(Y,ell)
    d mu_t/dt = b_t
    s_t = ell-E_t ell
    m_t = (Y-mu_t)' C_t^-1 b_t
    j_t = s_t-m_t.

Positive support on all32 profiles makes C_t positive definite. No ridge,
pseudoinverse or tuned rank is used. Weighted Hilbert projection gives

    E_t j_t = 0; Cov_t(Y,j_t) = 0
    Var_t ell = E_t m_t^2 + E_t j_t^2.

These zeros hold BY CONSTRUCTION, not because human traits are independent.
Integrating the actual marginal derivative returns mu1-mu0. Integrating
Var_t ell returns KL(p1||p0)+KL(p0||p1), the Jeffreys divergence. Retain all
26 higher Boolean-character derivatives, with both main and joint parts;
main directions can also alter configuration in a nonuniform reference.

## 4. Local invisibility versus a finite update

At the start reference, separately define the positive numeric tilts

    p_main proportional to p0 exp(m_0)
    p_joint proportional to p0 exp(j_0)
    delta_full = delta_main + delta_joint + delta_interaction.

The last identity keeps the signed finite interaction. It is not an additive
causal attribution or a partition of psychological information. The joint
score has zero initial derivative of five means, but its finite tilt can
change them. Its observed finite effect is small here, not forced to zero.
All complete laws and finite responses remain in the output, including the
small effects. These tilts are numeric counterfactuals, not new valid texts.

Full-relation five-mean delta RMS E/N/A/C/O:
0.000793036 / 0.001512245 / 0.000871178 / 0.000866374 / 0.001240981.
Frozen joint-only RMS:
0.000006989 / 0.000010085 / 0.000009062 / 0.000009211 / 0.000009372.
Main-only endpoint KL mean0.000049252; this preserves the distinction between
similar marginal changes and different complete joint laws.

For evidence edits, paired/product five-mean bank alignments are negative
pooled cosines -0.079623/-0.085267, with 91/192 and89/192 positive accounts.
The primary relation comparison's partial agreement does not automatically
transport to removing evidence. No sign flips or winners rescue this result.

## 5. Prior art and actual increment

HM82/S12 already used weighted projections and Fisher geometry; HM103/104
already separated response fibers and constructed path operations; HM177/209
already retained full target-profile dependence. This packet is not their
rediscovery as new mathematics. Its increment is a usable SOURCE-relative
update record joining marginal readings and joint configuration on the SAME
evidence, plus a data-based decision to separate their uses.

[Nielsen2020](https://franknielsen.github.io/entropy-22-01100-v2.pdf), pp23--24
and42, supplies exponential-family covariance/Fisher and exponential-arc
geometry. The finite identities above are directly derived standard
probability and projection statements, not a new theorem.

OpenAI/math pinned adc7f124, three-state `02-foundations.tex`110--210, was
reread. It distinguishes complete posterior sufficiency inside an existing
experiment from valid conditionally independent composition. Its three-state
tree theorem is not transferred to this Gaussian human-text instrument.
No nonlinear path is identified merely by choosing exponential interpolation.

## 6. Reproduction, scope and next mechanism

The frozen16-node calculation took0.075301s, with zero new neural/API/private/
training/application-label reads. Prepare/compute/verify/check each ran once.
All1,200 updates and31,200 configuration rows are retained. Literal sums at
2,400 node accounts give covariance/orthogonality error8.951e-18; integrated
endpoint error3.343e-16. Cold subprocess readback is exactly equal. This is
same-agent alternate arithmetic, not blind replication. 21focused tests and
four new Python lint/format checks pass. Primary scientific seals stay fixed;
source-bank alignment is a separately marked sealed post-outcome description.

Read-only reproduction:

    python scripts/run_grain_likelihood_transport_hm240.py check
    python scripts/run_grain_likelihood_transport_hm240.py cold --input results/grain_math_response_v1/likelihood_transport_r1/cold_input.json

Next inventory the existing HM171/173/192/209/233 source-meaning and SAME-event
vector-action interfaces. Find which source-supported target functional is
reusable across source populations and which is observer-specific joint
configuration, BEFORE another finite experiment. Add a measurement capability,
not another energy/rank/quadrature/target-accuracy table. No automatic covariance
deletion, coefficient rescue, new factor naming, or Gaussian-width search.
