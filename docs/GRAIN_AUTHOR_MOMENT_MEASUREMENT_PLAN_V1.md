# HM62: From Evidence Configurations to Distribution-Based Measurement

2026-10-09. Finite CPU-only protocol, not a prediction optimization. HM61 is
frozen; no new text, model calls, fitting or private-human use. Original scope
remains open.

## Scientific Decision

HM61 describes a response function on six jointly present behavior alternatives.
GRAIN describes how an author's local observations are distributed. The missing
connection is between a distribution and a target-linked observation functional,
not a more accurate emulation of the reader.

For a declared configuration distribution p_u, define the mean of per-configuration
readings, NOT a new LLM judgment of all of that author's history concatenated:

    s_uq = sum_x p_u(x) R_q(x) = sum_S a_qS m_uS
    m_uS = E_pu product_(j in S) X_j.

These are controlled distribution profiles of existing synthetic text materials.
They are not measured human traits. The identity is standard integration/Fourier
algebra. The empirical question is which moments act in the actual frozen
six-target response table, and what lower-moment GRAIN summaries cannot retain.

## Complete Fixed Battery

1. Preserve the exact HM61 table and its verification/manifest hashes. Use both
   lexical families, six queries, score and log-odds units. No target selection.
2. For ALL 63 nonempty subsets S, create p+/-=(1 +/- .5*chi_S)/64. Every profile
   has full support. They agree on all moments of order below |S| but differ on
   m_S. Evaluate both by direct integration and coefficient contraction.
3. Read saved body/query endpoint coefficients to report corresponding vector
   displacements. No new hidden-state calls. A low-order summary collision is
   not proof that the full embedding discarded the distinction.
4. For each family/unit/query and r=1,2,3,6, hold the reference moments through
   order r exactly fixed and optimize the minimum/maximum possible mean reading
   over valid distributions: p>=0, sum p=1, A_r p=mu_r. This is 96 finite
   identification intervals / 192 linear programs, not parameter search or CIs.
5. Save primal witnesses and lower/upper dual polynomials. Verify nonnegative
   probabilities, exact moments, dual inequalities and primal-dual gaps.
6. Compute family0 extrema first and seal them before applying these same
   distributions to family1. Family1's own extrema are descriptive oracle bounds,
   not inputs for source choices. The underlying HM61 application values were
   already observed: this is a result-after mechanism analysis, not fresh blind
   validation or a new independent lexical family.
7. Synthetic controls: additive table collapses first-moment intervals; a pure
   third-order table has equal means/covariance profiles but different readings;
   all moments reconstruct the distribution; invalid moments fail explicitly.

## Identification and Construction Rule

At an interior reference distribution, a target mean is uniquely determined by
the available moments iff its response vector lies in the row space of A_r.
Proof: row-space membership gives a linear function of mu. Otherwise choose a
nonzero target-active h in ker A_r; sufficiently small p0 +/- epsilon*h remain
valid and indistinguishable under the measured moments but change the target.
At boundary distributions, support restrictions can change this equivalence.

Use this as a construction diagnostic: retain target-active joint moments or
provide a declared conditional model instead of automatically treating all
covariation as noise. An infinite sample can estimate retained moments exactly
but cannot restore a discarded target-active direction.

This does not identify a psychological construct, prove the query's validity,
recover an author's B_u, or score a new natural text without an encoder. Choice,
condition, state and admissible event definitions remain separate construction
requirements. Do not call a profile difference personality change.

## Sources and Boundaries

[Bertsimas/Popescu 2005](https://web.mit.edu/dbertsim/www/papers/MomentProblems/Optimal-inequalities-in-probability-theory-A-convex-optimization-approach-SIAM15.pdf),
introduction/Definitions1.1--1.2, supplies the moment-feasible-distribution and
sharp-bound framework. This packet uses a finite LP with elementary primal/dual
proofs, not the paper's general SDP hierarchy. Targeted later PDF retrieval timed
out; do not claim full-paper verification.

OpenAI/math fixed adc7f1241b42e322a6451854ab7e4b4c146bf78a,
Posterior-replicas-and-conditional-information-in-Gaussian-regression-September-27-2026,
01-model-endpoint.tex lines1--190 re-read: residual fibers after a projection
motivate asking what a summary leaves unspecified. Gaussian/Haar independence,
spherical priors and finite-bit memory theorems are NOT applied to text profiles.
Old GRAIN M05/HM10 already used moments/interactions. New scope is actual HM61
multi-target distribution contractions, attainable moment sets and witnesses,
not the invention of moment theory.

## Acceptance and Stop

Deliver complete tables, witnesses, raw-to-derived binding and alternate-formula
checks, plus a report of which distribution information changes measurement.
The result may be additive or transport may reverse: retain it without rescue.
Stop this packet after 192 LPs /120 seconds; no order, regularization or accuracy
grid, no model/private/API/Claude/automation/publish activity. Keep failures.
