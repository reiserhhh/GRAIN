# V6 Selection-Preserving Design Protocol

Status: frozen simulation protocol, 2026-07-14.

## Problem

In naturally collected text, topic, situation, venue, and opportunity are not
automatically nuisance variables. An author may choose them in a way related to
their stable location in the population. Therefore the free-text condition mean
is generally

\[
E[Y\mid C=c] = \mu_c + E[a_u\mid C=c] + E[B_{u,c}\mid C=c],
\]

not merely a removable condition baseline. Subtracting it can remove a
selection-linked component of `a_u`.

## Simulation worlds

Both worlds use the same two-phase observation design.

1. **Free choice phase:** condition `C` is sampled from an author-specific
   softmax. It is the intended estimator for a choice/selection profile.
2. **Fixed condition phase:** every author receives every condition equally
   often, independently of their author level. It is the intended estimator
   for conditional level and response contrasts `B_u`.

The `selection_signal` world has author-dependent choice and no global
condition nuisance. The `condition_nuisance` world has an independent choice
process and strong shared condition effects. This pair rejects any claim that
one centering rule is universally correct.

## Estimands and gates

All references are fit on a deterministic discovery-author half and scored on
the held-out half. The outputs are:

- raw free author-level recovery;
- free condition-centred author-level recovery;
- balanced fixed-condition author-level recovery;
- free versus fixed response-contrast support and recovery;
- held-out free-choice log-score gain over the discovery-global condition rate.

The frozen full run requires: positive choice holdout gain in the
selection-linked world; raw-free recovery above free-centred recovery there;
free-centred recovery above raw-free recovery in the condition-only world;
full fixed-phase response support and recovery `>= .60`; and incomplete free
response support. These are planted-world gates only.

## Claim boundary

This test does not establish that Reddit topic choice is personality, that every
format feature is an opportunity, or that a future human study will reproduce
these magnitudes. It establishes a counterexample to universal condition
subtraction and motivates a two-phase GRAIN collection design: report free
selection and fixed-condition conditional expression separately.
