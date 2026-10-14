# HM196: propensity inference versus finite evidence inventory

## Question and project delta, declared before computation

HM174/175/191 use empirical SOURCE frequencies as reusable conditional iid
propensities. Their nonlinear evidence accumulation is not, by itself, evidence
of semantic reasoning. A different observation law samples distinct records
without replacement from the finite SOURCE inventory. Both laws have identical
SOURCE targets, author prior, coordinates, condition choices, eight-cell alphabet,
and every one-event target reading. Do their multi-event readings agree?

Existing `evidence_budget.py` has hypergeometric detector coverage, not the full
five-target posterior/continuation interface. HM189 has unequal-event geometric
cross-products; HM172 has fixed-bag/end path reordering; HM195 has actual ordered
transitions. None is replaced or rerun. This packet isolates the sampling regime
in the HM174 working law. It does NOT assert that new human comments were sampled
from the SOURCE archives, or that natural chronologies are exchangeable.

## Competing constructions and predictions

Let n_ua be the old integer count in cell a, N=34, and k_a the number of distinct
observed records in that cell. Use the SAME equal-author prior and old binary Y.

    L_R(u,E) = product_a (n_ua/N)^k_a
    L_F(u,E) = product_a (n_ua)_[k_a] / (N)_[sum k]
    w_u(E) proportional to L(u,E)
    mu_q(E) = sum_u w_u(E)Y_uq

Both singletons agree. If no category repeats, L_F/L_R is author-independent
because all inventories have the same N; every target posterior must agree.
For repeated categories the source-specific depletion factor can change the
reading even without a new semantic mechanism. Exhausted inventory is zero
support, NOT neutral evidence, a skipped unfavorable case, or a smoothing cue.

The finite continuation is (n_ua-k_a)/(N-|E|), not the old fixed n_ua/N.
With this correctly indexed continuation the same covariance identity holds:
mu_after = mu + Cov_w(Y,p_next)/P_w(next). A persistent author propensity and a
consumable historical inventory are different model objects, although both are
Bayesian. Neither is automatically the natural-language or native LLM law.

## Fixed materials and finite scope

- Read only the sealed HM174 numeric SOURCE bundle and all480 old cold families.
- Infer integer n by round(34*p), reject inconsistent counts; no parameter fit.
- Compute all8 singleton and all64 ordered pair comparisons for all5 targets.
- Replay every480 family under both laws, retaining identities and all steps.
- Repeated ID/same cell is the same event, not a second draw. Reused ID/different
  cell is invalid. Different ID/same category is a second draw under either law.
- No new predictions against application scales or native answers, no bootstrap,
  bin/rank/seed/N search, and no model promotion. Record every unsupported family.
- The infinite-inventory limit is an algebraic statement only, not an N grid.

## Prior art and mathematical scope

Niepert/Domingos, *Exchangeable Variable Models*, ICML2014, Sections2.1-2.2 and
Figure2 explicitly distinguish finite exchangeability from infinitely extendable
iid mixtures and represent finite laws by urns without replacement:
https://homes.cs.washington.edu/~pedrod/papers/mlc14.pdf . This is primary prior
art, not our new exchangeability theorem. Diaconis/Freedman1980 is its cited
foundation; the official Euclid fulltext endpoint returned no readable body,
so this packet does not claim to have read that original paper in full.

Pinned OpenAI/math adc7f124, three-state `02-foundations.tex`, lines98-116 and
152-215: posterior sufficiency is for a declared experiment and product
composition requires conditional independence. Used as observation-law design
inspiration ONLY. No tree, spin, reconstruction threshold, or new theorem is
transferred. Original18/100/44 remain OPEN.

## Execution and stop

Focused synthetic tests and Ruff precede sealing. `prepare`, `compute`, `verify`
each execute once at `results/grain_math_response_v1/evidence_sampling_r1`.
One alternate falling-factorial/count-product audit suffices. Do not optimize
precision or rerun frozen scientific stages. Preserve failures and old seals.
Deliver a report distinguishing mathematical construction, the observed magnitude
on old natural-text numeric inputs, and the still-unidentified natural sampling
law. No API/private text/new neural/train/Git/Claude/automation/lockbox/old queue.
