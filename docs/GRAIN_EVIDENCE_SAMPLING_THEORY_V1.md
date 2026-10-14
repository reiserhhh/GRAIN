# Observation law is part of the measurement: HM196

2026-10-12. A finite mechanism comparison, not a new sampling theorem or a
personality-accuracy promotion. Original18/100/44 and the native semantic bridge
remain OPEN. All prior results and scientific seals are preserved.

## 1. The new distinction in this interface

The same observed frequency table can denote two different objects:

- A reusable estimate of an author's expression propensity.
- A finite inventory of already collected records to sample without replacement.

Neither interpretation follows from the vector coordinates or target labels
alone. HM174/191 explicitly declared the first working law. We now construct
the second with the same386 SOURCE authors,34 records each,8 cells, equal author
weights and five binary targets. Existing hypergeometric detector coverage and
unequal-event geometry are prior project work, not this posterior interface.

For source count n_ua and observed counts k_a, N=34:

    L_R(u,E) = product_a (n_ua/N)^k_a
    L_F(u,E) = product_a (n_ua)_[k_a] / (N)_[|E|]
    w_u(E) = L(u,E) / sum_v L(v,E)
    mu_q(E) = sum_u w_u(E)Y_uq.

Here (n)_[k] is the falling factorial, including exact zero when k>n.
These are ordered-word probabilities. A count-vector multinomial coefficient
would be common to every source candidate and cancel in the posterior.
Different event identities in the same cell are distinct observations;
reusing an identical event ID does not multiply likelihood or consume inventory.

## 2. An exact mechanism signature, not a precision target

Both laws give identical one-event posteriors. If no observation cell repeats,

    L_F/L_R = N^|E| / (N)_[|E|],

which is the same for every source candidate, so the posterior remains
identical. The assumption of equal SOURCE inventory size is essential.

When cells repeat, on common positive support,

    log(L_F/L_R) = |E|log(N)-log((N)_[|E|])
                 +sum_a sum_(j=0..k_a-1) log(1-j/n_ua).

The last term is source-dependent material depletion. It is not an extra
personality trait, attention path, or semantic reasoning module. All SOURCE
vectors, labels and singleton target associations remain fixed.
The algebraic limit for fixed E and proportional inventories m*n is L_F/L_R->1
as m->infinity; no N/precision grid was run and no new observations are implied.

Conditional on a fixed source inventory, category-a repeat covariance is
-p_a(1-p_a)/(N-1); conditional iid covariance is0. Mixing source inventories
adds between-source variation. Thus context-dependent target updates can
contain both source reweighting and protocol-induced dependence. None of these
terms is automatically removed as noise.

## 3. Update state must include the evidence regime

Under R the next-source probability is n_ua/N. Under F it is

    p_u(a|E)=(n_ua-k_a)/(N-|E|).

With the correctly indexed probability p_next and posterior weight w,

    pi_a = sum_u w_u p_next(u,a)
    Gamma_qa = sum_u w_u Y_uq p_next(u,a)-mu_q*pi_a
    mu_q_after_a = mu_q + Gamma_qa/pi_a.

The covariance identity is mature Bayes algebra. The added construction rule
is that GRAIN's evidence state cannot silently substitute propensity for
remaining inventory, even when their initial target meanings agree. Record
identity, regime and remaining opportunity are part of the observation state.
Zero support is unavailable, not a neutral trait score or a reason to smooth.

## 4. Complete fixed-material results

All480 old evidence families are supported under both laws. They contain
24 historical application authors x2 views x2 raw/cue arms x5 subset variants,
not480 people; the3456 events across families are reused observations.
No new application labels, native answers, neural calls, raw/private text,
training, source fitting or predictor selection were used.

Singleton error is0; all56 distinct-category ordered pair posteriors agree
within8.88e-16. Repeated-category pair maximum difference is.0124696854.
On the complete old families, overall RMS posterior difference is.042272074.

| Target | RMS probability difference | Maximum absolute difference | Opposite prior-relative directions /480 families |
| --- | ---: | ---: | ---: |
| E | .040999 | .314881 | 11 |
| N | .045855 | .284573 | 18 |
| A | .034704 | .177839 | 29 |
| C | .036636 | .217529 | 26 |
| O | .051034 | .615940 | 12 |

These are observation-law sensitivities, not prediction gains, r values,
human trait changes, independent effect counts or uncertainty intervals.
96/2400 family-target prior-relative directions reverse. The material
depletion account is a concrete competing mechanism for accumulation; it
does not establish which law describes new human comments or a native LLM.
Application comments were NOT sampled from these SOURCE authors' archives.

## 5. What changes in the mainline

"Text->score" is insufficient as a construction description. Use

    (author reference, observed event representation, target anchor,
     evidence-source/sampling law, identity-qualified observation operation).

A score or nonlinear update is interpreted only within this declared tuple.
The next useful experiment must identify a common event-generating/reading
mechanism using actual observations or mapped interventions. It must not
pick the sampler with stronger scores, fit native answers, or continue a
budget/bin/rank grid. Existing HM195 actual continuations are a separate
empirical observation family, not confirmation of either resampling law.
HM191's memory witnesses remain valid under their declared propensity law;
this packet does not retrospectively call them false or psychological truth.

## 6. Prior art, verification and preserved metadata error

[Niepert/Domingos2014](https://homes.cs.washington.edu/~pedrod/papers/mlc14.pdf),
Sections2.1-2.2/Figure2 already develop finite exchangeability/urns and explain
why finite exchangeable laws need not be infinitely extendable iid mixtures.
The Diaconis/Freedman1980 official fulltext body was inaccessible; no full-read
claim is made. The pinned OpenAI/math three-state foundations98-116/152-215
motivates declaring the observation experiment and conditional-independence
requirements. Its tree/spin/threshold theorem is not transferred.

10 new/42 focused tests and3 Python lint/format checks pass. Science stages
prepare/compute/verify each ran once,compute1.485784 seconds CPU. A literal
falling-factorial/count-product audit checks4800 scalar values,maximum2.78e-15;
old shared-author reading replay differs2.55e-15. Same-agent alternate formula,
not independent blind replication. No scientific retry or postseal code edit.

The sealed prepare quality file incorrectly reports `cold_authors=2`: it reads
arm from key `arm/author/view/variant`. Actual author index count is24, confirmed
against the parent writer. This is a reporting error, not a change to any
likelihood, observation, family count or result. The original quality file and
code are retained; `metadata_erratum.json` supplies the corrected count.

Canonical: `results/grain_math_response_v1/evidence_sampling_r1`.
Run `scripts/run_grain_evidence_sampling_hm196.py prepare|compute|verify` ONLY
for a fresh declared output root. Do not reexecute the frozen r1 stages.

Goal ACTIVE, no background job, no Git/Claude/automation/lockbox/old queue.
